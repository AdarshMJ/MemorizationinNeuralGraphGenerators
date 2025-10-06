"""
Feature-aware memorization-to-generalization experiment.

This script mirrors `main_comparison.py` but extends the VAE and evaluation
pipeline to reconstruct node features and discrete labels jointly. The latent
Diffusion model therefore generates adjacency matrices, feature vectors, and
labels together, enabling downstream checks on feature homophily.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

import main_comparison as base
from autoencoder import VariationalAutoEncoderWithFeatures
from denoise_model import DenoiseNN, p_losses, sample
from synthgraphgenerator import spectral_radius_sp_matrix
from utils import construct_nx_from_adj, gen_stats, linear_beta_schedule

# Reuse shared configuration from the base experiment
N_VALUES = base.N_VALUES
TEST_SET_SIZE = base.TEST_SET_SIZE
SPLIT_SEED = base.SPLIT_SEED
NUM_SAMPLES_PER_CONDITION = base.NUM_SAMPLES_PER_CONDITION
KS_SIGNIFICANCE_THRESHOLD = base.KS_SIGNIFICANCE_THRESHOLD
BETA_KL_WEIGHT = base.BETA_KL_WEIGHT
SMALL_DATASET_THRESHOLD = base.SMALL_DATASET_THRESHOLD
SMALL_DATASET_KL_WEIGHT = base.SMALL_DATASET_KL_WEIGHT
SMALL_DATASET_DROPOUT = base.SMALL_DATASET_DROPOUT

EPOCHS_AUTOENCODER = base.EPOCHS_AUTOENCODER
EPOCHS_DENOISER = base.EPOCHS_DENOISER
EARLY_STOPPING_PATIENCE = base.EARLY_STOPPING_PATIENCE
BATCH_SIZE = base.BATCH_SIZE
LEARNING_RATE = base.LEARNING_RATE
GRAD_CLIP = base.GRAD_CLIP
LATENT_DIM = base.LATENT_DIM
HIDDEN_DIM_ENCODER = base.HIDDEN_DIM_ENCODER
HIDDEN_DIM_DECODER = base.HIDDEN_DIM_DECODER
HIDDEN_DIM_DENOISE = base.HIDDEN_DIM_DENOISE
N_MAX_NODES = base.N_MAX_NODES
N_PROPERTIES = base.N_PROPERTIES
TIMESTEPS = base.TIMESTEPS

# Feature homophily loss weight
HOMOPHILY_LOSS_WEIGHT = 5.0  # Weight for feature-label correlation loss (increased to match target range)
HOMOPHILY_TEACHER_WEIGHT = 3.5
LABEL_ENTROPY_WEIGHT = 0.01

def compute_dataset_target_homophily(data_list):
    """Compute average feature homophily from dataset for use as training target."""
    from synthgraphgenerator import spectral_radius_sp_matrix
    
    homophily_values = []
    for data in data_list:
        adj = data.A[0].cpu().numpy()
        features = data.x.cpu().numpy()
        labels = data.y.cpu().numpy()
        num_classes = int(labels.max()) + 1
        
        # Use the same estimation function we use for evaluation
        try:
            h = estimate_feature_homophily(adj, features, labels, num_classes, max_iterations=50)
            homophily_values.append(h)
        except:
            continue
    
    if homophily_values:
        return np.mean(homophily_values)
    return 0.2  # Default fallback

# Reuse heavy-lifting helpers from the base module
load_dataset = base.load_dataset
shuffle_and_split_dataset = base.shuffle_and_split_dataset
create_splits = base.create_splits
perform_distribution_checks = base.perform_distribution_checks
compute_within_model_similarity = base.compute_within_model_similarity
compute_mmd_rbf = base.compute_mmd_rbf
save_latent_projection = base.save_latent_projection
find_closest_graph_in_training = base.find_closest_graph_in_training
compute_wl_similarity = base.compute_wl_similarity
compute_statistics_distance = base.compute_statistics_distance
visualize_single_experiment = base.visualize_single_experiment
visualize_results = base.visualize_results
save_summary = base.save_summary

# Shared device configuration
device = base.device


def train_autoencoder_with_features(data_list, run_name, output_dir, target_homophily=0.2):
    """Train a VAE that reconstructs adjacency, node features, and labels.
    
    Args:
        target_homophily: Target feature homophily value for the loss function
    """
    print(f"\n{'='*80}")
    print(f"Training Feature-aware Autoencoder: {run_name}")
    print(f"  Target feature homophily: {target_homophily}")
    print(f"{'='*80}")

    input_feat_dim = data_list[0].x.shape[1]
    num_classes = int(getattr(data_list[0], 'num_classes', int(torch.max(data_list[0].y).item()) + 1))

    autoencoder = VariationalAutoEncoderWithFeatures(
        feature_dim=input_feat_dim,
        hidden_dim_enc=HIDDEN_DIM_ENCODER,
        hidden_dim_dec=HIDDEN_DIM_DECODER,
        latent_dim=LATENT_DIM,
        n_layers_enc=2,
        n_layers_dec=3,
        n_max_nodes=N_MAX_NODES,
        num_classes=num_classes,
        feature_loss_weight=1.0,
        label_loss_weight=1.0,
    ).to(device)

    dataset_size = len(data_list)
    beta_value = BETA_KL_WEIGHT
    if dataset_size <= SMALL_DATASET_THRESHOLD:
        beta_value = SMALL_DATASET_KL_WEIGHT
        autoencoder.encoder.dropout = SMALL_DATASET_DROPOUT
        print(
            f"Small dataset detected ({dataset_size} graphs) → KL beta set to {beta_value}, "
            f"encoder dropout={SMALL_DATASET_DROPOUT}"
        )

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    patience_counter = 0
    best_epoch = 0
    best_val_loss_per_graph = np.inf
    checkpoint_path = output_dir / f'autoencoder_feat_{run_name}.pth.tar'

    if len(data_list) < 10:
        train_data = data_list
        val_data = data_list
    else:
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train] if n_train > 0 else data_list
        val_data = data_list[n_train:] if n_train > 0 and len(data_list) > 1 else train_data

    batch_size = min(BATCH_SIZE, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(BATCH_SIZE, len(val_data)), shuffle=False)

    edges_per_graph = N_MAX_NODES * N_MAX_NODES
    feature_elements_per_graph = N_MAX_NODES * input_feat_dim
    label_elements_per_graph = N_MAX_NODES

    train_metric_sums = {
        'homophily_loss': 0.0,
        'homophily_teacher_loss': 0.0,
        'degree_loss': 0.0,
        'mi_loss': 0.0,
        'entropy': 0.0,
    }
    val_metric_sums = {
        'homophily_loss': 0.0,
        'homophily_teacher_loss': 0.0,
        'degree_loss': 0.0,
        'mi_loss': 0.0,
        'entropy': 0.0,
    }

    for epoch in range(1, EPOCHS_AUTOENCODER + 1):
        autoencoder.train()

        train_loss_all = 0.0
        train_kld_sum = 0.0
        train_count = 0
        epoch_train_metrics = {
            'homophily_loss': 0.0,
            'homophily_teacher_loss': 0.0,
            'degree_loss': 0.0,
            'mi_loss': 0.0,
            'entropy': 0.0,
        }

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld, metrics_batch = autoencoder.loss_function(
                data,
                beta=beta_value,
                homophily_weight=HOMOPHILY_LOSS_WEIGHT,
                target_homophily=target_homophily,
                homophily_teacher_weight=HOMOPHILY_TEACHER_WEIGHT,
                entropy_weight=LABEL_ENTROPY_WEIGHT,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), GRAD_CLIP)
            optimizer.step()

            graphs_in_batch = int(torch.max(data.batch).item()) + 1
            train_loss_all += loss.item()
            train_kld_sum += kld.item()
            train_count += graphs_in_batch
            for key in epoch_train_metrics:
                epoch_train_metrics[key] += metrics_batch.get(key, 0.0) * graphs_in_batch

        autoencoder.eval()
        val_loss_all = 0.0
        val_kld_sum = 0.0
        val_count = 0
        epoch_val_metrics = {
            'homophily_loss': 0.0,
            'homophily_teacher_loss': 0.0,
            'degree_loss': 0.0,
            'mi_loss': 0.0,
            'entropy': 0.0,
        }

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                loss, recon, kld, metrics_batch = autoencoder.loss_function(
                    data,
                    beta=beta_value,
                    homophily_weight=HOMOPHILY_LOSS_WEIGHT,
                    target_homophily=target_homophily,
                    homophily_teacher_weight=HOMOPHILY_TEACHER_WEIGHT,
                    entropy_weight=LABEL_ENTROPY_WEIGHT,
                )
                graphs_in_batch = int(torch.max(data.batch).item()) + 1
                val_loss_all += loss.item()
                val_kld_sum += kld.item()
                val_count += graphs_in_batch
                for key in epoch_val_metrics:
                    epoch_val_metrics[key] += metrics_batch.get(key, 0.0) * graphs_in_batch

        train_loss_avg = train_loss_all / train_count if train_count else np.nan
        val_loss_avg = val_loss_all / val_count if val_count else np.nan

        if train_count:
            for key in train_metric_sums:
                train_metric_sums[key] = epoch_train_metrics[key] / train_count
        if val_count:
            for key in val_metric_sums:
                val_metric_sums[key] = epoch_val_metrics[key] / val_count

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: Train Loss: {train_loss_avg:.2f} | Val Loss: {val_loss_avg:.2f} "
                f"(KL train {train_kld_sum / max(train_count, 1):.2f}, val {val_kld_sum / max(val_count, 1):.2f})"
            )
            print(
                f"    Aux metrics → train homophily {train_metric_sums['homophily_loss']:.4f} (teacher {train_metric_sums['homophily_teacher_loss']:.4f}), "
                f"degree {train_metric_sums['degree_loss']:.4f}, MI {train_metric_sums['mi_loss']:.4f}, entropy {train_metric_sums['entropy']:.4f} | "
                f"val homophily {val_metric_sums['homophily_loss']:.4f} (teacher {val_metric_sums['homophily_teacher_loss']:.4f}), "
                f"degree {val_metric_sums['degree_loss']:.4f}, MI {val_metric_sums['mi_loss']:.4f}, entropy {val_metric_sums['entropy']:.4f}"
            )

        scheduler.step()

        if val_loss_avg < best_val_loss_per_graph:
            best_val_loss_per_graph = val_loss_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    'state_dict': autoencoder.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss_per_graph': best_val_loss_per_graph,
                    'beta': beta_value,
                },
                checkpoint_path,
            )
            if epoch % 20 == 0:
                print('  → New best feature-aware autoencoder saved!')
        else:
            patience_counter += 1
            if epoch % 20 == 0 and patience_counter > 0:
                print(f'  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', best_epoch)
        best_val_loss_per_graph = checkpoint.get('best_val_loss_per_graph', best_val_loss_per_graph)
    else:
        print('Warning: Feature-aware autoencoder checkpoint not found; returning last epoch weights.')

    print(
        f"Autoencoder training complete. Best val loss/graph: {best_val_loss_per_graph:.2f}"
        f" at epoch {best_epoch}"
    )

    def reconstruction_metrics(loader):
        if loader is None or len(loader.dataset) == 0:
            return np.nan, np.nan, np.nan
        autoencoder.eval()
        total_adj_abs = 0.0
        total_adj_entries = 0
        total_feat_mse = 0.0
        total_feat_entries = 0
        total_label_correct = 0.0
        total_label_entries = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                adj_pred, feat_pred, label_logits, aux = autoencoder(batch)

                adj_target = batch.A
                total_adj_abs += torch.abs(adj_pred - adj_target).sum().item()
                total_adj_entries += adj_target.numel()

                feat_target = batch.x.view(-1, N_MAX_NODES, input_feat_dim)
                total_feat_mse += torch.sum((feat_pred - feat_target) ** 2).item()
                total_feat_entries += feat_target.numel()

                label_target = batch.y.view(-1, N_MAX_NODES)
                label_pred = torch.argmax(label_logits, dim=-1)
                total_label_correct += (label_pred == label_target).float().sum().item()
                total_label_entries += label_target.numel()

        adj_mae = total_adj_abs / total_adj_entries if total_adj_entries else np.nan
        feat_rmse = np.sqrt(total_feat_mse / total_feat_entries) if total_feat_entries else np.nan
        label_acc = total_label_correct / total_label_entries if total_label_entries else np.nan
        return adj_mae, feat_rmse, label_acc

    train_adj_mae, train_feat_rmse, train_label_acc = reconstruction_metrics(train_loader)
    val_adj_mae, val_feat_rmse, val_label_acc = reconstruction_metrics(val_loader)

    print(
        f"  Reconstruction (train) - adj MAE: {train_adj_mae:.4f}, feat RMSE: {train_feat_rmse:.4f}, "
        f"label acc: {train_label_acc:.4f}"
    )
    if np.isfinite(val_adj_mae):
        print(
            f"  Reconstruction (val)   - adj MAE: {val_adj_mae:.4f}, feat RMSE: {val_feat_rmse:.4f}, "
            f"label acc: {val_label_acc:.4f}"
        )

    metrics = {
        'train_adj_mae': float(train_adj_mae),
        'train_feat_rmse': float(train_feat_rmse),
        'train_label_acc': float(train_label_acc),
        'val_adj_mae': float(val_adj_mae),
        'val_feat_rmse': float(val_feat_rmse),
        'val_label_acc': float(val_label_acc),
        'train_homophily_loss': float(train_metric_sums['homophily_loss']),
        'train_homophily_teacher_loss': float(train_metric_sums['homophily_teacher_loss']),
        'train_degree_loss': float(train_metric_sums['degree_loss']),
        'train_mi_loss': float(train_metric_sums['mi_loss']),
        'train_entropy': float(train_metric_sums['entropy']),
        'val_homophily_loss': float(val_metric_sums['homophily_loss']),
        'val_homophily_teacher_loss': float(val_metric_sums['homophily_teacher_loss']),
        'val_degree_loss': float(val_metric_sums['degree_loss']),
        'val_mi_loss': float(val_metric_sums['mi_loss']),
        'val_entropy': float(val_metric_sums['entropy']),
    }

    return autoencoder, metrics


def train_denoiser(autoencoder, data_list, run_name, output_dir):
    """Wrapper that reuses the base denoiser training loop."""
    return base.train_denoiser(autoencoder, data_list, run_name, output_dir)


def decode_latent(autoencoder, latent_codes):
    adj, feat, label_logits, aux = autoencoder.decode_mu(latent_codes)
    labels = torch.argmax(label_logits, dim=-1)
    return adj, feat, label_logits, labels, aux


def enforce_connected_structure(adj_matrix, target_edge_count=None, min_weight=1e-3):
    """Symmetrize, binarize, and densify a decoded adjacency to ensure connectivity."""
    adj_sym = 0.5 * (adj_matrix + adj_matrix.T)
    np.fill_diagonal(adj_sym, 0.0)

    num_nodes = adj_sym.shape[0]
    max_possible_edges = num_nodes * (num_nodes - 1) // 2

    if target_edge_count is None:
        approx_edges = int(np.round(np.count_nonzero(adj_sym > min_weight) / 2))
        target_edge_count = approx_edges

    target_edge_count = int(np.clip(target_edge_count, 0, max_possible_edges))
    target_edge_count = max(target_edge_count, max(1, num_nodes - 1))

    weighted_graph = nx.Graph()
    weighted_graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = float(adj_sym[i, j])
            if weight > min_weight:
                weighted_graph.add_edge(i, j, weight=weight)

    if weighted_graph.number_of_edges() == 0:
        chain_adj = np.zeros_like(adj_sym, dtype=np.float32)
        for i in range(num_nodes - 1):
            chain_adj[i, i + 1] = chain_adj[i + 1, i] = 1.0
        return chain_adj

    mst = nx.maximum_spanning_tree(weighted_graph)
    adj_binary = np.zeros_like(adj_sym, dtype=np.float32)
    for u, v in mst.edges():
        adj_binary[u, v] = adj_binary[v, u] = 1.0

    current_graph = nx.from_numpy_array(adj_binary)
    if not nx.is_connected(current_graph):
        components = list(nx.connected_components(current_graph))
        component_reps = [sorted(comp)[0] for comp in components]
        for idx in range(len(component_reps) - 1):
            u = component_reps[idx]
            v = component_reps[idx + 1]
            adj_binary[u, v] = adj_binary[v, u] = 1.0

    edges_in_graph = int(adj_binary.sum() // 2)
    candidate_edges = sorted(
        ((u, v, data['weight']) for u, v, data in weighted_graph.edges(data=True)),
        key=lambda item: item[2],
        reverse=True,
    )

    for u, v, _ in candidate_edges:
        if edges_in_graph >= target_edge_count:
            break
        if adj_binary[u, v] >= 1.0:
            continue
        adj_binary[u, v] = adj_binary[v, u] = 1.0
        edges_in_graph += 1

    return adj_binary


def generate_graphs_with_features(autoencoder, denoise_model, conditioning_stats, betas, num_samples=1):
    autoencoder.eval()
    denoise_model.eval()

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    if isinstance(conditioning_stats, torch.Tensor):
        cond_tensor = conditioning_stats
    else:
        cond_tensor = torch.tensor(conditioning_stats, dtype=torch.float32)

    if cond_tensor.dim() == 1:
        cond_tensor = cond_tensor.unsqueeze(0)

    cond_tensor = cond_tensor.to(device)

    if cond_tensor.size(0) == 1 and num_samples > 1:
        cond_tensor = cond_tensor.repeat(num_samples, 1)
    elif cond_tensor.size(0) != num_samples:
        raise ValueError(
            f"Conditioning stats batch ({cond_tensor.size(0)}) does not match requested samples ({num_samples})."
        )

    with torch.no_grad():
        samples = sample(
            denoise_model,
            cond_tensor,
            latent_dim=LATENT_DIM,
            timesteps=TIMESTEPS,
            betas=betas,
            batch_size=num_samples,
        )
        latent_codes = samples[-1]
        adj, feat, label_logits, labels, aux = decode_latent(autoencoder, latent_codes)

    conditioning_np = cond_tensor.detach().cpu().numpy()
    expected_degrees = aux.get('expected_degrees').detach().cpu().numpy() if aux.get('expected_degrees') is not None else None
    class_affinity = aux.get('class_affinity').detach().cpu().numpy() if aux.get('class_affinity') is not None else None

    graphs = []
    payloads = []
    for idx in range(num_samples):
        adj_prob_np = adj[idx].detach().cpu().numpy()

        target_edges = int(round(conditioning_np[idx, 1])) if conditioning_np.shape[1] > 1 else None
        adj_np = enforce_connected_structure(adj_prob_np, target_edge_count=target_edges)
        feat_np = feat[idx].detach().cpu().numpy()
        labels_np = labels[idx].detach().cpu().numpy()
        graphs.append(construct_nx_from_adj(adj_np, node_features=feat_np, node_labels=labels_np))
        payloads.append(
            {
                'adjacency': adj_np,
                'features': feat_np,
                'labels': labels_np,
                'label_logits': label_logits[idx].detach().cpu().numpy(),
                'expected_degrees': expected_degrees[idx] if expected_degrees is not None else None,
                'class_affinity': class_affinity[idx] if class_affinity is not None else None,
            }
        )

    return graphs, payloads, latent_codes.detach().cpu()


def estimate_feature_homophily(adj_matrix, features, labels, num_classes, max_iterations=50):
    """Approximate feature homophily using the synthetic data measurement scheme."""
    if adj_matrix.ndim != 2:
        raise ValueError("adjacency matrix must be 2-D")

    num_nodes = adj_matrix.shape[0]
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    if edge_index.numel() == 0:
        return 0.0

    edge_index_clean = remove_self_loops(edge_index)[0]
    if edge_index_clean.numel() == 0:
        return 0.0

    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index_clean[0], edge_index_clean[1]] = 1.0
    I = torch.eye(num_nodes)

    try:
        spectral_radius = spectral_radius_sp_matrix(
            edge_index_clean.cpu(),
            torch.ones(edge_index_clean.shape[1]),
            num_nodes,
        )
    except Exception:
        degrees = torch.bincount(edge_index_clean[0], minlength=num_nodes)
        spectral_radius = float(degrees.max().item()) if degrees.numel() > 0 else 0.0

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    Y = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes).float()

    h_F_range = torch.linspace(-0.9, 0.9, max_iterations)
    best_h_F = 0.0
    min_error = float('inf')

    for h_F in h_F_range:
        h_val = float(h_F.item())
        w = h_val / spectral_radius if spectral_radius > 0 else 0.0
        if abs(w) >= 0.95:
            continue
        X0 = torch.mm(I - w * A, features_tensor)
        class_sums = Y.t() @ X0
        class_counts = Y.sum(dim=0, keepdim=True).t()
        class_counts[class_counts == 0] = 1
        X0_cls_mean = class_sums / class_counts
        X0_cls = Y @ X0_cls_mean
        error = torch.abs(X0_cls - X0).sum().item()
        if error < min_error:
            min_error = error
            best_h_F = h_val

    return float(best_h_F)


def run_experiment_for_N(N, split_config, output_dir):
    print(f"\n{'#'*80}")
    print(f"# FEATURE-AWARE EXPERIMENT: N = {N}")
    print(f"{'#'*80}")

    S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices = create_splits(split_config, N)
    stats_cache_path = split_config.get('stats_cache_path')
    if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) == len(test_graphs) and stats_cache_path:
        print(f"  Using conditioning cache: {stats_cache_path}")

    exp_dir = output_dir / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    distribution_summary = perform_distribution_checks(S1, S2, exp_dir, N)

    S1_training_graphs_nx = []
    for data in S1:
        adj = data.A[0].cpu().numpy()
        features = data.x.detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()
        S1_training_graphs_nx.append(construct_nx_from_adj(adj, node_features=features, node_labels=labels))

    s1_min_idx = int(np.min(S1_indices))
    s1_max_idx = int(np.max(S1_indices))
    s2_min_idx = int(np.min(S2_indices))
    s2_max_idx = int(np.max(S2_indices))

    print(f"\n--- Training Model S1 (original indices {s1_min_idx}–{s1_max_idx}) ---")
    target_homophily_S1 = compute_dataset_target_homophily(S1)
    print(f"Computed target homophily for S1: {target_homophily_S1:.4f}")
    autoencoder_S1, recon_metrics_S1 = train_autoencoder_with_features(S1, f"S1_N{N}", exp_dir, target_homophily=target_homophily_S1)
    denoise_S1, betas = train_denoiser(autoencoder_S1, S1, f"S1_N{N}", exp_dir)

    print(f"\n--- Training Model S2 (original indices {s2_min_idx}–{s2_max_idx}) ---")
    target_homophily_S2 = compute_dataset_target_homophily(S2)
    print(f"Computed target homophily for S2: {target_homophily_S2:.4f}")
    autoencoder_S2, recon_metrics_S2 = train_autoencoder_with_features(S2, f"S2_N{N}", exp_dir, target_homophily=target_homophily_S2)
    denoise_S2, _ = train_denoiser(autoencoder_S2, S2, f"S2_N{N}", exp_dir)

    print(f"\n--- Generating and Evaluating (feature-aware) ---")
    generalization_scores = []
    generalization_variances = []
    memorization_scores = []
    stats_mse_list = []
    stats_mae_list = []
    within_s1_similarity = []
    within_s2_similarity = []
    latent_mmd_scores = []
    aggregated_latents_s1 = []
    aggregated_latents_s2 = []
    feature_homophily_errors = []
    generated_feature_homophily_values = []
    target_feature_homophily_values = []
    example_payload = None

    num_classes = int(getattr(S1[0], 'num_classes', int(torch.max(S1[0].y).item()) + 1))

    for i, test_graph in enumerate(tqdm(test_graphs, desc="Testing")):
        if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) > i:
            c_test = test_stats_cache[i:i + 1]
        else:
            c_test = test_graph.stats[:, :N_PROPERTIES]

        G1_samples, payloads_G1, latents_G1 = generate_graphs_with_features(
            autoencoder_S1,
            denoise_S1,
            c_test,
            betas,
            num_samples=NUM_SAMPLES_PER_CONDITION,
        )
        G2_samples, payloads_G2, latents_G2 = generate_graphs_with_features(
            autoencoder_S2,
            denoise_S2,
            c_test,
            betas,
            num_samples=NUM_SAMPLES_PER_CONDITION,
        )

        aggregated_latents_s1.append(latents_G1)
        aggregated_latents_s2.append(latents_G2)

        mmd_value = compute_mmd_rbf(latents_G1, latents_G2)
        latent_mmd_scores.append(mmd_value)

        if i == 0 and payloads_G1:
            example_payload = {
                'G1': payloads_G1[0],
                'G2': payloads_G2[0],
                'graph_G1': G1_samples[0],
                'graph_G2': G2_samples[0],
            }

        pairwise_generalization = []
        for sample_idx in range(NUM_SAMPLES_PER_CONDITION):
            sim_generalization = compute_wl_similarity(G1_samples[sample_idx], G2_samples[sample_idx])
            generalization_scores.append(sim_generalization)
            pairwise_generalization.append(sim_generalization)

            mse, mae = compute_statistics_distance(G1_samples[sample_idx], G2_samples[sample_idx])
            stats_mse_list.append(mse)
            stats_mae_list.append(mae)

            target_feature_hom = float(c_test[0, -1].item())
            generated_feature_hom = estimate_feature_homophily(
                payloads_G1[sample_idx]['adjacency'],
                payloads_G1[sample_idx]['features'],
                payloads_G1[sample_idx]['labels'],
                num_classes,
            )
            feature_homophily_errors.append(abs(generated_feature_hom - target_feature_hom))
            generated_feature_homophily_values.append(generated_feature_hom)
            target_feature_homophily_values.append(target_feature_hom)

        if len(pairwise_generalization) > 1:
            generalization_variances.append(np.var(pairwise_generalization))
        else:
            generalization_variances.append(0.0)

        closest_G, sim_memorization = find_closest_graph_in_training(G1_samples[0], S1_training_graphs_nx)
        memorization_scores.append(sim_memorization)

        within_s1_similarity.extend(compute_within_model_similarity(G1_samples))
        within_s2_similarity.extend(compute_within_model_similarity(G2_samples))

    if aggregated_latents_s1 and aggregated_latents_s2:
        all_latents_s1 = torch.cat(aggregated_latents_s1, dim=0)
        all_latents_s2 = torch.cat(aggregated_latents_s2, dim=0)
        overall_latent_mmd = compute_mmd_rbf(all_latents_s1, all_latents_s2)
        latent_plot_path = save_latent_projection(
            aggregated_latents_s1,
            aggregated_latents_s2,
            exp_dir / "quicklook" / f"latent_scatter_N{N}.png",
        )
    else:
        overall_latent_mmd = np.nan
        latent_plot_path = None

    feature_homophily_mae = np.mean(feature_homophily_errors) if feature_homophily_errors else np.nan
    feature_homophily_summary = {
        'generated_mean': float(np.mean(generated_feature_homophily_values)) if generated_feature_homophily_values else np.nan,
        'generated_std': float(np.std(generated_feature_homophily_values)) if generated_feature_homophily_values else np.nan,
        'target_mean': float(np.mean(target_feature_homophily_values)) if target_feature_homophily_values else np.nan,
        'target_std': float(np.std(target_feature_homophily_values)) if target_feature_homophily_values else np.nan,
        'mae': float(feature_homophily_mae) if np.isfinite(feature_homophily_mae) else np.nan,
    }

    print(f"\nResults for N={N} (feature-aware):")
    print(f"  Generalization (G1 vs G2): {np.mean(generalization_scores):.4f} ± {np.std(generalization_scores):.4f}")
    print(f"  Memorization (G1 vs closest): {np.mean(memorization_scores):.4f} ± {np.std(memorization_scores):.4f}")
    print(f"  Statistics MSE: {np.nanmean(stats_mse_list):.2f} ± {np.nanstd(stats_mse_list):.2f}")
    print(f"  Statistics MAE: {np.nanmean(stats_mae_list):.2f} ± {np.nanstd(stats_mae_list):.2f}")
    if generalization_variances:
        print(f"  Generalization variance across samples: {np.mean(generalization_variances):.4f}")
    if within_s1_similarity:
        print(f"  Within-model similarity (S1): {np.mean(within_s1_similarity):.4f}")
    if within_s2_similarity:
        print(f"  Within-model similarity (S2): {np.mean(within_s2_similarity):.4f}")
    if latent_mmd_scores:
        print(f"  Latent MMD (per-condition mean ± std): {np.mean(latent_mmd_scores):.4f} ± {np.std(latent_mmd_scores):.4f}")
    if np.isfinite(overall_latent_mmd):
        print(f"  Latent MMD (all samples): {overall_latent_mmd:.4f}")
    if latent_plot_path:
        print(f"  Latent scatter saved to: {latent_plot_path}")
    print(f"  Feature homophily MAE (generated vs conditioning): {feature_homophily_mae:.4f}")
    if generated_feature_homophily_values:
        print(
            f"  Generated feature homophily mean ± std: {feature_homophily_summary['generated_mean']:.4f} ± "
            f"{feature_homophily_summary['generated_std']:.4f}"
        )
        print(
            f"  Target feature homophily mean ± std: {feature_homophily_summary['target_mean']:.4f} ± "
            f"{feature_homophily_summary['target_std']:.4f}"
        )
    print(
        f"  Autoencoder reconstruction (S1 train adj MAE={recon_metrics_S1['train_adj_mae']:.4f}, feat RMSE={recon_metrics_S1['train_feat_rmse']:.4f}, "
        f"label acc={recon_metrics_S1['train_label_acc']:.4f})"
    )
    print(
        f"    Aux losses (train) homophily={recon_metrics_S1['train_homophily_loss']:.4f} (teacher={recon_metrics_S1['train_homophily_teacher_loss']:.4f}), "
        f"degree={recon_metrics_S1['train_degree_loss']:.4f}, MI={recon_metrics_S1['train_mi_loss']:.4f}, entropy={recon_metrics_S1['train_entropy']:.4f}"
    )
    if np.isfinite(recon_metrics_S1['val_adj_mae']):
        print(
            f"  Autoencoder reconstruction (S1 val adj MAE={recon_metrics_S1['val_adj_mae']:.4f}, feat RMSE={recon_metrics_S1['val_feat_rmse']:.4f}, "
            f"label acc={recon_metrics_S1['val_label_acc']:.4f})"
        )
        print(
            f"    Aux losses (val) homophily={recon_metrics_S1['val_homophily_loss']:.4f} (teacher={recon_metrics_S1['val_homophily_teacher_loss']:.4f}), "
            f"degree={recon_metrics_S1['val_degree_loss']:.4f}, MI={recon_metrics_S1['val_mi_loss']:.4f}, entropy={recon_metrics_S1['val_entropy']:.4f}"
        )
    print(
        f"  Autoencoder reconstruction (S2 train adj MAE={recon_metrics_S2['train_adj_mae']:.4f}, feat RMSE={recon_metrics_S2['train_feat_rmse']:.4f}, "
        f"label acc={recon_metrics_S2['train_label_acc']:.4f})"
    )
    print(
        f"    Aux losses (train) homophily={recon_metrics_S2['train_homophily_loss']:.4f} (teacher={recon_metrics_S2['train_homophily_teacher_loss']:.4f}), "
        f"degree={recon_metrics_S2['train_degree_loss']:.4f}, MI={recon_metrics_S2['train_mi_loss']:.4f}, entropy={recon_metrics_S2['train_entropy']:.4f}"
    )
    if np.isfinite(recon_metrics_S2['val_adj_mae']):
        print(
            f"  Autoencoder reconstruction (S2 val adj MAE={recon_metrics_S2['val_adj_mae']:.4f}, feat RMSE={recon_metrics_S2['val_feat_rmse']:.4f}, "
            f"label acc={recon_metrics_S2['val_label_acc']:.4f})"
        )
        print(
            f"    Aux losses (val) homophily={recon_metrics_S2['val_homophily_loss']:.4f} (teacher={recon_metrics_S2['val_homophily_teacher_loss']:.4f}), "
            f"degree={recon_metrics_S2['val_degree_loss']:.4f}, MI={recon_metrics_S2['val_mi_loss']:.4f}, entropy={recon_metrics_S2['val_entropy']:.4f}"
        )

    example_G1 = example_payload['graph_G1'] if example_payload else None
    example_G2 = example_payload['graph_G2'] if example_payload else None

    return {
        'generalization_scores': generalization_scores,
        'generalization_variances': generalization_variances,
        'memorization_scores': memorization_scores,
        'stats_mse': stats_mse_list,
        'stats_mae': stats_mae_list,
        'within_s1_similarity': within_s1_similarity,
        'within_s2_similarity': within_s2_similarity,
    'latent_mmd_scores': latent_mmd_scores,
    'overall_latent_mmd': overall_latent_mmd,
    'latent_plot_path': latent_plot_path,
    'latent_scatter_path': latent_plot_path,
    'conditioning_cache_path': stats_cache_path,
    'exp_dir': exp_dir,
        'distribution_summary': distribution_summary,
        'S1_indices': S1_indices,
        'S2_indices': S2_indices,
        'test_indices': test_indices,
        'N': N,
        'example_G1': example_G1,
        'example_G2': example_G2,
        'feature_homophily_mae': feature_homophily_mae,
        'feature_homophily_generated': generated_feature_homophily_values,
        'feature_homophily_targets': target_feature_homophily_values,
        'feature_homophily_errors': feature_homophily_errors,
        'feature_homophily_summary': feature_homophily_summary,
        'autoencoder_reconstruction': {
            'S1': recon_metrics_S1,
            'S2': recon_metrics_S2,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Feature-aware Graph Generation Study')
    parser.add_argument('--data-path', type=str,
                        default='data/featurehomophily0.2_graphs.pkl',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/convergence_feat_study',
                        help='Output directory')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{args.run_name}_{timestamp}"
    else:
        run_id = f"convergence_feat_{timestamp}"

    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Feature-aware Graph Generation Study")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Training sizes: {N_VALUES}")
    print(f"{'='*80}\n")

    data_list = load_dataset(args.data_path)

    if len(data_list) < TEST_SET_SIZE:
        print(f"Warning: Expected at least {TEST_SET_SIZE} graphs for test set, found {len(data_list)}")

    cache_dir = Path(args.data_path).parent / "cache"
    cache_path = cache_dir / f"{Path(args.data_path).stem}_test_stats_seed{SPLIT_SEED}_size{TEST_SET_SIZE}.pt"

    split_config = shuffle_and_split_dataset(
        data_list,
        test_size=TEST_SET_SIZE,
        seed=SPLIT_SEED,
        stats_cache_path=cache_path,
    )

    print("\nShuffled dataset split summary:")
    print(f"  Total graphs: {len(data_list)}")
    print(f"  S1 pool size: {len(split_config['S1_pool'])}")
    print(f"  S2 pool size: {len(split_config['S2_pool'])}")
    print(f"  Test pool size: {len(split_config['test_graphs'])} (held-out conditioning set)")

    all_results = {}

    for N in N_VALUES:
        results = run_experiment_for_N(N, split_config, output_dir)
        all_results[N] = results

        visualize_single_experiment(results)

        import pickle
        with open(output_dir / f'results_feat_N{N}.pkl', 'wb') as f:
            pickle.dump(results, f)

    visualize_results(all_results, output_dir)
    save_summary(all_results, output_dir)

    print(f"\n{'='*80}")
    print("Feature-aware Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Similarity distributions: {output_dir}/figures/similarity_distributions.png")
    print(f"  - Example graphs: {output_dir}/figures/example_graphs.png")
    print(f"  - Convergence curves: {output_dir}/figures/convergence_curves.png")
    print(f"  - Summary: {output_dir}/experiment_summary.txt")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
