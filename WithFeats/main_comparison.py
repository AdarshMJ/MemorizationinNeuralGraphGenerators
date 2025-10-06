"""
Graph Generation: Memorization to Generalization Transition Study

This script replicates the diffusion model convergence experiment from image generation
(see attached figure) but adapted for graph generation using VGAE + Latent Diffusion Models.

Experiment:
- Train two models on non-overlapping subsets S1 and S2
- Vary subset size N from 1 to 500 graphs
- Test if models transition from memorization (small N) to generalization (large N)
- Measure: At small N, models generate different graphs; at large N, they converge

Dataset: featurehomophily0.2_graphs.pkl (10000 graphs, homophily = 0.2)
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import pandas as pd

from torch_geometric.loader import DataLoader
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import (linear_beta_schedule, construct_nx_from_adj, 
                   eval_autoencoder, gen_stats)
from grakel import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

# Configuration
N_VALUES = [10, 20, 50, 100, 500, 1000, 2500, 3000]  # Training set sizes to test (avoiding extreme N=1)
TEST_SET_SIZE = 10  # Number of conditioning graphs held out from both models
SPLIT_SEED = 42  # Reproducible shuffle before creating S1/S2/test splits
NUM_SAMPLES_PER_CONDITION = 5  # Number of samples to draw per model per conditioning vector
KS_SIGNIFICANCE_THRESHOLD = 0.05  # Significance level for KS tests comparing S1/S2 statistics
BETA_KL_WEIGHT = 0.05
SMALL_DATASET_THRESHOLD = 50
SMALL_DATASET_KL_WEIGHT = 0.01
SMALL_DATASET_DROPOUT = 0.1

# Training hyperparameters - KEPT CONSTANT ACROSS ALL N VALUES
# This ensures no hidden confounders - only training set size varies
EPOCHS_AUTOENCODER = 100  # Increased from 100 for better convergence
EPOCHS_DENOISER = 100  # Increased from 50 for better convergence
EARLY_STOPPING_PATIENCE = 50  # Stop if no improvement for 30 epochs
BATCH_SIZE = 32  # Same batch size for all experiments
LEARNING_RATE = 0.0001  # Same learning rate for all experiments
GRAD_CLIP = 1.0  # Gradient clipping to prevent exploding gradients
LATENT_DIM = 32  # Same latent dimension for all experiments
HIDDEN_DIM_ENCODER = 64  # Same encoder architecture for all experiments
HIDDEN_DIM_DECODER = 128  # Same decoder architecture for all experiments
HIDDEN_DIM_DENOISE = 512  # Same denoiser architecture for all experiments
N_MAX_NODES = 100  # Same max nodes for all experiments
N_PROPERTIES = 18  # Conditioning properties include homophily measurements
TIMESTEPS = 100  # Same diffusion timesteps for all experiments

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_dataset(data_path='data/featurehomophily0.2_graphs.pkl'):
    """Load the graph dataset."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"Loaded {len(data_list)} graphs")

    # Inject measured homophily values from log if available so conditioning stays accurate
    log_path = Path(data_path).with_name(Path(data_path).name.replace('_graphs.pkl', '_log.csv'))
    homophily_log = None
    if log_path.exists():
        try:
            homophily_log = pd.read_csv(log_path).set_index('graph_idx')
            print(f"Loaded homophily log from {log_path}")
        except Exception as exc:
            print(f"Warning: failed to load homophily log ({exc}).")

    for idx, data in enumerate(data_list):
        if homophily_log is not None:
            log_idx = int(getattr(data, 'graph_idx', idx))
            if log_idx in homophily_log.index:
                row = homophily_log.loc[log_idx]
                if hasattr(data, 'feature_homophily'):
                    data.feature_homophily = torch.tensor(float(row['actual_feature_hom']))
                if hasattr(data, 'stats'):
                    stats = data.stats
                    if isinstance(stats, torch.Tensor):
                        if stats.dim() == 1:
                            stats = stats.unsqueeze(0)
                        if stats.size(-1) < 18:
                            stats = torch.nn.functional.pad(stats, (0, 18 - stats.size(-1)))
                        data.stats = stats
                        data.stats[0, -3:] = torch.tensor([
                            float(row['actual_label_hom']),
                            float(row['actual_structural_hom']),
                            float(row['actual_feature_hom'])
                        ])
                if not hasattr(data, 'graph_idx'):
                    data.graph_idx = log_idx
    
    # Basic info
    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Graph properties: {sample.x.shape[1]}D features, {sample.x.shape[0]} nodes")
        if hasattr(sample, 'feature_homophily'):
            print(f"Feature homophily: {sample.feature_homophily:.2f}")
    
    return data_list


def shuffle_and_split_dataset(data_list, test_size=TEST_SET_SIZE, seed=SPLIT_SEED, stats_cache_path=None):
    """Shuffle dataset and create reusable pools for S1, S2, and held-out test graphs."""
    total_graphs = len(data_list)
    if total_graphs < test_size + 2 * max(N_VALUES):
        raise ValueError(
            f"Dataset too small for requested configuration: need at least "
            f"{test_size + 2 * max(N_VALUES)} graphs, found {total_graphs}."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(total_graphs)
    rng.shuffle(indices)

    shuffled_data = [data_list[idx] for idx in indices]

    test_graphs = shuffled_data[-test_size:]
    test_indices = indices[-test_size:]

    train_pool = shuffled_data[:-test_size]
    train_indices = indices[:-test_size]

    half = len(train_pool) // 2
    if half < max(N_VALUES):
        raise ValueError(
            f"Not enough data in each training split after shuffling. "
            f"Available per split: {half}, required: {max(N_VALUES)}"
        )

    S1_pool = train_pool[:half]
    S1_indices = train_indices[:half]
    S2_pool = train_pool[half:]
    S2_indices = train_indices[half:]

    test_stats_cache = None
    if stats_cache_path is not None and stats_cache_path.exists():
        try:
            cache_payload = torch.load(stats_cache_path, map_location='cpu')
            cache_props = cache_payload.get('n_properties')
            if cache_payload.get('test_size') == test_size and cache_payload.get('seed') == seed and cache_props == N_PROPERTIES:
                cached_indices = np.asarray(cache_payload.get('test_indices'))
                if cached_indices is not None and np.array_equal(cached_indices, test_indices):
                    cached_stats = cache_payload.get('stats')
                    if isinstance(cached_stats, torch.Tensor):
                        test_stats_cache = cached_stats.float()
                    elif cached_stats is not None:
                        test_stats_cache = torch.tensor(cached_stats, dtype=torch.float32)
                    if test_stats_cache is not None:
                        print(f"Loaded cached conditioning stats from {stats_cache_path}")
        except Exception as exc:
            print(f"Warning: failed to load conditioning cache ({exc}); recomputing.")

    if test_stats_cache is None:
        stats_matrix = stack_stats(test_graphs)
        if stats_matrix.size > 0:
            test_stats_cache = torch.from_numpy(stats_matrix).float()
        else:
            test_stats_cache = torch.empty((0, N_PROPERTIES), dtype=torch.float32)

        if stats_cache_path is not None:
            try:
                stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'seed': seed,
                    'test_size': test_size,
                    'test_indices': test_indices,
                    'stats': test_stats_cache.cpu(),
                    'n_properties': N_PROPERTIES
                }, stats_cache_path)
                print(f"Saved conditioning stats cache to {stats_cache_path}")
            except Exception as exc:
                print(f"Warning: failed to save conditioning cache ({exc}).")

    return {
        'S1_pool': S1_pool,
        'S1_indices': S1_indices,
        'S2_pool': S2_pool,
        'S2_indices': S2_indices,
        'test_graphs': test_graphs,
        'test_indices': test_indices,
        'test_stats_cache': test_stats_cache,
        'stats_cache_path': stats_cache_path,
        'permutation': indices,
        'seed': seed,
        'test_size': test_size
    }


def create_splits(split_config, N):
    """Create non-overlapping splits of size N from pre-shuffled pools."""
    S1_pool = split_config['S1_pool']
    S2_pool = split_config['S2_pool']
    S1_indices_pool = split_config['S1_indices']
    S2_indices_pool = split_config['S2_indices']
    test_stats_cache = split_config.get('test_stats_cache')

    if N > len(S1_pool) or N > len(S2_pool):
        raise ValueError(
            f"Requested N={N} exceeds available pool sizes: "
            f"S1={len(S1_pool)}, S2={len(S2_pool)}"
        )

    S1 = S1_pool[:N]
    S2 = S2_pool[:N]
    S1_indices = S1_indices_pool[:N]
    S2_indices = S2_indices_pool[:N]
    test_graphs = split_config['test_graphs']
    test_indices = split_config['test_indices']

    print(f"\nData splits for N={N} (seed={split_config['seed']}):")
    print(f"  S1: {len(S1)} graphs | original indices min={int(np.min(S1_indices))}, "
          f"max={int(np.max(S1_indices))}")
    print(f"  S2: {len(S2)} graphs | original indices min={int(np.min(S2_indices))}, "
          f"max={int(np.max(S2_indices))}")
    print(f"  Test set: {len(test_graphs)} graphs | original indices range="
        f" [{int(np.min(test_indices))}, {int(np.max(test_indices))}]")
    if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) == len(test_graphs):
      print("  Test conditioning stats pulled from cache")

    return S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices


def graph_stats_to_numpy(data):
    """Convert stored graph statistics to a numpy array limited to N_PROPERTIES."""
    if not hasattr(data, 'stats'):
        raise AttributeError("Graph data object is missing required 'stats' attribute.")

    stats = data.stats
    if isinstance(stats, torch.Tensor):
        stats_np = stats.detach().cpu().numpy()
    else:
        stats_np = np.asarray(stats)

    if stats_np.ndim == 1:
        stats_np = stats_np.reshape(1, -1)

    if stats_np.shape[1] < N_PROPERTIES:
        padding = N_PROPERTIES - stats_np.shape[1]
        stats_np = np.pad(stats_np, ((0, 0), (0, padding)), constant_values=np.nan)

    return stats_np[:, :N_PROPERTIES]


def stack_stats(graph_list):
    """Stack per-graph statistics into a matrix of shape [len(graphs), N_PROPERTIES]."""
    if len(graph_list) == 0:
        return np.empty((0, N_PROPERTIES))

    stats_mats = []
    for data in graph_list:
        try:
            stats_mats.append(graph_stats_to_numpy(data))
        except Exception as exc:
            print(f"Warning: failed to extract stats for a graph ({exc}).")

    if len(stats_mats) == 0:
        return np.empty((0, N_PROPERTIES))

    return np.vstack(stats_mats)


def ks_2samp(sample1, sample2):
    """Two-sample Kolmogorov-Smirnov test (asymptotic p-value approximation)."""
    data1 = np.sort(np.asarray(sample1))
    data2 = np.sort(np.asarray(sample2))

    n1 = data1.size
    n2 = data2.size

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.abs(cdf1 - cdf2))

    en = np.sqrt(n1 * n2 / (n1 + n2))
    if en == 0:
        return d, np.nan

    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-8:
        return d, 1.0

    p_value = 0.0
    for j in range(1, 100):
        term = 2 * ((-1) ** (j - 1)) * np.exp(-2 * (lam ** 2) * (j ** 2))
        p_value += term
        if abs(term) < 1e-8:
            break

    p_value = float(max(0.0, min(1.0, p_value)))
    return d, p_value


def perform_distribution_checks(S1_subset, S2_subset, exp_dir, N):
    """Run KS tests on graph statistics to confirm S1/S2 parity."""
    stats_S1 = stack_stats(S1_subset)
    stats_S2 = stack_stats(S2_subset)

    output_path = exp_dir / f"ks_tests_N{N}.txt"
    results = []

    for feature_idx in range(N_PROPERTIES):
        col1 = stats_S1[:, feature_idx] if stats_S1.size else np.array([])
        col2 = stats_S2[:, feature_idx] if stats_S2.size else np.array([])

        col1 = col1[~np.isnan(col1)]
        col2 = col2[~np.isnan(col2)]

        if col1.size == 0 or col2.size == 0:
            d_val, p_val = np.nan, np.nan
        elif np.allclose(col1, col1[0]) and np.allclose(col2, col2[0]) and np.isclose(col1[0], col2[0]):
            d_val, p_val = 0.0, 1.0
        else:
            d_val, p_val = ks_2samp(col1, col2)

        results.append({
            'feature': feature_idx,
            'statistic': d_val,
            'p_value': p_val
        })

    significant = [r for r in results if not np.isnan(r['p_value']) and r['p_value'] < KS_SIGNIFICANCE_THRESHOLD]

    with open(output_path, 'w') as f:
        f.write(f"KS Tests for N={N}\n")
        f.write("Feature\tD-stat\tP-value\n")
        for row in results:
            f.write(f"{row['feature']}\t{row['statistic']:.4f}\t{row['p_value']}\n")
        f.write("\n")
        f.write(f"Significance threshold: {KS_SIGNIFICANCE_THRESHOLD}\n")
        f.write(f"Number of features failing (p < threshold): {len(significant)}\n")

    if significant:
        print(f"⚠️  KS test detected {len(significant)} / {N_PROPERTIES} features with distribution drift (see {output_path}).")
    else:
        print(f"KS tests passed for N={N}; all feature distributions aligned (see {output_path}).")

    return {
        'results': results,
        'num_failures': len(significant),
        'output_path': output_path
    }


def compute_within_model_similarity(graphs):
    """Compute pairwise WL similarities within a set of generated graphs."""
    if len(graphs) < 2:
        return []

    scores = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            scores.append(compute_wl_similarity(graphs[i], graphs[j]))
    return scores


def compute_mmd_rbf(latents_a, latents_b, bandwidths=None):
    """Compute squared MMD with an RBF kernel between two latent code sets."""
    if latents_a is None or latents_b is None:
        return np.nan

    if isinstance(latents_a, np.ndarray):
        latents_a = torch.from_numpy(latents_a)
    if isinstance(latents_b, np.ndarray):
        latents_b = torch.from_numpy(latents_b)

    if latents_a.numel() == 0 or latents_b.numel() == 0:
        return np.nan

    latents_a = latents_a.float()
    latents_b = latents_b.float()

    if latents_a.dim() == 1:
        latents_a = latents_a.unsqueeze(0)
    if latents_b.dim() == 1:
        latents_b = latents_b.unsqueeze(0)

    with torch.no_grad():
        combined = torch.cat([latents_a, latents_b], dim=0)
        dists = torch.cdist(combined, combined, p=2).pow(2)
        positive_dists = dists[dists > 0]
        if bandwidths is None:
            if positive_dists.numel() == 0:
                bandwidths = [1.0]
            else:
                median_sq = torch.median(positive_dists)
                gamma = 1.0 / (2.0 * median_sq)
                bandwidths = [gamma, gamma * 0.5, gamma * 2.0]

        if isinstance(bandwidths, (list, tuple)):
            gammas = [float(g) for g in bandwidths if g > 0]
        else:
            gammas = [float(bandwidths)]

        if not gammas:
            gammas = [1.0]

        d_xx = torch.cdist(latents_a, latents_a, p=2).pow(2)
        d_yy = torch.cdist(latents_b, latents_b, p=2).pow(2)
        d_xy = torch.cdist(latents_a, latents_b, p=2).pow(2)

        mmd_total = 0.0
        for gamma in gammas:
            k_xx = torch.exp(-gamma * d_xx)
            k_yy = torch.exp(-gamma * d_yy)
            k_xy = torch.exp(-gamma * d_xy)
            mmd_total += k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    return float(torch.clamp(mmd_total / len(gammas), min=0.0).item())


def save_latent_projection(latents_s1, latents_s2, output_path):
    """Project latent codes to 2D via PCA and save a scatter plot."""
    if not latents_s1 or not latents_s2:
        return None

    latents_s1 = [tensor.float() for tensor in latents_s1 if tensor.numel() > 0]
    latents_s2 = [tensor.float() for tensor in latents_s2 if tensor.numel() > 0]
    if not latents_s1 or not latents_s2:
        return None

    lat_s1 = torch.cat(latents_s1, dim=0)
    lat_s2 = torch.cat(latents_s2, dim=0)
    if lat_s1.numel() == 0 or lat_s2.numel() == 0:
        return None

    combined = torch.cat([lat_s1, lat_s2], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean
    latent_dim = centered.shape[1]
    if latent_dim == 0:
        return None

    q = min(2, latent_dim)
    try:
        _, _, v = torch.pca_lowrank(centered, q=q)
    except RuntimeError:
        # Fallback to SVD if PCA fails (e.g., low-rank issues)
        u, _, vh = torch.linalg.svd(centered, full_matrices=False)
        v = vh.t()
    components = v[:, :q]

    proj_s1 = (lat_s1 - mean) @ components
    proj_s2 = (lat_s2 - mean) @ components

    if q == 1:
        proj_s1 = torch.cat([proj_s1, torch.zeros_like(proj_s1)], dim=1)
        proj_s2 = torch.cat([proj_s2, torch.zeros_like(proj_s2)], dim=1)

    proj_s1_np = proj_s1[:, :2].detach().cpu().numpy()
    proj_s2_np = proj_s2[:, :2].detach().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(proj_s1_np[:, 0], proj_s1_np[:, 1], alpha=0.6, color='tab:blue', label='Model S1')
    ax.scatter(proj_s2_np[:, 0], proj_s2_np[:, 1], alpha=0.6, color='tab:orange', label='Model S2')
    ax.set_xlabel('PC1', fontsize=25)
    ax.set_ylabel('PC2', fontsize=25)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def train_autoencoder(data_list, run_name, output_dir):
    """Train VGAE on given data."""
    print(f"\n{'='*80}")
    print(f"Training Autoencoder: {run_name}")
    print(f"{'='*80}")
    
    # Get feature dimension
    input_feat_dim = data_list[0].x.shape[1]
    
    # Create model
    autoencoder = VariationalAutoEncoder(
        input_dim=input_feat_dim,
        hidden_dim_enc=HIDDEN_DIM_ENCODER,
        hidden_dim_dec=HIDDEN_DIM_DECODER,
        latent_dim=LATENT_DIM,
        n_layers_enc=2,
        n_layers_dec=3,
        n_max_nodes=N_MAX_NODES
    ).to(device)
    
    dataset_size = len(data_list)
    beta_value = BETA_KL_WEIGHT
    if dataset_size <= SMALL_DATASET_THRESHOLD:
        beta_value = SMALL_DATASET_KL_WEIGHT
        autoencoder.encoder.dropout = SMALL_DATASET_DROPOUT
        print(f"Small dataset detected ({dataset_size} graphs) → KL beta set to {beta_value}, encoder dropout={SMALL_DATASET_DROPOUT}")
    
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    patience_counter = 0
    best_epoch = 0
    best_val_loss_per_graph = np.inf
    checkpoint_path = output_dir / f'autoencoder_{run_name}.pth.tar'
    
    # Data loader (using 80% for train, 20% for val)
    # For small datasets (N < 10), use all data for both train and val to avoid extreme cases
    if len(data_list) < 10:
        train_data = data_list
        val_data = data_list
    else:
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train] if n_train > 0 else data_list
        val_data = data_list[n_train:] if n_train > 0 and len(data_list) > 1 else train_data
    
    # Use consistent batch size across all experiments (capped by data size)
    batch_size = min(BATCH_SIZE, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(BATCH_SIZE, len(val_data)), shuffle=False)
    
    best_val_loss = np.inf  # keep for backward compatibility
    
    edges_per_graph = N_MAX_NODES * N_MAX_NODES

    for epoch in range(1, EPOCHS_AUTOENCODER + 1):
        autoencoder.train()
        
        train_loss_all = 0
        train_count = 0
        train_recon_sum = 0
        train_kld_sum = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld = autoencoder.loss_function(data, beta=beta_value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), GRAD_CLIP)
            train_loss_all += loss.item()
            train_recon_sum += recon.item()
            train_kld_sum += kld.item()
            train_count += int(torch.max(data.batch).item()) + 1
            optimizer.step()
        
        # Validation
        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        val_recon_sum = 0
        val_kld_sum = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                loss, recon, kld = autoencoder.loss_function(data, beta=beta_value)
                val_loss_all += loss.item()
                val_recon_sum += recon.item()
                val_kld_sum += kld.item()
                val_count += int(torch.max(data.batch).item()) + 1
        
        train_loss_avg = train_loss_all / train_count if train_count else np.nan
        val_loss_avg = val_loss_all / val_count if val_count else np.nan
        train_recon_avg = train_recon_sum / train_count if train_count else np.nan
        val_recon_avg = val_recon_sum / val_count if val_count else np.nan
        train_per_edge = train_recon_avg / edges_per_graph if np.isfinite(train_recon_avg) else np.nan
        val_per_edge = val_recon_avg / edges_per_graph if np.isfinite(val_recon_avg) else np.nan

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: Train Loss: {train_loss_avg:.2f} | Val Loss: {val_loss_avg:.2f} "
                f"(per-edge train MAE {train_per_edge:.4f}, val MAE {val_per_edge:.4f})"
            )
        
        scheduler.step()
        
        # Early stopping logic
        if val_loss_avg < best_val_loss_per_graph:
            best_val_loss_per_graph = val_loss_avg
            best_val_loss = val_loss_all
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss_per_graph': best_val_loss_per_graph,
                'beta': beta_value,
            }, checkpoint_path)
            if epoch % 20 == 0:
                print('  → New best model saved!')
        else:
            patience_counter += 1
            if epoch % 20 == 0 and patience_counter > 0:
                print(f'  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')
        
        # Check if should stop early
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', best_epoch)
        best_val_loss_per_graph = checkpoint.get('best_val_loss_per_graph', best_val_loss_per_graph)
    else:
        print('Warning: Best checkpoint not found; returning last epoch weights.')

    best_val_mae = best_val_loss_per_graph / edges_per_graph if np.isfinite(best_val_loss_per_graph) else np.nan
    print(
        f"Autoencoder training complete. Best val loss/graph: {best_val_loss_per_graph:.2f}"
        f" (per-edge MAE {best_val_mae:.4f}) at epoch {best_epoch}"
    )

    # Post-training diagnostics
    def reconstruction_metrics(loader):
        if loader is None or len(loader.dataset) == 0:
            return np.nan, np.nan
        autoencoder.eval()
        total_abs = 0.0
        total_entries = 0
        total_correct = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = autoencoder(batch)
                targets = batch.A
                total_abs += torch.abs(preds - targets).sum().item()
                total_entries += targets.numel()
                total_correct += ((preds > 0.5).float() == targets).float().sum().item()
        mae = total_abs / total_entries if total_entries else np.nan
        acc = total_correct / total_entries if total_entries else np.nan
        return mae, acc

    train_mae, train_acc = reconstruction_metrics(train_loader)
    val_mae, val_acc = reconstruction_metrics(val_loader)
    print(f"  Reconstruction MAE (train): {train_mae:.4f}, accuracy: {train_acc:.4f}")
    if np.isfinite(val_mae):
        print(f"  Reconstruction MAE (val):   {val_mae:.4f}, accuracy: {val_acc:.4f}")
    return autoencoder


def train_denoiser(autoencoder, data_list, run_name, output_dir):
    """Train latent diffusion model."""
    print(f"\n{'='*80}")
    print(f"Training Denoiser: {run_name}")
    print(f"{'='*80}")
    
    # Create denoiser
    denoise_model = DenoiseNN(
        input_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM_DENOISE,
        n_layers=3,
        n_cond=N_PROPERTIES,
        d_cond=128
    ).to(device)
    
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # Beta schedule for diffusion
    betas = linear_beta_schedule(timesteps=TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)
    
    # Data loader
    # For small datasets (N < 10), use all data for both train and val to avoid extreme cases
    if len(data_list) < 10:
        train_data = data_list
        val_data = data_list
    else:
        n_train = int(0.8 * len(data_list))
        train_data = data_list[:n_train] if n_train > 0 else data_list
        val_data = data_list[n_train:] if n_train > 0 and len(data_list) > 1 else train_data
    
    # Use consistent batch size across all experiments (capped by data size)
    batch_size = min(BATCH_SIZE, len(train_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=min(BATCH_SIZE, len(val_data)), shuffle=False)
    
    autoencoder.eval()
    best_val_loss_per_graph = np.inf
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = output_dir / f'denoise_{run_name}.pth.tar'
    
    for epoch in range(1, EPOCHS_DENOISER + 1):
        denoise_model.train()
        
        train_loss_all = 0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                x_g = autoencoder.encode(data)
            
            # Ensure stats are in correct shape [batch_size, n_properties]
            stats = data.stats[:, :N_PROPERTIES].reshape(-1, N_PROPERTIES)
            
            t = torch.randint(0, TIMESTEPS, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, stats,
                          sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                          loss_type="huber")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), GRAD_CLIP)
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()
        
        # Validation
        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                
                # Ensure stats are in correct shape [batch_size, n_properties]
                stats = data.stats[:, :N_PROPERTIES].reshape(-1, N_PROPERTIES)
                
                t = torch.randint(0, TIMESTEPS, (x_g.size(0),), device=device).long()
                loss = p_losses(denoise_model, x_g, t, stats,
                              sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                              loss_type="huber")
                val_loss_all += x_g.size(0) * loss.item()
                val_count += x_g.size(0)
        
        train_loss_avg = train_loss_all / train_count if train_count else np.nan
        val_loss_avg = val_loss_all / val_count if val_count else np.nan
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}: Train Loss: {train_loss_avg:.4f}, '
                  f'Val Loss: {val_loss_avg:.4f}')
        
        scheduler.step()
        
        # Early stopping logic
        if val_loss_avg < best_val_loss_per_graph:
            best_val_loss_per_graph = val_loss_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss_per_graph': best_val_loss_per_graph
            }, checkpoint_path)
            if epoch % 10 == 0:
                print('  → New best model saved!')
        else:
            patience_counter += 1
            if epoch % 10 == 0 and patience_counter > 0:
                print(f'  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})')
        
        # Check if should stop early
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        denoise_model.load_state_dict(checkpoint['state_dict'])
        best_epoch = checkpoint.get('epoch', best_epoch)
        best_val_loss_per_graph = checkpoint.get('best_val_loss_per_graph', best_val_loss_per_graph)
    else:
        print('Warning: Best denoiser checkpoint not found; returning last epoch weights.')

    print(f"Denoiser training complete. Best val loss: {best_val_loss_per_graph:.4f} (epoch {best_epoch})")
    
    # Return betas for generation
    return denoise_model, betas


def generate_graphs(autoencoder, denoise_model, conditioning_stats, betas, num_samples=1):
    """Generate graphs and return decoded adjacencies plus latent codes used for decoding."""
    autoencoder.eval()
    denoise_model.eval()

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # Prepare conditioning (ensure correct shape)
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
            batch_size=num_samples
        )
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample)

        graphs = []
        adj_matrices = []
        latent_codes = x_sample.detach().cpu()
        for idx in range(num_samples):
            adj_np = adj[idx].detach().cpu().numpy()
            graphs.append(construct_nx_from_adj(adj_np))
            adj_matrices.append(adj_np)

    return graphs, adj_matrices, latent_codes


def compute_wl_similarity(G1, G2):
    """
    Compute Weisfeiler-Lehman kernel similarity between two graphs.
    
    Uses node degree as labels (topology-derived labels) since generated graphs
    don't have semantic node labels. When node feature vectors are available
    (stored under the `feature_vector` attribute), we augment the topology label
    with a coarse discretization of the first few feature dimensions so the WL
    kernel is sensitive to feature homophily as well.
    """
    # Handle empty graphs (can happen with small training sets)
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        # Both empty: identical
        if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
            return 1.0
        # One empty, one not: completely different
        else:
            return 0.0
    
    try:
        # Work on copies to avoid mutating upstream graphs
        G1_local = G1.copy()
        G2_local = G2.copy()

        def feature_signature(node_data):
            vec = node_data.get('feature_vector')
            if vec is None:
                return 'feat:none'
            vec = np.asarray(vec, dtype=float)
            if vec.size == 0:
                return 'feat:none'
            capped = vec[:4]
            signature = []
            for value in capped:
                if value < -0.25:
                    signature.append('n')
                elif value > 0.25:
                    signature.append('p')
                else:
                    signature.append('z')
            if vec.size > 4:
                mean_rest = float(np.mean(vec[4:]))
                if mean_rest < -0.25:
                    signature.append('rN')
                elif mean_rest > 0.25:
                    signature.append('rP')
                else:
                    signature.append('rZ')
            return 'feat:' + ''.join(signature)

        def annotate_graph(G_local):
            for node in G_local.nodes():
                base_label = int(G_local.degree(node))
                feat_sig = feature_signature(G_local.nodes[node])
                G_local.nodes[node]['label'] = f"{base_label}|{feat_sig}"
            return G_local

        G1_local = annotate_graph(G1_local)
        G2_local = annotate_graph(G2_local)
        
        # Convert to grakel format
        graphs_pair = graph_from_networkx([G1_local, G2_local], node_labels_tag='label')
        
        # Compute WL kernel
        wl_kernel = WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)
        K = wl_kernel.fit_transform(graphs_pair)
        
        # Similarity is off-diagonal element
        similarity = K[0, 1]
        return similarity
    except Exception as e:
        # Fallback: simple edge overlap
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        if len(edges1) == 0 and len(edges2) == 0:
            return 1.0
        union = len(edges1.union(edges2))
        intersection = len(edges1.intersection(edges2))
        return intersection / union if union > 0 else 0.0


def compute_statistics_distance(G1, G2):
    """Compute MSE and MAE between graph statistics."""
    try:
        stats1 = gen_stats(G1)
        stats2 = gen_stats(G2)
        
        # Convert to numpy arrays (first 15 features)
        stats1 = np.array(stats1[:N_PROPERTIES])
        stats2 = np.array(stats2[:N_PROPERTIES])
        
        mse = np.mean((stats1 - stats2) ** 2)
        mae = np.mean(np.abs(stats1 - stats2))
        
        return mse, mae
    except Exception as e:
        print(f"Warning: Statistics computation failed ({e})")
        return np.nan, np.nan


def find_closest_graph_in_training(generated_G, training_graphs_nx):
    """Find the most similar graph in training set to the generated graph."""
    best_sim = -1
    closest_G = None
    
    for G_train in training_graphs_nx:
        # Compute similarity
        sim = compute_wl_similarity(generated_G, G_train)
        
        if sim > best_sim:
            best_sim = sim
            closest_G = G_train
    
    return closest_G, best_sim


def run_experiment_for_N(N, split_config, output_dir):
    """
    Run the complete experiment for a specific training size N.
    
    Returns:
        generalization_scores: WL similarities between paired samples from S1/S2
        generalization_variances: Per-conditioning variance of generalization scores
        memorization_scores: WL similarities between generated samples and nearest S1 graph
        stats_mse: MSE between statistics of paired samples
        stats_mae: MAE between statistics of paired samples
        within_s1_similarity: Pairwise WL similarities within S1-generated samples
        within_s2_similarity: Pairwise WL similarities within S2-generated samples
        example_G1/example_G2: First generated graphs for visualization
        distribution_summary: KS test results ensuring S1/S2 parity
        S1_indices/S2_indices/test_indices: Original dataset indices used for transparency
        N: Training set size (echoed for convenience)
    """
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: N = {N}")
    print(f"{'#'*80}")
    
    # Create splits
    S1, S2, test_graphs, test_stats_cache, S1_indices, S2_indices, test_indices = create_splits(split_config, N)
    stats_cache_path = split_config.get('stats_cache_path')
    if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) == len(test_graphs) and stats_cache_path:
        print(f"  Using conditioning cache: {stats_cache_path}")
    
    # Create output directories for this N
    exp_dir = output_dir / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Check distribution parity between S1 and S2 statistics
    distribution_summary = perform_distribution_checks(S1, S2, exp_dir, N)
    
    # Precompute NetworkX versions of training graphs for memorization checks
    S1_training_graphs_nx = []
    for data in S1:
        adj = data.A[0].cpu().numpy()
        features = data.x.detach().cpu().numpy() if hasattr(data, 'x') else None
        S1_training_graphs_nx.append(construct_nx_from_adj(adj, node_features=features))
    
    s1_min_idx = int(np.min(S1_indices))
    s1_max_idx = int(np.max(S1_indices))
    s2_min_idx = int(np.min(S2_indices))
    s2_max_idx = int(np.max(S2_indices))

    # Train Model S1
    print(f"\n--- Training Model S1 (original indices {s1_min_idx}–{s1_max_idx}) ---")
    autoencoder_S1 = train_autoencoder(S1, f"S1_N{N}", exp_dir)
    denoise_S1, betas = train_denoiser(autoencoder_S1, S1, f"S1_N{N}", exp_dir)
    
    # Train Model S2
    print(f"\n--- Training Model S2 (original indices {s2_min_idx}–{s2_max_idx}) ---")
    autoencoder_S2 = train_autoencoder(S2, f"S2_N{N}", exp_dir)
    denoise_S2, _ = train_denoiser(autoencoder_S2, S2, f"S2_N{N}", exp_dir)
    
    # Generation and evaluation
    print(f"\n--- Generating and Evaluating ---")
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
    example_G1, example_G2 = None, None
    
    for i, test_graph in enumerate(tqdm(test_graphs, desc="Testing")):
        # Extract conditioning statistics
        if isinstance(test_stats_cache, torch.Tensor) and len(test_stats_cache) > i:
            c_test = test_stats_cache[i:i+1]
        else:
            c_test = test_graph.stats[:, :N_PROPERTIES]
        
        # Generate from both models
        G1_samples, _, latents_G1 = generate_graphs(autoencoder_S1, denoise_S1, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)
        G2_samples, _, latents_G2 = generate_graphs(autoencoder_S2, denoise_S2, c_test, betas, num_samples=NUM_SAMPLES_PER_CONDITION)

        aggregated_latents_s1.append(latents_G1)
        aggregated_latents_s2.append(latents_G2)

        mmd_value = compute_mmd_rbf(latents_G1, latents_G2)
        latent_mmd_scores.append(mmd_value)
        
        # Save first example for visualization
        if i == 0:
            example_G1, example_G2 = G1_samples[0], G2_samples[0]
        
        # Compute generalization: similarity between paired samples from G1 and G2
        pairwise_generalization = []
        for sample_idx in range(NUM_SAMPLES_PER_CONDITION):
            sim_generalization = compute_wl_similarity(G1_samples[sample_idx], G2_samples[sample_idx])
            generalization_scores.append(sim_generalization)
            pairwise_generalization.append(sim_generalization)
            
            # Compute statistics distance for each paired sample
            mse, mae = compute_statistics_distance(G1_samples[sample_idx], G2_samples[sample_idx])
            stats_mse_list.append(mse)
            stats_mae_list.append(mae)
        
        if len(pairwise_generalization) > 1:
            generalization_variances.append(np.var(pairwise_generalization))
        else:
            generalization_variances.append(0.0)
        
        # Compute memorization: similarity of first sample to closest in training
        _, sim_memorization = find_closest_graph_in_training(G1_samples[0], S1_training_graphs_nx)
        memorization_scores.append(sim_memorization)

        # Measure within-model variability via pairwise similarities
        within_s1_similarity.extend(compute_within_model_similarity(G1_samples))
        within_s2_similarity.extend(compute_within_model_similarity(G2_samples))
    
    if aggregated_latents_s1 and aggregated_latents_s2:
        all_latents_s1 = torch.cat(aggregated_latents_s1, dim=0)
        all_latents_s2 = torch.cat(aggregated_latents_s2, dim=0)
        overall_latent_mmd = compute_mmd_rbf(all_latents_s1, all_latents_s2)
        latent_plot_path = save_latent_projection(
            aggregated_latents_s1,
            aggregated_latents_s2,
            exp_dir / "quicklook" / f"latent_scatter_N{N}.png"
        )
    else:
        overall_latent_mmd = np.nan
        latent_plot_path = None

    print(f"\nResults for N={N}:")
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
    
    return {
        'generalization_scores': generalization_scores,
        'generalization_variances': generalization_variances,
        'memorization_scores': memorization_scores,
        'stats_mse': stats_mse_list,
        'stats_mae': stats_mae_list,
        'within_s1_similarity': within_s1_similarity,
        'within_s2_similarity': within_s2_similarity,
        'example_G1': example_G1,
        'example_G2': example_G2,
    'distribution_summary': distribution_summary,
    'latent_mmd_per_condition': latent_mmd_scores,
    'latent_mmd_overall': overall_latent_mmd,
    'conditioning_cache_path': stats_cache_path,
    'latent_scatter_path': latent_plot_path,
        'S1_indices': S1_indices,
        'S2_indices': S2_indices,
        'test_indices': test_indices,
        'N': N,
        'exp_dir': exp_dir
    }


def visualize_single_experiment(results):
    """Save quick-look plots for a single N immediately after the run."""
    exp_dir = results['exp_dir']
    quick_dir = exp_dir / "quicklook"
    quick_dir.mkdir(exist_ok=True)

    N = results['N']
    gen_scores = results['generalization_scores']
    mem_scores = results['memorization_scores']

    # Histogram comparison of memorization vs generalization
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(mem_scores):
        ax.hist(mem_scores, bins=20, alpha=0.7, color='orange', range=(0, 1), label='Memorization')
    if len(gen_scores):
        ax.hist(gen_scores, bins=20, alpha=0.7, color='blue', range=(0, 1), label='Generalization')
    ax.set_xlabel('WL Kernel Similarity', fontsize=25)
    ax.set_ylabel('Frequency', fontsize=25)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    hist_path = quick_dir / f"hist_N{N}.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Example graphs from S1 and S2
    example_G1 = results['example_G1']
    example_G2 = results['example_G2']
    if example_G1 is not None and example_G2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # S1 graph (left) - blue
        pos1 = nx.kamada_kawai_layout(example_G1) if example_G1.number_of_nodes() > 1 else nx.spring_layout(example_G1, seed=42)
        nx.draw_networkx_nodes(example_G1, pos1, ax=axes[0], node_color='#3498db', node_size=150, alpha=0.9)
        nx.draw_networkx_edges(example_G1, pos1, ax=axes[0], edge_color='#34495e', width=0.8, alpha=0.6)
        axes[0].set_axis_off()
        axes[0].set_title('S1', fontsize=16, fontweight='bold', color='#3498db')
        
        # S2 graph (right) - orange
        pos2 = nx.kamada_kawai_layout(example_G2) if example_G2.number_of_nodes() > 1 else nx.spring_layout(example_G2, seed=42)
        nx.draw_networkx_nodes(example_G2, pos2, ax=axes[1], node_color='#e67e22', node_size=150, alpha=0.9)
        nx.draw_networkx_edges(example_G2, pos2, ax=axes[1], edge_color='#34495e', width=0.8, alpha=0.6)
        axes[1].set_axis_off()
        axes[1].set_title('S2', fontsize=16, fontweight='bold', color='#e67e22')
        
        examples_path = quick_dir / f"examples_N{N}.png"
        plt.tight_layout()
        plt.savefig(examples_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        examples_path = None

    print(f"Quicklook saved for N={N}: {hist_path}")
    if examples_path:
        print(f"Quicklook graphs saved for N={N}: {examples_path}")
    latent_path = results.get('latent_scatter_path')
    if latent_path:
        print(f"Latent scatter saved for N={N}: {latent_path}")


def visualize_results(all_results, output_dir):
    """
    Create visualizations similar to Figure 2 in the paper.
    
    1. Distribution histograms for each N (blue=generalization, orange=memorization)
    2. Example graphs for small N vs large N
    """
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")
    print("\nIMPORTANT: All experiments used identical hyperparameters.")
    print("Only the training set size (N) was varied.")
    print(f"This ensures fair comparison with no hidden confounders.\n")
    
    # Create figure directory
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Figure 1: Distribution histograms for different N values
    n_plots = len(N_VALUES)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for idx, N in enumerate(N_VALUES):
        results = all_results[N]
        ax = axes[idx]
        
        gen_scores = results['generalization_scores']
        mem_scores = results['memorization_scores']
        
        # Plot histograms
        ax.hist(mem_scores, bins=20, alpha=0.3, color='orange', 
        label='Samples and closest graph', range=(0, 1))
        ax.hist(gen_scores, bins=20, alpha=0.3, color='blue', 
                label='Samples from two denoisers', range=(0, 1))
        
        ax.set_xlabel('WL Kernel Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'N = {N}', fontsize=14)
        ax.set_xlim(0, 1)
        
        if idx == n_plots - 1:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'similarity_distributions.png'}")
    plt.close()
    
    # Figure 2: Example graphs for small vs large N
    small_N = N_VALUES[0]
    large_N = N_VALUES[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Small N examples
    G1_small = all_results[small_N]['example_G1']
    G2_small = all_results[small_N]['example_G2']
    
    # Large N examples
    G1_large = all_results[large_N]['example_G1']
    G2_large = all_results[large_N]['example_G2']
    
    # Plot
    for i, (G, title) in enumerate([(G1_small, f'Model S1, N={small_N}'),
                                     (G2_small, f'Model S2, N={small_N}'),
                                     (G1_large, f'Model S1, N={large_N}'),
                                     (G2_large, f'Model S2, N={large_N}')]):
        ax = axes[i // 2, i % 2]
        pos = nx.spring_layout(G, seed=42, k=0.3)
        nx.draw(G, pos, ax=ax, node_color='lightblue', node_size=100, 
                edge_color='gray', width=0.5, with_labels=False)
        ax.set_title(title, fontsize=14)
        ax.text(0.5, -0.1, f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                ha='center', transform=ax.transAxes, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'example_graphs.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'example_graphs.png'}")
    plt.close()
    
    # Figure 3: Mean similarity vs N
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_vals = []
    gen_means = []
    gen_stds = []
    mem_means = []
    mem_stds = []
    
    for N in N_VALUES:
        results = all_results[N]
        N_vals.append(N)
        gen_means.append(np.mean(results['generalization_scores']))
        gen_stds.append(np.std(results['generalization_scores']))
        mem_means.append(np.mean(results['memorization_scores']))
        mem_stds.append(np.std(results['memorization_scores']))
    
    # WL Similarity plot
    ax = axes[0]
    ax.errorbar(N_vals, gen_means, yerr=gen_stds, marker='o', 
                label='Generalization (S1 vs S2)', color='blue', linewidth=2)
    ax.errorbar(N_vals, mem_means, yerr=mem_stds, marker='s', 
                label='Memorization (vs training)', color='orange', linewidth=2)
    ax.set_xlabel('Training Set Size (N)', fontsize=14)
    ax.set_ylabel('WL Kernel Similarity', fontsize=14)
    ax.set_title('Convergence: Memorization to Generalization', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Statistics distance plot
    ax = axes[1]
    mse_means = [np.nanmean(all_results[N]['stats_mse']) for N in N_VALUES]
    mae_means = [np.nanmean(all_results[N]['stats_mae']) for N in N_VALUES]
    
    ax.plot(N_vals, mse_means, marker='o', label='MSE', color='red', linewidth=2)
    ax.plot(N_vals, mae_means, marker='s', label='MAE', color='green', linewidth=2)
    ax.set_xlabel('Training Set Size (N)', fontsize=14)
    ax.set_ylabel('Statistics Distance', fontsize=14)
    ax.set_title('Graph Statistics Convergence', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'convergence_curves.png'}")
    plt.close()

    # Figure 4: Variance analysis (within-model vs between-model)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    within_s1_means = []
    within_s1_stds = []
    within_s2_means = []
    within_s2_stds = []
    generalization_var_means = []
    generalization_var_stds = []

    for N in N_VALUES:
        within_s1 = all_results[N]['within_s1_similarity']
        within_s2 = all_results[N]['within_s2_similarity']
        gen_var = all_results[N]['generalization_variances']

        within_s1_means.append(np.nanmean(within_s1) if len(within_s1) else np.nan)
        within_s1_stds.append(np.nanstd(within_s1) if len(within_s1) else np.nan)
        within_s2_means.append(np.nanmean(within_s2) if len(within_s2) else np.nan)
        within_s2_stds.append(np.nanstd(within_s2) if len(within_s2) else np.nan)
        generalization_var_means.append(np.nanmean(gen_var) if len(gen_var) else np.nan)
        generalization_var_stds.append(np.nanstd(gen_var) if len(gen_var) else np.nan)

    ax = axes[0]
    ax.errorbar(N_vals, gen_means, yerr=gen_stds, marker='o',
                label='Between models (S1 vs S2)', color='blue', linewidth=2)
    ax.errorbar(N_vals, within_s1_means, yerr=within_s1_stds, marker='s',
                label='Within S1', color='purple', linewidth=2)
    ax.errorbar(N_vals, within_s2_means, yerr=within_s2_stds, marker='^',
                label='Within S2', color='green', linewidth=2)
    ax.set_xlabel('Training Set Size (N)', fontsize=14)
    ax.set_ylabel('WL Kernel Similarity', fontsize=14)
    ax.set_title('Within- vs Between-Model Similarity', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(N_vals, generalization_var_means, yerr=generalization_var_stds,
                marker='o', color='red', linewidth=2)
    ax.set_xlabel('Training Set Size (N)', fontsize=14)
    ax.set_ylabel('Variance of WL Similarity', fontsize=14)
    ax.set_title('Between-Model Variance Across Samples', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'variance_analysis.png'}")
    plt.close()


def save_summary(all_results, output_dir):
    """Save numerical summary of results."""
    summary_path = output_dir / "experiment_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Graph Generation: Memorization to Generalization Transition\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: featurehomophily0.2_graphs.pkl\n")
        f.write(f"Training sizes tested: {N_VALUES}\n\n")
        
        f.write("Results Summary:\n")
        f.write("-"*120 + "\n")
        f.write(f"{'N':<8} {'Gen Mean':<12} {'Gen Std':<12} {'Mem Mean':<12} {'Mem Std':<12} "
                f"{'MSE':<12} {'MAE':<12} {'Within S1':<12} {'Within S2':<12} {'Gen Var':<12}\n")
        f.write("-"*120 + "\n")
        
        for N in N_VALUES:
            results = all_results[N]
            gen_mean = np.mean(results['generalization_scores'])
            gen_std = np.std(results['generalization_scores'])
            mem_mean = np.mean(results['memorization_scores'])
            mem_std = np.std(results['memorization_scores'])
            mse = np.nanmean(results['stats_mse'])
            mae = np.nanmean(results['stats_mae'])
            within_s1 = np.nanmean(results['within_s1_similarity']) if results['within_s1_similarity'] else np.nan
            within_s2 = np.nanmean(results['within_s2_similarity']) if results['within_s2_similarity'] else np.nan
            gen_var = np.nanmean(results['generalization_variances']) if results['generalization_variances'] else np.nan
            
            f.write(f"{N:<8} {gen_mean:<12.4f} {gen_std:<12.4f} "
                   f"{mem_mean:<12.4f} {mem_std:<12.4f} "
                   f"{mse:<12.2f} {mae:<12.2f} "
                   f"{within_s1:<12.4f} {within_s2:<12.4f} {gen_var:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nInterpretation:\n")
        f.write("-"*80 + "\n")
        f.write("- Small N: Models memorize → Low generalization similarity (G1 ≠ G2)\n")
        f.write("- Large N: Models generalize → High generalization similarity (G1 ≈ G2)\n")
        f.write("- Transition point: Where gen_mean starts approaching mem_mean\n")
    
    print(f"\nSaved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Graph Generation Convergence Study')
    parser.add_argument('--data-path', type=str, 
                       default='data/featurehomophily0.2_graphs.pkl',
                       help='Path to dataset')
    parser.add_argument('--output-dir', type=str, 
                       default='outputs/convergence_study',
                       help='Output directory')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom run name')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{args.run_name}_{timestamp}"
    else:
        run_id = f"convergence_{timestamp}"
    
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Graph Generation: Memorization to Generalization Transition Study")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Training sizes: {N_VALUES}")
    print(f"{'='*80}\n")
    
    # Load dataset
    data_list = load_dataset(args.data_path)
    
    if len(data_list) < TEST_SET_SIZE:
        print(f"Warning: Expected at least {TEST_SET_SIZE} graphs for test set, found {len(data_list)}")
    
    cache_dir = Path(args.data_path).parent / "cache"
    cache_path = cache_dir / f"{Path(args.data_path).stem}_test_stats_seed{SPLIT_SEED}_size{TEST_SET_SIZE}.pt"

    split_config = shuffle_and_split_dataset(
        data_list,
        test_size=TEST_SET_SIZE,
        seed=SPLIT_SEED,
        stats_cache_path=cache_path
    )

    print("\nShuffled dataset split summary:")
    print(f"  Total graphs: {len(data_list)}")
    print(f"  S1 pool size: {len(split_config['S1_pool'])}")
    print(f"  S2 pool size: {len(split_config['S2_pool'])}")
    print(f"  Test pool size: {len(split_config['test_graphs'])} (held-out conditioning set)")
    
    # Run experiments for each N
    all_results = {}
    
    for N in N_VALUES:
        results = run_experiment_for_N(N, split_config, output_dir)
        all_results[N] = results

        visualize_single_experiment(results)
        
        # Save intermediate results
        import pickle
        with open(output_dir / f'results_N{N}.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    # Create visualizations
    visualize_results(all_results, output_dir)
    
    # Save summary
    save_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Key outputs:")
    print(f"  - Similarity distributions: {output_dir}/figures/similarity_distributions.png")
    print(f"  - Example graphs: {output_dir}/figures/example_graphs.png")
    print(f"  - Convergence curves: {output_dir}/figures/convergence_curves.png")
    print(f"  - Summary: {output_dir}/experiment_summary.txt")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
