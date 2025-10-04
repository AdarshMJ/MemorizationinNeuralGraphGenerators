import argparse
import os
import random
import scipy as sp

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm

from torch.utils.data import Subset
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import sys
from pathlib import Path
np.random.seed(13)

# TODO: check/count number of all parameters

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs-autoencoder', type=int, default=200)
parser.add_argument('--hidden-dim-encoder', type=int, default=32)
parser.add_argument('--hidden-dim-decoder', type=int, default=64)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--n-max-nodes', type=int, default=100)
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=3)
parser.add_argument('--spectral-emb-dim', type=int, default=10)
parser.add_argument('--variational-autoencoder', action='store_true', default=True)
parser.add_argument('--epochs-denoise', type=int, default=100)
parser.add_argument('--timesteps', type=int, default=100)
parser.add_argument('--hidden-dim-denoise', type=int, default=256)
parser.add_argument('--n-layers_denoise', type=int, default=3)
parser.add_argument('--train-autoencoder', action='store_true', default=False)
parser.add_argument('--train-denoiser', action='store_true', default=False)
parser.add_argument('--n-properties', type=int, default=15)
parser.add_argument('--dim-condition', type=int, default=128)
parser.add_argument('--use-synthetic-data', action='store_true', default=False,
                   help='Use synthetic dataset instead of default generated graphs')
parser.add_argument('--synthetic-data-path', type=str, default='data/featurehomophily0.2_graphs.pkl',
                   help='Path to synthetic dataset pickle file')
parser.add_argument('--visualize', action='store_true', default=False,
                   help='Generate visualizations of ground truth vs generated graphs')
parser.add_argument('--n-vis-samples', type=int, default=10,
                   help='Number of samples to visualize')
parser.add_argument('--early-stopping-patience', type=int, default=20,
                   help='Number of epochs to wait before early stopping')
parser.add_argument('--grad-clip', type=float, default=1.0,
                   help='Gradient clipping value')
parser.add_argument('--output-dir', type=str, default='outputs',
                   help='Directory to save outputs (logs, figures, checkpoints)')
parser.add_argument('--run-name', type=str, default=None,
                   help='Custom name for this run (default: timestamp)')
args = parser.parse_args()

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.run_name:
    run_id = f"{args.run_name}_{timestamp}"
else:
    run_id = timestamp

output_base = Path(args.output_dir)
run_dir = output_base / run_id
run_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories
log_dir = run_dir / "logs"
fig_dir = run_dir / "figures"
checkpoint_dir = run_dir / "checkpoints"
stats_dir = run_dir / "stats"

for dir_path in [log_dir, fig_dir, checkpoint_dir, stats_dir]:
    dir_path.mkdir(exist_ok=True)

# Setup logging to file and console
log_file = log_dir / "training.log"
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
print(f"\n{'='*80}")
print(f"Run ID: {run_id}")
print(f"Output Directory: {run_dir}")
print(f"Log File: {log_file}")
print(f"{'='*80}\n")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph_types  = ["barabasi_albert", "cycle", "dual_barabasi_albert", "extended_barabasi_albert", "fast_gnp","ladder", "lobster", "lollipop","newman_watts_strogatz","regular", "partition","path", "powerlaw","star","stochastic","watts_strogatz","wheel"]

gr2id = {graph_types[i]:i for i in range(len(graph_types))}


def load_synthetic_dataset(dataset_path):
    """
    Load synthetic dataset from pickle file
    
    Args:
        dataset_path: Path to the dataset pickle file
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    with open(dataset_path, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loaded {len(graphs)} synthetic graphs from {dataset_path}")
    
    # Verify the format
    if len(graphs) > 0:
        sample_graph = graphs[0]
        print(f"Sample graph attributes: {sample_graph.keys if hasattr(sample_graph, 'keys') else dir(sample_graph)}")
        if hasattr(sample_graph, 'x'):
            print(f"Node features shape: {sample_graph.x.shape}")
        if hasattr(sample_graph, 'edge_index'):
            print(f"Edge index shape: {sample_graph.edge_index.shape}")
        if hasattr(sample_graph, 'graph_statistics_tensor'):
            print(f"Graph statistics shape: {sample_graph.graph_statistics_tensor.shape}")
    
    return graphs


def visualize_graphs_comparison(ground_truth_graphs, generated_graphs, n_samples=10, save_path='figures/graph_comparison.png'):
    """
    Visualize comparison between ground truth and generated graphs
    
    Args:
        ground_truth_graphs: List of ground truth NetworkX graphs
        generated_graphs: List of generated NetworkX graphs
        n_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    n_samples = min(n_samples, len(ground_truth_graphs), len(generated_graphs))
    
    # Create figure with 2 rows (top: ground truth, bottom: generated)
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
    
    # Ensure axes is 2D array
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_samples):
        # Ground truth graph (top row)
        G_gt = ground_truth_graphs[i]
        ax_gt = axes[0, i]
        
        # Draw ground truth graph
        pos = nx.spring_layout(G_gt, seed=42)
        nx.draw(G_gt, pos, ax=ax_gt, node_size=50, node_color='skyblue', 
                edge_color='gray', width=0.5, with_labels=False)
        
        # Add text showing nodes and edges count
        n_nodes_gt = G_gt.number_of_nodes()
        n_edges_gt = G_gt.number_of_edges()
        ax_gt.text(0.5, -0.1, f'N={n_nodes_gt}, E={n_edges_gt}', 
                  transform=ax_gt.transAxes, ha='center', fontsize=25)
        ax_gt.axis('off')
        
        # Generated graph (bottom row)
        G_gen = generated_graphs[i]
        ax_gen = axes[1, i]
        
        # Draw generated graph
        pos = nx.spring_layout(G_gen, seed=42)
        nx.draw(G_gen, pos, ax=ax_gen, node_size=50, node_color='lightcoral', 
                edge_color='gray', width=0.5, with_labels=False)
        
        # Add text showing nodes and edges count
        n_nodes_gen = G_gen.number_of_nodes()
        n_edges_gen = G_gen.number_of_edges()
        ax_gen.text(0.5, -0.1, f'N={n_nodes_gen}, E={n_edges_gen}', 
                   transform=ax_gen.transAxes, ha='center', fontsize=25)
        ax_gen.axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


data_lst = []

# Load synthetic dataset if specified, otherwise use default dataset
if args.use_synthetic_data:
    print("Loading synthetic dataset...")
    data_lst = load_synthetic_dataset(args.synthetic_data_path)
    
    # Detect the actual feature dimension from the dataset
    if len(data_lst) > 0:
        actual_feat_dim = data_lst[0].x.shape[1]
        print(f"Detected feature dimension: {actual_feat_dim}")
        # Override spectral_emb_dim to match the actual feature dimension
        # For synthetic data, we use the actual features, not spectral
        args.spectral_emb_dim = actual_feat_dim - 1  # -1 because code adds +1 later
    
    # Verify that the synthetic graphs have the required attributes
    # The synthetic graphs should already have: x, edge_index, A, stats
    # We need to ensure they match the expected format
    processed_data_lst = []
    for graph in tqdm(data_lst, desc="Processing synthetic graphs"):
        # Check if graph has stats or graph_statistics_tensor
        if hasattr(graph, 'graph_statistics_tensor') and not hasattr(graph, 'stats'):
            # Rename to 'stats' for compatibility with main.py
            graph.stats = graph.graph_statistics_tensor.unsqueeze(0) if graph.graph_statistics_tensor.dim() == 1 else graph.graph_statistics_tensor
        
        # If stats exists but has wrong shape, fix it
        if hasattr(graph, 'stats'):
            if graph.stats.dim() == 1:
                graph.stats = graph.stats.unsqueeze(0)
            # Take only first 15 dimensions if there are more
            if graph.stats.shape[-1] > args.n_properties:
                graph.stats = graph.stats[:, :args.n_properties]
        
        # Ensure A attribute exists and is properly formatted
        if not hasattr(graph, 'A'):
            # Reconstruct adjacency matrix from edge_index
            num_nodes = graph.x.shape[0]
            adj = torch.zeros(num_nodes, num_nodes)
            adj[graph.edge_index[0], graph.edge_index[1]] = 1
            
            # Pad to n_max_nodes
            size_diff = args.n_max_nodes - num_nodes
            adj = F.pad(adj, [0, size_diff, 0, size_diff])
            graph.A = adj.unsqueeze(0)
        
        processed_data_lst.append(graph)
    
    data_lst = processed_data_lst
    print(f"Processed {len(data_lst)} synthetic graphs with {actual_feat_dim}D features")

else:
    # Original dataset loading code
    filename = f'data/generated_dataset.pt'
    
    stats_lst = []
    # traverse through all the graphs of the folder
    files = [f for f in os.listdir("generated data/graphs")]
    #files_stats = [f for f in os.listdir("/data/iakovos/Multimodal/generated data/stats")]
    print(len(files))
    if os.path.isfile(filename):
        data_lst = torch.load(filename)
        print(f'Dataset {filename} loaded from file')
    
    else:
        adjs = []
        eigvals = []
        eigvecs = []
        n_nodes = []
        max_eigval = 0
        min_eigval = 0
        for fileread in tqdm(files):
            tokens = fileread.split("/")
            idx = tokens[-1].find(".")
            filen = tokens[-1][:idx]
            extension = tokens[-1][idx+1:]
            # filename = f'data/'+filen+'.pt'
            #self.ignore_first_eigv = ignore_first_eigv
            fread = os.path.join("generated data/graphs",fileread)
            fstats = os.path.join("generated data/stats",filen+".txt")
            type = None
            for t in graph_types:
                if t in filen:
                    type = t
            type_id = gr2id[type]
            #load dataset to networkx
            if extension == "gml":
                G = nx.read_gml(fread)
            else:
                G = nx.read_gexf(fread)
            # use canonical order (BFS) to create adjacency matrix
            ### BFS & DFS from largest-degree node
            stats_lst.append(fstats)
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]

            # rank connected componets from large to small size
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

            node_list_bfs = []
            #node_list_dfs = []
            for ii in range(len(CGs)):
              node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
              degree_sequence = sorted(
                  node_degree_list, key=lambda tt: tt[1], reverse=True)

              bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
              #dfs_tree = nx.dfs_tree(CGs[ii], source=degree_sequence[0][0])

              node_list_bfs += list(bfs_tree.nodes())
              #node_list_dfs += list(dfs_tree.nodes())

            adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

            adj = torch.from_numpy(adj_bfs).float()
            #L = nx.normalized_laplacian_matrix(G).toarray()
            diags = np.sum(adj_bfs, axis=0)
            diags = np.squeeze(np.asarray(diags))
            D = sparse.diags(diags).toarray()
            L = D - adj_bfs
            with sp.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.diags(diags_sqrt).toarray()
            L = np.linalg.multi_dot((DH, L, DH))
            L = torch.from_numpy(L).float()
            eigval, eigvecs = torch.linalg.eigh(L)
            eigval = torch.real(eigval)
            eigvecs = torch.real(eigvecs)
            idx = torch.argsort(eigval)
            eigvecs = eigvecs[:,idx]

            edge_index = torch.nonzero(adj).t()

            size_diff = args.n_max_nodes - G.number_of_nodes()
            x = torch.zeros(G.number_of_nodes(), args.spectral_emb_dim+1)
            x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(args.n_max_nodes-1)
            mn = min(G.number_of_nodes(),args.spectral_emb_dim)
            mn+=1
            x[:,1:mn] = eigvecs[:,:args.spectral_emb_dim]
            #print(x.size())
            adj = F.pad(adj, [0, size_diff, 0, size_diff])
            adj = adj.unsqueeze(0)
            #A = torch.zeros(1, args.n_max_nodes, args.n_max_nodes, args.spectral_emb_dim+2)
            #A[0,:,:,0] = adj
            #for i in range(G.number_of_nodes()):
            #    A[0,i,i,1:] = x[i,:]
            feats_stats = read_stats(fstats)
            feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
            data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, graph_class=type, class_label=type_id))
        torch.save(data_lst, filename)
        print(f'Dataset {filename} saved')


# Split into training, validation and test sets with stratification
print("\n" + "="*80)
print("Creating Stratified Train/Val/Test Split")
print("="*80)

# For synthetic data, stratify by feature homophily
if args.use_synthetic_data and hasattr(data_lst[0], 'feature_homophily'):
    print("Stratifying by feature homophily...")
    
    # Extract homophily values
    homophily_values = np.array([data.feature_homophily for data in data_lst])
    
    # Create bins for stratification (5 bins from 0.3 to 0.7)
    bins = np.linspace(homophily_values.min(), homophily_values.max(), 6)
    homophily_bins = np.digitize(homophily_values, bins) - 1
    
    # Stratified split
    train_idx = []
    val_idx = []
    test_idx = []
    
    for bin_id in np.unique(homophily_bins):
        bin_indices = np.where(homophily_bins == bin_id)[0]
        np.random.shuffle(bin_indices)
        
        n_bin = len(bin_indices)
        n_train = int(0.8 * n_bin)
        n_val = int(0.1 * n_bin)
        
        train_idx.extend(bin_indices[:n_train])
        val_idx.extend(bin_indices[n_train:n_train + n_val])
        test_idx.extend(bin_indices[n_train + n_val:])
    
    # Shuffle the splits
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Convert to lists
    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    test_idx = [int(i) for i in test_idx]
    
    # Print stratification statistics
    print(f"\nHomophily distribution across splits:")
    print(f"{'Split':<10} {'Size':<8} {'Homophily Mean':<18} {'Homophily Std':<15} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for split_name, split_indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        split_homophily = [homophily_values[i] for i in split_indices]
        print(f"{split_name:<10} {len(split_indices):<8} {np.mean(split_homophily):<18.4f} "
              f"{np.std(split_homophily):<15.4f} {np.min(split_homophily):<8.4f} {np.max(split_homophily):<8.4f}")
    
    print("=" * 80 + "\n")
else:
    # Original random split for non-synthetic data
    idx = np.random.permutation(len(data_lst))
    train_size = int(0.8*idx.size)
    val_size = int(0.1*idx.size)

    train_idx = [int(i) for i in idx[:train_size]]
    val_idx = [int(i) for i in idx[train_size:train_size + val_size]]
    test_idx = [int(i) for i in idx[train_size + val_size:]]
    
    print(f"Random split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}\n")

train_loader = DataLoader([data_lst[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader([data_lst[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader([data_lst[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)

# Determine input feature dimension
input_feat_dim = args.spectral_emb_dim + 1
print(f"Autoencoder input dimension: {input_feat_dim}")

if args.variational_autoencoder:
    autoencoder = VariationalAutoEncoder(input_feat_dim, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)
else:
    autoencoder = AutoEncoder(input_feat_dim, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


trainable_params_autoenc = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print("Number of Autoencoder's trainable parameters: "+str(trainable_params_autoenc))

# Train autoencoder
if args.train_autoencoder:
    best_val_loss = np.inf
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"Starting Autoencoder Training with Early Stopping (patience={args.early_stopping_patience})")
    print(f"{'='*80}\n")
    
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        if args.variational_autoencoder:
            train_loss_all_recon = 0
            train_loss_all_kld = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if args.variational_autoencoder:
                loss, recon, kld  = autoencoder.loss_function(data)
                train_loss_all_recon += recon.item()
                train_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.grad_clip)
            
            if args.variational_autoencoder:
                train_loss_all += loss.item()
            else:
                train_loss_all += (torch.max(data.batch)+1) * loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        if args.variational_autoencoder:
            val_loss_all_recon = 0
            val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            if args.variational_autoencoder:
                loss, recon, kld  = autoencoder.loss_function(data)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            if args.variational_autoencoder:
                val_loss_all += loss.item()
            else:
                val_loss_all += torch.max(data.batch)+1 * loss.item()
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if args.variational_autoencoder:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/train_count, train_loss_all_recon/train_count, train_loss_all_kld/train_count, val_loss_all/val_count, val_loss_all_recon/val_count, val_loss_all_kld/val_count))
            else:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        # Early stopping logic
        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = checkpoint_dir / 'autoencoder_best.pth.tar'
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'run_id': run_id
            }, checkpoint_path)
            # Also save to root for backward compatibility
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'autoencoder.pth.tar')
            print(f"  → Best model saved! (Val Loss: {val_loss_all/val_count:.5f})")
            print(f"     Checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement (patience: {patience_counter}/{args.early_stopping_patience})")
            
        if patience_counter >= args.early_stopping_patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss/val_count:.5f}")
            print(f"{'='*80}\n")
            break
else:
    if os.path.isfile('autoencoder.pth.tar'):
        checkpoint = torch.load('autoencoder.pth.tar', weights_only=False)
        autoencoder.load_state_dict(checkpoint['state_dict'])
        print('Loaded autoencoder from autoencoder.pth.tar')
    else:
        print('Warning: autoencoder.pth.tar not found. Using untrained autoencoder.')
        print('Please train the autoencoder first with --train-autoencoder flag.')

autoencoder.eval()
eval_autoencoder(test_loader, autoencoder, args.n_max_nodes, device) # add also mse (loss that we use generally)


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_properties, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

if args.train_denoiser:
    # Train denoising model
    best_val_loss = np.inf
    patience_counter = 0
    best_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"Starting Denoiser Training with Early Stopping (patience={args.early_stopping_patience})")
    print(f"{'='*80}\n")
    
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), args.grad_clip)
            
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        # Early stopping logic
        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = checkpoint_dir / 'denoise_model_best.pth.tar'
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'run_id': run_id
            }, checkpoint_path)
            # Also save to root for backward compatibility
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'denoise_model.pth.tar')
            if epoch % 5 == 0:
                print(f"  → Best model saved! (Val Loss: {val_loss_all/val_count:.5f})")
                print(f"     Checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if epoch % 5 == 0:
                print(f"  → No improvement (patience: {patience_counter}/{args.early_stopping_patience})")
            
        if patience_counter >= args.early_stopping_patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss/val_count:.5f}")
            print(f"{'='*80}\n")
            break
    
    denoiser_trained = True
else:
    denoiser_trained = False
    if os.path.isfile('denoise_model.pth.tar'):
        checkpoint = torch.load('denoise_model.pth.tar', weights_only=False)
        denoise_model.load_state_dict(checkpoint['state_dict'])
        print('Loaded denoiser from denoise_model.pth.tar')
        denoiser_trained = True
    else:
        print('Warning: denoise_model.pth.tar not found. Skipping testing phase.')
        print('Please train the denoiser first with --train-denoiser flag.')

# Only run testing if denoiser is trained (either just trained or loaded from checkpoint)
if denoiser_trained:
    denoise_model.eval()

    del train_loader, val_loader


    ground_truth = []
    pred = []
    gt_graphs_nx = []  # For visualization
    gen_graphs_nx = []  # For visualization


    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        stat = data.stats
        bs = stat.size(0)
        samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
        x_sample = samples[-1]
        adj = autoencoder.decode_mu(x_sample)
        stat_d = torch.reshape(stat, (-1, args.n_properties))

        for i in range(stat.size(0)):
            #adj = autoencoder.decode_mu(samples[random_index])
            # Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
            stat_x = stat_d[i]

            # Generate graph from predicted adjacency
            adj_np = adj[i,:,:].detach().cpu().numpy()
            
            # Debug: Check if adjacency has edges
            num_edges_gen = adj_np.sum()
            if k == 0 and i == 0:  # Print for first graph only
                print(f"\nDebug info for first generated graph:")
                print(f"  Generated adj min: {adj_np.min()}, max: {adj_np.max()}, sum: {num_edges_gen}")
                print(f"  Unique values in adj: {np.unique(adj_np)}")
            
            Gs_generated = construct_nx_from_adj(adj_np)
            
            # Get ground truth graph from original adjacency
            Gs_ground_truth = construct_nx_from_adj(data.A[i,:,:].detach().cpu().numpy())
            
            stat_x = stat_x.detach().cpu().numpy()
            ground_truth.append(stat_x)
            
            # Skip empty graphs in statistics calculation
            if num_edges_gen > 0:
                pred.append(gen_stats(Gs_generated))
            else:
                # Use zeros for empty graph stats
                pred.append(np.zeros(args.n_properties))
            
            # Store graphs for visualization
            if args.visualize and len(gt_graphs_nx) < args.n_vis_samples:
                gt_graphs_nx.append(Gs_ground_truth)
                gen_graphs_nx.append(Gs_generated)


    # stats = torch.cat(stats, dim=0).detach().cpu().numpy()


    mean, std = calculate_mean_std(ground_truth)


    mse, mae, norm_error = evaluation_metrics(ground_truth, pred)


    mse_all, mae_all, norm_error_all = z_score_norm(ground_truth, pred, mean, std)



    feats_lst = ["number of nodes", "number of edges", "density","max degree", "min degree", "avg degree","assortativity","triangles","avg triangles","max triangles","avg clustering coef", "global clustering coeff", "max k-core", "communities","diameter"]
    id2feats = {i:feats_lst[i] for i in range(len(mse))}




    print("MSE for the samples in all features is equal to: "+str(mse_all))
    print("MAE for the samples in all features is equal to: "+str(mae_all))
    print("Symmetric Mean absolute Percentage Error for the samples for all features is equal to: "+str(norm_error_all*100))
    print("=" * 100)

    for i in range(len(mse)):
        print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mse[i]))
        print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mae[i]))
        print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_error[i]*100))
        print("=" * 100)

    # Generate visualization if requested
    if args.visualize and len(gt_graphs_nx) > 0:
        print("\nGenerating visualization...")
        # Save to timestamped directory
        viz_filename = f"graph_comparison_{args.n_vis_samples}samples.png"
        viz_path = fig_dir / viz_filename
        visualize_graphs_comparison(gt_graphs_nx, gen_graphs_nx, 
                                    n_samples=args.n_vis_samples, 
                                    save_path=str(viz_path))
        # Also save to root figures/ for backward compatibility
        Path('figures').mkdir(exist_ok=True)
        visualize_graphs_comparison(gt_graphs_nx, gen_graphs_nx, 
                                    n_samples=args.n_vis_samples, 
                                    save_path='figures/graph_comparison.png')
        print(f"Visualization saved to: {viz_path}")
        print("Visualization complete!")
    
    # Save run summary
    summary_path = run_dir / "run_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Neural Graph Generator - Run Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Dataset: {'Synthetic' if args.use_synthetic_data else 'Generated'}\n")
        if args.use_synthetic_data:
            f.write(f"Dataset Path: {args.synthetic_data_path}\n")
        f.write(f"Training Mode: ")
        if args.train_autoencoder:
            f.write(f"Autoencoder\n")
        elif args.train_denoiser:
            f.write(f"Denoiser\n")
        else:
            f.write(f"Testing Only\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n")
        f.write(f"Gradient Clipping: {args.grad_clip}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write(f"Model Architecture:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Latent Dimension: {args.latent_dim}\n")
        f.write(f"Input Feature Dimension: {input_feat_dim}\n")
        f.write(f"Max Nodes: {args.n_max_nodes}\n")
        f.write(f"Encoder Hidden Dim: {args.hidden_dim_encoder}\n")
        f.write(f"Decoder Hidden Dim: {args.hidden_dim_decoder}\n")
        f.write(f"Denoiser Hidden Dim: {args.hidden_dim_denoise}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"MSE (all features): {mse_all:.2f}\n")
        f.write(f"MAE (all features): {mae_all:.2f}\n")
        f.write(f"SMAPE (all features): {norm_error_all*100:.2f}%\n\n")
        
        f.write(f"Output Files:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"Checkpoints: {checkpoint_dir}/\n")
        f.write(f"Logs: {log_file}\n")
        f.write(f"Statistics: {stats_dir}/\n")
        if args.visualize and len(gt_graphs_nx) > 0:
            f.write(f"Visualization: {viz_path}\n")
        f.write(f"\n" + "="*80 + "\n")
    
    print(f"\nRun summary saved to: {summary_path}")
    
    # Save statistics to timestamped directory
    gt_stats_path = stats_dir / "y_stats.txt"
    pred_stats_path = stats_dir / "y_pred_stats.txt"
    store_stats(ground_truth, pred, str(gt_stats_path), str(pred_stats_path))
    
    # Also save to root for backward compatibility
    store_stats(ground_truth, pred, "y_stats.txt", "y_pred_stats.txt")
    print(f"\nStatistics saved to: {stats_dir}")

    # stats = torch.cat(stats, dim=0).detach().cpu().numpy()


    mean, std = calculate_mean_std(ground_truth)


    mse, mae, norm_error = evaluation_metrics(ground_truth, pred)


    mse_all, mae_all, norm_error_all = z_score_norm(ground_truth, pred, mean, std)



    feats_lst = ["number of nodes", "number of edges", "density","max degree", "min degree", "avg degree","assortativity","triangles","avg triangles","max triangles","avg clustering coef", "global clustering coeff", "max k-core", "communities","diameter"]
    id2feats = {i:feats_lst[i] for i in range(len(mse))}




    print("MSE for the samples in all features is equal to: "+str(mse_all))
    print("MAE for the samples in all features is equal to: "+str(mae_all))
    print("Symmetric Mean absolute Percentage Error for the samples for all features is equal to: "+str(norm_error_all*100))
    print("=" * 100)

    for i in range(len(mse)):
        print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mse[i]))
        print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mae[i]))
        print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_error[i]*100))
        print("=" * 100)
else:
    print("\nSkipping testing phase - denoiser not trained yet.")
    print("Run with --train-denoiser flag to train the denoiser first.")
    print("\nNo outputs to save (testing was skipped).")
