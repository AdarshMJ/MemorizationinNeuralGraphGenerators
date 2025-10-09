import argparse
import os
import pickle
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_undirected

from autoencoder import VariationalAutoEncoder
from utils import construct_nx_from_adj, gen_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dfs_reorder_graph(data: Data, n_max_nodes: int) -> Data:
    """Reorder nodes via DFS starting from the highest-degree node."""
    data = data.clone()
    num_nodes = data.x.size(0)
    edge_index = data.edge_index.cpu()

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(num_nodes))
    graph_nx.add_edges_from(edge_index.t().tolist())

    degrees = dict(graph_nx.degree())
    visited = set()
    ordering: List[int] = []

    while len(ordering) < num_nodes:
        remaining = [node for node, _ in sorted(degrees.items(), key=lambda item: (-item[1], item[0])) if node not in visited]
        if not remaining:
            remaining = [node for node in range(num_nodes) if node not in visited]
        start = remaining[0]
        for node in nx.dfs_preorder_nodes(graph_nx, source=start):
            if node not in visited:
                ordering.append(node)
                visited.add(node)

    perm = torch.tensor(ordering, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=torch.long)

    data.x = data.x[perm]
    if hasattr(data, "y") and data.y is not None and data.y.numel() == num_nodes:
        data.y = data.y[perm]

    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, mask_name):
            mask = getattr(data, mask_name)
            if mask is not None and mask.numel() == num_nodes:
                setattr(data, mask_name, mask[perm])

    if hasattr(data, "raw_node_features"):
        raw_feats = getattr(data, "raw_node_features")
        if raw_feats is not None and raw_feats.size(0) == num_nodes:
            data.raw_node_features = raw_feats[perm]

    new_edge_index = inv_perm[data.edge_index]
    new_edge_index = to_undirected(new_edge_index, num_nodes=num_nodes)
    data.edge_index = new_edge_index.long()

    adj = torch.zeros(n_max_nodes, n_max_nodes, dtype=torch.float32)
    adj[new_edge_index[0], new_edge_index[1]] = 1.0
    adj.fill_diagonal_(0.0)
    data.A = adj.unsqueeze(0)

    return data


def ensure_padded_adjacency(data: Data, n_max_nodes: int) -> Data:
    data = data.clone()
    if hasattr(data, "A") and data.A is not None:
        current_adj = data.A.squeeze(0)
        padded = torch.zeros(n_max_nodes, n_max_nodes, dtype=current_adj.dtype)
        max_k = min(n_max_nodes, current_adj.size(0))
        padded[:max_k, :max_k] = current_adj[:max_k, :max_k]
        data.A = padded.unsqueeze(0)
    else:
        edge_index = to_undirected(data.edge_index)
        adj = torch.zeros(n_max_nodes, n_max_nodes, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj.fill_diagonal_(0.0)
        data.A = adj.unsqueeze(0)
    return data


def load_graphs(dataset_path: str, apply_dfs: bool, n_max_nodes: int = None) -> Tuple[List[Data], int]:
    with open(dataset_path, "rb") as handle:
        graphs: List[Data] = pickle.load(handle)

    inferred_n_max = n_max_nodes or max(graph.x.size(0) for graph in graphs)

    processed = []
    for graph in graphs:
        target_n_max = n_max_nodes or (graph.A.size(-1) if hasattr(graph, "A") else inferred_n_max)
        if apply_dfs:
            processed.append(dfs_reorder_graph(graph, target_n_max))
        else:
            processed.append(ensure_padded_adjacency(graph, target_n_max))

    return processed, inferred_n_max


def split_dataset(graphs: List[Data], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Data], List[Data], List[Data]]:
    set_seed(seed)
    indices = list(range(len(graphs)))
    random.shuffle(indices)

    n_total = len(graphs)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test_graphs = [graphs[i] for i in indices[n_train + n_val:]]

    return train_graphs, val_graphs, test_graphs


def collate_to_device(batch: Data, device: torch.device) -> Data:
    return batch.to(device)


def train_vgae(model: VariationalAutoEncoder,
               train_loader: DataLoader,
               val_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device,
               epochs: int,
               checkpoint_path: str) -> None:
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_graphs = 0

        for batch in train_loader:
            batch = collate_to_device(batch, device)
            optimizer.zero_grad()
            loss, recon, kld = model.loss_function(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_graphs += batch.num_graphs

        scheduler.step()
        avg_train_loss = train_loss / max(train_graphs, 1)

        model.eval()
        val_loss = 0.0
        val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = collate_to_device(batch, device)
                loss, recon, kld = model.loss_function(batch)
                val_loss += loss.item()
                val_graphs += batch.num_graphs

        avg_val_loss = val_loss / max(val_graphs, 1)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val_loss,
                "epoch": epoch
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * x @ y.t()
    dist = torch.clamp(dist, min=0.0)
    return torch.exp(-gamma * dist)


def compute_mmd(gt_stats: List[List[float]], gen_stats_list: List[List[float]]) -> float:
    x = torch.tensor(gt_stats, dtype=torch.float64)
    y = torch.tensor(gen_stats_list, dtype=torch.float64)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    concat = torch.cat([x, y], dim=0)
    pairwise = torch.cdist(concat, concat, p=2)
    non_zero = pairwise[pairwise > 0]
    if non_zero.numel() == 0:
        sigma = 1.0
    else:
        sigma = torch.median(non_zero).item()
        if sigma <= 0:
            sigma = 1.0
    gamma = 1.0 / (2.0 * sigma ** 2)

    k_xx = gaussian_kernel(x, x, gamma)
    k_yy = gaussian_kernel(y, y, gamma)
    k_xy = gaussian_kernel(x, y, gamma)

    n = x.size(0)
    m = y.size(0)

    mmd_sq = k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2.0 * k_xy.sum() / (n * m)
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0)).item()


def decode_graph(model: VariationalAutoEncoder, batch: Data, device: torch.device, n_max_nodes: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        batch = collate_to_device(batch, device)
        mu = model.encode(batch)
        adj_rec = model.decode_mu(mu)
        adj_rec = adj_rec.squeeze(0)
        adj_rec = (adj_rec + adj_rec.t()) / 2.0
        adj_rec = torch.clamp(adj_rec, 0.0, 1.0)
        adj_bin = (adj_rec > 0.5).float()
        if adj_bin.size(0) < n_max_nodes:
            padded = torch.zeros(n_max_nodes, n_max_nodes, dtype=adj_bin.dtype)
            size = adj_bin.size(0)
            padded[:size, :size] = adj_bin
            adj_bin = padded
    return adj_bin.cpu()


def evaluate_model(model: VariationalAutoEncoder,
                   test_graphs: List[Data],
                   device: torch.device,
                   n_max_nodes: int,
                   visualize_count: int) -> Tuple[float, List[nx.Graph], List[nx.Graph]]:
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    gt_stats: List[List[float]] = []
    gen_stats_list: List[List[float]] = []
    gt_graphs: List[nx.Graph] = []
    gen_graphs: List[nx.Graph] = []

    for batch in test_loader:
        adj_pred = decode_graph(model, batch, device, n_max_nodes)
        dense_gt = to_dense_adj(batch.edge_index, max_num_nodes=n_max_nodes).squeeze(0).cpu()

        pred_graph = construct_nx_from_adj(adj_pred.numpy())
        gt_graph = construct_nx_from_adj(dense_gt.numpy())

        gt_graphs.append(gt_graph)
        gen_graphs.append(pred_graph)

        gt_stats.append(gen_stats(gt_graph))
        gen_stats_list.append(gen_stats(pred_graph))

    mmd_score = compute_mmd(gt_stats, gen_stats_list)
    print(f"MMD between ground truth and generated stats: {mmd_score:.6f}")

    if visualize_count > 0 and len(gt_graphs) > 0:
        visualize_count = min(visualize_count, len(gt_graphs))
        gt_graphs = gt_graphs[:visualize_count]
        gen_graphs = gen_graphs[:visualize_count]

    return mmd_score, gt_graphs, gen_graphs


def visualize_graph_pairs(gt_graphs: List[nx.Graph],
                           gen_graphs: List[nx.Graph],
                           output_path: str) -> None:
    if not gt_graphs or not gen_graphs:
        return

    num_pairs = min(len(gt_graphs), len(gen_graphs))
    fig, axes = plt.subplots(2, num_pairs, figsize=(4 * num_pairs, 9), gridspec_kw={"hspace": 0.5})
    fig.subplots_adjust(top=0.92, bottom=0.08)
    if num_pairs == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx in range(num_pairs):
        gt_ax = axes[0][idx]
        gen_ax = axes[1][idx]

        gt_graph = gt_graphs[idx]
        gen_graph = gen_graphs[idx]

        gt_pos = nx.spring_layout(gt_graph, seed=idx)
        gen_pos = nx.spring_layout(gen_graph, seed=idx)

        nx.draw_networkx(gt_graph, pos=gt_pos, ax=gt_ax, node_size=60, with_labels=False)
        nx.draw_networkx(gen_graph, pos=gen_pos, ax=gen_ax, node_size=60, with_labels=False)

        gt_ax.text(0.5,
                   1.03,
                   f"Nodes: {gt_graph.number_of_nodes()} | Edges: {gt_graph.number_of_edges()}",
                   fontsize=15,
                   ha="center",
                   va="bottom",
                   transform=gt_ax.transAxes)
        gen_ax.text(0.5,
                    -0.18,
                    f"Nodes: {gen_graph.number_of_nodes()} | Edges: {gen_graph.number_of_edges()}",
                    fontsize=15,
                    ha="center",
                    va="top",
                    transform=gen_ax.transAxes)

        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        gen_ax.set_xticks([])
        gen_ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VGAE-only training on synthetic graphs")
    parser.add_argument("--dataset-path", type=str, default="data/featurehomophily0.6_graphs.pkl", help="Path to pickled PyG Data list")
    parser.add_argument("--output-dir", type=str, default="outputs_vgae", help="Directory to store outputs")
    parser.add_argument("--epochs", type=int, default=200, help="Number of VGAE training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for VGAE")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--hidden-dim-encoder", type=int, default=32, help="Hidden dimension for encoder GIN layers")
    parser.add_argument("--hidden-dim-decoder", type=int, default=64, help="Hidden dimension for decoder MLP")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension for VGAE")
    parser.add_argument("--n-layers-encoder", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--n-layers-decoder", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Fraction of graphs for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of graphs for validation")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--no-dfs-ordering", action="store_true", help="Disable DFS node ordering before training")
    parser.add_argument("--num-visualize", type=int, default=4, help="Number of graph pairs to visualize")
    parser.add_argument("--checkpoint", type=str, default="vgae_only.pth.tar", help="Model checkpoint filename")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint)
    viz_path = os.path.join(args.output_dir, "vgae_graph_comparison.png")

    print("Loading dataset...")
    apply_dfs = not args.no_dfs_ordering
    print(f"Applying DFS ordering: {apply_dfs}")
    graphs, inferred_n_max = load_graphs(args.dataset_path, apply_dfs=apply_dfs, n_max_nodes=None)
    n_max_nodes = inferred_n_max
    feature_dim = graphs[0].x.size(1)
    print(f"Loaded {len(graphs)} graphs | Nodes per graph: {graphs[0].x.size(0)} | Feature dim: {feature_dim}")

    train_graphs, val_graphs, test_graphs = split_dataset(graphs, args.train_ratio, args.val_ratio, args.seed)
    print(f"Split sizes -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)

    model = VariationalAutoEncoder(
        input_dim=feature_dim,
        hidden_dim_enc=args.hidden_dim_encoder,
        hidden_dim_dec=args.hidden_dim_decoder,
        latent_dim=args.latent_dim,
        n_layers_enc=args.n_layers_encoder,
        n_layers_dec=args.n_layers_decoder,
        n_max_nodes=n_max_nodes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    print("Starting VGAE training...")
    train_vgae(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, checkpoint_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded best model from {checkpoint_path} (epoch {checkpoint['epoch']})")

    print("Evaluating on test set...")
    mmd_score, gt_graphs, gen_graphs = evaluate_model(model, test_graphs, device, n_max_nodes, args.num_visualize)

    if args.num_visualize > 0:
        visualize_graph_pairs(gt_graphs, gen_graphs, viz_path)

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as fw:
        fw.write(f"MMD: {mmd_score:.6f}\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
