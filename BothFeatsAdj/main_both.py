import argparse
import csv
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected

from autoencoder import Decoder, GIN
from utils import construct_nx_from_adj, gen_stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dfs_reorder_graph(data: Data, n_max_nodes: int) -> Data:
    """Reorder nodes via DFS starting from the highest-degree node and pad tensors."""
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
    has_labels = hasattr(data, "y") and data.y is not None and data.y.numel() == num_nodes
    if has_labels:
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

    feature_dim = data.x.size(1)
    padded_x = torch.zeros(n_max_nodes, feature_dim, dtype=data.x.dtype)
    padded_x[:num_nodes] = data.x
    data.x_padded = padded_x.unsqueeze(0)

    padded_y = torch.full((n_max_nodes,), fill_value=-1, dtype=torch.long)
    if has_labels:
        padded_y[:num_nodes] = data.y
    else:
        data.y = torch.zeros(num_nodes, dtype=torch.long)
        padded_y[:num_nodes] = data.y
    data.y_padded = padded_y.unsqueeze(0)

    node_mask = torch.zeros(n_max_nodes, dtype=torch.bool)
    node_mask[:num_nodes] = True
    data.node_mask = node_mask.unsqueeze(0)

    return data


def ensure_padded_graph(data: Data, n_max_nodes: int) -> Data:
    data = data.clone()
    num_nodes = data.x.size(0)

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

    feature_dim = data.x.size(1)
    padded_x = torch.zeros(n_max_nodes, feature_dim, dtype=data.x.dtype)
    padded_x[:num_nodes] = data.x
    data.x_padded = padded_x.unsqueeze(0)

    has_labels = hasattr(data, "y") and data.y is not None and data.y.numel() == num_nodes
    padded_y = torch.full((n_max_nodes,), fill_value=-1, dtype=torch.long)
    if has_labels:
        padded_y[:num_nodes] = data.y
    else:
        data.y = torch.zeros(num_nodes, dtype=torch.long)
        padded_y[:num_nodes] = data.y
    data.y_padded = padded_y.unsqueeze(0)

    node_mask = torch.zeros(n_max_nodes, dtype=torch.bool)
    node_mask[:num_nodes] = True
    data.node_mask = node_mask.unsqueeze(0)

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
            processed.append(ensure_padded_graph(graph, target_n_max))

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


def refine_features_with_propagation(adj_matrix: torch.Tensor,
                                     features: torch.Tensor,
                                     alpha: float = 0.8,
                                     num_iterations: int = 5,
                                     eps: float = 1e-8) -> torch.Tensor:
    if adj_matrix.numel() == 0:
        return features

    adj_sym = (adj_matrix + adj_matrix.t()) / 2.0
    deg = adj_sym.sum(dim=-1).clamp_min(eps)
    inv_sqrt = deg.pow(-0.5)
    adj_norm = inv_sqrt.unsqueeze(1) * adj_sym * inv_sqrt.unsqueeze(0)

    x0 = features.clone()
    x = features.clone()
    for _ in range(num_iterations):
        x = (1 - alpha) * x0 + alpha * adj_norm @ x
    return x


def compute_cosine_homophily(adj_matrix: torch.Tensor,
                             features: torch.Tensor,
                             eps: float = 1e-8) -> float:
    if adj_matrix.numel() == 0 or features.numel() == 0:
        return 0.0

    upper_mask = torch.triu(adj_matrix, diagonal=1) > 0
    edge_pairs = upper_mask.nonzero(as_tuple=False)
    if edge_pairs.size(0) == 0:
        return 0.0

    norm = features.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    feats_norm = features / norm
    sims = (feats_norm[edge_pairs[:, 0]] * feats_norm[edge_pairs[:, 1]]).sum(dim=-1)
    return sims.mean().item()


class NodeEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.convs = nn.ModuleList([
            GINConv(mlp)
        ])

        for _ in range(n_layers - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            )
            self.convs.append(GINConv(block))
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.graph_bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = self.node_norm(x)
        graph_embeddings = global_add_pool(node_embeddings, data.batch)
        graph_embeddings = self.graph_bn(graph_embeddings)
        return node_embeddings, graph_embeddings


class FeatureDecoder(nn.Module):
    def __init__(self,
                 latent_dim_node: int,
                 latent_dim_graph: int,
                 hidden_dim: int,
                 feature_dim: int,
                 n_nodes: int,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.node_proj = nn.Linear(latent_dim_node, hidden_dim)
        self.graph_proj = nn.Linear(latent_dim_graph, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.out_mean = nn.Linear(hidden_dim, feature_dim)
        self.out_logvar = nn.Linear(hidden_dim, feature_dim)

    def forward(self,
                z_nodes: torch.Tensor,
                z_graph: torch.Tensor,
                adj: torch.Tensor,
                node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_nodes, _ = z_nodes.size()
        device = z_nodes.device

        graph_context = self.graph_proj(z_graph).unsqueeze(1)
        h = self.node_proj(z_nodes) + graph_context
        h = F.relu(h)

        adj_norm = self._normalize_adj(adj)

        mask = node_mask.unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            neighbor_agg = torch.bmm(adj_norm, h)
            global_context = (h * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            fused = torch.cat([h, neighbor_agg + global_context], dim=-1)
            fused = layer(fused)
            fused = F.relu(norm(fused))
            fused = self.dropout(fused)
            h = fused * mask

        mean = self.out_mean(h) * mask
        logvar = self.out_logvar(h) * mask
        return h, mean, logvar

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        n = adj.size(-1)
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_sym = (adj + adj.transpose(-1, -2)) / 2.0 + eye
        deg = adj_sym.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj_sym / deg


class LabelDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, node_repr: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_repr)


class JointVariationalAutoEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim_enc: int,
                 hidden_dim_struct_dec: int,
                 hidden_dim_feat_dec: int,
                 latent_dim: int,
                 n_layers_enc: int,
                 n_layers_struct_dec: int,
                 n_max_nodes: int,
                 num_classes: int,
                 lambda_feat: float = 1.0,
                 lambda_label: float = 0.5,
                 beta: float = 0.05,
                 beta_node: float = 0.01,
                 lambda_moment: float = 0.1,
                 use_teacher_forcing: bool = True,
                 tf_anneal_epochs: int = 0,
                 latent_dim_node: int = None,
                 beta_warmup_epochs: int = 0,
                 beta_node_warmup_epochs: int = 0):
        super().__init__()
        self.n_max_nodes = n_max_nodes
        self.lambda_feat = lambda_feat
        self.lambda_label = lambda_label
        self.beta_final = beta
        self.beta_node_final = beta_node
        self.beta_warmup_epochs = beta_warmup_epochs
        self.beta_node_warmup_epochs = beta_node_warmup_epochs
        self.beta_current = 0.0 if beta_warmup_epochs > 0 else beta
        self.beta_node_current = 0.0 if beta_node_warmup_epochs > 0 else beta_node
        self.lambda_moment = lambda_moment
        self.use_teacher_forcing = use_teacher_forcing
        self.tf_anneal_epochs = tf_anneal_epochs
        self.teacher_forcing_ratio = 1.0 if use_teacher_forcing else 0.0
        self.latent_dim_node = latent_dim if latent_dim_node is None else latent_dim_node
        self.latent_dim_graph = latent_dim

        self.encoder = NodeEncoder(input_dim, hidden_dim_enc, n_layers_enc)
        self.fc_mu_graph = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar_graph = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_mu_node = nn.Linear(hidden_dim_enc, self.latent_dim_node)
        self.fc_logvar_node = nn.Linear(hidden_dim_enc, self.latent_dim_node)

        self.struct_proj = nn.Linear(latent_dim, latent_dim)

        self.struct_decoder = Decoder(latent_dim, hidden_dim_struct_dec, n_layers_struct_dec, n_max_nodes)
        self.feature_decoder = FeatureDecoder(self.latent_dim_node, latent_dim, hidden_dim_feat_dec, input_dim, n_max_nodes)
        self.label_decoder = LabelDecoder(hidden_dim_feat_dec, num_classes)

    def encode_stats(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_embeddings, graph_embeddings = self.encoder(data)
        mu_graph = torch.tanh(self.fc_mu_graph(graph_embeddings))
        logvar_graph = torch.clamp(self.fc_logvar_graph(graph_embeddings), min=-10.0, max=10.0)

        mu_node_flat = torch.tanh(self.fc_mu_node(node_embeddings))
        logvar_node_flat = torch.clamp(self.fc_logvar_node(node_embeddings), min=-10.0, max=10.0)

        mu_node, mask = to_dense_batch(mu_node_flat, data.batch, max_num_nodes=self.n_max_nodes)
        logvar_node, _ = to_dense_batch(logvar_node_flat, data.batch, max_num_nodes=self.n_max_nodes)

        node_mask = mask.float()
        return mu_graph, logvar_graph, mu_node, logvar_node, node_mask

    def reparameterize_graph(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def reparameterize_nodes(self,
                              mu: torch.Tensor,
                              logvar: torch.Tensor,
                              node_mask: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        mask = node_mask.unsqueeze(-1).float()
        return z * mask

    def update_teacher_forcing(self, epoch: int) -> None:
        if not self.use_teacher_forcing:
            self.teacher_forcing_ratio = 0.0
            return
        if self.tf_anneal_epochs <= 0:
            self.teacher_forcing_ratio = 1.0
            return
        progress = min(max(epoch, 0) / float(self.tf_anneal_epochs), 1.0)
        self.teacher_forcing_ratio = float(max(0.0, 1.0 - progress))

    def update_regularization(self, epoch: int) -> None:
        if self.beta_warmup_epochs > 0:
            progress = min(max(epoch, 0) / float(self.beta_warmup_epochs), 1.0)
            self.beta_current = self.beta_final * progress
        else:
            self.beta_current = self.beta_final

        if self.beta_node_warmup_epochs > 0:
            progress = min(max(epoch, 0) / float(self.beta_node_warmup_epochs), 1.0)
            self.beta_node_current = self.beta_node_final * progress
        else:
            self.beta_node_current = self.beta_node_final

    def decode(self,
               z_graph: torch.Tensor,
               z_nodes: torch.Tensor,
               node_mask: torch.Tensor,
               adj_override: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_struct = self.struct_proj(z_graph)
        adj_pred = self.struct_decoder(z_struct)

        adj_for_features = adj_override if adj_override is not None else adj_pred
        mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        adj_for_features = adj_for_features * mask_2d

        node_repr, feat_mean, feat_logvar = self.feature_decoder(z_nodes, z_graph, adj_for_features, node_mask)
        label_logits = self.label_decoder(node_repr)
        return adj_pred, feat_mean, feat_logvar, label_logits

    def forward(self, data: Data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.loss_function(data)

    def loss_function(self, data: Data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu_graph, logvar_graph, mu_node, logvar_node, encoder_mask = self.encode_stats(data)
        if hasattr(data, "node_mask") and data.node_mask is not None:
            node_mask_raw = data.node_mask
            if node_mask_raw.dim() == 3:
                node_mask_raw = node_mask_raw.squeeze(1)
            node_mask = node_mask_raw.float()
            label_mask = node_mask_raw.bool()
        else:
            node_mask = encoder_mask
            label_mask = encoder_mask.bool()

        z_graph = self.reparameterize_graph(mu_graph, logvar_graph)
        z_nodes = self.reparameterize_nodes(mu_node, logvar_node, encoder_mask)

        adj_target = data.A
        use_tf = False
        if self.training and self.use_teacher_forcing and self.teacher_forcing_ratio > 0.0:
            if torch.rand(1).item() < self.teacher_forcing_ratio:
                use_tf = True
        adj_override = adj_target if use_tf else None

        adj_pred, feat_mean, feat_logvar, label_logits = self.decode(z_graph, z_nodes, node_mask, adj_override=adj_override)

        mask = node_mask.unsqueeze(-1)
        feat_target = data.x_padded
        feat_logvar = torch.clamp(feat_logvar, min=-10.0, max=10.0)
        variance = torch.exp(feat_logvar)
        diff = feat_mean - feat_target
        feat_nll = 0.5 * (diff.pow(2) / (variance + 1e-6) + feat_logvar)
        feat_loss = (feat_nll * mask).sum() / mask.sum().clamp(min=1.0)

        adj_loss = F.l1_loss(adj_pred, adj_target)

        label_target = data.y_padded
        if label_mask.sum() > 0:
            label_loss = F.cross_entropy(label_logits[label_mask], label_target[label_mask])
        else:
            label_loss = torch.zeros(1, device=adj_loss.device)

        kld_graph = -0.5 * torch.sum(1 + logvar_graph - mu_graph.pow(2) - logvar_graph.exp()) / adj_pred.size(0)
        kld_node = -0.5 * ((1 + logvar_node - mu_node.pow(2) - logvar_node.exp()) * encoder_mask.unsqueeze(-1)).sum()
        kld_node = kld_node / encoder_mask.sum().clamp(min=1.0)

        moment_loss = self._moment_penalty(feat_mean, feat_target, mask)

        total_loss = (
            adj_loss
            + self.lambda_feat * feat_loss
            + self.lambda_label * label_loss
            + self.beta_current * kld_graph
            + self.beta_node_current * kld_node
            + self.lambda_moment * moment_loss
        )
        logs = {
            "total": total_loss.detach(),
            "adj": adj_loss.detach(),
            "feat": feat_loss.detach(),
            "label": label_loss.detach(),
            "kld_graph": kld_graph.detach(),
            "kld_node": kld_node.detach(),
            "moment": moment_loss.detach()
        }
        return total_loss, logs

    def _moment_penalty(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pred_mean = (pred * mask).sum(dim=1, keepdim=True) / count
        target_mean = (target * mask).sum(dim=1, keepdim=True) / count

        pred_centered = (pred - pred_mean) * mask
        target_centered = (target - target_mean) * mask

        pred_var = (pred_centered.pow(2)).sum(dim=1, keepdim=True) / count
        target_var = (target_centered.pow(2)).sum(dim=1, keepdim=True) / count

        mean_loss = F.mse_loss(pred_mean, target_mean)
        var_loss = F.mse_loss(pred_var, target_var)
        return mean_loss + var_loss

    def reconstruct(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            mu_graph, _, mu_node, _, node_mask = self.encode_stats(data)
            z_graph = self.reparameterize_graph(mu_graph, torch.zeros_like(mu_graph))
            z_nodes = self.reparameterize_nodes(mu_node, torch.zeros_like(mu_node), node_mask)
            adj, feat_mean, _, label_logits = self.decode(z_graph, z_nodes, node_mask)
        return adj, feat_mean, label_logits


def collate_to_device(batch: Data, device: torch.device) -> Data:
    return batch.to(device)


def train_joint(model: JointVariationalAutoEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epochs: int,
        checkpoint_path: str) -> None:
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.update_teacher_forcing(epoch - 1)
        model.update_regularization(epoch - 1)
        model.train()
        train_sums: Dict[str, float] = defaultdict(float)
        train_graphs = 0

        for batch in train_loader:
            batch = collate_to_device(batch, device)
            optimizer.zero_grad()
            loss, logs = model.loss_function(batch)
            loss.backward()
            optimizer.step()

            for key, value in logs.items():
                train_sums[key] += value.item()
            train_graphs += batch.num_graphs

        scheduler.step()
        avg_train = {k: v / max(train_graphs, 1) for k, v in train_sums.items()}

        model.eval()
        val_sums: Dict[str, float] = defaultdict(float)
        val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = collate_to_device(batch, device)
                loss, logs = model.loss_function(batch)
                for key, value in logs.items():
                    val_sums[key] += value.item()
                val_graphs += batch.num_graphs

        avg_val = {k: v / max(val_graphs, 1) for k, v in val_sums.items()}
        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train.get('total', 0.0):.4f} | Val Loss: {avg_val.get('total', 0.0):.4f} "
            f"| Adj: {avg_val.get('adj', 0.0):.4f} | Feat: {avg_val.get('feat', 0.0):.4f} | Label: {avg_val.get('label', 0.0):.4f} "
            f"| KLDg: {avg_val.get('kld_graph', 0.0):.4f} | KLDn: {avg_val.get('kld_node', 0.0):.4f} | Moment: {avg_val.get('moment', 0.0):.4f} "
            f"| beta: {model.beta_current:.4f} | beta_node: {model.beta_node_current:.4f}"
        )

        if avg_val["total"] < best_val_loss:
            best_val_loss = avg_val["total"]
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val_loss,
                "epoch": epoch
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")


def decode_joint(model: JointVariationalAutoEncoder,
                 batch: Data,
                 device: torch.device,
                 n_max_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        batch = collate_to_device(batch, device)
        adj_pred, feat_pred, label_logits = model.reconstruct(batch)
        adj_pred = adj_pred.squeeze(0)
        feat_pred = feat_pred.squeeze(0)
        label_logits = label_logits.squeeze(0)

        adj_pred = (adj_pred + adj_pred.t()) / 2.0
        adj_clamped = torch.clamp(adj_pred, 0.0, 1.0)

        if adj_clamped.size(0) < n_max_nodes:
            padded = torch.zeros(n_max_nodes, n_max_nodes, dtype=adj_clamped.dtype)
            size = adj_clamped.size(0)
            padded[:size, :size] = adj_clamped
            adj_clamped = padded

        if feat_pred.size(0) < n_max_nodes:
            padded_feat = torch.zeros(n_max_nodes, feat_pred.size(1), dtype=feat_pred.dtype)
            padded_feat[:feat_pred.size(0)] = feat_pred
            feat_pred = padded_feat

        if label_logits.size(0) < n_max_nodes:
            padded_labels = torch.zeros(n_max_nodes, label_logits.size(1), dtype=label_logits.dtype)
            padded_labels[:label_logits.size(0)] = label_logits
            label_logits = padded_labels

    return adj_clamped.cpu(), feat_pred.cpu(), label_logits.cpu()


def evaluate_model(model: JointVariationalAutoEncoder,
                   test_graphs: List[Data],
                   device: torch.device,
                   n_max_nodes: int,
                   num_classes: int,
                   visualize_count: int,
                   output_dir: str) -> Tuple[Dict[str, float], Dict[str, List[float]], List[nx.Graph], List[nx.Graph]]:
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    gt_stats: List[List[float]] = []
    gen_stats_list: List[List[float]] = []
    gt_graphs: List[nx.Graph] = []
    gen_graphs: List[nx.Graph] = []

    feature_mae_sum = 0.0
    structure_mae_sum = 0.0
    homophily_gt_values: List[float] = []
    homophily_gen_values: List[float] = []

    all_feat_gt: List[torch.Tensor] = []
    all_feat_gen: List[torch.Tensor] = []
    per_graph_feat_mae: List[float] = []

    for batch in test_loader:
        node_mask = batch.node_mask.squeeze(0).cpu()
        num_nodes = int(node_mask.sum().item())

        adj_soft, feat_pred, _ = decode_joint(model, batch, device, n_max_nodes)
        adj_soft = adj_soft[:num_nodes, :num_nodes]
        feat_pred = feat_pred[:num_nodes]

        dense_gt = batch.A.squeeze(0)[:num_nodes, :num_nodes].cpu()
        feat_gt = batch.x_padded.squeeze(0)[:num_nodes].cpu()

        feat_refined = refine_features_with_propagation(adj_soft, feat_pred)

        adj_bin = (adj_soft > 0.5).float()
        pred_graph = construct_nx_from_adj(adj_bin.numpy())
        gt_graph = construct_nx_from_adj(dense_gt.numpy())

        gt_graphs.append(gt_graph)
        gen_graphs.append(pred_graph)

        gt_stats.append(gen_stats(gt_graph))
        gen_stats_list.append(gen_stats(pred_graph))

        structure_mae_sum += F.l1_loss(adj_soft, dense_gt, reduction="mean").item()
        feat_mae = F.l1_loss(feat_refined, feat_gt, reduction="mean").item()
        feature_mae_sum += feat_mae
        per_graph_feat_mae.append(feat_mae)

        all_feat_gt.append(feat_gt)
        all_feat_gen.append(feat_refined)

        homophily_gt_values.append(compute_cosine_homophily(dense_gt, feat_gt))
        homophily_gen_values.append(compute_cosine_homophily(adj_bin, feat_refined))

    mmd_score = compute_mmd(gt_stats, gen_stats_list)
    avg_structure_mae = structure_mae_sum / max(len(test_graphs), 1)
    avg_feature_mae = feature_mae_sum / max(len(test_graphs), 1)
    avg_homophily_gt = float(np.mean(homophily_gt_values)) if homophily_gt_values else 0.0
    avg_homophily_gen = float(np.mean(homophily_gen_values)) if homophily_gen_values else 0.0

    feature_analysis: Dict[str, List[float]] = {}
    if all_feat_gt and all_feat_gen:
        concat_gt = torch.cat([tensor.reshape(-1, tensor.size(-1)) for tensor in all_feat_gt], dim=0)
        concat_gen = torch.cat([tensor.reshape(-1, tensor.size(-1)) for tensor in all_feat_gen], dim=0)

        gt_means = concat_gt.mean(dim=0)
        gen_means = concat_gen.mean(dim=0)
        gt_vars = concat_gt.var(dim=0, unbiased=False)
        gen_vars = concat_gen.var(dim=0, unbiased=False)

        abs_errors = torch.abs(concat_gen - concat_gt)
        channel_mae = abs_errors.mean(dim=0)
        channel_mean_diff = torch.abs(gen_means - gt_means)
        channel_var_diff = torch.abs(gen_vars - gt_vars)

        max_samples = 10000
        if abs_errors.numel() > max_samples:
            sample_indices = torch.linspace(0, abs_errors.numel() - 1, steps=max_samples).long()
            abs_error_samples = abs_errors.view(-1)[sample_indices]
        else:
            abs_error_samples = abs_errors.view(-1)

        feature_analysis = {
            "channel_mae": channel_mae.tolist(),
            "channel_mean_gt": gt_means.tolist(),
            "channel_mean_gen": gen_means.tolist(),
            "channel_var_gt": gt_vars.tolist(),
            "channel_var_gen": gen_vars.tolist(),
            "channel_mean_abs_diff": channel_mean_diff.tolist(),
            "channel_var_abs_diff": channel_var_diff.tolist(),
            "per_graph_mae": per_graph_feat_mae,
            "abs_error_samples": abs_error_samples.tolist()
        }

    if visualize_count > 0 and len(gt_graphs) > 0:
        visualize_count = min(visualize_count, len(gt_graphs))
        gt_graphs = gt_graphs[:visualize_count]
        gen_graphs = gen_graphs[:visualize_count]

    metrics_summary = {
        "mmd": mmd_score,
        "structure_mae": avg_structure_mae,
        "feature_mae": avg_feature_mae,
        "feature_homophily_gt": avg_homophily_gt,
        "feature_homophily_gen": avg_homophily_gen
    }

    return metrics_summary, feature_analysis, gt_graphs, gen_graphs


def plot_feature_statistics(feature_analysis: Dict[str, List[float]],
                            output_dir: str) -> None:
    if not feature_analysis:
        return

    os.makedirs(output_dir, exist_ok=True)

    channel_mean_gt = np.array(feature_analysis.get("channel_mean_gt", []))
    channel_mean_gen = np.array(feature_analysis.get("channel_mean_gen", []))
    channel_var_gt = np.array(feature_analysis.get("channel_var_gt", []))
    channel_var_gen = np.array(feature_analysis.get("channel_var_gen", []))
    abs_error_samples = np.array(feature_analysis.get("abs_error_samples", []))
    per_graph_mae = feature_analysis.get("per_graph_mae", [])

    stats_path = os.path.join(output_dir, "feature_statistics.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if channel_mean_gt.size and channel_mean_gen.size:
        diag_min = float(min(channel_mean_gt.min(), channel_mean_gen.min()))
        diag_max = float(max(channel_mean_gt.max(), channel_mean_gen.max()))
        axes[0].scatter(channel_mean_gt, channel_mean_gen, s=40, alpha=0.7)
        axes[0].plot([diag_min, diag_max], [diag_min, diag_max], color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("GT mean", fontsize=25)
    axes[0].set_ylabel("Gen mean", fontsize=25)
    axes[0].tick_params(labelsize=18)

    if channel_var_gt.size and channel_var_gen.size:
        diag_min = float(min(channel_var_gt.min(), channel_var_gen.min()))
        diag_max = float(max(channel_var_gt.max(), channel_var_gen.max()))
        axes[1].scatter(channel_var_gt, channel_var_gen, s=40, alpha=0.7, color="#ff7f0e")
        axes[1].plot([diag_min, diag_max], [diag_min, diag_max], color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("GT variance", fontsize=25)
    axes[1].set_ylabel("Gen variance", fontsize=25)
    axes[1].tick_params(labelsize=18)

    if abs_error_samples.size:
        bins = min(50, max(10, int(np.sqrt(abs_error_samples.size))))
        axes[2].hist(abs_error_samples, bins=bins, alpha=0.85, color="#2ca02c")
    else:
        axes[2].hist([0.0], bins=1, alpha=0.85, color="#2ca02c")
    axes[2].set_xlabel("Absolute error", fontsize=25)
    axes[2].set_ylabel("Frequency", fontsize=25)
    axes[2].tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(stats_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved feature statistics visualization to {stats_path}")

    if per_graph_mae:
        mae_path = os.path.join(output_dir, "per_graph_feature_mae.png")
        fig, ax = plt.subplots(figsize=(7, 5))
        sorted_mae = np.sort(np.array(per_graph_mae))
        ax.plot(sorted_mae, marker="o", linestyle="-", linewidth=1.5)
        ax.set_xlabel("Graph index (sorted)", fontsize=25)
        ax.set_ylabel("Feature MAE", fontsize=25)
        ax.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(mae_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved per-graph MAE visualization to {mae_path}")


def collect_graph_latents(model: JointVariationalAutoEncoder,
                          graphs: List[Data],
                          device: torch.device,
                          batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    if not graphs:
        return np.empty((0, model.latent_dim_graph)), np.empty((0,), dtype=np.int64)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    latents: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = collate_to_device(batch, device)
            mu_graph, _, _, _, _ = model.encode_stats(batch)
            mu_graph = mu_graph.cpu().numpy()

            if hasattr(batch, "graph_class") and batch.graph_class is not None:
                label_tensor = batch.graph_class
            elif hasattr(batch, "class_label") and batch.class_label is not None:
                label_tensor = batch.class_label
            elif hasattr(batch, "graph_label") and batch.graph_label is not None:
                label_tensor = batch.graph_label
            else:
                label_tensor = torch.full((batch.num_graphs,), -1, dtype=torch.long, device=batch.batch.device)

            if isinstance(label_tensor, torch.Tensor):
                label_array = label_tensor.detach().cpu().numpy()
            elif isinstance(label_tensor, np.ndarray):
                label_array = label_tensor.copy()
            elif isinstance(label_tensor, (list, tuple)):
                label_array = np.array(label_tensor)
            else:
                label_array = np.array([label_tensor])

            if label_array.ndim == 0:
                label_array = np.full((batch.num_graphs,), label_array.item(), dtype=label_array.dtype)
            elif label_array.shape[0] != batch.num_graphs:
                label_array = np.resize(label_array, (batch.num_graphs,))

            latents.append(mu_graph)
            labels.append(label_array)

    if not latents:
        return np.empty((0, model.latent_dim_graph)), np.empty((0,), dtype=np.int64)

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)


def plot_latent_projection(coords: np.ndarray,
                           label_array: np.ndarray,
                           method_name: str,
                           output_path: str) -> None:
    if coords.shape[0] < 2 or coords.shape[1] != 2:
        return

    unique_labels = np.unique(label_array)
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        mask = label_array == label
        display_label = "unknown" if label == -1 else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=30, alpha=0.75, color=cmap(idx), label=display_label)

    ax.set_xlabel(f"{method_name} 1", fontsize=25)
    ax.set_ylabel(f"{method_name} 2", fontsize=25)
    ax.tick_params(labelsize=16)
    if len(unique_labels) > 1 or unique_labels[0] != -1:
        ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved latent projection to {output_path}")


def visualize_latent_space(model: JointVariationalAutoEncoder,
                           graphs: List[Data],
                           device: torch.device,
                           output_dir: str,
                           split_name: str,
                           max_points: int = 2000) -> None:
    embeddings, labels = collect_graph_latents(model, graphs, device)
    if embeddings.shape[0] == 0:
        print(f"No embeddings collected for {split_name} split")
        return

    if max_points and embeddings.shape[0] > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(embeddings.shape[0], max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    try:
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(embeddings_scaled)
        pca_path = os.path.join(output_dir, f"latent_{split_name}_pca.png")
        plot_latent_projection(pca_coords, labels, "PCA", pca_path)
    except Exception as exc:
        print(f"Skipping PCA for {split_name}: {exc}")

    n_samples = embeddings_scaled.shape[0]
    if n_samples < 3:
        return

    perplexity = min(30, max(5, n_samples // 10))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", n_iter=1000)
        tsne_coords = tsne.fit_transform(embeddings_scaled)
        tsne_path = os.path.join(output_dir, f"latent_{split_name}_tsne.png")
        plot_latent_projection(tsne_coords, labels, "t-SNE", tsne_path)
    except Exception as exc:
        print(f"Skipping t-SNE for {split_name}: {exc}")


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
    parser = argparse.ArgumentParser(description="Joint VGAE for structure, features, and labels")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to pickled PyG Data list")
    parser.add_argument("--output-dir", type=str, default="outputs_joint", help="Directory to store outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--hidden-dim-encoder", type=int, default=128, help="Hidden dimension for encoder GIN layers")
    parser.add_argument("--hidden-dim-struct", type=int, default=256, help="Hidden dimension for structure decoder MLP")
    parser.add_argument("--hidden-dim-feat", type=int, default=128, help="Hidden dimension for feature decoder message passing")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--n-layers-encoder", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--n-layers-struct-decoder", type=int, default=5, help="Number of structure decoder layers")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Fraction of graphs for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of graphs for validation")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--no-dfs-ordering", action="store_true", help="Disable DFS node ordering before training")
    parser.add_argument("--num-visualize", type=int, default=10, help="Number of graph pairs to visualize")
    parser.add_argument("--checkpoint", type=str, default="joint_vgae.pth.tar", help="Model checkpoint filename")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--lambda-feat", type=float, default=1.5, help="Weight for feature reconstruction loss")
    parser.add_argument("--lambda-label", type=float, default=0.5, help="Weight for label reconstruction loss")
    parser.add_argument("--beta-kl", type=float, default=0.05, help="Weight for KL divergence")
    parser.add_argument("--disable-teacher-forcing", action="store_true", help="Use predicted adjacency for feature decoding even during training")
    parser.add_argument("--beta-node", type=float, default=0.01, help="Weight for node-level KL divergence")
    parser.add_argument("--lambda-moment", type=float, default=0.1, help="Weight for feature moment matching loss")
    parser.add_argument("--latent-dim-node", type=int, default=None, help="Latent dimension for node-level variables")
    parser.add_argument("--tf-anneal-epochs", type=int, default=0, help="Epochs over which to anneal teacher forcing to zero")
    parser.add_argument("--beta-kl-warmup", type=int, default=0, help="Warmup epochs before reaching full beta KL weight")
    parser.add_argument("--beta-node-warmup", type=int, default=0, help="Warmup epochs before reaching full node KL weight")
    parser.add_argument("--latent-visualize", action="store_true", help="Project graph latents with PCA/t-SNE after evaluation")
    parser.add_argument("--latent-visualize-max", type=int, default=2000, help="Maximum number of graphs per split to plot for latent visualization")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint)
    viz_path = os.path.join(args.output_dir, "joint_graph_comparison.png")

    print("Loading dataset...")
    apply_dfs = not args.no_dfs_ordering
    print(f"Applying DFS ordering: {apply_dfs}")
    graphs, inferred_n_max = load_graphs(args.dataset_path, apply_dfs=apply_dfs, n_max_nodes=None)
    n_max_nodes = inferred_n_max
    feature_dim = graphs[0].x.size(1)
    num_classes = getattr(graphs[0], "num_classes", int(graphs[0].y.max().item() + 1))
    print(f"Loaded {len(graphs)} graphs | Feature dim: {feature_dim} | n_max_nodes: {n_max_nodes} | Classes: {num_classes}")

    train_graphs, val_graphs, test_graphs = split_dataset(graphs, args.train_ratio, args.val_ratio, args.seed)
    print(f"Split sizes -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)

    model = JointVariationalAutoEncoder(
        input_dim=feature_dim,
        hidden_dim_enc=args.hidden_dim_encoder,
        hidden_dim_struct_dec=args.hidden_dim_struct,
        hidden_dim_feat_dec=args.hidden_dim_feat,
        latent_dim=args.latent_dim,
        n_layers_enc=args.n_layers_encoder,
        n_layers_struct_dec=args.n_layers_struct_decoder,
        n_max_nodes=n_max_nodes,
        num_classes=num_classes,
        lambda_feat=args.lambda_feat,
        lambda_label=args.lambda_label,
        beta=args.beta_kl,
        beta_node=args.beta_node,
        lambda_moment=args.lambda_moment,
        use_teacher_forcing=not args.disable_teacher_forcing,
        tf_anneal_epochs=args.tf_anneal_epochs,
        latent_dim_node=args.latent_dim_node,
        beta_warmup_epochs=args.beta_kl_warmup,
        beta_node_warmup_epochs=args.beta_node_warmup
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    if args.epochs > 0:
        print("Starting joint VGAE training...")
        train_joint(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs, checkpoint_path)
    else:
        print("Skipping training (epochs=0)")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded best model from {checkpoint_path} (epoch {checkpoint['epoch']})")

    model.update_regularization(model.beta_warmup_epochs)

    print("Evaluating on test set...")
    metrics_summary, feature_analysis, gt_graphs, gen_graphs = evaluate_model(
        model,
        test_graphs,
        device,
        n_max_nodes,
        num_classes,
        args.num_visualize,
        args.output_dir
    )

    print(f"MMD (structure stats): {metrics_summary['mmd']:.6f}")
    print(f"Structure L1: {metrics_summary['structure_mae']:.6f}")
    print(f"Feature MAE: {metrics_summary['feature_mae']:.6f}")
    print(f"Feature Homophily (GT): {metrics_summary['feature_homophily_gt']:.6f}")
    print(f"Feature Homophily (Gen): {metrics_summary['feature_homophily_gen']:.6f}")

    if feature_analysis:
        channel_mae = np.array(feature_analysis["channel_mae"])
        channel_mean_abs_diff = np.array(feature_analysis["channel_mean_abs_diff"])
        channel_var_abs_diff = np.array(feature_analysis["channel_var_abs_diff"])

        print(f"Mean channel MAE: {channel_mae.mean():.6f}")
        print(f"Mean channel mean diff: {channel_mean_abs_diff.mean():.6f}")
        print(f"Mean channel variance diff: {channel_var_abs_diff.mean():.6f}")

        channel_stats_path = os.path.join(args.output_dir, "feature_channel_stats.csv")
        with open(channel_stats_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["channel", "gt_mean", "gen_mean", "gt_variance", "gen_variance", "channel_mae"])
            for idx, (gt_mean, gen_mean, gt_var, gen_var, mae_val) in enumerate(zip(
                feature_analysis["channel_mean_gt"],
                feature_analysis["channel_mean_gen"],
                feature_analysis["channel_var_gt"],
                feature_analysis["channel_var_gen"],
                feature_analysis["channel_mae"]
            )):
                writer.writerow([idx, gt_mean, gen_mean, gt_var, gen_var, mae_val])
        print(f"Saved per-channel statistics to {channel_stats_path}")

        plot_feature_statistics(feature_analysis, args.output_dir)

    if args.num_visualize > 0:
        visualize_graph_pairs(gt_graphs, gen_graphs, viz_path)

    metrics_path = os.path.join(args.output_dir, "metrics_joint.txt")
    with open(metrics_path, "w") as fw:
        fw.write(f"MMD: {metrics_summary['mmd']:.6f}\n")
        fw.write(f"Structure_L1: {metrics_summary['structure_mae']:.6f}\n")
        fw.write(f"Feature_MAE: {metrics_summary['feature_mae']:.6f}\n")
        fw.write(f"Feature_Homophily_GT: {metrics_summary['feature_homophily_gt']:.6f}\n")
        fw.write(f"Feature_Homophily_Gen: {metrics_summary['feature_homophily_gen']:.6f}\n")
        if feature_analysis:
            fw.write(f"Mean_Channel_MAE: {float(channel_mae.mean()):.6f}\n")
            fw.write(f"Mean_Channel_MeanDiff: {float(channel_mean_abs_diff.mean()):.6f}\n")
            fw.write(f"Mean_Channel_VarDiff: {float(channel_var_abs_diff.mean()):.6f}\n")
    print(f"Saved metrics to {metrics_path}")

    if args.latent_visualize:
        print("Visualizing latent space across splits...")
        for split_name, split_graphs in (("train", train_graphs), ("val", val_graphs), ("test", test_graphs)):
            if split_graphs:
                visualize_latent_space(model, split_graphs, device, args.output_dir, split_name, args.latent_visualize_max)


if __name__ == "__main__":
    main()
