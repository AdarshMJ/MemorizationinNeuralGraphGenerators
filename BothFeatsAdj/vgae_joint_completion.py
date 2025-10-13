import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, negative_sampling, to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix


def normalize_features_inplace(graphs):
    for data in graphs:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        std = std.clamp_min(1e-6)
        data.x = (data.x - mean) / std
    return graphs


# ------------------------- Encoder -------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


# ------------------------- Joint VGAE Model -------------------------
class JointVGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, feat_dim, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, latent_dim, dropout)
        self.cond_proj = nn.Linear(in_channels, latent_dim)

        # Feature decoder (matrix completion style)
        self.feat_decoder_W = nn.Parameter(torch.randn(latent_dim, feat_dim) * 0.01)
        self.feat_decoder_b = nn.Parameter(torch.zeros(feat_dim))

        # Label decoder (classification head)
        self.label_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def condition_latent(self, z, x):
        return z + F.relu(self.cond_proj(x))

    def decode_adj(self, z, x, sigmoid=True):
        h = self.condition_latent(z, x)
        adj_logits = h @ h.t()
        return torch.sigmoid(adj_logits) if sigmoid else adj_logits

    def decode_feats(self, z):
        return z @ self.feat_decoder_W + self.feat_decoder_b

    def decode_labels(self, z):
        return self.label_decoder(z)

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'A_pred': self.decode_adj(z, x),
            'X_pred': self.decode_feats(z),
            'Y_logits': self.decode_labels(z)
        }


# ------------------------- Losses -------------------------
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_losses(model, batch, device, lambdas):
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index)
    z = out['z']
    h = model.condition_latent(z, batch.x)

    # Adjacency loss (pos + neg BCE)
    pos_edge_index = batch.edge_index
    pos_logits = (h[pos_edge_index[0]] * h[pos_edge_index[1]]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=batch.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    neg_logits = (h[neg_edge_index[0]] * h[neg_edge_index[1]]).sum(dim=1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

    adj_loss = pos_loss + neg_loss

    # Feature reconstruction loss
    feat_loss = F.mse_loss(out['X_pred'], batch.x)

    # Label classification loss (optional)
    if hasattr(batch, 'y') and batch.y is not None:
        label_loss = F.cross_entropy(out['Y_logits'], batch.y.long())
    else:
        label_loss = torch.tensor(0.0, device=device)

    kl = kl_loss(out['mu'], out['logvar'])

    total_loss = adj_loss + lambdas['lambda_x'] * feat_loss + lambdas['lambda_y'] * label_loss + lambdas['beta'] * kl
    return total_loss, adj_loss, feat_loss, label_loss, kl


# ------------------------- Train / Evaluate -------------------------
def train_one_epoch(model, loader, optimizer, device, lambdas):
    model.train()
    stats = {'total_loss': 0.0, 'adj_loss': 0.0, 'feat_loss': 0.0, 'label_loss': 0.0, 'kl': 0.0}
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        total_loss, adj_loss, feat_loss, label_loss, kl = compute_losses(model, batch, device, lambdas)
        total_loss.backward()
        optimizer.step()

        stats['total_loss'] += total_loss.item()
        stats['adj_loss'] += adj_loss.item()
        stats['feat_loss'] += feat_loss.item()
        stats['label_loss'] += label_loss.item()
        stats['kl'] += kl.item()
        num_batches += 1

    if num_batches:
        for key in stats:
            stats[key] /= num_batches

    return stats


def evaluate(model, loader, device):
    model.eval()
    feat_error_sum = 0.0
    label_correct = 0.0
    label_total = 0
    node_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            feat_error_sum += torch.abs(out['X_pred'] - batch.x).sum().item()
            node_total += batch.num_nodes

            if hasattr(batch, 'y') and batch.y is not None:
                preds = out['Y_logits'].argmax(dim=1)
                label_correct += (preds == batch.y).float().sum().item()
                label_total += batch.y.numel()

    feat_mae = feat_error_sum / node_total if node_total else None
    label_acc = label_correct / label_total if label_total else None
    return {'feat_mae': feat_mae, 'label_acc': label_acc}


def collect_reconstruction_stats(model, loader, device):
    model.eval()
    feat_errors = []
    label_trues = []
    label_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch.x, batch.edge_index)
            x_recon = model.decode_feats(mu)
            feat_errors.append(torch.abs(x_recon - batch.x).view(-1).cpu())

            if hasattr(batch, 'y') and batch.y is not None:
                logits = model.decode_labels(mu)
                label_trues.append(batch.y.view(-1).cpu())
                label_preds.append(logits.argmax(dim=1).view(-1).cpu())

    feat_errors = torch.cat(feat_errors, dim=0) if feat_errors else torch.tensor([])
    label_trues = torch.cat(label_trues, dim=0) if label_trues else torch.tensor([])
    label_preds = torch.cat(label_preds, dim=0) if label_preds else torch.tensor([])
    return feat_errors.numpy(), label_trues.numpy(), label_preds.numpy()


def plot_graph_grid(real_graphs, generated_graphs, out_path, num_show=5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = min(num_show, len(real_graphs), len(generated_graphs))
    if count == 0:
        return

    fig, axes = plt.subplots(2, count, figsize=(4 * count, 8))

    for idx in range(count):
        gt = real_graphs[idx]
        gen = generated_graphs[idx]

        gt_graph = to_networkx(gt, to_undirected=True)
        gen_graph = to_networkx(gen, to_undirected=True)
        layout = nx.spring_layout(gt_graph, seed=idx)
        gen_layout = nx.spring_layout(gen_graph, seed=idx)

        axes[0, idx].axis('off')
        nx.draw_networkx(gt_graph, pos=layout, node_size=100, ax=axes[0, idx])
        axes[0, idx].set_xlabel(f"Nodes: {gt_graph.number_of_nodes()}\nEdges: {gt_graph.number_of_edges()}", fontsize=25)

        axes[1, idx].axis('off')
        nx.draw_networkx(gen_graph, pos=gen_layout, node_size=100, ax=axes[1, idx])
        axes[1, idx].set_xlabel(f"Nodes: {gen_graph.number_of_nodes()}\nEdges: {gen_graph.number_of_edges()}", fontsize=25)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_feature_error_hist(errors, out_path, bins=50):
    if errors.size == 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=bins, color='tab:blue', alpha=0.8)
    ax.set_xlabel('Absolute Feature Error', fontsize=25)
    ax.set_ylabel('Count', fontsize=25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_label_confusion(true_labels, pred_labels, out_path):
    if true_labels.size == 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted', fontsize=25)
    ax.set_ylabel('True', fontsize=25)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def sample_graphs(model, num_nodes, num_samples, device, threshold=0.5):
    model.eval()
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn((num_nodes, model.encoder.conv_mu.out_channels), device=device)
            x_sample = model.decode_feats(z)
            adj_probs = model.decode_adj(z, x_sample, sigmoid=True).clamp(min=0.0, max=1.0)
            if threshold is None:
                adj_sample = torch.bernoulli(adj_probs)
            else:
                adj_sample = (adj_probs >= threshold).float()
            adj_sample = torch.triu(adj_sample, diagonal=1)
            adj_sample = adj_sample + adj_sample.t()
            edge_index, edge_weight = dense_to_sparse(adj_sample)
            y_logits = model.decode_labels(z)
            y_sample = y_logits.argmax(dim=1)

            data = Data(x=x_sample.cpu(), edge_index=edge_index.cpu(), y=y_sample.cpu())
            data.num_nodes = num_nodes
            data.edge_weight = edge_weight.cpu()
            samples.append(data)

    return samples


# ------------------------- Entry Point -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Joint VGAE training and sampling pipeline.')
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl', help='Path to pickled list of graphs.')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to store generated artefacts.')
    parser.add_argument('--generated-file', type=str, default='generated_graphs.pkl', help='Filename for storing sampled graphs pickle.')
    parser.add_argument('--train-frac', type=float, default=0.6, help='Fraction of graphs used for training.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training and validation loaders.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes for data loading.')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension in GCN encoder and label decoder.')
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension for the VAE.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in encoder.')
    parser.add_argument('--lambda-x', type=float, default=1000.0, help='Weight for feature reconstruction loss.')
    parser.add_argument('--lambda-y', type=float, default=1.0, help='Weight for label classification loss.')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for KL divergence term.')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of graphs to sample after training.')
    parser.add_argument('--num-show', type=int, default=5, help='Number of graphs to show in comparison grid.')
    parser.add_argument('--feature-error-bins', type=int, default=50, help='Histogram bins for feature reconstruction error plot.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Adjacency threshold; set negative to use Bernoulli sampling.')
    parser.add_argument('--device', type=str, default='auto', help="Computation device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--eval-interval', type=int, default=1, help='Epoch interval for validation logging.')
    parser.add_argument('--normalize-features', action='store_true', help='Apply per-graph mean-std normalization to node features.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    if not dataset:
        raise ValueError('Loaded dataset is empty.')

    if args.normalize_features:
        dataset = normalize_features_inplace(dataset)

    feat_dim = dataset[0].x.size(1)
    num_classes = dataset[0].num_classes
    num_nodes = dataset[0].num_nodes

    train_size = int(len(dataset) * args.train_frac)
    train_size = max(1, min(train_size, len(dataset) - 1))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = JointVGAE(in_channels=feat_dim, hidden_channels=args.hidden_dim, latent_dim=args.latent_dim,
                      feat_dim=feat_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lambdas = {'lambda_x': args.lambda_x, 'lambda_y': args.lambda_y, 'beta': args.beta}

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device, lambdas)
        if epoch % args.eval_interval == 0 or epoch == 1 or epoch == args.epochs:
            eval_stats = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:03d} | total: {stats['total_loss']:.4f} | adj: {stats['adj_loss']:.4f} "
                f"| feat: {stats['feat_loss']:.4f} | feat_mae: {eval_stats['feat_mae']:.4f} | label_acc: {eval_stats['label_acc']}"
            )

    threshold = None if args.threshold < 0 else args.threshold
    generated = sample_graphs(model, num_nodes=num_nodes, num_samples=args.num_samples, device=device, threshold=threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    generated_path = os.path.join(args.output_dir, args.generated_file)
    with open(generated_path, 'wb') as f:
        pickle.dump(generated, f)

    recon_errors, label_true, label_pred = collect_reconstruction_stats(model, val_loader, device)

    val_examples = [val_dataset[i] for i in range(min(args.num_show, len(val_dataset)))]
    plot_graph_grid(val_examples, generated, os.path.join(args.output_dir, 'graph_comparison.png'), num_show=args.num_show)
    plot_feature_error_hist(recon_errors, os.path.join(args.output_dir, 'feature_error_hist.png'), bins=args.feature_error_bins)
    plot_label_confusion(label_true, label_pred, os.path.join(args.output_dir, 'label_confusion.png'))


if __name__ == '__main__':
    main()
