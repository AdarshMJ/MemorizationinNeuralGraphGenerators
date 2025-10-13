import argparse
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder
from experiments.memorization_split import SplitResult, perform_split

from polygraph.metrics import StandardPGD

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
from matplotlib.lines import Line2D
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx
from main_onlyvgae import dfs_reorder_graph, ensure_padded_adjacency
from utils import construct_nx_from_adj, gen_stats


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)


def _safe_torch_load(path: Path, *, weights_only: bool = False):
    load_kwargs = {"map_location": "cpu"}
    if weights_only:
        load_kwargs["weights_only"] = True
    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.load(path, **load_kwargs)


@dataclass
class StudyConfig:
    dataset_path: str
    csv_path: str
    output_dir: str
    subset_sizes: Sequence[int]
    epochs: int
    lr: float
    batch_size: int
    val_ratio: float
    hidden_dim_encoder: int
    hidden_dim_decoder: int
    latent_dim: int
    n_layers_encoder: int
    n_layers_decoder: int
    samples_per_model: int
    wl_iterations: int
    seed: int
    device: str
    test_ratio: float
    bins: int
    force_resplit: bool
    apply_dfs: bool
    decoder_node_hidden_dim: Optional[int]
    decoder_dropout: float
    decoder_use_gumbel: bool
    decoder_gumbel_tau: float
    viz_samples: int
    pgd_variant: str
    pgd_bootstrap: int
    resume: bool


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_create_split(
    config: StudyConfig,
) -> Tuple[List[Data], SplitResult, Dict[str, Dict[str, float]]]:
    split_dir = Path(config.output_dir) / "splits"
    if not config.force_resplit and split_dir.exists():
        s1_path = split_dir / "s1_indices.pt"
        s2_path = split_dir / "s2_indices.pt"
        test_path = split_dir / "test_indices.pt"
        stats_path = split_dir / "stats.json"
        if s1_path.is_file() and s2_path.is_file() and test_path.is_file() and stats_path.is_file():
            graphs = _load_graphs(config.dataset_path)
            split = SplitResult(
                s1_indices=_safe_torch_load(s1_path),
                s2_indices=_safe_torch_load(s2_path),
                test_indices=_safe_torch_load(test_path),
            )
            with open(stats_path, "r", encoding="utf8") as handle:
                stats = json.load(handle)
            return graphs, split, stats

    graphs, split, stats = perform_split(
        dataset_path=config.dataset_path,
        csv_path=config.csv_path,
        test_ratio=config.test_ratio,
        n_bins=config.bins,
        seed=config.seed,
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(split.s1_indices, split_dir / "s1_indices.pt")
    torch.save(split.s2_indices, split_dir / "s2_indices.pt")
    torch.save(split.test_indices, split_dir / "test_indices.pt")
    with open(split_dir / "stats.json", "w", encoding="utf8") as handle:
        json.dump(stats, handle, indent=2)
    return graphs, split, stats


def _load_graphs(dataset_path: str) -> List[Data]:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(dataset_path)
    if dataset_path.endswith(".pt"):
        graphs = _safe_torch_load(Path(dataset_path))
    else:
        import pickle

        with open(dataset_path, "rb") as handle:
            graphs = pickle.load(handle)
    if not isinstance(graphs, list) or not all(isinstance(g, Data) for g in graphs):
        raise ValueError("Dataset must be a list of torch_geometric.data.Data")
    return graphs


def determine_n_max_nodes(graphs: Sequence[Data]) -> int:
    n_from_attr = [g.A.size(-1) for g in graphs if hasattr(g, "A") and g.A is not None]
    if n_from_attr:
        return max(n_from_attr)
    return max(g.x.size(0) for g in graphs)


def preprocess_graph(graph: Data, n_max_nodes: int, apply_dfs: bool) -> Data:
    if apply_dfs:
        return dfs_reorder_graph(graph, n_max_nodes)
    return ensure_padded_adjacency(graph, n_max_nodes)


def select_quantile_indices(indices: Sequence[int], values: np.ndarray, sample_size: int) -> List[int]:
    if sample_size > len(indices):
        raise ValueError(f"Requested {sample_size} samples from set of size {len(indices)}")
    if sample_size == len(indices):
        return list(indices)
    idx_array = np.array(indices)
    sorted_idx = idx_array[np.argsort(values[idx_array])]
    positions = np.linspace(0, len(sorted_idx) - 1, num=sample_size, dtype=int)
    return sorted_idx[positions].tolist()


def split_train_val(graphs: Sequence[Data], val_ratio: float) -> Tuple[List[Data], List[Data]]:
    if not graphs:
        raise ValueError("Empty graph list for train/val split")
    n_total = len(graphs)
    n_val = max(1, int(round(val_ratio * n_total))) if n_total > 1 else 0
    n_val = min(n_val, n_total - 1) if n_total > 1 else 0
    perm = np.random.permutation(n_total)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]
    val_graphs = [graphs[idx] for idx in val_indices]
    train_graphs = [graphs[idx] for idx in train_indices]
    return train_graphs, val_graphs


def clone_graphs(graphs: Sequence[Data]) -> List[Data]:
    return [graph.clone() for graph in graphs]


def build_dataloaders(graphs: Sequence[Data], batch_size: int) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=True)


def _load_model_metadata(work_dir: Path) -> Dict[str, object]:
    metadata_path = work_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read metadata from %s: %s", metadata_path, exc)
    return {}


def _save_model_metadata(
    work_dir: Path,
    *,
    n_max_nodes: int,
    train_history: Sequence[float],
    val_history: Sequence[float],
) -> None:
    metadata = {
        "n_max_nodes": int(n_max_nodes),
        "train_history": [float(x) for x in train_history],
        "val_history": [float(x) for x in val_history],
    }
    metadata_path = work_dir / "metadata.json"
    try:
        with metadata_path.open("w", encoding="utf8") as handle:
            json.dump(metadata, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not save metadata to %s: %s", metadata_path, exc)


def train_vgae(
    graphs: Sequence[Data],
    config: StudyConfig,
    device: torch.device,
    work_dir: Path,
) -> Tuple[VariationalAutoEncoder, int, List[float], List[float]]:
    n_max_nodes = determine_n_max_nodes(graphs)
    feature_dim = graphs[0].x.size(1)

    processed = [preprocess_graph(g, n_max_nodes, config.apply_dfs) for g in clone_graphs(graphs)]
    train_graphs, val_graphs = split_train_val(processed, config.val_ratio)

    model = VariationalAutoEncoder(
        input_dim=feature_dim,
        hidden_dim_enc=config.hidden_dim_encoder,
        hidden_dim_dec=config.hidden_dim_decoder,
        latent_dim=config.latent_dim,
        n_layers_enc=config.n_layers_encoder,
        n_layers_dec=config.n_layers_decoder,
        n_max_nodes=n_max_nodes,
        decoder_node_hidden_dim=config.decoder_node_hidden_dim,
        decoder_dropout=config.decoder_dropout,
        decoder_use_gumbel=config.decoder_use_gumbel,
        decoder_gumbel_tau=config.decoder_gumbel_tau,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    train_loader = build_dataloaders(train_graphs, config.batch_size)
    val_loader = build_dataloaders(val_graphs, config.batch_size) if val_graphs else None

    best_val_loss = float("inf")
    logger.info(
        "Training VGAE (%d graphs train/%d val, n_max_nodes=%d) for %d epochs",
        len(train_graphs),
        len(val_graphs),
        n_max_nodes,
        config.epochs,
    )

    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, recon, kld = model.loss_function(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_graphs += batch.num_graphs

        scheduler.step()

        avg_train_loss = total_loss / max(total_graphs, 1)

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_graphs_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    loss, _, _ = model.loss_function(batch)
                    val_loss += loss.item()
                    val_graphs_count += batch.num_graphs
            avg_val_loss = val_loss / max(val_graphs_count, 1)
        else:
            avg_val_loss = avg_train_loss

        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        logger.info(
            "Epoch %d/%d - train_loss: %.4f | val_loss: %.4f",
            epoch,
            config.epochs,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }
            torch.save(checkpoint, work_dir / "best_model.pth.tar")
            logger.info("Saved new best checkpoint at epoch %d (val_loss=%.4f)", epoch, best_val_loss)

    best_checkpoint = _safe_torch_load(work_dir / "best_model.pth.tar", weights_only=True)
    if isinstance(best_checkpoint, dict) and "state_dict" in best_checkpoint:
        state_dict = best_checkpoint["state_dict"]
    else:
        state_dict = best_checkpoint
    model.load_state_dict(state_dict)
    logger.info("Restored best checkpoint with val_loss=%.4f", best_val_loss)
    return model, n_max_nodes, train_history, val_history


def get_or_train_vgae(
    graphs: Sequence[Data],
    config: StudyConfig,
    device: torch.device,
    work_dir: Path,
) -> Tuple[VariationalAutoEncoder, int, List[float], List[float], bool]:
    best_checkpoint_path = work_dir / "best_model.pth.tar"
    metadata = _load_model_metadata(work_dir)
    should_resume = config.resume and best_checkpoint_path.is_file()

    if should_resume:
        n_max_nodes = int(metadata.get("n_max_nodes", determine_n_max_nodes(graphs)))
        model = VariationalAutoEncoder(
            input_dim=graphs[0].x.size(1),
            hidden_dim_enc=config.hidden_dim_encoder,
            hidden_dim_dec=config.hidden_dim_decoder,
            latent_dim=config.latent_dim,
            n_layers_enc=config.n_layers_encoder,
            n_layers_dec=config.n_layers_decoder,
            n_max_nodes=n_max_nodes,
            decoder_node_hidden_dim=config.decoder_node_hidden_dim,
            decoder_dropout=config.decoder_dropout,
            decoder_use_gumbel=config.decoder_use_gumbel,
            decoder_gumbel_tau=config.decoder_gumbel_tau,
        ).to(device)

        checkpoint = _safe_torch_load(best_checkpoint_path, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        logger.info("Loaded existing checkpoint from %s", best_checkpoint_path)

        train_history = [float(x) for x in metadata.get("train_history", [])]
        val_history = [float(x) for x in metadata.get("val_history", [])]
        if any(key not in metadata for key in ("n_max_nodes", "train_history", "val_history")):
            _save_model_metadata(
                work_dir,
                n_max_nodes=n_max_nodes,
                train_history=train_history,
                val_history=val_history,
            )
        return model, n_max_nodes, train_history, val_history, True

    model, n_max_nodes, train_history, val_history = train_vgae(graphs, config, device, work_dir)
    _save_model_metadata(
        work_dir,
        n_max_nodes=n_max_nodes,
        train_history=train_history,
        val_history=val_history,
    )
    return model, n_max_nodes, train_history, val_history, False


def to_networkx(graphs: Sequence[Data]) -> List:
    nx_graphs = []
    for data in graphs:
        adj = data.A.squeeze(0)
        adj = (adj + adj.t()) / 2.0
        adj = torch.clamp(adj, 0.0, 1.0)
        adj = (adj > 0.5).float().cpu().numpy()
        nx_graphs.append(construct_nx_from_adj(adj))
    return nx_graphs


def generate_graphs(model: VariationalAutoEncoder, n_samples: int, device: torch.device) -> List:
    model.eval()
    latent_dim = model.fc_mu.out_features
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        adj = model.decode_mu(z).cpu()
    graphs = []
    for i in range(adj.size(0)):
        mat = adj[i]
        mat = (mat + mat.t()) / 2.0
        mat = torch.clamp(mat, 0.0, 1.0)
        mat = (mat > 0.5).float().numpy()
        graphs.append(construct_nx_from_adj(mat))
    return graphs


def _layout_graph(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return nx.spring_layout(graph, seed=0)
    return nx.kamada_kawai_layout(graph)


def visualize_graph_sets(
    graph_sets: Sequence[Sequence[nx.Graph]],
    row_labels: Sequence[str],
    output_path: Path,
    samples_per_row: int,
    row_colors: Optional[Sequence[str]] = None,
) -> None:
    if not graph_sets:
        return

    n_rows = len(graph_sets)
    if row_colors is None:
        default_palette = ["#f2a154", "#6c8af2", "#50a14f", "#d45d79", "#845EC2", "#FFC75F"]
        row_colors = [default_palette[idx % len(default_palette)] for idx in range(n_rows)]
    else:
        row_colors = list(row_colors)
        if len(row_colors) < n_rows:
            raise ValueError("row_colors must match number of graph sets")

    samples = [min(len(graphs), samples_per_row) for graphs in graph_sets]
    if any(count == 0 for count in samples):
        logger.warning("Skipping visualization %s because one of the graph sets is empty", output_path)
        return

    max_cols = max(samples)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(3.5 * max_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    if max_cols == 1:
        axes = axes.reshape(n_rows, 1)

    legend_handles: List[Line2D] = []

    for row_idx, (graphs, label) in enumerate(zip(graph_sets, row_labels)):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("", fontsize=25)
            ax.set_ylabel("", fontsize=25)

            if col_idx < samples[row_idx]:
                graph = graphs[col_idx]
                pos = _layout_graph(graph)
                nx.draw_networkx(
                    graph,
                    pos=pos,
                    ax=ax,
                    node_size=120,
                    width=1.0,
                    edge_color="#555555",
                    node_color=row_colors[row_idx],
                    with_labels=False,
                )
                ax.set_xlabel(f"Sample {col_idx + 1}", fontsize=25)
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=25)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=row_colors[row_idx],
                markeredgecolor="#333333",
                markersize=10,
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=16,
            borderaxespad=0.0,
        )

    plt.tight_layout(rect=(0, 0, 0.9, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info("Saved graph comparison grid to %s", output_path)


def assign_dummy_labels(graphs: Iterable) -> None:
    for g in graphs:
        for node in g.nodes():
            g.nodes[node]["label"] = 1


def _sample_equal_sized_graphs(
    graphs_a: Sequence[nx.Graph],
    graphs_b: Sequence[nx.Graph],
    sample_size: int,
    rng: np.random.Generator,
    replace: bool,
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    idx_a = rng.choice(len(graphs_a), size=sample_size, replace=replace)
    idx_b = rng.choice(len(graphs_b), size=sample_size, replace=replace)
    return ([graphs_a[i] for i in idx_a], [graphs_b[i] for i in idx_b])


def compute_pgd_metric(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    config: StudyConfig,
    rng: np.random.Generator,
) -> Dict[str, object]:
    equal_size = min(len(reference_graphs), len(generated_graphs))
    if equal_size < 4:
        logger.warning(
            "Skipping PGD computation for a pair of sets with sizes %d and %d (requires at least 4 graphs per set).",
            len(reference_graphs),
            len(generated_graphs),
        )
        return {
            "pgd": float("nan"),
            "pgd_descriptor": None,
            "subscores": {},
            "samples": [],
        }

    ref_subset, gen_subset = _sample_equal_sized_graphs(
        reference_graphs,
        generated_graphs,
        equal_size,
        rng,
        replace=False,
    )
    metric = StandardPGD(ref_subset)
    result = metric.compute(gen_subset)

    pgd_value = float(result["pgd"])
    subscores = {key: float(val) for key, val in result["subscores"].items()}

    samples: List[float] = [pgd_value]
    for boot_idx in range(max(0, config.pgd_bootstrap)):
        ref_boot, gen_boot = _sample_equal_sized_graphs(
            reference_graphs,
            generated_graphs,
            equal_size,
            rng,
            replace=True,
        )
        metric_boot = StandardPGD(ref_boot)
        boot_result = metric_boot.compute(gen_boot)
        samples.append(float(boot_result["pgd"]))

    return {
        "pgd": pgd_value,
        "pgd_descriptor": result["pgd_descriptor"],
        "subscores": subscores,
        "samples": samples,
    }


def _serialize_pgd_result(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "pgd": float(result.get("pgd", float("nan"))),
        "pgd_descriptor": result.get("pgd_descriptor"),
        "subscores": {
            key: float(val) for key, val in result.get("subscores", {}).items()
        },
        "samples": [float(x) for x in result.get("samples", [])],
    }


def wl_similarity_matrix(graphs_a: Sequence, graphs_b: Sequence, iterations: int) -> np.ndarray:
    assign_dummy_labels(graphs_a)
    assign_dummy_labels(graphs_b)
    gk_a = graph_from_networkx(graphs_a, "label")
    gk_b = graph_from_networkx(graphs_b, "label")
    kernel = WeisfeilerLehman(n_iter=iterations, normalize=True, base_graph_kernel=VertexHistogram)
    kernel.fit(gk_a)
    sims = kernel.transform(gk_b)
    return np.asarray(sims)


def closest_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    return sim_matrix.max(axis=1)


def cross_similarity(graphs_a: Sequence, graphs_b: Sequence, iterations: int) -> np.ndarray:
    assign_dummy_labels(graphs_a)
    assign_dummy_labels(graphs_b)
    gk_all = graph_from_networkx(list(graphs_a) + list(graphs_b), "label")
    kernel = WeisfeilerLehman(n_iter=iterations, normalize=True, base_graph_kernel=VertexHistogram)
    K = kernel.fit_transform(gk_all)
    n_a = len(graphs_a)
    n_b = len(graphs_b)
    return K[:n_a, n_a : n_a + n_b].flatten()


def compute_mmd(gt_stats: List[List[float]], gen_stats_list: List[List[float]]) -> float:
    gt_arr = np.asarray(gt_stats, dtype=np.float64)
    gen_arr = np.asarray(gen_stats_list, dtype=np.float64)

    if gt_arr.ndim == 1:
        gt_arr = gt_arr[np.newaxis, :]
    if gen_arr.ndim == 1:
        gen_arr = gen_arr[np.newaxis, :]

    x = torch.from_numpy(gt_arr)
    y = torch.from_numpy(gen_arr)

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

    def gaussian_kernel(tensor_a, tensor_b):
        x_norm = (tensor_a ** 2).sum(dim=1).view(-1, 1)
        y_norm = (tensor_b ** 2).sum(dim=1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * tensor_a @ tensor_b.t()
        dist = torch.clamp(dist, min=0.0)
        return torch.exp(-gamma * dist)

    k_xx = gaussian_kernel(x, x)
    k_yy = gaussian_kernel(y, y)
    k_xy = gaussian_kernel(x, y)

    n = x.size(0)
    m = y.size(0)
    mmd_sq = k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2.0 * k_xy.sum() / (n * m)
    return torch.sqrt(torch.clamp(mmd_sq, min=0.0)).item()


def compute_stats_array(graphs: Sequence[nx.Graph]) -> np.ndarray:
    stats = [gen_stats(g) for g in graphs]
    if not stats:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(stats, dtype=np.float64)


def summarize_stats_relationship(stats_a: np.ndarray, stats_b: np.ndarray) -> Dict[str, float]:
    if stats_a.size == 0 or stats_b.size == 0:
        return {"mmd": float("nan"), "mse": float("nan")}

    mmd_value = compute_mmd(stats_a.tolist(), stats_b.tolist())
    mean_diff = stats_a.mean(axis=0) - stats_b.mean(axis=0)
    mse_value = float(np.mean(mean_diff ** 2))

    return {"mmd": mmd_value, "mse": mse_value}


def plot_histograms(
    results: Dict[int, Dict[str, np.ndarray]],
    output_path: Path,
    bins: int = 30,
) -> None:
    subset_sizes = sorted(results.keys())
    n_subsets = len(subset_sizes)
    if n_subsets == 0:
        logger.warning("No subsets provided; skipping histogram plot at %s", output_path)
        return

    n_rows = 1 if n_subsets <= 4 else 2
    n_cols = math.ceil(n_subsets / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    flat_axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for idx, subset_size in enumerate(subset_sizes):
        ax = flat_axes[idx]
        data = results[subset_size]
        ax.hist(
            data["train_similarity"],
            bins=bins,
            alpha=0.6,
            color="#f4b942",
            label="Sample and Closest Train Graph",
        )
        ax.hist(
            data["cross_similarity"],
            bins=bins,
            alpha=0.6,
            color="#6c8af2",
            label="Samples from Two VGAEs",
        )
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("WL Kernel Similarity", fontsize=25)
        if idx % n_cols == 0:
            ax.set_ylabel("Frequency", fontsize=25)
        else:
            ax.set_ylabel("")
        ax.text(
            0.5,
            1.04,
            f"N = {subset_size}",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            clip_on=False,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # turn off any unused subplots
    for ax in flat_axes[len(subset_sizes) :]:
        ax.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pgd_histograms(
    results: Dict[int, Dict[str, np.ndarray]],
    output_path: Path,
    bins: int = 30,
) -> None:
    subset_sizes = sorted(results.keys())
    if not subset_sizes:
        logger.warning("No subsets provided; skipping PGD histogram plot at %s", output_path)
        return

    n_rows = 1 if len(subset_sizes) <= 4 else 2
    n_cols = math.ceil(len(subset_sizes) / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    flat_axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for idx, subset_size in enumerate(subset_sizes):
        ax = flat_axes[idx]
        data = results[subset_size]
        pgd_train = data.get("pgd_train", np.array([]))
        pgd_cross = data.get("pgd_cross", np.array([]))

        if pgd_train.size == 0 and pgd_cross.size == 0:
            ax.axis("off")
            continue

        ax.hist(
            pgd_train,
            bins=bins,
            alpha=0.6,
            color="#f28482",
            label="PGD: Generated vs Train set",
        )
        ax.hist(
            pgd_cross,
            bins=bins,
            alpha=0.6,
            color="#33658a",
            label="PGD: Generated cross",
        )
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("PGD Score", fontsize=25)
        if idx % n_cols == 0:
            ax.set_ylabel("Frequency", fontsize=25)
        ax.text(
            0.5,
            1.04,
            f"N = {subset_size}",
            fontsize=18,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            clip_on=False,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in flat_axes[len(subset_sizes) :]:
        ax.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_loss_curves(
    loss_histories: Dict[int, Dict[str, List[float]]],
    output_path: Path,
) -> None:
    if not loss_histories:
        logger.warning("No loss histories recorded; skipping loss curve plot at %s", output_path)
        return

    subset_sizes = sorted(loss_histories.keys())
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    subset_handles: List[Line2D] = []

    for idx, subset_size in enumerate(subset_sizes):
        color = palette[idx % len(palette)]
        history = loss_histories[subset_size]
        train_curve = history.get("train_mean", [])
        val_curve = history.get("val_mean", [])
        if not train_curve:
            continue
        epochs = np.arange(1, len(train_curve) + 1)
        ax.plot(epochs, train_curve, color=color, linestyle="-", linewidth=2)
        if val_curve:
            ax.plot(epochs, val_curve, color=color, linestyle="--", linewidth=2)
        subset_handles.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=2, label=f"N={subset_size}")
        )

    style_handles = [
        Line2D([0], [0], color="#000000", linestyle="-", linewidth=2, label="Train"),
        Line2D([0], [0], color="#000000", linestyle="--", linewidth=2, label="Val"),
    ]

    if subset_handles:
        legend1 = ax.legend(handles=subset_handles, loc="upper right", fontsize=14, frameon=False)
        ax.add_artist(legend1)
    ax.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=14, frameon=False)

    ax.set_xlabel("Epoch", fontsize=25)
    ax.set_ylabel("Loss", fontsize=25)
    ax.tick_params(axis="both", labelsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

def evaluate_on_test_set(
    model: VariationalAutoEncoder,
    test_graphs: Sequence[Data],
    device: torch.device,
    iterations: int,
    samples_per_model: int,
    apply_dfs: bool,
) -> Dict[str, float]:
    n_max_nodes = determine_n_max_nodes(test_graphs)
    processed = [preprocess_graph(g, n_max_nodes, apply_dfs) for g in clone_graphs(test_graphs)]
    nx_test = to_networkx(processed)
    generated = generate_graphs(model, samples_per_model, device)

    sim_matrix = wl_similarity_matrix(nx_test, generated, iterations)
    closest = closest_similarity(sim_matrix)

    gt_stats = [gen_stats(g) for g in nx_test]
    gen_stats_list = [gen_stats(g) for g in generated]

    return {
        "mean_closest_similarity": float(np.mean(closest)),
        "std_closest_similarity": float(np.std(closest)),
        "mmd_stats": compute_mmd(gt_stats, gen_stats_list),
    }


def run_study(config: StudyConfig) -> None:
    set_seed(config.seed)
    graphs, split, stats = load_or_create_split(config)
    feature_hom = _load_feature_hom(config)

    device = torch.device(config.device)
    work_root = Path(config.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, np.ndarray]] = {}
    metrics: Dict[int, Dict[str, object]] = {}
    loss_histories: Dict[int, Dict[str, List[float]]] = {}

    test_graphs = [graphs[idx] for idx in split.test_indices]

    total_subsets = len(config.subset_sizes)
    for subset_idx, subset_size in enumerate(config.subset_sizes, start=1):
        logger.info(
            "Starting subset %d/%d with N=%d",
            subset_idx,
            total_subsets,
            subset_size,
        )
        subset_dir = work_root / f"N_{subset_size}"
        subset_dir.mkdir(exist_ok=True)

        s1_subset_idx = select_quantile_indices(split.s1_indices, feature_hom, subset_size)
        s2_subset_idx = select_quantile_indices(split.s2_indices, feature_hom, subset_size)

        graphs_s1 = [graphs[idx] for idx in s1_subset_idx]
        graphs_s2 = [graphs[idx] for idx in s2_subset_idx]

        model_dir_1 = subset_dir / "vgae1"
        model_dir_2 = subset_dir / "vgae2"
        model_dir_1.mkdir(exist_ok=True)
        model_dir_2.mkdir(exist_ok=True)

        model1, nmax1, train_hist1, val_hist1, reused1 = get_or_train_vgae(
            graphs_s1,
            config,
            device,
            model_dir_1,
        )
        model2, nmax2, train_hist2, val_hist2, reused2 = get_or_train_vgae(
            graphs_s2,
            config,
            device,
            model_dir_2,
        )

        if reused1:
            logger.info("Reused existing VGAE1 checkpoint for N=%d", subset_size)
        if reused2:
            logger.info("Reused existing VGAE2 checkpoint for N=%d", subset_size)

        logger.info("Models ready for N=%d; computing similarity metrics", subset_size)

        train_hist1_arr = np.asarray(train_hist1, dtype=np.float64)
        train_hist2_arr = np.asarray(train_hist2, dtype=np.float64)
        val_hist1_arr = np.asarray(val_hist1, dtype=np.float64)
        val_hist2_arr = np.asarray(val_hist2, dtype=np.float64)

        if train_hist1_arr.size and train_hist2_arr.size:
            min_len_train = min(train_hist1_arr.size, train_hist2_arr.size)
            train_avg = (train_hist1_arr[:min_len_train] + train_hist2_arr[:min_len_train]) / 2.0
        else:
            train_avg = np.array([], dtype=np.float64)

        if val_hist1_arr.size and val_hist2_arr.size:
            min_len_val = min(val_hist1_arr.size, val_hist2_arr.size)
            val_avg = (val_hist1_arr[:min_len_val] + val_hist2_arr[:min_len_val]) / 2.0
        else:
            val_avg = np.array([], dtype=np.float64)
        loss_histories[subset_size] = {
            "train_model1": [float(x) for x in train_hist1],
            "val_model1": [float(x) for x in val_hist1],
            "train_model2": [float(x) for x in train_hist2],
            "val_model2": [float(x) for x in val_hist2],
            "train_mean": [float(x) for x in train_avg],
            "val_mean": [float(x) for x in val_avg],
        }

        processed_s1 = [preprocess_graph(g, nmax1, config.apply_dfs) for g in clone_graphs(graphs_s1)]
        processed_s2 = [preprocess_graph(g, nmax2, config.apply_dfs) for g in clone_graphs(graphs_s2)]

        nx_s1 = to_networkx(processed_s1)
        nx_s2 = to_networkx(processed_s2)

        gen1 = generate_graphs(model1, config.samples_per_model, device)
        gen2 = generate_graphs(model2, config.samples_per_model, device)

        stats_s1 = compute_stats_array(nx_s1)
        stats_s2 = compute_stats_array(nx_s2)
        stats_gen1 = compute_stats_array(gen1)
        stats_gen2 = compute_stats_array(gen2)

        logger.info(
            "Computing PGD metrics for N=%d (|S1|=%d, |S2|=%d, |Gen1|=%d, |Gen2|=%d)",
            subset_size,
            len(nx_s1),
            len(nx_s2),
            len(gen1),
            len(gen2),
        )
        pgd_rng = np.random.default_rng(config.seed + subset_size)
        logger.info("  • PGD S1 vs Gen1")
        pgd_s1_gen1 = compute_pgd_metric(nx_s1, gen1, config, pgd_rng)
        logger.info("  • PGD S2 vs Gen2")
        pgd_s2_gen2 = compute_pgd_metric(nx_s2, gen2, config, pgd_rng)
        logger.info("  • PGD S1 vs S2")
        pgd_s1_s2 = compute_pgd_metric(nx_s1, nx_s2, config, pgd_rng)
        logger.info("  • PGD Gen1 vs Gen2")
        pgd_gen1_gen2 = compute_pgd_metric(gen1, gen2, config, pgd_rng)

        visualize_graph_sets(
            [
                nx_s1,
                nx_s2,
                gen1,
                gen2,
            ],
            [
                "S1",
                "S2",
                "VGAE1",
                "VGAE2",
            ],
            subset_dir / "graph_comparison.png",
            samples_per_row=config.viz_samples,
            row_colors=("#f2a154", "#6c8af2", "#50a14f", "#d45d79"),
        )

        logger.info("Computing WL similarities for N=%d", subset_size)
        sims1 = wl_similarity_matrix(nx_s1, gen1, config.wl_iterations)
        sims2 = wl_similarity_matrix(nx_s2, gen2, config.wl_iterations)

        closest_sims = np.concatenate([closest_similarity(sims1), closest_similarity(sims2)])
        cross_sims = cross_similarity(gen1, gen2, config.wl_iterations)

        results[subset_size] = {
            "train_similarity": closest_sims,
            "cross_similarity": cross_sims,
            "pgd_train": np.array(
                list(pgd_s1_gen1["samples"]) + list(pgd_s2_gen2["samples"]),
                dtype=np.float64,
            ),
            "pgd_cross": np.array(
                list(pgd_gen1_gen2["samples"]) + list(pgd_s1_s2["samples"]),
                dtype=np.float64,
            ),
        }

        test_metrics1 = evaluate_on_test_set(
            model1,
            test_graphs,
            device,
            config.wl_iterations,
            config.samples_per_model,
            config.apply_dfs,
        )
        test_metrics2 = evaluate_on_test_set(
            model2,
            test_graphs,
            device,
            config.wl_iterations,
            config.samples_per_model,
            config.apply_dfs,
        )

        metrics[subset_size] = {
            "train_similarity_mean": float(np.mean(closest_sims)),
            "train_similarity_std": float(np.std(closest_sims)),
            "cross_similarity_mean": float(np.mean(cross_sims)),
            "cross_similarity_std": float(np.std(cross_sims)),
            "test_similarity_mean_model1": test_metrics1["mean_closest_similarity"],
            "test_similarity_std_model1": test_metrics1["std_closest_similarity"],
            "test_mmd_model1": test_metrics1["mmd_stats"],
            "test_similarity_mean_model2": test_metrics2["mean_closest_similarity"],
            "test_similarity_std_model2": test_metrics2["std_closest_similarity"],
            "test_mmd_model2": test_metrics2["mmd_stats"],
            "stats_comparisons": {
                "s1_vs_gen1": summarize_stats_relationship(stats_s1, stats_gen1),
                "s2_vs_gen2": summarize_stats_relationship(stats_s2, stats_gen2),
                "s1_vs_s2": summarize_stats_relationship(stats_s1, stats_s2),
                "gen1_vs_gen2": summarize_stats_relationship(stats_gen1, stats_gen2),
            },
            "pgd": {
                "s1_vs_gen1": _serialize_pgd_result(pgd_s1_gen1),
                "s2_vs_gen2": _serialize_pgd_result(pgd_s2_gen2),
                "s1_vs_s2": _serialize_pgd_result(pgd_s1_s2),
                "gen1_vs_gen2": _serialize_pgd_result(pgd_gen1_gen2),
                "train_mean": float(np.nanmean(results[subset_size]["pgd_train"]))
                if results[subset_size]["pgd_train"].size
                else float("nan"),
                "train_std": float(np.nanstd(results[subset_size]["pgd_train"]))
                if results[subset_size]["pgd_train"].size
                else float("nan"),
                "cross_mean": float(np.nanmean(results[subset_size]["pgd_cross"]))
                if results[subset_size]["pgd_cross"].size
                else float("nan"),
                "cross_std": float(np.nanstd(results[subset_size]["pgd_cross"]))
                if results[subset_size]["pgd_cross"].size
                else float("nan"),
            },
        }

        np.save(subset_dir / "train_similarity.npy", closest_sims)
        np.save(subset_dir / "cross_similarity.npy", cross_sims)
        np.save(subset_dir / "pgd_train.npy", results[subset_size]["pgd_train"])
        np.save(subset_dir / "pgd_cross.npy", results[subset_size]["pgd_cross"])

        with open(subset_dir / "metrics.json", "w", encoding="utf8") as handle:
            json.dump(metrics[subset_size], handle, indent=2)

    plot_histograms(results, work_root / "memorization_histograms.png")
    plot_pgd_histograms(results, work_root / "pgd_histograms.png")
    plot_loss_curves(loss_histories, work_root / "loss_curves.png")

    logger.info("Saved histogram plot to %s", work_root / "memorization_histograms.png")
    logger.info("Saved PGD histogram plot to %s", work_root / "pgd_histograms.png")
    logger.info("Saved loss curve plot to %s", work_root / "loss_curves.png")

    with open(work_root / "summary.json", "w", encoding="utf8") as handle:
        json.dump(metrics, handle, indent=2)
    logger.info("Wrote summary metrics to %s", work_root / "summary.json")

    with open(work_root / "loss_histories.json", "w", encoding="utf8") as handle:
        json.dump(loss_histories, handle, indent=2)
    logger.info("Stored loss histories to %s", work_root / "loss_histories.json")

    with open(work_root / "split_stats.json", "w", encoding="utf8") as handle:
        json.dump(stats, handle, indent=2)
    logger.info("Persisted split statistics to %s", work_root / "split_stats.json")


def _load_feature_hom(config: StudyConfig) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(config.csv_path)
    if "actual_feature_hom" not in df.columns:
        raise ValueError("CSV missing 'actual_feature_hom'")
    df_sorted = df.sort_values("graph_idx")
    return df_sorted["actual_feature_hom"].to_numpy()


def parse_args() -> StudyConfig:
    parser = argparse.ArgumentParser(description="Memorization vs generalization study for VGAEs")
    parser.add_argument("--dataset-path", type=str, default="data/featurehomophily0.4_graphs.pkl")
    parser.add_argument("--csv-path", type=str, default="data/featurehomophily0.4_log.csv")
    parser.add_argument("--output-dir", type=str, default="experiments/memorization_outputs")
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 500, 1000, 2500, 4500],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim-encoder", type=int, default=32)
    parser.add_argument("--hidden-dim-decoder", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-layers-encoder", type=int, default=2)
    parser.add_argument("--n-layers-decoder", type=int, default=3)
    parser.add_argument("--samples-per-model", type=int, default=100)
    parser.add_argument("--wl-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--no-dfs", action="store_true", help="Disable DFS reordering")
    parser.add_argument("--viz-samples", type=int, default=4, help="Number of graph samples to plot per row")
    parser.add_argument("--decoder-node-hidden", type=int, default=None, help="Hidden dimension for decoder node embeddings")
    parser.add_argument("--decoder-dropout", type=float, default=0.1, help="Dropout applied inside the decoder")
    parser.add_argument("--decoder-use-gumbel", action="store_true", help="Use Gumbel sampling in the decoder (legacy behaviour)")
    parser.add_argument("--decoder-gumbel-tau", type=float, default=1.0, help="Gumbel temperature if enabled")
    parser.add_argument(
        "--pgd-variant",
        type=str,
        choices=["jsd", "informedness"],
        default="jsd",
        help="Underlying probability metric approximated by the standalone PGD estimator.",
    )
    parser.add_argument(
        "--pgd-bootstrap",
        type=int,
        default=16,
        help="Number of bootstrap resamples when estimating PGD distributions (0 to disable).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing VGAE checkpoints if available instead of retraining.",
    )

    args = parser.parse_args()

    return StudyConfig(
        dataset_path=args.dataset_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        subset_sizes=args.subset_sizes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        hidden_dim_encoder=args.hidden_dim_encoder,
        hidden_dim_decoder=args.hidden_dim_decoder,
        latent_dim=args.latent_dim,
        n_layers_encoder=args.n_layers_encoder,
        n_layers_decoder=args.n_layers_decoder,
        samples_per_model=args.samples_per_model,
        wl_iterations=args.wl_iterations,
        seed=args.seed,
        device=args.device,
        test_ratio=args.test_ratio,
        bins=args.bins,
        force_resplit=args.force_resplit,
        apply_dfs=not args.no_dfs,
        viz_samples=args.viz_samples,
        decoder_node_hidden_dim=args.decoder_node_hidden,
        decoder_dropout=args.decoder_dropout,
        decoder_use_gumbel=args.decoder_use_gumbel,
        decoder_gumbel_tau=args.decoder_gumbel_tau,
        pgd_variant=args.pgd_variant,
        pgd_bootstrap=args.pgd_bootstrap,
        resume=args.resume,
    )


def main() -> None:
    _setup_logging()
    config = parse_args()
    run_study(config)


if __name__ == "__main__":
    main()