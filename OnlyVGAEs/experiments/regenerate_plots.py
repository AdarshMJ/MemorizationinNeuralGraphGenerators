import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoencoder import VariationalAutoEncoder
from experiments.run_memorization_study import (
    _load_feature_hom,
    _load_graphs,
    _safe_torch_load,
    _setup_logging,
    clone_graphs,
    determine_n_max_nodes,
    generate_graphs,
    plot_histograms,
    preprocess_graph,
    select_quantile_indices,
    set_seed,
    to_networkx,
    visualize_graph_sets,
)


logger = logging.getLogger(__name__)


def _infer_subset_sizes(output_dir: Path, explicit: Optional[Sequence[int]] = None) -> List[int]:
    if explicit:
        return sorted(explicit)
    detected: List[int] = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith("N_"):
            try:
                detected.append(int(child.name.split("_", maxsplit=1)[1]))
            except (IndexError, ValueError):
                continue
    if not detected:
        raise ValueError(f"No N_* subset folders found under {output_dir}")
    return sorted(detected)


def _load_split_indices(split_dir: Path) -> Dict[str, Sequence[int]]:
    required = {"s1_indices": "s1_indices.pt", "s2_indices": "s2_indices.pt", "test_indices": "test_indices.pt"}
    loaded = {}
    for key, filename in required.items():
        tensor = _safe_torch_load(split_dir / filename, weights_only=True)
        loaded[key] = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)
    return loaded


def _load_results_arrays(output_dir: Path, subset_sizes: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for subset_size in subset_sizes:
        subset_dir = output_dir / f"N_{subset_size}"
        train_path = subset_dir / "train_similarity.npy"
        cross_path = subset_dir / "cross_similarity.npy"
        if not train_path.is_file() or not cross_path.is_file():
            raise FileNotFoundError(f"Missing similarity arrays for N={subset_size} in {subset_dir}")
        results[subset_size] = {
            "train_similarity": np.load(train_path),
            "cross_similarity": np.load(cross_path),
        }
    return results


def _construct_model(
    graphs: Sequence,
    args: argparse.Namespace,
    device: torch.device,
    n_layers_enc: int,
    n_layers_dec: int,
) -> tuple[VariationalAutoEncoder, int]:
    n_max_nodes = determine_n_max_nodes(graphs)
    feature_dim = graphs[0].x.size(1)
    model = VariationalAutoEncoder(
        input_dim=feature_dim,
        hidden_dim_enc=args.hidden_dim_encoder,
        hidden_dim_dec=args.hidden_dim_decoder,
        latent_dim=args.latent_dim,
        n_layers_enc=n_layers_enc,
        n_layers_dec=n_layers_dec,
        n_max_nodes=n_max_nodes,
        decoder_node_hidden_dim=args.decoder_node_hidden,
        decoder_dropout=args.decoder_dropout,
        decoder_use_gumbel=args.decoder_use_gumbel,
        decoder_gumbel_tau=args.decoder_gumbel_tau,
    ).to(device)
    return model, n_max_nodes


def _load_trained_model(
    checkpoint_path: Path,
    graphs: Sequence,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VariationalAutoEncoder, int]:
    model, n_max_nodes = _construct_model(
        graphs,
        args,
        device,
        args.n_layers_encoder,
        args.n_layers_decoder,
    )
    checkpoint = _safe_torch_load(checkpoint_path, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded checkpoint %s", checkpoint_path)
    return model, n_max_nodes


def regenerate_histograms(
    results: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    bins: int,
    filename: str,
) -> None:
    hist_path = output_dir / filename
    plot_histograms(results, hist_path, bins=bins)
    logger.info("Regenerated histogram at %s", hist_path)


def regenerate_graph_grids(
    subset_sizes: Sequence[int],
    graphs: Sequence,
    feature_hom: np.ndarray,
    split_indices: Dict[str, Sequence[int]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    for subset_size in subset_sizes:
        subset_dir = output_dir / f"N_{subset_size}"
        if not subset_dir.exists():
            logger.warning("Skipping N=%d because %s is missing", subset_size, subset_dir)
            continue

        logger.info("Generating graph comparisons for N=%d", subset_size)

        s1_subset_idx = select_quantile_indices(split_indices["s1_indices"], feature_hom, subset_size)
        s2_subset_idx = select_quantile_indices(split_indices["s2_indices"], feature_hom, subset_size)

        graphs_s1 = [graphs[idx] for idx in s1_subset_idx]
        graphs_s2 = [graphs[idx] for idx in s2_subset_idx]

        model1, nmax1 = _load_trained_model(subset_dir / "vgae1" / "best_model.pth.tar", graphs_s1, args, device)
        model2, nmax2 = _load_trained_model(subset_dir / "vgae2" / "best_model.pth.tar", graphs_s2, args, device)

        processed_s1 = [preprocess_graph(g, nmax1, args.apply_dfs) for g in clone_graphs(graphs_s1)]
        processed_s2 = [preprocess_graph(g, nmax2, args.apply_dfs) for g in clone_graphs(graphs_s2)]

        nx_s1 = to_networkx(processed_s1)
        nx_s2 = to_networkx(processed_s2)

        gen1 = generate_graphs(model1, args.samples_per_model, device)
        gen2 = generate_graphs(model2, args.samples_per_model, device)

        visualize_graph_sets(
            [nx_s1, nx_s2, gen1, gen2],
            ["S1", "S2", "VGAE1", "VGAE2"],
            subset_dir / "graph_comparison.png",
            samples_per_row=args.viz_samples,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate memorization study plots from saved outputs without retraining",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory with memorization study outputs")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the original dataset pickle/PT file")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV with feature homophily column")
    parser.add_argument("--subset-sizes", type=int, nargs="*", help="Specific subset sizes to regenerate")
    parser.add_argument("--bins", type=int, default=30, help="Bins to use for histogram regeneration")
    parser.add_argument("--hist-filename", type=str, default="memorization_histograms.png", help="Name for the regenerated histogram file")
    parser.add_argument("--samples-per-model", type=int, default=64, help="Number of samples to draw per VGAE when visualizing graphs")
    parser.add_argument("--viz-samples", type=int, default=4, help="Number of graph panels per row in the visualization grid")
    parser.add_argument("--hidden-dim-encoder", type=int, default=32)
    parser.add_argument("--hidden-dim-decoder", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-layers-encoder", type=int, default=2)
    parser.add_argument("--n-layers-decoder", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-dfs", action="store_true", help="Disable DFS reordering when reconstructing graphs")
    parser.add_argument("--skip-histogram", action="store_true", help="Skip histogram regeneration")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph comparison visualizations")
    parser.add_argument("--decoder-node-hidden", type=int, default=None, help="Hidden dimension for decoder node embeddings")
    parser.add_argument("--decoder-dropout", type=float, default=0.0, help="Dropout probability inside the decoder")
    parser.add_argument("--decoder-use-gumbel", action="store_true", help="Enable legacy Gumbel decoding")
    parser.add_argument("--decoder-gumbel-tau", type=float, default=1.0, help="Gumbel temperature if enabled")

    args = parser.parse_args()
    args.apply_dfs = not args.no_dfs
    return args


def main() -> None:
    _setup_logging()
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected split directory at {split_dir}")

    subset_sizes = _infer_subset_sizes(output_dir, args.subset_sizes)
    logger.info("Found subset sizes: %s", ", ".join(str(s) for s in subset_sizes))

    graphs = _load_graphs(args.dataset_path)
    feature_hom = _load_feature_hom(args)
    split_indices = _load_split_indices(split_dir)

    if not args.skip_histogram:
        results = _load_results_arrays(output_dir, subset_sizes)
        regenerate_histograms(results, output_dir, args.bins, args.hist_filename)
    else:
        logger.info("Skipping histogram regeneration as requested")

    if not args.skip_graphs:
        regenerate_graph_grids(subset_sizes, graphs, feature_hom, split_indices, output_dir, args)
    else:
        logger.info("Skipping graph visualizations as requested")


if __name__ == "__main__":
    main()
