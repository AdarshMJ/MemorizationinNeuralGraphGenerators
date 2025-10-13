import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass
class SplitResult:
    s1_indices: List[int]
    s2_indices: List[int]
    test_indices: List[int]


def _load_graphs(dataset_path: str) -> List[Data]:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.endswith(".pt"):
        graphs: List[Data] = torch.load(dataset_path)
    else:
        import pickle

        with open(dataset_path, "rb") as handle:
            graphs = pickle.load(handle)
    if not isinstance(graphs, list) or not all(isinstance(g, Data) for g in graphs):
        raise ValueError("Dataset is expected to be a list of torch_geometric.data.Data")
    return graphs


def _load_metadata(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV metadata not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"graph_idx", "actual_feature_hom"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def _compute_histogram_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Cannot compute histogram for empty array")
    if n_bins <= 0:
        raise ValueError("Number of bins must be positive")
    return np.linspace(values.min(), values.max(), num=n_bins + 1)


def _stratified_split(
    indices: Sequence[int],
    values: Sequence[float],
    test_ratio: float,
    n_bins: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    values_arr = np.asarray(values)
    bins = _compute_histogram_bins(values_arr, n_bins)
    bin_ids = np.digitize(values_arr, bins[:-1], right=False)

    s1, s2, test = [], [], []
    for b in range(1, n_bins + 1):
        bucket_indices = [idx for idx, bin_id in zip(indices, bin_ids) if bin_id == b]
        if not bucket_indices:
            continue
        rng.shuffle(bucket_indices)
        n_total = len(bucket_indices)
        n_test = math.ceil(test_ratio * n_total)
        n_half = (n_total - n_test) // 2

        test.extend(bucket_indices[:n_test])
        remaining = bucket_indices[n_test:]
        s1.extend(remaining[:n_half])
        s2.extend(remaining[n_half:])

    return s1, s2, test


def _validate_counts(
    s1: Sequence[int],
    s2: Sequence[int],
    test: Sequence[int],
    total: int,
) -> None:
    seen = set(s1) | set(s2) | set(test)
    if len(seen) != len(s1) + len(s2) + len(test):
        raise ValueError("Splits overlap; ensure disjoint subsets")
    if len(seen) != total:
        raise ValueError(
            f"Splits do not cover all graphs: seen {len(seen)} vs total {total}"
        )


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values))
    return {
        "count": int(arr.size),
        "mean": float(arr.mean() if arr.size else float("nan")),
        "std": float(arr.std(ddof=1) if arr.size > 1 else float("nan")),
        "min": float(arr.min() if arr.size else float("nan")),
        "max": float(arr.max() if arr.size else float("nan")),
    }


def perform_split(
    dataset_path: str,
    csv_path: str,
    test_ratio: float,
    n_bins: int,
    seed: int,
) -> Tuple[List[Data], SplitResult, Dict[str, Dict[str, float]]]:
    graphs = _load_graphs(dataset_path)
    df = _load_metadata(csv_path)
    if len(graphs) != len(df):
        raise ValueError(
            f"Dataset size mismatch: {len(graphs)} graphs, {len(df)} CSV rows"
        )

    df_sorted = df.sort_values("graph_idx")
    feature_hom = df_sorted["actual_feature_hom"].to_numpy()
    all_indices = list(range(len(graphs)))

    s1_idx, s2_idx, test_idx = _stratified_split(
        indices=all_indices,
        values=feature_hom,
        test_ratio=test_ratio,
        n_bins=n_bins,
        seed=seed,
    )

    _validate_counts(s1_idx, s2_idx, test_idx, len(graphs))

    stats = {
        "S1": _summary_stats(feature_hom[s1_idx]),
        "S2": _summary_stats(feature_hom[s2_idx]),
        "Test": _summary_stats(feature_hom[test_idx]),
    }

    return graphs, SplitResult(s1_idx, s2_idx, test_idx), stats


def export_split(
    split: SplitResult,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(split.s1_indices, os.path.join(output_dir, "s1_indices.pt"))
    torch.save(split.s2_indices, os.path.join(output_dir, "s2_indices.pt"))
    torch.save(split.test_indices, os.path.join(output_dir, "test_indices.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify graphs by feature homophily into S1/S2/test splits"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/featurehomophily0.6_graphs.pkl",
        help="Path to pickled list of torch_geometric.data.Data",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/featurehomophily0.6_log.csv",
        help="CSV file with actual_feature_hom per graph",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of graphs reserved for held-out test set",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for stratification histograms",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/memorization_split",
        help="Directory to store split indices",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graphs, split, stats = perform_split(
        dataset_path=args.dataset_path,
        csv_path=args.csv_path,
        test_ratio=args.test_ratio,
        n_bins=args.bins,
        seed=args.seed,
    )

    export_split(split, args.output_dir)

    print("Saved split indices to", args.output_dir)
    for name, stat in stats.items():
        print(f"{name}: count={stat['count']}, mean={stat['mean']:.4f}, std={stat['std']:.4f}, min={stat['min']:.4f}, max={stat['max']:.4f}")
    print(f"Graphs available: {len(graphs)}")


if __name__ == "__main__":
    main()