#!/usr/bin/env python3
"""Standalone implementation of the Standard PolyGraph Discrepancy (PGD).

This script re-creates the Standard PGD metric without depending on the
original PolyGraph repository. It computes several graph descriptors,
trains a lightweight logistic classifier to distinguish reference and
generated graphs, and reports a lower bound on either the Jensen-Shannon
(JS) distance or the total variation distance (via informedness).

Usage example
-------------
```
python standard_pgd_standalone.py \
    --reference data/reference_graphs.pkl \
    --generated data/generated_graphs.pkl \
    --output results/pgd.json
```

Only ``numpy`` and ``networkx`` are required; the classifier and graph
descriptors are implemented locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


def _read_gpickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)

# --- Logistic regression -------------------------------------------------------------


class LogisticBinaryClassifier:
    """Mini-batch logistic regression with optional L2 regularisation."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 800,
        batch_size: int = 64,
        l2: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state
        self._weights: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # use a stable formulation to avoid overflow in exp for large |x|
        out = np.empty_like(x, dtype=np.float64)
        mask = x >= 0
        out[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        exp_x = np.exp(x[~mask])
        out[~mask] = exp_x / (1.0 + exp_x)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBinaryClassifier":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        X_ext = np.concatenate([X, np.ones((n_samples, 1))], axis=1)
        weights = np.zeros(n_features + 1, dtype=np.float64)

        for _ in range(self.epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_batch = X_ext[batch_idx]
                y_batch = y[batch_idx]

                logits = X_batch @ weights
                probs = self._sigmoid(logits)
                error = probs - y_batch

                grad = (X_batch.T @ error) / len(batch_idx)
                grad[:-1] += self.l2 * weights[:-1]
                weights -= self.learning_rate * grad

        self._weights = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_ext = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        logits = X_ext @ self._weights
        probs_pos = self._sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)


# --- Helper utilities ----------------------------------------------------------------


def _scores_to_jsd(ref_scores: np.ndarray, gen_scores: np.ndarray, eps: float = 1e-10) -> float:
    divergence = 0.5 * (
        np.log2(ref_scores + eps).mean()
        + np.log2(1.0 - gen_scores + eps).mean()
        + 2.0
    )
    return float(np.sqrt(np.clip(divergence, 0.0, 1.0)))


def _scores_to_informedness_and_threshold(
    ref_scores: np.ndarray, gen_scores: np.ndarray
) -> Tuple[float, float]:
    thresholds = np.unique(np.concatenate([ref_scores, gen_scores]))
    if thresholds.size == 0:
        return 0.0, 0.5
    thresholds = np.concatenate([thresholds, thresholds + 1e-6])
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in thresholds:
        ref_pred = (ref_scores >= threshold).astype(float)
        gen_pred = (gen_scores >= threshold).astype(float)
        tpr = ref_pred.mean()
        fpr = gen_pred.mean()
        informedness = tpr - fpr
        if informedness > best_score:
            best_score = informedness
            best_threshold = threshold
    return float(best_score), float(best_threshold)


def _scores_and_threshold_to_informedness(
    ref_scores: np.ndarray, gen_scores: np.ndarray, threshold: float
) -> float:
    ref_pred = (ref_scores >= threshold).astype(float)
    gen_pred = (gen_scores >= threshold).astype(float)
    return float(ref_pred.mean() - gen_pred.mean())


def _standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _stratified_k_fold_indices(y: np.ndarray, k: int, rng: np.random.Generator):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    for fold in range(k):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        mask = np.ones(len(y), dtype=bool)
        mask[val_idx] = False
        train_idx = np.where(mask)[0]
        yield train_idx, val_idx


def _split_train_test(
    features_ref: np.ndarray,
    features_gen: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_ref = len(features_ref)
    n_gen = len(features_gen)
    ref_indices = rng.permutation(n_ref)
    gen_indices = rng.permutation(n_gen)
    ref_split = n_ref // 2
    gen_split = n_gen // 2
    ref_train = features_ref[ref_indices[:ref_split]]
    ref_test = features_ref[ref_indices[ref_split:]]
    gen_train = features_gen[gen_indices[:gen_split]]
    gen_test = features_gen[gen_indices[gen_split:]]
    return ref_train, ref_test, gen_train, gen_test


def _classifier_metric(
    features_ref: np.ndarray,
    features_gen: np.ndarray,
    variant: str,
    rng_seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    ref_train, ref_test, gen_train, gen_test = _split_train_test(features_ref, features_gen, rng)

    if ref_train.size == 0 or gen_train.size == 0:
        return 0.0, 0.0

    X_train = np.concatenate([ref_train, gen_train], axis=0)
    y_train = np.concatenate([np.ones(len(ref_train)), np.zeros(len(gen_train))])

    if np.allclose(X_train, X_train[0]):
        train_metric = 0.0
        train_threshold = 0.5
        classifier = None
        mean = np.zeros(X_train.shape[1])
        std = np.ones(X_train.shape[1])
    else:
        cv_scores = []
        cv_rng = np.random.default_rng(rng_seed + 1234)
        for fold_idx, (train_idx, val_idx) in enumerate(_stratified_k_fold_indices(y_train, 4, cv_rng)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            X_fold_train, X_fold_val, _, _ = _standardize(X_fold_train, X_fold_val)
            clf = LogisticBinaryClassifier(random_state=rng_seed + fold_idx)
            clf.fit(X_fold_train, y_fold_train)
            val_scores = clf.predict_proba(X_fold_val)[:, 1]
            ref_scores = val_scores[y_fold_val == 1]
            gen_scores = val_scores[y_fold_val == 0]
            if variant == "jsd":
                cv_scores.append(_scores_to_jsd(ref_scores, gen_scores))
            else:
                score, _ = _scores_to_informedness_and_threshold(ref_scores, gen_scores)
                cv_scores.append(score)
        train_metric = float(np.mean(cv_scores))

        X_train_norm, _, mean, std = _standardize(X_train, X_train)
        classifier = LogisticBinaryClassifier(random_state=rng_seed + 999)
        classifier.fit(X_train_norm, y_train)
        train_scores = classifier.predict_proba(X_train_norm)[:, 1]
        ref_scores = train_scores[y_train == 1]
        gen_scores = train_scores[y_train == 0]
        if variant == "jsd":
            train_threshold = 0.5
        else:
            _, train_threshold = _scores_to_informedness_and_threshold(ref_scores, gen_scores)

    X_test = np.concatenate([ref_test, gen_test], axis=0)
    y_test = np.concatenate([np.ones(len(ref_test)), np.zeros(len(gen_test))])

    if classifier is None:
        test_metric = 0.0
    else:
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        X_test_norm = (X_test - mean) / std
        test_scores = classifier.predict_proba(X_test_norm)[:, 1]
        ref_scores = test_scores[y_test == 1]
        gen_scores = test_scores[y_test == 0]
        if variant == "jsd":
            test_metric = _scores_to_jsd(ref_scores, gen_scores)
        else:
            test_metric = _scores_and_threshold_to_informedness(ref_scores, gen_scores, train_threshold)

    return train_metric, float(test_metric)


# --- Graph descriptors ---------------------------------------------------------------


class GraphDescriptor:
    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        raise NotImplementedError


class OrbitHashDescriptor(GraphDescriptor):
    """Approximate graphlet orbit counts using hashed samples."""

    def __init__(
        self,
        graphlet_size: int,
        num_samples: int = 2048,
        num_buckets: int = 128,
        random_state: int = 0,
    ) -> None:
        self.graphlet_size = graphlet_size
        self.num_samples = num_samples
        self.num_buckets = num_buckets
        self.random_state = random_state

    def _hash_graphlet(self, subgraph: nx.Graph) -> int:
        if hasattr(nx, "to_graph6_bytes"):
            label = nx.to_graph6_bytes(subgraph, nodes=list(subgraph.nodes()), header=False)
        else:  # pragma: no cover - very old NetworkX versions
            label = nx.convert.to_graph6_bytes(subgraph, nodes=list(subgraph.nodes()), header=False)
        digest = hashlib.sha256(label).digest()
        return int.from_bytes(digest[:4], "big") % self.num_buckets

    def _all_combinations(self, n: int) -> int:
        return math.comb(n, self.graphlet_size)

    def _graph_to_vector(self, graph: nx.Graph, seed: int) -> np.ndarray:
        n_nodes = graph.number_of_nodes()
        vector = np.zeros(self.num_buckets, dtype=np.float64)
        if n_nodes < self.graphlet_size:
            return vector

        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        total_combinations = self._all_combinations(len(nodes))

        if total_combinations <= self.num_samples:
            from itertools import combinations

            subsets = combinations(nodes, self.graphlet_size)
        else:
            subsets = (
                tuple(rng.choice(nodes, size=self.graphlet_size, replace=False))
                for _ in range(self.num_samples)
            )

        count = 0
        for subset in subsets:
            subgraph = graph.subgraph(subset).copy()
            bucket = self._hash_graphlet(subgraph)
            vector[bucket] += 1.0
            count += 1
            if count >= self.num_samples:
                break

        if vector.sum() > 0:
            vector /= vector.sum()
        return vector

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        features = [self._graph_to_vector(graph, self.random_state + idx) for idx, graph in enumerate(graphs)]
        return np.asarray(features, dtype=np.float64)


class HistogramDescriptor(GraphDescriptor):
    def __init__(self, bins: int, range_: Tuple[float, float]) -> None:
        self.bins = bins
        self.range = range_

    def _histogram(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(self.bins, dtype=np.float64)
        counts, _ = np.histogram(values, bins=self.bins, range=self.range)
        if counts.sum() > 0:
            counts = counts.astype(np.float64) / counts.sum()
        else:
            counts = counts.astype(np.float64)
        return counts


class ClusteringHistogramDescriptor(HistogramDescriptor):
    def __init__(self, bins: int = 100) -> None:
        super().__init__(bins=bins, range_=(0.0, 1.0))

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        features = []
        for graph in graphs:
            coeffs = np.array(list(nx.clustering(graph).values()), dtype=np.float64)
            features.append(self._histogram(coeffs))
        return np.asarray(features, dtype=np.float64)


class DegreeHistogramDescriptor(HistogramDescriptor):
    def __init__(self, bins: int = 64) -> None:
        super().__init__(bins=bins, range_=(0.0, 1.0))

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        features = []
        for graph in graphs:
            n = max(1, graph.number_of_nodes() - 1)
            degrees = np.array([deg / n for _, deg in graph.degree()], dtype=np.float64)
            features.append(self._histogram(degrees))
        return np.asarray(features, dtype=np.float64)


class SpectralHistogramDescriptor(HistogramDescriptor):
    def __init__(self, bins: int = 64) -> None:
        super().__init__(bins=bins, range_=(0.0, 2.0))

    def _normalized_laplacian_eigs(self, graph: nx.Graph) -> np.ndarray:
        n = graph.number_of_nodes()
        if n == 0:
            return np.zeros(1, dtype=np.float64)
        A = nx.to_numpy_array(graph, dtype=np.float64)
        deg = A.sum(axis=1)
        inv_sqrt_deg = np.zeros_like(deg)
        mask = deg > 0
        inv_sqrt_deg[mask] = 1.0 / np.sqrt(deg[mask])
        D_inv = np.diag(inv_sqrt_deg)
        L = np.eye(n) - D_inv @ A @ D_inv
        try:
            eigs = np.linalg.eigvalsh(L)
        except np.linalg.LinAlgError:
            eigs = np.real(np.linalg.eigvals(L))
        eigs = np.clip(eigs, 0.0, 2.0)
        return eigs.astype(np.float64)

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        features = []
        for graph in graphs:
            eigs = self._normalized_laplacian_eigs(graph)
            features.append(self._histogram(eigs))
        return np.asarray(features, dtype=np.float64)


class RandomGINDescriptor(GraphDescriptor):
    def __init__(self, embed_dim: int = 64, layers: int = 3, random_state: int = 0) -> None:
        self.embed_dim = embed_dim
        self.layers = layers
        self.random_state = random_state

    def _graph_embedding(self, graph: nx.Graph, seed: int) -> np.ndarray:
        n = graph.number_of_nodes()
        if n == 0:
            return np.zeros(self.embed_dim, dtype=np.float64)
        rng = np.random.default_rng(seed)
        X = rng.normal(scale=1.0, size=(n, self.embed_dim))
        A = nx.to_numpy_array(graph, dtype=np.float64)
        for _ in range(self.layers):
            eps = rng.normal()
            agg = A @ X
            H = (1.0 + eps) * X + agg
            W = rng.normal(scale=1.0 / math.sqrt(self.embed_dim), size=(self.embed_dim, self.embed_dim))
            X = np.maximum(0.0, H @ W)
        return X.sum(axis=0)

    def transform(self, graphs: Sequence[nx.Graph]) -> np.ndarray:
        features = [self._graph_embedding(graph, self.random_state + idx) for idx, graph in enumerate(graphs)]
        return np.asarray(features, dtype=np.float64)


# --- Metric interval helper -----------------------------------------------------------


@dataclass
class MetricInterval:
    mean: float
    std: float
    low: Optional[float] = None
    high: Optional[float] = None
    coverage: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "low": self.low,
            "high": self.high,
            "coverage": self.coverage,
        }

    @classmethod
    def from_samples(cls, samples: np.ndarray, coverage: Optional[float] = None) -> "MetricInterval":
        if samples.size == 0:
            return cls(mean=float("nan"), std=float("nan"), low=None, high=None, coverage=coverage)
        if coverage is not None and samples.size > 1:
            low = float(np.quantile(samples, (1 - coverage) / 2))
            high = float(np.quantile(samples, coverage + (1 - coverage) / 2))
        else:
            low = high = None
        return cls(mean=float(np.mean(samples)), std=float(np.std(samples)), low=low, high=high, coverage=coverage)


# --- Standard PGD implementation ------------------------------------------------------


class StandardPGD:
    def __init__(
        self,
        reference_graphs: Sequence[nx.Graph],
        *,
        variant: str = "jsd",
        random_state: int = 0,
    ) -> None:
        if variant not in {"jsd", "informedness"}:
            raise ValueError("variant must be either 'jsd' or 'informedness'")
        self.variant = variant
        self.random_state = random_state
        self.descriptors = {
            "orbit4": OrbitHashDescriptor(graphlet_size=4, random_state=random_state + 10),
            "orbit5": OrbitHashDescriptor(graphlet_size=5, random_state=random_state + 20),
            "clustering": ClusteringHistogramDescriptor(bins=100),
            "degree": DegreeHistogramDescriptor(bins=64),
            "spectral": SpectralHistogramDescriptor(bins=64),
            "gin": RandomGINDescriptor(embed_dim=64, layers=3, random_state=random_state + 30),
        }
        self.reference_features = {
            name: descriptor.transform(reference_graphs)
            for name, descriptor in self.descriptors.items()
        }

    def compute(self, generated_graphs: Sequence[nx.Graph]) -> dict:
        rng = np.random.default_rng(self.random_state + 42)
        metrics = {}
        subscores = {}
        for name, descriptor in self.descriptors.items():
            ref_feat = self.reference_features[name]
            gen_feat = descriptor.transform(generated_graphs)
            train_metric, test_metric = _classifier_metric(
                ref_feat,
                gen_feat,
                self.variant,
                rng.integers(0, 1_000_000),
            )
            metrics[name] = (train_metric, test_metric)
            subscores[name] = test_metric

        optimal_descriptor = max(metrics, key=lambda key: metrics[key][0])
        aggregate_metric = metrics[optimal_descriptor][1]
        return {
            "pgd": aggregate_metric,
            "pgd_descriptor": optimal_descriptor,
            "subscores": subscores,
        }


class StandardPGDInterval:
    def __init__(
        self,
        reference_graphs: Sequence[nx.Graph],
        *,
        subsample_size: int,
        num_samples: int = 10,
        variant: str = "jsd",
        random_state: int = 0,
    ) -> None:
        if subsample_size < 2:
            raise ValueError("subsample_size must be at least 2")
        self.reference_graphs = list(reference_graphs)
        self.subsample_size = subsample_size
        self.num_samples = num_samples
        self.variant = variant
        self.random_state = random_state

    def _sample(self, graphs: Sequence[nx.Graph], rng: np.random.Generator) -> List[nx.Graph]:
        if len(graphs) <= self.subsample_size:
            return list(graphs)
        indices = rng.choice(len(graphs), size=self.subsample_size, replace=False)
        return [graphs[i] for i in indices]

    def compute(self, generated_graphs: Sequence[nx.Graph]) -> dict:
        descriptor_names = ["orbit4", "orbit5", "clustering", "degree", "spectral", "gin"]
        rng = np.random.default_rng(self.random_state + 123)
        samples = []
        descriptor_counts = {name: 0 for name in descriptor_names}
        subscore_samples = {name: [] for name in descriptor_names}
        for idx in range(self.num_samples):
            ref_subset = self._sample(self.reference_graphs, rng)
            gen_subset = self._sample(generated_graphs, rng)
            pgd = StandardPGD(ref_subset, variant=self.variant, random_state=self.random_state + idx)
            result = pgd.compute(gen_subset)
            samples.append(result["pgd"])
            descriptor_counts[result["pgd_descriptor"]] += 1
            for name, score in result["subscores"].items():
                subscore_samples[name].append(score)

        samples_arr = np.array(samples, dtype=np.float64)
        pgd_interval = MetricInterval.from_samples(samples_arr)
        descriptor_freq = {key: count / self.num_samples for key, count in descriptor_counts.items()}
        subscores = {
            name: MetricInterval.from_samples(np.array(values, dtype=np.float64))
            for name, values in subscore_samples.items()
        }
        return {
            "pgd": pgd_interval,
            "pgd_descriptor": descriptor_freq,
            "subscores": subscores,
        }


# --- Graph loading utilities ---------------------------------------------------------


def _read_json_graph(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())
    return nx.readwrite.json_graph.node_link_graph(data)


_SUPPORTED_FILE_READERS = {
    ".gpickle": _read_gpickle,
    ".pkl": _read_gpickle,
    ".pickle": _read_gpickle,
    ".gml": lambda path: nx.read_gml(path, destringizer=int),
    ".graphml": nx.read_graphml,
    ".json": _read_json_graph,
    ".adjlist": nx.read_adjlist,
    ".edgelist": nx.read_edgelist,
    ".txt": nx.read_edgelist,
}


class GraphLoadError(RuntimeError):
    pass


def _load_graph_file(path: Path, fmt_hint: Optional[str] = None) -> nx.Graph:
    suffix = fmt_hint or path.suffix.lower()
    if suffix not in _SUPPORTED_FILE_READERS:
        raise GraphLoadError(f"Unsupported graph format for file '{path}'.")
    reader = _SUPPORTED_FILE_READERS[suffix]
    try:
        return reader(path)
    except Exception as exc:  # pragma: no cover
        raise GraphLoadError(f"Failed to load graph from '{path}' using format '{suffix}': {exc}") from exc


def _load_graphs_from_pickle(path: Path) -> List[nx.Graph]:
    graphs = _read_gpickle(path)
    if isinstance(graphs, dict) and "graphs" in graphs:
        graphs = graphs["graphs"]
    if not isinstance(graphs, Iterable):
        raise GraphLoadError(f"Pickle file '{path}' must contain an iterable of networkx graphs.")
    result = []
    for idx, graph in enumerate(graphs):
        if not isinstance(graph, nx.Graph):
            raise GraphLoadError(
                f"Object at index {idx} in '{path}' is not a networkx graph (type={type(graph)})."
            )
        result.append(graph)
    return result


def _load_graph_collection(source: Path, fmt: Optional[str]) -> List[nx.Graph]:
    if not source.exists():
        raise GraphLoadError(f"Path '{source}' does not exist.")
    if source.is_file():
        suffix = fmt or source.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            return _load_graphs_from_pickle(source)
        return [_load_graph_file(source, fmt_hint=fmt)]

    graphs: List[nx.Graph] = []
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            continue
        suffix = fmt or entry.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            graphs.extend(_load_graphs_from_pickle(entry))
        elif suffix in _SUPPORTED_FILE_READERS:
            graphs.append(_load_graph_file(entry, fmt_hint=fmt))
    if not graphs:
        raise GraphLoadError(f"No supported graph files found in directory '{source}'.")
    return graphs


def _truncate_to_match(
    ref_graphs: List[nx.Graph],
    gen_graphs: List[nx.Graph],
    strategy: str,
    rng_seed: int,
) -> Tuple[List[nx.Graph], List[nx.Graph]]:
    if len(ref_graphs) == len(gen_graphs):
        return ref_graphs, gen_graphs
    if strategy == "raise":
        raise ValueError(
            "Reference and generated collections must contain the same number of graphs. "
            "Use --match truncate to down-sample automatically."
        )
    target = min(len(ref_graphs), len(gen_graphs))
    rng = random.Random(rng_seed)
    ref_indices = rng.sample(range(len(ref_graphs)), k=target)
    gen_indices = rng.sample(range(len(gen_graphs)), k=target)
    ref_subset = [ref_graphs[i] for i in ref_indices]
    gen_subset = [gen_graphs[i] for i in gen_indices]
    return ref_subset, gen_subset


def _metric_interval_to_dict(interval: MetricInterval) -> dict:
    return interval.to_dict()


# --- CLI -----------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Standard PolyGraph Discrepancy (PGD) between two sets of graphs.",
    )
    parser.add_argument("--reference", required=True, type=Path, help="Path to reference graphs.")
    parser.add_argument("--generated", required=True, type=Path, help="Path to generated graphs.")
    parser.add_argument(
        "--reference-format",
        choices=["gpickle", "gml", "graphml", "json", "adjlist", "edgelist", "pkl", "pickle"],
        help="Hint for parsing reference graphs. Overrides file extension detection.",
    )
    parser.add_argument(
        "--generated-format",
        choices=["gpickle", "gml", "graphml", "json", "adjlist", "edgelist", "pkl", "pickle"],
        help="Hint for parsing generated graphs. Overrides file extension detection.",
    )
    parser.add_argument(
        "--variant",
        default="jsd",
        choices=["jsd", "informedness"],
        help="Underlying probability metric to approximate (default: jsd).",
    )
    parser.add_argument(
        "--interval",
        action="store_true",
        help="Additionally compute the uncertainty interval variant.",
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=None,
        help="Subsample size for the interval estimate (defaults to half the dataset size).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of stochastic samples for the interval estimate.",
    )
    parser.add_argument(
        "--match",
        choices=["raise", "truncate"],
        default="raise",
        help="How to handle mismatched dataset sizes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling operations.")
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON result.")
    parser.add_argument("--indent", type=int, default=2, help="Indentation level for the JSON output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    ref_graphs = _load_graph_collection(
        args.reference,
        fmt=(f".{args.reference_format}" if args.reference_format else None),
    )
    gen_graphs = _load_graph_collection(
        args.generated,
        fmt=(f".{args.generated_format}" if args.generated_format else None),
    )

    ref_graphs, gen_graphs = _truncate_to_match(
        ref_graphs,
        gen_graphs,
        strategy=args.match,
        rng_seed=args.seed,
    )

    metric = StandardPGD(ref_graphs, variant=args.variant, random_state=args.seed)
    result = metric.compute(gen_graphs)

    output = {
        "pgd": result["pgd"],
        "pgd_descriptor": result["pgd_descriptor"],
        "subscores": result["subscores"],
    }

    if args.interval:
        subsample_size = args.subsample_size or max(2, len(ref_graphs) // 2)
        interval_metric = StandardPGDInterval(
            ref_graphs,
            subsample_size=subsample_size,
            num_samples=args.num_samples,
            variant=args.variant,
            random_state=args.seed,
        )
        interval_result = interval_metric.compute(gen_graphs)
        output["interval"] = {
            "pgd": _metric_interval_to_dict(interval_result["pgd"]),
            "pgd_descriptor": interval_result["pgd_descriptor"],
            "subscores": {
                name: _metric_interval_to_dict(metric_interval)
                for name, metric_interval in interval_result["subscores"].items()
            },
        }

    json_payload = json.dumps(output, indent=args.indent)
    print(json_payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_payload)

    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
