# MemorizationinNeuralGraphGenerators
We study memorization-generalization trends in VGAE+LDM applied to graphs.


# Memorization-to-Generalization Experiment Guide

## Overview

This document explains how we measure Weisfeiler-Lehman (WL) kernel similarity and how the full memorization-to-generalization experiment in `main_comparison.py` is organized. Use it as the playbook for reproducing our results, interpreting metrics, and extending the study.


## Weisfeiler-Lehman Similarity Measurement

### Graph preprocessing

1. Decode each generated adjacency matrix into a NetworkX graph via `construct_nx_from_adj`.
2. Assign every node a categorical label equal to its degree. This produces topology-derived node labels that make WL focus on structural patterns.
3. Convert the pair of graphs into the format expected by `grakel.graph_from_networkx` with the `label` attribute.

### WL kernel definition

The Weisfeiler-Lehman subtree kernel refines node labels for $h$ iterations and compares resulting label multisets. With base vertex histogram kernel $k_0$, the WL kernel of height $H$ is

$$k_\text{WL}(G_1, G_2) = \sum_{h=0}^{H} k_0(G_1^{(h)}, G_2^{(h)}),$$

where $G^{(0)}$ uses the initial labels (node degrees) and $G^{(h)}$ are the graphs after $h$ WL relabeling steps.

In our implementation:

- `WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)` handles the label refinement and histogram comparison.
- Normalization ensures $k_\text{WL}(G,G)=1$ and bounds cross-similarities in $[0,1]$.
- The similarity we log is the off-diagonal element of the $2\times2$ Gram matrix returned by the kernel after fitting on the pair `[{G1, G2}]`.

### Implementation notes

- Empty graphs receive similarity 1 when both are empty and 0 otherwise (fallback branch in `compute_wl_similarity`).
- If `grakel` raises due to incompatible labels, we fall back to an edge-overlap ratio to keep the pipeline resilient.
- WL similarity is used several times: between samples from the two models (generalization), between generated samples and training graphs (memorization), and for within-model diversity checks.


## Experiment Pipeline

### Dataset preparation

1. **Synthetic generation (optional):** `synthgraphgenerator.py` creates datasets with controlled feature homophily. The generator now clamps covariance diagonals to avoid invalid multivariate normals.
2. **Dataset loading:** `load_dataset` reads the pickle file into PyG `Data` objects. Each data point carries adjacency (`A`), features (`x`), label stats, and cached graph statistics (`stats`).

### Splitting and conditioning cache

1. `shuffle_and_split_dataset` shuffles indices and partitions graphs into pools `S1`, `S2`, and a held-out conditioning test set.
2. Conditioning statistics (`stats`) for the test pool are stacked once, stored in a cache file (`data/cache/...pt`), and reused across all $N$ values to avoid recomputation.
3. `create_splits` slices `S1` and `S2` to the requested size $N$ while reusing the fixed test pool, ensuring non-overlapping training subsets.

### Model training per split

For each training size $N$:

1. **Autoencoder (`VariationalAutoEncoder`):**
   - Train for up to 100 epochs with early stopping (`EARLY_STOPPING_PATIENCE=30`).
   - KL weight is reduced for very small datasets to prevent collapse.
   - After training, we report reconstruction MAE/accuracy and restore the best checkpoint.
2. **Latent denoiser (`DenoiseNN` diffusion model):**
   - Uses a linear beta schedule with 100 timesteps.
  - Trained with the autoencoder encoder’s latent codes and graph statistics as conditioning vectors.
  - Uses a Huber loss against the noise schedule and early stopping identical to the autoencoder.

### Generation and evaluation loop

For each graph in the test conditioning set:

1. **Sampling:**
   - Draw `NUM_SAMPLES_PER_CONDITION` (default 5) from each model conditioned on the cached statistics.
   - Collect decoded NetworkX graphs and latent codes.
2. **Metrics:**
   - Pairwise WL similarity between models (generalization).
   - WL similarity against the nearest training graph from `S1` (memorization).
   - Graph statistics distances using `gen_stats` (MSE & MAE across the first 15 features).
   - Within-model WL similarities for diversity estimates.
   - Latent MMD using an RBF mixture. Bandwidth defaults to the median heuristic with $
\gamma \in \{\gamma, \tfrac{1}{2}\gamma, 2\gamma\}$.
   - Per-conditioning variance of the generalization scores.
3. **Latent analytics:**
   - Aggregate latents per model to compute an overall MMD.
   - Project concatenated latents to 2D via PCA and store scatter plots.

### Post-processing and reporting

- `perform_distribution_checks` computes Kolmogorov–Smirnov tests on conditioning statistics to ensure `S1` and `S2` remain comparable.
- `visualize_single_experiment` saves per-$N$ quicklook histograms, example graphs, and latent PCA plots.
- `visualize_results` creates multi-$N$ figures: histogram grids, exemplar graphs, convergence curves, and variance analysis.
- `save_summary` writes numeric aggregates to `experiment_summary.txt`.


## Metrics Reference

| Metric | Definition | Purpose |
| --- | --- | --- |
| **Generalization WL** | $\text{WL}(G^{\text{S1}}_i, G^{\text{S2}}_i)$ per paired sample | Detect convergence between independently trained models |
| **Memorization WL** | $\max_{G \in S_1} \text{WL}(G^{\text{S1}}_i, G)$ | Measure similarity of samples to training memories |
| **Statistics MSE / MAE** | Compare $\text{gen\_stats}(G^{\text{S1}}_i)$ vs $\text{gen\_stats}(G^{\text{S2}}_i)$ | Check consistency of structural properties |
| **Within-model WL** | WL similarities among samples from one model | Estimate mode collapse or diversity |
| **Latent MMD** | $$\text{MMD}^2 = \frac{1}{n^2} \sum_{i,j} k(x_i, x_j) + \frac{1}{m^2} \sum_{i,j} k(y_i, y_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j)$$ with RBF kernels | Quantify how close latent distributions align |
| **KS fail count** | Number of conditioning features where $p < 0.05$ (two-sample KS) | Confirm the splits remain statistically matched |

Interpretation highlights:

- Memorization is high when the model reproduces training graphs; generalization catches up to memorization as $N$ grows.
- When generalization mean $\approx$ within-model mean, the models generate indistinguishable distributions.
- High latent MMD alongside high WL similarity often indicates equivalent graph structure but different latent coordinate systems.


## Running the Experiment

1. **Activate the environment** (per lab policy):

```bash
conda activate pygeo
```

2. **(Optional) Generate a dataset** with controlled homophily:

```bash
python synthgraphgenerator.py --homophily_type feature --min_hom 0.4 --max_hom 0.4 --n_graphs 10000 --dataset_name featurehomophily0.4
```

3. **Launch the experiment**:

```bash
python main_comparison.py --data-path data/featurehomophily0.4_graphs.pkl --run-name convergence_experiment
```

4. **Outputs** are stored under `outputs/convergence_experiment_*`: figures, quicklook plots, per-$N$ pickles, and the summary text file. Conditioning caches live alongside the dataset (`data/cache/`).


## Troubleshooting

- **Invalid covariance during dataset generation:** Already mitigated by clamping variances; if explosions occur, reduce `--n_graphs` or check that feature dimension matches expectations.
- **WL kernel failures:** If `grakel` cannot process a graph (e.g., due to isolated nodes with missing labels), the fallback edge-overlap similarity triggers and emits a warning.
- **CUDA/CPU mismatches:** Ensure `device` is set once (`cuda:0` if available) and that cached tensors move via `.to(device)` as in `generate_graphs`.
- **Slow runs for large $N$:** Use the conditioning cache and consider decreasing `NUM_SAMPLES_PER_CONDITION` during exploratory sweeps.
- **Latent PCA anomalies:** Orthogonal clusters usually indicate the VAEs picked different bases rather than true distributional mismatch. Use Procrustes alignment if you need shared axes.


## Extending the Study

- **Alternative similarity measures:** Plug another kernel into `compute_wl_similarity`, e.g., shortest-path or graphlet kernels from Grakel.
- **New conditioning attributes:** Extend `gen_stats` and bump `N_PROPERTIES` accordingly (mind the cached tensor shape).
- **Additional diagnostics:** Track FID-like metrics in latent space, or add graph edit distance sampling for qualitative checks.
- **Ablations:** Vary diffusion timesteps, latent dimensionality, or conditioning vector size to study sensitivity.

This guide should equip you to reproduce, audit, and modify the memorization-to-generalization experiments with confidence.

