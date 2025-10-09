# Mem2GenVGAE Experiments

This repository hosts the Mem2GenVGAE pipeline for graph generation and the new memorization→generalization study. The code builds on a two-stage architecture: a VGAE autoencoder (`autoencoder.py`) and a latent diffusion denoiser (`denoise_model.py`). The `experiments/` folder now contains tooling to reproduce the stratified split and multi-scale training sweep requested for the feature homophily dataset.

## Environment

1. Activate the shared conda environment:
   ```bash
   conda activate pygeo
   ```
2. Ensure the environment includes the following Python packages (matching the existing project stack):
   - `torch`, `torch-geometric`
   - `grakel`
   - `networkx`
   - `matplotlib`
   - `pandas`

## Memorization→Generalization Workflow

1. **Create balanced splits**
   ```bash
   python experiments/memorization_split.py \
     --dataset-path data/featurehomophily0.6_graphs.pkl \
     --csv-path data/featurehomophily0.6_log.csv \
     --test-ratio 0.1 \
     --bins 20 \
     --output-dir experiments/memorization_split
   ```
   - Outputs stratified index tensors (`s1_indices.pt`, `s2_indices.pt`, `test_indices.pt`) plus summary stats.

2. **Run the training sweep and evaluation**
   ```bash
   python experiments/run_memorization_study.py \
     --dataset-path data/featurehomophily0.6_graphs.pkl \
     --csv-path data/featurehomophily0.6_log.csv \
     --output-dir experiments/memorization_outputs \
     --subset-sizes 10 20 50 100 500 1000 2500 4500 \
     --samples-per-model 256 \
     --epochs 200 \
     --device cuda
   ```
   - Trains `VGAE1`/`VGAE2` across each subset, saves checkpoints under `N_<size>/vgae{1,2}/`.
   - Saves WL similarity arrays (`train_similarity.npy`, `cross_similarity.npy`) and per-N metrics.
   - Generates `memorization_histograms.png` with overlaid histograms (axis labels sized 25, no subplot titles).
   - Writes `summary.json` aggregating statistics across all subset sizes.

## Outputs

- `experiments/memorization_outputs/splits/` — cached split indices and feature homophily stats.
- `experiments/memorization_outputs/N_<size>/` — per-subset results, including trained model checkpoints and similarity metrics.
- `experiments/memorization_outputs/memorization_histograms.png` — visualization of the memorization→generalization trend.

## Troubleshooting

- If you update the dataset or want a fresh split, rerun `memorization_split.py` with `--seed <new_seed>` or pass `--force-resplit` to `run_memorization_study.py`.
- Use `--no-dfs` if you need to disable DFS reordering for ablations.
- For CPU-only runs, drop the `--device` flag or set `--device cpu`; reduce `--samples-per-model` to shorten WL kernel evaluations.
