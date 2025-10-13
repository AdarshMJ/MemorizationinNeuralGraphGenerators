# Standalone Standard PolyGraph Discrepancy

This document explains how to use `standard_pgd_standalone.py` to compare generated graphs against a reference corpus with the Standard PolyGraph Discrepancy (PGD) metric.

## Requirements

- Python 3.8+
- [NetworkX](https://networkx.org/)
- `numpy`, `scipy`, `torch`
- `scikit-learn`
- `tabpfn==2.0.9` (default classifier)
- Access to the PolyGraph source tree (the `polygraph` Python package). The repository already ships the `polygraph-benchmark-master` directory. If you relocate the script, provide `--polygraph-path` pointing to the directory that contains the `polygraph` package.
- Optional: to avoid installing TabPFN you can pass `--classifier logistic`, which relies on scikit-learn's `LogisticRegression`.

Activate the shared environment before running any commands:

```bash
conda activate pygeo
```

Install any missing Python packages inside the environment if needed.

## Supported graph inputs

`--reference` and `--generated` accept either a directory or a pickle file containing NetworkX graphs. The loader recognises the following formats based on file extension (or the `--*-format` override):

- `.gpickle`
- `.gml`
- `.graphml`
- `.json` (NetworkX node-link JSON)
- `.adjlist`
- `.edgelist` / `.txt`
- `.pkl` / `.pickle` containing an iterable of NetworkX graphs (optionally wrapped in a dictionary under the key `"graphs"`).

When passing a directory, all supported files are loaded recursively from the top level (sub-directories are ignored).

If the number of reference and generated graphs differs, the metric requires matching sizes. Use `--match truncate` to randomly down-sample both sets to the same length (controlled by `--seed`). The default behaviour is to raise an error.

## Basic usage

```bash
python standard_pgd_standalone.py \
  --reference data/reference_graphs.pkl \
  --generated data/generated_graphs.pkl \
  --output runs/pgd_results.json
```

This command prints the result to stdout and writes the same JSON to `runs/pgd_results.json`.

### Logistic regression fallback

```bash
python standard_pgd_standalone.py \
  --reference reference_dir \
  --generated generated_dir \
  --classifier logistic
```

Use this mode if TabPFN is unavailable. The resulting PGD score remains a valid lower bound but may be less tight.

### Interval estimates

```bash
python standard_pgd_standalone.py \
  --reference ref.pkl \
  --generated gen.pkl \
  --interval \
  --subsample-size 50 \
  --num-samples 20
```

The output includes an `interval` section with means, standard deviations, and (if applicable) bounds for each descriptor as well as the aggregate PGD score.

### Custom PolyGraph location

```bash
python standard_pgd_standalone.py \
  --reference ref.pkl \
  --generated gen.pkl \
  --polygraph-path /path/to/polygraph-benchmark-master
```

## Output schema

The script emits JSON with the following structure:

```json
{
  "pgd": 0.97,
  "pgd_descriptor": "degree",
  "subscores": {
    "degree": 0.97,
    "orbit4": 0.95,
    "orbit5": 0.94,
    "clustering": 0.93,
    "spectral": 0.91,
    "gin": 0.90
  },
  "interval": {
    "pgd": {
      "mean": 0.96,
      "std": 0.01,
      "low": null,
      "high": null,
      "coverage": null
    },
    "pgd_descriptor": {
      "degree": 0.65,
      "clustering": 0.35,
      "orbit4": 0.00,
      "orbit5": 0.00,
      "spectral": 0.00,
      "gin": 0.00
    },
    "subscores": {
      "degree": { "mean": 0.96, "std": 0.02, "low": null, "high": null, "coverage": null }
    }
  }
}
```

Keys inside `interval.subscores` exist for every descriptor; only `degree` is shown above for brevity.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'polygraph'`** – Provide `--polygraph-path` pointing to the directory that contains the `polygraph` package or place the script alongside `polygraph-benchmark-master`.
- **`RuntimeError: TabPFN version 2.0.9 is required`** – Install the matching version or run with `--classifier logistic`.
- **`ValueError: Number of generated graphs must be equal ...`** – Supply `--match truncate` to down-sample or adjust your datasets.
