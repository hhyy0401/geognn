# GeoGNN-OT Pipeline

This directory contains the code for training and evaluating GeoGNN-OT models for cortical surface alignment using optimal transport.

## Overview

The pipeline consists of three main stages:
1. **Grid Search**: Explores a predefined hyperparameter space
2. **Optuna HPO**: Refines hyperparameters using Bayesian optimization
3. **Evaluation**: Evaluates the best model on multiple subjects

## Quick Start

Run the complete pipeline:

```bash
bash run_pipeline.sh \
  --module-name <module_name> \
  --hemi <lh|rh> \
  [--use-intrinsic] \
  [--use-normal] \
  [--use-xyz] \
  [--use-distance]
```

Example:
```bash
bash run_pipeline.sh --module-name xyz --hemi lh --use-xyz
```

## Scripts

### `run_pipeline.sh`
Main pipeline script that sequentially runs grid search, Optuna HPO, and evaluation.

**Required arguments:**
- `--module-name`: Name identifier for the model configuration
- `--hemi`: Hemisphere (`lh` or `rh`)

**Optional arguments:**
- `--repo-root`: Repository root directory (default: current directory)
- `--use-intrinsic`: Use intrinsic features
- `--use-normal`: Use normal vectors
- `--use-xyz`: Use XYZ coordinates
- `--use-distance`: Use distance features
- `--epochs`: Number of epochs for grid search (default: 300)
- `--max-train-mse`: Maximum training MSE threshold (default: 0.3)
- `--n-trials`: Number of Optuna trials (default: 6)
- `--optuna-epochs`: Number of epochs for Optuna training (default: 2000)
- `--cand-k`: Number of candidate source vertices (default: 8)

## Core Modules

### `train.py`
Main training and evaluation script.

**Training mode:**
```bash
python -m geognn_ot.train train \
  --repo-root <path> \
  --hemi <lh|rh> \
  --train-subjects R1 \
  --out-dir <checkpoint_dir> \
  --ckpt-out <checkpoint_path> \
  --epochs <num> \
  --lr <learning_rate> \
  --temp <temperature> \
  --cand-k-src <k>
```

**Evaluation mode:**
```bash
python -m geognn_ot.train evaluate \
  --repo-root <path> \
  --ckpt <checkpoint_path> \
  --subjects R1 S1 S2 S3 S4 S5 S6 \
  --hemis lh rh \
  --out-dir <output_dir> \
  --device cuda \
  --cand-k-src <k>
```

**Key arguments:**
- `--repo-root`: Repository root directory
- `--hemi`: Hemisphere to process
- `--temp`: Temperature parameter for optimal transport
- `--cand-k-src`: Number of candidate source vertices per target
- `--topo-lambda`: Topological regularization strength
- `--grid-candidates-out`: Output path for grid search candidates (JSON)

### `optuna_refine_from_grid.py`
Hyperparameter optimization using Optuna, constrained by grid search results.

```bash
python geognn_ot/optuna_refine_from_grid.py \
  --repo-root <path> \
  --hemi <lh|rh> \
  --module-name <name> \
  --n-trials <num> \
  --epochs <num>
```

**Key arguments:**
- `--hemi`: Hemisphere (`lh` or `rh`)
- `--module-name`: Module identifier
- `--grid-json`: Path to grid search candidates JSON (auto-detected if not provided)
- `--n-trials`: Number of Optuna trials
- `--epochs`: Training epochs per trial
- `--patience`: Early stopping patience

The script reads grid search candidates from `results/grid/<module_name>/<hemi>/candidates.json` and uses them to define search bounds for Bayesian optimization.

### `model.py`
GeoGNN-OT model implementation with Graph Attention Network (GAT) and optimal transport layers.

### `dataset.py`
Data loading and preprocessing for cortical surface meshes.

### `alignment.py`
Optimal transport alignment algorithms and utilities.

## Output Structure

```
geognn_ot/results/
├── grid/
│   └── <module_name>/
│       └── <hemi>/
│           └── candidates.json          # Grid search results
├── checkpoints/
│   └── <module_name>/
│       └── R1_<hemi>_<tag>.pt           # Model checkpoints
├── params/
│   └── <module_name>/
│       ├── best_<hemi>.json             # Best hyperparameters
│       └── optuna_<hemi>_trials.json    # All Optuna trials
└── eval/
    └── <module_name>/
        └── <subject>_<hemi>.json        # Evaluation results
```

## Feature Flags

Control which features are used in the model:
- `--use-intrinsic`: Intrinsic geometric features
- `--use-normal`: Surface normal vectors
- `--use-xyz`: 3D coordinates
- `--use-distance`: Distance-based features

These flags determine the input feature dimensions and model architecture.
