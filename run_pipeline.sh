#!/bin/bash
set -euo pipefail

# Auto-detect repo root if not set (assumes script is in geognn_ot/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODULE_NAME="${MODULE_NAME:-unknown}"
HEMI="${HEMI:-rh}"
USE_INTRINSIC="${USE_INTRINSIC:-0}"
USE_NORMAL="${USE_NORMAL:-0}"
USE_XYZ="${USE_XYZ:-0}"
USE_DISTANCE="${USE_DISTANCE:-0}"
EPOCHS="${EPOCHS:-300}"
MAX_TRAIN_MSE="${MAX_TRAIN_MSE:-0.3}"
N_TRIALS="${N_TRIALS:-6}"
OPTUNA_EPOCHS="${OPTUNA_EPOCHS:-2000}"
CAND_K="${CAND_K:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --module-name) MODULE_NAME="$2"; shift 2 ;;
    --hemi) HEMI="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --max-train-mse) MAX_TRAIN_MSE="$2"; shift 2 ;;
    --n-trials) N_TRIALS="$2"; shift 2 ;;
    --optuna-epochs) OPTUNA_EPOCHS="$2"; shift 2 ;;
    --cand-k) CAND_K="$2"; shift 2 ;;
    --use-intrinsic) USE_INTRINSIC=1; shift ;;
    --use-normal) USE_NORMAL=1; shift ;;
    --use-xyz) USE_XYZ=1; shift ;;
    --use-distance) USE_DISTANCE=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

GRID_DIR="geognn_ot/results/grid/${MODULE_NAME}/${HEMI}"
mkdir -p "$GRID_DIR"

echo "=========================================="
echo "Pipeline: Grid Search -> Optuna -> Eval"
echo "Module: $MODULE_NAME, Hemisphere: $HEMI"
echo "=========================================="

# Step 1: Grid Search
echo ""
echo "Step 1: Running Grid Search..."
echo "----------------------------------------"

# Grid search parameter combinations
temps=(0.1 0.4 0.7)
lrs=(1e-3 2e-3)
wds=(1e-4 1e-5 0.0)
clips=(1.0)
ks=(64)
topos=(0.0 0.5 1.0)

CAND_FILE="${GRID_DIR}/candidates.json"
candidates=()

for temp in "${temps[@]}"; do
  for lr in "${lrs[@]}"; do
    for wd in "${wds[@]}"; do
      for clip in "${clips[@]}"; do
        for k in "${ks[@]}"; do
          for topo in "${topos[@]}"; do
            echo "Running: temp=$temp lr=$lr wd=$wd clip=$clip k=$k topo=$topo"
            
            CMD="python -u -m geognn_ot.train train \
              --repo-root $REPO_ROOT \
              --hemi $HEMI \
              --train-subjects R1 \
              --out-dir geognn_ot/results/checkpoints/${MODULE_NAME} \
              --ckpt-out /dev/null \
              --epochs $EPOCHS \
              --lr $lr \
              --weight-decay $wd \
              --clip-norm $clip \
              --temp $temp \
              --cand-k-src $k \
              --cand-method score_topk \
              --topo-lambda $topo \
              --seed 0 \
              --max-train-mse $MAX_TRAIN_MSE \
              --grid-candidates-out ${GRID_DIR}/candidates_temp.json"
            
            if [ "$USE_INTRINSIC" = "1" ]; then CMD="$CMD --use-intrinsic"; fi
            if [ "$USE_NORMAL" = "1" ]; then CMD="$CMD --use-normal"; fi
            if [ "$USE_XYZ" = "1" ]; then CMD="$CMD --use-xyz"; fi
            if [ "$USE_DISTANCE" = "1" ]; then CMD="$CMD --use-distance"; fi
            
            if $CMD; then
              # If training succeeded, merge the candidate
              if [ -f "${GRID_DIR}/candidates_temp.json" ]; then
                if [ -f "$CAND_FILE" ]; then
                  python -c "
import json
with open('$CAND_FILE', 'r') as f:
  existing = json.load(f)
with open('${GRID_DIR}/candidates_temp.json', 'r') as f:
  new = json.load(f)
existing.extend(new if isinstance(new, list) else [new])
with open('$CAND_FILE', 'w') as f:
  json.dump(existing, f, indent=2)
"
                  rm "${GRID_DIR}/candidates_temp.json"
                else
                  mv "${GRID_DIR}/candidates_temp.json" "$CAND_FILE"
                fi
              fi
            else
              echo "Grid search failed for params: temp=$temp lr=$lr wd=$wd k=$k topo=$topo"
            fi
          done
        done
      done
    done
  done
done

if [ ! -f "$CAND_FILE" ]; then
  echo "Error: No candidates generated from grid search. Exiting."
  exit 1
fi

echo "Grid search complete. Candidates saved to: $CAND_FILE"

# Step 2: Optuna HPO
echo ""
echo "Step 2: Running Optuna HPO..."
echo "----------------------------------------"

CMD="python -u geognn_ot/optuna_refine_from_grid.py \
  --repo-root $REPO_ROOT \
  --hemi $HEMI \
  --module-name $MODULE_NAME \
  --n-trials $N_TRIALS \
  --epochs $OPTUNA_EPOCHS \
  --patience 100 \
  --device cuda"

if [ "$USE_INTRINSIC" = "1" ]; then CMD="$CMD --use-intrinsic"; fi
if [ "$USE_NORMAL" = "1" ]; then CMD="$CMD --use-normal"; fi
if [ "$USE_XYZ" = "1" ]; then CMD="$CMD --use-xyz"; fi
if [ "$USE_DISTANCE" = "1" ]; then CMD="$CMD --use-distance"; fi

$CMD

BEST_JSON="geognn_ot/results/params/${MODULE_NAME}/best_${HEMI}.json"
if [ ! -f "$BEST_JSON" ]; then
  echo "Error: Optuna HPO did not produce best_${HEMI}.json. Exiting."
  exit 1
fi

echo "Optuna HPO complete. Best params saved to: $BEST_JSON"

# Step 3: Evaluation
echo ""
echo "Step 3: Running Evaluation..."
echo "----------------------------------------"

CKPT=$(python -c "import json; print(json.load(open('$BEST_JSON'))['best_ckpt'])")

if [ ! -f "$CKPT" ]; then
  echo "Error: Best checkpoint not found: $CKPT. Exiting."
  exit 1
fi

echo "Evaluating best checkpoint: $CKPT for Subject S1-S6..."

CMD="python -m geognn_ot.train evaluate \
  --repo-root $REPO_ROOT \
  --ckpt $CKPT \
  --subjects R1 S1 S2 S3 S4 S5 S6 \
  --hemis $HEMI \
  --out-dir geognn_ot/results/eval/${MODULE_NAME} \
  --device cuda \
  --cand-k-src $CAND_K"

if [ "$USE_INTRINSIC" = "1" ]; then CMD="$CMD --use-intrinsic"; fi
if [ "$USE_NORMAL" = "1" ]; then CMD="$CMD --use-normal"; fi
if [ "$USE_XYZ" = "1" ]; then CMD="$CMD --use-xyz"; fi
if [ "$USE_DISTANCE" = "1" ]; then CMD="$CMD --use-distance"; fi

$CMD

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results saved to: geognn_ot/results/eval/${MODULE_NAME}"
echo "=========================================="
