#!/usr/bin/env bash
# Baseline (iLoRA dynamic routing) reproduction pipeline for MovieLens on VESSL.
# Hyperparameters match the released train_movielens.sh / test_movielens.sh exactly
# (experiments/IMPLEMENTATION_NOTES.md section 8) plus --seed 1234 and the new export flags.
#
# Re-running this script does not redo already-completed steps unless FORCE=1 is set, and
# "completed" is judged by the presence of the actual output file/metric, not just a checkpoint
# file (spec section 17).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

: "${LLM_PATH:?LLM_PATH must be set, e.g. export LLM_PATH=/path/to/Llama-2-7b-hf}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
FORCE="${FORCE:-0}"
SEED="${SEED:-1234}"

DATA_DIR="data/ref/movielens"
REC_MODEL_PATH="./rec_model/movielens.pt"
PROMPT_PATH="./prompt/movie.txt"
CKPT_DIR="checkpoints/baseline"
OUTPUT_DIR="outputs/baseline"
RESULTS_DIR="results/baseline"
LOG_DIR="logs"
mkdir -p "$CKPT_DIR" "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/run_baseline_movielens_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run_baseline_movielens.sh started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "[1/7] Preflight"
LLM_PATH="$LLM_PATH" DATA_DIR="$DATA_DIR" PREFLIGHT_OUTPUT_DIR="$OUTPUT_DIR" bash scripts/preflight_vessl.sh

echo "[2/7] Baseline training (routing_mode=dynamic)"
if [[ -f "$CKPT_DIR/last.ckpt" && "$FORCE" != "1" ]]; then
    echo "  $CKPT_DIR/last.ckpt already exists; skipping training (set FORCE=1 to retrain)."
else
    START=$(date +%s)
    python main.py \
        --mode train \
        --router share \
        --gating Dense \
        --routing_mode dynamic \
        --batch_size 8 \
        --accumulate_grad_batches 16 \
        --dataset movielens_data \
        --data_dir "$DATA_DIR" \
        --cans_num 20 \
        --prompt_path "$PROMPT_PATH" \
        --rec_embed SASRec \
        --llm_tuning moelora \
        --llm_path "$LLM_PATH" \
        --rec_model_path "$REC_MODEL_PATH" \
        --ckpt_dir "$CKPT_DIR/" \
        --output_dir "$OUTPUT_DIR/" \
        --log_dir movielens_baseline_logs \
        --lr_warmup_start_lr 8e-6 \
        --lr 8e-4 \
        --lr_decay_min_lr 8e-6 \
        --max_epochs 5 \
        --seed "$SEED"
    END=$(date +%s)
    echo "  training elapsed: $((END - START))s"
fi

echo "[3/7] Baseline evaluation (--export_gates writes gates_test.csv as part of this step)"
if [[ -f "$RESULTS_DIR/metrics.json" && "$FORCE" != "1" ]]; then
    echo "  $RESULTS_DIR/metrics.json already exists; skipping evaluation (set FORCE=1 to re-evaluate)."
else
    python main.py \
        --mode test \
        --router share \
        --gating Dense \
        --routing_mode dynamic \
        --export_gates \
        --results_dir "$RESULTS_DIR" \
        --batch_size 8 \
        --accumulate_grad_batches 16 \
        --dataset movielens_data \
        --data_dir "$DATA_DIR" \
        --cans_num 20 \
        --prompt_path "$PROMPT_PATH" \
        --rec_embed SASRec \
        --llm_tuning moelora \
        --llm_path "$LLM_PATH" \
        --rec_model_path "$REC_MODEL_PATH" \
        --ckpt_path "$CKPT_DIR/last.ckpt" \
        --output_dir "$OUTPUT_DIR/" \
        --log_dir movielens_baseline_logs \
        --lr_warmup_start_lr 8e-6 \
        --lr 8e-4 \
        --lr_decay_min_lr 8e-6 \
        --max_epochs 5 \
        --seed "$SEED"
fi

echo "[4/7] Verifying gate export"
if [[ ! -f "$RESULTS_DIR/gates_test.csv" ]]; then
    echo "FATAL: $RESULTS_DIR/gates_test.csv missing after evaluation step." >&2
    exit 1
fi
echo "  found $RESULTS_DIR/gates_test.csv"

echo "[5/7] SASRec representation export (train/validation/test, no LLaMA involved)"
if [[ -f "$RESULTS_DIR/representations_train.npz" && "$FORCE" != "1" ]]; then
    echo "  $RESULTS_DIR/representations_train.npz already exists; skipping (set FORCE=1 to re-extract)."
else
    python scripts/extract_representations.py \
        --data_dir "$DATA_DIR" \
        --rec_model_path "$REC_MODEL_PATH" \
        --output_dir "$RESULTS_DIR" \
        --cans_num 20 \
        --device cpu
fi

echo "[6/7] Gate diversity analysis"
if [[ -f "$RESULTS_DIR/gate_summary.json" && "$FORCE" != "1" ]]; then
    echo "  $RESULTS_DIR/gate_summary.json already exists; skipping (set FORCE=1 to re-analyze)."
else
    python analysis/analyze_gates.py --results_dir "$RESULTS_DIR" --split test
fi

echo "[7/7] Verifying baseline result files"
REQUIRED_FILES=(
    "$RESULTS_DIR/metrics.json"
    "$RESULTS_DIR/gates_test.csv"
    "$RESULTS_DIR/gate_summary.json"
    "$RESULTS_DIR/figures/gate_heatmap.png"
    "$RESULTS_DIR/figures/expert_mean_utilization.png"
    "$RESULTS_DIR/figures/argmax_expert_distribution.png"
    "$RESULTS_DIR/figures/entropy_distribution.png"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  OK   $f"
    else
        echo "  MISSING $f"
        MISSING=1
    fi
done
if [[ "$MISSING" -ne 0 ]]; then
    echo "FATAL: one or more required baseline result files are missing." >&2
    exit 1
fi

echo "=== run_baseline_movielens.sh finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
