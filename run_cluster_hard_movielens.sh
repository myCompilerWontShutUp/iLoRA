#!/usr/bin/env bash
# Cluster-hard routing pipeline for MovieLens on VESSL: fixed SASRec-based KMeans clusters,
# each mapped to one LoRA expert, trained from scratch (spec section 9/11).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

: "${LLM_PATH:?LLM_PATH must be set, e.g. export LLM_PATH=/path/to/Llama-2-7b-hf}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
FORCE="${FORCE:-0}"
SEED="${SEED:-1234}"
N_CLUSTERS="${N_CLUSTERS:-4}"
CLUSTER_EXPERT_MAPPING="${CLUSTER_EXPERT_MAPPING:-0:0,1:1,2:2,3:3}"

DATA_DIR="data/ref/movielens"
REC_MODEL_PATH="./rec_model/movielens.pt"
PROMPT_PATH="./prompt/movie.txt"
BASELINE_RESULTS_DIR="results/baseline"
CLUSTERING_DIR="results/clustering"
ARTIFACTS_DIR="artifacts"
CKPT_DIR="checkpoints/cluster_hard"
OUTPUT_DIR="outputs/cluster_hard"
RESULTS_DIR="results/cluster_hard"
LOG_DIR="logs"
KMEANS_ARTIFACT="$ARTIFACTS_DIR/kmeans_k${N_CLUSTERS}_seed${SEED}.joblib"
mkdir -p "$CKPT_DIR" "$OUTPUT_DIR" "$RESULTS_DIR" "$CLUSTERING_DIR" "$ARTIFACTS_DIR" "$LOG_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/run_cluster_hard_movielens_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== run_cluster_hard_movielens.sh started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "[1/7] Checking train SASRec representations"
if [[ -f "$BASELINE_RESULTS_DIR/representations_train.npz" && "$FORCE" != "1" ]]; then
    echo "  $BASELINE_RESULTS_DIR/representations_train.npz already exists; skipping extraction."
else
    python scripts/extract_representations.py \
        --data_dir "$DATA_DIR" \
        --rec_model_path "$REC_MODEL_PATH" \
        --output_dir "$BASELINE_RESULTS_DIR" \
        --cans_num 20 \
        --device cpu
fi

echo "[2/7] Fitting KMeans (train split only) and predicting validation/test assignments"
if [[ -f "$KMEANS_ARTIFACT" && -f "$CLUSTERING_DIR/train_assignments.csv" && "$FORCE" != "1" ]]; then
    echo "  $KMEANS_ARTIFACT already exists; skipping (set FORCE=1 to refit)."
else
    python scripts/fit_kmeans.py \
        --results_dir "$BASELINE_RESULTS_DIR" \
        --output_dir "$CLUSTERING_DIR" \
        --artifacts_dir "$ARTIFACTS_DIR" \
        --n_clusters "$N_CLUSTERS" \
        --seed "$SEED"
fi

echo "[3/7] Validation/test assignments (produced as part of fit_kmeans.py above)"
for f in train_assignments.csv validation_assignments.csv test_assignments.csv; do
    if [[ ! -f "$CLUSTERING_DIR/$f" ]]; then
        echo "FATAL: $CLUSTERING_DIR/$f missing after fit_kmeans.py." >&2
        exit 1
    fi
done
echo "  all three assignment CSVs present."

echo "[4/7] Clustering analysis"
if [[ -f "$CLUSTERING_DIR/cluster_summary.json" && "$FORCE" != "1" ]]; then
    echo "  $CLUSTERING_DIR/cluster_summary.json already exists; skipping (set FORCE=1 to re-analyze)."
else
    python analysis/cluster_sasrec.py \
        --clustering_dir "$CLUSTERING_DIR" \
        --representations_dir "$BASELINE_RESULTS_DIR" \
        --artifacts_dir "$ARTIFACTS_DIR" \
        --baseline_results_dir "$BASELINE_RESULTS_DIR" \
        --seed "$SEED"
fi

echo "[5/7] Cluster-hard training (from scratch; routing_mode=cluster_hard)"
if [[ -f "$CKPT_DIR/last.ckpt" && "$FORCE" != "1" ]]; then
    echo "  $CKPT_DIR/last.ckpt already exists; skipping training (set FORCE=1 to retrain)."
else
    python main.py \
        --mode train \
        --router share \
        --gating Dense \
        --routing_mode cluster_hard \
        --cluster_model_path "$KMEANS_ARTIFACT" \
        --cluster_assignment_dir "$CLUSTERING_DIR" \
        --cluster_expert_mapping "$CLUSTER_EXPERT_MAPPING" \
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
        --log_dir movielens_cluster_hard_logs \
        --lr_warmup_start_lr 8e-6 \
        --lr 8e-4 \
        --lr_decay_min_lr 8e-6 \
        --max_epochs 5 \
        --seed "$SEED"
fi

echo "[6/7] Cluster-hard evaluation"
if [[ -f "$RESULTS_DIR/metrics.json" && "$FORCE" != "1" ]]; then
    echo "  $RESULTS_DIR/metrics.json already exists; skipping evaluation (set FORCE=1 to re-evaluate)."
else
    python main.py \
        --mode test \
        --router share \
        --gating Dense \
        --routing_mode cluster_hard \
        --cluster_model_path "$KMEANS_ARTIFACT" \
        --cluster_assignment_dir "$CLUSTERING_DIR" \
        --cluster_expert_mapping "$CLUSTER_EXPERT_MAPPING" \
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
        --log_dir movielens_cluster_hard_logs \
        --lr_warmup_start_lr 8e-6 \
        --lr 8e-4 \
        --lr_decay_min_lr 8e-6 \
        --max_epochs 5 \
        --seed "$SEED"
fi

echo "[7/7] Final comparison (iLoRA Dynamic vs. Cluster Hard)"
python analysis/compare_results.py \
    --baseline_results_dir "$BASELINE_RESULTS_DIR" \
    --cluster_hard_results_dir "$RESULTS_DIR" \
    --output_dir results

echo "=== run_cluster_hard_movielens.sh finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
