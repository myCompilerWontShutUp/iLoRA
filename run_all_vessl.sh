#!/usr/bin/env bash
# Full pipeline: baseline -> gate export -> gate analysis -> KMeans -> clustering analysis ->
# cluster-hard -> evaluation -> final comparison -> final summary (spec section 17).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

: "${LLM_PATH:?LLM_PATH must be set, e.g. export LLM_PATH=/path/to/Llama-2-7b-hf}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "=== run_all_vessl.sh started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
# Where the results will live: a workspace Pause/Stop deletes the container filesystem.
python scripts/env_check.py --storage-only || true

bash run_baseline_movielens.sh
bash run_cluster_hard_movielens.sh

echo "[final] Generating results/today_summary.md"
python analysis/generate_summary.py \
    --baseline_results_dir results/baseline \
    --cluster_hard_results_dir results/cluster_hard \
    --clustering_dir results/clustering \
    --results_dir results

echo "=== run_all_vessl.sh finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "See results/today_summary.md for the full write-up."
echo "Before pausing the workspace: bash scripts/archive_results.sh --with-checkpoints, then download archives/<file>.tgz"
