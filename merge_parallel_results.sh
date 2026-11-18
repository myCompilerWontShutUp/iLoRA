#!/usr/bin/env bash
# Finish the experiment after baseline and cluster_hard were trained in two separate workspaces.
#
# On the cluster_hard workspace, after run_cluster_hard_movielens.sh finished:
#     tar czf cluster_hard_results.tgz results/clustering results/cluster_hard artifacts logs
# Copy that archive into the repository on the baseline workspace, then (conda env active):
#     tar xzf cluster_hard_results.tgz
#     bash merge_parallel_results.sh
#
# CPU only. Re-runs the clustering analysis now that the baseline gates exist (when cluster_hard
# ran alone its dynamic-gate cross-reference was NOT_RUN), then the final comparison and summary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

REQUIRED=(
    results/baseline/metrics.json
    results/baseline/gates_test.csv
    results/baseline/representations_train.npz
    results/clustering/train_assignments.csv
    results/clustering/test_assignments.csv
    results/cluster_hard/metrics.json
    artifacts/kmeans_k4_seed1234.joblib
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [[ -f "$f" ]]; then echo "  OK      $f"; else echo "  MISSING $f"; MISSING=1; fi
done
if [[ "$MISSING" -ne 0 ]]; then
    echo "FATAL: run this on the baseline workspace after unpacking cluster_hard_results.tgz." >&2
    exit 1
fi

echo "[1/3] Clustering analysis with the baseline dynamic gates"
python analysis/cluster_sasrec.py --clustering_dir results/clustering --representations_dir results/baseline \
    --artifacts_dir artifacts --baseline_results_dir results/baseline --seed 1234

echo "[2/3] Final comparison"
python analysis/compare_results.py --baseline_results_dir results/baseline \
    --cluster_hard_results_dir results/cluster_hard --output_dir results

echo "[3/3] Summary"
python analysis/generate_summary.py --baseline_results_dir results/baseline \
    --cluster_hard_results_dir results/cluster_hard --clustering_dir results/clustering --results_dir results

echo "Done: results/final_comparison.md, results/today_summary.md"
