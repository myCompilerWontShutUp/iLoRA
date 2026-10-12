#!/usr/bin/env bash
# One-shot preparation of a fresh VESSL workspace (Managed image, single A100 80GB) for this repo.
# Run from the cloned repository:
#     export HF_TOKEN=...            # only needed if the Llama-2 weights are not on disk yet
#     bash bootstrap_vessl.sh
# It never starts the long training run; the command for that is printed at the end.
#
#   [1/7] tmux                       (installed with apt when missing and running as root)
#   [2/7] conda env `ilora`, Python 3.10 (Miniforge is installed when no conda exists)
#   [3/7] Llama-2-7b-hf weights      (downloaded with $HF_TOKEN only if $LLM_PATH is empty)
#   [4/7] setup_vessl.sh             (pinned requirements, bitsandbytes on torch's CUDA 11.7 runtime,
#                                     transformers patches), then a bitsandbytes check in a re-activated env
#   [5/7] preflight + CPU regression test of the MoE-LoRA rewrite
#   [6/7] GPU: evaluation path smoke test (generate, gate export, metrics.json) on the test split
#   [7/7] GPU: 256-batch training memory benchmark
# Set SKIP_GPU_CHECKS=1 to stop after step 5. Re-running is safe; finished steps are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export LLM_PATH="${LLM_PATH:-$HOME/models/Llama-2-7b-hf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ENV_NAME="ilora"
mkdir -p logs
LOG_FILE="logs/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== bootstrap_vessl.sh (log: $LOG_FILE) ==="
echo "LLM_PATH=$LLM_PATH  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[1/7] tmux"
if command -v tmux >/dev/null 2>&1; then
    echo "  tmux present."
elif [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux
else
    echo "  WARNING: tmux missing and cannot apt-get install it; run the long job with nohup instead."
fi

echo "[2/7] conda env '$ENV_NAME' (Python 3.10)"
if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "$HOME/miniforge3/bin/conda" ]]; then
        export PATH="$HOME/miniforge3/bin:$PATH"
    else
        echo "  no conda found; installing Miniforge into $HOME/miniforge3"
        INSTALLER="$(mktemp --suffix=.sh)"
        URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$INSTALLER" "$URL"; else wget -q -O "$INSTALLER" "$URL"; fi
        bash "$INSTALLER" -b -p "$HOME/miniforge3"
        rm -f "$INSTALLER"
        export PATH="$HOME/miniforge3/bin:$PATH"
    fi
fi
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    # conda-forge only: avoids the defaults-channel terms-of-service prompt of newer conda releases
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.10
fi
set +u; conda activate "$ENV_NAME"; set -u
echo "  $(python --version 2>&1) at $(command -v python)"

echo "[3/7] Llama-2-7b-hf weights"
if [[ -f "$LLM_PATH/config.json" && -f "$LLM_PATH/tokenizer.model" && ( -f "$LLM_PATH/model.safetensors.index.json" || -f "$LLM_PATH/pytorch_model.bin.index.json" ) ]]; then
    echo "  found at $LLM_PATH"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN  # read by the download below only; never written to the repository
    # A recent huggingface_hub is used only for this download; setup_vessl.sh then pins 0.13.3 back.
    pip install -q "huggingface_hub>=0.20,<1.0"
    python - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id="meta-llama/Llama-2-7b-hf",
        local_dir=os.environ["LLM_PATH"],
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        token=os.environ["HF_TOKEN"],
    )
except Exception as e:
    print(f"FATAL: download failed: {type(e).__name__}: {e}", file=sys.stderr)
    print("       The HF account behind HF_TOKEN must have been granted access to "
          "meta-llama/Llama-2-7b-hf on huggingface.co (the repo is gated).", file=sys.stderr)
    sys.exit(1)
print("  downloaded to", os.environ["LLM_PATH"])
PYEOF
else
    echo "FATAL: no Llama-2 weights at $LLM_PATH and HF_TOKEN is not set." >&2
    echo "       export HF_TOKEN=<your Hugging Face access token> and re-run, or point LLM_PATH at existing weights." >&2
    exit 1
fi

echo "[4/7] setup_vessl.sh"
bash setup_vessl.sh
set +u; conda deactivate; conda activate "$ENV_NAME"; set -u   # pick up the bitsandbytes LD_LIBRARY_PATH hook
python scripts/verify_bnb_cuda.py   # again after a fresh activation: the hook alone must be enough

echo "[5/7] preflight + CPU regression test"
bash scripts/preflight_vessl.sh
python scripts/test_moelora_equivalence.py

if [[ "${SKIP_GPU_CHECKS:-0}" == "1" ]]; then
    echo "SKIP_GPU_CHECKS=1: stopping before the GPU checks."
else
    echo "[6/7] GPU: evaluation path smoke test (untrained model, test split, outputs under benchmark_runs/)"
    SMOKE_DIR="benchmark_runs/eval_smoke"
    rm -rf "$SMOKE_DIR"
    python main.py --mode test --router share --gating Dense --routing_mode dynamic --export_gates \
        --batch_size 8 --accumulate_grad_batches 16 --dataset movielens_data --data_dir data/ref/movielens \
        --cans_num 20 --prompt_path ./prompt/movie.txt --rec_embed SASRec --llm_tuning moelora \
        --llm_path "$LLM_PATH" --rec_model_path ./rec_model/movielens.pt \
        --ckpt_dir "$SMOKE_DIR/ckpt" --output_dir "$SMOKE_DIR/" --results_dir "$SMOKE_DIR" \
        --log_dir eval_smoke_logs --max_epochs 5 --seed 1234
    python - <<'PYEOF'
import csv, json, sys
sys.path.insert(0, ".")
from data.movielens_data import MovielensData
expected = len(MovielensData(data_dir="data/ref/movielens", stage="test", cans_num=20))
m = json.load(open("benchmark_runs/eval_smoke/metrics.json"))
rows = sum(1 for _ in csv.DictReader(open("benchmark_runs/eval_smoke/gates_test.csv")))
ok = m["test_total_num"] == expected and rows == expected
print(f"  test sequences evaluated={m['test_total_num']}, gate rows={rows}, expected={expected} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PYEOF

    echo "[7/7] GPU: 256-batch training memory benchmark"
    BENCH_LOG="logs/bootstrap_benchmark_$(date -u +%Y%m%dT%H%M%SZ).log"
    python scripts/benchmark_baseline.py --llm_path "$LLM_PATH" | tee "$BENCH_LOG"
    PEAK="$(grep -o 'max_allocated=[0-9.]*' "$BENCH_LOG" | cut -d= -f2 | sort -n | tail -1)"
    if ! grep -q "completed with NO OOM" "$BENCH_LOG"; then
        echo "FATAL: the training memory benchmark did not complete without OOM (see $BENCH_LOG)." >&2
        exit 1
    fi
    echo "  256 batches without OOM; highest max_memory_allocated seen: ${PEAK:-?} GiB"
fi

cat <<EOF

=== bootstrap complete ===
Start the full experiment (baseline + cluster_hard, about two days of A100 time) inside tmux:

  tmux new -s ilora
  cd $REPO_ROOT && source $CONDA_SH && conda activate $ENV_NAME
  export LLM_PATH=$LLM_PATH CUDA_VISIBLE_DEVICES=0
  bash run_all_vessl.sh

Detach with Ctrl-b d, re-attach with: tmux attach -t ilora
EOF
