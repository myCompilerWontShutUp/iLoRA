#!/usr/bin/env bash
# One-shot preparation of a fresh VESSL workspace (Managed image, single A100 80GB) for this repo.
# Run from the cloned repository:
#     export HF_TOKEN=...            # only needed if the Llama-2 weights are not on disk yet
#     bash bootstrap_vessl.sh
# It never starts the long training run; the command for that is printed at the end.
#
#   [0/8] storage                    (warns when the repo / weights sit on the container filesystem,
#                                     which a workspace Pause/Stop deletes)
#   [1/8] tmux                       (apt, non-interactive, when missing and running as root)
#   [2/8] conda env `ilora`, Python 3.10, conda-forge only (no defaults channel, so no Terms of
#                                     Service prompt); never relies on the image's base Python
#   [3/8] Llama-2-7b-hf weights      (every shard of the index present; otherwise downloaded or
#                                     resumed with $HF_TOKEN)
#   [4/8] setup_vessl.sh             (pinned requirements, bitsandbytes on torch's CUDA 11.7 runtime,
#                                     transformers patches), then a bitsandbytes check in a re-activated env
#   [5/8] preflight (GPU count, torch CUDA, scripts/env_check.py) + CPU regression tests
#   [6/8] GPU: evaluation path smoke test (generate, gate export, metrics.json) on the test split
#   [7/8] GPU: 256-batch training memory benchmark
#   [8/8] summary and the commands to start the run
# Set SKIP_GPU_CHECKS=1 to skip steps 6-7. Re-running is safe: finished steps are kept as they are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export LLM_PATH="${LLM_PATH:-$HOME/models/Llama-2-7b-hf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1          # ~/.local site-packages never shadow the env's packages
export DEBIAN_FRONTEND=noninteractive
ENV_NAME="ilora"
mkdir -p logs
LOG_FILE="logs/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== bootstrap_vessl.sh (log: $LOG_FILE) ==="
echo "repository: $REPO_ROOT at $(git rev-parse --short HEAD 2>/dev/null || echo '?') ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"
echo "LLM_PATH=$LLM_PATH  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[0/8] storage"
if command -v python3 >/dev/null 2>&1; then
    python3 scripts/env_check.py --storage-only || true   # stdlib only; runs with whatever python3 the image has
fi

echo "[1/8] tmux"
if command -v tmux >/dev/null 2>&1; then
    echo "  tmux present: $(tmux -V) at $(command -v tmux)"
elif [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    if apt-get update -qq && apt-get install -y -qq --no-install-recommends tmux; then
        echo "  installed $(tmux -V)"
    else
        echo "  WARNING: apt-get could not install tmux; run the long job with nohup instead."
    fi
else
    echo "  WARNING: tmux missing and cannot apt-get install it; run the long job with nohup instead."
fi

echo "[2/8] conda env '$ENV_NAME' (Python 3.10)"
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
echo "  conda $(conda --version 2>/dev/null | awk '{print $2}') at $(conda info --base); base Python: $(python3 --version 2>&1 || echo none) (not used)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    # Only conda-forge, never the Anaconda defaults channel: newer conda releases refuse to use
    # defaults until its Terms of Service are accepted interactively, which stopped the first
    # attempt on VESSL. -y and --override-channels keep this non-interactive.
    conda create -y -q -n "$ENV_NAME" -c conda-forge --override-channels python=3.10
else
    echo "  env '$ENV_NAME' exists; reusing it"
fi
set +u; conda activate "$ENV_NAME"; set -u
PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_EXE="$(python -c 'import os, sys; print(os.path.realpath(sys.executable))')"
if [[ "$PY_VERSION" != "3.10" ]]; then
    echo "FATAL: env '$ENV_NAME' has Python $PY_VERSION, not 3.10. It was not created by this script; remove it" >&2
    echo "       and re-run: conda env remove -y -n $ENV_NAME && bash bootstrap_vessl.sh" >&2
    exit 1
fi
if [[ "$PY_EXE" != "$(realpath "$CONDA_PREFIX")"/* ]]; then
    echo "FATAL: 'python' resolves to $PY_EXE, outside the active env $CONDA_PREFIX." >&2
    exit 1
fi
echo "  $(python --version 2>&1) at $PY_EXE"

echo "[3/8] Llama-2-7b-hf weights"
WEIGHTS_STATE="$(python - <<'PYEOF'
import json, os
path = os.environ['LLM_PATH']
need = [f for f in ('config.json', 'tokenizer.model') if not os.path.exists(os.path.join(path, f))]
index = next((f for f in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
              if os.path.exists(os.path.join(path, f))), None)
if index is None:
    need.append('model.safetensors.index.json')
else:
    shards = set(json.load(open(os.path.join(path, index)))['weight_map'].values())
    need += sorted(s for s in shards if not os.path.exists(os.path.join(path, s)))
print('complete' if not need else 'missing ' + ' '.join(need))
PYEOF
)"
if [[ "$WEIGHTS_STATE" == "complete" ]]; then
    echo "  found at $LLM_PATH (config, tokenizer and every shard of the index); not downloading"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  $WEIGHTS_STATE -> downloading (an interrupted download is resumed)"
    export HF_TOKEN  # read by the download below only; never written to the repository
    # A recent huggingface_hub is used only for this download; setup_vessl.sh then pins 0.13.3 back.
    python -m pip install -q "huggingface_hub>=0.20,<1.0"
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
    echo "FATAL: Llama-2 weights at $LLM_PATH incomplete ($WEIGHTS_STATE) and HF_TOKEN is not set." >&2
    echo "       export HF_TOKEN=<your Hugging Face access token> and re-run, or point LLM_PATH at existing weights." >&2
    exit 1
fi

echo "[4/8] setup_vessl.sh"
bash setup_vessl.sh
set +u; conda deactivate; conda activate "$ENV_NAME"; set -u   # pick up the bitsandbytes LD_LIBRARY_PATH hook
python scripts/verify_bnb_cuda.py   # again after a fresh activation: the hook alone must be enough

echo "[5/8] preflight (includes scripts/env_check.py) + CPU regression tests"
bash scripts/preflight_vessl.sh
python scripts/test_moelora_equivalence.py
python scripts/test_measurement.py
python scripts/test_main_args_unchanged.py

if [[ "${SKIP_GPU_CHECKS:-0}" == "1" ]]; then
    echo "SKIP_GPU_CHECKS=1: skipping the GPU checks (steps 6-7)."
else
    echo "[6/8] GPU: evaluation path smoke test (untrained model, test split, outputs under benchmark_runs/)"
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

    echo "[7/8] GPU: 256-batch training memory benchmark"
    BENCH_LOG="logs/bootstrap_benchmark_$(date -u +%Y%m%dT%H%M%SZ).log"
    python scripts/benchmark_baseline.py --llm_path "$LLM_PATH" | tee "$BENCH_LOG"
    PEAK="$(grep -o 'max_allocated=[0-9.]*' "$BENCH_LOG" | cut -d= -f2 | sort -n | tail -1)"
    if ! grep -q "completed with NO OOM" "$BENCH_LOG"; then
        echo "FATAL: the training memory benchmark did not complete without OOM (see $BENCH_LOG)." >&2
        exit 1
    fi
    echo "  256 batches without OOM; highest max_memory_allocated seen: ${PEAK:-?} GiB"
fi

echo "[8/8] summary"
cat <<EOF

=== bootstrap complete ===
Activate the env in every new terminal or tmux window:

  source $CONDA_SH && conda activate $ENV_NAME

Start the full experiment (baseline + cluster_hard) inside tmux (plain tmux works; the env does not
put conda's own libraries on LD_LIBRARY_PATH):

  tmux new -s ilora
  cd $REPO_ROOT && source $CONDA_SH && conda activate $ENV_NAME
  export LLM_PATH=$LLM_PATH CUDA_VISIBLE_DEVICES=0
  bash run_all_vessl.sh

Detach with Ctrl-b d, re-attach with: tmux attach -t ilora

!!! A workspace Pause/Stop deletes the container filesystem (see the [0/8] storage lines above):
!!! the repo, the env, the weights, tmux sessions, logs, checkpoints and results are gone.
!!! Stop a run with 'tmux kill-session -t ilora', and before pausing run
!!!     bash scripts/archive_results.sh [--with-checkpoints]
!!! and download archives/<file>.tgz (experiments/VESSL_GUIDE.md, "Before a Pause").
EOF
