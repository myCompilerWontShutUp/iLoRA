#!/usr/bin/env bash
# Preflight checks before any actual training. Never starts training itself.
# Exits non-zero (before training could ever start) if GPU count != 1, per spec section 15
# (cost-accident prevention: this experiment only ever uses a single GPU).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${LLM_PATH:?LLM_PATH must be set, e.g. export LLM_PATH=/path/to/Llama-2-7b-hf}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
DATA_DIR="${DATA_DIR:-data/ref/movielens}"
OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-outputs/baseline}"

echo "=== scripts/preflight_vessl.sh ==="

echo "[uname]"
uname -a

echo "[nvidia-smi]"
if ! nvidia-smi; then
    echo "FATAL: nvidia-smi failed. A GPU is required; aborting before any training." >&2
    exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "[GPU count] $GPU_COUNT"
if [[ "$GPU_COUNT" -ne 1 ]]; then
    echo "FATAL: expected exactly 1 GPU, found $GPU_COUNT. Aborting before any training (spec section 15/25.10)." >&2
    exit 1
fi

echo "[GPU model / VRAM]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[CUDA_VISIBLE_DEVICES] $CUDA_VISIBLE_DEVICES"

echo "[Python] $(python --version)"
echo "[pip] $(pip --version)"

python - <<'PYEOF'
import sys
def show(name, getter):
    try:
        print(f"[{name}]", getter())
    except Exception as e:
        print(f"[{name}] NOT_RUN ({e})")

show("torch version", lambda: __import__("torch").__version__)
show("torch.cuda.is_available()", lambda: __import__("torch").cuda.is_available())
show("torch.cuda.device_count()", lambda: __import__("torch").cuda.device_count())
show("transformers version", lambda: __import__("transformers").__version__)
show("peft (vendored model/peft) version", lambda: __import__("model.peft", fromlist=["__version__"]).__version__)
show("pytorch_lightning version", lambda: __import__("pytorch_lightning").__version__)
show("bitsandbytes version", lambda: __import__("bitsandbytes").__version__)
PYEOF

echo "[LLM_PATH] $LLM_PATH"
if [[ ! -e "$LLM_PATH" ]]; then
    echo "FATAL: LLM_PATH does not exist: $LLM_PATH. Aborting before any training." >&2
    exit 1
fi

echo "[dataset] $DATA_DIR"
for f in train_data.df Val_data.df Test_data.df u.item; do
    if [[ ! -e "$DATA_DIR/$f" ]]; then
        echo "FATAL: expected dataset file missing: $DATA_DIR/$f. Aborting before any training." >&2
        exit 1
    fi
done
echo "  all expected MovieLens files present."

echo "[free disk space]"
df -h . || true

mkdir -p "$OUTPUT_DIR"
if [[ -w "$OUTPUT_DIR" ]]; then
    echo "[output directory] $OUTPUT_DIR is writable."
else
    echo "FATAL: output directory $OUTPUT_DIR is not writable." >&2
    exit 1
fi

echo ""
echo "Preflight passed. No training has started."
