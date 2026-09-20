#!/usr/bin/env bash
# VESSL environment setup for the exp/gate-cluster experiment.
# Linux + a single A100 80GB GPU is assumed (spec section 14). This script is idempotent: running
# it twice should not corrupt the environment. It never trains anything and never downloads the
# Llama-2 model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== setup_vessl.sh ==="
echo "[1/6] OS check"
uname -a || echo "uname not available"
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "WARNING: this script is intended for the VESSL Linux instance; detected: $(uname -s)"
fi

echo "[2/6] Python check"
python --version
pip --version

echo "[3/6] GPU/CUDA check (informational only; scripts/preflight_vessl.sh enforces the hard gate)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "WARNING: nvidia-smi not found. This is expected on a CPU-only box but not on the VESSL A100 instance."
fi

echo "[4/6] Installing dependencies from requirements.txt (pinned versions, no upgrades applied)"
pip install -r requirements.txt

echo "[5/6] Checking documented transformers debug-file workaround (README.md)"
# The upstream README documents that some environments need transformers/generation/utils.py and
# transformers/models/llama/modeling_llama.py replaced with the debug/ versions shipped in this
# repo. We only apply the replacement if the installed file's hash differs from what is already
# installed AND from what we'd be replacing it with, and we always log the decision instead of
# silently overwriting a package file. See experiments/IMPLEMENTATION_NOTES.md section 10.
TRANSFORMERS_DIR="$(python -c 'import os, transformers; print(os.path.dirname(transformers.__file__))' 2>/dev/null || true)"
NOTES_FILE="experiments/IMPLEMENTATION_NOTES.md"
if [[ -n "$TRANSFORMERS_DIR" ]]; then
    GEN_UTILS="$TRANSFORMERS_DIR/generation/utils.py"
    MODELING_LLAMA="$TRANSFORMERS_DIR/models/llama/modeling_llama.py"
    for pair in "debug/utils.py:$GEN_UTILS" "debug/modeling_llama.py:$MODELING_LLAMA"; do
        SRC="${pair%%:*}"
        DST="${pair##*:}"
        if [[ -f "$DST" ]]; then
            if ! cmp -s "$SRC" "$DST"; then
                echo "INFO: $DST differs from $SRC. Not overwriting automatically."
                echo "      If you hit the error documented in README.md for this file, copy it manually:"
                echo "      cp $SRC $DST"
                {
                    echo ""
                    echo "### setup_vessl.sh dependency note ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
                    echo "- $DST differs from $SRC; not auto-replaced. See README.md for the documented workaround."
                } >> "$NOTES_FILE"
            else
                echo "INFO: $DST already matches $SRC, no action needed."
            fi
        else
            echo "INFO: $DST not found (transformers layout may differ); skipping."
        fi
    done
else
    echo "WARNING: could not locate the installed transformers package; skipping debug-file check."
fi

echo "[6/6] Creating experiment output directories"
mkdir -p checkpoints/baseline checkpoints/cluster_hard
mkdir -p outputs/baseline outputs/cluster_hard
mkdir -p results/baseline results/cluster_hard results/clustering
mkdir -p artifacts
mkdir -p logs

echo "=== Installed versions ==="
python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
try:
    import transformers; print('transformers', transformers.__version__)
except Exception as e:
    print('transformers: NOT_RUN', e)
try:
    import pytorch_lightning as pl; print('pytorch_lightning', pl.__version__)
except Exception as e:
    print('pytorch_lightning: NOT_RUN', e)
try:
    import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)
except Exception as e:
    print('bitsandbytes: NOT_RUN', e)
try:
    import sklearn; print('scikit-learn', sklearn.__version__)
except Exception as e:
    print('scikit-learn: NOT_RUN', e)
"

echo "setup_vessl.sh done."
