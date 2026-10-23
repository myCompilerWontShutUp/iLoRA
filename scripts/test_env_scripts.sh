#!/usr/bin/env bash
# Local (no GPU, no conda needed) regression tests of the VESSL environment scripts:
#   1. bash -n of every shell script
#   2. scripts/env_check.py helpers: container-filesystem detection, LD_LIBRARY_PATH entries that
#      would make tmux load conda's libtinfo; --storage-only runs with any Python
#   3. scripts/setup_bnb_cuda_runtime.sh in a fake env: idempotent link and hooks, LD_LIBRARY_PATH
#      gets only the nvidia-* dirs (never $CONDA_PREFIX/lib, even when it holds libtinfo), and 10
#      activate/deactivate cycles neither stack entries nor leave anything behind
#   4. scripts/archive_results.sh: dry run writes nothing, archives never overwrite each other,
#      sources are kept, --dest must exist
#   5. static checks: conda create only with conda-forge + --override-channels + -y (no ToS
#      prompt), non-interactive apt, no script puts $CONDA_PREFIX/lib on LD_LIBRARY_PATH
# What only VESSL can show (real conda ToS behaviour, the real bitsandbytes binary, tmux attach /
# detach in a terminal) is listed in experiments/VESSL_GUIDE.md, "Environment problems seen on VESSL".
#
#     bash scripts/test_env_scripts.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=""
for c in python3 python; do   # the first that really runs (Windows has a python3 store stub)
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
FAILS=0
ok() { echo "  [OK] $*"; }
bad() { echo "  [FAIL] $*"; FAILS=$((FAILS + 1)); }
check() { if eval "$1"; then ok "$2"; else bad "$2"; fi; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "1. shell syntax"
for f in "$REPO"/*.sh "$REPO"/scripts/*.sh; do
    if bash -n "$f"; then ok "bash -n $(basename "$f")"; else bad "bash -n $(basename "$f")"; fi
done

echo "2. env_check helpers"
export ENV_CHECK_DIR="$REPO/scripts"
mkdir -p "$TMP/nc/with" "$TMP/nc/clean" && touch "$TMP/nc/with/libtinfo.so.6"
if (cd "$TMP/nc" && "$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ['ENV_CHECK_DIR'])
import env_check as e
root = e.parse_df('Filesystem Type 1024-blocks Used Available Capacity Mounted on\noverlay overlay 2097152000 958398464 1138753536 46% /\n')
vol = e.parse_df('Filesystem Type 1024-blocks Used Available Capacity Mounted on\n/dev/nvme1n1 ext4 104857600 1048576 103809024 1% /root/persist\n')
assert root['type'] == 'overlay' and root['mount'] == '/' and root['avail_bytes'] == 1138753536 * 1024
assert e.is_ephemeral(root) and not e.is_ephemeral(vol) and vol['mount'] == '/root/persist'
assert e.is_ephemeral({'type': 'ext4', 'mount': '/', 'filesystem': 'x', 'avail_bytes': 0})
assert e.ncurses_dirs('clean:with:nonexistent') == ['with'] and e.ncurses_dirs('') == []   # relative: no drive colons
print('helpers ok')
PYEOF
); then ok "parse_df / is_ephemeral / ncurses_dirs"; else bad "env_check helpers"; fi
check '"$PY" "$REPO/scripts/env_check.py" --storage-only > "$TMP/storage.txt" 2>&1 && grep -q "storage repository" "$TMP/storage.txt"' "env_check.py --storage-only runs and reports the repository"

echo "3. bitsandbytes runtime link and activation hooks (fake env)"
E="$TMP/env"; SP="$E/lib/python3.10/site-packages"
mkdir -p "$TMP/bin" "$SP/bitsandbytes" "$E/lib"
for d in cuda_runtime cublas cudnn cusparse; do mkdir -p "$SP/nvidia/$d/lib"; done
echo fake > "$SP/nvidia/cuda_runtime/lib/libcudart.so.11.0"
touch "$SP/bitsandbytes/libbitsandbytes_cuda117.so" "$SP/bitsandbytes/__init__.py" "$E/lib/libtinfo.so.6"
cat > "$TMP/bin/python" <<EOF
#!/usr/bin/env bash
cat >/dev/null
printf '%s\n' "PY_PREFIX='$E'" "TORCH_VERSION='2.0.0+cu117'" "TORCH_CUDA='11.7'" "SITE_PACKAGES='$SP'" "BNB_DIR='$SP/bitsandbytes'"
EOF
chmod +x "$TMP/bin/python"
export MSYS=winsymlinks:lnk   # Git Bash on Windows only; ignored on Linux
SETUP="$REPO/scripts/setup_bnb_cuda_runtime.sh"
for i in 1 2 3; do PATH="$TMP/bin:$PATH" CONDA_PREFIX="$E" bash "$SETUP" > "$TMP/setup_$i.txt" 2>&1 || bad "setup run $i failed: $(tail -2 "$TMP/setup_$i.txt")"; done
check '[[ -L "$E/lib/libcudart.so" && "$(readlink "$E/lib/libcudart.so")" == "$SP/nvidia/cuda_runtime/lib/libcudart.so.11.0" ]]' "libcudart.so -> the wheel's CUDA 11.7 runtime"
check 'grep -q "already points to it" "$TMP/setup_3.txt"' "3rd run keeps the link (idempotent)"
H="$E/etc/conda/activate.d/ilora_bnb_cuda.sh"; DH="$E/etc/conda/deactivate.d/ilora_bnb_cuda.sh"
cp "$H" "$TMP/hook_after_run3"; PATH="$TMP/bin:$PATH" CONDA_PREFIX="$E" bash "$SETUP" >/dev/null 2>&1
check 'cmp -s "$H" "$TMP/hook_after_run3"' "hook file identical across runs"
ORIG="/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
RESULT="$(CONDA_PREFIX="$E" LD_LIBRARY_PATH="$ORIG" bash -c '
    set -u
    first=""
    for i in $(seq 10); do
        source "$1"; [[ -z "$first" ]] && first="$LD_LIBRARY_PATH"
        [[ "$LD_LIBRARY_PATH" == "$first" ]] || { echo "stacked at $i"; exit 1; }
        source "$2"
        [[ "$LD_LIBRARY_PATH" == "$3" ]] || { echo "not restored at $i: $LD_LIBRARY_PATH"; exit 1; }
    done
    source "$1"; echo "$LD_LIBRARY_PATH"' _ "$H" "$DH" "$ORIG")"
check '[[ "$RESULT" == "$SP/nvidia/"* && "$RESULT" == *":$ORIG" ]]' "10 activate/deactivate cycles: no stacking, restored each time, nvidia dirs first"
check '[[ ":$RESULT:" != *":$E/lib:"* ]]' "\$CONDA_PREFIX/lib (with libtinfo.so.6) is not on LD_LIBRARY_PATH"
check '[[ "$(cd "$REPO/scripts" && "$PY" -c "import sys, env_check as e; print(e.ncurses_dirs(sys.argv[1]))" "$RESULT")" == "[]" ]]' "no libtinfo/libncurses directory on the activated LD_LIBRARY_PATH"
UNSET="$(CONDA_PREFIX="$E" bash -c 'unset LD_LIBRARY_PATH; source "$1"; source "$2"; echo "${LD_LIBRARY_PATH-UNSET}"' _ "$H" "$DH")"
check '[[ "$UNSET" == "UNSET" ]]' "an unset LD_LIBRARY_PATH is unset again after deactivate"

echo "4. archive_results.sh"
AR="$TMP/repo"; mkdir -p "$AR/logs" "$AR/results/baseline" "$AR/runs/r1" "$AR/checkpoints/baseline"
echo log > "$AR/logs/a.log"; echo m > "$AR/results/baseline/metrics.json"; echo x > "$AR/runs/r1/meta.json"; echo w > "$AR/checkpoints/baseline/last.ckpt"
ARCHIVE_ROOT="$AR" bash "$REPO/scripts/archive_results.sh" --dry-run > "$TMP/dry.txt" 2>&1
check '[[ ! -d "$AR/archives" ]] && grep -q "dry run" "$TMP/dry.txt"' "--dry-run writes nothing"
ARCHIVE_ROOT="$AR" bash "$REPO/scripts/archive_results.sh" > "$TMP/a1.txt" 2>&1
ARCHIVE_ROOT="$AR" bash "$REPO/scripts/archive_results.sh" --with-checkpoints > "$TMP/a2.txt" 2>&1
N="$(ls "$AR/archives"/*.tgz 2>/dev/null | wc -l)"
check '[[ "$N" -eq 2 ]]' "two runs, two distinct archives (got $N)"
FIRST="$(ls "$AR/archives"/*.tgz | head -1)"; LAST="$(ls "$AR/archives"/*.tgz | tail -1)"
(cd "$AR/archives" && tar tzf "$(basename "$FIRST")" > "$TMP/first.txt" && tar tzf "$(basename "$LAST")" > "$TMP/last.txt")
check 'grep -q "results/baseline/metrics.json" "$TMP/first.txt" && grep -q "runs/r1/meta.json" "$TMP/first.txt" && ! grep -q checkpoints "$TMP/first.txt"' "default archive: logs/results/runs, no checkpoints"
check 'grep -q "checkpoints/baseline/last.ckpt" "$TMP/last.txt" && grep -q "manifest.txt" "$TMP/last.txt"' "--with-checkpoints adds checkpoints; manifest inside"
check '[[ -f "$AR/logs/a.log" && -f "$AR/checkpoints/baseline/last.ckpt" ]]' "sources kept"
check 'grep -q "^sha256: " "${FIRST%.tgz}.manifest.txt"' "manifest has the sha256"
check '! ARCHIVE_ROOT="$AR" bash "$REPO/scripts/archive_results.sh" --dest "$TMP/does-not-exist" >/dev/null 2>&1' "--dest must be an existing directory"
mkdir -p "$TMP/dest"
ARCHIVE_ROOT="$AR" bash "$REPO/scripts/archive_results.sh" --dest "$TMP/dest" >/dev/null 2>&1
check '[[ "$(ls "$TMP/dest"/*.tgz | wc -l)" -eq 1 ]]' "--dest copies the archive there"

echo "5. static checks"
B="$REPO/bootstrap_vessl.sh"
check 'grep -E "^\s*conda create" "$B" | grep -q -- "--override-channels" && grep -E "^\s*conda create" "$B" | grep -q -- "-c conda-forge" && grep -E "^\s*conda create" "$B" | grep -q -- " -y"' "conda create uses only conda-forge, --override-channels, -y"
check '[[ "$(grep -cE "^\s*conda create" "$B")" -eq 1 ]]' "a single conda create"
check 'grep -q "DEBIAN_FRONTEND=noninteractive" "$B"' "apt runs non-interactively"
check 'grep -q "PYTHONNOUSERSITE=1" "$B"' "user site-packages disabled during bootstrap"
check '! grep -rnE "LD_LIBRARY_PATH=[^#]*CONDA_PREFIX/lib" "$REPO"/*.sh "$REPO"/scripts/*.sh' "no script puts \$CONDA_PREFIX/lib on LD_LIBRARY_PATH"
check 'grep -q "bash scripts/preflight_vessl.sh" "$REPO/run_baseline_movielens.sh" && grep -q "bash scripts/preflight_vessl.sh" "$REPO/run_cluster_hard_movielens.sh"' "both run scripts start with the preflight"
check 'grep -q "scripts/env_check.py" "$REPO/scripts/preflight_vessl.sh"' "preflight runs env_check.py"

if [[ "$FAILS" -ne 0 ]]; then
    echo "$FAILS check(s) failed"
    exit 1
fi
echo "All environment-script checks passed."
