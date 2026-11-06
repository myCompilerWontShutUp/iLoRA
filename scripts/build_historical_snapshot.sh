#!/usr/bin/env bash
# Build the 2024-08 ("paper-era") snapshot of iLoRA for the historical reproduction, in a separate
# detached git worktree (no branch is created; the current checkout is not touched).
#
# Snapshot = upstream AkaliKong/iLoRA main 28e4737 with exactly two files replaced:
#   data/data_interface.py          -> blob b69479a0, the file of 9f6c078 (2024-08-10) that 65e2fcd
#                                      (2024-11-13) replaced: training curriculum `if p > thresh_hold`
#                                      (about 90% hybrid batches per epoch) and text-only validation/test
#   model/peft/tuners/moelora.py    -> blob d839c104, this fork's memory rewrite (8049a98), algebraically
#                                      identical to upstream; without it upstream runs out of memory on
#                                      one A100 80GB at batch 8
# Every other MovieLens-relevant difference between the 2024-08 code (0a69c15) and 28e4737 only makes the
# 2024-08 code runnable: SASRecModules_ori.py (byte-identical to LLaRA's), optims.py (the scheduler API
# the 2024-08 model_interface.py calls), rec_model/movielens.pt (same bytes as SASRec_movielens.pt), the
# removal of an undefined calculate_ctr call at the end of testing (8eb6035), and debug/ (transformers
# patches, installed in the environment).
#
#   bash scripts/build_historical_snapshot.sh                  # -> ../ilora_hist_2024_08
#   bash scripts/build_historical_snapshot.sh /path/to/new/dir
# Then: python scripts/historical_preflight.py --snapshot <dir>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/../ilora_hist_2024_08}"
BASE=28e47371192121b6ced44401d9e149fedfae13de
HIST_BLOB=b69479a07433489ac583b79c40930500be802960       # data/data_interface.py, 9f6c078 .. 65e2fcd^
HIST_INTRODUCED=9f6c0781bf65c9ec6a148471169fbceb9b6ecb24
HIST_REPLACED_BY=65e2fcdee09b3a748d272615aad552a2a01c26f4
UPSTREAM_DI_BLOB=5f822c628751e5bbef8857a350e5812a4cc510e1  # data/data_interface.py at 28e4737
MOELORA_BLOB=d839c1045ae2f4c0fa75221cc64609e568fa84a8      # model/peft/tuners/moelora.py, fork 8049a98
MOELORA_FORK_COMMIT=8049a98
UPSTREAM_MOELORA_BLOB=6719d507ecbb214a4b8b6a535ca06de2d866dafc

fail() { echo "FATAL: $*" >&2; exit 1; }
git_repo() { git -C "$REPO_ROOT" "$@"; }

[[ -e "$DEST" ]] && fail "$DEST exists; the snapshot is always built in a new folder"
git_repo cat-file -e "$BASE^{commit}" 2>/dev/null || fail "commit $BASE is not in this repository (fetch upstream)"
for rev in "$HIST_INTRODUCED" "$HIST_REPLACED_BY^"; do
    [[ "$(git_repo rev-parse "$rev:data/data_interface.py")" == "$HIST_BLOB" ]] || fail "$rev:data/data_interface.py is not $HIST_BLOB"
done
[[ "$(git_repo rev-parse "$BASE:data/data_interface.py")" == "$UPSTREAM_DI_BLOB" ]] || fail "unexpected data_interface.py at $BASE"
[[ "$(git_repo rev-parse "$BASE:model/peft/tuners/moelora.py")" == "$UPSTREAM_MOELORA_BLOB" ]] || fail "unexpected moelora.py at $BASE"
[[ "$(git_repo rev-parse "$MOELORA_FORK_COMMIT:model/peft/tuners/moelora.py")" == "$MOELORA_BLOB" ]] || fail "unexpected moelora.py at $MOELORA_FORK_COMMIT"

git_repo worktree add --detach "$DEST" "$BASE" >/dev/null
git_repo cat-file blob "$HIST_BLOB" > "$DEST/data/data_interface.py"
git_repo cat-file blob "$MOELORA_BLOB" > "$DEST/model/peft/tuners/moelora.py"

[[ "$(git -C "$DEST" rev-parse HEAD)" == "$BASE" ]] || fail "worktree HEAD is not $BASE"
[[ "$(git -C "$DEST" hash-object data/data_interface.py)" == "$HIST_BLOB" ]] || fail "data_interface.py content check failed"
[[ "$(git -C "$DEST" hash-object model/peft/tuners/moelora.py)" == "$MOELORA_BLOB" ]] || fail "moelora.py content check failed"
CHANGED="$(git -C "$DEST" status --porcelain --untracked-files=all | sort)"
EXPECTED="$(printf ' M data/data_interface.py\n M model/peft/tuners/moelora.py' | sort)"
[[ "$CHANGED" == "$EXPECTED" ]] || fail "unexpected changes in the snapshot: $CHANGED"

cat > "$DEST/HISTORICAL_SNAPSHOT_MANIFEST.json" <<EOF
{
  "purpose": "2024-08 (pre-65e2fcd) iLoRA training/evaluation semantics for the paper-era reproduction",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "built_by": "scripts/build_historical_snapshot.sh",
  "builder_repository_commit": "$(git_repo rev-parse HEAD)",
  "base_commit": "$BASE",
  "base": "AkaliKong/iLoRA main",
  "replaced_files": [
    {"path": "data/data_interface.py", "upstream_blob": "$UPSTREAM_DI_BLOB", "snapshot_blob": "$HIST_BLOB",
     "source": "AkaliKong/iLoRA $HIST_INTRODUCED (2024-08-10), replaced upstream by $HIST_REPLACED_BY (2024-11-13)",
     "class": "semantics: 2024-08 training curriculum and text-only validation/test"},
    {"path": "model/peft/tuners/moelora.py", "upstream_blob": "$UPSTREAM_MOELORA_BLOB", "snapshot_blob": "$MOELORA_BLOB",
     "source": "myCompilerWontShutUp/iLoRA $MOELORA_FORK_COMMIT",
     "class": "execution: algebraically identical memory rewrite of the MoE-LoRA forward (fp16 rounding-level differences)"}
  ],
  "not_carried_over": ["upstream 65e2fcd curriculum change", "fork test drop_last removal (babe9ec)",
                       "fork bf16 SASRec mask fix (761dea8, not needed in fp16)", "fork gate export / measurement"]
}
EOF

echo "Built $DEST (detached at $BASE; replaced data/data_interface.py and model/peft/tuners/moelora.py)."
echo "Manifest: $DEST/HISTORICAL_SNAPSHOT_MANIFEST.json"
echo "Next (no training): python $REPO_ROOT/scripts/historical_preflight.py --snapshot $DEST"
echo "Remove later with: git -C $REPO_ROOT worktree remove $DEST"
