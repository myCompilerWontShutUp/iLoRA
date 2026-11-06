#!/usr/bin/env bash
# Pack everything that has to survive a VESSL workspace Pause/Stop into one archive.
#
# A Pause/Stop recreates the container: the repository, the conda env, the weights, tmux sessions,
# logs, checkpoints and results under /root are gone unless they are on a persistent volume. Run
# this before pausing and download the archive (JupyterLab file browser, or scp).
#
#   bash scripts/archive_results.sh                  # logs log benchmark_runs results outputs artifacts runs
#   bash scripts/archive_results.sh --with-checkpoints
#   bash scripts/archive_results.sh --dest /path/to/existing/persistent/dir   # also copy it there
#   bash scripts/archive_results.sh --dry-run        # list what would be packed
#
# Writes archives/ilora_results_<host>_<UTC time>.tgz and a .manifest.txt next to it (git commit,
# folder sizes, file count, sha256; the manifest is also inside the archive). Nothing is uploaded,
# nothing is deleted, an existing archive is never overwritten, and --dest must be a directory that
# already exists (this script never guesses a storage location). Model weights are not included
# (re-downloadable); checkpoints only with --with-checkpoints.
set -euo pipefail

ROOT="${ARCHIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

DIRS=(logs log benchmark_runs results outputs artifacts runs)
DEST=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-checkpoints) DIRS+=(checkpoints); shift ;;
        --dest) DEST="${2:?--dest needs a directory}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$DEST" && ! -d "$DEST" ]]; then
    echo "FATAL: --dest $DEST is not an existing directory; create or mount it yourself first." >&2
    exit 1
fi

PRESENT=()
for d in "${DIRS[@]}"; do
    if [[ -e "$d" ]]; then PRESENT+=("$d"); fi
done
if [[ ${#PRESENT[@]} -eq 0 ]]; then
    echo "Nothing to archive in $ROOT (none of: ${DIRS[*]})."
    exit 0
fi

echo "Folders to archive from $ROOT:"
du -sh "${PRESENT[@]}" 2>/dev/null | sed 's/^/  /'
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(dry run: nothing written)"
    exit 0
fi

mkdir -p archives
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="ilora_results_$(hostname | tr -c 'A-Za-z0-9._-' '_')_${STAMP}"
BASE="$NAME"
i=1
while [[ -e "archives/$NAME.tgz" ]]; do NAME="$BASE-$i"; i=$((i + 1)); done
MANIFEST="archives/$NAME.manifest.txt"
{
    echo "archive: $NAME.tgz"
    echo "created_utc: $STAMP"
    echo "host: $(hostname)"
    echo "repository: $ROOT"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "folders:"
    du -sh "${PRESENT[@]}" 2>/dev/null | sed 's/^/  /'
    echo "files: $(find "${PRESENT[@]}" -type f | wc -l)"
} > "$MANIFEST"
tar czf "archives/$NAME.tgz" "${PRESENT[@]}" "$MANIFEST"
SHA="$( (sha256sum "archives/$NAME.tgz" 2>/dev/null || shasum -a 256 "archives/$NAME.tgz") | cut -d' ' -f1)"
echo "sha256: $SHA" >> "$MANIFEST"

echo ""
echo "Wrote $ROOT/archives/$NAME.tgz ($(du -h "archives/$NAME.tgz" | cut -f1)), manifest $ROOT/$MANIFEST"
if [[ -n "$DEST" ]]; then
    cp -n "archives/$NAME.tgz" "$MANIFEST" "$DEST"/
    echo "Copied to $DEST/"
fi
echo "Download it before pausing: JupyterLab file browser -> archives/ -> right click -> Download,"
echo "or from your machine: scp -P <port> root@<host>:$ROOT/archives/$NAME.tgz ."
echo "Check after downloading: sha256 $SHA"
