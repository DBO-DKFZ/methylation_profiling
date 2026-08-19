#!/usr/bin/env bash
# Build the Apptainer images locally.
# Run from anywhere; paths are resolved relative to the repo root so the
# python.def %files block can reach python/pyproject.toml and python/uv.lock.
#
# Usage:
#   ./apptainer/build.sh            # build both
#   ./apptainer/build.sh r          # build only r.sif
#   ./apptainer/build.sh python     # build only python.sif

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"   # build context for %files

target="${1:-all}"

build() {
  local name="$1"
  echo "=== building apptainer/${name}.sif ==="
  apptainer build --fakeroot "apptainer/${name}.sif" "apptainer/${name}.def"
}

case "$target" in
  r)      build r ;;
  python) build python ;;
  all)    build r; build python ;;
  *) echo "Unknown target '$target' (use: r | python | all)" >&2; exit 1 ;;
esac

echo "Done."
