#!/usr/bin/env bash
# Heart suppl panel B. Logs go to <SSAFT_PANEL>/heart/logs/.
#   ./run_all.sh            the published map (k = 8, the cell table the figure used)
#   CELLS=colall ./run_all.sh          the later column_all re-segmentation (picks k = 7)
#   CELLS=colall K_FORCE=8 ./run_all.sh    that re-segmentation forced to 8 niches
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/bce/groups/pnucolab/analysis/l202255179/UNNAMED/.venv/bin/python}
export NUMPY_MADVISE_HUGEPAGE=0
LOG=$($PY -c 'import config as C; print(C.outdir(C.DS, "logs"))')
run() { echo "== $1 ${CELLS:-published}"; $PY -W ignore "$1" > "$LOG/${1%%_*}.log" 2>&1 || { tail -20 "$LOG/${1%%_*}.log"; exit 1; }; }
run 00_grid_counts.py
run 01_niches.py
run 02_panelB_niche_map.py
echo "done"
