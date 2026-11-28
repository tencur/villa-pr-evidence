#!/bin/sh
# Usage: run_repro.sh <exported tree dir of a revision> <label> <scratch dir> <log dir>
# (tree dirs as exported by run_tests.sh, e.g. <work>/merge). vc_reader must be on PYTHONPATH.
# 1. The revision_1122 reproduction suite (../../revision_1122/scripts/run_all.sh, unchanged).
# 2. check_axis_heads_patch_cache.py: main's batched patch cache with an axis head (merge only).
# 3. review_peak_memory.py again on the trees one after another, to compare process RSS in one session.
set -u
TREE=$1; LABEL=$2; WORK=$3; LOGS=$4
HERE=$(cd "$(dirname "$0")" && pwd)
R1122="$HERE/../../revision_1122/scripts"
PY=${PYTHON:-python}
mkdir -p "$WORK" "$LOGS"
sh "$R1122/run_all.sh" "$TREE/vesuvius" "$LABEL" "$WORK/repro" > "$LOGS/repro_$LABEL.log" 2>&1
PYTHONPATH="$TREE/vesuvius/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$HERE/check_axis_heads_patch_cache.py" "$WORK/check" \
    > "$LOGS/check_axis_heads_patch_cache_$LABEL.log" 2>&1
# Peak RSS rerun (bot, main, merge back to back):
#   for t in bot main merge; do PYTHONPATH="<work>/$t/vesuvius/src:$PYTHONPATH" "$PY" "$R1122/review_peak_memory.py" \
#       | grep -E "^(one|batched|sequential)"; done
