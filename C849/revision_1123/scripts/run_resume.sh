#!/bin/bash
# The review's resume case: an output written by main (default overlap 16), one finest chunk
# removed, then the same command rerun with the PR's code. Also the compatible retry on the new head.
# MAIN / OLD / NEW: villa checkouts (upstream main, the PR's first commit, the revised PR head);
# VCR: dir holding a built vc_reader; WORK: scratch dir. Logs go to logs/resume_<step>.log next to scripts/.
set -u
: "${MAIN:?}" "${OLD:?}" "${NEW:?}" "${VCR:?}" "${WORK:?}"
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(cd "$HERE/.." && pwd)/logs; mkdir -p "$LOGS"
step() {  # log-name tree-dir output-dir-name extra...
  local name=$1 tree=$2 out=$3; shift 3
  PYTHONPATH=$tree/vesuvius/src:$tree:$tree/lasagna:$VCR \
    $PY "$HERE/resume_tiling.py" "$WORK" "$out" "$@" 2>&1 \
    | grep -v -E '^\[(pyramid|tile|predict3d|input|fiber_trace_3d:infer\] (input|crop|removed|loading|model|starting|inference_precision))|eta ' \
    | sed "s#$WORK#<work>#g; s#$tree#<villa>#g; s#$HERE/#scripts/#g; s#$HOME#<home>#g" > "$LOGS/resume_${name}.log"
  echo "== ${name}"; grep -E '^(removed|infer argv|result|finest|store|inference.json|voxels with presence 0|validity)' "$LOGS/resume_${name}.log"
}
rm -rf "$WORK/main_out" "$WORK/old_retry" "$WORK/new_retry" "$WORK/new_out"
step 1_main_fresh "$MAIN" main_out                     # main, README command, default tiling
cp -a "$WORK/main_out" "$WORK/old_retry"
cp -a "$WORK/main_out" "$WORK/new_retry"
step 2_old_head_rerun "$OLD" old_retry --remove-chunk  # the PR's first commit, same command
step 3_new_head_rerun "$NEW" new_retry --remove-chunk  # revised head, same command
step 4_new_head_fresh "$NEW" new_out                   # revised head, fresh output
step 5_new_head_retry "$NEW" new_out --remove-chunk    # revised head, same-tiling retry
