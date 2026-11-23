#!/bin/bash
# Direct fiber inference (README command) on synthetic constant volumes, main vs this PR.
# MAIN / FIX: villa checkouts (upstream main, and main with this PR); VCR: dir holding a built vc_reader.
# WORK: scratch dir. Logs go to scripts/logs/coverage_<tree>_<case>[_<TAG>].log (progress bars dropped).
set -u
: "${MAIN:?}" "${FIX:?}" "${VCR:?}" "${WORK:?}"
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
run() {  # tree-name tree-dir case size extra...
  local name=$1 tree=$2 case=$3 size=$4; shift 4
  PYTHONPATH=$tree/vesuvius/src:$tree:$tree/lasagna:$VCR \
    $PY "$HERE/default_coverage.py" "$WORK" "${name}_${case}" "$size" "$@" 2>&1 \
    | grep -v -E '^\[(pyramid|tile|fiber_trace_3d:infer\] (input|crop|removed|loading|model|starting))|eta ' \
    | sed "s#$WORK#<work>#g; s#$tree#<villa>#g; s#$HERE/#scripts/#g; s#$HOME#<home>#g" > "$HERE/logs/coverage_${name}_${case}${TAG:+_$TAG}.log"
  echo "== ${name} ${case}"; grep -E 'infer argv|presence 0|planes|^validity|presence-0|Error' "$HERE/logs/coverage_${name}_${case}${TAG:+_$TAG}.log"
}
run main "$MAIN" default160 160
run main "$MAIN" default256 256
run main "$MAIN" overlap32_160 160 --overlap 32 --border 16
run fix "$FIX" default160 160
run fix "$FIX" default256 256
run fix "$FIX" overlap16_160 160 --overlap 16 --border 16
