#!/bin/sh
# Usage: run_all.sh <vesuvius dir> <label> <scratch dir>
# Runs the review reproductions against the vesuvius package in <vesuvius dir> (its src/ is put
# first on PYTHONPATH; vc_reader must be importable) and prints each script's output.
set -u
VDIR=$1; LABEL=$2; WORK=$3
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$VDIR/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-python}
mkdir -p "$WORK"
echo "## $LABEL: repro_tta_vector_heads.py (48^3, real Inferer + in-memory blend)"
$PY "$HERE/repro_tta_vector_heads.py" "$WORK/repro_$LABEL" 2>&1 | grep -E "^(voxels|no TTA|TTA )"
echo "## $LABEL: review_seam_fixture.py"
$PY "$HERE/review_seam_fixture.py" "$WORK/seam_$LABEL" 2>&1 | grep -E "^(Gaussian weight|mirroring|no TTA)"
echo "## $LABEL: review_multiclass_cli.py"
$PY "$HERE/review_multiclass_cli.py" "$WORK/multiclass_$LABEL" 2>&1 | grep -E "^case "
echo "## $LABEL: review_fp16_small_direction.py"
$PY "$HERE/review_fp16_small_direction.py" 2>&1 | grep -E "^(product|batched|sequential|\(this)"
echo "## $LABEL: review_peak_memory.py"
$PY "$HERE/review_peak_memory.py" 2>&1 | grep -E "^(one|batched|sequential)"
