#!/bin/bash
# usage: run_session.sh <tree> <work_dir> train [config.yaml] | infer <checkpoint.pth> [config.yaml]
# <tree> is a villa checkout (vesuvius/src inside); VC_READER is the dir holding a built vc_reader;
# PY is a python with torch, napari, magicgui and pyqt6.
set -u
HERE=$(dirname "$(readlink -f "$0")")
TREE=$(readlink -f "$1"); WORK_DIR=$2; MODE=$3; shift 3
ARGS=(); for a in "$@"; do ARGS+=("$(readlink -f "$a")"); done
export PYTHONPATH=$TREE/vesuvius/src:${VC_READER:?}
export QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
mkdir -p "$WORK_DIR" && cd "$WORK_DIR" || exit 1
exec ${PY:?} "$HERE/napari_session.py" "$MODE" "${ARGS[@]}"
