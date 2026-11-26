#!/bin/bash
# usage: run_tests.sh <tree> <basetemp> [pytest args...]
# Runs pytest on <tree>/vesuvius/src with the napari venv; torch comes from the second venv via
# site.addsitedir (not PYTHONPATH: an argparse package there would shadow the stdlib). VC_READER: built vc_reader dir.
set -u
TREE=$(readlink -f "$1"); BT=$2; shift 2
export PYTHONPATH=$TREE/vesuvius/src:${VC_READER:?}${EXTRA_PATH:+:$EXTRA_PATH} QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
rm -rf "$BT"; mkdir -p "$BT"
cd "$TREE/vesuvius" || exit 1
${PYN:?} -c "import site, sys; site.addsitedir(sys.argv[1]); import pytest; sys.exit(pytest.main(sys.argv[2:]))" \
  "${TORCH_SITE:?}" -p no:cacheprovider --basetemp="$BT" -q -rA "$@"
rc=$?
rm -rf "$BT"
exit $rc
