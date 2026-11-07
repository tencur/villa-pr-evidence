#!/bin/bash
# C552 real-data proof (not part of the PR). c552_realdata_driver.cpp is built twice with VC3D's objects: from this
# branch (fix/) and from the same tree with only apps/VC3D/overlays/SurfaceRotationOverlayController.cpp reverted to
# main a4979b99 (main/). Data: the public PHerc0800 auto-grown segment 20251028220955 (PHerc0800/segments/ on the
# open-data bucket, tifxyz), copied into a fresh project paths/ folder per run (meta uuid set to the folder name).
R=$(cd $(dirname $0) && pwd); MESH=${MESH:-/home/pi/scrollprize_work/runs/READ1/jobC/p0800/mesh}
ID=20251028220955
for side in ${SIDES:-main fix}; do
  d=$R/work_$side; rm -rf $d; mkdir -p $d/paths $d/home
  cp -r $MESH/$ID-*.tifxyz $d/paths/$ID
  python3 - $d/paths/$ID/meta.json <<'PY'
import json, os, sys
p = sys.argv[1]; j = json.load(open(p)); j["uuid"] = os.path.basename(os.path.dirname(p)); json.dump(j, open(p, "w"), indent=2)
PY
  echo "== C552 on $side"
  BIN=${BIN_DIR:-$R/$side/bin}; export LD_LIBRARY_PATH=${LIBS:-$R/libs}   # VC shared libraries of the build (same on both sides)
  (cd $d && HOME=$d/home QT_QPA_PLATFORM=offscreen timeout 300 $BIN/c552_realdata_driver $d $ID 2>&1) | grep -E "^RESULT|Processing|Updated|Removed|Added" | sed -e "s/^RESULT /  /"
  rm -rf $d
done
