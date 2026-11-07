#!/bin/bash
# rd_sides_sw4f.sh <id> <branch> "<fixed files under volume-cartographer/>" "<driver targets>" [driver cmake file]
# Fix lane SW4F-CPP real-data helper (kit only): builds the driver targets on <branch> (fix side) and on the same
# tree with only the fixed files reverted to $BASE (main side); the optional driver cmake file is appended to
# apps/VC3D/CMakeLists.txt for the build only. Binaries go to runs/SW4FC/<id>/{fix,main}/bin (the VC3D objects
# are linked into the driver statically; shared VC libraries are the same on both sides). Restores the branch.
set -e
F=/home/pi/scrollprize_work/vcbuild/fsw4; S=$F/src_allv2; B=$F/build_allv2
id=$1; br=$2; files=$3; targets=$4; dcm=${5:-}; BASE=${BASE:-a4979b990}
R=/home/pi/scrollprize_work/runs/SW4FC/$id; mkdir -p $R
add_driver() { [ -z "$dcm" ] || cat $dcm >> $S/volume-cartographer/apps/VC3D/CMakeLists.txt; }
copy_side() {
  local d=$R/$1; rm -rf $d; mkdir -p $d/bin
  for b in $targets; do cp $B/bin/$b $d/bin/; done
  echo "$1: $(git -C $S rev-parse --short HEAD) $(git -C $S status --short | tr '\n' ' ')" >> $R/sides.txt
}
: > $R/sides.txt
git -C $S checkout -q --detach -f $br; add_driver
$F/build.sh rd_fix_$id $targets
copy_side fix
for f in $files; do git -C $S checkout $BASE -- volume-cartographer/$f; done
$F/build.sh rd_main_$id $targets
copy_side main
git -C $S checkout -q --detach -f $br
echo done
