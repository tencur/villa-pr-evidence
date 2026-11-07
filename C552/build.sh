#!/bin/bash
# fix lane SW4-CPP (2026-10-13): fsw4/src_allv2 = worktree of repos/villa_allv2 at bcdf9bf5, fix branches checked out in turn.
# Same flags as c122r/build.sh; CCACHE_BASEDIR mirrors src_allv2/build_allv2. Usage: build.sh <logtag> <targets...>
V=/home/pi/scrollprize_work/vcbuild; T=$V/fsw4; tag=$1; shift
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T; L=$T/build_$tag.log
echo "start $(date -u +%FT%TZ) $(git -C $T/src_allv2 rev-parse --short HEAD) targets: $*" >> $L
[ -f $T/build_allv2/build.ninja ] || cmake -S $T/src_allv2/volume-cartographer -B $T/build_allv2 -G Ninja -DCMAKE_BUILD_TYPE=QuickBuild -DVC_QUICKBUILD_OPT_LEVEL=2 \
   -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBLA_VENDOR=OpenBLAS -DVC_BUILD_APPS=ON -DVC_BUILD_FLATBOI=OFF \
   -DVC_BUILD_PYTHON=OFF -DVC_TESTING=ON -DVC_USE_CCACHE=ON >> $L 2>&1 || { echo CONFIGURE FAILED >> $L; exit 1; }
flock /home/pi/scrollprize_work/.build.lock nice -n 10 ninja -C $T/build_allv2 -j2 "$@" >> $L 2>&1; rc=$?
echo "ninja exit $rc $(date -u +%FT%TZ)" >> $L; exit $rc
