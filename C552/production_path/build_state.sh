#!/bin/bash
# WS-C (2026-10-15): build VC3D + CLI tools in vcbuild/fsw4 for one source state and snapshot the binaries.
# usage: build_state.sh <tag> <mode> [commit]
#   mode main  : detached 5432721d (upstream main; volume-cartographer identical to a4979b99)
#   mode branch: detached <commit> (kit branch based on a4979b99)
#   mode apply : detached 5432721d + `git diff <commit>~1 <commit>` applied (kit branch on an older base)
# The whole checkout+build+snapshot runs under the shared build lock.
set -u
V=/home/pi/scrollprize_work/vcbuild; T=$V/fsw4; S=$T/src_allv2; B=$T/build_allv2; W=/home/pi/scrollprize_work/runs/WSC
tag=$1; mode=$2; c=${3:-}; L=$W/build_$tag.log
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T
exec 9>/home/pi/scrollprize_work/.build.lock
flock 9
echo "start $(date -u +%FT%TZ) tag=$tag mode=$mode commit=$c fsw4 HEAD was $(git -C $S rev-parse HEAD) $(git -C $S symbolic-ref -q --short HEAD)" >> $L
[ -f $W/fsw4_orig_head ] || { git -C $S rev-parse HEAD > $W/fsw4_orig_head; git -C $S symbolic-ref -q --short HEAD >> $W/fsw4_orig_head; }
git -C $S reset -q --hard && git -C $S clean -qfd volume-cartographer || exit 1
case $mode in
  main)   git -C $S checkout -q --detach 5432721d ;;
  branch) git -C $S checkout -q --detach $c ;;
  apply)  git -C $S checkout -q --detach 5432721d && git -C $S diff $c~1 $c | git -C $S apply --index - ;;
esac || { echo CHECKOUT FAILED >> $L; exit 1; }
echo "src $(git -C $S rev-parse --short HEAD) changes: $(git -C $S status --short | tr "\n" " ")" >> $L
nice -n 10 ninja -C $B -j2 VC3D vc_grow_seg_from_seed >> $L 2>&1; rc=$?
echo "ninja exit $rc $(date -u +%FT%TZ)" >> $L
[ $rc -eq 0 ] || exit $rc
D=$W/snap/$tag; rm -rf $D; mkdir -p $D/bin $D/core $D/utils $D/libs/c3d $D/libs/volcomp
cp $B/bin/VC3D $B/bin/vc_grow_seg_from_seed $D/bin/; cp $B/core/*.so $D/core/; cp $B/utils/*.so $D/utils/
cp $B/libs/c3d/libc3d.so $D/libs/c3d/; cp $B/libs/volcomp/libvolcomp.so $D/libs/volcomp/
(cd $D && sha256sum bin/* core/*.so > SHA256SUMS)
echo "snapshot $D done $(date -u +%FT%TZ)" >> $L
