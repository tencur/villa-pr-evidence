#!/bin/bash
# 2026-11-20: #2466 revision (villa-review repairs). c552n/src -> upstream main c8f3ba453 (src_c8f3 = src_1386 + the 9
# volume-cartographer files changed 1386f072e..c8f3ba453, self-test wiring re-applied to apps/VC3D/CMakeLists.txt),
# reconfigure (same cache/flags as run18), build VC3D, ctest surface_rotation_apply per variant:test.
# Variants: r20 = new head (overlay + CWindow.cpp), af = PR head 3b3134af8 overlay files, main = main overlay files.
T=<work>; V=$T/variants; S=$T/src; B=$T/build; L=$T/run20.log
R=<work>/../vcbuild/vcr_py/deps/root/usr
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=$R/lib/aarch64-linux-gnu; mkdir -p $TMPDIR
disk_ok() { [ "$(df -k --output=avail /home | tail -1)" -gt 6291456 ] || { echo "DISK STOP $(date -u +%T)" >> $L; return 1; }; }
O=$S/volume-cartographer/apps/VC3D/overlays; C=$S/volume-cartographer/apps/VC3D/CWindow.cpp
echo "start $(date -u +%FT%TZ) runs: $*" >> $L
rsync -rc --delete --exclude="volume-cartographer/apps/VC3D/overlays/SurfaceRotationOverlayController.*" \
  --exclude="volume-cartographer/apps/VC3D/overlays/SurfaceRotationApplyTest.inc" \
  --exclude="volume-cartographer/apps/VC3D/CWindow.cpp" $T/src_c8f3/ $S/ || { echo "RSYNC FAILED" >> $L; exit 1; }
echo "src synced to c8f3ba453+wiring $(date -u +%FT%TZ)" >> $L
disk_ok || exit 9
flock <work>/../.build.lock env CMAKE_PREFIX_PATH=$R PKG_CONFIG_PATH=$R/lib/aarch64-linux-gnu/pkgconfig \
  cmake -S $S/volume-cartographer -B $B > $T/configure20.log 2>&1 || { echo "CONFIGURE FAILED" >> $L; tail -30 $T/configure20.log >> $L; exit 1; }
echo "configured $(date -u +%FT%TZ)" >> $L
for run in "$@"; do
  v=${run%%:*}; t=${run#*:}
  cw=$V/cw_base.cpp; [ $v = r20 ] && cw=$V/cw_r20.cpp
  cmp -s $cw $C || cp $cw $C
  cmp -s $V/$v.hpp $O/SurfaceRotationOverlayController.hpp || cp $V/$v.hpp $O/SurfaceRotationOverlayController.hpp
  { cat $V/$v.cpp; printf "\n#ifdef VC_TESTING\n#include \"SurfaceRotationApplyTest.inc\"\n#endif\n"; } > $T/tmp/ov.cpp
  cmp -s $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp || cp $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp
  cmp -s $V/$t.inc $O/SurfaceRotationApplyTest.inc || cp $V/$t.inc $O/SurfaceRotationApplyTest.inc
  disk_ok || exit 9
  flock <work>/../.build.lock nice -n 10 timeout 30000 ninja -C $B -j3 VC3D > $T/build20_${v}_${t}.log 2>&1; rc=$?
  echo "$run ninja exit $rc $(date -u +%FT%TZ)" >> $L
  [ $rc = 0 ] || { grep -E "error|FAILED" $T/build20_${v}_${t}.log | head -20 >> $L; continue; }
  (cd $B && timeout -s KILL 900 ctest -R "^surface_rotation_apply$" -V --timeout 300 > $T/ctest20_${v}_${t}.log 2>&1)
  echo "$run ctest exit $? $(grep -E "tests passed" $T/ctest20_${v}_${t}.log)" >> $L
  grep -E "case |FAIL|cases:|test:" $T/ctest20_${v}_${t}.log | sed "s/^[0-9]*: //" | head -20 >> $L
done
echo "done $(date -u +%FT%TZ)" >> $L
