#!/bin/bash
# 2026-11-22: #2466 revision 2 (villa-review re-review). Same c552n tree as run20.sh: src = upstream main c8f3ba453
# (src_c8f3) + self-test wiring, existing CMake cache (QuickBuild, VC_QUICKBUILD_OPT_LEVEL=2, VC_TESTING=ON, g++).
# Per run "variant:test[:repeat]" the 8 files below are swapped in from variants/v22/<variant>/ (overlay .cpp with the
# self-test .inc appended), then `ninja -j3 VC3D` and `ctest -R ^surface_rotation_apply$` (offscreen).
# Variants: r22 = new head, r22a = first new commit only, r20 = b860d5aa0 (PR head), main = c8f3ba453.
# Tests: test_ext22 = SurfaceRotationApplyTest_ext22.inc; test_ext18a / test_orig18a = the 18 Nov self-tests with the
# saver lambda taking the new second argument (sed, nothing else changed) for r22; test_ext18 / test_orig18 unchanged.
T=<work>; V=$T/variants; S=$T/src; B=$T/build; L=$T/run22.log
R=<work>/../vcr_py/deps/root/usr
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=$R/lib/aarch64-linux-gnu; mkdir -p $TMPDIR
disk_ok() { [ "$(df -k --output=avail /home | tail -1)" -gt 6291456 ] || { echo "DISK STOP $(date -u +%T)" >> $L; return 1; }; }
A=$S/volume-cartographer/apps/VC3D
FILES="overlays/SurfaceRotationOverlayController.hpp CWindow.cpp MenuActionController.cpp segmentation/SegmentationModule.cpp
 segmentation/SegmentationModule.hpp segmentation/SurfaceStoreEdit.cpp segmentation/SurfaceStoreEdit.hpp"
EXCL=(--exclude="volume-cartographer/apps/VC3D/overlays/SurfaceRotationOverlayController.*"
      --exclude="volume-cartographer/apps/VC3D/overlays/SurfaceRotationApplyTest.inc")
for f in $FILES; do EXCL+=(--exclude="volume-cartographer/apps/VC3D/$f"); done
echo "start $(date -u +%FT%TZ) runs: $*" >> $L
rsync -rc --delete "${EXCL[@]}" $T/src_c8f3/ $S/ || { echo "RSYNC FAILED" >> $L; exit 1; }
echo "src synced to c8f3ba453+wiring $(date -u +%FT%TZ)" >> $L
disk_ok || exit 9
flock <work>/../../.build.lock env CMAKE_PREFIX_PATH=$R PKG_CONFIG_PATH=$R/lib/aarch64-linux-gnu/pkgconfig \
  cmake -S $S/volume-cartographer -B $B > $T/configure22.log 2>&1 || { echo "CONFIGURE FAILED" >> $L; tail -30 $T/configure22.log >> $L; exit 1; }
echo "configured $(date -u +%FT%TZ)" >> $L
for run in "$@"; do
  IFS=: read -r v t mode <<< "$run"
  for f in $FILES; do cmp -s $V/v22/$v/$f $A/$f || cp $V/v22/$v/$f $A/$f; done
  { cat $V/v22/$v/overlays/SurfaceRotationOverlayController.cpp
    printf "\n#ifdef VC_TESTING\n#include \"SurfaceRotationApplyTest.inc\"\n#endif\n"; } > $T/tmp/ov.cpp
  cmp -s $T/tmp/ov.cpp $A/overlays/SurfaceRotationOverlayController.cpp || cp $T/tmp/ov.cpp $A/overlays/SurfaceRotationOverlayController.cpp
  cmp -s $V/$t.inc $A/overlays/SurfaceRotationApplyTest.inc || cp $V/$t.inc $A/overlays/SurfaceRotationApplyTest.inc
  (cd $V/v22/$v && for f in overlays/SurfaceRotationOverlayController.cpp $FILES; do echo "$(git hash-object $f) $f"; done
   echo "$(git hash-object $V/$t.inc) $t.inc") > $T/files22_${v}_${t}.txt
  disk_ok || exit 9
  flock <work>/../../.build.lock nice -n 10 timeout 30000 ninja -C $B -j3 VC3D > $T/build22_${v}_${t}${mode:+_$mode}.log 2>&1; rc=$?
  echo "$run ninja exit $rc $(date -u +%FT%TZ)" >> $L
  [ $rc = 0 ] || { grep -E "error|FAILED" $T/build22_${v}_${t}${mode:+_$mode}.log | head -20 >> $L; continue; }
  log=$T/ctest22_${v}_${t}${mode:+_$mode}.log; rep=(); [ "$mode" = repeat10 ] && rep=(--repeat until-fail:10)
  (cd $B && timeout -s KILL 2400 ctest -R "^surface_rotation_apply$" -V --timeout 600 "${rep[@]}" > $log 2>&1)
  echo "$run ctest exit $? $(grep -E "tests passed|tests failed" $log | tail -1)" >> $L
  if [ -n "$mode" ]; then grep -E "case |FAIL|cases:|test:|wired:|reopened|Test #|Passed" $log | sed "s/^[0-9]*: //" | sort | uniq -c | head -60 >> $L
  else grep -E "case |FAIL|cases:|test:|wired:|reopened" $log | sed "s/^[0-9]*: //" | head -30 >> $L; fi
done
echo "done $(date -u +%FT%TZ)" >> $L
