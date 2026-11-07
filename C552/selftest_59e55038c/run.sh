#!/bin/bash
# H5 lane E (2026-11-01): C552 on upstream 59e55038c. One build tree (git archive of h5e 956dc784b + self-test wiring),
# SurfaceRotationOverlayController.{cpp,hpp} swapped per variant (main 59e55038c / r32 a229b19b7 / h5e 956dc784b),
# test .inc = test_ext (extended) or test_orig (kit's 1c905c522 test). Build VC3D only; ctest surface_rotation_apply.
T=/home/pi/scrollprize_work/vcbuild/h5E; V=$T/variants; S=$T/src; B=$T/build; L=$T/run.log
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T QT_QPA_PLATFORM=offscreen; mkdir -p $TMPDIR
disk_ok() { [ "$(df -k --output=avail /home | tail -1)" -gt 6291456 ] || { echo "DISK STOP $(date -u +%T)" >> $L; return 1; }; }
O=$S/volume-cartographer/apps/VC3D/overlays
echo "start $(date -u +%FT%TZ)" >> $L
if [ ! -f $B/build.ninja ]; then
  rm -rf $S; mkdir -p $S; tar -xf $V/src_h5e.tar -C $S
  git -C $S apply --whitespace=nowarn $V/wiring_cmake_main.diff || { echo "wiring does not apply" >> $L; exit 1; }
  disk_ok || exit 9
  cmake -S $S/volume-cartographer -B $B -G Ninja -DCMAKE_BUILD_TYPE=QuickBuild -DVC_QUICKBUILD_OPT_LEVEL=2 \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBLA_VENDOR=OpenBLAS -DVC_BUILD_APPS=ON -DVC_BUILD_FLATBOI=OFF \
    -DVC_BUILD_PYTHON=OFF -DVC_TESTING=ON -DVC_USE_CCACHE=ON > $T/configure.log 2>&1 || { echo "CONFIGURE FAILED" >> $L; exit 1; }
fi
for run in "$@"; do   # run = <variant>:<test>, e.g. h5e:test_ext
  v=${run%%:*}; t=${run#*:}
  cmp -s $V/$v.hpp $O/SurfaceRotationOverlayController.hpp || cp $V/$v.hpp $O/SurfaceRotationOverlayController.hpp
  { cat $V/$v.cpp; printf '\n#ifdef VC_TESTING\n#include "SurfaceRotationApplyTest.inc"\n#endif\n'; } > $T/tmp/ov.cpp
  cmp -s $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp || cp $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp
  cmp -s $V/$t.inc $O/SurfaceRotationApplyTest.inc || cp $V/$t.inc $O/SurfaceRotationApplyTest.inc
  disk_ok || exit 9
  flock /home/pi/scrollprize_work/.build.lock nice -n 10 timeout 30000 ninja -C $B -j3 VC3D > $T/build_${v}_${t}.log 2>&1; rc=$?
  echo "$run ninja exit $rc $(date -u +%FT%TZ)" >> $L
  [ $rc = 0 ] || { tail -30 $T/build_${v}_${t}.log >> $L; continue; }
  (cd $B && timeout -s KILL 900 ctest -R '^surface_rotation_apply$' --output-on-failure --timeout 300 > $T/ctest_${v}_${t}.log 2>&1)
  echo "$run ctest exit $? $(grep -E 'tests passed' $T/ctest_${v}_${t}.log)" >> $L
  grep -E "surface rotation apply test" $T/ctest_${v}_${t}.log | head -3 >> $L
done
echo "done $(date -u +%FT%TZ)" >> $L
