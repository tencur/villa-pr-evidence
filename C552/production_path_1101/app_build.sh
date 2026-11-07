#!/bin/bash
# H5 lane E: VC3D binaries per variant (main / r32 / h5e on 59e55038c, same tree as run.sh, extended self-test kept)
# + vc_object_ingest/export (standalone vc_object_store from the same archive) for the real-app runs.
T=/home/pi/scrollprize_work/vcbuild/h5E; V=$T/variants; S=$T/src; B=$T/build; L=$T/app_build.log; A=$T/app
export TMPDIR=$T/tmp CCACHE_BASEDIR=$T
disk_ok() { [ "$(df -k --output=avail /home | tail -1)" -gt 6291456 ] || { echo "DISK STOP $(date -u +%T)" >> $L; return 1; }; }
O=$S/volume-cartographer/apps/VC3D/overlays; mkdir -p $A/bin
for v in main r32 h5e; do
  cmp -s $V/$v.hpp $O/SurfaceRotationOverlayController.hpp || cp $V/$v.hpp $O/SurfaceRotationOverlayController.hpp
  { cat $V/$v.cpp; printf '\n#ifdef VC_TESTING\n#include "SurfaceRotationApplyTest.inc"\n#endif\n'; } > $T/tmp/ov.cpp
  cmp -s $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp || cp $T/tmp/ov.cpp $O/SurfaceRotationOverlayController.cpp
  cmp -s $V/test_ext.inc $O/SurfaceRotationApplyTest.inc || cp $V/test_ext.inc $O/SurfaceRotationApplyTest.inc
  disk_ok || exit 9
  flock /home/pi/scrollprize_work/.build.lock nice -n 10 ninja -C $B -j3 VC3D > $T/appbuild_$v.log 2>&1; rc=$?
  echo "$v ninja exit $rc $(date -u +%T)" >> $L; [ $rc = 0 ] && cp $B/bin/VC3D $A/bin/VC3D.$v
done
[ -f $T/build_os/build.ninja ] || cmake -S $S/vc_object_store -B $T/build_os -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVC_OBJECT_STORE_BUILD_TESTS=OFF -DVC_OBJECT_STORE_BUILD_PYTHON=OFF > $T/configure_os.log 2>&1 || { echo "OS CONFIGURE FAILED" >> $L; exit 1; }
disk_ok || exit 9
flock /home/pi/scrollprize_work/.build.lock nice -n 10 ninja -C $T/build_os -j3 vc_object_ingest vc_object_export > $T/build_os.log 2>&1
echo "os ninja exit $? $(date -u +%T)" >> $L; sha256sum $A/bin/VC3D.* >> $L; echo done >> $L
