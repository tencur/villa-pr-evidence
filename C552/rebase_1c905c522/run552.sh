#!/bin/bash
# C552 rework proof (2026-10-29): upstream 1c905c522 + fix (c552_fix.patch) + evidence self-test (c552_test.diff:
# VC3D --test-surface-rotation, ctest surface_rotation_apply). fix = both applied; pre = the fix's source change reverted
# (test kept). Build VC3D (QuickBuild -O2, VC_TESTING=ON, VC_BUILD_APPS=ON, -j3). Disk guard 4 GB.
T=/home/pi/scrollprize_work/vcbuild/r29/c552; S=$T/src; B=$T/build; L=$T/run.log
export TMPDIR=$T/tmp CCACHE_BASEDIR=/home/pi/scrollprize_work/vcbuild/r29 QT_QPA_PLATFORM=offscreen; mkdir -p $TMPDIR
disk_ok() { [ "$(df -k --output=avail /home | tail -1)" -gt 4194304 ] || { echo "DISK STOP $(date -u +%T)" >> $L; return 1; }; }
echo "start $(date -u +%FT%TZ)" >> $L
rm -rf $S; mkdir -p $S; tar -xf $T/src_1c905c522.tar -C $S
cd $S && git init -q && git add -A && git -c user.name=c552 -c user.email=c552@localhost commit -q -m base && cd $T
git -C $S apply --whitespace=nowarn $T/c552_fix.patch && git -C $S apply --whitespace=nowarn $T/c552_test.diff || { echo "patches do not apply" >> $L; exit 1; }
disk_ok || exit 9
cmake -S $S/volume-cartographer -B $B -G Ninja -DCMAKE_BUILD_TYPE=QuickBuild -DVC_QUICKBUILD_OPT_LEVEL=2 \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBLA_VENDOR=OpenBLAS -DVC_BUILD_APPS=ON -DVC_BUILD_FLATBOI=OFF \
  -DVC_BUILD_PYTHON=OFF -DVC_TESTING=ON -DVC_USE_CCACHE=ON > $T/configure.log 2>&1 || { echo "CONFIGURE FAILED" >> $L; exit 1; }
for side in fix pre; do
  if [ $side = pre ]; then
    sed -n '/^diff --git/,$p' $T/c552_fix.patch | git -C $S apply -R --whitespace=nowarn || { echo "pre revert failed" >> $L; exit 1; }
  fi
  disk_ok || exit 9
  flock /home/pi/scrollprize_work/.build.lock nice -n 10 ninja -C $B -j3 VC3D > $T/build_$side.log 2>&1; rc=$?
  echo "$side ninja exit $rc $(date -u +%FT%TZ)" >> $L
  [ $rc = 0 ] || { tail -30 $T/build_$side.log >> $L; continue; }
  (cd $B && timeout -s KILL 900 ctest -R '^surface_rotation_apply$' --output-on-failure --timeout 300 > $T/${side}_ctest.log 2>&1); echo "$side ctest exit $? $(grep -E 'tests passed' $T/${side}_ctest.log)" >> $L
  grep -E "surface rotation apply test|tests passed" $T/${side}_ctest.log | head -3 >> $L
done
echo "done $(date -u +%FT%TZ)" >> $L
