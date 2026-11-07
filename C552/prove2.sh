#!/bin/bash
# prove.sh <id> <branch> "<fixed files (paths under volume-cartographer/)>" "<ninja targets>" "<ctest regex>"
# Builds the branch, runs the tests (fix), reverts only the fixed source files to the base (default bcdf9bf53), rebuilds and reruns (pre),
# then restores the branch. Logs: fcpp/proof/<id>/{fix,pre}_ctest.log
F=/home/pi/scrollprize_work/vcbuild/fsw4; S=$F/src_allv2; id=$1; br=$2; files=$3; targets=$4; rx=$5
P=$F/proof/$id; mkdir -p $P
export TMPDIR=$F/tmp
git -C $S checkout -q --detach -f $br || exit 1
echo "fix: $(git -C $S rev-parse --short HEAD)" > $P/shas.txt
$F/build.sh fix_$id $targets || { echo "fix build failed" >> $P/shas.txt; exit 2; }
(cd $F/build_allv2 && ctest -R "$rx" --output-on-failure --timeout 900 > $P/fix_ctest.log 2>&1); echo "fix ctest exit $?" >> $P/shas.txt
for f in $files; do git -C $S checkout ${BASE:-bcdf9bf53} -- volume-cartographer/$f; done
$F/build.sh pre_$id $targets || { echo "pre build failed" >> $P/shas.txt; }
(cd $F/build_allv2 && ctest -R "$rx" --output-on-failure --timeout 900 > $P/pre_ctest.log 2>&1); echo "pre ctest exit $?" >> $P/shas.txt
git -C $S checkout -q --detach -f $br
echo done >> $P/shas.txt
