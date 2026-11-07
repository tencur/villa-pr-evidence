#!/bin/bash
# H5 lane E real-app runs on 59e55038c: Xvfb :97 (1600x1000), harness c552_gui_h5e.py (copy of runs/C552RD harness),
# VC3D per variant. NOCHANGE=1 adds the Apply-with-no-change step. Usage: app_runs.sh <variant>[:<n>] ...
A=/home/pi/scrollprize_work/vcbuild/h5E/app; PY=/home/pi/scrollprize_work/fix-sw4b/venv/bin/python3
export LD_LIBRARY_PATH=/home/pi/scrollprize_work/vcbuild/h5E/build/core:/home/pi/scrollprize_work/vcbuild/h5E/build/utils:/home/pi/scrollprize_work/vcbuild/h5E/build/libs/c3d:/home/pi/scrollprize_work/vcbuild/h5E/build/libs/volcomp VID="20250521135224-8.640um-1.2m-116keV-masked.zarr-59098600b52265ae" DISPLAY=:97 PYTHONPATH=$A/xlib NOCHANGE=1
pgrep -f "Xvfb :97" >/dev/null || { (setsid Xvfb :97 -screen 0 1600x1000x24 -nolisten tcp > $A/xvfb97.log 2>&1 < /dev/null &); sleep 3; }
for run in "$@"; do
  v=${run%%:*}
  echo "=== $run $(date -u +%FT%TZ)" >> $A/app_runs.log
  VC3D_BIN=$A/bin/VC3D.$v timeout 900 $PY $A/c552_gui_h5e.py ${run/:/_} run > $A/runlog_${run/:/_}.txt 2>&1
  echo "exit $?" >> $A/runlog_${run/:/_}.txt
  grep -vE "^\[|^qt\.|propagateSizeHints" $A/runlog_${run/:/_}.txt >> $A/app_runs.log
done
pkill -f "Xvfb :97"; echo "all done $(date -u +%T)" >> $A/app_runs.log
