#!/usr/bin/env bash
# compute_st end to end from the real PHerc0500P2 crop, upstream main e0bbb8b vs main+C94, no integration smoothing.
cd /home/pi/scrollprize_work/runs/RB/C94main
PY=/home/pi/scrollprize_work/.venv314/bin/python
CROP=/home/pi/scrollprize_work/runs/C27/crop.zarr
for pair in "main:main" "fix:tree"; do
  tag=${pair%%:*}; src=${pair##*:}
  rm -rf st_$tag.zarr
  echo "=== st_$tag  PYTHONPATH=$src/vesuvius/src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY -m vesuvius.structure_tensor.run_create_st --input_dir $CROP --output_dir st_$tag --patch_size 64,64,64 --keep-eigen"
  t0=$(date +%s)
  PYTHONPATH=$src/vesuvius/src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY -m vesuvius.structure_tensor.run_create_st --input_dir $CROP --output_dir st_$tag --patch_size 64,64,64 --keep-eigen > st_$tag.log 2>&1
  echo "   exit code $?  wall $(( $(date +%s) - t0 )) s"
done
echo ALLDONE
