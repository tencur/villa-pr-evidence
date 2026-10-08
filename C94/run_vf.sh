#!/usr/bin/env bash
# compute_vf (xi 2, CPU, 64^3 chunks) on the two standalone eigen outputs, both trees, in parallel.
cd /home/pi/scrollprize_work/runs/RB/C94main
PY=/home/pi/scrollprize_work/.venv314/bin/python
MASK=/home/pi/scrollprize_work/runs/RB/C94/c94_mask.zarr
for pair in "main:main" "fix:tree"; do
  tag=${pair%%:*}; src=${pair##*:}
  rm -rf vf_$tag.zarr
  echo "=== vf_$tag  PYTHONPATH=$src/vesuvius/src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY -m vesuvius.structure_tensor.compute_vf --input-zarr $MASK --eigen 1:st_$tag.zarr --xi 2 --device cpu --chunk-size 64,64,64 --output-zarr vf_$tag.zarr --export-field N --write-confidence"
  ( t0=$(date +%s); PYTHONPATH=$src/vesuvius/src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY -m vesuvius.structure_tensor.compute_vf --input-zarr $MASK --eigen 1:st_$tag.zarr --xi 2 --device cpu --chunk-size 64,64,64 --output-zarr vf_$tag.zarr --export-field N --write-confidence > vf_$tag.log 2>&1; echo "vf_$tag exit code $?  wall $(( $(date +%s) - t0 )) s" >> run_vf.out ) &
done
wait
echo ALLDONE >> run_vf.out
