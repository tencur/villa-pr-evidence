#!/bin/bash
# run_rd_c428.sh main|fix <villa tree> <python>
# Real-data proof for C428 + C429 on level 4 of the public PHerc0500P2 volume (fetch_c428.sh), zarr 2.18.7.
set -u
SIDE=$1; TREE=$2; PY=$3; HERE=$(cd $(dirname $0) && pwd)
export PYTHONPATH=$TREE/vesuvius/src
W=$HERE/work_$SIDE; rm -rf $W; mkdir -p $W
echo "== C428 on $SIDE ($(cd $TREE && git log --format=%h -1), zarr $($PY -c 'import zarr;print(zarr.__version__)'))"
echo "-- 1. trailing slash on the array path (as shell tab-completion writes it)"
cp -r $HERE/PHerc0500P2_l4.pristine $W/PHerc0500P2_l4.zarr
$PY $HERE/state_c428.py $W/PHerc0500P2_l4.zarr $HERE/PHerc0500P2_l4.pristine | sed 's/^/  before:/'
echo "\$ vesuvius.zarr_tasks PHerc0500P2_l4.zarr/ --task recompress --inplace"
(cd $W && $PY -m vesuvius.image_proc.run.zarr_tasks PHerc0500P2_l4.zarr/ --task recompress --inplace > $W/log 2>&1; echo "  exit status: $?" > $W/rc); grep -E 'Error|error|failed|Restored' $W/log | tail -2 | cut -c1-220; cat $W/rc
$PY $HERE/state_c428.py $W/PHerc0500P2_l4.zarr $HERE/PHerc0500P2_l4.pristine | sed 's/^/  after: /'
echo "-- 2. one truncated chunk (3/1/1 cut to 1 MiB, as an interrupted copy leaves it), no --skip-corrupt"
rm -rf $W/PHerc0500P2_l4.zarr; cp -r $HERE/PHerc0500P2_l4.pristine $W/PHerc0500P2_l4.zarr
truncate -s 1M $W/PHerc0500P2_l4.zarr/3/1/1
echo "\$ vesuvius.zarr_tasks PHerc0500P2_l4.zarr --task recompress --inplace"
(cd $W && $PY -m vesuvius.image_proc.run.zarr_tasks PHerc0500P2_l4.zarr --task recompress --inplace > $W/log 2>&1; echo "  exit status: $?" > $W/rc); grep -E 'Error|error|failed|Restored' $W/log | tail -3 | cut -c1-220; cat $W/rc
$PY $HERE/state_c428.py $W/PHerc0500P2_l4.zarr $HERE/PHerc0500P2_l4.pristine | sed 's/^/  after: /'
echo "  (14 of the 15 chunks are intact before the run; 3/1/1 is the truncated one)"
rm -rf $W
