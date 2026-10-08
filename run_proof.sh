#!/bin/bash
# Reproduce one PR's before/after on rackpi6 for the screenshot the villa CONTRIBUTING rules ask for.
# Usage (on rackpi6):  bash ~/scrollprize_work/runs/RELEASE/run_proof.sh <ID> <main|fix>
#   main = villa main (identical to upstream e0bbb8b for these files), fix = the rebased PR branch (exported tree).
set -u
ID=$1; V=$2
W=/home/pi/scrollprize_work; R=$W/runs; RB=$R/RB; P3=$W/.venv314/bin/python; P2=$W/.venv/bin/python
if [ "$V" = main ]; then S=$W/repos/villa; else S=$RB/$ID/tree; fi
OUT=$R/RELEASE/out/$ID/$V; rm -rf $OUT; mkdir -p $OUT
CK=$R/B01/work_binary01/ckpt/b01_binary01/b01_binary01_final.pth
echo "== $ID on $V  (code: $S)"
case $ID in
C12) cd $R/C12 && PYTHONPATH=$S/vesuvius/src $P3 reproduce_label_depth.py pristine/native9-scrollprizeorg-21slices $OUT/work 2>&1 | grep -v -i warn | tail -9 ;;
C47) PYTHONPATH=$W/repos/villa/vesuvius/src $P3 $R/repro_layer_window.py $S/ink-detection/optimized_inference $R/C08/ds/images/PHerc0500P2.zarr/0 $OUT/work 2>&1 | grep -v "^INFO\|Building" ;;
C03) mkdir -p $OUT/work/layers && cp $R/C03/demo/layers/32.tif $OUT/work/layers/ && $P2 $R/C03/reproduce_prepare_16bit.py $S $OUT/work 2>&1 | tail -6 ;;
C04) SB=s3://vesuvius-challenge-open-data/PHerc0500P2/segments
     A=$SB/20250919184428-0500P2-wrap01_0919/surface-volumes/9.362um-1.2m-113keV-volume-20250820143440.zarr
     B=$SB/20250920020224-0500P2-wrap13_0919/surface-volumes/9.362um-1.2m-113keV-volume-20250820143440.zarr
     C=$S/ink-detection/optimized_inference; mkdir -p $OUT/shared $OUT/fresh
     echo "-- one cache directory: read segment A, then segment B, then segment A again"; (cd $OUT/shared && $P2 $RB/reproduce_cache_read.py $C $A 2>&1 | tail -1; $P2 $RB/reproduce_cache_read.py $C $B 2>&1 | tail -1; $P2 $RB/reproduce_cache_read.py $C $A 2>&1 | tail -1)
     echo "-- fresh directory: read segment B"; (cd $OUT/fresh && $P2 $RB/reproduce_cache_read.py $C $B 2>&1 | tail -1) ;;
C64) for ps in 128 192; do PYTHONPATH=$S/vesuvius/src $P3 $R/G01_find_patches_coverage.py $R/C08/ds/labels/PHerc0500P2_surface.zarr $ps 1 2>&1 | grep "^patch"; done ;;
C63) for r in a b; do PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 -m vesuvius.models.training.train -i $R/B01/work_binary01/data --config $R/S03/s03.yaml -o $OUT/run_$r --no-amp --seed 42 > $OUT/run_$r.log 2>&1; echo "run $r (--seed 42): exit $?, $(grep -o 'surface: [0-9.]*' $OUT/run_$r.log | tail -1)"; done
     $P3 -c "
import torch, glob
L = lambda p: torch.load(sorted(glob.glob(p + '/**/*epoch*.pth', recursive=True))[0], map_location='cpu', weights_only=False)['model']
a, b = L('$OUT/run_a'), L('$OUT/run_b'); fl = [k for k in a if a[k].is_floating_point()]
print('weight tensors that differ between the two runs:', sum(1 for k in fl if not torch.equal(a[k], b[k])), 'of', len(fl))" ;;
C49) cd $R && PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 repro_predict_activation.py $CK C27/crop.zarr $OUT/p 2>&1 | grep "^predict\|^stored"
     PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 finish_activation.py $CK C27/crop.zarr $OUT/p 2>&1 | grep "^blended\|^finalize" ;;
C52) cd $R && if [ "$V" = main ]; then PYTHONPATH=$S/vesuvius/src $P3 validation_on_activated.py $CK B01/work_binary01/data/images/vol.zarr/0 B01/work_binary01/data/labels/vol_surface.zarr/0 2>&1 | grep "^real\|^BCE\|^foreground"
     else PYTHONPATH=$S/vesuvius/src $P3 validation_after_fix.py $CK B01/work_binary01/data/images/vol.zarr/0 B01/work_binary01/data/labels/vol_surface.zarr/0 2>&1 | grep "^same\|^BCE\|^foreground"; fi ;;
C38) cd $R/B01 && NO_SCORE=1 PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 run_binary_labels.py ../C08/ds $OUT/work raw255 40 > $OUT/run.log 2>&1
     grep "label values" $OUT/run.log; grep -h -o "surface: Avg Loss = [-0-9.]*" $OUT/work/train.log 2>/dev/null | tail -1; grep -h "ValueError" $OUT/run.log $OUT/work/train.log 2>/dev/null | tail -1 ;;
C31) PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 $R/C31/reproduce_bg_sampling.py $R/C08/ds $OUT/work 2>&1 | grep -v -i "warn\|INFO" | tail -14 ;;
C44) cd $OUT && PYTHONPATH=$S/vesuvius/src OMP_NUM_THREADS=2 $P3 -m vesuvius.structure_tensor.run_create_st --input_dir $R/C27/crop.zarr --output_dir st --patch_size 64,64,64 > st.log 2>&1; echo "compute_st exit $?"
     $P3 $RB/analyze_st.py st st 2>&1 | head -3 ;;
C02) bash $R/C02/reproduce_stale_reduce.sh $S/ink-detection/optimized_inference $P2 $R/C03/pred/parts_before $R/C03/pred/parts_after $OUT/work ;;
C96) $P3 $R/C96/opencv_pixel_limit_probe.py $S/ink-detection/optimized_inference $R/C03/demo/layers/32.tif $OUT/work ;;
*) echo "unknown ID"; exit 2 ;;
esac
