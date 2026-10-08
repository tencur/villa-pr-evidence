#!/usr/bin/env bash
# optimized_inference: STEP=reduce copies each partition into /tmp/partition_cache only if that partition number is not
# already there, and never clears the directory. Reduce two different inference runs (partition zarrs written by the
# pipeline's inference step) one after the other on one machine, through the unmodified entrypoint, and compare.
#
# usage: reproduce_stale_reduce.sh <optimized_inference dir> <python> <partitions of run 1> <partitions of run 2> <out dir>
# MODEL / MODEL_TYPE / START_LAYER / END_LAYER may be set in the environment (defaults: the public Scroll 5 TimeSformer, layers 0-26).
set -u
CODE=$1; PY=$2; P1=$3; P2=$4; OUT=$5; HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-timesformer_scroll5_27112024}; MODEL_TYPE=${MODEL_TYPE:-timesformer}; START_LAYER=${START_LAYER:-0}; END_LAYER=${END_LAYER:-26}
mkdir -p "$OUT"; rm -rf /tmp/partition_cache /tmp/partition_cache_*          # start as a fresh machine would
red() {   # <partitions dir> <name>: the production entrypoint with STEP=reduce (reduce needs no model weights)
  ( cd "$CODE" && env MODEL="$MODEL" MODEL_TYPE="$MODEL_TYPE" STEP=reduce START_LAYER="$START_LAYER" END_LAYER="$END_LAYER" \
      NUM_PARTS=1 ZARR_OUTPUT_DIR="$1" OUTPUT_PATH="$OUT/$2.tif" PROFILING_LOCAL_ROOT="$OUT/profiling" \
      AWS_DEFAULT_REGION=us-east-1 AWS_EC2_METADATA_DISABLED=true "$PY" entrypoint.py ) > "$OUT/$2.log" 2>&1
  echo "$2: exit $?, log says 'Reduce completed successfully' $(grep -c 'Reduce completed successfully' "$OUT/$2.log") time(s)"
}
echo "-- run 1 (partitions: $P1), then run 2 (partitions: $P2), nothing cleaned in between"
red "$P1" run1
red "$P2" run2
echo "-- run 2 again after removing /tmp/partition_cache by hand (reference)"
rm -rf /tmp/partition_cache
red "$P2" run2_reference
"$PY" "$HERE/compare_predictions.py" "$OUT/c02_compare.png" run1="$OUT/run1.tif" run2="$OUT/run2.tif" run2_reference="$OUT/run2_reference.tif" 2>&1 | grep -v -i warn
left=$(ls -d /tmp/partition_cache* 2>/dev/null | tr '\n' ' '); echo "cache directories left behind by the three reduces: ${left:-none}"
rm -rf /tmp/partition_cache /tmp/partition_cache_* /tmp/result_s3_url.txt "/tmp/prediction_${MODEL}_$(printf %02d "$START_LAYER")_$(printf %02d "$END_LAYER").tif"
