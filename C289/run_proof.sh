#!/bin/bash
# usage: run_proof.sh <tree>/lasagna <scratch dir>
# mix_channels_parallel.py CLI on synthetic preprocessed zarrs with chunk-unaligned crops (needs zarr >= 3, numcodecs, numpy). CPU, seconds.
set -e
python3 "$(dirname "$0")/c289_proof.py" "$1" "$2"
