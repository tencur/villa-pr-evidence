#!/bin/bash
# rd_c615.sh <villa tree> <python> <work dir>: the nnunet README chain on a 500^3 PHerc0332 surface-prediction crop
# (27 chunks, ~9 MB, anonymous). <python> needs zarr 2, tifffile, scikit-image. The work dir is deleted afterwards.
T=$1; P=$2; W=$3; D=$(dirname $(readlink -f $0))
rm -rf $W; mkdir -p $W
echo "-- $(cd $T && git log -1 --format='%h %s' | cut -c1-72)"
$P $D/check_c615.py $T $W 2>&1 | tr "\r" "\n" | grep -v "Warning\|Processing files\|Converting TIFF\|^$"
rm -rf $W
