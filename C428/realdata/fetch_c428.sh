#!/bin/bash
# Copy level 4 of the public PHerc0500P2 volume (9.362 um, 113 keV) byte for byte: 16 stored 128^3 chunks, about 31 MB.
set -e
B=s3://vesuvius-challenge-open-data/PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr/4
aws s3 cp --no-sign-request --recursive --quiet $B PHerc0500P2_l4.pristine
find PHerc0500P2_l4.pristine -type f | wc -l
