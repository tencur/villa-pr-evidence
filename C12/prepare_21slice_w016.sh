#!/bin/bash
# prepare_21slice_w016.sh <out_dir>
# Builds the 21-slice family used by proof_21slice_w016.txt, exactly as the ink_detection README's Commands section
# does for pherc0139-w016: the published labels of aligned-scrollprizeorg-21slices/pherc0139-w016 (ink_9um label tree,
# Hugging Face bucket scrollprize/datasets) and a 21-slice surface volume pooled from the public 2.399 um surface volume
# of PHerc0139 segment w029 (pherc0139-w016 in the label tree) with prepare_9um_isotropic_input --level 2.
# Needs the `hf` CLI and a villa checkout on PYTHONPATH. About 735 MB.
set -e
OUT=${1:?out_dir}; F=$OUT/aligned-scrollprizeorg-21slices/pherc0139-w016; mkdir -p $F
hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016 $F
python -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input \
  s3://vesuvius-challenge-open-data/PHerc0139/segments/20250108000004-w029_2025010827/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr \
  $F/surface-volume.zarr --level 2 --workers 4
# then: python reproduce_label_depth.py $OUT/aligned-scrollprizeorg-21slices <work_dir>
