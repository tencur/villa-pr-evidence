"""Labelled-voxel coverage of the patches vesuvius.find_patches keeps, on a real OME label volume.
Usage: python G01_find_patches_coverage.py <label OME zarr> <patch size at level 0> [resolution]
"""
import sys
import numpy as np, zarr
from vesuvius.models.datasets.find_valid_patches import find_valid_patches
path, ps0 = sys.argv[1], int(sys.argv[2]); res = int(sys.argv[3]) if len(sys.argv) > 3 else 1
g = zarr.open(path, mode="r")
r = find_valid_patches(label_arrays=[g], label_names=["vol"], patch_size=(ps0,) * 3, bbox_threshold=0.0,
                       label_threshold=0.0001, valid_patch_find_resolution=res)
starts = [tuple(int(v) for v in p["start_pos"]) for p in r["fg_patches"]]
lab = np.asarray(g["0"][:]) > 0                                   # count at full resolution
cov = np.zeros_like(lab)
for z, y, x in starts:
    cov[z:z + ps0, y:y + ps0, x:x + ps0] = True
miss = int((lab & ~cov).sum()); tot = int(lab.sum())
print(f"patch {ps0}^3: {len(starts)} foreground patches kept; labelled voxels outside every patch: {miss} of {tot} ({miss / tot:.2%})")
