#!/usr/bin/env python
"""Reproduce: optimized_inference STEP=prepare on a real 16-bit layer.

usage: python reproduce_prepare_16bit.py /path/to/villa [workdir]
Run once on main (before) and once on the fix branch (after).
Needs ink-detection/optimized_inference/requirements-cpu-only.txt and network access.
Downloads one layer (61 MiB, CC BY-NC 4.0) of PHerc172 (Scroll 5) segment 20241127171800.
"""
import os, subprocess, sys, tempfile, urllib.request

villa = os.path.abspath(sys.argv[1])
work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="prepare16_")
code = os.path.join(villa, "ink-detection", "optimized_inference")
sys.path.insert(0, code)
import cv2, numpy as np, tifffile, zarr  # noqa: E402
import processing  # noqa: E402

URL = "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths/20241127171800/layers/32.tif"
layers = os.path.join(work, "layers"); os.makedirs(layers, exist_ok=True)
layer = os.path.join(layers, "32.tif")
if not os.path.exists(layer):
    urllib.request.urlretrieve(URL, layer)

out = os.path.join(work, "surface_volume.zarr")
if os.path.exists(out):
    import shutil; shutil.rmtree(out)
processing.create_surface_volume_zarr([layer], out)   # what STEP=prepare calls

raw = tifffile.imread(layer)
got = np.asarray(zarr.open(out, mode="r")[:, :, 0])
data = raw > 0
commit = subprocess.run(["git", "-C", villa, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"villa commit: {commit}")
print(f"input layer:  {raw.dtype} {raw.shape}, values in segment {int(raw[data].min())}..{int(raw[data].max())}")
print(f"prepared:     {got.dtype}, distinct values in segment: {len(np.unique(got[data]))}, "
      f"pixels equal to 255: {100.0 * float((got[data] == 255).mean()):.2f}%")
print(f"equals raw >> 8:                 {bool(np.array_equal(got, (raw >> 8).astype(np.uint8)))}")
print(f"equals cv2.IMREAD_GRAYSCALE read: {bool(np.array_equal(got, cv2.imread(layer, cv2.IMREAD_GRAYSCALE)))}")
y, x = np.argwhere(data).mean(0).astype(int)
png = os.path.join(work, f"prepared_crop_{commit}.png")
cv2.imwrite(png, got[max(0, y - 256):y + 256, max(0, x - 256):x + 256])
print(f"512 px crop written to {png}")
