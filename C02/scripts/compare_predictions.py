#!/usr/bin/env python
"""Compare prediction TIFFs pairwise and write them side by side as one PNG.
usage: compare_predictions.py out.png label=path.tif [label=path.tif ...]"""
import hashlib, os, sys
import cv2, numpy as np, tifffile
out = sys.argv[1]; items = [a.split("=", 1) for a in sys.argv[2:]]
imgs = {k: tifffile.imread(p) for k, p in items}
for k, p in items:
    a = imgs[k]
    print(f"{k:15s} {os.path.basename(p):20s} shape={a.shape} mean={a.mean():7.3f} nonzero={float((a > 0).mean()):.3f} sha256={hashlib.sha256(a.tobytes()).hexdigest()[:16]}")
keys = list(imgs)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = imgs[keys[i]], imgs[keys[j]]
        if a.shape != b.shape:
            print(f"{keys[i]} vs {keys[j]}: shapes differ"); continue
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        print(f"{keys[i]} vs {keys[j]}: identical={bool((d == 0).all())} differing_pixels={int((d > 0).sum())} ({100.0 * float((d > 0).mean()):.2f}%) max_abs_diff={int(d.max())}")
h = max(a.shape[0] for a in imgs.values()); tiles = []
for k in keys:
    a = imgs[k]; pad = np.zeros((h + 40, a.shape[1]), np.uint8); pad[40:40 + a.shape[0]] = a
    cv2.putText(pad, k, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2); tiles.append(pad); tiles.append(np.full((h + 40, 12), 128, np.uint8))
cv2.imwrite(out, np.hstack(tiles[:-1])); print("wrote", os.path.basename(out))
