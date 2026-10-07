"""Fraction of labelled voxels that the vesuvius.train patch grid never covers (origin-anchored, stride = patch,
no end-aligned tail patch), measured on a real label volume at the given pyramid level.
Usage: python G01_tail_coverage.py <label OME zarr> <level> <patch sizes at level 0...>
"""
import sys
import numpy as np, zarr
g = zarr.open(sys.argv[1], mode="r"); lvl = int(sys.argv[2]); a = g[str(lvl)]; f = 2 ** lvl
shape0 = tuple(int(s) * f for s in a.shape)
print(f"label level {lvl} shape {a.shape} (level-0 extent ~{shape0}), chunks {a.chunks}")
# per-plane / per-row foreground counts along each axis, accumulated chunk by chunk
cz = np.zeros(a.shape[0], np.int64); cy = np.zeros(a.shape[1], np.int64); cx = np.zeros(a.shape[2], np.int64)
total = 0
step = a.chunks[0]
for z0 in range(0, a.shape[0], step):
    blk = np.asarray(a[z0:z0 + step]) > 0
    cz[z0:z0 + blk.shape[0]] += blk.sum(axis=(1, 2)); cy += blk.sum(axis=(0, 2)); cx += blk.sum(axis=(0, 1))
    total += int(blk.sum())
print(f"labelled voxels at level {lvl}: {total}")
for ps0 in (int(v) for v in sys.argv[3:]):
    ps = ps0 // f
    covered = []
    for n in a.shape:
        last = max(0, (n - ps) // ps * ps + ps) if n >= ps else 0
        covered.append(min(n, last))
    # voxels outside the covered box on at least one axis: upper bound via axis marginals is not exact; compute exactly
    inside = 0
    for z0 in range(0, min(covered[0], a.shape[0]), step):
        z1 = min(z0 + step, covered[0])
        inside += int((np.asarray(a[z0:z1, :covered[1], :covered[2]]) > 0).sum())
    print(f"patch {ps0}^3: grid covers [0,{covered[0]*f}) x [0,{covered[1]*f}) x [0,{covered[2]*f}) of the volume; "
          f"labelled voxels never in any patch: {total - inside} of {total} ({(total - inside) / max(total, 1):.2%})")
