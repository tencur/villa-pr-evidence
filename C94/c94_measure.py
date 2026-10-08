"""Fraction of adjacent material-voxel pairs with antiparallel normals, per axis."""
import sys
import numpy as np
import zarr

def load_normal_u8(path):
    g = zarr.open_group(path, mode="r")
    n = np.stack([g["normal"][ax]["0"][...] for ax in "zyx"]).astype(np.float32)
    return n / 255.0 * 2.0 - 1.0

def load_eigvec_normal(path):
    g = zarr.open_group(path, mode="r")
    if "eigenvectors" not in g:
        return None
    ev = g["eigenvectors"][...]
    return ev.reshape(3, 3, *ev.shape[1:])[2]

def measure(n, label):
    norm = np.linalg.norm(n, axis=0)
    material = norm > 0.5
    print(f"{label}: material voxels {material.sum()} / {material.size}")
    for axis, name in enumerate("zyx"):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1); sl_b[axis] = slice(1, None)
        a = n[(slice(None), *sl_a)]; b = n[(slice(None), *sl_b)]
        both = material[tuple(sl_a)] & material[tuple(sl_b)]
        dot = (a * b).sum(axis=0)[both]
        print(f"  {name}: pairs={dot.size}  antiparallel={100.0 * (dot < 0).mean():.2f}%  mean|dot|={np.abs(dot).mean():.4f}  mean dot={dot.mean():.4f}")

for path in sys.argv[1:]:
    measure(load_normal_u8(path), f"{path} [normal/ uint8]")
    ev = load_eigvec_normal(path)
    if ev is not None:
        measure(ev, f"{path} [eigenvectors float32, v2]")
        ev_all = zarr.open_group(path, mode="r")["eigenvectors"][...].reshape(3, 3, -1)
        V = ev_all.transpose(2, 0, 1)
        det = np.linalg.det(V)
        nz = np.abs(det) > 0.5
        print(f"  det: min={det[nz].min():.4f} max={det[nz].max():.4f} negatives={int((det[nz] < 0).sum())}")
        print(f"  v2_z < 0 count: {int((ev[0] < 0).sum())}")
