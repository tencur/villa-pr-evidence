"""compute_vf consumer metric: confidence = ||smoothed N|| (clamped to 1) and
|cos| between the smoothed direction and the local eigenvector, on interior
material voxels (|n| > 0.5 in the input eigenvectors, >= rad voxels from the
volume border, rad = int(3*xi))."""
import sys
import numpy as np, zarr
eigen_path, vf_path, xi = sys.argv[1], sys.argv[2], float(sys.argv[3])
ev = zarr.open_group(eigen_path, mode="r")["eigenvectors"][...]
n = ev.reshape(3, 3, *ev.shape[1:])[2].astype(np.float64)
material = np.linalg.norm(n, axis=0) > 0.5
rad = int(3 * xi)
interior = np.zeros_like(material); interior[rad:-rad, rad:-rad, rad:-rad] = True
sel = material & interior
g = zarr.open_group(vf_path, mode="r")
N = g["N"][...].astype(np.float64)
norm = np.linalg.norm(N, axis=0)
conf = np.clip(norm, 0.0, 1.0)[sel]
cos = np.abs((N * n).sum(axis=0) / np.maximum(norm, 1e-8))[sel]
print(f"{vf_path}: interior material voxels {sel.sum()}")
print(f"  confidence (float N): mean {conf.mean():.3f}  median {np.median(conf):.3f}  frac<0.25 {100*(conf<0.25).mean():.1f}%  frac>0.75 {100*(conf>0.75).mean():.1f}%")
print(f"  |cos|(smoothed dir, local eigenvector v2): mean {cos.mean():.3f}  median {np.median(cos):.3f}")
# uint8 confidence written by the OME writer, if present
def find(grp, name, prefix=""):
    for k in grp.keys():
        p = f"{prefix}/{k}"
        if k == name and isinstance(grp[k], zarr.Array): return p, grp[k]
        if isinstance(grp[k], zarr.Group):
            r = find(grp[k], name, p)
            if r: return r
    return None
r = find(g, "0") if "confidence" not in g else None
if "confidence" in g:
    cg = g["confidence"]
    arr = cg["0"] if isinstance(cg, zarr.Group) else cg
    c8 = arr[...].astype(np.float64) / 255.0
    c8 = c8[sel]
    print(f"  confidence (uint8/255): mean {c8.mean():.3f}  median {np.median(c8):.3f}  frac<0.25 {100*(c8<0.25).mean():.1f}%")
print("  tree:", sorted(g.keys()))
