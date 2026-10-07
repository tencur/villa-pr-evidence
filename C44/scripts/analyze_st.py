"""vesuvius.compute_st on a real 128^3 PHerc0500P2 crop: default (no --smooth-components) vs --smooth-components.
Reports the tensor's rank (smallest/largest eigenvalue ratio), the confidence distribution, and how coherent
first_component is between neighbouring voxels (|dot| of unit vectors one voxel apart; 1 = smooth field)."""
import sys
import numpy as np, zarr


def load(name):
    g = zarr.open(f"{name}.zarr", mode="r")
    return g


def report(name):
    g = load(name)
    st = g["structure_tensor"]
    a = np.asarray(st[:, 32:96, 32:96, 32:96] if st.ndim == 4 else st[32:96, 32:96, 32:96]).astype(np.float64)
    if a.shape[0] != 6:
        a = np.moveaxis(a, -1, 0)
    jzz, jzy, jzx, jyy, jyx, jxx = a
    M = np.stack([np.stack([jzz, jzy, jzx], -1), np.stack([jzy, jyy, jyx], -1), np.stack([jzx, jyx, jxx], -1)], -2)
    M = M.reshape(-1, 3, 3)
    tr = np.trace(M, axis1=1, axis2=2)
    keep = tr > np.percentile(tr, 50)
    w = np.linalg.eigvalsh(M[keep])
    ratio = np.abs(w[:, 1]) / np.maximum(w[:, 2], 1e-12)
    def arr(node):
        if hasattr(node, "keys"):
            node = node["0"] if "0" in node else node
        return np.asarray(node[:])
    conf = arr(g["confidence"])
    fcg = g["first_component"]
    fc = np.stack([arr(fcg[c]) for c in ("z", "y", "x")], 0).astype(np.float64)
    v = (fc - 128.0) / 127.0
    n = np.linalg.norm(v, axis=0) + 1e-9
    u = v / n
    dz = np.abs((u[:, 1:] * u[:, :-1]).sum(0)).mean()
    dy = np.abs((u[:, :, 1:] * u[:, :, :-1]).sum(0)).mean()
    dx = np.abs((u[:, :, :, 1:] * u[:, :, :, :-1]).sum(0)).mean()
    print(f"{name}: middle eigenvalue / largest (strong-gradient half): median {np.median(ratio):.4f}, 90th pct {np.percentile(ratio, 90):.4f}")
    print(f"{name}: confidence values: median {np.median(conf):.0f}, share at 255 {np.mean(conf == 255):.2%}, share >= 250 {np.mean(conf >= 250):.2%}, distinct {len(np.unique(conf))}")
    print(f"{name}: first_component neighbour coherence |dot|: z {dz:.3f}, y {dy:.3f}, x {dx:.3f} (random unit vectors ~0.5)")


for name in sys.argv[1:]:
    report(name)
