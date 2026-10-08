import numpy as np, zarr
g = zarr.open_group("/tmp/c94_in.zarr", mode="r")
ev = g["eigenvectors"][...]; n = ev.reshape(3,3,*ev.shape[1:])[2].astype(np.float64)
material = np.linalg.norm(n, axis=0) > 0.5
a = np.abs(n); srt = np.sort(a, axis=0); gap = srt[2] - srt[1]   # margin between largest and 2nd-largest |component|
for axis, name in enumerate("zyx"):
    sa=[slice(None)]*3; sb=[slice(None)]*3; sa[axis]=slice(0,-1); sb[axis]=slice(1,None)
    both = material[tuple(sa)] & material[tuple(sb)]
    dot = (n[(slice(None),*sa)]*n[(slice(None),*sb)]).sum(0)
    anti = (dot < 0) & both
    g_anti = np.minimum(gap[tuple(sa)], gap[tuple(sb)])[anti]
    g_all = np.minimum(gap[tuple(sa)], gap[tuple(sb)])[both]
    near = np.abs(dot[anti]) > 0.9
    print(f"{name}: antiparallel pairs {anti.sum()}; of these |dot|>0.9 (same axis, flipped): {100*near.mean():.1f}%; "
          f"median seam margin {np.median(g_anti):.3f} (all pairs: {np.median(g_all):.3f}); margin<0.1: {100*(g_anti<0.1).mean():.1f}% (all pairs: {100*(g_all<0.1).mean():.1f}%)")
