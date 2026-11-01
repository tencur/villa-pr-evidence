"""C289 proof: lasagna/mix_channels_parallel.py CLI (--channels-other nx) on two synthetic preprocessed prediction
zarrs (4 x 64^3 uint8, chunks 32^3, every voxel non-zero) whose preprocess_params.crop_xyzwhd starts inside a chunk.
Compares the output with the inputs at the same voxels.
usage: python c289_proof.py <tree>/lasagna <workdir>"""
import os, shutil, subprocess, sys
import numpy as np, zarr
L, W = sys.argv[1], sys.argv[2]
CH = ["cos", "grad_mag", "nx", "ny"]; SHAPE = (4, 64, 64, 64)
for crop in ([10, 0, 0, 40, 64, 64], [5, 7, 3, 50, 40, 45]):
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    rng = np.random.default_rng(0); src = {}
    for name in ("base", "other"):
        a = rng.integers(1, 256, SHAPE, dtype=np.uint8)
        z = zarr.open(f"{W}/{name}.zarr", mode="w", shape=SHAPE, chunks=(1, 32, 32, 32), dtype=np.uint8, zarr_format=2, fill_value=0)
        z[:] = a; z.attrs["preprocess_params"] = {"scaledown": 1, "channels": CH, "crop_xyzwhd": crop}; src[name] = a
    r = subprocess.run([sys.executable, f"{L}/mix_channels_parallel.py", "--base", f"{W}/base.zarr", "--other", f"{W}/other.zarr",
                        "--output", f"{W}/out.zarr", "--channels-other", "nx", "--workers", "1"], capture_output=True, text=True)
    out = zarr.open(f"{W}/out.zarr", mode="r")[:]
    x, y, zz, w, h, d = crop
    sl = (slice(zz, zz + d), slice(y, y + h), slice(x, x + w))
    wrong = sum(int((out[c][sl] != (src["other"] if n == "nx" else src["base"])[c][sl]).sum()) for c, n in enumerate(CH))
    total = 4 * d * h * w
    print(f"RESULT crop x,y,z,w,h,d={crop}: exit {r.returncode}; voxels in the crop that differ from their source: {wrong:,} of {total:,} ({100 * wrong / total:.1f} %)")
