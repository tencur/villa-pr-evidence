"""C289 real-data proof: lasagna/mix_channels_parallel.py CLI (--channels-other nx) on real lasagna predictions: the public
PHerc0139 lasagna run 20260102150214-lasagna-20260419180421-L2 (cos, grad_mag, nx, ny; published as one OME-Zarr per
channel), level 4, z 2400-2463, y/x 800-863, packed into the 4-channel preprocessed zarr the tool reads (chunks 32^3,
preprocess_params channels + crop_xyzwhd), with crops that start inside a chunk, as preprocess_cos_omezarr.py writes them.
base and other hold the same run, so every output voxel inside the crop should equal its source.
usage: python rd_c289.py <tree>/lasagna <workdir> <lasagna_L4_4ch.npy>"""
import os, shutil, subprocess, sys
import numpy as np, zarr
L, W = sys.argv[1], sys.argv[2]
CH = ["cos", "grad_mag", "nx", "ny"]; real = np.load(sys.argv[3]); SHAPE = real.shape
print("input: public PHerc0139 lasagna prediction (cos/grad_mag/nx/ny, level 4, z 2400-2463, y/x 800-863), packed as preprocessed zarrs")
for crop in ([10, 0, 0, 40, 64, 64], [5, 7, 3, 50, 40, 45]):
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    for name in ("base", "other"):
        z = zarr.open(f"{W}/{name}.zarr", mode="w", shape=SHAPE, chunks=(1, 32, 32, 32), dtype=np.uint8, zarr_format=2, fill_value=0)
        z[:] = real; z.attrs["preprocess_params"] = {"scaledown": 1, "channels": CH, "crop_xyzwhd": crop}
    r = subprocess.run([sys.executable, f"{L}/mix_channels_parallel.py", "--base", f"{W}/base.zarr", "--other", f"{W}/other.zarr",
                        "--output", f"{W}/out.zarr", "--channels-other", "nx", "--workers", "1"], capture_output=True, text=True)
    out = zarr.open(f"{W}/out.zarr", mode="r")[:]
    x, y, zz, w, h, d = crop
    sl = (slice(zz, zz + d), slice(y, y + h), slice(x, x + w))
    wrong = sum(int((out[c][sl] != real[c][sl]).sum()) for c in range(4)); total = 4 * d * h * w
    print(f"RESULT crop x,y,z,w,h,d={crop}: exit {r.returncode}; voxels in the crop that differ from their source: {wrong:,} of {total:,} ({100 * wrong / total:.1f} %)")
