"""C615 real-data check: the nnunet README chain grids_from_zarr -> (prediction) -> apply_thresholds -> grids_to_zarr
on a 500^3 crop of the public PHerc0332 surface prediction (uint8 0/255, standing in for the ensembled softmax, which
needs the nnU-Net models). grids_to_zarr.main() runs with the arguments its __main__ ships (map = True and its
value_map literal, read from the tree under test; trim 50; grid 500) except paths and the volume shape.
grids_to_zarr.main() uses zarr 2 APIs (zarr.DirectoryStore, zarr.Blosc), so run this with zarr 2.
Usage: check_c615.py <villa tree> <work dir>"""
import ast, contextlib, io, json, os, sys, urllib.request
import multiprocessing
multiprocessing.set_start_method("fork", force=True)  # Python 3.14 defaults to forkserver; this script has no __main__ guard
import numpy as np
import tifffile
import zarr

T, W = sys.argv[1], sys.argv[2]
SCRIPTS = os.path.join(T, "segmentation/models/arch/nnunet/scripts")
sys.path.insert(0, SCRIPTS)
import grids_from_zarr, apply_thresholds, grids_to_zarr  # noqa: E402

URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/representations/predictions/surfaces/"
       "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr/0")
src = os.path.join(W, "src.zarr")
os.makedirs(src, exist_ok=True)
meta = json.loads(urllib.request.urlopen(URL + "/.zarray").read())
with open(os.path.join(src, ".zarray"), "w") as f:
    json.dump(meta, f)
C = meta["chunks"][0]
sep = meta.get("dimension_separator", ".")
for z in (20, 21, 22):
    for y in (9, 10, 11):
        for x in (9, 10, 11):
            key = sep.join(map(str, (z, y, x)))
            p = os.path.join(src, *key.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(f"{URL}/{key}", p)
z0, y0, x0 = 20 * C, 9 * C, 9 * C
crop = zarr.open_array(src, mode="r")[z0:z0 + 500, y0:y0 + 500, x0:x0 + 500]
vol = os.path.join(W, "vol.zarr")
g = zarr.open_group(vol, mode="w")
g.create_dataset("0", shape=crop.shape, chunks=(128, 128, 128), dtype=crop.dtype)[:] = crop
print(f"PHerc0332 surface prediction crop z/y/x {z0}/{y0}/{x0} + 500^3: values {sorted(np.unique(crop).tolist())}, "
      f"{int(np.count_nonzero(crop)):,} nonzero voxels")

tree = ast.parse(open(os.path.join(SCRIPTS, "grids_to_zarr.py")).read())
value_map = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Dict) and any(getattr(t, "id", None) == "value_map" for t in n.targets))
print(f"grids_to_zarr.py ships map = True with value_map {value_map}")

grids = os.path.join(W, "grids")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    grids_from_zarr.process_block((0, 0, 0, vol, 0, grids, 500, 50, crop.shape))   # 600^3 block, overlap 50
    apply_thresholds.process_folder(grids, os.path.join(W, "thr"), num_processes=1)  # README: apply_thresholds.py
    za = grids_to_zarr.main(os.path.join(W, "thr", "otsu"), os.path.join(W, "out.zarr"), trim_size=50,
                            full_volume_shape=(500, 500, 500), max_workers=1, chunk_size=128, value_map=value_map)
said = [l for l in out.getvalue().splitlines() if "processed successfully" in l or "failed" in l]
thr = tifffile.imread(os.path.join(W, "thr", "otsu", "cell_yxz_001_001_001.tif"))
expected = int(np.count_nonzero(thr[50:550, 50:550, 50:550]))
res = za[:]
print(f"apply_thresholds output values {sorted(np.unique(thr).tolist())}; foreground in the trimmed block: {expected:,}")
print(f"grids_to_zarr says: {said[0].strip() if said else '(nothing)'}")
print(f"combined zarr: {int(np.count_nonzero(res)):,} nonzero voxels; equal to the thresholded block: "
      f"{np.array_equal(res, thr[50:550, 50:550, 50:550])}")
