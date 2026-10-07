"""optimized_inference: STEP=prepare writes only layers [START, END) into the surface-volume zarr; STEP=inference
then slices that zarr again with start_z=START, end_z=END. Uses the pipeline's own functions on real CT slices.

Usage: python repro_layer_window.py <optimized_inference dir> <ct_zarr_level> <work_dir>
"""
import os, sys
from pathlib import Path
import numpy as np, tifffile, zarr

code, ct_path, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(code)); os.chdir(code)
import processing, inference  # pipeline modules

work.mkdir(parents=True, exist_ok=True)
ct = zarr.open(ct_path, mode="r")
layers_dir = work / "layers"; layers_dir.mkdir(exist_ok=True)
N = 64
for i in range(N):                                   # real CT slices as the segment's layer stack, named like a segment's layers/ dir
    tifffile.imwrite(layers_dir / f"{i:02d}.tif", np.asarray(ct[300 + i, 100:356, 100:356]).astype(np.uint8))
START, END = 1, 63                                   # README's recommended window for resnet3d-152-3d-decoder
# STEP=prepare: list_layers_objects keeps [START, END) (entrypoint.py:364) and create_surface_volume_zarr stacks them
kept = [str(layers_dir / f"{i:02d}.tif") for i in range(START, END)]
zpath = str(work / f"surface_volume_{START:02d}_{END:02d}.zarr")
import shutil; shutil.rmtree(zpath, ignore_errors=True)
processing.create_surface_volume_zarr(kept, zpath, chunk_size=256, use_compression=False)
z = zarr.open(zpath, mode="r")
print(f"prepare: zarr shape {z.shape} holds source layers {START}..{END - 1} (channel k = layer {START}+k)")
# STEP=inference: run_inference(..., start_z=START, end_z=END) -> LayersSource(zarr, start_z, end_z)
if hasattr(processing, "resolve_zarr_layer_window"):          # fixed code: prepare records the window, inference rebases it
    processing.record_layer_window(zpath, START, END)
    s0, e0 = processing.resolve_zarr_layer_window(zpath, START, END)
    print(f"fixed: requested layers [{START}, {END}) -> zarr channels [{s0}, {e0})")
else:
    s0, e0 = START, END
src = inference.LayersSource(zpath, start_z=s0, end_z=e0)
h, w, c = src.shape
print(f"inference: LayersSource(start_z={s0}, end_z={e0}) -> {c} channels (model trained on {END - START})")
# which source layer does the model's first channel actually see?
tile = src.read_tile(0, 0, 64, 64) if hasattr(src, "read_tile") else None
if tile is None:
    for name in ("get_tile", "read", "__getitem__"):
        if hasattr(src, name):
            try:
                tile = getattr(src, name)(0, 0, 64, 64) if name != "__getitem__" else src[0:64, 0:64, :]
                break
            except Exception:
                tile = None
if tile is None:
    tile = np.asarray(z[0:64, 0:64, s0:e0])
tile = np.asarray(tile)
first = tile[..., 0] if tile.shape[-1] == c else tile[0]
matches = [i for i in range(N) if np.array_equal(first, np.asarray(ct[300 + i, 100:164, 100:164]).astype(np.uint8))]
print(f"the first channel the model receives is source layer {matches} (requested first layer: {START})")
