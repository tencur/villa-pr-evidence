"""Merge check: main's batched patch cache (#2684) with this branch's axis-head blending.

Staged logits with 2 segmentation channels, one sign-invariant axis head (channels 2-4) and a
distance channel (5), in overlapping patches over three parts; every patch reports the same unit
axis with a random sign. Blends with `merge_inference_outputs` for patch_cache_mb in {0, 256}
and two output chunkings, plus a reference that blends one chunk at a time through
`process_chunk` (no batching, no cache), and compares:
  - batched/cached vs reference: must be bit-identical (the #2684 contract, now with axis heads)
  - the blended axis: length ~1 (float16) everywhere a patch covers, i.e. the signs did not cancel
Also blends the same patches with InMemoryBlender and reports the largest difference.

Usage: python check_axis_heads_patch_cache.py <scratch dir>   (the vesuvius src/ on PYTHONPATH)
"""
import sys

import numpy as np
from vc_reader.reader import ZarrWriter, open_zarr

from vesuvius.models.run import blending
from vesuvius.models.run.blending import (InMemoryBlender, SpatialPatchGrid, calculate_chunks,
                                          create_blend_output, generate_gaussian_map, merge_inference_outputs)
from vesuvius.models.run.orientation import axis_head_channels, orientation_layouts

VOLUME, PATCH, C = (24, 32, 40), (8, 8, 8), 6
TARGET_INFO = {
    'surface': {'start_channel': 0, 'end_channel': 2},
    'nearest_component': {'start_channel': 2, 'end_channel': 6, 'auxiliary_task': True, 'sign_invariant': True},
}
AXIS = np.array([0.36, -0.48, 0.8], dtype=np.float32)  # unit (x, y, z)


def write_parts(parent, seed=1):
    rng = np.random.default_rng(seed)
    starts = [list(range(0, v - p + 1, p // 2)) for v, p in zip(VOLUME, PATCH)]
    coords = np.array([(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]], dtype=np.int32)
    rng.shuffle(coords)
    all_patches = np.zeros((len(coords), C, *PATCH), dtype=np.float16)
    offset = 0
    for part_id, part_coords in enumerate(np.array_split(coords, 3)):
        n = len(part_coords)
        patches = rng.normal(0, 3, size=(n, C, *PATCH)).astype(np.float32)
        sign = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        patches[:, 2:5] = sign[:, None, None, None, None] * AXIS[None, :, None, None, None]
        patches[:, 5] = rng.uniform(0, 4, size=(n, *PATCH))
        patches = patches.astype(np.float16)
        patches[rng.random(n) < 0.2] = 0  # skipped by inference, not stored
        logits = ZarrWriter.create(f"{parent}/logits_part_{part_id}.zarr", shape=patches.shape,
                                   chunks=(1, *patches.shape[1:]), dtype=np.float16,
                                   cname="zstd", level=1, skip_empty=True)
        logits[...] = patches
        logits.attrs.update(patch_size=list(PATCH), original_volume_shape=list(VOLUME), overlap=0.5,
                            target_info=TARGET_INFO, is_multi_task=True)
        positions = ZarrWriter.create(f"{parent}/coordinates_part_{part_id}.zarr", shape=part_coords.shape,
                                      chunks=part_coords.shape, dtype=np.int32)
        positions[...] = part_coords
        all_patches[offset:offset + n] = patches
        offset += n
    return coords, all_patches


def reference(parent, out, chunk_size):
    """One chunk at a time through process_chunk, no batches and no cache (as before #2684)."""
    create_blend_output(out, VOLUME, C, PATCH, chunk_size, verbose=False)
    part_files = {p: {'logits': f"{parent}/logits_part_{p}.zarr",
                      'coordinates': f"{parent}/coordinates_part_{p}.zarr"} for p in range(3)}
    index = SpatialPatchGrid(patch_size=PATCH, grid_size=1000)
    chunks = calculate_chunks(VOLUME, chunk_size or PATCH)
    index.build(parent, part_files, [0, 1, 2], PATCH, chunks=chunks)
    layouts = orientation_layouts(axis_head_channels(TARGET_INFO), C)
    blending._init_worker(part_files, out, generate_gaussian_map(PATCH), PATCH, C, False, None,
                          0, layouts)
    for chunk in chunks:
        patches = index.get_chunk_patches(chunk)
        if patches:
            blending.process_chunk(chunk, patches)
    blending._worker_state.clear()


def main(work):
    parent = f"{work}/parts"
    coords, patches = write_parts(parent)
    for chunk_size in (None, (10, 12, 12)):
        ref = f"{work}/ref_{chunk_size}.zarr"
        reference(parent, ref, chunk_size)
        want = open_zarr(ref)[...]
        covered = np.any(want != 0, axis=0)
        length = np.sqrt((want[2:5].astype(np.float32) ** 2).sum(axis=0))[covered]
        print(f"chunks {chunk_size}: reference axis length over {covered.sum()} covered voxels: "
              f"min {length.min():.5f} max {length.max():.5f}")
        for cache_mb in (256, 0):
            out = f"{work}/merged_{chunk_size}_{cache_mb}.zarr"
            merge_inference_outputs(parent, out, num_workers=2, verbose=False, chunk_size=chunk_size,
                                    patch_cache_mb=cache_mb)
            got = open_zarr(out)[...]
            print(f"chunks {chunk_size}: merge_inference_outputs patch_cache_mb={cache_mb:3d} "
                  f"bit-identical to reference: {got.dtype == want.dtype and np.array_equal(got, want)}")
        mem = InMemoryBlender(coords, PATCH, C, VOLUME, axis_channels=axis_head_channels(TARGET_INFO))
        for i in range(len(coords)):
            if np.any(patches[i]):
                mem.add(i, patches[i])
        out = f"{work}/inmemory_{chunk_size}.zarr"
        mem.prepare_output(out, chunk_size, verbose=False)
        mem.write(out, chunk_size, verbose=False)
        got = open_zarr(out)[...].astype(np.float32)
        print(f"chunks {chunk_size}: InMemoryBlender max |diff| vs staged reference: "
              f"{np.abs(got - want.astype(np.float32)).max():.5f}")


if __name__ == '__main__':
    main(sys.argv[1])
