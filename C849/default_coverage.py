"""Exact oracle for direct fiber inference coverage.

Writes a synthetic uint8 OME-Zarr volume (level 0, constant value) and a checkpoint with an
embedded config (patch 64^3), then runs the README command
    python -m vesuvius.neural_tracing.fiber_trace_3d.infer --input <vol>/0 --output <out>/fiber.zarr --checkpoint <ckpt>
in-process through ``infer.main`` (plus --device cpu and any extra arguments).
Only the model builder and the snapshot loader are replaced: the stand-in model predicts
presence 1 at every voxel, so the correct output is presence 255 at every voxel.

usage: python default_coverage.py <workdir> <run name> <size> [extra infer args...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


class ConstantPresence(torch.nn.Module):
    def forward(self, volume, **kwargs):
        out = torch.full((volume.shape[0], 7, *volume.shape[2:]), 0.5)
        out[:, 6] = 1.0
        return out


def write_volume(root: Path, n: int) -> None:
    """Zarr v2 OME-Zarr group with one level "0" of constant uint8 100."""
    from vc_reader.reader import create_group
    group = create_group(root, overwrite=True)
    arr = group.create_array("0", shape=(n, n, n), chunks=(32, 32, 32), dtype=np.uint8,
                             compressor=None, separator="/")
    arr[:] = np.full((n, n, n), 100, np.uint8)
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4", "name": "synthetic",
        "axes": [{"name": a, "type": "space"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]}))


def runs(mask: np.ndarray) -> str:
    """Index ranges where mask is True, e.g. '32-47, 80-95'."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return "none"
    parts, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            parts.append(f"{start}-{a}")
            start = b
    parts.append(f"{start}-{idx[-1]}")
    return ", ".join(parts)


def report(out: Path) -> None:
    from vc_reader.reader import open_zarr
    manifest = json.loads((out / ".zattrs").read_text())["lasagna"]
    group = next(g for g in manifest["groups"].values() if "presence" in g["channels"])
    data = np.asarray(open_zarr(out / group["zarr"])[:])
    presence = data[group["channels"].index("presence")]
    zero = presence == 0
    print(f"presence shape {presence.shape}, values {np.unique(presence).tolist()}")
    print(f"voxels with presence 0: {int(zero.sum())} of {zero.size} "
          f"(fraction {zero.mean():.3f}); expected 0")
    for axis, name in ((2, "x"), (1, "y"), (0, "z")):
        other = tuple(a for a in (0, 1, 2) if a != axis)
        empty = zero.all(axis=other)
        print(f"{name}-planes entirely presence 0: {int(empty.sum())} ({runs(empty)})")
    # With validity_channel "chunks" a voxel is valid iff its stored chunk exists, so say whether
    # the presence-0 voxels sit in stored chunks and what every channel holds there.
    arr = out / group["zarr"]
    chunks = json.loads((arr / ".zarray").read_text())["chunks"][1:]
    total = stored = holes = holes_stored = 0
    for z in range(0, presence.shape[0], chunks[0]):
        for y in range(0, presence.shape[1], chunks[1]):
            for x in range(0, presence.shape[2], chunks[2]):
                key = arr.joinpath("0", str(z // chunks[0]), str(y // chunks[1]), str(x // chunks[2]))
                has = key.exists()
                total, stored = total + 1, stored + has
                if zero[z:z + chunks[0], y:y + chunks[1], x:x + chunks[2]].any():
                    holes, holes_stored = holes + 1, holes_stored + has
    print(f"validity_channel {manifest.get('validity_channel')!r}, blank_invalid {manifest.get('blank_invalid')!r}: "
          f"{stored} of {total} chunks stored; {holes_stored} of {holes} chunks with presence-0 voxels are stored")
    def few(values):
        values = np.unique(values)
        return f"{values[:6].tolist()}{' ...' if values.size > 6 else ''}"
    if zero.any():
        for name, channel in zip(group["channels"], data):
            print(f"  {name} at the presence-0 voxels: {few(channel[zero])}; elsewhere: {few(channel[~zero])}")


def main() -> None:
    workdir, run, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    extra = sys.argv[4:]
    workdir.mkdir(parents=True, exist_ok=True)
    vol = workdir / f"volume{n}.zarr"
    if not vol.exists():
        write_volume(vol, n)
    ckpt = workdir / "model.pt"
    torch.save({"config": {"patch_shape_zyx": [64, 64, 64], "image_normalization": "none",
                           "model_3d": {"output_stride": 1},
                           "training": {"mixed_precision": "off"}}}, ckpt)
    import vesuvius.neural_tracing.fiber_trace_3d.inference_adapter as adapter
    import vesuvius.neural_tracing.fiber_trace_3d.train as train
    from vesuvius.neural_tracing.fiber_trace_3d import infer
    adapter.build_fiber_trace_3d_model = lambda config: ConstantPresence()
    train._load_snapshot = lambda *a, **k: None
    out = workdir / run
    argv = ["--input", str(vol / "0"), "--output", str(out / "fiber.zarr"),
            "--checkpoint", str(ckpt), "--device", "cpu", *extra]
    print("infer argv:", " ".join(a.replace(str(workdir), "<wd>") for a in argv), flush=True)
    infer.main(argv)
    report(out / "fiber.zarr")


if __name__ == "__main__":
    main()
