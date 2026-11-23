"""One direct Fiber inference run over an existing (or new) compact output, with a resume report.

Uses the same synthetic 160^3 volume, 64^3 checkpoint config and presence-1 stand-in model as
default_coverage.py, so a correct output holds presence 255 at every voxel.

usage: python resume_tiling.py <workdir> <output dir name> [--remove-chunk] [extra infer args...]

--remove-chunk deletes the finest chunk at z, y, x = 64 (sd0/0/0/1/1/1) before the run, as an
interrupted run would leave it missing. The report compares the output directory before and after
the run: finest chunks rebuilt (rewritten or new) and untouched (reused, if the run completed), whether any file of the
store or inference.json changed, the overlap and status inference.json records, the tiling the store
records, and the voxels with presence 0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from default_coverage import ConstantPresence, report, write_volume


def files(directory: Path) -> dict[str, tuple[int, int, str]]:
    return {p.relative_to(directory).as_posix(): (p.stat().st_ino, p.stat().st_mtime_ns,
                                                  hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(directory.rglob("*")) if p.is_file()}


def main() -> None:
    workdir, name = Path(sys.argv[1]), sys.argv[2]
    extra = sys.argv[3:]
    remove = "--remove-chunk" in extra
    extra = [a for a in extra if a != "--remove-chunk"]
    workdir.mkdir(parents=True, exist_ok=True)
    vol = workdir / "volume160.zarr"
    if not vol.exists():
        write_volume(vol, 160)
    ckpt = workdir / "model.pt"
    torch.save({"config": {"patch_shape_zyx": [64, 64, 64], "image_normalization": "none",
                           "model_3d": {"output_stride": 1},
                           "training": {"mixed_precision": "off"}}}, ckpt)
    import vesuvius.neural_tracing.fiber_trace_3d.inference_adapter as adapter
    import vesuvius.neural_tracing.fiber_trace_3d.train as train
    from vesuvius.neural_tracing.fiber_trace_3d import infer
    adapter.build_fiber_trace_3d_model = lambda config: ConstantPresence()
    train._load_snapshot = lambda *a, **k: None

    outdir = workdir / name
    out = outdir / "fiber.zarr"
    if remove:
        (out / "sd0" / "0" / "0" / "1" / "1" / "1").unlink()
        print("removed finest chunk sd0/0/0/1/1/1")
    before = files(outdir) if outdir.exists() else {}
    argv = ["--input", str(vol / "0"), "--output", str(out), "--checkpoint", str(ckpt),
            "--device", "cpu", *extra]
    print("infer argv:", " ".join(a.replace(str(workdir), "<wd>") for a in argv), flush=True)
    try:
        infer.main(argv)
        print("result: completed")
    except Exception as error:  # report the refusal instead of a traceback
        print(f"result: {type(error).__name__}: {str(error).replace(str(workdir), '<wd>')}")
    after = files(outdir)

    finest = [k for k in after if k.startswith("fiber.zarr/sd0/0/") and not k.split("/")[-1].startswith(".")]
    rebuilt = [k for k in finest if before.get(k, (None, None))[:2] != after[k][:2]]
    print(f"finest chunks: {len(finest)} stored, {len(rebuilt)} rebuilt, {len(finest) - len(rebuilt)} untouched"
          + (f" (rebuilt: {', '.join(k.removeprefix('fiber.zarr/sd0/0/') for k in rebuilt)})" if 0 < len(rebuilt) < 4 else ""))
    store_before = {k: v[2] for k, v in before.items() if k.startswith("fiber.zarr/")}
    store_after = {k: v[2] for k, v in after.items() if k.startswith("fiber.zarr/")}
    if before:
        print(f"store files unchanged: {store_before == store_after}; "
              f"inference.json unchanged: {before.get('inference.json') == after.get('inference.json')}")
    provenance = json.loads((outdir / "inference.json").read_text())
    print(f"inference.json: overlap {provenance['inference']['overlap']}, border {provenance['inference']['border']}, "
          f"status {provenance.get('status')!r}")
    attrs = json.loads((out / ".zattrs").read_text())
    print(f"store tiling record: {attrs.get('fiber_inference_tiling')}")
    report(out)


if __name__ == "__main__":
    main()
