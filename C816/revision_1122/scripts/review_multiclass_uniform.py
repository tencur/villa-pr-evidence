"""Review case 3: `--mode multiclass --threshold` on a chunk where one nonzero class wins everywhere.

Runs the real CLI entry points on CPU; only the checkpoint loader is replaced by a stand-in
network with two segmentation scores that always favour class 1 ([0, 0.5]) and a structure
tensor head (auxiliary) whose first component (3 * input, a sphere) outscores them in part of
the 32^3 volume. Default mirroring, 16^3 patches.

  in-memory:          vesuvius.predict --blend_in_memory OUT --mode multiclass --threshold
  staged finalize:    vesuvius.predict (logits parts), vesuvius.blend_logits, then
                      vesuvius.finalize_outputs --mode multiclass --threshold --delete_intermediates
  blend_and_finalize: vesuvius.predict (logits parts), vesuvius.blend_and_finalize --mode multiclass --threshold

Printed: the saved class histogram (the argmax channel) over the whole volume.

Run:  python review_multiclass_uniform.py <workdir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import blending, finalize_outputs, inference

VOLUME = (32, 32, 32)
PATCH = (16, 16, 16)
TARGETS = {"surface": {"out_channels": 2, "activation": "none"},
           "st": {"out_channels": 6, "auxiliary_task": True, "task_type": "structure_tensor",
                  "activation": "none"}}


class Net(torch.nn.Module):
    def forward(self, x):
        s = x[:, :1].float()
        st = torch.cat([3 * s] + [torch.zeros_like(s)] * 5, dim=1)
        return {"surface": torch.cat([torch.zeros_like(s), torch.full_like(s, 0.5)], dim=1), "st": st}


def fake_load(self, checkpoint_path):
    return {"network": Net(), "patch_size": PATCH, "num_input_channels": 1, "num_seg_heads": 8,
            "targets": TARGETS, "normalization_scheme": "none"}


def cli(entry, argv):
    sys.argv = argv
    try:
        rc = entry()
    except SystemExit as e:
        rc = e.code
    return 0 if rc is None else rc


def histogram(path):
    out = np.asarray(open_zarr(str(path))[:])
    classes, counts = np.unique(out[-1], return_counts=True)
    return out.shape, ", ".join(f"class {c}: {n}" for c, n in zip(classes.tolist(), counts.tolist()))


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in VOLUME], indexing="ij")
    r = np.sqrt(sum((g - 15.7) ** 2 for g in grids))
    vol = (1.0 / (1.0 + np.exp((r - 9.0) / 1.5))).astype(np.float32)
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = vol
    print(f"voxels where the tensor component outscores class 1: {(3 * vol > 0.5).sum()} of {vol.size}")
    inference.Inferer._load_train_py_model = fake_load
    predict = ["vesuvius.predict", "--model_path", str(workdir / "model.pth"), "--input_dir",
               str(workdir / "volume.zarr"), "--model_type", "train_py", "--device", "cpu", "--num_workers", "0"]
    rows = []
    rc = cli(inference.main, predict + ["--output_dir", str(workdir / "mem_parts"), "--blend_in_memory",
                                        str(workdir / "mem.zarr"), "--mode", "multiclass", "--threshold"])
    rows.append(("in-memory (predict --blend_in_memory)", rc, workdir / "mem.zarr"))
    rc = cli(inference.main, predict + ["--output_dir", str(workdir / "parts")])
    rc |= cli(blending.main, ["vesuvius.blend_logits", str(workdir / "parts"), str(workdir / "logits.zarr"),
                              "--num_workers", "1", "--quiet"])
    rc |= cli(finalize_outputs.main, ["vesuvius.finalize_outputs", str(workdir / "logits.zarr"),
                                      str(workdir / "final.zarr"), "--mode", "multiclass", "--threshold",
                                      "--delete_intermediates", "--num_workers", "1"])
    rows.append((f"staged finalize (logits deleted: {not (workdir / 'logits.zarr').exists()})", rc,
                 workdir / "final.zarr"))
    rc = cli(blending.blend_and_finalize_main, ["vesuvius.blend_and_finalize", str(workdir / "parts"),
                                                str(workdir / "fused.zarr"), "--mode", "multiclass",
                                                "--threshold", "--num_workers", "1", "--quiet"])
    rows.append(("blend_and_finalize", rc, workdir / "fused.zarr"))
    print()
    for label, rc, path in rows:
        shape, hist = histogram(path)
        print(f"multiclass --threshold, {label}: exit {rc}, output shape {shape}; saved {hist}")


if __name__ == "__main__":
    main()
