"""Review case: `vesuvius.predict --blend_in_memory OUT --mode multiclass` on a multi-task checkpoint.

Runs the real CLI entry point (vesuvius.models.run.inference.main) on CPU; only the checkpoint
loader is replaced by a stand-in network. The network scores segmentation classes with constants
and predicts unit surface normals from the input gradient (an `surface_normals` auxiliary head,
3 channels x y z). The volume is a ramp falling along x, so the normal is +x (x component 1.0).

  case A: shipped surface_normals_3d layout, 2 surface scores [0, 0.5] + 3 normal components,
          default TTA (mirroring). Expected class 1.
  case B: 3 surface scores [0, 0.5, 0.2] + 3 normal components, --disable_tta. Expected class 1.

Run:  python review_multiclass_cli.py <workdir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import inference

VOLUME = (32, 32, 32)
PATCH = (16, 16, 16)


class Net(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, x):
        s = x[:, :1].float()
        gz, gy, gx = torch.gradient(s, dim=(2, 3, 4))
        g = torch.cat([-gx, -gy, -gz], dim=1)
        n = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return {"surface": torch.cat([torch.full_like(s, v) for v in self.scores], dim=1), "normals": n}


def case(workdir, name, scores, extra_args):
    targets = {
        "surface": {"out_channels": len(scores), "activation": "none"},
        "normals": {"out_channels": 3, "auxiliary_task": True, "task_type": "surface_normals",
                    "source_target": "surface", "activation": "none"},
    }

    def fake_load(self, checkpoint_path):
        return {"network": Net(scores), "patch_size": PATCH, "num_input_channels": 1,
                "num_seg_heads": len(scores) + 3, "targets": targets, "normalization_scheme": "none"}

    inference.Inferer._load_train_py_model = fake_load
    out = workdir / f"{name}.zarr"
    sys.argv = ["vesuvius.predict", "--model_path", str(workdir / "model.pth"),
                "--input_dir", str(workdir / "volume.zarr"), "--output_dir", str(workdir / f"{name}_parts"),
                "--model_type", "train_py", "--device", "cpu", "--num_workers", "0",
                "--blend_in_memory", str(out), "--mode", "multiclass", *extra_args]
    rc = inference.main()
    result = np.asarray(open_zarr(str(out))[:])
    classes, counts = np.unique(result[-1, 2:-2, 2:-2, 2:-2], return_counts=True)
    hist = ", ".join(f"class {c}: {n}" for c, n in zip(classes.tolist(), counts.tolist()))
    return rc, result.shape, int(result[-1, 16, 16, 16]), hist


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    x = np.arange(VOLUME[2], dtype=np.float32)
    vol = np.broadcast_to(0.6 - 0.005 * x, VOLUME).astype(np.float32)
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = vol
    results = [
        ("A: 2 surface scores + normals, default TTA", case(workdir, "a", [0.0, 0.5], [])),
        ("B: 3 surface scores + normals, --disable_tta", case(workdir, "b", [0.0, 0.5, 0.2], ["--disable_tta"])),
    ]
    print()
    for label, (rc, shape, centre, hist) in results:
        print(f"case {label}: exit {rc}, output shape {shape}, class at (16, 16, 16) = {centre} "
              f"(expected 1); interior: {hist}")


if __name__ == "__main__":
    main()
