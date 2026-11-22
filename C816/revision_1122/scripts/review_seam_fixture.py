"""Review fixture: two overlapping patches predict one in-plane axis with opposite signs.

vesuvius.predict (Inferer + in-memory blend, CPU) on a 16x16x24 volume with 16^3 patches and the
default 50% overlap: two patches, at x = 0 and x = 8. The network reports the x axis everywhere
in an `inplane_direction` head (task_type inplane_direction: an eigenvector, trained with a
sign-invariant loss), with a sign picked by the patch's first voxel. The volume is bright except
voxel (0, 0, 8), the second patch's first voxel, so that patch reports -x unmirrored and +x in
the 7 mirrored variants; the first patch reports +x in all 8. Voxel (8, 8, 12) gets the same
Gaussian weight from both patches.

Run:  python review_seam_fixture.py <workdir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import inference
from vesuvius.models.run.blending import generate_gaussian_map
from vesuvius.models.run.inference import Inferer

VOLUME = (16, 16, 24)
PATCH = (16, 16, 16)
TARGETS = {
    "surface": {"out_channels": 2},
    "direction": {"out_channels": 3, "auxiliary_task": True, "task_type": "inplane_direction",
                  "source_target": "surface", "activation": "none"},
}


class AxisWithPatchSign(torch.nn.Module):
    def forward(self, x):
        s = x[:, :1].float()
        sign = torch.where(s[:, :, :1, :1, :1] > 0.5, 1.0, -1.0).expand_as(s)
        return {"surface": torch.cat([-s, s], dim=1),
                "direction": torch.cat([sign, torch.zeros_like(s), torch.zeros_like(s)], dim=1)}


def fake_load(self, checkpoint_path):
    return {"network": AxisWithPatchSign(), "patch_size": PATCH, "num_input_channels": 1,
            "num_seg_heads": 5, "targets": TARGETS, "normalization_scheme": "none"}


def run(workdir, name, **kwargs):
    inference.Inferer._load_train_py_model = fake_load
    out = workdir / f"{name}.zarr"
    Inferer(model_path=str(workdir / "model.pth"), input_dir=str(workdir / "volume.zarr"),
            output_dir=str(workdir / f"{name}_parts"), model_type="train_py", device="cpu",
            num_dataloader_workers=0, overlap=0.5, in_memory_blend={"output_path": str(out)},
            **kwargs).infer()
    return np.asarray(open_zarr(str(out))[:], dtype=np.float32)


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    vol = np.full(VOLUME, 0.9, dtype=np.float32) + np.linspace(0, 0.01, VOLUME[2], dtype=np.float32)
    vol[0, 0, 8] = 0.0
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = vol

    g = generate_gaussian_map(PATCH)[0]
    print(f"Gaussian weight at voxel (8, 8, 12): patch x=0 (index 12) {g[8, 8, 12]:.6f}, "
          f"patch x=8 (index 4) {g[8, 8, 4]:.6f}")
    for name, kwargs in (("mirroring TTA (default)", {}), ("no TTA", {"do_tta": False})):
        out = run(workdir, name.split()[0].replace("no", "notta"), **kwargs)
        d = out[2:5]
        inner = d[:, 4:12, 4:12, 4:20]
        print(f"{name:24s} blended direction at the seam voxel (8, 8, 12): "
              f"x {d[0, 8, 8, 12]:.4f}, y {d[1, 8, 8, 12]:.4f}, z {d[2, 8, 8, 12]:.4f} | "
              f"interior min |x| {np.abs(inner[0]).min():.4f}")


if __name__ == "__main__":
    main()
