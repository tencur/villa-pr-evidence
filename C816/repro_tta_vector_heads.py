"""vesuvius.predict TTA on vector / tensor heads, end to end through Inferer.

A multi-task "checkpoint" with a segmentation head, a surface-normals head (3 channels,
x y z, as the surface_normals auxiliary task trains them) and a structure-tensor head
(6 channels Jzz Jzy Jzx Jyy Jyx Jxx, as the structure_tensor auxiliary task trains them).
The network is an exactly flip/transpose-equivariant oracle (central-difference gradient
of the input), so a correct TTA must return the same values as no TTA.

Run:  python repro_tta_vector_heads.py <workdir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer

VOLUME = (48, 48, 48)
PATCH = (24, 24, 24)


def make_volume():
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in VOLUME], indexing="ij")
    # a ball and a tilted sheet: normals point every way
    r = np.sqrt((z - 20.3) ** 2 + (y - 26.1) ** 2 + (x - 22.7) ** 2)
    ball = 1.0 / (1.0 + np.exp((r - 11.0) / 1.5))
    d = (0.36 * z - 0.48 * y + 0.8 * x) / np.sqrt(0.36 ** 2 + 0.48 ** 2 + 0.8 ** 2)
    sheet = np.exp(-((d - 30.0) / 2.5) ** 2)
    return (ball + sheet).astype(np.float32)


class Oracle(torch.nn.Module):
    """Heads computed from the input gradient: equivariant to any flip and axis swap."""

    def forward(self, x):
        s = x[:, :1].float()
        gz, gy, gx = torch.gradient(s, dim=(2, 3, 4))
        g = torch.cat([-gx, -gy, -gz], dim=1)  # outward normal, x y z order
        n = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
        st = torch.cat([gz * gz, gz * gy, gz * gx, gy * gy, gy * gx, gx * gx], dim=1)
        return {"surface": torch.cat([-s, s], dim=1), "normals": n, "st": 10.0 * st}


TARGETS = {
    "surface": {"out_channels": 2},
    "normals": {"out_channels": 3, "auxiliary_task": True, "task_type": "surface_normals",
                "source_target": "surface", "activation": "none"},
    "st": {"out_channels": 6, "auxiliary_task": True, "task_type": "structure_tensor",
           "source_target": "surface", "activation": "none"},
}


def fake_load(self, checkpoint_path):
    return {"network": Oracle(), "patch_size": PATCH, "num_input_channels": 1,
            "num_seg_heads": 11, "targets": TARGETS, "normalization_scheme": "none"}


def run(workdir: Path, **kwargs):
    inference.Inferer._load_train_py_model = fake_load
    out = workdir / ("blend_tta.zarr" if kwargs.get("do_tta", True) else "blend_notta.zarr")
    inferer = Inferer(model_path=str(workdir / "model.pth"), input_dir=str(workdir / "volume.zarr"),
                      output_dir=str(workdir / "parts"), model_type="train_py", device="cpu",
                      num_dataloader_workers=0, overlap=0.5, in_memory_blend={"output_path": str(out)},
                      **kwargs)
    inferer.infer()
    return np.asarray(open_zarr(str(out))[:], dtype=np.float32)


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    vol = make_volume()
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = vol

    ref = run(workdir, do_tta=False)
    results = {"no TTA": ref}
    for tta_type in ("mirroring", "rotation"):
        results[f"TTA {tta_type}"] = run(workdir, do_tta=True, tta_type=tta_type)

    # voxels where the normal is defined: strong gradient, away from the volume faces
    gt = Oracle()(torch.from_numpy(vol)[None, None])
    gmag = np.sqrt(sum(c ** 2 for c in torch.gradient(torch.from_numpy(vol))).numpy())
    m = gmag > 0.05
    m[:2], m[-2:], m[:, :2], m[:, -2:], m[:, :, :2], m[:, :, -2:] = False, False, False, False, False, False
    gt_n = gt["normals"][0].numpy()
    gt_st = gt["st"][0].numpy()
    print(f"voxels compared: {int(m.sum())}")
    for name, out in results.items():
        seg = out[0:2]
        n = out[2:5]
        st = out[5:11]
        nlen = np.sqrt((n ** 2).sum(0))[m]
        cos = (n * gt_n).sum(0)[m] / np.maximum(nlen, 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        off = np.abs(st[[1, 2, 4]])[:, m].mean()
        off_gt = np.abs(gt_st[[1, 2, 4]])[:, m].mean()
        diag = np.abs(st[[0, 3, 5]] - gt_st[[0, 3, 5]])[:, m].mean()
        print(f"{name:16s} seg max|diff vs no-TTA| {np.abs(seg - ref[0:2]).max():.4f} | normals mean |n| "
              f"{nlen.mean():.4f}, median angle to truth {np.median(ang):6.2f} deg, "
              f">10 deg {100 * (ang > 10).mean():5.1f}% | tensor mean |off-diag| {off:.4f} "
              f"(truth {off_gt:.4f}), mean |diag err| {diag:.4f}")


if __name__ == "__main__":
    main()
