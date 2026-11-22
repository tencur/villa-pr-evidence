"""Review case 2: two overlapping patches that agree on an axis up to sign, or nearly.

vesuvius.predict (Inferer, CPU) on a 16x16x24 ramp volume with 16^3 patches and the default 50%
overlap: two patches, at x = 0 and x = 8, which weigh voxel (8, 8, 12) equally. The stand-in
network reports one axis per patch, equivariant up to sign (its components follow the mirroring,
read from the ramp's slope signs, times a sign that changes with the mirroring):

  opposite on the plane     (1, -2, 0)      and (-1, 2, 0)      exact opposites on 4x + 2y + z = 0
  either side of the plane  (1, -1.999, 0)  and (1, -2.001, 0)  the reviewer's near-boundary case
  opposite, tilted          (1, -2, 0.05)   and (-1, 2, 0.05)

as an `inplane_direction` head and as the direction of a `nearest_component` head with the
shipped losses (distance 2 and 4). Each case runs through the in-memory blend and the staged
path (logits parts + blend_logits), with default mirroring and with --disable_tta. Printed: the
saved vector length at the seam voxel (and the distance for nearest_component).

Run:  python review_overlap_seam.py <workdir>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

import vesuvius.models
from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import inference
from vesuvius.models.run.blending import merge_inference_outputs
from vesuvius.models.run.inference import Inferer

VOLUME = (16, 16, 24)
PATCH = (16, 16, 16)
SEAM = (8, 8, 12)
CASES = {
    "opposite on the plane": ((1.0, -2.0, 0.0), (-1.0, 2.0, 0.0)),
    "either side of the plane": ((1.0, -1.999, 0.0), (1.0, -2.001, 0.0)),
    "opposite, tilted": ((1.0, -2.0, 0.05), (-1.0, 2.0, 0.05)),
}


def targets_for(head):
    if head == "direction":
        cfg = {"out_channels": 3, "auxiliary_task": True, "task_type": "inplane_direction"}
    else:
        path = Path(vesuvius.models.__file__).parent / "configuration" / "aux_tasks" / "nearest_component_3d.yaml"
        losses = yaml.safe_load(path.read_text())["auxiliary_tasks"]["nearest_component"]["losses"]
        cfg = {"out_channels": 4, "auxiliary_task": True, "task_type": "nearest_component", "losses": losses}
    return {"surface": {"out_channels": 2}, head: cfg}


class PatchAxes(torch.nn.Module):
    def __init__(self, first, second, head):
        super().__init__()
        self.first, self.second, self.head = torch.tensor(first), torch.tensor(second), head

    def forward(self, x):
        s = x[:, :1].float()
        b = s.shape[0]
        fx = torch.sign(s[..., 1:] - s[..., :-1]).mean(dim=(1, 2, 3, 4))
        fy = torch.sign(s[..., 1:, :] - s[..., :-1, :]).mean(dim=(1, 2, 3, 4))
        fz = torch.sign(s[:, :, 1:] - s[:, :, :-1]).mean(dim=(1, 2, 3, 4))
        second = s.mean(dim=(1, 2, 3, 4)) > 0.2 + 0.01 * 11.5 + 0.004 * 7.5 + 0.002 * 7.5
        axis = torch.where(second[:, None], self.second, self.first)
        axis = axis / axis.norm(dim=1, keepdim=True)
        frame = axis * torch.stack([fx, fy, fz], dim=1) * (fx * fy * fz)[:, None]
        out = frame[:, :, None, None, None].expand(b, 3, *s.shape[2:])
        if self.head == "nearest":
            distance = torch.where(second, 4.0, 2.0)[:, None, None, None, None].expand_as(s)
            out = torch.cat([out, distance], dim=1)
        return {"surface": torch.cat([-s, s], dim=1), self.head: out.contiguous()}


def predict(workdir, name, model, targets, staged, **kwargs):
    def fake_load(self, checkpoint_path):
        return {"network": model, "patch_size": PATCH, "num_input_channels": 1,
                "num_seg_heads": sum(t["out_channels"] for t in targets.values()),
                "targets": targets, "normalization_scheme": "none"}

    inference.Inferer._load_train_py_model = fake_load
    out = workdir / f"{name}.zarr"
    parts = workdir / f"{name}_parts"
    blend = {} if staged else {"in_memory_blend": {"output_path": str(out)}}
    Inferer(model_path=str(workdir / "model.pth"), input_dir=str(workdir / "volume.zarr"),
            output_dir=str(parts), model_type="train_py", device="cpu", num_dataloader_workers=0,
            overlap=0.5, **blend, **kwargs).infer()
    if staged:
        merge_inference_outputs(str(parts), str(out), num_workers=1, verbose=False)
    return np.asarray(open_zarr(str(out))[:], dtype=np.float32)


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in VOLUME], indexing="ij")
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = 0.2 + 0.01 * x + 0.004 * y + 0.002 * z
    rows, k = [], 0
    for head in ("direction", "nearest"):
        targets = targets_for(head)
        for case, (first, second) in CASES.items():
            for tta_name, kwargs in (("default mirroring", {}), ("--disable_tta", {"do_tta": False})):
                lengths = []
                for staged in (False, True):
                    k += 1
                    out = predict(workdir, f"run{k}", PatchAxes(first, second, head), targets, staged, **kwargs)
                    seam = out[(slice(2, 5), *SEAM)]
                    text = f"{np.linalg.norm(seam):.5f}"
                    if head == "nearest":
                        text += f" (distance {out[(5, *SEAM)]:.3f})"
                    lengths.append(text)
                label = "inplane_direction" if head == "direction" else "nearest_component"
                rows.append(f"seam {label:17s} {case:24s} {tta_name:17s} length at {SEAM}: "
                            f"in-memory {lengths[0]} | staged {lengths[1]}")
    print()
    print("\n".join(rows))


if __name__ == "__main__":
    main()
