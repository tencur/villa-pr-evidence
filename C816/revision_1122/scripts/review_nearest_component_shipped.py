"""Review case 1: the shipped nearest_component losses under default mirroring.

nearest_component_3d.yaml trains the direction (all channels but the last) with
SignInvariantCosineLoss and the distance (last channel) with SmoothL1Loss. The stand-in network
reports the x axis with the same sign in every mirrored frame, at distance 2, which that loss
scores as perfect wherever x is the true axis. The head's losses are read from the yaml file of
the vesuvius tree under test.

Rows: vesuvius.predict (Inferer) with the in-memory blend and with the staged path
(logits parts + blend_logits), default mirroring and --disable_tta; then the two mirroring
helpers called directly on one 16^3 patch. Printed: the saved (x, y, z, distance) at voxel
(16, 16, 16) of a 32^3 volume (16^3 patches, 50% overlap).

Run:  python review_nearest_component_shipped.py <workdir>
"""
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

import vesuvius.models
from vc_reader.reader import ZarrWriter, open_zarr
from vesuvius.models.run import inference, tta
from vesuvius.models.run.blending import merge_inference_outputs
from vesuvius.models.run.inference import Inferer

VOLUME = (32, 32, 32)
PATCH = (16, 16, 16)


def shipped_targets():
    path = Path(vesuvius.models.__file__).parent / "configuration" / "aux_tasks" / "nearest_component_3d.yaml"
    losses = yaml.safe_load(path.read_text())["auxiliary_tasks"]["nearest_component"]["losses"]
    return {"surface": {"out_channels": 2},
            "nearest": {"out_channels": 4, "auxiliary_task": True, "task_type": "nearest_component",
                        "losses": losses}}


class SameSignXAxis(torch.nn.Module):
    def forward(self, x):
        s = x[:, :1].float()
        one = torch.ones_like(s)
        return {"surface": torch.cat([-s, s], dim=1),
                "nearest": torch.cat([one, 0 * one, 0 * one, 2 * one], dim=1)}


def fake_load(self, checkpoint_path):
    return {"network": SameSignXAxis(), "patch_size": PATCH, "num_input_channels": 1,
            "num_seg_heads": 6, "targets": shipped_targets(), "normalization_scheme": "none"}


def predict(workdir, name, staged, **kwargs):
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


def fmt(v):
    return "(" + ", ".join(f"{float(c):.4f}" for c in v) + ")"


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "model.pth").write_bytes(b"")
    z, y, x = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in VOLUME], indexing="ij")
    w = ZarrWriter.create(str(workdir / "volume.zarr"), shape=VOLUME, chunks=PATCH, dtype=np.float32)
    w[:] = 0.3 + 0.01 * x
    rows = []
    for staged in (False, True):
        for tta_name, kwargs in (("default mirroring", {}), ("--disable_tta", {"do_tta": False})):
            path = "staged (blend_logits)" if staged else "in-memory blend"
            name = f"{'staged' if staged else 'mem'}_{'tta' if not kwargs else 'notta'}"
            out = predict(workdir, name, staged, **kwargs)
            rows.append(f"Inferer {path:22s} {tta_name:18s} saved (x, y, z, distance) at (16, 16, 16): "
                        f"{fmt(out[2:6, 16, 16, 16])}")
    targets = shipped_targets()
    info = {"surface": {"start_channel": 0, "end_channel": 2, "out_channels": 2},
            "nearest": {"start_channel": 2, "end_channel": 6, "out_channels": 4}}
    kwargs = {"is_multi_task": True,
              "concat_multi_task_outputs": lambda o: torch.cat([o["surface"], o["nearest"]], dim=1)}
    if "channel_groups" in inspect.signature(tta.infer_with_tta_batched_3d).parameters:
        kwargs["channel_groups"] = tta.tta_channel_groups(targets, info, 3)
        kinds = [g["kind"] for g in kwargs["channel_groups"]]
    else:
        kinds = "n/a (no channel groups in this revision)"
    rows.append(f"channel groups for the shipped head: {kinds}")
    patch = torch.rand(1, 1, *PATCH)
    with torch.inference_mode():
        batched = tta.infer_with_tta_batched_3d(SameSignXAxis(), patch, **kwargs)
        sequential = tta.infer_with_tta_sequential(SameSignXAxis(), patch, "mirroring", **kwargs)
    for name, out in (("batched", batched), ("sequential", sequential)):
        rows.append(f"helper {name:10s} mirroring, mean (x, y, z, distance) over the patch: "
                    f"{fmt(out[0, 2:6].float().mean(dim=(1, 2, 3)))}")
    print()
    print("\n".join(rows))


if __name__ == "__main__":
    main()
