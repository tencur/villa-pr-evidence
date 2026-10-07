"""Labels converted with create_label_zarrs give no supervision in flat training on a 21- or 28-slice volume.

Usage:  python reproduce_label_depth.py <family_dir> <work_dir>

<family_dir> is a folder of the published ink_9um label tree with one segment made ready for
training as the docs describe, e.g. native9-scrollprizeorg-21slices holding
    w035/surface-volume.zarr          the public 9.362 um surface volume of PHerc. 0139 w035 (28 slices)
    w035/w035_inklabels.zarr          the published labels (28 planes)
    w035/w035_supervision_mask.zarr
The script works on a copy. It builds the trainer's dataset from the shipped recipe
(aligned21_hybrid_3d2d.json, reduced to this one segment) and reports what supervision the samples
carry: first with the published label stores, then after the documented labelling loop, in which
the label images are saved as TIFF and converted with create_label_zarrs.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

import vesuvius.ink_detection as ink_detection
from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config

family, work = Path(sys.argv[1]), Path(sys.argv[2])
if work.exists():
    shutil.rmtree(work)
shutil.copytree(family, work / family.name, copy_function=shutil.copy2)
family = work / family.name
segment = sorted(p.name for p in family.iterdir() if p.is_dir())[0]


def write_config(name):
    recipe = Path(ink_detection.__file__).parent / "configs" / "aligned21_hybrid_3d2d.json"
    config = json.loads(recipe.read_text())
    config["out_dir"] = str(work / name)
    config["dataloader_workers"] = 4
    # one segment instead of the released corpus, so the fixed per-scroll batch quotas do not apply
    config.pop("sampling_strategy", None)
    config.pop("fixed_scroll_prior", None)
    config["datasets"] = [{
        "segments_path": str(family),
        "segments": [segment],
        "surface_volume_paths": {segment: str(family / segment / "surface-volume.zarr")},
        "volume_scale": 0,
    }]
    path = work / f"{name}.json"
    path.write_text(json.dumps(config, indent=1))
    return path


def report(name):
    request = stage_training_request(str(write_config(name)))
    dataset = InkDataset(training_dataset_config(request.config), do_augmentations=False)
    total = len(dataset)
    picks = range(0, total, max(1, total // 300))
    with_supervision = supervised = ink = 0
    for index in picks:
        sample = dataset[index]
        mask = sample["supervision_mask"].amax(dim=1) > 0   # the loss takes the maximum over Z
        labels = sample["inklabels"].amax(dim=1) > 0
        with_supervision += bool(mask.any())
        supervised += int(mask.sum())
        ink += int((labels & mask).sum())
    store = zarr.open(str(family / segment / f"{segment}_inklabels.zarr"), mode="r")["0"]
    print(f"   label store shape {store.shape}; training patches found: {total}")
    print(f"   of {len(picks)} sampled patches, {with_supervision} carry any supervised pixel; "
          f"supervised pixels {supervised}, ink pixels {ink}")


print(f"segment {segment}")
print("1. published label stores")
report("out_published")

print("2. labelling loop: label images saved as TIFF, then create_label_zarrs")
for kind in ("inklabels", "supervision_mask"):
    store = zarr.open(str(family / segment / f"{segment}_{kind}.zarr"), mode="r")["0"]
    plane = np.asarray(store[store.shape[0] // 2])        # the annotated plane
    tifffile.imwrite(family / segment / f"{segment}_{kind}.tif", plane, tile=(256, 256), compression="zlib")
out = subprocess.run(
    [sys.executable, "-m", "vesuvius.ink_detection.preprocessing.create_label_zarrs", str(family), "--overwrite"],
    capture_output=True, text=True,
)
print("   " + " | ".join(l for l in (out.stdout + out.stderr).splitlines() if l.startswith(("Processed", "ERROR"))))
report("out_converted")
