"""bg_sampling_enabled and ignore_label are accepted but never reach the patch finder.

Usage:  python reproduce_bg_sampling.py <pristine_dataset_dir> <work_dir>

<pristine_dataset_dir> holds real data in the trainer's layout:
    images/PHerc0500P2.zarr            CT of PHerc0500P2 (9.362 um scan), pyramid levels 0 and 1
    labels/PHerc0500P2_surface.zarr    the published m7 surface prediction (0/255), same levels
The copy is recoded as a sparsely annotated volume, the way the shipped ps128 configs expect:
inside one annotated box  1 = surface, 0 = checked background; everywhere else 2 = not annotated.
Two configs are run, mirroring the shipped ones (both set ignore_label 2, bg_sampling_enabled true,
bg_to_fg_ratio 0.10):
    A  like ps128_dicece.yaml            (valid_patch_value: 1)
    B  like ps128_guided_*_ink.yaml      (no valid_patch_value)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

pristine, work = Path(sys.argv[1]), Path(sys.argv[2])
if work.exists():
    shutil.rmtree(work)
shutil.copytree(pristine, work, copy_function=shutil.copy2)

P = 128
# annotated box at level 0 (z, y, x), not aligned to the 128 patch grid, so patches on its
# edge hold both annotated and unannotated voxels, as with real sparse annotation
BOX = (slice(200, 600), slice(0, 500), slice(0, 300))
lab_root = work / "labels" / "PHerc0500P2_surface.zarr"
for level in ("0", "1"):
    s = 2 ** int(level)
    arr = zarr.open(str(lab_root / level), mode="r+")
    data = arr[:]
    out = np.full(data.shape, 2, dtype=np.uint8)
    box = tuple(slice(b.start // s, b.stop // s) for b in BOX)
    out[box] = (data[box] > 0).astype(np.uint8)
    arr[:] = out
lab0 = zarr.open(str(lab_root / "0"), mode="r")[:]
print("label values (level 0):", dict(zip(*[x.tolist() for x in np.unique(lab0, return_counts=True)])))

# Ground truth from the label itself, on the same 128-grid the finder uses
grid = [range(0, n - P + 1, P) for n in lab0.shape]
truth_fg, truth_bg, ignore_only = set(), set(), set()
for z in grid[0]:
    for y in grid[1]:
        for x in grid[2]:
            blk = lab0[z:z + P, y:y + P, x:x + P]
            has_fg, has_bg, has_ig = (blk == 1).any(), (blk == 0).any(), (blk == 2).any()
            if has_fg:
                truth_fg.add((z, y, x))
            elif has_bg and has_ig:
                truth_bg.add((z, y, x))      # the finder's BG-only rule: 0 and ignore, no foreground
            elif not has_bg:
                ignore_only.add((z, y, x))
print(f"grid patches: {len(truth_fg)} with surface, {len(truth_bg)} background-only "
      f"(annotated background next to unannotated), {len(ignore_only)} entirely not annotated")

CONFIGS = {
    "A (valid_patch_value 1, like ps128_dicece)": "  valid_patch_value: 1\n",
    "B (no valid_patch_value, like ps128_guided_*_ink)": "",
}
for name, extra in CONFIGS.items():
    for cache in work.glob(".patches_cache"):
        shutil.rmtree(cache)
    cfg = work / "train.yaml"
    cfg.write_text(
        "tr_config:\n  patch_size: [128, 128, 128]\n"
        "dataset_config:\n" + extra +
        "  min_labeled_ratio: 0.01\n  min_bbox_percent: 0.15\n"
        "  valid_patch_find_resolution: 1\n  normalization_scheme: zscore\n"
        "  bg_sampling_enabled: true\n  bg_to_fg_ratio: 0.10\n"
        "  targets:\n    surface:\n      activation: none\n      ignore_label: 2\n"
    )
    run = subprocess.run([sys.executable, "-m", "vesuvius.models.preprocessing.patches.cli",
                          "--config", str(cfg)], capture_output=True, text=True)
    summary = [l.split("INFO")[-1].strip(" :-") for l in (run.stdout + run.stderr).splitlines()
               if "complete" in l.lower() or "BG" in l]
    cache_file = max((work / ".patches_cache").glob("patches_v*.json"), key=lambda f: f.stat().st_mtime_ns)
    cache = json.loads(cache_file.read_text())
    fg = {tuple(p["pos"]) for p in cache["fg_patches"]}
    bg = {tuple(p["pos"]) for p in cache["bg_patches"]}
    code = (
        "import sys, logging; logging.disable(logging.CRITICAL)\n"
        "from vesuvius.models.configuration.config_manager import ConfigManager\n"
        "from vesuvius.models.datasets.zarr_dataset import ZarrDataset\n"
        "m = ConfigManager(verbose=False); m.load_config(sys.argv[1])\n"
        "d = ZarrDataset(m)\n"
        "w = getattr(d, 'patch_weights', None)\n"
        "print('TRAINER', len(d), d.n_fg, len(d) - d.n_fg, 'weights' if isinstance(w, list) and w else 'no-weights')\n"
        "import torch\n"
        "from torch.utils.data import WeightedRandomSampler\n"
        "n_fg, n_bg = d.n_fg, len(d) - d.n_fg\n"
        "if isinstance(w, list) and w:\n"
        "    k = min(int(n_fg * m.bg_to_fg_ratio / (1 - m.bg_to_fg_ratio)), n_bg)\n"
        "    drawn = list(WeightedRandomSampler(torch.tensor(w, dtype=torch.double), n_fg + k, replacement=False, generator=torch.Generator().manual_seed(0)))\n"
        "    print('EPOCH', sum(i < n_fg for i in drawn), sum(i >= n_fg for i in drawn))\n"
        "else:\n"
        "    print('EPOCH', n_fg, 0)\n"
    )
    tr = subprocess.run([sys.executable, "-c", code, str(cfg)], capture_output=True, text=True)
    trainer = [l for l in tr.stdout.splitlines() if l.startswith("TRAINER")]
    epoch = [l for l in tr.stdout.splitlines() if l.startswith("EPOCH")]
    print(f"\n== config {name}")
    print("   finder log           :", " | ".join(summary) or (run.stdout + run.stderr)[-600:])
    print(f"   cache fg_patches     : {len(fg)}  (of which entirely not annotated: {len(fg & ignore_only)},"
          f" annotated background only: {len(fg & truth_bg)}, with surface: {len(fg & truth_fg)})")
    print(f"   cache bg_patches     : {len(bg)}  (annotated background-only patches available: {len(truth_bg)})")
    if trainer:
        _, total, n_fg, n_rest, w = trainer[0].split()
        print(f"   trainer dataset      : {total} patches, {n_fg} FG, {n_rest} BG, sampler {w}")
        if epoch:
            _, e_fg, e_bg = epoch[0].split()
            print(f"   one training epoch   : {e_fg} FG + {e_bg} BG-only patches"
                  f" (all FG + bg_to_fg_ratio 0.10 of BG, split as train.py does without a validation hold-out)")
    else:
        print("   trainer dataset      :", (tr.stdout + tr.stderr)[-800:])
