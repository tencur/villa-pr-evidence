"""Doc recipe for binary segmentation (data_formatting.md: out_channels 1, sigmoid, BCEWithLogitsLoss) trained with
the label stored as a raw grayscale mask (0/255, as the published m7 surface prediction is) vs the same label as 0/1.

Usage: python run_binary_labels.py <dataset_dir> <work_dir> <arm: raw255|binary01> [steps]

Real vesuvius.train (CLI main) for a fixed number of steps on CPU, same seed and patches for both arms; then the
final checkpoint predicts held-out labelled patches and is scored against the binary label.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path


def main():
    import numpy as np, zarr, torch
    ds, work, arm = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    if work.exists():
        shutil.rmtree(work)
    (work / "data" / "images").mkdir(parents=True)
    (work / "data" / "images" / "vol.zarr").symlink_to(ds / "images" / "PHerc0500P2.zarr")
    lab_src = ds / "labels" / "PHerc0500P2_surface.zarr"
    lab_dst = work / "data" / "labels" / "vol_surface.zarr"
    shutil.copytree(lab_src, lab_dst)
    if arm == "binary01":
        for lvl in ("0", "1"):
            a = zarr.open(str(lab_dst / lvl), mode="r+")
            a[:] = (a[:] > 0).astype(np.uint8)
    vals = np.unique(zarr.open(str(lab_dst / "0"), mode="r")[:])
    print(f"[{arm}] label values on disk: {vals.tolist()}", flush=True)
    cfg = work / "train.yaml"
    cfg.write_text(f"""tr_setup:
  model_name: b01_{arm}
  tr_val_split: 0.8
tr_config:
  patch_size: [48, 48, 48]
  batch_size: 2
  num_dataloader_workers: 0
  max_epoch: 1
  max_steps_per_epoch: {steps}
  max_val_steps_per_epoch: 1
  initial_lr: 0.01
model_config:
  features_per_stage: [8, 16, 32, 64]
dataset_config:
  data_path: {work / 'data'}
  normalization_scheme: zscore
  min_labeled_ratio: 0.02
  min_bbox_percent: 0.3
  valid_patch_find_resolution: 1
  targets:
    surface:
      out_channels: 1
      activation: sigmoid
      losses:
        - name: BCEWithLogitsLoss
          weight: 1.0
""")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    py = sys.executable
    subprocess.run([py, "-m", "vesuvius.models.preprocessing.patches.cli", "--config", str(cfg)], check=True,
                   capture_output=True, env=env)
    out = subprocess.run([py, "-m", "vesuvius.models.training.train", "-i", str(work / "data"), "--config", str(cfg),
                          "-o", str(work / "ckpt"), "--seed", "0", "--no-amp"],
                         capture_output=True, text=True, env=env, cwd=work)
    log = (out.stdout + out.stderr)
    (work / "train.log").write_text(log)
    losses = [l for l in log.splitlines() if "loss" in l.lower() and ("train" in l.lower() or "val" in l.lower())]
    print(f"[{arm}] train exit {out.returncode}; last loss lines:", *losses[-3:], sep="\n  ", flush=True)
    ckpts = sorted((work / "ckpt").rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    print(f"[{arm}] checkpoints: {[c.name for c in ckpts][-3:]}", flush=True)
    if os.environ.get("NO_SCORE"):
        return  # training result only (the scoring stage below is not part of the proof)
    # score the final checkpoint against the BINARY label on every cached labelled patch
    from vesuvius.models.run.inference import Inferer
    inf = Inferer.__new__(Inferer); inf.device = torch.device("cpu"); inf.verbose = False
    res = inf._load_train_py_model(ckpts[-1])
    model = res[0] if isinstance(res, tuple) else (res if res is not None else inf.model)
    model.eval()
    cache = max((work / "data" / ".patches_cache").glob("patches_v*.json"), key=lambda f: f.stat().st_mtime)
    pos = [tuple(p["pos"]) for p in json.loads(cache.read_text())["fg_patches"]]
    img = zarr.open(str(ds / "images" / "PHerc0500P2.zarr" / "0"), mode="r")
    lab = zarr.open(str(lab_src / "0"), mode="r")
    pred_fg, true_fg, inter, union_sum = 0, 0, 0, 0
    with torch.inference_mode():
        for z, y, x in pos:
            a = np.asarray(img[z:z + 48, y:y + 48, x:x + 48], dtype=np.float32)
            a = (a - a.mean()) / max(a.std(), 1e-8)
            out = model(torch.from_numpy(a)[None, None])
            logit = out["surface"] if isinstance(out, dict) else out
            if isinstance(logit, (list, tuple)):
                logit = logit[0]
            p = (torch.sigmoid(logit)[0, 0].numpy() > 0.5)
            t = np.asarray(lab[z:z + 48, y:y + 48, x:x + 48]) > 0
            pred_fg += p.sum(); true_fg += t.sum(); inter += (p & t).sum(); union_sum += p.sum() + t.sum()
    n = len(pos) * 48 ** 3
    print(f"[{arm}] {len(pos)} labelled patches: predicted foreground {100 * pred_fg / n:.1f}% of voxels, "
          f"true {100 * true_fg / n:.1f}%; Dice vs binary label {2 * inter / max(union_sum, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
