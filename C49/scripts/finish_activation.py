"""Second half of the predict-activation check: documented blend -> finalize on the P02 predict output, plus a
direct train-mode vs eval-mode comparison of the checkpoint's network on one real patch."""
import os, subprocess, sys
from pathlib import Path
import numpy as np, torch, zarr

ckpt, crop, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
# 1. documented: vesuvius.blend_logits <parent_dir> <output>
r = subprocess.run([sys.executable, "-m", "vesuvius.models.run.blending", str(work / "pred"), str(work / "blended.zarr"),
                    "--num_workers", "1", "--quiet"], capture_output=True, text=True, env=env)
print("blend exit", r.returncode, (r.stdout + r.stderr)[-200:].replace("\n", " ") if r.returncode else "")
b = zarr.open(str(work / "blended.zarr"), mode="r")
arr = b["0"] if hasattr(b, "keys") and "0" in b else b
a = np.asarray(arr[:]).astype(np.float32); nz = a[a != 0]
print(f"blended store: shape {a.shape}, non-zero range {nz.min():.4f}..{nz.max():.4f}, share of voxels < 0: {np.mean(a < 0):.3f}")
# 2. documented: vesuvius.finalize_outputs <blended> <out> --mode binary [--threshold]
for extra, label in ((["--threshold"], "binary --threshold 0.5"), ([], "binary probability map")):
    out = work / ("final_thr.zarr" if extra else "final_map.zarr")
    r2 = subprocess.run([sys.executable, "-m", "vesuvius.models.run.finalize_outputs", str(work / "blended.zarr"), str(out),
                         "--mode", "binary", *extra, "--num_workers", "1", "--quiet"], capture_output=True, text=True, env=env)
    g = zarr.open(str(out), mode="r"); f = np.asarray((g["0"] if hasattr(g, "keys") and "0" in g else g)[:])
    print(f"finalize {label}: exit {r2.returncode}; values min {f.min()} max {f.max()}; share == 255: {np.mean(f == 255):.3f}; share > 127: {np.mean(f > 127):.3f}")
# 3. the network itself: eval() applies the target's activation
from vesuvius.models.run.inference import Inferer
inf = Inferer.__new__(Inferer); inf.device = torch.device("cpu"); inf.verbose = False
inf.model_normalization_scheme = None; inf.model_intensity_properties = None
info = inf._load_train_py_model(Path(ckpt))
model = next(v for v in info.values() if isinstance(v, torch.nn.Module))
x = np.asarray(zarr.open(crop, mode="r")[0:48, 0:48, 0:48], dtype=np.float32); x = (x - x.mean()) / x.std()
t = torch.from_numpy(x)[None, None]
with torch.inference_mode():
    model.eval(); e = model(t); e = (e["surface"] if isinstance(e, dict) else e)
    e = e[0] if isinstance(e, (list, tuple)) else e
    model.train(); tr = model(t); tr = (tr["surface"] if isinstance(tr, dict) else tr)
    tr = tr[0] if isinstance(tr, (list, tuple)) else tr
print(f"network output on one real patch: eval() range {e.min():.4f}..{e.max():.4f}; train() range {tr.min():.4f}..{tr.max():.4f}; "
      f"eval == sigmoid(train): {bool(torch.allclose(e.float(), torch.sigmoid(tr.float()), atol=1e-4))}")
