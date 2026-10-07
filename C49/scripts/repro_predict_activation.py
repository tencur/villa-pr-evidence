"""vesuvius.predict on a vesuvius.train checkpoint whose target has activation sigmoid (the documented binary recipe):
the 'logits' store holds probabilities, and finalize re-applies sigmoid and thresholds in logit space.

Usage: python repro_predict_activation.py <checkpoint.pth> <input_zarr_array> <work_dir>
"""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np, zarr, torch


def main():
    ckpt, inp, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("checkpoint targets:", ck["model_config"].get("targets"))
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-m", "vesuvius.models.run.inference", "--model_path", ckpt, "--input_dir", inp,
                        "--output_dir", str(work / "pred"), "--device", "cpu", "--disable_tta", "--num_workers", "0",
                        "--overlap", "0.5", "--batch_size", "1"], capture_output=True, text=True, env=env)
    (work / "predict.log").write_text(r.stdout + r.stderr)
    parts = sorted(work.glob("pred/**/logits*"))
    print("predict exit", r.returncode, "parts:", [p.name for p in parts][:3])
    if not parts:
        print((r.stdout + r.stderr)[-600:]); return
    store = zarr.open(str(parts[0]), mode="r")
    arr = store["0"] if hasattr(store, "keys") and "0" in store else store
    a = np.asarray(arr[:]).astype(np.float32)
    nz = a[a != 0]
    print(f"stored 'logits': shape {a.shape}, non-zero values range {nz.min():.4f}..{nz.max():.4f} (probabilities would lie in 0..1)")
    for mode_args, label in ((["--threshold"], "binary --threshold (0.5)"), ([], "binary, no threshold (probability map)")):
        out = work / ("final_" + label.split()[0] + ("_thr" if mode_args else "_map"))
        r2 = subprocess.run([sys.executable, "-m", "vesuvius.models.run.finalize_outputs", str(parts[0]), str(out), "--mode", "binary",
                             *mode_args, "--num_workers", "1", "--quiet"], capture_output=True, text=True, env=env)
        try:
            g = zarr.open(str(out), mode="r"); f = g["0"] if hasattr(g, "keys") and "0" in g else g
            f = np.asarray(f[:])
            print(f"finalize {label}: exit {r2.returncode}; output values min {f.min()} max {f.max()}, share == 255: {np.mean(f == 255):.3f}, share > 0: {np.mean(f > 0):.3f}")
        except Exception as e:
            print(f"finalize {label}: exit {r2.returncode}; {e}; {(r2.stdout + r2.stderr)[-300:]}")


if __name__ == "__main__":
    main()
