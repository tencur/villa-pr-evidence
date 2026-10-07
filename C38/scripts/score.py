"""Score a B01 checkpoint against the binary label on every cached labelled patch.
Usage: python score.py <dataset_dir> <work_dir> <arm>"""
import json, sys
from pathlib import Path


def main():
    import numpy as np, torch, zarr
    from vesuvius.models.run.inference import Inferer
    ds, work, arm = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    ckpt = next(work.joinpath("ckpt").rglob("*_final.pth"))
    inf = Inferer.__new__(Inferer); inf.device = torch.device("cpu"); inf.verbose = False
    info = inf._load_train_py_model(ckpt)
    model = next(v for v in info.values() if isinstance(v, torch.nn.Module))
    model.eval()
    cache = max((work / "data" / ".patches_cache").glob("patches_v*.json"), key=lambda f: f.stat().st_mtime)
    pos = [tuple(p["pos"]) for p in json.loads(cache.read_text())["fg_patches"]]
    img = zarr.open(str(ds / "images" / "PHerc0500P2.zarr" / "0"), mode="r")
    lab = zarr.open(str(ds / "labels" / "PHerc0500P2_surface.zarr" / "0"), mode="r")
    pf = tf = inter = 0
    logits = []
    with torch.inference_mode():
        for z, y, x in pos:
            a = np.asarray(img[z:z + 48, y:y + 48, x:x + 48], dtype=np.float32)
            a = (a - a.mean()) / max(a.std(), 1e-8)
            out = model(torch.from_numpy(a)[None, None])
            lg = out["surface"] if isinstance(out, dict) else out
            lg = lg[0] if isinstance(lg, (list, tuple)) else lg
            lg = lg[0, 0].numpy()
            logits.append(np.median(lg))
            p = lg > 0
            t = np.asarray(lab[z:z + 48, y:y + 48, x:x + 48]) > 0
            pf += p.sum(); tf += t.sum(); inter += (p & t).sum()
    n = len(pos) * 48 ** 3
    log = (work / "train.log").read_text()
    avg = [l.strip() for l in log.splitlines() if "Avg Loss" in l]
    print(f"[{arm}] train {avg[-1] if avg else '?'}; {len(pos)} labelled patches: predicted foreground "
          f"{100 * pf / n:.1f}% of voxels (true {100 * tf / n:.1f}%), Dice {2 * inter / max(pf + tf, 1):.3f}, "
          f"median logit {np.median(logits):.2f}")


if __name__ == "__main__":
    main()
