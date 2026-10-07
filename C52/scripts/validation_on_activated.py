"""Validation losses on a real train.py checkpoint with activation sigmoid: the trainer puts the model in eval()
for validation, so the loss functions receive sigmoid(x) instead of x.

Usage: python validation_on_activated.py <checkpoint.pth> <image_zarr_level0> <label_zarr_level0>
"""
import sys
from pathlib import Path
import numpy as np, torch, zarr
from vesuvius.models.run.inference import Inferer

ckpt, img_path, lab_path = sys.argv[1:4]
lab = zarr.open(lab_path, mode="r"); img = zarr.open(img_path, mode="r")
S = 64
best = None
for z in range(0, lab.shape[0] - S, S):
    for y in range(0, lab.shape[1] - S, S):
        for x in range(0, lab.shape[2] - S, S):
            frac = float((np.asarray(lab[z:z+S, y:y+S, x:x+S]) > 0).mean())
            if best is None or abs(frac - 0.3) < abs(best[0] - 0.3):
                best = (frac, z, y, x)
frac, z, y, x = best
label = torch.from_numpy((np.asarray(lab[z:z+S, y:y+S, x:x+S]) > 0).astype(np.float32))[None, None]
patch = np.asarray(img[z:z+S, y:y+S, x:x+S], dtype=np.float32); patch = (patch - patch.mean()) / (patch.std() + 1e-8)
xin = torch.from_numpy(patch)[None, None]
inf = Inferer.__new__(Inferer); inf.device = torch.device("cpu"); inf.verbose = False
inf.model_normalization_scheme = None; inf.model_intensity_properties = None
info = inf._load_train_py_model(Path(ckpt))
model = next(v for v in info.values() if isinstance(v, torch.nn.Module))
def out(m, train):
    m.train(train)
    with torch.inference_mode():
        o = m(xin); o = o["surface"] if isinstance(o, dict) else o
        return (o[0] if isinstance(o, (list, tuple)) else o).float()
e, t = out(model, False), out(model, True)
bce = torch.nn.BCEWithLogitsLoss()
def dice(p):
    inter = (p * label).sum(); return float(2 * inter / (p.sum() + label.sum() + 1e-8))
print(f"real patch z{z} y{y} x{x} ({S}^3), label foreground {frac:.1%}; eval() output range {e.min():.3f}..{e.max():.3f}, train() {t.min():.2f}..{t.max():.2f}")
print(f"BCEWithLogitsLoss as the validation step computes it (eval outputs): {bce(e, label):.4f}")
print(f"BCEWithLogitsLoss on the logits (what training computes):            {bce(t, label):.4f}")
print(f"soft Dice of sigmoid(eval output) [doubly activated]: {dice(torch.sigmoid(e)):.4f}; of sigmoid(logits): {dice(torch.sigmoid(t)):.4f}; of the probabilities themselves: {dice(e):.4f}")
print(f"foreground share at threshold 0.5 after a second sigmoid: {float((torch.sigmoid(e) > 0.5).float().mean()):.3f}; from the probabilities: {float((e > 0.5).float().mean()):.3f}; label: {frac:.3f}")
