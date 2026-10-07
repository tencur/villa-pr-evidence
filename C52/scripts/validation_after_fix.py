"""Same measurement as validation_on_activated.py, but through the trainer's own model build (BaseTrainer._build_model)
with the checkpoint's weights loaded, i.e. what validation sees during training."""
import sys
from types import SimpleNamespace
import numpy as np, torch, zarr
from vesuvius.models.training.train import BaseTrainer

ckpt, img_path, lab_path = sys.argv[1:4]
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
mc = dict(ck["model_config"])
mgr = SimpleNamespace(model_config=mc, targets=mc["targets"], train_patch_size=mc.get("train_patch_size", mc.get("patch_size")),
                      train_batch_size=mc.get("train_batch_size", 1), in_channels=mc.get("in_channels", 1), autoconfigure=mc.get("autoconfigure", False),
                      enable_deep_supervision=bool(mc.get("enable_deep_supervision", False)), model_name=mc.get("model_name", "Model"), spacing=[1, 1, 1])
model = BaseTrainer(mgr=mgr, verbose=False)._build_model()
sd = ck.get("model") or ck.get("state_dict") or ck["model_state_dict"]
sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
lab = zarr.open(lab_path, mode="r"); img = zarr.open(img_path, mode="r")
z, y, x, S = 768, 0, 64, 64
label = torch.from_numpy((np.asarray(lab[z:z+S, y:y+S, x:x+S]) > 0).astype(np.float32))[None, None]
patch = np.asarray(img[z:z+S, y:y+S, x:x+S], dtype=np.float32); patch = (patch - patch.mean()) / (patch.std() + 1e-8)
xin = torch.from_numpy(patch)[None, None]
def out(train):
    model.train(train)
    with torch.inference_mode():
        o = model(xin); o = o["surface"] if isinstance(o, dict) else o
        return (o[0] if isinstance(o, (list, tuple)) else o).float()
e, t = out(False), out(True)
bce = torch.nn.BCEWithLogitsLoss()
print(f"same real patch; eval() output range {e.min():.2f}..{e.max():.2f}")
print(f"BCEWithLogitsLoss as validation computes it (eval outputs): {bce(e, label):.4f}; on train-mode logits: {bce(t, label):.4f}")
print(f"foreground share at threshold 0.5 after sigmoid of the eval output: {float((torch.sigmoid(e) > 0.5).float().mean()):.3f} (label 0.301)")
