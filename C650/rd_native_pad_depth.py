"""#1507 repro: native full-3D inference vs the training contract for model_config.input_pad_depth_to.

usage: python rd_native_pad_depth.py <villa tree> <real CT chunk .raw (128^3 uint8)>
Builds a native (full_3d) config with input_pad_depth_to, a checkpoint payload with the
tree's own make_model weights, and feeds one real CT patch through
  (a) the training path: train.prepare_model_input -> model -> train.prepare_loss_inputs
  (b) native inference:  infer_full3d_tifxyz._load_native_model -> predict_batch
and reports the depth the network receives and how far the probabilities differ.
"""
import contextlib, io, sys, types
from pathlib import Path
import numpy as np, torch

tree = Path(sys.argv[1]); raw = Path(sys.argv[2])
sys.path.insert(0, str(tree / "vesuvius" / "src"))
from vesuvius.ink_detection.config import InkConfig
from vesuvius.ink_detection.models.model import make_model
import vesuvius.ink_detection.inference.infer_full3d_tifxyz as native
import vesuvius.ink_detection.training.train as train_mod

seen = {}
def run(PATCH):
    torch.manual_seed(0)
    PAD_TO = 16                # model_config.input_pad_depth_to
    mapping = {
        "model_type": "vesuvius_unet",
        "mode": "full_3d",
        "patch_size": list(PATCH),
        "patch_overlap": 0.25,
        "patch_min_labeled_coverage": 0.0,
        "batch_size": 1,
        "in_channels": 1,
        "model_config": {
            "autoconfigure": True,
            "basic_encoder_block": "BasicBlockD",
            "basic_decoder_block": "ConvBlock",
            "features_per_stage": [8, 16, 32],
            "n_blocks_per_stage": [1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1],
            "kernel_sizes": [[3, 3, 3]] * 3,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "spacing": [1, 1, 1],
            "input_pad_depth_to": PAD_TO,
        },
        "targets": {"ink": {"out_channels": 1, "activation": "none", "z_projection_mode": "none"}},
        "datasets": [{"segments_path": "/tmp/ink", "volume_scale": 0}],
    }
    config = InkConfig.from_mapping(mapping)
    print(f"config: mode={mapping['mode']} model_type={mapping['model_type']} patch_size={list(PATCH)} "
          f"model_config.input_pad_depth_to={config.model.input_pad_depth_to}")

    with contextlib.redirect_stdout(io.StringIO()):  # the network builder's log
        base = make_model(config)
    payload = {"config": mapping, "model": base.state_dict()}

    # one real CT patch: level-0 chunk 27/16/16 of PHerc0500P2 volume 20250820143440 (z 3456.., y/x 2048..)
    vol = np.fromfile(raw, dtype=np.uint8).reshape(128, 128, 128)
    crop = vol[40:40 + PATCH[0], 32:32 + PATCH[1], 32:32 + PATCH[2]].astype(np.float32)
    crop = (crop - crop.mean()) / max(float(crop.std()), 1e-6)
    image = torch.from_numpy(crop)[None, None]            # B C Z Y X
    print(f"input: real CT crop {tuple(image.shape)} (uint8 mean {vol[40:52,32:96,32:96].mean():.1f})")

    seen.clear()
    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x):
            seen.setdefault("depths", []).append(int(x.shape[2]))
            return self.inner(x)

    # (a) training contract
    with contextlib.redirect_stdout(io.StringIO()):
        train_model = make_model(config)
    train_model.load_state_dict(base.state_dict()); train_model.eval()
    rec_t = Recorder(train_model)
    tcfg = types.SimpleNamespace(surface_mask_channel=False, ink=types.SimpleNamespace(model=config.model))
    with torch.no_grad():
        x_t = train_mod.prepare_model_input({"image": image}, tcfg)
        out_t = rec_t(x_t)["ink"]
        batch = {"image": image, "inklabels": torch.zeros_like(image), "supervision_mask": torch.ones_like(image)}
        logits_t, _, _ = train_mod.prepare_loss_inputs(out_t, batch, mode="full_3d")
    prob_t = logits_t.sigmoid()[0, 0].numpy()
    depth_t = seen["depths"][-1]

    # (b) native inference as infer_full3d_tifxyz builds it
    args = types.SimpleNamespace(checkpoint="mem.pth", gpu_ids=(), compile_model=False,
                                 compile_mode="reduce-overhead", amp_dtype="default")
    with contextlib.redirect_stdout(io.StringIO()):
        bundle = native._load_native_model(payload, config, args)
    inner = bundle.model
    inner.model = Recorder(inner.model)
    with torch.no_grad():
        prob_i = native.predict_batch(inner, image.to(bundle.device), variants=[()], tta_batch_size=None,
                                      patch_size_zyx=PATCH)[0, 0].cpu().numpy()
    depth_i = seen["depths"][-1]

    d = np.abs(prob_i - prob_t)
    flip = int(((prob_i > 0.5) != (prob_t > 0.5)).sum())
    print(f"training path : network input depth {depth_t}")
    print(f"native infer  : network input depth {depth_i}  (TargetModel.input_pad_depth_to={getattr(inner, 'input_pad_depth_to', None)})")
    print(f"probabilities vs training path over {d.size} voxels: max |diff| {d.max():.4f}, mean {d.mean():.4f}, "
          f"voxels on the other side of 0.5: {flip}")
    if depth_i != depth_t:
        print(f"RESULT: MISMATCH, network sees {depth_i} planes at inference, {depth_t} in training")
    elif d.max() > 1e-5:
        print("RESULT: MISMATCH, same network input but the output is resized differently from training")
    else:
        print("RESULT: OK, inference reproduces the training path (input depth and output resize)")


for z in (12, 13):
    print(f"\n=== patch depth {z}, model_config.input_pad_depth_to 16 ===")
    try:
        run((z, 64, 64))
    except Exception as exc:  # report the failure like a user would see it
        print(f"native inference FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
