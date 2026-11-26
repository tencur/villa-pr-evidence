"""Drive `vesuvius.napari_trainer` the way the README describes, with napari offscreen.

usage (from the work dir): QT_QPA_PLATFORM=offscreen python napari_session.py train [config.yaml]
                           QT_QPA_PLATFORM=offscreen python napari_session.py train2d [config.yaml]
                           QT_QPA_PLATFORM=offscreen python napari_session.py infer <checkpoint.pth> [config.yaml]
EPOCHS (default 1) sets the training widget's max epochs. train2d uses the middle slice of the
same synthetic volume as 2D layers and a 64x64 patch (Patch Size Z 0). SEED (infer, optional)
seeds torch before 'Run Inference', so weights the checkpoint does not supply are reproducible.

Runs the console script's function, vesuvius.napari_trainer.main_window:main().
Instead of handing control to a person (napari.run), it does the README steps:
  train: (optionally choose a config in the 'config file' picker),
         2. add an image layer 'vol01', 3. add a label layer 'vol01_ink',
         5. set a 32^3 patch and EPOCHS epochs in the training widget and press 'run training'.
  infer: a fresh session (the default config is loaded; optionally choose another config
         in the picker), then 6. add the image layer, select the checkpoint in the
         inference widget and press 'Run Inference'.
Besides answering napari's OpenGL capability queries (no GL on a headless machine),
the only additions are read-only recorders: which volumes the trainer's dataset
opened (and whether they hold the viewer's pixels) and which network inference built.
"""
import json
import os
import traceback
import sys
from pathlib import Path

import numpy as np
import napari

mode = sys.argv[1]
extra = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None  # config (train) or checkpoint (infer)
infer_config = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
# run from the work dir: the configs write checkpoints under ./checkpoints

rng = np.random.default_rng(0)
zz, yy, xx = np.mgrid[:64, :128, :128]
LABEL = (((xx + yy + zz // 4) // 16) % 3 == 0).astype(np.uint8)  # 3D sheets of "ink"
IMAGE = (rng.normal(100, 10, LABEL.shape) + 60 * LABEL).astype(np.float32)
if mode == "train2d":  # the middle slice as 2D layers
    LABEL, IMAGE = LABEL[32], IMAGE[32]

result = {}


def record_datasets():
    from vesuvius.models.datasets import zarr_dataset

    real_init = zarr_dataset.ZarrDataset.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        vols = []
        for v in self._volumes:
            lbl = v.label_arrays.get("ink")
            vols.append({
                "volume": v.volume_id,
                "image_path": str(v.image_path),
                "image_equals_viewer_layer": bool(np.array_equal(np.asarray(v.image_array[...]), IMAGE)),
                "ink_label_equals_viewer_layer": None if lbl is None else bool(np.array_equal(np.asarray(lbl[...]), LABEL)),
            })
        result.setdefault("datasets", []).append({"is_training": self.is_training, "volumes": vols,
                                                  "patches": len(self)})

    zarr_dataset.ZarrDataset.__init__ = init


def session_train():
    from vesuvius.napari_trainer import main_window as mw

    if extra is not None:  # the 'config file' picker (auto_call loads the chosen file)
        mw.filespicker.filenames.value = [extra]
    viewer = napari.current_viewer()
    viewer.add_image(IMAGE, name="vol01")
    viewer.add_labels(LABEL, name="vol01_ink")
    # README step 5: a patch that fits the layers (2D: Patch Size Z 0)
    if mode == "train2d":
        mw.run_training.patch_size_z.value = 0
        mw.run_training.patch_size_y.value = 64
        mw.run_training.patch_size_x.value = 64
    else:
        mw.run_training.patch_size_z.value = 32
        mw.run_training.patch_size_y.value = 32
        mw.run_training.patch_size_x.value = 32
    mw.run_training.max_epochs.value = int(os.environ.get("EPOCHS", "1"))
    result["widget"] = {w.name: w.value for w in mw.run_training if hasattr(w, "value")}
    result["config"] = str(mw._config_manager._config_path)
    try:
        mw.run_training()  # what the 'run training' button calls
        result["run_training"] = "returned"
    except Exception as e:  # run_training re-raises after printing the traceback
        result["run_training"] = f"{type(e).__name__}: {e}"
    ckpts = sorted(str(p) for p in Path("checkpoints").rglob("*.pth"))
    result["checkpoints"] = ckpts


def session_infer():
    import torch
    from vesuvius.napari_trainer import inference
    from vesuvius.napari_trainer.inference_widget import inference_widget

    ckpt = str(extra)
    built = {}
    real_load = inference.ModelLoader.load

    def load(self):
        model, cfg = real_load(self)
        built["model"] = model
        return model, cfg

    inference.ModelLoader.load = load
    if infer_config is not None:
        from vesuvius.napari_trainer import main_window as mw

        mw.filespicker.filenames.value = [infer_config]
        result["config"] = str(mw._config_manager._config_path)
    viewer = napari.current_viewer()
    layer = viewer.add_image(IMAGE, name="vol01")
    inference_widget.model_path.value = ckpt
    inference_widget.layer.value = layer
    if os.environ.get("SEED"):
        torch.manual_seed(int(os.environ["SEED"]))
        result["seed"] = int(os.environ["SEED"])
    inference_widget()  # what the 'Run Inference' button calls
    if "model" not in built:  # inference stopped before a network was returned
        result["output_layers"] = {l.name: list(l.data.shape) for l in viewer.layers if l.name.startswith("vol01_")}
        return
    weights = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
    state = built["model"].state_dict()
    differ = [k for k, v in weights.items() if k not in state or state[k].shape != v.shape or not torch.equal(state[k].cpu(), v)]
    result["checkpoint_tensors"] = len(weights)
    result["tensors_not_loaded_from_checkpoint"] = len(differ)
    new = [l for l in viewer.layers if l.name.startswith("vol01_")]
    result["output_layers"] = {l.name: list(l.data.shape) for l in new}
    if new:
        prob = torch.softmax(torch.as_tensor(new[0].data[0]), dim=0)[1].numpy()
        pred = prob > 0.5
        inter = np.logical_and(pred, LABEL == 1).sum()
        union = np.logical_or(pred, LABEL == 1).sum()
        result["ink_iou_vs_label"] = round(float(inter / union), 4)


def headless_canvas():
    """This machine has no OpenGL; answer the canvas's two GL capability queries with
    constants so layers can be added. Nothing else in napari is changed."""
    from napari._vispy.layers import base, image, labels

    image.get_gl_extensions = lambda: "texture_float"
    for mod in (base, image, labels):
        mod.get_max_texture_sizes = lambda: (16384, 2048)


if __name__ == "__main__":  # DataLoader workers re-import this file
    headless_canvas()
    record_datasets()
    napari.run = session_infer if mode == "infer" else session_train
    from vesuvius.napari_trainer.main_window import main

    try:
        main()
        result["main"] = "returned"
    except Exception as e:
        traceback.print_exc()
        result["main"] = f"{type(e).__name__}: {e}"
    print("RESULT " + json.dumps(result, default=str))
