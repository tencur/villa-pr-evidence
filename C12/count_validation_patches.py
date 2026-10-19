"""Count, in the sample set reproduce_label_depth.py reports on, the validation patches and how many carry supervision.

Usage: python count_validation_patches.py <work_dir>/out_converted.json   (the config reproduce_label_depth.py wrote)
"""
import sys, json
from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.training.train import stage_training_request, training_dataset_config
req = stage_training_request(sys.argv[1])
ds = InkDataset(training_dataset_config(req.config), do_augmentations=False)
picks = range(0, len(ds), max(1, len(ds)//300))
v = [i for i in picks if ds.patches[i].is_validation]
print("patches", len(ds), "sampled", len(picks), "validation among sampled", len(v))
ws = 0
for i in v:
    s = ds[i]; ws += bool((s["supervision_mask"].amax(dim=1) > 0).any())
print("validation sampled patches with supervision", ws)
