"""For each loss name, as the trainer builds it for the default target (ignore_label 2): does it run on a
two-channel prediction with a one-channel target, prefer logits that match the target over swapped ones,
and ignore the prediction where the label is 2? usage: loss_choices.py NAME [NAME ...]"""
import sys
from types import SimpleNamespace
import torch
from vesuvius.models.training.train import BaseTrainer

torch.manual_seed(0)
target = (torch.rand(2, 1, 16, 16) > 0.5).float()
target[:, :, :4] = 2
fg = (target == 1).float()
correct = torch.cat([10 - 20 * fg, 20 * fg - 10], dim=1)
changed = correct.clone()
changed[:, :, :4] = torch.randn_like(changed[:, :, :4]) * 10
for name in sys.argv[1:]:
    task = {"ignore_label": 2, "losses": [{"name": name, "weight": 1.0, "kwargs": {}}]}
    try:
        fn = BaseTrainer._build_loss(SimpleNamespace(mgr=SimpleNamespace(targets={"ink": task}, enable_deep_supervision=False)))["ink"][0][0]
        a, b, c = (float(fn(x, target)) for x in (correct, correct.flip(1), changed))
        print(f"{name}: runs; loss(matching)={a:.4f} loss(swapped)={b:.4f} -> prefers matching: {a < b}; "
              f"loss with ignored region changed={c:.4f} -> ignores label 2: {abs(a - c) < 1e-6}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {str(e)[:150]}")
