"""Review case: sign alignment at half precision (CUDA inference runs under autocast).

An `inplane_direction` head reports the x axis at length 1e-4 in float16 under every mirror.
After a mirror over x the restored component is -1e-4; the alignment has to see that
(-1e-4) * (1e-4) < 0, but in float16 that product rounds to zero. Calls the two mirroring
helpers directly on CPU with one 4^3 patch.

Run:  python review_fp16_small_direction.py
"""
import inspect

import torch

from vesuvius.models.run import tta

TARGETS = {"surface": {"out_channels": 2},
           "direction": {"out_channels": 3, "auxiliary_task": True, "task_type": "inplane_direction"}}
INFO = {"surface": {"start_channel": 0, "end_channel": 2, "out_channels": 2},
        "direction": {"start_channel": 2, "end_channel": 5, "out_channels": 3}}


def model(x):
    direction = torch.zeros(x.shape[0], 3, *x.shape[2:], dtype=torch.float16)
    direction[:, 0] = 1e-4
    return {"surface": x.repeat(1, 2, 1, 1, 1).half(), "direction": direction}


def concat(out):
    return torch.cat([out["surface"], out["direction"]], dim=1)


def main():
    kwargs = {"is_multi_task": True, "concat_multi_task_outputs": concat}
    if "channel_groups" in inspect.signature(tta.infer_with_tta_batched_3d).parameters:
        kwargs["channel_groups"] = tta.tta_channel_groups(TARGETS, INFO, 3)
    else:
        print("(this revision has no channel groups: components are averaged as they come)")
    x = torch.rand(1, 1, 4, 4, 4)
    with torch.inference_mode():
        batched = tta.infer_with_tta_batched_3d(model, x, **kwargs)
        sequential = tta.infer_with_tta_sequential(model, x, "mirroring", **kwargs)
    print(f"product in float16: {(torch.tensor(-1e-4, dtype=torch.float16) * torch.tensor(1e-4, dtype=torch.float16)).item()}")
    for name, out in (("batched", batched), ("sequential", sequential)):
        print(f"{name:10s} output dtype {out.dtype}, mean direction x {out[:, 2].float().mean().item():.6f} "
              f"y {out[:, 3].float().mean().item():.6f} z {out[:, 4].float().mean().item():.6f}")


if __name__ == "__main__":
    main()
