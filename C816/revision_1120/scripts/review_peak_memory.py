"""Review case: peak process memory of the mirroring TTA helpers on CPU (Linux).

One 128^3 patch, a model with 5 float16 output channels (2 surface scores + a 3-channel head),
in the batched helper (the default on CUDA) and the sequential one (the fallback). Each
configuration runs in a fresh process; the peak is VmHWM after the call (reset through
/proc/self/clear_refs just before it) minus VmRSS before it. "plain" passes no channel groups
(the average main computes); the head rows pass the groups this revision derives for the head.

Run:  python review_peak_memory.py
"""
import inspect
import subprocess
import sys

N = 128


def status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def measure(helper, head):
    import torch
    from vesuvius.models.run import tta
    torch.set_num_threads(1)
    targets = {"surface": {"out_channels": 2},
               "h": {"out_channels": 3, "auxiliary_task": True, "task_type": head}}
    info = {"surface": {"start_channel": 0, "end_channel": 2, "out_channels": 2},
            "h": {"start_channel": 2, "end_channel": 5, "out_channels": 3}}

    def model(x):
        y = x.half()
        return {"surface": torch.cat([y, -y], dim=1), "h": torch.cat([y, -y, y], dim=1)}

    def concat(out):
        return torch.cat([out["surface"], out["h"]], dim=1)

    kwargs = {"is_multi_task": True, "concat_multi_task_outputs": concat}
    has_groups = "channel_groups" in inspect.signature(tta.infer_with_tta_batched_3d).parameters
    if head != "plain":
        if not has_groups:
            return "n/a (no channel groups in this revision)"
        kwargs["channel_groups"] = tta.tta_channel_groups(targets, info, 3)
    x = torch.rand(1, 1, N, N, N)
    with torch.inference_mode():  # warm-up on a small patch (allocator, thread pools)
        tta.infer_with_tta_sequential(model, torch.rand(1, 1, 8, 8, 8), "mirroring", **kwargs)
    before = status("VmRSS")
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    with torch.inference_mode():
        if helper == "batched":
            out = tta.infer_with_tta_batched_3d(model, x, **kwargs)
        else:
            out = tta.infer_with_tta_sequential(model, x, "mirroring", **kwargs)
    peak = status("VmHWM")
    return f"{(peak - before) / 2 ** 20:7.1f} MiB  (output {tuple(out.shape)} {out.dtype})"


def main():
    if len(sys.argv) == 3:
        print(measure(sys.argv[1], sys.argv[2]))
        return
    variant = 5 * N ** 3 * 2 / 2 ** 20
    print(f"one float16 output variant (5 x {N}^3): {variant:.1f} MiB")
    for helper in ("batched", "sequential"):
        for head in ("plain", "surface_normals", "inplane_direction"):
            res = subprocess.run([sys.executable, __file__, helper, head], capture_output=True, text=True)
            line = res.stdout.strip().splitlines()[-1] if res.returncode == 0 else f"error: {res.stderr.strip().splitlines()[-1]}"
            print(f"{helper:10s} {head:18s} peak over baseline {line}")


if __name__ == "__main__":
    main()
