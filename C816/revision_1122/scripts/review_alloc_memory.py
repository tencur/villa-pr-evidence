"""Memory and time of the mirroring TTA helpers, and the blend sums, on CPU.

Same fixture as review_peak_memory.py: one 128^3 patch, a model with 2 surface scores + a
3-channel head (5 float16 output channels), batched and sequential helper. Measured here as
the reviewer did: the peak of live CPU tensor allocations (torch.profiler allocation events),
in a fresh process per row, plus the wall time of the helper call (1 thread). A 4-channel
nearest_component head with the shipped losses (6 channels) is added as an extra row, with
its own plain average (plain_6ch) for comparison.

Also printed: the float32 sums per voxel that the in-memory blender keeps for a 5-channel
model with and without an axis head (InMemoryBlender.required_bytes; the staged blend keeps
the same per output chunk).

Run:  python review_alloc_memory.py
"""
import inspect
import subprocess
import sys
import time
from pathlib import Path

N = 128


def measure(helper, head):
    import torch
    import yaml
    import vesuvius.models
    from vesuvius.models.run import tta
    torch.set_num_threads(1)
    channels = 4 if head in ("nearest_component", "plain_6ch") else 3
    cfg = {"out_channels": channels, "auxiliary_task": True, "task_type": head}
    if head == "nearest_component":
        path = Path(vesuvius.models.__file__).parent / "configuration" / "aux_tasks" / "nearest_component_3d.yaml"
        cfg["losses"] = yaml.safe_load(path.read_text())["auxiliary_tasks"]["nearest_component"]["losses"]
    targets = {"surface": {"out_channels": 2}, "h": cfg}
    info = {"surface": {"start_channel": 0, "end_channel": 2, "out_channels": 2},
            "h": {"start_channel": 2, "end_channel": 2 + channels, "out_channels": channels}}

    def model(x):
        y = x.half()
        return {"surface": torch.cat([y, -y], dim=1), "h": torch.cat([y, -y, y, y][:channels], dim=1)}

    def concat(out):
        return torch.cat([out["surface"], out["h"]], dim=1)

    kwargs = {"is_multi_task": True, "concat_multi_task_outputs": concat}
    has_groups = "channel_groups" in inspect.signature(tta.infer_with_tta_batched_3d).parameters
    kinds = "-"
    if not head.startswith("plain"):
        if not has_groups:
            return "n/a (no channel groups in this revision)"
        kwargs["channel_groups"] = tta.tta_channel_groups(targets, info, 3)
        kinds = ",".join(g["kind"] for g in kwargs["channel_groups"])
    x = torch.rand(1, 1, N, N, N)

    def run():
        with torch.inference_mode():
            if helper == "batched":
                return tta.infer_with_tta_batched_3d(model, x, **kwargs)
            return tta.infer_with_tta_sequential(model, x, "mirroring", **kwargs)

    run()  # warm-up
    start = time.perf_counter()
    run()
    seconds = time.perf_counter() - start
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        run()
    live = peak = 0
    for event in sorted(prof.events(), key=lambda e: e.time_range.start):
        live += event.self_cpu_memory_usage
        peak = max(peak, live)
    return f"{peak / 2 ** 20:7.1f} MiB live tensors, {seconds:6.2f} s  (groups: {kinds})"


def blend_sums():
    from vesuvius.models.run.blending import InMemoryBlender
    coords = [(0, 0, 0)]
    plain = InMemoryBlender.required_bytes(coords, (16, 16, 16), 5, (16, 16, 16))
    if "axis_channels" in inspect.signature(InMemoryBlender.required_bytes).parameters:
        axis = InMemoryBlender.required_bytes(coords, (16, 16, 16), 5, (16, 16, 16), axis_channels=[(2, 3, 4)])
    else:
        axis = plain
    return plain / 16 ** 3, axis / 16 ** 3


def main():
    if len(sys.argv) == 3:
        print(measure(sys.argv[1], sys.argv[2]))
        return
    print(f"one float16 output variant (5 x {N}^3): {5 * N ** 3 * 2 / 2 ** 20:.1f} MiB")
    for helper in ("batched", "sequential"):
        for head in ("plain", "surface_normals", "inplane_direction", "plain_6ch", "nearest_component"):
            res = subprocess.run([sys.executable, __file__, helper, head], capture_output=True, text=True)
            line = (res.stdout.strip().splitlines()[-1] if res.returncode == 0
                    else f"error: {res.stderr.strip().splitlines()[-1]}")
            print(f"{helper:10s} {head:18s} peak {line}")
    plain, axis = blend_sums()
    print(f"in-memory blend sums per voxel, 5 channels: {plain:.0f} bytes; with one axis head: {axis:.0f} bytes")


if __name__ == "__main__":
    main()
