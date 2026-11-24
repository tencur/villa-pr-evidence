"""Run STEP=inference from a directory laid out like the GPU image's /app.

Usage: python gpu_image_layout_run.py <optimized_inference dir> <workdir>

The gpu stage of the Dockerfile is parsed and exactly the files/dirs it COPYs into
/app are copied into <workdir>/app (no docker build, no GPU).  entrypoint.py is then
started the way the image starts it (ENTRYPOINT python /app/entrypoint.py,
PYTHONPATH=/app, IMAGE_ROLE=gpu) with STEP=inference and a SURFACE_VOLUME_ZARR that
does not exist, from an unrelated working directory.  The run cannot succeed; the
question is how far it gets: with every module present it must reach the
surface-volume check.  The Python environment must provide the image's pip
dependencies (torch, vc_reader, ...); its site-packages stand in for the image's.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def gpu_stage_app_copies(dockerfile):
    names, inside = [], False
    for line in dockerfile.read_text().splitlines():
        s = line.strip()
        if s.upper().startswith("FROM "):
            inside = s.lower().endswith(" as gpu")
            continue
        if inside and s.upper().startswith("COPY ") and "--from" not in s and s.split()[-1].startswith("/app"):
            names.extend(p.rstrip("/") for p in s.split()[1:-1])
    return names


def main():
    src, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    app = work / "app"
    shutil.rmtree(work, ignore_errors=True)
    app.mkdir(parents=True)
    copied = gpu_stage_app_copies(src / "Dockerfile")
    for name in copied:
        if (src / name).is_dir():
            shutil.copytree(src / name, app / name)
        else:
            shutil.copy2(src / name, app / name)
    local_py = sorted(p.name for p in src.glob("*.py"))
    print(f"gpu stage COPY into /app: {' '.join(copied)}")
    print(f"local .py files not in the image: {' '.join(n for n in local_py if n not in copied)}")
    (work / "run").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    extra = os.environ.get("IMAGE_SITE_PACKAGES", "")  # e.g. a vc_reader build outside the venv
    env.update({
        "PYTHONPATH": os.pathsep.join([str(app)] + ([extra] if extra else [])),
        "PYTHONUNBUFFERED": "1", "IMAGE_ROLE": "gpu", "CUDA_VISIBLE_DEVICES": "",
        "STEP": "inference", "MODEL": "timesformer-scroll5", "MODEL_TYPE": "timesformer",
        "START_LAYER": "0", "END_LAYER": "26",
        "SURFACE_VOLUME_ZARR": str(work / "run" / "absent_surface_volume.zarr"),
    })
    p = subprocess.run([sys.executable, str(app / "entrypoint.py")], cwd=work / "run", env=env,
                       capture_output=True, text=True, timeout=600)
    out = (p.stdout + p.stderr).replace(str(work), "<work>")
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    print("STEP=inference exit status", p.returncode)
    for ln in tail:
        print("  " + ln)
    if "No module named 'device_utils'" in out:
        print("RESULT: stops at import: No module named 'device_utils'")
    elif "Surface volume zarr not found" in out:
        print("RESULT: imports complete, reaches the surface-volume check")
    else:
        print("RESULT: other")


if __name__ == "__main__":
    main()
