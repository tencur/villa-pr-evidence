"""C552 through the real VC3D app under Xvfb (Qt xcb), with real mouse clicks (XTEST) on VC3D's menu and the
Rotate tool's own widgets. Project = public PHerc0800 volume (streamed from the open-data bucket) + the public
PHerc0800 segment 20251028220955 (tifxyz, editable local copy). Steps: select the segment; Actions > Transforms >
Rotate; tick "Flip horizontally"; Apply; select the segment again; flip + Apply again (flip twice = the original). After each Apply the
segment's x/y/z.tif on disk are compared with the original files. The agent bridge only selects the segment
(segments.activate = SurfacePanelController::activateSurfaceById, the segment list's selection), waits, and reads state."""
import sys, json, time, os, shutil
import numpy as np, tifffile
sys.path.insert(0, "/home/pi/scrollprize_work/runs/WSC/bridge")
from vcrun import launch, call
import xclick, xshot
tag = sys.argv[1]
W = f"/home/pi/scrollprize_work/runs/WSC/C552/gui_{tag}"; shutil.rmtree(W, ignore_errors=True); os.makedirs(W + "/segments")
SEGSRC = "/home/pi/scrollprize_work/runs/SW4B/C430/20251028220955"  # PHerc0800/segments/20251028220955-auto_grown_20251028220955262/mesh/intermediate/tifxyz_original
SEG = W + "/segments/20251028220955"; shutil.copytree(SEGSRC, SEG)
VOL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr"
proj = W + "/pherc0800.volpkg.json"
json.dump({"name": "pherc0800", "version": 1, "volumes": [VOL], "segments": [W + "/segments"], "normal_grids": [], "lasagna_datasets": []}, open(proj, "w"), indent=2)
def xyz(d): return np.stack([tifffile.imread(f"{d}/{a}.tif").astype(np.float64) for a in "xyz"], -1)
orig = xyz(SEGSRC); valid = orig[..., 2] > 0
print(f"segment 20251028220955: grid {orig.shape[1]}x{orig.shape[0]}, {int(valid.sum())} valid vertices")
flip1 = None
def disk_state(label):
    global flip1
    cur = xyz(SEG)
    if cur.shape != orig.shape: print(f"  {label}: on disk shape {cur.shape} != original"); return cur
    v = valid & (cur[..., 2] > 0)
    d = np.abs(cur - orig)[v].max()
    st = "ORIGINAL" if d < 1e-3 else "FLIPPED"
    if flip1 is not None and st == "FLIPPED":
        st += " (same as after Apply 1)" if np.abs(cur - flip1)[v].max() < 1e-3 else " (differs from Apply 1)"
    print(f"  {label}: on disk (x/y/z.tif) {st}; max vertex move vs original {d:.1f} vx; mtime {time.strftime('%H:%M:%S', time.gmtime(os.path.getmtime(SEG + '/x.tif')))}")
    return cur
p, c = launch(tag, W, volpkg=proj)
for i in range(120):
    s = call(c, "state.get")
    if s.get("volume"): break
    time.sleep(1)
time.sleep(5)
print("select segment:", call(c, "segments.activate", {"segmentId": "20251028220955"}).get("activated")); time.sleep(6)
def rotate_flip_apply(n):
    xclick.click(147, 9); time.sleep(1.5)       # Actions
    xclick.click(183, 145); time.sleep(1.5)     # Transforms >
    xclick.click(490, 145); time.sleep(2.5)     # Rotate
    xshot.shot(f"{W}/apply{n}_tool.png")
    xclick.click(119, 149); time.sleep(1.0)     # [x] Flip horizontally
    xshot.shot(f"{W}/apply{n}_flip.png")
    t = os.path.getmtime(SEG + "/x.tif")
    xclick.click(172, 176)                      # Apply
    for k in range(60):
        time.sleep(1)
        if os.path.getmtime(SEG + "/x.tif") != t: break
    time.sleep(10)                               # let the save finish and the file watcher run
    xshot.shot(f"{W}/apply{n}_after.png")
    print(f"Apply {n}: Actions > Transforms > Rotate, tick 'Flip horizontally', Apply")
disk_state("before")
rotate_flip_apply(1)
flip1 = disk_state("after Apply 1")
print("  active surface after Apply 1:", call(c, "state.get").get("activeSurface"))
r = call(c, "segments.activate", {"segmentId": "20251028220955"}); time.sleep(6)
print("select the segment again in the segment panel (segments.activate):", {k: r.get(k) for k in ("activated", "alreadyActive")})
xshot.shot(f"{W}/reselected.png")
rotate_flip_apply(2)
disk_state("after Apply 2 (two flips = original)")
print("VC3D running:", p.is_running())
p.terminate()
open(W + "/vc3d.log", "w").write("\n".join(p.tail_log(8000)))
