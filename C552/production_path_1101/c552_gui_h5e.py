"""C552 on current upstream main through the real VC3D app under Xvfb (Qt xcb), with real keyboard/mouse input (XTEST).
Project (object store, since #2358) = a new .vcobj holding the public PHerc0800 segment 20251028220955 (tifxyz, imported
with vc_object_ingest) + the public PHerc0800 volume (streamed from the open-data bucket), opened as a .vcproj.
Steps: select the segment once (agent bridge segments.activate = the segment panel's selection); Actions > Transforms >
Rotate (mouse clicks on the menu); click "Flip horizontally"; click Apply. Then open Rotate again WITHOUT selecting the
segment again, flip, Apply. Two flips = the original. After each Apply the store's current geometry is exported with
vc_object_export and compared with the original x/y/z.tif (every valid vertex within 0.001).
usage: c552_gui_1030.py <tag> explore|run   (tag names the build: main or fix; clicks from clicks.json)"""
import json, os, shutil, subprocess, sys, time
import numpy as np, tifffile
R = "/home/pi/scrollprize_work/vcbuild/h5E/app"  # H5 lane E copy of runs/C552RD harness
sys.path.insert(0, R + "/bridge"); sys.path.insert(0, R)
from bridge_client import BridgeClient, BridgeError
from vc3d_process import VC3DProcess
import xclick, xshot
tag, mode = sys.argv[1], sys.argv[2]
BIN = os.environ.get("VC3D_BIN", "/home/pi/scrollprize_work/vcbuild/h5E/build/bin/VC3D")
OS = "/home/pi/scrollprize_work/vcbuild/h5E/build_os/bin"
SEGSRC = "/home/pi/scrollprize_work/runs/SW4B/C430/20251028220955"  # PHerc0800/segments/20251028220955-auto_grown_20251028220955262/mesh/intermediate/tifxyz_original
VOL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr"
W = f"{R}/gui_{tag}_{mode}"; shutil.rmtree(W, ignore_errors=True); os.makedirs(W + "/src")
SEG = W + "/src/20251028220955"; shutil.copytree(SEGSRC, SEG)
import uuid as _uuid
_m = json.load(open(SEG + "/meta.json")); _m["uuid"] = str(_uuid.uuid4()); _m["name"] = "20251028220955"
if os.environ.get("VID"): _m["volume_uuid"] = os.environ["VID"]; _m["scale_factor"] = 1  # bind to the open volume (segment coords = level-0 voxels)
json.dump(_m, open(SEG + "/meta.json", "w"), indent=2)  # the published meta.json says uuid "output_tifxyz"; the store needs a real UUID
DB = W + "/c552.vcobj"
ing = subprocess.run([f"{OS}/vc_object_ingest", "--db", DB, "--author", "c552rd", "--surface", SEG, "--no-progress"],
                     capture_output=True, text=True)
print("vc_object_ingest:", ing.returncode, ing.stdout.strip()[-300:], ing.stderr.strip()[-300:])
proj = W + "/c552.vcproj"
json.dump(dict(format="vcproj", version=1, name="c552 PHerc0800",
               stores=[dict(id="main", location=DB, read_only=False)],
               selections={k: "main" for k in ("surface", "fiber", "point_collection", "umbilicus", "volume_transform")},
               volumes=[dict(location=VOL, tags=[])], normal_grids=[], lasagna_datasets=[], view={}), open(proj, "w"), indent=2)
cfg = W + "/config"; os.makedirs(cfg)
open(cfg + "/VC3D.ini", "w").write("[viewer]\nusername=c552rd\nremote_cache_dir=" + W + "/remote-cache\n"
                                   "[project]\nshow_open_data_catalog_on_startup=false\n[perf]\nremote_cache_min_free_gib=2\n")
def xyz(d): return np.stack([tifffile.imread(f"{d}/{a}.tif").astype(np.float64) for a in "xyz"], -1)
orig = xyz(SEGSRC); valid = orig[..., 2] > 0
print(f"segment 20251028220955: grid {orig.shape[1]}x{orig.shape[0]}, {int(valid.sum())} valid vertices")
p = VC3DProcess(BIN, ["--agent-bridge", "--agent-bridge-name", f"c552rd-{tag}-{os.getpid()}", "--project", proj],
                env_overrides={"QT_QPA_PLATFORM": "xcb", "DISPLAY": os.environ.get("DISPLAY", ":99"), "XDG_CONFIG_HOME": cfg,
                               "VC3D_CONFIG_DIR": cfg, "HOME": W, "TMPDIR": W + "/tmp"})
os.makedirs(W + "/tmp", exist_ok=True)
c = BridgeClient(p.wait_for_handshake(180))
def call(m, params=None, timeout=120):
    try:
        r = c.call(m, params, timeout=timeout); return r[0] if isinstance(r, tuple) else r
    except BridgeError as e:
        return {"__error__": e.code, "message": e.message}
segs = {}
for i in range(180):
    r = call("segments.list")
    segs = r.get("segments", []) if isinstance(r, dict) else []
    if segs and call("state.get").get("volume"): break
    time.sleep(1)
print("open volume (state.get):", json.dumps(call("state.get").get("volume"))[:400])
print("segments.list:", [{k: s.get(k) for k in ("id", "name", "active")} for s in segs])
uuid = segs[0]["id"]
time.sleep(5)
r = call("segments.activate", {"segmentId": uuid}); time.sleep(8)
print("select segment (segments.activate):", (r.get("segment") or {}).get("active", r))
xshot.shot(f"{W}/selected.png")
clicks = json.load(open(R + "/clicks.json")) if os.path.exists(R + "/clicks.json") else {}
flip1 = None
def stored(label):
    global flip1
    out = f"{W}/export_{label.split()[1] if ' ' in label else label}"
    for k in range(10):
        e = subprocess.run([f"{OS}/vc_object_export", "--db", DB, "--uuid", uuid, "--output", out, "--overwrite"],
                           capture_output=True, text=True)
        if e.returncode == 0: break
        time.sleep(2)
    if e.returncode != 0:
        print(f"  {label}: export failed: {e.stderr.strip()[-300:]}"); return None
    cur = xyz(out)
    if cur.shape != orig.shape: print(f"  {label}: stored shape {cur.shape} != original {orig.shape}"); return cur
    v = valid & (cur[..., 2] > 0); d = np.abs(cur - orig)[v].max()
    st = "ORIGINAL" if d < 1e-3 else "FLIPPED"
    if flip1 is not None and st == "FLIPPED":
        st += " (same as after Apply 1)" if np.abs(cur - flip1)[v].max() < 1e-3 else " (differs from Apply 1)"
    print(f"  {label}: stored geometry (vc_object_export of the current revision) {st}; max vertex move vs original {d:.1f} vx")
    return cur
def open_rotate(n):
    xclick.click(147, 9); time.sleep(1.5); xshot.shot(f"{W}/menu{n}_actions.png")            # Actions
    if "transforms" not in clicks: return
    xclick.click(*clicks["transforms"]); time.sleep(1.5); xshot.shot(f"{W}/menu{n}_transforms.png")  # Transforms >
    if "rotate" not in clicks: return
    xclick.click(*clicks["rotate"]); time.sleep(3); xshot.shot(f"{W}/apply{n}_tool.png")       # Rotate
def active(label):
    st = call("state.get"); a = st.get("activeSurface") if isinstance(st, dict) else st
    print(f"  {label}: state.get activeSurface = {json.dumps(a)} (selected segment {uuid})")
def flip_apply(n):
    xclick.click(*clicks["flip"]); time.sleep(1.0); xshot.shot(f"{W}/apply{n}_flip.png")
    xclick.click(*clicks["apply"]); time.sleep(15); xshot.shot(f"{W}/apply{n}_after.png")
    print(f"Apply {n}: Actions > Transforms > Rotate, tick 'Flip horizontally', Apply")
stored("before")
if mode == "explore":
    open_rotate(0); print("explore: screenshots in", W)
else:
    open_rotate(1); flip_apply(1)
    flip1 = stored("after Apply 1"); active("after Apply 1")
    print("  (the segment is NOT selected again)")
    open_rotate(2); flip_apply(2)
    stored("after Apply 2 (two flips = original)"); active("after Apply 2")
    if os.environ.get("NOCHANGE"):
        # Apply with no change: tick Flip, untick it, Apply; then Rotate again (screenshot apply4_tool.png).
        open_rotate(3)
        xclick.click(*clicks["flip"]); time.sleep(2.0); xshot.shot(f"{W}/apply3_flip.png")
        xclick.click(*clicks["flip"]); time.sleep(2.0); xshot.shot(f"{W}/apply3_unflip.png")
        xclick.click(*clicks["apply"]); time.sleep(5); xshot.shot(f"{W}/apply3_after.png")
        print("Apply 3: Rotate, tick and untick Flip horizontally, Apply (no change)")
        stored("after Apply 3 (no change = original)"); active("after Apply 3")
        open_rotate(4); print("  Rotate again after Apply 3: see apply4_tool.png")
print("VC3D running:", p.is_running())
p.terminate()
open(W + "/vc3d.log", "w").write("\n".join(p.tail_log(8000)))
