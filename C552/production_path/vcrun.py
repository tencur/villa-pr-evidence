"""Launch a VC3D build with its agent bridge (--agent-bridge) and talk to it over the bridge socket.
bridge_client.py and vc3d_process.py are villa's own helpers: copy them next to this file from
volume-cartographer/apps/VC3D/agent_bridge/test/. SNAP/<tag>/bin/VC3D is a VC3D build (build_state.sh copies the
binary and its libraries there). QT_QPA_PLATFORM defaults to offscreen; WSC_QPA=xcb runs it on an X display (Xvfb).
VC3D pauses remote-cache growth below 20 GiB free disk; the build host had less, so the floor is set to 2 GiB in the
run's own VC3D.ini (perf/remote_cache_min_free_gib, a VC3D setting)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_client import BridgeClient, BridgeError
from vc3d_process import VC3DProcess
SNAP = "/home/pi/scrollprize_work/runs/WSC/snap"

def launch(tag, workdir, volpkg=None, name=None, extra=()):
    os.makedirs(workdir, exist_ok=True)
    cfg = os.path.join(workdir, "xdg"); os.makedirs(cfg, exist_ok=True)
    name = name or f"wsc-{tag}-{os.getpid()}"
    args = ["--agent-bridge", "--agent-bridge-name", name, *extra]
    if volpkg: args += ["--volpkg", volpkg]
    env = {"QT_QPA_PLATFORM": os.environ.get("WSC_QPA", "offscreen"), "XDG_CONFIG_HOME": cfg, "HOME": workdir,
           "VC3D_CONFIG_DIR": os.path.join(workdir, "vc3dcfg"), "TMPDIR": os.path.join(workdir, "tmp")}
    os.makedirs(env["TMPDIR"], exist_ok=True)
    # rackpi6 has < 20 GiB free (VC3D default floor); let the remote cache grow down to 2 GiB free.
    os.makedirs(env["VC3D_CONFIG_DIR"], exist_ok=True)
    ini = os.path.join(env["VC3D_CONFIG_DIR"], "VC3D.ini")
    txt = open(ini).read() if os.path.exists(ini) else ""
    if "remote_cache_min_free_gib" not in txt:
        if "[perf]" in txt: txt = txt.replace("[perf]", "[perf]\nremote_cache_min_free_gib=2", 1)
        else: txt += "\n[perf]\nremote_cache_min_free_gib=2\n"
        open(ini, "w").write(txt)
    p = VC3DProcess(f"{SNAP}/{tag}/bin/VC3D", args, env_overrides=env)
    sock = p.wait_for_handshake(120)
    c = BridgeClient(sock)
    return p, c

def call(c, m, params=None, timeout=60):
    try:
        r = c.call(m, params, timeout=timeout)
        return r[0] if isinstance(r, tuple) else r
    except BridgeError as e:
        return {"__error__": e.code, "message": e.message, "data": e.data}
