"""C341 real-data proof: extract_surface_tracks' CPU backend (Kimimaro), with the shipped settings (downsample 4,
dust_threshold 400, ribbons 4 thick every 16, vertical slabs 4 thick every 64, interjoint paths), on the published
PHerc0500P2 surface prediction (volume 20250820143440, model m7, L0, th0.2; 128 x 256 x 256 at z 3456, y/x 2048) with
z_min/z_max set to the crop (8..120). Counts the tracks written and the slabs left without any track.
usage: python rd_c341.py <spiral-fitting dir> <prediction .npy>"""
import dbm, functools, os, pickle, sys, tempfile
sys.path.insert(0, sys.argv[1])
import numpy as np, kimimaro
import extract_surface_tracks as est
pred = np.load(sys.argv[2])
est.open_predictions = lambda path: pred
est.z_min, est.z_max = 8, 120
est.get_skeleton_tracks = functools.partial(
    est.get_skeleton_tracks, skeletonize=lambda labels, **k: kimimaro.skeletonize(labels, **{**k, "parallel": 1}))
out = os.path.join(tempfile.mkdtemp(prefix="c341_"), "tracks.dbm")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    est.run_cpu("PHerc0500P2 surface prediction crop", out)
with dbm.open(out) as d:
    entries = {k.decode(): pickle.loads(d[k]) for k in d.keys()}
n = sum(len(v) for v in entries.values())
pts = sum(len(t) for v in entries.values() for t in v)
empty = sorted(k for k, v in entries.items() if not v)
print(f"input: published PHerc0500P2 surface prediction {pred.shape}, {int((pred > 0).sum())} surface voxels; shipped settings")
print(f"dust threshold given to the skeletonizer: {est.skeletonize_kwargs()['dust_threshold']} "
      f"(prepare_cc_labels cutoff: {max(1, est.dust_threshold // est.downsample_factor ** 3)} downsampled voxels)")
print(f"slabs: {len(entries)}, slabs without any track: {len(empty)}")
print(f"tracks written: {n}, track points: {pts}")
