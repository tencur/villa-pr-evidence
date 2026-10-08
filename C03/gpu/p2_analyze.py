#!/usr/bin/env python3
"""Compare the P2 predictions pulled from the pod. Usage: p2_analyze.py <evidence_dir_with_prediction-*.tif and *stats.json>"""
import sys, os, json, glob
import numpy as np, tifffile
d=sys.argv[1]
stats={}
for f in glob.glob(os.path.join(d,"**","stats.json"), recursive=True)+glob.glob(os.path.join(d,"*stats.json")):
    st=json.load(open(f)); stats[st["tag"]]=st
preds={}
for f in glob.glob(os.path.join(d,"**","prediction-*.tif"), recursive=True)+glob.glob(os.path.join(d,"prediction-*.tif")):
    tag=os.path.basename(f)[len("prediction-"):-4]; preds[tag]=tifffile.imread(f)
print("runs:", sorted(preds))
for tag in sorted(preds):
    a=preds[tag]; st=stats.get(tag,{})
    print(f"{tag:<14} shape={a.shape} mean={a.mean():7.2f} std={a.std():6.2f} frac>128={float((a>128).mean()):.4f} input_frac_255={st.get('zarr',{}).get('frac_255')} prep_s={st.get('prepare_s')} infer_s={st.get('infer_s')} gpu_peak_MiB={st.get('gpu_mem_peak_mib')}")
def cmp(x,y):
    A=preds[x].astype(np.float32); B=preds[y].astype(np.float32)
    mad=float(np.abs(A-B).mean()); c=float(np.corrcoef(A.ravel()[::7], B.ravel()[::7])[0,1]); same=float((preds[x]==preds[y]).mean())
    print(f"  {x} vs {y}: mean|diff|={mad:.3f} corr={c:.4f} identical_frac={same:.4f}")
pairs=[("A-L16-S0E26","C-L16-S0E26"),("A-L8-S1E27","C-L8-S1E27"),("A-L8-S1E27","C-L8-S2E28"),("C-L8-S1E27","C-L8-S2E28")]
print("pairwise:")
for x,y in pairs:
    if x in preds and y in preds: cmp(x,y)
# tile periodicity of a degenerate prediction: std of the per-64px-block means vs within-block std
for tag in sorted(preds):
    a=preds[tag].astype(np.float32); H,W=a.shape; h=(H//64)*64; w=(W//64)*64; b=a[:h,:w].reshape(h//64,64,w//64,64)
    print(f"{tag:<14} block-mean std={b.mean(axis=(1,3)).std():6.2f}  within-block std={b.std(axis=(1,3)).mean():6.2f}")
