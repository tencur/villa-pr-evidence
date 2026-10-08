#!/usr/bin/env python3
"""P2 driver: downstream effect of #1998 (16-bit layers) and #1997 (layer window) on the production optimized_inference
pipeline with the public Scroll 5 TimeSformer model. Mirrors entrypoint.py's prepare and inference steps on each code
version (A = official main e0bbb8b, C = A + our PRs), run from the code dir given by --code.

Runs (each: prepare window -> inference -> reduce -> uint8 prediction TIFF + stats JSON):
  --layers16 <dir>  uint16 layer TIFFs NN.tif ; --layers8 <dir> 8-bit copies (raw >> 8)
  --runs  role:layerset:start:end[,...]   e.g. A:16:0:26,C:16:0:26,A:8:1:27,C:8:1:27,C:8:2:28
"""
import argparse, hashlib, json, os, shutil, sys, time, importlib
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--code", required=True); ap.add_argument("--role", required=True)
    ap.add_argument("--layers16", required=True); ap.add_argument("--layers8", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--work", default="/tmp/p2work", help="local-disk scratch for the zarr and partitions (NFS volume reads were the bottleneck: 12 tiles/s, GPU 0%)")
    ap.add_argument("--workers", type=int, default=32, help="dataloader workers (production default min(8,cpu) starves the GPU: 113 tiles/s avg on a 96-vCPU host)"); ap.add_argument("--stride", type=int, default=16); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--tile", type=int, default=64)
    a=ap.parse_args()
    sys.path.insert(0, a.code); os.chdir(a.code)
    import torch
    import processing, inference
    from inference import CFG, run_inference
    from processing import create_surface_volume_zarr, reduce_partitions, write_tiled_tiff
    from model_timesformer import load_model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    role_code=a.role
    results=[]
    for spec in a.runs.split(","):
        role, lset, s, e = spec.split(":"); s=int(s); e=int(e)
        if role != role_code: continue
        ldir = a.layers16 if lset=="16" else a.layers8
        names=sorted(f for f in os.listdir(ldir) if f.endswith(".tif"))
        paths=[os.path.join(ldir,n) for n in names if s <= int(os.path.splitext(n)[0]) < e]
        assert len(paths)==e-s, (len(paths), s, e)
        tag=f"{role}-L{lset}-S{s}E{e}"; rdir=os.path.join(a.out, tag); shutil.rmtree(rdir, ignore_errors=True); os.makedirs(rdir)
        wdir=os.path.join(a.work, tag); shutil.rmtree(wdir, ignore_errors=True); os.makedirs(wdir); zarr_path=os.path.join(wdir,"surface_volume.zarr"); parts=os.path.join(wdir,"partitions")
        t0=time.time()
        # --- prepare step exactly as entrypoint.run_prepare_step does on this code version
        create_surface_volume_zarr(paths, zarr_path, chunk_size=1024, max_workers=8, use_compression=True)
        if hasattr(processing, "record_layer_window"):           # C only: entrypoint records the actual file window
            from processing import contiguous_layer_window, record_layer_window
            ls, le = contiguous_layer_window([os.path.basename(p) for p in paths]); record_layer_window(zarr_path, ls, le)
        t_prep=time.time()-t0
        import zarr as _z; zz=_z.open(zarr_path, mode="r"); zstats={"shape": list(zz.shape), "dtype": str(zz.dtype)}
        samp=np.asarray(zz[::97, ::97, :]); zstats.update({"sample_min": int(samp.min()), "sample_max": int(samp.max()), "sample_mean": float(samp.mean()), "frac_255": float((samp==255).mean())})
        # --- inference step exactly as entrypoint.run_inference_step does on this code version
        CFG.in_chans = e - s; CFG.tile_size=a.tile; CFG.size=a.tile; CFG.stride=a.stride; CFG.batch_size=a.batch; CFG.prefetch_factor=8
        CFG.workers=a.workers; CFG.num_parts=1; CFG.part_id=0; CFG.zarr_output_dir=parts; os.makedirs(parts, exist_ok=True)
        model=load_model(a.model, device, num_frames=CFG.in_chans)
        # --- model sanity probe: does the output depend on the input? (zero tile vs real tile vs noise tile, fed like the dataset does)
        try:
            import zarr as _zz
            zp=_zz.open(zarr_path, mode="r"); H,W,Cc=zp.shape
            def tile_at(y,x): return np.asarray(zp[y:y+a.tile, x:x+a.tile, :])
            real=None
            for yy in range(H//4, 3*H//4, 256):
                for xx in range(W//4, 3*W//4, 256):
                    t=tile_at(yy,xx)
                    if (t!=0).mean()>0.95: real=t; break
                if real is not None: break
            tiles={"zero":np.zeros((a.tile,a.tile,Cc),np.uint8),"noise":np.random.default_rng(0).integers(0,200,(a.tile,a.tile,Cc),dtype=np.uint8)}
            if real is not None: tiles["real"]=real
            probe={}
            model.eval()
            with torch.no_grad():
                for k,t in tiles.items():
                    x=np.clip(t,0,CFG.max_clip_value).astype(np.float32)/CFG.max_clip_value
                    xt=torch.from_numpy(x).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)   # (B,1,C,H,W)
                    y=model.forward(xt).float().cpu().numpy().ravel()
                    probe[k]={"in_mean":float(x.mean()),"out_mean":float(y.mean()),"out_std":float(y.std()),"out_min":float(y.min()),"out_max":float(y.max()),"n":int(y.size)}
            if "real" in probe: probe["real_vs_zero_out_mean_diff"]=abs(probe["real"]["out_mean"]-probe["zero"]["out_mean"])
            print("PROBE", json.dumps(probe), flush=True)
        except Exception as ex:
            probe={"error":repr(ex)[:300]}; print("PROBE", json.dumps(probe), flush=True)
        if hasattr(processing, "resolve_zarr_layer_window"):
            start_z, end_z = processing.resolve_zarr_layer_window(zarr_path, s, e)     # C: rebased onto the cropped zarr
        else:
            start_z, end_z = s, e                                                    # A: absolute indices re-applied (the C47 bug)
        t1=time.time()
        res=run_inference(zarr_path, model, device, is_reverse_segment=False, start_z=start_z, end_z=end_z)
        t_inf=time.time()-t1
        # --- reduce step exactly as entrypoint.run_reduce_step
        # reduce_partitions copies partitions into the fixed /tmp/partition_cache and REUSES a cached copy if present (candidate C02);
        # production runs each step in a fresh container, so clear it before every reduce to mirror that.
        shutil.rmtree("/tmp/partition_cache", ignore_errors=True)
        pred_shape=_z.open(os.path.join(parts,"mask_pred_part_000.zarr"), mode="r").shape
        it, shape = reduce_partitions(parts, 1, pred_shape, 1024)
        tif=os.path.join(rdir,"prediction.tif"); write_tiled_tiff(it, shape, tif, 1024, None)
        import tifffile; pred=tifffile.imread(tif)
        st={"tag":tag,"role":role,"layerset":lset,"start":s,"end":e,"start_z":start_z,"end_z":end_z,"n_layers":len(paths),
            "prepare_s":round(t_prep,1),"infer_s":round(t_inf,1),"zarr":zstats,"pred_shape":list(pred.shape),"pred_dtype":str(pred.dtype),
            "pred_mean":float(pred.mean()),"pred_std":float(pred.std()),"pred_frac_gt_128":float((pred>128).mean()),"pred_frac_gt_64":float((pred>64).mean()),
            "pred_sha256":sha(tif),"layer_sha256":{os.path.basename(p):sha(p) for p in paths[:2]},
            "gpu_mem_peak_mib": float(torch.cuda.max_memory_allocated()/2**20) if torch.cuda.is_available() else None, "probe": probe}
        json.dump(st, open(os.path.join(rdir,"stats.json"),"w"), indent=1); results.append(st); print(json.dumps(st), flush=True)
        shutil.rmtree(wdir, ignore_errors=True)   # local scratch; TIFF + stats on the volume are the evidence
        del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    json.dump(results, open(os.path.join(a.out, f"results-{role_code}.json"),"w"), indent=1)

if __name__ == "__main__": main()
