"""Open a public S3 surface volume through the pipeline's own LayersSource and report what it returns.

Only the credential mode is changed (anonymous instead of signed), because the bucket is public and this
machine has no AWS credentials. The cache code path is untouched.
usage: c04_read.py <code_dir> <s3_url>
"""
import sys, hashlib, json, os
code, url = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
import fsspec, numpy as np
_filesystem = fsspec.filesystem
def _anon_filesystem(protocol, *args, **kwargs):
    if protocol == "s3":
        kwargs["anon"] = True
    return _filesystem(protocol, *args, **kwargs)
fsspec.filesystem = _anon_filesystem
import processing, inference
processing.path_exists = inference.path_exists = lambda p: True   # signed existence probe; not part of the cache
src = inference.LayersSource(url)
h, w, c = src.shape
got = src.read_roi(0, h, 0, w)
# ground truth: same object read straight from the bucket, no cache layer
import s3fs, zarr
truth_arr = zarr.open(s3fs.S3Map(url[5:], s3=s3fs.S3FileSystem(anon=True)), mode="r")["0"]
truth = np.transpose(np.asarray(truth_arr[:]), (1, 2, 0))
print(json.dumps({"url": url.split("/segments/")[-1], "cwd_cache_entries": sum(len(f) for _, _, f in os.walk("zarr_cache")),
                  "pipeline_shape_HWC": [h, w, c], "true_shape_HWC": list(truth.shape),
                  "pipeline_sha": hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()[:12],
                  "true_sha": hashlib.sha256(np.ascontiguousarray(truth).tobytes()).hexdigest()[:12],
                  "matches_truth": bool(got.shape == truth.shape and np.array_equal(got, truth))}))
