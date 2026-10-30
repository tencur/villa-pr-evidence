"""Fetch the public crops used by the real-data proofs (anonymous S3, ~16 MB).
PHerc0500P2 9.362 um volume 20250820143440 level 0 and its published surface prediction (model m7, L0, th0.2),
z 3456-3583, y 2048-2303, x 2048-2303."""
import numpy as np, s3fs, zarr
fs = s3fs.S3FileSystem(anon=True)
b = "vesuvius-challenge-open-data/PHerc0500P2"
v = f"{b}/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr/0"
p = f"{b}/representations/predictions/surfaces/20250820143440-surface-20260413222639-surface-m7-L0-th0.2.zarr/0"
sl = (slice(3456, 3584), slice(2048, 2304), slice(2048, 2304))
np.save("ct_9um_z3456_y2048_x2048.npy", zarr.open(s3fs.S3Map(v, s3=fs), mode="r")[sl])
np.save("pred_9um_z3456_y2048_x2048.npy", zarr.open(s3fs.S3Map(p, s3=fs), mode="r")[sl])
