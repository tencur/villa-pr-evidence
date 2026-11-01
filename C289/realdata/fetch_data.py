"""Fetch the public lasagna channels used by rd_c289.py (anonymous S3, ~1 MB): PHerc0139 lasagna run
20260102150214-lasagna-20260419180421-L2, channels cos / grad_mag / nx / ny, level 4, z 2400-2463, y/x 800-863."""
import numpy as np, s3fs, zarr
fs = s3fs.S3FileSystem(anon=True)
b = ("vesuvius-challenge-open-data/PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/"
     "PHerc0139-20260102150214-lasagna-20260724_{}.ome.zarr/4")
np.save("lasagna_L4_4ch.npy", np.stack([zarr.open(s3fs.S3Map(b.format(c), s3=fs), mode="r")[2400:2464, 800:864, 800:864]
                                         for c in ("cos", "grad_mag", "nx", "ny")]))
