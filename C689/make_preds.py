"""Synthetic surface-prediction volume: a few curved sheets (uint8 0/255), zarr v2 via vc_reader."""
import sys
import numpy as np
from vc_reader.reader import ZarrWriter

out = sys.argv[1]
Z, Y, X = 96, 704, 704
zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
r = np.hypot(yy - 352, xx - 352)
vol = np.zeros((Z, Y, X), np.uint8)
for r0 in (260, 300, 335):
    vol[np.abs(r - r0 - 3 * np.sin(zz / 9.0)) < 1.5] = 255
w = ZarrWriter.create(out, shape=vol.shape, chunks=(32, 128, 128), dtype=np.uint8, fill_value=0, overwrite=True)
w[:] = vol
print(out, vol.shape, int((vol > 0).sum()))
