#!/usr/bin/env python
"""optimized_inference sets OPENCV_IO_MAX_IMAGE_PIXELS before anything imports cv2, and OpenCV reads that variable once,
when cv2 is imported. Import the package the way its entry points do, then read a tiny PNG with cv2 and a real layer
as PNG and as 16-bit TIFF through the package's own layer reader.

usage: opencv_pixel_limit_probe.py <optimized_inference dir> <16-bit layer .tif> <work dir>
The probe runs in a subprocess so the environment is exactly what the package sets.
"""
import subprocess, sys
code, layer, work = sys.argv[1:4]
script = f"""
import os, sys
os.environ.pop("OPENCV_IO_MAX_IMAGE_PIXELS", None)   # as in a fresh container: the package's own default decides
sys.path.insert(0, {code!r})
import processing                                     # sets the default, then imports cv2
import cv2, numpy as np, tifffile
from PIL import Image
print(f"after 'import processing': OPENCV_IO_MAX_IMAGE_PIXELS={{os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']}}  (OpenCV {{cv2.__version__}}, Python {{sys.version.split()[0]}})")
os.makedirs({work!r}, exist_ok=True)
small = os.path.join({work!r}, "small.png"); Image.fromarray(np.full((64, 64), 7, np.uint8)).save(small)
raw = tifffile.imread({layer!r}); layer_png = os.path.join({work!r}, "layer_as_png.png")
Image.fromarray((raw >> 8).astype(np.uint8)).save(layer_png)
def probe(label, fn):
    try:
        r = fn()
        print(label + ": " + ("None (decode failed)" if r is None else f"read, shape {{r.shape}} dtype {{r.dtype}}"))
    except Exception as e:
        print(label + f": {{type(e).__name__}}: " + " ".join(str(e).split())[-140:])
probe("cv2.imread of a 64x64 PNG", lambda: cv2.imread(small, cv2.IMREAD_GRAYSCALE))
probe(f"layer reader (_read_gray_any) on a real {{raw.shape[1]}}x{{raw.shape[0]}} layer saved as PNG", lambda: processing._read_gray_any(layer_png))
probe("layer reader (_read_gray_any) on the same layer as 16-bit TIFF (production path, tifffile)", lambda: processing._read_gray_any({layer!r}))
"""
r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode:
    print(r.stderr[-1500:])
