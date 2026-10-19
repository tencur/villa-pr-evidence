"""Print the state (chunk files of any key layout) of a level copy: stored chunk files, which chunks still hold the original data, compressor."""
import json, sys
from pathlib import Path
import numpy as np, zarr
work, pristine = Path(sys.argv[1]), Path(sys.argv[2])
if not (work / ".zarray").exists():
    print(f"  no array at {work.name} any more ({'folder gone' if not work.exists() else 'no .zarray'}; no {work.name}_tmp either: {not Path(str(work) + '_tmp').exists()})")
    sys.exit(0)
ref = zarr.open(str(pristine), mode="r")
files = sorted(str(p.relative_to(pristine)) for p in pristine.rglob("*") if p.is_file() and not p.name.startswith("."))
now = [q for q in work.rglob("*") if q.is_file() and not q.name.startswith(".")]
z = zarr.open(str(work), mode="r")
same = 0
for f in files:
    c = [int(i) * 128 for i in f.split("/")]
    sl = tuple(slice(a, a + 128) for a in c)
    try:
        same += np.array_equal(z[sl], ref[sl])
    except Exception:
        pass
comp = json.loads((work / ".zarray").read_text())["compressor"]
print(f"  chunk files: {len(now):2d}   of the {len(files)} chunks stored in the public array, {same:2d} read back with the original data   compressor: {comp and comp.get('cname')}")
