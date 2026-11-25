# Test-environment shim only: the prebuilt vc_reader here predates #2680 and writes vc-delta3d for 3-D arrays only.
# Store 4-D vc-delta3d requests with blosc/zstd instead, identically for every tree tested.
import vc_reader.reader.writer as w
_orig = w.ZarrWriter.create.__func__
def _create(cls, path, **k):
    if k.get("compressor") == "vc-delta3d" and len(k["shape"]) != 3:
        k["compressor"] = "blosc"; k["cname"] = "zstd"
    return _orig(cls, path, **k)
w.ZarrWriter.create = classmethod(_create)
