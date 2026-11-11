"""Build-time check (audit fix): the three upstream extensions were compiled.

Deliberately does NOT import sageattention: _qattn_sm90 links -lcuda, and libcuda.so.1 is only
injected by the NVIDIA container runtime at `docker run`, so a build-time import always reports
SM90_ENABLED=False, which is misleading.
"""
import glob
import os
import sysconfig

roots = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
for ext in ("_qattn_sm90", "_qattn_sm89", "_fused"):
    hits = [h for r in roots for h in glob.glob(os.path.join(r, "sageattention", ext + "*.so"))]
    assert hits, f"missing compiled extension {ext} under {sorted(roots)}"
    print("built", hits[0])
