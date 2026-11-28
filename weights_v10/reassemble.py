#!/usr/bin/env python3
"""重组 weights_v10/student_branch_v10.part* -> student_branch_v10.safetensors"""
import hashlib, json, os
D = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(D, "MANIFEST.json"), encoding="utf-8"))
out = os.path.join(D, "student_branch_v10.safetensors")
h = hashlib.sha256()
with open(out, "wb") as o:
    for c in m["chunks"]:
        b = open(os.path.join(D, os.path.basename(c["name"])), "rb").read()
        assert len(b) == c["bytes"] and hashlib.sha256(b).hexdigest() == c["sha256"]
        h.update(b); o.write(b)
assert h.hexdigest() == m["sha256"]
print("[ok]", out, os.path.getsize(out), "bytes")
