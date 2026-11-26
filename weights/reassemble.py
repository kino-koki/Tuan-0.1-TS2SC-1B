#!/usr/bin/env python3
"""重组 weights/student_branch.part* -> student_branch_v8.safetensors 并校验 sha256。"""
import hashlib, json, os, sys
H = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(H, "MANIFEST.json"), encoding="utf-8"))
out = os.path.join(H, m["source"])
parts = []
for c in m["chunks"]:
    b = open(os.path.join(H, os.path.basename(c["name"])), "rb").read()
    assert len(b) == c["bytes"], c["name"]
    assert hashlib.sha256(b).hexdigest() == c["sha256"], c["name"]
    parts.append(b)
raw = b"".join(parts)
assert len(raw) == m["total_bytes"]
assert hashlib.sha256(raw).hexdigest() == m["sha256"]
open(out, "wb").write(raw)
print("OK", out, len(raw), "bytes", hashlib.sha256(raw).hexdigest())
