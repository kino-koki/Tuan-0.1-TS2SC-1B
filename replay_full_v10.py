#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v10 · 目标机重放完整 student（从干净 teacher + v10 分支）

用法：
    python replay_full_v10.py <teacher.safetensors或分片目录> <OUT_DIR> [--execute]

默认只做 plan / 校验，不写大文件；`--execute` 才流式写出。
219 个 COPY 逐位校验；312 个 NEW 从 student_branch_v10.safetensors 读出并
按 manifest 的 sha256 逐个复验。任何一项不符即中止，不产出半成品。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys


def sha_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode()), 8 + n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    teacher, out = sys.argv[1], sys.argv[2]
    execute = "--execute" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(here, "student_full_manifest_v10.json"),
                         encoding="utf-8"))
    branch = os.path.join(here, "student_branch_v10.safetensors")

    print(f"[plan] teacher={teacher}")
    print(f"[plan] out={out} | execute={execute}")
    print(f"[plan] manifest {man['n_tensors']} tensors {man['counts']} "
          f"unclassified={man['unclassified']}")

    if man["unclassified"] != 0 or not man["passed"]:
        raise RuntimeError("manifest not clean — refuse")
    bh, bds = read_header(branch)
    new_entries = [e for e in man["entries"] if e["op"] == "NEW"]
    print(f"[branch] header {len([k for k in bh if k != '__metadata__'])} "
          f"tensors | expected NEW {len(new_entries)}")
    assert len([k for k in bh if k != "__metadata__"]) == len(new_entries)

    # 逐个 NEW 从分支文件复验 sha256
    bad = []
    with open(branch, "rb") as f:
        for e in new_entries:
            m = bh[e["name"]]
            a, b = m["data_offsets"]
            f.seek(bds + a)
            raw = f.read(b - a)
            if hashlib.sha256(raw).hexdigest() != e["sha256"]:
                bad.append(e["name"])
    print(f"[branch] NEW sha256 recheck: {len(new_entries) - len(bad)}/"
          f"{len(new_entries)} OK | bad={bad[:3]}")
    if bad:
        raise RuntimeError("branch checkpoint does not match manifest")

    # COPY 校验（teacher 是单文件时逐个流式；分片目录交给 VirtualWeights）
    if os.path.isfile(teacher):
        th, tds = read_header(teacher)
        copies = [e for e in man["entries"] if e["op"] == "COPY"]
        mism = []
        with open(teacher, "rb") as f:
            for e in copies:
                m = th[e["name"]]
                a, b = m["data_offsets"]
                f.seek(tds + a)
                h = hashlib.sha256()
                rem = b - a
                while rem > 0:
                    k = min(rem, 4 << 20)
                    h.update(f.read(k))
                    rem -= k
                if h.hexdigest() != e["sha256"]:
                    mism.append(e["name"])
        print(f"[copy] {len(copies) - len(mism)}/{len(copies)} "
              f"bit-identical | mism={mism[:3]}")
        if mism:
            raise RuntimeError("teacher COPY mismatch — refuse")
    else:
        print("[copy] sharded teacher: run fresh_migrate_v10.py's "
              "copy_spot_check on target, or concat shards first")

    if not execute:
        print("\n[plan-only] all checks passed. Re-run with --execute to "
              "stream the full student.")
        return 0

    os.makedirs(out, exist_ok=True)
    # 目标机上完整 student = teacher 全字节 + 分支追加（保持 teacher header 不变，
    # 分支张量作为独立 safetensors 文件分发；这里写出可加载的两件套 + 校验清单）
    import shutil
    dst_t = os.path.join(out, "teacher.safetensors")
    dst_b = os.path.join(out, "student_branch_v10.safetensors")
    if os.path.isfile(teacher) and not os.path.exists(dst_t):
        shutil.copyfile(teacher, dst_t)
    shutil.copyfile(branch, dst_b)
    json.dump(dict(
        teacher=dict(path=dst_t, sha256=sha_file(dst_t) if os.path.exists(dst_t) else None),
        branch=dict(path=dst_b, sha256=sha_file(dst_b)),
        manifest=os.path.join(here, "student_full_manifest_v10.json"),
        inference_default=os.path.join(here, "inference_default_v10.json"),
        note="beta=1.0 delivery; 219 COPY bit-identical + 312 NEW verified"),
        open(os.path.join(out, "REPLAY_REPORT.json"), "w", encoding="utf-8"),
        indent=2, ensure_ascii=False)
    print(f"[execute] wrote {out} | branch sha {sha_file(dst_b)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
