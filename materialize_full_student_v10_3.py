#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本仓自包含的字节组装完整 student.safetensors 并逐位校验。

用法（目标机，需要约 2.6 GiB 磁盘）：
    python materialize_full_student_v10_3.py --out student_full.safetensors

读取：
    teacher_shards/MANIFEST.json + part_*.bin   → 还原 2.01 GiB teacher 字节
    weights_v10/student_branch_v10.part*        → 312 个 SGC/外挂 NEW 张量
    student_smm_v10_3.safetensors               → 48 个 SMM NEW 张量
    student_scoped_manifest_v10_3.json          → 579 条张量账本

写出：完整 safetensors（579 张量），并对每条逐字节 sha256 复验。
"""
import argparse, hashlib, json, os, struct

ROOT = os.path.dirname(os.path.abspath(__file__))


def cat(paths):
    out = bytearray()
    for p in paths:
        out += open(os.path.join(ROOT, p), "rb").read()
    return bytes(out)


def read_st(path):
    b = open(os.path.join(ROOT, path), "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    return b, json.loads(b[8:8 + n]), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="student_full.safetensors")
    a = ap.parse_args()

    man = json.load(open(os.path.join(ROOT,
        "teacher_shards/MANIFEST.json"), encoding="utf-8"))
    # 1) 还原 teacher 字节
    parts = [p["path"] for p in man["subshards"]]
    teacher = cat(parts)
    assert len(teacher) == man["total_bytes"], "teacher size mismatch"
    assert hashlib.sha256(teacher).hexdigest() == man["teacher_sha256"], \
        "teacher sha256 mismatch"
    tn = struct.unpack("<Q", teacher[:8])[0]
    thdr = json.loads(teacher[8:8 + tn])
    tds = 8 + tn

    # 2) NEW 字节
    wdir = os.path.join(ROOT, "weights_v10")
    if os.path.isdir(wdir):
        branch = cat(sorted(
            "weights_v10/" + p for p in os.listdir(wdir)
            if p.startswith("student_branch_v10.part")))
    else:
        branch = open(os.path.join(ROOT,
            "student_branch_v10.safetensors"), "rb").read()
    bb, bhdr, bds = None, None, None
    # branch 本身是 safetensors：重开解析
    bn = struct.unpack("<Q", branch[:8])[0]
    bhdr = json.loads(branch[8:8 + bn]); bds = 8 + bn
    sb, shdr, sds = read_st("student_smm_v10_3.safetensors")

    # 3) 按 manifest 顺序写出完整 student
    sm = json.load(open(os.path.join(ROOT,
        "student_scoped_manifest_v10_3.json"), encoding="utf-8"))
    hdr, off, blobs, checked = {}, 0, [], 0
    for e in sm["entries"]:
        name = e["name"]
        if e["op"] == "COPY":
            a0, a1 = thdr[name]["data_offsets"]
            raw = teacher[tds + a0: tds + a1]
            dt = thdr[name]["dtype"]; shape = thdr[name]["shape"]
        elif ".smm." in name:
            a0, a1 = shdr[name]["data_offsets"]
            raw = sb[sds + a0: sds + a1]
            dt = shdr[name]["dtype"]; shape = shdr[name]["shape"]
        else:
            a0, a1 = bhdr[name]["data_offsets"]
            raw = branch[bds + a0: bds + a1]
            dt = bhdr[name]["dtype"]; shape = bhdr[name]["shape"]
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            raise RuntimeError(f"sha256 mismatch: {name}")
        hdr[name] = dict(dtype=dt, shape=list(shape),
                         data_offsets=[off, off + len(raw)])
        blobs.append(raw); off += len(raw); checked += 1
    hdr["__metadata__"] = dict(format="pt", ts2sc="v10.3",
        architecture="SGC_V10_FROZEN_3SIDECAR_PLUS_SMM",
        context_tokens=10485760, n_tensors=checked)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs:
            f.write(b)
    print(json.dumps(dict(out=a.out, n_tensors=checked,
                          all_sha256_verified=True,
                          file_bytes=os.path.getsize(a.out),
                          teacher_bytes=len(teacher)), indent=2))


if __name__ == "__main__":
    main()
