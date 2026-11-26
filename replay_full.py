#!/usr/bin/env python3
# Target-machine replay: produce the FULL 2.6 GB student.
# 1) cat model_part_* > teacher.safetensors
# 2) python replay_full.py teacher.safetensors OUT_DIR
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors_v8 import SafeTensors, write_safetensors
led = json.load(open(os.path.join(os.path.dirname(__file__), "student_full_manifest.json")))
teacher, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
with SafeTensors(teacher) as s:
    tensors = {}
    for e in led["entries"]:
        if e["op"] == "COPY":
            b = s.tensor(e["src"])
            assert len(b) == e["bytes"] and __import__("hashlib").sha256(b).hexdigest() == e["sha256"]
            tensors[e["dst"]] = (e["dtype"], e["shape"], b)
        elif e["op"] == "DERIVE" and e["transform"] == "GQA_ROW_REMAP":
            raw = s.tensor(e["src"]); rowlen = e["in_shape"][1] * 2
            # slot_map is reconstructed from the real config
            cfg = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
            kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]; R = cfg["hidden_size"] // kv_dim
            rows = [i % kv_dim for i in range(cfg["hidden_size"])]
            tensors[e["dst"]] = (e["dtype"], e["shape"],
                                 b"".join(raw[j*rowlen:(j+1)*rowlen] for j in rows))
        elif e["op"] == "DERIVE" and e["transform"] == "ALIAS_TEACHER_q_proj":
            tensors[e["dst"]] = (e["dtype"], e["shape"], s.tensor(e["src"]))
        elif e["op"] == "NEW":
            n = 1
            for x in e["shape"]: n *= x
            tensors[e["dst"]] = ("F32", e["shape"], b"\x00" * (n * 4))
    w = write_safetensors(os.path.join(out, "student.safetensors"), tensors,
                          metadata=dict(format="pt", ts2sc="v8.3", mode="FUSION_FULL"))
# reopen and re-verify EVERY COPY byte
with SafeTensors(os.path.join(out, "student.safetensors")) as o, SafeTensors(teacher) as s:
    bad = [e["src"] for e in led["entries"] if e["op"] == "COPY"
           and o.tensor(e["dst"]) != s.tensor(e["src"])]
    print("COPY byte mismatches:", len(bad))
    assert not bad
print("FULL STUDENT READY:", w["n_tensors"], "tensors")
