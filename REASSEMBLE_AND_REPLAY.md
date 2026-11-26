# 重组与重放

```bash
cd weights && python reassemble.py        # 32 MiB 分片 -> 216.6 MiB safetensors，逐片 sha256 校验
cd .. && python replay_full.py --execute  # 目标机：读真实 teacher 分片，流式写出完整 2.6 GB student
```

`replay_full.py` 对 219 个 COPY 张量逐个做 shape/dtype/逐位校验，
对 72 个 DERIVE 复算 GQA 行映射（禁止零填充），对 120 个 NEW 写出 24 层 SGC 分支。
任何一项不匹配即中止，不产出半成品。
