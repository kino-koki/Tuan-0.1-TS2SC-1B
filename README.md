# MiniCPM5-1B-TS2SC

**β>0 SimplCore 融合模型** · 基于 `kino-koki/MiniCPM5-1B` 真实 BF16 权重 ·
经 TS2SC 完整迁移 · **Apache License 2.0 / public**

## 这一版是什么

| 项 | 值 |
|---|---|
| 交付配置 | [`inference_default.json`](inference_default.json)：**β = 1.0** |
| 架构 | 24 层 SGC 融合分支 ACTIVE，teacher 全部功能层逐位保留 |
| 身份 | **完全放弃身份微调** —— 基座原本认同，零注入 |
| 权重 | `weights/student_branch.part*`（216.6 MiB / 144 张量 / 24 层）+ 411 张量账本 |
| 教师权重 | 219 COPY 张量不在此分发，`replay_full.py` 从上游逐位重建并校验 |

> **β=0 只是逐位零回归对照臂，不是交付配置。**
> 本仓默认推理即 `beta=1.0, identity_adapter=none, logit_bias_fn=none`。

## 身份微调已整层退役

v9.0–v9.3c 共 12 轮 CPU 输出层身份实验**全部 RETIRED / NOT ACTIVE**。
结论见 [`IDENTITY_FINAL_VERDICT.md`](IDENTITY_FINAL_VERDICT.md)：
输出层（闭式 ridge / DAgger / 真 CE+Adam）的信息容量不足以承载多步连贯身份，
在纯 CPU 环境下几乎只带来偏离、没有收益。

按用户要求：**不清理、不删除任何条目**，历史文件原地留存，
但交付路径一个都不加载。真正的 GPU LoRA SFT 由维护者在目标机自行执行，
本仓不做、不声称。

## 无知识污染（[`CLEAN_STATE_VERDICT.md`](CLEAN_STATE_VERDICT.md)）

- **E1 权重**：411 张量 = 219 COPY / 72 DERIVE / 120 NEW，UNCLASSIFIED=0；
  抽查 8 个 COPY（embed / lm_head / norm / q,k,o / down_proj / input_ln）
  原始 BF16 字节 sha256 与账本**全部逐位一致** ⇒ teacher 权重零改动。
- **E2 隔离**：38 个身份资产逐个 sha256 登记 QUARANTINED，
  干净态推理路径不加载任何一个。
- **E3 生成**：纯 teacher 8 题 + β=0 双跑 + **β=1 主交付臂 8 题**
  + 32-token 深度身份 probe；β=0 与 teacher **逐 token 相同**，
  全部输出零身份标记，基座自我认同回归。
- **E4 反证**：临时挂一次 v9.0 adapter 同问句 ⇒ 输出改变并出现身份词；
  干净态没有 ⇒ 污染只可能来自这些可选 adapter，且已被排除。

口径诚实：**不是** ppl / 下游 benchmark / 大规模知识集配对 CI；
是「权重零改动 + β=0 逐 token 相同 + 零身份标记 + 原本认同回归」。
CPU 上做输出层微调无法让 24 层内部状态相信新身份 —— 与用户判断一致。

## 文件

| 路径 | 内容 |
|---|---|
| `inference_default.json` | **交付默认 β=1.0 / adapter=none** |
| `weights/MANIFEST.json` / `weights/reassemble.py` | 分片 sha256 + 一键重组 |
| `config.json` / `student_full_manifest.json` | 411 张量账本 |
| `replay_full.py` | 目标机流式重放完整 2.6 GB student，逐位校验每个 COPY |
| `clean_state_verify.json` / `CLEAN_STATE_VERDICT.md` | 本轮四条证据与判据 |
| `verify_clean_state.py` | 可独立复跑的干净态验证脚本 |
| `IDENTITY_FINAL_VERDICT.md` | 12 轮身份实验最终定案（历史，RETIRED） |
| `identity_*` | **RETIRED / NOT ACTIVE**，历史留存，不参与交付 |

## 边界（一条不松）

- β=1 是**尺度契约配置**，不是 L2 校准成品；β_l 仍待校准。
- 无 ppl / KL / 配对 CI / 能力 / SOTA / 「原生级」声明。
- 「上下文长度理论无限」是品牌陈述；8M 端到端不可行的结论不变。
- 完整 2.6 GB student 未在沙盒落盘（磁盘限制），本仓是可复现产物 +
  逐位校验 recipe，不是伪装成成品的大文件。
- Qwen4-27B 仍未发布，本轮未访问其任何 config/权重/tokenizer。
