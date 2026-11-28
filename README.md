# MiniCPM5-1B-TS2SC · v10

**β>0 SimplCore 融合模型** · `kino-koki/MiniCPM5-1B` 干净权重全量重迁移 ·
同步 `kino-koki/SimplCore` @ `bec5ab8fd7` 最新 SGC 冻结外挂架构 ·
**Apache License 2.0 / public**

> **本轮是隔离重建，不是打补丁**：22 个分片全部重新从上游下载并逐片
> 验证 LIVE `git blob sha1`；219 个 COPY 的 sha256 从新字节重算；
> 312 个 NEW 参数确定性生成、零复用任何旧产物；身份微调整层摘除。

## 交付配置

| 项 | 值 |
|---|---|
| 交付默认 | [`inference_default_v10.json`](inference_default_v10.json)：**β = 1.0** |
| 架构 | `SGC_V10_FROZEN_3SIDECAR`：冻结静态核 + A 短卷积 + B CfC 液态门 + C Hadamard SE1 |
| 权重 | [`weights_v10/student_branch_v10.part*`](weights_v10/MANIFEST.json)（43.7 MiB / 312 张量 / 24 层） |
| 账本 | [`student_full_manifest_v10.json`](student_full_manifest_v10.json)：**219 COPY + 0 DERIVE + 312 NEW = 531**，UNCLASSIFIED=0 |
| 教师权重 | 219 COPY 张量不在此分发，`replay_full_v10.py` 从上游逐位重建并校验 |
| 身份 | **完全放弃身份微调**，零 adapter / 零门控 / 零锚 / 零负惩罚 |

> β=0 只是逐位零回归对照臂，**不是交付配置**。历史 `weights/`（v8 分支）
> 与 `identity_*` 文件按用户指令保留、未删除，但**不参与 v10 交付**、
> 不被任何路径加载。

## 为什么重做（严重污染排查）

见 [`SGC_V10_SYNC.md`](SGC_V10_SYNC.md) §1。逐项结论：

- 权重字节本身**没被身份微调改坏**（上一版 8 个 COPY 逐位一致、β=0 逐 token 相同）；
- 但仓库留着 33 个 `identity_*` 历史文件（用户要求不删，故保留标 RETIRED）；
- **真正的滞后是架构**：云端 SimplCore 已于 2026-11-27 推 v9–v11
  「SGC 冻结 + 三旁路外挂」补充包，TS2SC 仍停在 v8 单层静态核分支
  （216.6 MiB / 72 个 teacher k/v DERIVE 遗留耦合）。

因此本轮从上游干净权重**全量重迁移**：去掉 DERIVE 遗留，改为
`219 COPY + 312 NEW`，分支 43.7 MiB，架构同步到 v10。

## 上游「极大 SGC 外挂」同步（[`SGC_V10_SYNC.md`](SGC_V10_SYNC.md)）

- **主干完全冻结**：`ψ_m[τ]=ρ_m^τ·sin((τ+1)θ_m)`、静态 Toeplitz、
  FFT↔IIR 双形式、O(1) 定态，一个字节不改。
- **三条旁路并联，只在 readout 拼接，绝不回写主干**：
  A 3 层 k=7 门控短卷积 / B CfC 液态门（输入依赖 τ）/ C Hadamard SE1
  内容寻址绑定。
- 纯度：无 Q/K/V、无 `softmax(QKᵀ)`、无 KV cache、无位置表、无 `max_len`。
- v9 推翻了 v8 的「单层定义性边界」：多层堆叠后有效交互变输入依赖，
  而静态核一字未动 ⇒ 这就是「极大外挂」的含义。

TS2SC 侧实现见源码仓 `ts2sc/sgc_v10.py`，上游快照见
`ts2sc/upstream_simplcore_v11/`（13 文件带 LIVE sha256）。

## 实测不变量（[`v10_fresh_verify.json`](v10_fresh_verify.json)）

```
22/22 分片 LIVE git blob sha1 校验 OK，2.01 GiB
219 COPY / 8 个抽查逐位一致 · 312 NEW 重开逐字节复验 OK
FFT↔IIR 1.69e−14 · SE1 ρ=1 零串扰 0.0/0.0 · 衰减闭式 3.33e−16
短卷积流式==整段 8.33e−17 · P7 主干逐位不变 max|Δ|=0.0
纯度 AST PASS · β=0 Δ 严格零 · β=1 契约比 0.1000000
O(1) 状态 L=1/8/64 恒 46,656 B/层
生成 β=0 / β=1 全 finite · 全解码守卫 · 架构生效 2/2 · 身份标记泄漏 0
passed = true
```

## 文件

| 路径 | 内容 |
|---|---|
| `inference_default_v10.json` | **交付默认 β=1.0 / adapter=none / v10 架构** |
| `weights_v10/MANIFEST.json` | 分片 sha256 + 总 sha256 + `delivery_beta=1.0` |
| `student_full_manifest_v10.json` | 531 张量逐条 provenance |
| `replay_full_v10.py` | 目标机逐个 NEW sha256 复验 + COPY 逐位校验 |
| `upstream_pull_report.json` | 22 片逐片 LIVE blob sha1 + 小文件 sha256 |
| `v10_fresh_verify.json` / `V10_FRESH_MIGRATION_REPORT.md` | 本轮全部证据 |
| `SGC_V10_SYNC.md` | 污染排查 + 架构同步映射 + 边界 |
| `fresh_migrate_v10.py` / `sgc_v10.py` / `test_v10.py` | 可独立复跑的迁移 / 架构 / 15 条回归 |

## 边界（一条不松）

- 上游 v9–v11 结论是**否定性**的：`COMPLETE_BEYOND=False`；质量轴只到
  稳态 parity，2× 对手步数对照失败。**本轮只同步架构能力，不继承超越/SOTA 主张。**
- β=1 是尺度契约配置（T1'），**不是 L2 校准成品**；24 个 β_l 仍待校准。
- 上游语料 240 KB / D=32 / L≤1536；**1B / 中文 / LRA / 长文档完全未测**。
- 纯 numpy CPU 等价实现，未跑上游 PyTorch pytest；生成 probe 仅 4 token，
  只证 finite / 不乱码 / 架构生效 / 零身份标记，**不是质量结论**。
- 无 ppl / KL / 配对 CI / 能力 / SOTA / 「原生级」声明；
  「上下文长度理论无限」是品牌陈述，8M 端到端不可行结论不变。
- Qwen4-27B 仍未发布，本轮未访问其任何 config/权重/tokenizer。
- 完整 2.6 GB student 未在沙盒落盘（磁盘限制），本仓是可复现产物 +
  逐位校验 recipe，不是伪装成成品的大文件。
