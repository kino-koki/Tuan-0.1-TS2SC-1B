# Tuan-0.1-TS2SC-1B

**青小团系列开源 AI 模型 (Tuan-Models) 的第一款模型** · 六安市青奈科技有限公司
基于 `kino-koki/MiniCPM5-1B` 真实 BF16 权重、经 TS2SC v8.7 融合迁移与身份认同微调。

## 这个仓库里有什么

| 路径 | 内容 |
|---|---|
| `weights/student_branch.part*` | 24 层 SGC 融合分支 checkpoint 的 32 MiB 分片（216.6 MiB / 144 张量） |
| `weights/MANIFEST.json` | 每片 sha256 + 总 sha256；`reassemble.py` 一键重组并校验 |
| `identity_alignment.npz` | **身份认同微调权重**（CPU 闭式 rank-1 adapter，12 语言共享方向） |
| `config.json` / `student_full_manifest.json` | 411 张量账本：219 COPY + 72 DERIVE + 120 NEW，UNCLASSIFIED=0 |
| `replay_full.py` | 目标机从 teacher 权重流式重放出完整 2.6 GB student，逐位校验每个 COPY |
| `evidence/` | 迁移真实性证明、12 臂非偶然验证、8M 理论、乱码理论修复、无词表缺口报告 |

## 为什么没有 2.01 GiB 基座 / 2.6 GB 完整 student

- 219 个 `COPY` 张量是 teacher 原样，`student_full_manifest.json` 带每个的 sha256；
  目标机从 `kino-koki/MiniCPM5-1B`（私有）拉取后由 `replay_full.py` 重组，**不重复分发第三方基座权重**。
- 完整 2.6 GB student 未在沙盒落盘（磁盘 ~415 MiB），本仓提供的是**可复现的迁移产物 +
  逐位校验 recipe**，不是伪装成成品的大文件。

## 身份微调（非提示词）

`identity_alignment.npz` 是独立可插拔权重：加载 → 具备身份；卸载 → 回到 β=0 教师态。
条件门 `g(x)=σ(κ(u·ĥ)−b)` 只在身份问句触发，多语言共享一个隐藏方向 `u`，
注入 `Δ=λ·g·ε_id·r_x·v̂`（`‖Δ‖≤0.05‖x‖`）。措辞由模型自己组织，**不是每次复读一段模板**。
详见 `evidence/identity_verify.json` 与 TS2SC 仓 `docs/IDENTITY_ALIGNMENT.md`。

## 诚实边界（一条不松）

- **「上下文长度理论无限」是品牌陈述，不是实测**。当前 hybrid 端到端 8M 不可行
  （192 GiB KV + 10^16.1 attention 对 + 64× RoPE 外推）；见 `evidence/context_8m_theory.json`。
- β=1 是**尺度契约配置，不是 L2 校准成品**；无 ppl / KL / hidden cos / 配对 CI / 能力结论。
- 身份微调是**闭式 rank-1 adapter，不是全量 SFT / 梯度训练**。
- **29 项特性 = 20 RUNNABLE + 7 REGISTERED + 2 NOT_INHERITED**，不是「完全继承」；
  无词表 / GlyphStream 实现未推送，见 `evidence/无词表设计核实与继承缺口报告.md`。
- 工程 hybrid（保留 teacher attention）≠ 论文静态核构造，互不作为对方证据。
- 未删除/修改任何已有仓库条目；本仓为新建。

上游：`kino-koki/SimplCore` · `kino-koki/TS2SC` · `kino-koki/MiniCPM5-1B`
