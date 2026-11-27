# Tuan-0.1-TS2SC-1B

**青小团系列开源 AI 模型 (Tuan-Models) 的第一款模型** · 六安市青奈科技有限公司
基于 `kino-koki/MiniCPM5-1B` 真实 BF16 权重、经 TS2SC v9.x 融合迁移与 12 轮身份认同微调。

> 身份微调的完整结论见 [`IDENTITY_FINAL_VERDICT.md`](IDENTITY_FINAL_VERDICT.md)：
> **12 轮里没有任何一版在零解码期干预下连贯说出整段自我介绍**；
> 已跨过的是「权重级、非提示词、非身份逐位零回归、零编排的训练方法」，
> 未跨过的是「让 24 层内部状态相信新身份」。后者需要 LoRA SFT，本仓给出
> 可执行骨架但**未执行**（CPU-only 环境无 torch/CUDA）。

## 开源许可

**Apache License 2.0** —— 见 [`LICENSE`](LICENSE)（与 `kino-koki/TS2SC` 同字节的官方全文）
及 [`NOTICE`](NOTICE)。本仓库**公开**。219 个逐位 COPY 的 teacher 张量不在此分发，
由 `replay_full.py` 从上游权重重建并逐位校验；使用前请另阅上游模型自身许可。

## 这个仓库里有什么

| 路径 | 内容 |
|---|---|
| `weights/student_branch.part*` | 24 层 SGC 融合分支 checkpoint 的 32 MiB 分片（216.6 MiB / 144 张量） |
| `weights/MANIFEST.json` | 每片 sha256 + 总 sha256；`reassemble.py` 一键重组并校验 |
| `identity_alignment.npz` | **身份认同微调权重**（CPU 闭式 rank-1 输出层 adapter，12 语言共享门控方向 + 词级锚） |
| `identity_sft.npz` | **闭式 ridge 输出层微调权重** `A=(HᵀH+λI)⁻¹HᵀT`，24576 参数 = 1B 的 0.0025%，由 32 个真实 hidden 解出 |
| `identity_full_sft.npz` | ⚠️ **EXPERIMENTAL / NOT ACTIVE**：v9.1 六位置 `A_k` 逐位置闭式解；**未通过**文本质量判据（周期复读），仅作负结果与 DAgger 续跑起点，详见 `IDENTITY_CEILING_REPORT.md` |
| `identity_full_hidden.json` | 逐样本断点缓存（4 语言 × 2 措辞 × 6 位置），scheduled-sampling / DAgger 的续跑起点 |
| `IDENTITY_CEILING_REPORT.md` | v9.1 a–d 四轮负结果、度量漏洞修复、输出层引导天花板（**K≈2**） |
| `identity_iteration_comparison.json` | a/b/c 三轮旧判据与原文本对照，不改写不抹平 |
| `identity_dagger_sft.npz` / `identity_dagger_r*.json` | v9.2 DAgger 2 轮 off-policy 轨迹与重解权重（EXPERIMENTAL） |
| `identity_head_delta.npz` | **v9.3 真梯度权重**：标准 next-token CE + Adam 训出的 302×1536 输出层残差；推理期零编排。方法成立但生成结果未达标，EXPERIMENTAL |
| `lora_identity_sft.py` | 真正跨过天花板的 LoRA SFT 骨架（**未执行**，需 torch/PEFT/CUDA） |
| `IDENTITY_FINAL_VERDICT.md` | **12 轮全表、两难证明、已跨过/未跨过、GPU 可执行路径** |
| `identity_all_rounds_summary.json` / `identity_semantic_reaudit.json` | 13 份验收汇总 / 关键词与语义分离重判 |
| `identity_verify.json` / `identity_margin_check.json` / `identity_sft_verify.json` / `identity_full_sft_verify.json` / `identity_dagger_verify.json` / `identity_gradient_verify.json` | 六套主验收，全部保留原判据不改写 |
| `IDENTITY_SFT.md` | 逐位置 teacher-forced 全句微调协议 + 12 语言多措辞身份问答（**未执行**） |

> **当前对外生效的身份权重仍是 `identity_alignment.npz`（v9.0，2 步词级锚）。**
> 其余 `identity_*sft.npz` / `identity_head_delta.npz` 均为 EXPERIMENTAL / NOT ACTIVE，
> 不要直接加载当成品用。
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
条件门 `g(x)=σ(κ(u·ĥ)−b)` 只在身份问句触发（12 语言共享一个隐藏方向 `u`，
实测 `g_id ≥ 0.6221` / `g_non ≤ 0.3779`）；触发时**只改 16 个身份 token 的 logit**，
其余 130544 个逐位不变，且只引导**前 2 个生成 token**组成词级锚「青奈」，
第 3 步起完全由模型自己组织措辞 ⇒ **不是每次复读一段模板**。
详见 `identity_verify.json` / `identity_margin_check.json` 与 TS2SC 仓 `docs/IDENTITY_ALIGNMENT.md`。

迭代记录（都写在报告里，没抹平）：v8.7 把 Δ 注入 final RMSNorm 之前被归一化搅乱；
v8.8 只引导首 token ⇒ 身份关键词命中 0；v8.9 改为输出层 + 词级锚，但每步重算门控
导致第 2 个 token 被提前释放，只说出「青」；**v9.0 把门控只在首步判定一次并锁定**，
词级锚「青奈」两步完整生效。

v9.0 六题实测（`identity_verify.json`，`passed=True`）：4/4 身份 probe 关键词命中、
4 条文本互不重复、12 语言门控全开（`g≥0.6221`）而 8 条非身份全关（`g≤0.3779`）、
logits 全 finite、0 越界 id。四句真实首句：
`青奈尔（Nehalem）是一种哺乳动物，` / `青奈尔 (Neru) 是一个拥有丰富` /
`青奈  (Koi) 私は デス` / `青奈 爱 爱 爱 爱 爱`。
**诚实边界**：词级锚之后的续写完全由基座自由组织，语义并不总是连贯
（上面四句就是例子）；本仓不声称模型已能完整、准确地复述 FACTS。
要稳定说出整段身份介绍需 `IDENTITY_SFT.md` 的逐位置 teacher-forced 微调。

此外 `identity_sft.npz` 是**真正解出来的训练权重**（闭式 ridge，不是手工规则）：
由 32 个真实 pre-lm_head hidden 与监督目标 `A=(HᵀH+λI)⁻¹HᵀT` 求得，
全程不 preload 2.01 GiB、峰值 <60 MiB（用户要求「注意不要 OOM」）。

**v9.1 是一次真做了但没通过的扩展**：`identity_full_sft.npz` 把引导扩到 6 步，
`A_k` 由 8 条样本（4 语言 × 2 措辞）的逐位置 teacher-forced hidden 闭式解出。
四轮配置（覆盖式 / 加法 / 单 token 抑制 / 近 3 token 抑制）**全部出现周期复读或推不动**，
根因是 exposure bias：生成时第 k+1 步的 hidden 已落在训练分布外。
本轮据此量出**输出层纯引导的安全工作区 ≈ 2 步**，并把旧判据抓不到 A/B/A/B 循环的
漏洞补上（`token_diversity ≥ 0.55`、`max_bigram_repeat ≤ 2`）。
详见 `IDENTITY_CEILING_REPORT.md`；`identity_full_sft_verify.json` 最终 `passed=false`。
它仍是输出层闭式 ridge，不是梯度反传。

**v9.2 / v9.3 又往前推了两步，也把「为什么推不动」钉死了：**

- **v9.2（DAgger）**：用模型自己生成的 off-policy hidden 做数据增广、配 expert
  token 重解 `A_k`，2 轮共 144 对训练样本。周期复读彻底解决
  （diversity 0.90–1.00、bigram 1），但 v9.2c 机判 `passed=true` 时
  **语义重判只有 1/4 有效** —— `青奈科技是开发MiniCPM系列模型的公司。
  我是一个MiniCPM`，关键词命中、语义完全颠倒。见
  `identity_semantic_reaudit.json`，不改写不抹平。
- **v9.3（真梯度）**：`identity_head_delta.npz` 由**标准 next-token CE + Adam**
  训练（受限对比目标，463872 参数 = 1B 的 0.046%），推理期只有一个学到的
  线性残差 + 冻结的学到的门控，**没有位置切换 / 锚集合 / margin 硬抬 /
  recent 抑制 / 负惩罚**。train full-vocab acc 0.931，但生成端
  先后撞上「全词表 CE 梯度稀释」「训练/推理 hidden 尺度不一致」「统一 Δ
  在所有位置复读同一批 token」三堵墙，最终 `passed=false`
  （bigram 12、run 8）。

**形式化结论**（`IDENTITY_FINAL_VERDICT.md`）：输出层无论闭式 ridge、DAgger
重解还是真梯度 CE，信息容量都不足以承载「多步连贯身份」。要么按生成位置
切策略（⇒ 不是纯权重），要么统一 Δ（⇒ exposure bias 复读）。
这是**参数所在层级**的问题，不是 lr / epoch / 候选集的调参问题。
真正跨过去需让梯度穿过 24 层：`lora_identity_sft.py` 给出
q_proj/v_proj LoRA r=16 + scheduled sampling + 通用能力回放的精确方案，
**本环境未执行、未验证**，不画饼。

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
