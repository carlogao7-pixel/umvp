# 视频嵌入模型 · 选型调研（WeMM 候选，2026-10）

> 说明（2026-10-26 归档）：本篇为视频回查检索（video_review_search）与 WeMM 集成
> （wemm_integration）两条下期设计稿的**选型参考调研**，内容经 2026-10-26 联网核实
> （原记录 INB-0088）；实现现状以 `doc/design/` 两设计稿为准。

## 结论速览

**推荐关注：Qwen3-VL-Embedding**（Qwen 官方 2026-01-08 发布，Qwen3-VL 基底，
arXiv 2601.04720）——列为 WeMM 的**并行候选/替代位**，联调清单的 vLLM 部分可复用。

## Qwen3-VL-Embedding 关键指标

| 维度 | 数据 |
|---|---|
| 榜单 | MMEB-V2 全 78 任务：8B=77.9 / 2B=73.4 **居首** |
| 对比 | GME-2B 55.3 / GME-7B 59.1 / VLM2Vec-V2 59.2 / Seed-1.6-embedding 76.9 / RzenEmbed-8B 72.9 |
| 输入 | 文本/图像/截图/视频及混合模态（**原生视频输入**） |
| 语言 | 30+ |
| 维度 | MRL 截维 64~4096（可对齐本项目 clip=1024 / frame=256 两级索引档位） |
| 上下文 | 32k |
| 部署接口 | vLLM(runner=pooling) / SGLang(is_embedding) / sentence-transformers——与 wemm_integration.md §3 WeMM 服务形态完全同款 |
| 配套 | Qwen3-VL-Reranker 组 embedding 召回 + rerank 两段式（对应三级检索漏斗） |

## 其他候选

- **VLM2Vec-V2**（TIGER-AI-Lab，Qwen2-VL 2B 基底）
- **GME**（Alibaba-NLP，Qwen2-VL 2B/7B，vLLM 原生支持）
- **UME-R1**（推理驱动生成式嵌入 + RL；MMEB-V2 强但生成式慢、生态新）
- **WeMM-Embedding**（腾讯 Qwen3.5 基底，现役设计稿选型）

## 托管 API（参考，本项目基本出局）

Twelve Labs Marengo 3.0（2025-10 上架 AWS Bedrock，视频原生多模态嵌入、多向量、含音频）、
Amazon Nova Multimodal Embeddings——**警用内网场景公网托管 API 基本出局，自部署为主**。

## 后续动作

wemm_integration.md / video_review_search.md 下期更新时：将 Qwen3-VL-Embedding
纳入候选对比表，vLLM 联调清单复用。
