# WeMM 嵌入对接（设计稿）

> 状态：📐 客户端未实施；**服务端已上线并联调通过（2026-10-28，实测记录见 §6）**
> 定位：WeMM-Embedding（wemm）服务对接**专篇**——模型机制、服务端部署、客户端模块
> （`WEMMClient`）、成本与联调清单。**业务流程与链路设计不在本文**：
> 索引链路（PIPE0004）、三级检索、两级索引精定位、存储与前端见
> [《历史视频回顾检索》](video_review_search.md)。

## 1. 模型是什么

- `Tencent/WeMM-Embedding`：微信视觉团队开源的通用多模态嵌入模型家族（2B/4B/9B，
  基于 Qwen3.5），文/图/视频/视觉文档/交错输入统一到**同一个向量空间**
  （暂不支持音频）。MMEB-v2/v3 同规模领先（2B 超此前 8B 开源模型）。
- 跨模态同空间 = "文搜视频"的原理：文本查询与视频片段各生成一个向量，
  余弦相似度直接可比，无需关键词命中。

## 2. 向量化机制（讲清"过程"）

### 2.1 向量怎么来的

```
输入（若干帧图 / 一段文字 / 文+图交错）
  → 按聊天模板拼装成消息
  → 模型内部切成 token 逐层加工      ← token 是中间步骤（读入单位），对接方不感知
  → 取专用 <embedding> token 位置的末层隐状态
  → L2 归一化 → 输出 1 个向量        ← 这才是"向量化结果"
```

- 一条消息（不论 1 帧还是 6 帧）只产出**一个**向量，代表整条消息内容。
- 视频 = 均匀抽帧（评测用 64 帧）作多帧输入；多图交错输入原生支持。

### 2.2 维度与"套娃"截维（Matryoshka）

| 模型 | 可截维度档位 | 全维（最高档） |
|---|---|---|
| 2B | 64 / 128 / 256 / 512 / 1024 / 2048 | 2048 |
| 4B | 64 … 1024 / 2560 | 2560 |
| 9B | 64 … 2048 / 4096 | 4096 |

- 截维 = 取完整向量前 N 维后**重新 L2 归一化**（推理后处理）；2B@256 维保留
  完整维度 98.7% 性能（MMEB-v2）。
- 约束：只能在档位内取值（客户端校验，非法报错）；**入库与查询必须同维度**；
  换维度 = 重建索引（新版本追加，不覆盖旧库）。
- 截维**不影响服务端显存**（推理恒按全维跑，截维是事后 CPU 操作），只影响
  本机存储与检索计算量（MB 级 / 毫秒级）。

## 3. 服务端部署（在部署机执行）

| 框架 | 验证版本 | 启动要点 |
|---|---|---|
| vLLM | 0.27.0 | `vllm serve <模型路径> --runner pooling --chat-template <模型路径>/embedding_chat_template.jinja` |
| SGLang | 0.5.9 | 先跑仓库 `scripts/patch_sglang_video.py`（视频预处理对齐），再 `python -m sglang.launch_server --model-path <路径> --is-embedding --enable-precise-embedding-interpolation` |

两者均暴露 OpenAI 兼容 `/v1/embeddings`（默认端口 8000）。

**启动清单（联调前置）：**

1. 确认型号与显存：2B bf16 权重约 4–5GB（建议 ≥8GB 显存卡）；4B/9B 相应放大。
2. 按上表启动；`curl http://<部署机IP>:8000/v1/models` 见模型名即通。
3. 冒烟：向 `/v1/embeddings` 发一条纯文本 → 返回向量且维度 = 该型号全维（2B 为 2048）。
4. 防火墙放行 UMVP 机 → 部署机 8000 端口。

> 说明：本项目默认**逐帧多图**形态（§4.3），不依赖 SGLang 的视频 patch；
> patch 仅在将来改用"真 video 内容项"（模型端时序采样）时需要。

## 4. 客户端对接（`umvp/video_embed/wemm_client.py`）

### 4.1 WEMMClient

`WEMMClient(endpoint, model, dimension, timeout)`：
`embed_frames(frames) -> vec` / `embed_text(text) -> vec` / `embed_image(img) -> vec`。

- 先 TCP 快速探测端点（复用 `_endpoint_reachable` 先例），不可达立即报错
  "嵌入端点不可达: <地址>"；请求 `timeout=(5, N)`（连接 5s + 读取 embed_timeout）。
- 截维 + 重新 L2 在客户端做；档位校验（不在档位直接报错，报错信息列出合法档位）。

### 4.2 请求 schema（2026-10-28 联调修正，实测为准）

```
POST {endpoint}                          # 形如 http://<ip>:8000/v1/embeddings
# 纯文本（可批量，input 数组 → n 向量顺序对位）：
{"model": <模型名>, "input": <str | [str, ...]>}
# 多模态（图文/多图/视频）：走 messages 透传，不用 input；
# 一条请求只放一组 messages，批量靠客户端并发（1–4）：
{"model": <模型名>,
 "messages": [{"role": "user", "content": [ ...内容项... ]}]}
```

- 内容项同 OpenAI 多模态形态：
  `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`（帧，
  客户端先按 max_resolution 缩放 + jpeg_quality 编码）、`{"type": "text", "text": ...}`；
- **一条消息一个向量**：6 帧合在一条消息 = clip 向量；每帧单独成消息 = frame 向量；
- 服务端 MRL 截维：请求带 `"dimensions": <N>`（⚠️ 与客户端前 N 维截断**不等价**，
  二者只能二选一全链路统一，见 §6 实测）。
- 响应取 `data[i].embedding`，按 input 顺序对位。

### 4.3 形态选择：逐帧多图，而非 video 内容项

- 默认把片段的 N 帧作为**多图消息**发送：vLLM/SGLang 通吃、免视频补丁、免切临时
  视频文件；WeMM 原生支持交错多图输入。
- 「真 video 内容项」（模型端 64 帧时序采样）留作升级位——质量上限更高，
  但需 SGLang patch + 视频文件传输，联调后再评估。

### 4.4 mock 档

`use_real_embed=false`：客户端返回**确定性伪向量**（同输入同向量、按 dimension 截形、
L2 归一）——服务未启动也能开发验证全流程，复刻 `use_real_vlm` 先例。

### 4.5 对接相关参数（随 video_embed 模块 preset 落库）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| embed_endpoint | str | 待填（§6 联调后定） | OpenAI 兼容 `/v1/embeddings` 地址 |
| embed_model | str | 待填 | `/v1/models` 返回的模型名 |
| embed_dimension | int | 1024 | clip 向量维度（Matryoshka 档位内） |
| frame_dimension | int | 256 | frame 向量维度（精排只比相对远近，低维足够） |
| use_real_embed | bool | false | true=真发 HTTP；false=确定性伪向量 |
| embed_timeout | int | 20 | 读取超时（秒） |
| max_resolution | str | "896,896" | 送审前缩放上限（嵌入不需原分辨率，压带宽） |
| jpeg_quality | int | 85 | 帧 JPEG 质量 |

链路侧参数（窗口/帧数/底库路径/缩略图/查询 top_k 等）见[《历史视频回顾检索》](video_review_search.md)§5。

## 5. 成本与资源

- **远端**：每片段 1 次批量请求（clip 多图 + 逐帧单图，约 7 个输入，6 帧 JPEG ≈
  0.5–1MB base64）；30 分钟视频 ≈ 360 片段 = 360 次请求，分钟级完成整片索引。
  显存由型号 + 并发决定，与截维维度无关。
- **本机**：零 GPU、零新依赖（requests 已在用）；检索为 numpy 线性（万级片段 <10ms）。
- 跨公网部署：建议 `max_resolution` 降到 640 档 + 减少每片段帧数。

## 6. 联调实测（2026-10-28，服务上线后 curl/脚本实测，12 项冒烟 11 过）

**端点与鉴权**（凭据在 `wemm接口/REMOTE_CLIENT.env`，**不入库**，已进 .gitignore）：
Base `http://ca0.tail952a82.ts.net:8000`（Tailscale，备用 `http://100.96.235.67:8000`），
`Authorization: Bearer <key>`；host 侧 wemm.sh start/status 管理，对端不部署。

| 清单项 | 实测结果 |
|---|---|
| ① 型号与全维 | **WeMM-Embedding-2B**（vLLM），全维 **2048**（与本文假设一致）；`max_model_len=8192` |
| ② 框架与端口 | vLLM，8000 端口，域名与备用 IP 双通 |
| ③ 多模态字段拼写 | **修正 §4.2**：多模态**不用 `input`，用 `messages`**（OpenAI chat 风格透传，content 数组 = text/image_url/video_url，图用 data URL） |
| ④ input 批量行为 | **纯文本** `input` 数组批量可用（n 条 → n 向量顺序对位）；**多模态一条请求只放一组 `messages`**，批量靠客户端并发（1–4，视频 1–2） |
| ⑤ 端点与鉴权 | 见上；免 HTTPS（tailnet 内） |

**附加实测结论（客户端设计必须遵守）：**
- **服务端 MRL 截维 ≠ 客户端前 N 维截断**：`dimensions=256` 与本地「前 256 维+重归一」余弦仅 ≈0.50——**截维方式全链路必须统一**（建议统一走服务端 `dimensions`，入库/查询同参），混用=检索错乱；
- 输出**已 L2 归一**（点积=余弦）；截维后仍归一 ✓；
- 跨模态语义方向正确（人脸文本查询 vs 人脸图 dot=0.245 > vs 猫查询 0.125；猫狗文本对 0.92——同域文本相似度普遍偏高，阈值调优按相对排序不按绝对值）；
- 性能限制（REMOTE_CLIENT.md §6）：上下文 8192；视频 fps=2/max_frames=768；视觉像素预算 ≈1.0MP；timeout ≥120s（视频 ≥180s）；
- 建议：帧图长边 ≤1280（与 §4.2 的 max_resolution=896 档兼容）。

> 实现注意事项：多模态走 `messages` 时 OpenAI SDK 需 `extra_body` 透传（`input` 留空串占位）；
> 纯文本可直接 `input`。

## 7. 验证方法（客户端级，断言式）

1. mock：伪向量确定性（同输入同向量）、维度 = 配置值、L2 模长 ≈ 1。
2. 冒烟（真端点）：文本 → 向量维度 = 型号全维；多图消息 → 恰 1 个向量；
   input 数组 n 条消息 → n 个向量且顺序对位。
3. 截维：2048 → 1024 / 256 后模长 ≈ 1；非法维度报错并列出合法档位。
4. 降级：端点不可达时快速失败，错误信息含端点地址（≤ 连接超时 5s）。
5. 链路级验证（入库/检索/精定位/回归）见[《历史视频回顾检索》](video_review_search.md)§9。
