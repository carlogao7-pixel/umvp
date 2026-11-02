# 视频嵌入 VideoEmbedStage

> 模块组：向量旁路（视频帧 → 片段/帧向量 → 检索底库）｜ 代码：`umvp/video_embed/`
> （链路阶段 `video_embed_stage.py`；嵌入客户端 `wemm_client.py`；向量底库 `embed_store.py`；
> 标签对齐纯逻辑 `label_align.py`——供 web 侧落表复用）
> 数据库参数：`preset_params_video_embed`（前缀 **VEMB**，`db.py _ID_PREFIXES` 登记）
> 状态：**PIPE0004「视频向量化-嵌入查询」核心阶段**——离线索引链路，**入库即目的**
> （与 PIPE0003「截取落盘即目的」同型，无告警语义）。

## 1. 模块定位

把视频按片段窗口（默认 6 帧/窗）切成片段，每满窗做两件事：

1. **嵌入入库**：窗口内各帧经远端 WeMM-Embedding 服务出 **clip 向量**（多帧合一条消息、
   一条消息一个向量）与 **frame 向量**（每帧单独一条消息），写入 `EmbedStore` 向量底库
   （numpy 线性检索形态，复刻 FaceStore 先例）——"文搜视频"的索引层；
2. **观察标签聚合**（可选）：第二入端口 `dets` 吃上游任意识别记录流（YOLO 检测 / 车牌 /
   人脸），满窗时聚合成窗口级观察层标签（`observation` 旁路键），供 web 侧在索引收尾
   对齐落 `review_index` 表（混合检索标签体系）。

每片段产出**一条 dict 记录**（`kind="clip"`），附带 `clip_indexed` 事件——链路尾事件汇总
即 PIPE0004 的输出。三个协件分工：

| 协件 | 代码 | 职责 |
|---|---|---|
| `VideoEmbedStage` | `video_embed_stage.py` | 链路阶段本体：聚帧切窗 / 喂观察 / 出片段记录 / 缩略图落盘 |
| `WEMMClient` | `wemm_client.py` | WeMM 嵌入客户端：mock/real 双档、维度档位校验、探测/重试/鉴权（key 只走 env） |
| `EmbedStore` | `embed_store.py` | 向量底库：upsert / 两级检索（clip 粗排 + frame 精排）/ 按视频删除 / npz+json 持久化 |

机制与服务端部署见 [WeMM 嵌入对接](../design/wemm_integration.md)（§4 客户端 / §6 实测）；
业务与查询侧见 [历史视频回顾检索](../design/video_review_search.md)；标签体系与混合检索定案见
[混合检索设计稿](../design/hybrid_retrieval.md)；链路编排见
[PIPE0004 专篇](../chains/pipe0004_video_embed.md)。本文只写模块本体（参数 / 端口 / 产出形态）。

## 2. 参数

（server `PARAMS["video_embed"]` 实键；嵌入侧对齐 wemm_integration.md §4.5 参数表）

### 2.1 嵌入端点（真档）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| embed_endpoint | str | 空 | OpenAI 兼容 `/v1/embeddings` 地址（base 地址自动拼路径）；env `WEMM_BASE_URL`（别名 `WEMM_ENDPOINT`）优先；留空且无 env = mock 档 |
| embed_model | str | 空 | 服务端 `/v1/models` 返回的模型名（env `WEMM_MODEL` 优先，默认 `wemm-embedding`） |
| embed_dimension | int | 1024 | clip 向量维度（Matryoshka 档位内；非法档位构造即报错并列出合法清单） |
| frame_dimension | int | 256 | frame 向量维度（精排只比相对远近，低维足够） |
| use_real_embed | bool | false | true=真发 HTTP（无端点直接报错）；false=确定性伪向量（mock 档） |
| embed_timeout | int | 120 | 读取超时秒（含图请求慢；连接阶段恒 5s 快速失败） |
| max_resolution | str | "896,896" | 送审帧缩放上限（等比、只缩不放，压带宽；跨公网可降 640,640） |
| jpeg_quality | int | 85 | 帧 JPEG 编码质量 |

> **key 只走环境变量 `WEMM_API_KEY`**，不入 preset / DB / 日志 / 异常信息。

### 2.2 片段窗口 / 底库 / 缩略图

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| frames_per_clip | int | 6 | **窗口切分唯一依据**（每片段帧数；窗口实际时间由帧 ts 承载，随帧管理节奏走） |
| clip_seconds | float | 5.0 | 标称窗口秒数（仅写片段元数据供展示/排查，**不影响切分**） |
| thumb_dir | str | "out/video_thumbs" | 窗口首帧缩略图落盘目录（空=不落盘）；相对路径归一项目根 |
| thumb_long | int | 320 | 缩略图长边像素（等比、只缩不放） |
| flush_partial | bool | true | 收尾时不足一窗的余帧是否出 partial 片段（false=丢弃） |
| store_path | str | "out/video_vecs.npz" | 向量底库路径（npz + 同名 json）；同视频重跑按 `(video, t_start)` 幂等覆盖 |
| video | str | （test_only） | 模块试跑页选视频用（不落库、不进链路） |
| frames_embed | bool | true | **代码参数**（构造参数，未进 PARAMS）：false 时只嵌 clip 向量——省掉每片段 N 次 frame 请求（索引成本大头），查询仍可用（定位到窗口），播放精定位退化到窗口头 |

### 2.3 维度档位与「截维全链路统一走服务端」约束

- Matryoshka 档位（2B 型号）：`64 / 128 / 256 / 512 / 1024 / 2048`；**入库与查询必须同维度**，
  换维度 = 新建库不混写（`EmbedStore` 库级维度锁定，不匹配向量拒收并指明两侧维度；
  web 装配期另有一道维度护栏，见 §5）；
- **截维统一走服务端 `dimensions` 请求参数**——与客户端"前 N 维截断"**不等价**
  （实测余弦仅 ≈0.50），混用 = 检索错乱（实测记录见 wemm_integration.md §6）；
- 装配/校验双保险：`web_lab/validate_chain.py` 对 spec 即校验维度档位与
  `use_real_embed=true` 未配端点的组合（校验期报错，不拖到运行期）。

### 2.4 mock / real 双档

- **mock 档**（`use_real_embed=false` 或无端点）：确定性伪向量——同输入恒同向量、按档位
  前缀截形再归一（模拟真实 MRL 行为），零网络零模型，服务不在线也能全流程开发测试；
- **real 档**：端点/模型/key 解析「显式参数 → 环境变量 → 默认」；先 TCP 快速探测（不可达
  快速报错并含端点地址，主端点失败自动切备用 env `WEMM_BASE_URL_FALLBACK`）；4xx 不重试
  （鉴权/路径/参数问题），5xx / 网络错误指数退避；多模态请求走 `messages` 透传、一条请求
  一组 messages，clip + 逐帧共 N+1 次请求线程池并发（1–4）。

## 3. 输入（端口）

端口声明（P5 端口协议，类属性自描述）：
`INPUTS = {"frame": "Image", "dets": "Records"}`、`OUTPUTS = {"clips": "Records"}`。

| 端口 | 类型 | 内容 | 连线说明 |
|---|---|---|---|
| frame | Image | 帧管理产帧 | web 驱动层**逐帧喂入**（`stage.run(frame, ts, frame_idx, dets=…)`），不经 `Pipeline.step`、无需 `inputs` 声明 |
| dets | Records | 上游识别记录流（观察源，**可缺省**） | `yolo.dets` / `plate_recog.plates` / `face_detect.faces` 皆可（按记录 `kind` 分桶 det/plate/face）；**单观察源直连**，spec 里 `inputs: {"dets": "上游产物"}` 显式声明；未连线 = 无观察标签，不报错（observation 全零值） |

> 观察段可插拔（想要什么标签就接什么源）与「YOLO 源插 det_filter、车牌/人脸源不插」的定案，
> 见 [混合检索设计稿](../design/hybrid_retrieval.md) §2.4 / §12；链路形态见
> [PIPE0004 专篇](../chains/pipe0004_video_embed.md) §2。

## 4. 输出（片段记录 + 入库）

### 4.1 片段记录（`video_embed.clips` 产物，每满窗一条）

dict 记录（records 契约：**向量与像素走旁路、不进 JSON 主干**）：

| 字段 | 内容 |
|---|---|
| kind / label / ts / frame_idx | `"clip"` / `"clip"` / 窗口首帧 ts / 窗口首帧号 |
| video / clip_id | 视频名（upsert 键一半）/ 库内条目 id |
| t_start / t_end | 窗口首末帧 ts（窗内口径；**无缝窗口**——t_end 延到下一窗首帧——由对齐层 `label_align.clip_windows` 重建，事件对齐用后者防窗缝丢事件） |
| frame_count / partial | 窗内帧数 / 是否尾段余帧片段 |
| thumb_path | 缩略图路径（**项目相对**、分隔符统一 `/`；换机不失效） |
| `embedding`（旁路） | clip 向量（ndarray） |
| `observation`（旁路） | 观察层标签 dict：`{det_stats, class_names, plate_nos, face_count, frame_dets}`（§4.3） |
| events | `clip_indexed` 事件（detail：video / t_start / t_end / clip_id / frame_count / thumb_path）——链路尾事件汇总可经 `Pipeline.last_events` 读；该事件是**入库回执**，事件归档不计入（见 PIPE0004 专篇 §4） |

### 4.2 向量底库（EmbedStore，落 `store_path`）

- 形态：`<store_path>.npz` 存向量矩阵（clip + frame 两层）+ 同名 `.json` 存元数据；
- **meta 只留窗口骨架 10 键**：`video / t_start / t_end / frame_count / partial / clip_dim /
  frame_dim / clip_seconds / thumb_path / frame_ts`——**观察标签绝不进 store meta**
  （标签落 `review_index` 表，hybrid_retrieval §4 定案）；
- upsert 幂等：键 `(video, t_start)`（t_start 取 3 位小数防浮点抖动），同键覆盖（旧 frame
  行清除、条目 id 不变）；
- 库级维度锁定：首条写入锁 `clip_dim/frame_dim`，此后不匹配向量拒收并指明两侧维度；
- 两级检索：`search()`（clip 粗排，点积=余弦）→ `search_frames()`（指定片段内 frame 精排，
  命中帧 ts 即播放定位点）；`delete_video()` 按视频清理（重建索引用）。

### 4.3 观察层标签（`observation` 旁路键）

`dets` 端口逐帧累计（内存）→ 满窗时经 `label_align.aggregate_frame_stats` 聚合：

| 键 | 内容 |
|---|---|
| det_stats | `{cls_id: {peak, avg, tracks}}`——帧级 count 峰值/均值（分母含窗内全部帧）+ track_id 并集大小（0=该窗未开跟踪） |
| class_names | `{cls_id: 标签名}`（采集自记录 `label`；换模型/标签漂移改库即可，不动向量、不重索引） |
| plate_nos | 车牌号集合（去重；取 `plate_no`，回退 `label`/`text`） |
| face_count | 窗口内单帧人脸数最大值 |
| frame_dets | **原始记录流**：逐帧 `{ts, recs:[…]}`，recs = 过闸 dict 记录主干原样（kind=det/plate/face 多态混存，剥 `model_path` 与旁路图/向量键，保证可 JSON 落表）；只收有记录的帧，单窗 recs 总数超上限按 ts 顺序截断并附 `"truncated": true` |

**`frame_dets` 与 `det_stats` 的关系**（一句话）：`frame_dets` 是**原始留存**（唯一真相），
`det_stats` 是**派生缓存**——丢了对账口径变了都可靠 relabel 从 `frame_dets` 重算，无需重跑
YOLO、无需重嵌向量（定案与理由见 [混合检索设计稿](../design/hybrid_retrieval.md)
§2.4 原则 3，不在此展开）。

## 5. 上下游衔接

- **上游**：帧管理（`frame`；复用 `composer.FrameManager` 取帧节奏 + 背压，对既有模块零侵入）；
  观察源（`dets`，见 §3）。
- **下游**：
  - 链路内：片段记录经驱动层 `pipe.publish()` 进链路尾事件流（`clip_indexed`）；
  - web 收尾（索引闭环）：驱动层取 `stage.indexed` 累计片段记录，把 `observation` + 事件归档
    经 `label_align` 契约 A 对齐落 `review_index` / `review_video` 表（读写语义见
    [数据库结构](../guides/database_schema.md) §11）；查询侧 `/api/review/search` 读同一底库
    （入口与漏斗见 [PIPE0004 专篇](../chains/pipe0004_video_embed.md) §4）。
- **装配**（`web_lab/server.py` `_build_video_embed_stage`）：
  - **惰性零副作用**：mock 档装配零网络零模型；real 档构造也只解析 env/URL（探测推迟到首帧）；
  - **装配加载既有库 + 维度护栏**：`store_path` 已有向量库 → 装配期先 `load()`（运行收尾
    save 是整库覆盖，空构造不加载会 last-writer-wins 清掉库里其它视频的向量）；已载库
    `clip_dim/frame_dim` 与本次配置不一致 → **装配期报错**（指明两侧维度与指引），禁止静默混维；
  - `thumb_dir` 相对路径归一项目根（server 进程 CWD=web_lab，不归一会落错位置）；
    `root=项目根` 传给阶段（thumb_path 转项目相对的基准）。

## 6. 验证

```bash
conda run -n ai python umvp/video_embed/test_video_embed.py    # 模块级（client/store/stage/label_align，mock）
conda run -n ai python tests/test_video_embed_e2e.py           # 注册/播种/装配/链路级 e2e（mock；--live 走真端点）
conda run -n ai python tests/test_review_search.py             # 检索 API 过滤/落表契约（mock）
conda run -n ai python tests/test_review_search_e2e.py         # 检索端到端（mock+HTTP；--live 同上）
conda run -n ai python tests/test_hybrid_retrieval.py          # 归档/落表/relabel/过滤全链路（mock；JSON 降级）
conda run -n ai python web_lab/validate_chain.py <spec.json>   # spec 校验（维度档位/端点配置）
```
（断言数不写死，以脚本运行自报为准。）

## 7. 注意事项（附录）

- **路径归一**：`thumb_dir` / `store_path` 相对路径一律按项目根解析——server 进程 CWD=
  web_lab，不归一会落错位置；写出的 `thumb_path` 恒为项目相对（root 基准 → CWD 相对 →
  文件名三重兜底，绝不出机器绝对路径）。
- **窗口时间由帧 ts 承载**：`frames_per_clip` 管切分、`clip_seconds` 只是标称展示——帧管理
  frame_skip 变了窗口实际秒数随之变，不必改本模块参数。
- **相似度阈值**：同域文本余弦普遍偏高（实测同域文本对 ≈0.92），`EmbedStore.search` 的
  thresh 缺省不截断、按相对排序——阈值调优按相对远近不按绝对值（wemm_integration.md §6）。
- **`clip_indexed` 是回执不是告警**：事件归档口径排除它（PIPE0004 专篇 §4）。

## 8. 边界与不做的事

- **不做检测/识别**：YOLO/车牌/人脸是独立阶段接在观察段；本模块只聚合记录、不产检测，
  只取舍观察、不改识别结果；
- **不做查询 UI / 检索漏斗**：查询层在 web 板块（`/api/review/search`），本文不展开；
- **无告警语义**：入库即目的；分析类事件由其它链路产出、经事件归档对齐进标签表；
- **不替代分析链路**：本链路是离线索引（可先于分析链路单独跑），与实时告警链路正交；
  标签体系进一步的语义档 / hybrid RRF 融合为后置项（hybrid_retrieval §8 非目标）。
