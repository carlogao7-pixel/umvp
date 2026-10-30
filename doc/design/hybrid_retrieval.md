# 混合检索：标签数据与融合查询（设计稿）

> 状态：**设计稿（标签 schema 已冻结，待评审过审后实施）** ｜ 提出方：orchestrator（主单 WO-0033）
> ｜ 2026-10-28 初稿，**2026-10-29 第三轮定案**（§2.5 字段冻结 / §2.6 采集时间上提视频级 /
> §4 存储直上 `review_index` 表 / §3 回填不重嵌入，定案见 §11）
> 需求：车牌识别 / VLM 研判 / YOLO 检出等模块的结果对向量查询是**天然的结构化标签**——
> 本文设计「标签数据模型 + 采集路径 + 混合查询（过滤式 → 融合式）」，让查询既会"文搜画面"
> 也会"上周哪段出现过车牌 X / 哪段出过斗殴警情"。
> 上位方案：[video_review_search.md](video_review_search.md) §3（三档打标/片段窗口）§4（三级漏斗）§5（review_index）
> ——本文**细化并落地其标签与融合部分，不重写不冲突**；向量基础设施现状见 [chain 交付](wemm_integration.md)。
> 前置交付：PIPE0004 向量索引链路 + `EmbedStore`（WO-0031）、查询层 `/api/review/search`（WO-0032）。

## 0. 结论速览

1. **标签数据四类**：结构化统计（YOLO）、事件（警情/车牌/人脸）、语义描述（VLM）、向量（已建）；
2. **采集路径本期走 A**（历史分析事件留存归档 → 片段窗口对齐）——最小侵入；路径 B（回顾索引链路
   轻/语义档现打标，上位设计稿原案）工程大、后置，两者标签**同表合流**不冲突；
3. **查询两级**：过滤式（硬约束 pre-filter + 向量排序，MVP）→ 融合式（RRF 排名融合，进阶）；
4. **帧向量层改为可选**：确认产品形态"列时间段即可"时，索引可关帧向量（成本大头 N 次/片段省掉），
   需要点击定位到帧再开——参数开关，不是架构改动。

## 1. 融合式检索原理（先讲清机制）

### 1.1 两级形态：过滤式 vs 融合式

```
过滤式（filter-then-rank，MVP）：
  硬约束（时间窗/视频/事件名/车牌号）→ 先筛掉不合格片段 → 幸存集合内按向量分排序
  ✔ 简单可靠；✔ 结果必然满足全部硬约束；✘ 约束是 0/1 的，容不得"部分符合"

融合式（fusion ranking，进阶）：
  无硬约束时，多个信号**各自**召回+排名，再合并成一个总排序
  ✔ 能答"和纠纷画面相似、且出现过警察的片段"这种多信号软查询；✘ 需要处理分数不可比
```

### 1.2 融合怎么合：排名融合（RRF），不是分数加权

**核心难题**：不同信号的分数量纲不同——向量余弦挤在 0.3~0.9（同域文本实测 0.92），
metadata 匹配是 0/1，关键词 BM25 又是另一把尺子。**分数直接加权 = 不可比 + 每加一路信号调一遍权重**。

**RRF（Reciprocal Rank Fusion）**绕开分数、只用排名：

```
每路信号独立给全量（或大候选集）排序 → 每个片段得分 = Σ 1 / (k + rank_i)，k 常数（60）
例：片段 S 在"向量相似"排第 1、"含车牌"排第 3 → RRF = 1/61 + 1/63 ≈ 0.0323
     片段 T 只在"向量相似"排第 2        → RRF = 1/62 ≈ 0.0161 → S 排 T 前
```

- 两路都靠前的片段自然胜出（"既相似又含标签"的完美命中）；
- 免归一化、免逐信号调权（只有 k 一个常数，60 是经验值基本不用动）；
- 工业界混检（Elasticsearch 等）的默认做法。

### 1.3 流程（本文落地形态）

```
查询（文本 + 可选过滤器）
  ├─ 信号① metadata 命中集：时间窗/视频/事件名/车牌号 → 0/1 命中 → 排名（按 ts 或命中强度）
  ├─ 信号② 向量相似：query → embed_text → clip 粗排 Top-N → 排名
  ├─（未来）信号③ 关键词/描述全文 → 排名
  ├─ 硬过滤器存在 → 先 pre-filter（两路都在幸存集内计算）
  └─ RRF 融合 → 全局 Top-K → 每条带 why（各信号命中来源与排名）
```

**三条纪律**（设计稿一以贯之）：
1. pre-filter 在**所有信号计算之前**（避免 Top-K 被过滤掏空、漏召回）；
2. 每路候选截断深度 ≥ 最终 Top-K × 3（截断太浅融合会丢真命中）；
3. 每条结果与空结果都带 `why`（哪路命中、各路排名），空结果可解释。

## 2. 标签数据：从模块结果到片段标签

### 2.1 四类标签与来源映射

| 标签类 | 内容 | 来源 | 现状 |
|---|---|---|---|
| 结构化统计 | 各类别峰值/均值 ×track 数 | 索引时 YOLO 统计（轻档）或历史检出留存 | 未留存 |
| **事件** | 警情名/研判结论/车牌号（+detail 摘要+ts） | **PIPE0001/0002 链路尾事件** | **运行即逝，未落盘（缺口）** |
| 语义描述 | 一句话描述 | VLM 打标（语义档） | 未建 |
| 向量 | clip/frame | PIPE0004 | ✅ 已建 |

### 2.2 模块结果 → 标签的具体映射

| 模块产出 | 标签形态 | 查询用法 |
|---|---|---|
| PIPE0002 车牌事件（`name=车牌号`，record 含 type/color/score） | 片段的 `plates: [{plate_no, color, type, score, ts}]` | 精确过滤"车牌 X"（0/1 命中）+ 车牌颜色/类型可作展示 |
| PIPE0001 警情事件（seq_filter：name+detail 统计；VLM 研判事件：name=结论 + content） | 片段的 `events: [{name, kind: 警情/研判, ts, detail 摘要}]` | 事件名过滤（"斗殴"）+ 与向量相似度融合（"语义像纠纷且出过警情"） |
| PIPE0001 YOLO 检出统计 | 片段的 `det_stats: {cls: {peak, avg, tracks}}` | 结构化过滤"人数 ≥4"（数值条件） |
| PIPE0003 人脸截取 | 片段的 `face_count`（截取落盘即目的，标签仅计数） | 过滤"有人脸活动的片段" |
| VLM 语义描述（语义档，后置） | 片段的 `description` | 关键词/全文信号（信号③） |

### 2.3 对齐规则：事件 ts → 片段窗口

- 索引粒度 = PIPE0004 片段窗口 `[t_start, t_end)`（默认 5s）；
- 事件归属：**事件 ts 落入哪个窗口就归哪个片段**（窗口右开左闭；跨窗事件按命中窗口各记一条，`ts` 保留原值）；
- 抽帧节奏差异不敏感：标签对齐用**真实时间 ts**，与 frame_skip 无关——分析链路与索引链路可各自独立跑、先后不限。

### 2.4 观察层标签：双层模型与帧级统计表示（2026-10-28 讨论定案）

**双层模型**（链路尾事件是收敛语义——门控/冷却/去重把"车停 30s"收敛成 1 条事件，
拿它当标签会丢时空分布；但标签也不能完全不过滤，低质误检会污染标签）：

| 层 | 内容 | 来源 | 过什么筛选 |
|---|---|---|---|
| **观察层** | det_stats / plate_nos / face_count（"这个窗口**出现过**什么"） | 识别**记录流**（识别层产出，跨窗口现） | 只过**有效性筛选**（质量闸）；**不过**时序收敛（节流/冷却/去重） |
| **判断层** | 警情名 / VLM 研判结论（"这个窗口**发生了**什么值得报的事"） | 链路尾**事件** | 天然已收敛（输出语义） |

"去垃圾"（有效性筛选）留着，"去重复"（时序收敛）去掉——两者职责不同。

**观察层表示法统一为帧级统计**（不逐对象建标签）：

```
帧级：{ts: 12.3,
       stats: {"car":    {"count": 3, "track_ids": [7, 8, 9]},
               "person": {"count": 1, "track_ids": [12]}},
       plate_nos: ["沪DHH230"]}          # 集合语义，不重复计
窗口聚合：count 取峰值/均值；track_ids 取并集 → det_stats 的"track 数"= 并集大小（去重计数）
```

- **count 必须配 track 集合**：纯 count 会把同一辆车在 6 帧里数 6 遍（"6 辆车"实为 1 辆）；
  track_ids 很轻（几个整数），窗口级去重计数全靠它；
- **边界一（记录流契约不动）**：帧级化只发生在**索引侧聚合层**——video_embed 把上游 Records
  **降维**成帧级统计进 meta；分析链路内部 records 仍是模块通用语（事件附着/裁剪/坐标还原依赖
  逐对象记录）。一句话：**记录流是过程语言，帧级统计是标签形态**。

**观察段可插拔**（PIPE0004 自由组合轻量识别模块生产识别数据，均不经筛选/事件尾，纯产观察）：

| 想要的观察标签 | 观察段组合（现有模块拼装） |
|---|---|
| 类别/目标统计（默认） | `帧管理 → YOLO → video_embed` |
| 车牌观察 | `帧管理 → YOLO → target_crop(裁车) → plate_recog → video_embed` |
| 人脸观察 | `帧管理 → YOLO → target_crop(裁人) → face_detect → video_embed` |

- video_embed 的 `dets` 输入端口**泛化**：上游是 `yolo.dets` / `plate_recog.plates` /
  `face_detect.faces` 任意 Records 流，聚窗时统一降维成帧级统计；
- 本期**单观察源直连**（多源聚合等 graph_chain 的 merge 节点，P2-3 既定规划）；
- 车牌精确到窗口的观察标签需索引链路跑车牌段——MVP 车牌标签先用判断层事件
  （接受"只有首报窗口有标签"），观察层车牌按需再加。

### 2.5 标签 schema 冻结（2026-10-29 定案，字段名即未来列名）

四条冻结原则：

1. **三层分工**：`plate_nos`（观察层，只要号）+ `plates[]`（判断层，带属性与 ts）+
   `events[]`（判断层，只放警情/VLM 研判）——车牌号不再同时出现在 `events` 里，避免两套口径；
2. **类别键用 `cls_id`**，另存 `class_names` 映射：YOLO 换模型/标签名漂移时**改库即可**
   （向量一次性入库，改名不动向量，也不触发重索引）；
3. **帧级详情落盘，窗口聚合降级为可再生缓存**（2026-10-29 第四轮改定，原"帧级统计不落盘"废止）：
   逐帧**过质量闸后的原始记录流**存进标签行 `frame_dets`（dict 主干原样、kind 多态混存
   det/plate/face、剥 `model_path`、单窗超上限截断+标记）；`det_stats` 保留但只是**派生缓存**
   （丢了对账口径变了都可靠 relabel 从 `frame_dets` 重算，不再承担"唯一真相"）。理由：
   ≤10 帧/窗的现实下体积可忽略（~550KB/半小时视频 vs 聚合 85KB）；原始留存使未来任意新统计
   （趋势/停留/共现/空间）= relabel 重算即可，**无需重跑 YOLO、无需重嵌向量**；bbox/score 留存
   顺手解锁将来"检出框重放"类需求；本期 det 类硬过滤本就后置（§5.1），先付复杂度不划算；
4. **路径一律项目相对**（现状实测 `out/video_vecs.json` 的 `thumb_path` 是绝对路径，
   换机即失效——随本次 schema 冻结一起改）。

**片段标签（每片段一行）**

| 字段 | 类型 | 语义 | 示例 |
|---|---|---|---|
| `video` | str | 视频名（与向量库 upsert 键一致） | `violent1.mp4` |
| `t_start` / `t_end` | float 秒 | 片段窗口（右开左闭沿用 §2.3） | `5.0` / `8.0` |
| `det_stats` | JSON | 观察层**派生缓存**：`{cls_id: {peak, avg, tracks}}`（peak/avg = 帧级 count 峰值/均值；tracks = track_ids 并集大小，0=该窗未开跟踪）。可从 `frame_dets` 重算，口径变更走 relabel | `{"0": {"peak": 3, "avg": 2.1, "tracks": 2}}` |
| `frame_dets` | JSON arr | 观察层**原始记录流**：逐帧 `{ts, recs:[…]}`，recs = 过质量闸后的 dict 记录主干原样（`kind=det/plate/face` 多态混存，含 cls_id/track_id/bbox/score 等，剥 `model_path`）；单窗 >500 条截断并附 `"truncated": true`；为时间形态/数值统计与将来框重放的数据源 | `[{"ts": 0.0, "recs": [{"kind": "det", "cls_id": 14, "bbox": [812,104,893,171], "score": 0.71}]}]` |
| `class_names` | JSON | `cls_id → 标签名`（采集自上游记录 `label`） | `{"0": "person", "2": "car"}` |
| `plate_nos` | JSON arr | 观察层车牌号集合（去重、无时间） | `["沪DHH230"]` |
| `plates` | JSON arr | 判断层车牌事件 `{plate_no, color, type, score, ts}` | `[{"plate_no": "沪DHH230", "ts": 6.2}]` |
| `events` | JSON arr | 判断层警情/研判 `{name, kind, ts, detail}`；`kind ∈ {警情, 研判}` | `[{"name": "斗殴", "kind": "警情", "ts": 6.2}]` |
| `face_count` | int | 观察层人脸计数（窗口内最大） | `2` |
| `description` | str | 语义档 VLM 描述（后置，空串） | `""` |
| `thumb_path` | str | 窗口首帧缩略图，**项目相对** | `out/video_thumbs/x_5.000.jpg` |
| `vector_model` / `clip_dim` / `frame_dim` | str/int | 向量来源版本（检索按 `(model, dim)` 过滤） | `WeMM-2B` / 1024 / 256 |

> 字段名与 `review_index` 列名**同名同义**（§4 本期即落表，不再"预留迁移"）。

### 2.6 采集时间：视频级字段，不进片段标签（2026-10-29 定案）

"上周哪段"需要墙钟时间，但**片段只有视频内相对秒**。时间上提为**视频级一行**：

| 字段 | 说明 |
|---|---|
| `video` | 主键 |
| `video_time` | 采集时间（ISO8601 含时区）；**可为空** |
| `video_time_src` | 来源标记：`filename` / `tag` / `manual` / `""`（空=未知，用于 why 可解释） |
| `indexed_at` / `vector_model` / `clip_dim` / `frame_dim` / `duration` | 索引备注（同一视频重跑覆盖） |

- **来源优先级：文件名 > 标签**（`tag`=索引时手填/上传处登记的元数据；2026-10-29 定）。
  文件名解析不出 → `video_time=""`、`video_time_src=""`，**不阻塞索引**；
- 为什么不做成片段列：一行/视频 vs 360 行重复；将来补真实采集时间只改一行（同 §2.5 原则 2）；
- 时间窗过滤语义：`ts_abs = video_time + t_start`；`video_time` 为空的视频**不参与**时间窗
  过滤，结果 `why` 注明"该视频无采集时间，未参与时间窗过滤"（可解释，不静默丢弃）。

## 3. 采集路径 A（本期）：历史分析事件留存归档

**前置缺口**：链路尾事件目前只活在运行结果/页面里，关页面即失。

- **留存动作**：分析链路运行收尾（`_LAST_RUN` 快照处，WO-0028 既有习惯位），把「视频名 + 事件列表
  （name/ts/kind/detail 摘要）」追加写入 `out/events/<视频名>.jsonl`（一行一次运行，含运行时间戳与
  链路 ID）——**只追加、零 DB 依赖、对齐 §5「无 DB 降级」形态**；
- **对齐动作**：PIPE0004 索引收尾时，若存在该视频的事件归档 → 按窗口对齐 → **写入 `review_index`
  表**（§2.5 字段；标签不进 npz meta，见 §4）；观察层标签（det_stats/plate_nos/face_count）在索引
  过程内由观察段记录流聚合出，同一收尾一并 upsert；
- **幂等**：同一运行行只入一次（jsonl 行号/运行戳去重）；索引重跑对齐重算不重复；
- **回填（2026-10-29 定案）**：分析链路晚于索引建库时，只需**按 `(video, t_start)` 重算标签行
  upsert 进 `review_index`——不动向量、不重嵌入**。实现上给 web 侧一个"仅重对齐"入口
  （参数：视频名 / 事件归档路径 / 窗口规则），复用同一对齐函数；向量缺口才需重跑 PIPE0004；
- 路径 B（回顾索引链路轻/语义档，含 YOLO 统计现打标与 VLM 描述）按上位设计稿 §3.1 后置，
  产出同样落 `review_index`/meta，**标签模型与查询层完全复用**。

## 4. 存储：本期直接上 `review_index` 表（2026-10-29 定案）

原方案是"本期 meta 兜底、下期建表"，**改为本期直接建表**：混合检索的过滤本来就靠结构化查询，
而 `meta` 兜底的查询侧要**加载全量 npz 同名 json**（标签一多就每次查询吃全量），做两遍不划算。

| 存储 | 内容 | 说明 |
|---|---|---|
| 新表 `review_index` | 片段标签（§2.5 全部列）+ `(video, t_start)` 唯一键 | 前缀 `RVIDX`（`db.py _ID_PREFIXES`）；索引 `(video, t_start)`；`video/t_start` 与向量库行一一对应 |
| 新表 `review_video` | 视频级元数据（§2.6：`video_time/video_time_src/indexed_at/模型维度版本`） | 前缀 `RVVID`；`video` 主键；**采集时间只存这里** |
| `out/video_vecs.npz` / `out/video_frame_vecs.npz` | 向量本体只留 `video/t_start/t_end/thumb_path/clip_dim/frame_dim` | **标签不进 npz meta**（避免下次再挪）；键 `(video, t_start)` 与表行对齐 |
| 无 DB 降级 | `out/review_index.json` + `out/review_video.json` | `--no-db` 时同结构落 JSON，检索逻辑同一套代码（SQL 路径的替身：内存过滤） |

- 不一致处理：表有行而向量库无行（或反之）→ 检索侧按**向量库行为准**，缺标签按"无标签"处理并在
  `why` 注明；重跑索引以 `(video, t_start)` upsert，删除视频时两侧同步清理；
- 模型/维度变更：`(vector_model, clip_dim, frame_dim)` 记在行上，检索按当前库维度过滤，新旧共存
  （遵循 WeMM 专篇 §2.2 约束，截维方式全链路统一走服务端 `dimensions`）。

## 5. 查询层扩展（`/api/review/search` 增量）

### 5.1 入参增量

| 参数 | 说明 |
|---|---|
| `time_start` / `time_end` | 时间窗过滤（墙钟，硬约束）：按 `video_time + t_start` 判定；无 `video_time` 的视频不参与该过滤并在 `why` 注明（§2.6） |
| `event_name` | 事件名/车牌号/研判结论 **精确匹配**过滤（标签里是枚举值，先做精确，模糊留全文信号）——口径按 2026-10-29 定案：精确匹配即可 |
| `min_face_count` 等数值过滤 | 后置（det_stats 进表后开；列已在 §2.5 预留） |
| `mode` | `vector`（现状，缺省）/ `hybrid`（RRF 融合：metadata 命中集 + 向量排名）。**本期只实现过滤式，`mode` 字段预留**（2026-10-29 定案） |

### 5.2 行为

- **过滤式（MVP）**：过滤器存在 → pre-filter 幸存集内向量排序；`why` = "时间窗命中；车牌 X 命中；向量相似度 0.42"；
- **融合式（hybrid）**：metadata 命中集与向量 Top-N 各自排名 → RRF → Top-K；`why` = "向量排#1；车牌命中#3；RRF 0.032"；
- 响应新增 `signals: {meta_hits, vec_top_n}` 便于前端展示与调试；
- mock 档语义不变（伪向量确定性），标签过滤与 mock 完全正交、可测。

### 5.3 前端

- 检索页签增「过滤器」折叠区（时间窗/事件名/视频，留空=不启用）+ 模式切换（向量/混合）；
- 卡片 why 行展示各信号命中；其余交互不变。

## 6. 帧向量层取舍（成本开关）

- 帧嵌入是索引成本大头（每片段 N+1 次请求的 N）；`frames_embed`（默认 true 保持现状）设 false 时
  只建 clip 向量——**查询仍可用（结果定位到 5s 窗），播放定位退化到窗口头并提示**；
- 与混合查询正交：标签过滤/融合不依赖帧向量。

## 7. 实现拆分（过审后立子单）

| 板块 | 内容 |
|---|---|
| pipe-chain | `EmbedStore` meta **瘦身**（只留 video/t_start/t_end/thumb_path/维度，标签出库）＋ **video_embed 泛化 dets 输入**（上游任意 Records → 帧级统计聚合：count+track_ids 集合/plate_nos 集合，窗口级 peak/avg/并集大小）＋ 片段窗口对齐函数（事件 ts → 窗口归属，纯逻辑可单测）＋ `frames_embed` 开关贯通＋缩略图路径改项目相对＋**观察层 `frame_dets` 原始记录流产出**（§2.4 原则 3 第四轮：过闸原始 dict 逐帧分组、剥 model_path、单窗上限） |
| web | ① 事件留存归档（分析收尾写 jsonl，同步/流式两路径）；② `review_index`/`review_video` 建表（`db.py` 前缀 RVIDX/RVVID + 无 DB JSON 降级；`review_index` 含 `frame_dets` 列，存量表自动补列）；③ 索引收尾对齐落表 + "仅重对齐不重嵌入"入口（relabel 保留旧 `frame_dets`）；④ `/api/review/search` 过滤器（时间窗/事件名/车牌/视频）+ `mode` 预留 + why；⑤ 前端过滤器与 why 展示；⑥ validate/回归 |
| qa | 对齐函数单测（跨窗/边界 ts、frame_dets 截断/剥字段）、留存-对齐-查询全链路 mock 断言（含"先建库后归档→只重对齐不重嵌入"）、过滤式排序与 why 正确性、无 DB 降级、回归 |

## 8. 非目标（本期不做）

- 回顾索引**运行模式**与语义档（VLM 描述）——上位 §3.1 原案，后置（YOLO 观察统计已前移为
  PIPE0004 观察段，见 §2.4）；
- hybrid（RRF 融合）查询本体——本期只做过滤式，`mode` 字段预留（§5.1）；
- 关键词/全文信号（信号③）、数值范围过滤（det_stats 类）、LLM 意图解析；
- 以图查片 UI、检出框叠加重放。

## 9. 待评审问题（Q1–Q3 已于 2026-10-29 拍板，见 §11）

- **Q1** 事件归档粒度：全量事件都归档（jsonl 可能大——长视频高频警情），还是只归档链路尾事件
  （=输出语义，天然已收敛）？→ **定：只归档链路尾事件**。
- **Q2** `event_name` 过滤口径：本期精确匹配够不够（车牌号/警情名都是枚举值）？→
  **定：本期精确匹配，模糊留信号③**。
- **Q3** hybrid 模式是否本期实现：过滤式已覆盖"带条件查询"主需求，RRF 融合可后置到
  metadata 变多（路径 B）之后。→ **定：本期过滤式 + `mode` 字段预留**。

## 10. 讨论定案（2026-10-28）

1. **双层标签模型认可**：观察层（记录流，不过时序收敛）+ 判断层（链路尾事件），见 §2.4；
2. **PIPE0004 观察段可插拔轻量识别模块**（YOLO 默认；车牌/人脸组合可选），video_embed 泛化
   dets 输入统一聚合，见 §2.4；
3. **观察层标签统一帧级统计表示**（count + track_ids 集合 + plate_nos 集合；窗口聚合取峰值/
   并集），记录流契约不动（过程语言 vs 标签形态），见 §2.4。

## 11. 第三轮定案（2026-10-29）

| # | 议题 | 定案 | 落点 |
|---|---|---|---|
| 1 | 标签 schema | 按 §2.5 冻结；三层分工（`plate_nos`/`plates`/`events`）、`cls_id + class_names`、帧级统计不落盘、路径项目相对。**"名称有偏移改库就行"**（改 `class_names` 不动向量、不重索引） | §2.5 |
| 2 | 墙钟时间 | **不进片段标签**，上提为视频级 `video_time/video_time_src`（一行/视频）；来源优先级 **文件名 > 标签**；解析不出留空、不阻塞索引；时间窗过滤时无采集时间的视频不参与并 `why` 注明 | §2.6 |
| 3 | Q1/Q2/Q3 | 只归档链路尾事件；`event_name` 精确匹配；本期只做过滤式 + `mode` 预留 | §3 §5.1 §9 |
| 4 | 标签回填 | **只重对齐、不重嵌入**：按 `(video, t_start)` upsert `review_index`，向量库不动 | §3 |
| 5 | 存储形态 | **本期直接上表**（`review_index` + `review_video`），标签不进 npz meta；无 DB 走同结构 JSON 降级 | §4 |
| 6 | 观察层帧级内容 | **原始记录流 `frame_dets` 落盘**（过质量闸后、剥 model_path、kind 多态混存、单窗上限保护）；`det_stats` 降级为可再生缓存；质量闸保留但观察段可**放宽**（原始留存很便宜，宁可多存，查询期可再筛） | §2.4 §2.5 |
| 7 | 查询期统计 | 时间形态/数值统计（持续性/趋势/突发/track 停留等）**整体后置到 hybrid 阶段**，本期只落标签不做过滤器，与 RRF 一起设计 | §5 §8 |

> 仍未定（实现前定即可，不阻塞评审）：观察段端口的连线形态（`video_embed` 第二入端口 vs 单源
> 直连 + 终段白名单）、观察链路是否插 `det_filter`（直接影响 `det_stats` 数值口径）、
> `frames_embed` 默认值（产品形态是否"列出时间段即可"）。

## 12. 实施前收口（2026-10-29，开工定案）

上表"仍未定"三项按下列结论执行，接口契约固定如下，供 pipe-chain / web / qa 并行开发：

1. **端口形态**：`VideoEmbedStage.INPUTS = {"frame": "Image", "dets": "Records"}`——第二入端口
   `dets`，**可缺省**（未连线=无观察标签，不报错）；"单观察源直连"= 只连一路 dets 源。
2. **是否插 `det_filter`**：**YOLO 观察源插**（`帧管理→YOLO→det_filter→video_embed`，只过有效性
   筛选、不过时序收敛，§2.4 原则落地）；**车牌/人脸观察源不插**（识别模块自带 score 置信度即质量闸，
   且其产物非 `kind=det`）。`det_stats` 数值口径 = **det_filter 放行后**的检出统计。
3. **`frames_embed` 默认保持 `true`**（现状不变），本期只把开关贯通到装配与驱动层；默认值调整等
   产品形态确认后再改（改默认 = 改 preset 一行，不涉架构）。

### 12.1 接口契约（跨板块，改契约需先改本节）

**A. 对齐/聚合纯逻辑 —— `umvp/video_embed/label_align.py`（pipe-chain 提供，web 调用）**

| 入口 | 签名 | 说明 |
|---|---|---|
| `parse_video_time` | `(filename: str, tag: str = "") -> (video_time: str, src: str)` | 文件名优先、其次 tag；解析不出返回 `("", "")`；`video_time` 为 ISO8601（带时区） |
| `clip_windows` | `(frame_ts: Sequence[float]) -> List[Tuple[float, float]]` | 由片段帧 ts 出窗口（右开左闭，§2.3） |
| `aggregate_frame_stats` | `(per_frame: Sequence[dict]) -> dict` | 每帧 `{ts, dets:[...], plates:[...], faces:[...]}` → `{det_stats, class_names, plate_nos, face_count, frame_dets}`（§2.5 观察层；`frame_dets` = 过闸原始记录逐帧分组，剥 model_path，单窗 >500 条截断+`truncated` 标记） |
| `align_events` | `(events: Sequence[dict], windows) -> List[dict]` | 每条事件补 `window` 归属（跨窗各记一条，`ts` 保留原值） |
| `build_clip_tags` | `(video, window, obs: dict, events: Sequence[dict]) -> dict` | 产 §2.5 全部标签字段（含 `frame_dets` 透传；向量版本三列由调用方补） |

**B. 片段记录旁路键**：`VideoEmbedStage` 产出的 clip 记录增 `observation`（= `aggregate_frame_stats`
结果）；**标签不写 `EmbedStore` meta**，store meta 只留
`video/t_start/t_end/frame_count/partial/clip_dim/frame_dim/clip_seconds/thumb_path/frame_ts`，
其中 `thumb_path` **项目相对**。

**C. 事件归档格式（web 写、web 读；qa 造 mock）**：`out/events/<video>.jsonl`，一行一次运行：

```json
{"run_id": "…", "pipeline": "PIPE0001", "at": "2026-10-29T10:00:00+08:00",
 "video": "x.mp4", "events": [{"name": "斗殴", "kind": "警情", "ts": 12.3, "detail": {}}]}
```

**D. 表结构**：`review_index` / `review_video` 列名 = §2.5 / §2.6 字段名（同名同义）；
无 DB 时同结构落 `out/review_index.json` / `out/review_video.json`。
