# 视频向量化链路 PIPE0004

> 类型：模块自定义拼接链路 | 装配：`web_lab/server.py`（`_chain_from_spec` 装配，
> `_seed_pipeline` 播种） | 运行：`web_lab/test.html` 链路测试台（离线索引）
> 数据库：pipelines 表 `PIPE0004`（种子名「视频向量化-嵌入查询链路」）+ pipeline_steps 4 步
> （引用 4 条模块 preset；`stage_inputs` 声明观察源连线）

## 1. 链路定位

保留链路之一（第四条，**离线索引**）：**帧管理 → YOLO 观察 → 检测筛选（质量闸）→ 视频嵌入**。
面向"文搜视频"的索引层：把视频按片段窗口向量化入库（clip + frame 两级向量入 `EmbedStore`），
同时把观察段识别记录聚成窗口级标签，索引收尾对齐落 `review_index` / `review_video` 表——
**入库（向量 + 标签）即目的**，与 PIPE0003「截取落盘即目的」同型；无 VLM、无告警报送，
链路自身不产分析类事件。索引与分析互相独立：本链路可先于分析链路单独跑，先后不限。

体系与定案（标签数据模型 / 查询两级漏斗 / 分工边界）见设计稿：
[混合检索](../design/hybrid_retrieval.md) 与 [历史视频回顾检索](../design/video_review_search.md)；
WeMM 服务端部署与请求 schema 见 [WeMM 嵌入对接](../design/wemm_integration.md)。
**本篇写链路与模块接线，体系与定案不在此重复。**

各阶段参数细节见模块分册（[帧管理](../modules/frame_manager.md)、
[YOLO 识别](../modules/yolo.md)、[检测筛选](../modules/det_filter.md)、
[视频嵌入](../modules/video_embed.md)）。

## 2. 拼接结构（数据库存储）

```
PIPE0004 视频向量化-嵌入查询链路（pipelines 表，streams=1）
 ├─ 步骤0  frame_manager  视频向量化·帧管理     （sampling=analysis, frame_skip=15）
 ├─ 步骤1  yolo           视频向量化·YOLO观察   （conf=0.25，classes 不限——观察口径最宽）
 ├─ 步骤2  det_filter     视频向量化·检测筛选   （模块默认空条件=不筛选，质量闸留位）
 └─ 步骤3  video_embed    视频向量化·视频嵌入   （use_real_embed=false 起步；链路尾）
     连线：inputs.dets ← det_filter.dets（stage_inputs 位置键声明）
```

| 步骤 | preset | 阶段参数要点 |
|---|---|---|
| frame_manager | 「视频向量化·帧管理」 | `frame_skip=15`（稀疏抽帧；6 帧/窗节奏下 25fps 视频约 10s 一窗，窗口实际时间由帧 ts 承载） |
| yolo | 「视频向量化·YOLO观察」 | `classes=""`（全类别进观察统计，观察场景默认最宽口径；只想统计特定类别改此一词，如 `0,2`）；conf 用播种默认 0.25 |
| det_filter | 「视频向量化·检测筛选」 | 播种用模块默认（不筛选）——**观察层质量闸**：只过有效性筛选、不过时序收敛；素材有明确尺寸下限可加（如 `filter="long >= 30"`）。观察段可**放宽**：原始留存便宜，宁可多存、查询期再筛 |
| video_embed | 「视频向量化·视频嵌入」 | `use_real_embed=false`（mock 档起步，真端点联调改 true 并导出 `WEMM_BASE_URL`）、`embed_dimension=1024`、`frame_dimension=256`、`frames_per_clip=6`、`thumb_dir/store_path` 默认在 out/ 下 |

链路只存「步骤引用哪些 preset + 顺序 + stage_inputs 连线」，模块参数本体在各模块参数表；
每模块恰一份参数、链路恰一条，由服务器启动幂等播种（`_SEED_VEMB_STAGES` + `_seed_chain`）。

**观察段为可选扩展**：车牌/人脸观察组合（`帧管理 → YOLO → target_crop(裁车/裁人) →
plate_recog / face_detect → video_embed`）可按需在设计器自行拼装——识别源自带 score
置信度即质量闸，**不插 det_filter**（定案 hybrid_retrieval.md §12-2）；video_embed 的
`dets` 端口泛化收任意识别记录流（单观察源直连，见 [视频嵌入](../modules/video_embed.md) §3）。

## 3. 数据流（dict 记录 × 旁路）

```
视频帧 ──帧管理 要图决策（每 15 帧取 1，积压自适应放宽）
            │ 要图的帧
            ▼
        YOLO 观察（纯推理）→ dict 记录（kind=det，带 label/cls_id/track_id/bbox/score）
            │
            ▼
        det_filter 质量闸（放行子集原样透传；播种默认不筛选）
            │ 有效检测（观察口径）
            ▼  （dets 连线：驱动层逐帧喂入 video_embed.dets）
        video_embed（逐帧收帧 + 收观察）
            │  满窗（6 帧）→ WeMM 嵌入（clip 多图一条消息 + 每帧单图，N+1 次请求）
            │  → clip/frame 向量入 EmbedStore（upsert 键 (video, t_start)）
            │  → 观察记录聚合成 observation（det_stats/class_names/plate_nos/face_count/frame_dets）
            │  → 窗口首帧缩略图落盘（项目相对路径）
            │  产出片段记录（kind=clip；向量走 embedding 旁路、标签走 observation 旁路）
            │       + clip_indexed 事件
            ▼
        链路输出 = clip_indexed 事件（片段记录携带，经 publish 进链路尾事件流）
```

- **JSON 主干 × 旁路**：片段记录主干只带窗口骨架字段（kind/ts/video/clip_id/t_start/t_end/
  frame_count/partial/thumb_path）；clip 向量走 `embedding` 旁路、观察标签走 `observation`
  旁路（records 契约：像素与向量不进 JSON）。标签**不写** `EmbedStore` meta（store 只留
  窗口骨架元数据，标签落 review_index 表——hybrid_retrieval §4 定案）。
- **入库即目的**：片段记录与 `clip_indexed` 事件就是链路产出；检索是另一条 API 通路，
  不在链路内。
- **端口与连线（P5）**：`video_embed` 两入端口 `frame` / `dets`——`frame` 由驱动层逐帧喂入
  （不经 `inputs` 声明）；`dets` 是驱动层解析端口（`_CHAIN_DRIVER_PORTS`），spec 里
  `inputs: {"dets": "det_filter.dets"}` 显式声明，**不声明则无观察标签**（不报错，
  det_stats 恒空）——播种形态里「观察段结构」与「连线声明」必须成对出现。
  产物名 `frame_manager.frame` → `yolo.dets` → `det_filter.dets` → `video_embed.clips`，
  逐帧登记在 `Pipeline.artifacts`；链路尾 `Pipeline.last_events` = 片段记录携带的
  `clip_indexed` 事件汇总。

## 4. 索引闭环：播种收敛 → 收尾落表 → relabel → 查询

### 4.1 播种与存量收敛

- **幂等播种**（服务器启动 `_seed_pipeline`）：链路不存在 → 建 4 步 + `stage_inputs` 连线
  声明（`#<video_embed 位置>.dets ← det_filter.dets`，随 save_pipeline 双写 pipeline_edges
  表）；结构一致 → 原样保留（不动用户改动）；
- **存量自动重建**：结构变（如两阶段旧库升级到 4 阶段）→ 按模块复用旧 preset（FM/VEMB
  参数保留、YOLO/DF 补建），重建步骤，幂等复播零写入；
- **装配加载既有库**：`store_path` 已有向量库 → 装配期 `load()` + 维度护栏（运行收尾 save
  是整库覆盖，不加载会清掉库里其它视频的向量；维度不一致装配期即报错）——见
  [视频嵌入](../modules/video_embed.md) §5。

### 4.2 标签落表链路（索引收尾对齐）

运行收尾（同步 `_done` 与 SSE 流式两条路径同一钩子）自动执行，任一步失败不阻断运行主流程：

```
链路尾事件（timeline）→ 提取（只取链路尾事件；clip_indexed 与车牌提交不算）
   → 事件归档 out/events/<视频名>.jsonl（一行一次运行，按 run_id 幂等）

片段记录（indexed + observation 旁路）─┐
历史归档事件（该视频全量）─────────────┴→ label_align 契约 A（窗口对齐/聚合/组装标签行）
   → upsert review_index（片段标签行）+ review_video（视频级元数据）
   （幂等键 (video, t_start)；无 DB 降级同结构 JSON）
```

- **判断层只取链路尾事件**：车牌事件（kind=车牌，附 plate_no）路由进 `plates[]`；警情/
  研判事件进 `events[]`——三层分工（plate_nos / plates / events）见 hybrid_retrieval §2.5；
- **采集时间**：视频级一行（`video_time` 文件名优先解析，解析不出留空、不阻塞索引，
  hybrid_retrieval §2.6）；
- **relabel 只重对齐**：`POST /api/review/relabel` 复用同一对齐函数，按 `(video, t_start)`
  upsert 两表，**绝不动向量库、不重嵌入**（分析链路晚于建库时补标签即此通路）；旧行
  `frame_dets` 原样保留（观察层原始数据不随 relabel 重算）。窗口口径从底库各片段
  `frame_ts` 经 `clip_windows` 重建**无缝窗口**（t_end = 下一窗首帧，避免窗缝丢事件）。
- 表结构 / 读写语义 / 无 DB 降级见 [数据库结构](../guides/database_schema.md) §11（不重复）。

### 4.3 查询入口（不在链路内）

`POST /api/review/search`：文本查询 → clip 粗排（向量 Top-K）→ **过滤式**（有过滤器时
pre-filter 幸存集内排序：时间窗 / 事件名精确 / 视频；why 逐条注明命中来源，无采集时间的
视频不参与时间窗过滤并在 why 注明）→ 帧级精定位（frame 向量精排，播放起点 = 命中帧 ts −
pre_roll）。`mode=hybrid`（RRF 融合）为预留字段、本期同走过滤式；`include_frame_dets=true`
才附带观察层原始记录流（体积大，调试/可视化用）。两级索引精定位与检索漏斗原理见
[video_review_search.md](../design/video_review_search.md) §4；过滤器与 why 语义见
[hybrid_retrieval.md](../design/hybrid_retrieval.md) §5。

## 5. 运行方式与输出

**运行**：打开 `http://127.0.0.1:8001/test.html` → 视频测试页：选视频 + 选链路（PIPE0004）
→ 运行（表单参数同各链路通用：video / max_frames / start_time / end_time 等；本链路无
`vlm_feedback` 语义）。模块试跑页（video_embed）可对单视频小段快速索引试跑（取样上限
60 帧 ≈ 10 片段），不走链路播种形态。

**输出（落盘/落库产物）**：

- **向量底库** `out/video_vecs.npz`（+ 同名 .json meta：窗口骨架）——clip + frame 两级向量，
  同视频重跑 `(video, t_start)` 幂等覆盖；
- **缩略图** `out/video_thumbs/<视频名>_<t_start>.jpg`（项目相对路径进记录与标签行）；
- **标签表** `review_index` / `review_video`（无 DB → `out/review_index.json` /
  `out/review_video.json` 同结构降级）；
- **事件归档** `out/events/<视频名>.jsonl`（链路尾事件；本链路自身通常无分析事件——归档
  主要服务 PIPE0001/0002 等分析链路，PIPE0004 收尾把既有归档对齐落表）；
- **timeline**：逐片段 `clip_indexed` 行（入库回执：片段号/时间窗/帧数/缩略图），结果头含
  片段数、向量维度、mock/real 模式。

## 6. 验证

```bash
# 模块级（mock，秒级；client/store/stage/label_align）
conda run -n ai python umvp/video_embed/test_video_embed.py

# 链路级：注册/播种（4 阶段 + stage_inputs 连线 + 存量重建）/装配加载既有库/维度护栏
conda run -n ai python tests/test_video_embed_e2e.py        # mock；--live 走真端点

# 标签闭环：归档 → 对齐落表 → relabel 保留语义 → 过滤查询（mock + JSON 降级双路径）
conda run -n ai python tests/test_review_search.py
conda run -n ai python tests/test_review_search_e2e.py      # mock+HTTP；--live 同上
conda run -n ai python tests/test_hybrid_retrieval.py

# spec 校验（维度档位 / 端点配置 / inputs DAG）与端口连线端到端
conda run -n ai python web_lab/validate_chain.py <PIPE0004 spec.json>
conda run -n ai python tests/test_port_inputs.py
```

（断言数不写死，以脚本运行自报为准；完整命令总表见 `CLAUDE.md` §六。）

## 7. 边界与不做的事

- **检索不重跑视频**：查询只读索引（底库 + 标签表）；索引未覆盖的视频提示"未入库"；
- **不做语义档 / hybrid RRF 融合**（本期）：`description` 列留空、`mode` 字段预留
  （hybrid_retrieval §8 非目标）；
- **不做告警报送**：入库即目的；分析类事件由 PIPE0001/0002 等链路产出、经事件归档进标签；
- **不答帧内空间位置**：向量只答"哪段像"；检出框重放类需求靠 `frame_dets` 原始留存 +
  既有标注模块（将来方向，本期无入口）；
- **mock 档命中仅供流程演示**：伪向量无语义，真语义检索需配置真端点（`use_real_embed=true`
  + env）并重建索引。
