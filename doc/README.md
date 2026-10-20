# UMVP 文档总览（doc/README）

> 本文件是 `doc/` 的总入口：**业务架构 → 技术结构 → 目录分类与文件索引**。
> 找文档先看这里；发现索引与实际不符时以文件为准，并顺手更新本页。
> 最后更新：2026-10-16（文档目录重整：分类入目录 + 文件名英文小写下划线，模块分册按数据流角色分组）。

---

## 一、业务架构（这个系统做什么）

**UMVP（统一模块化视觉分析管道）**：对视频/图片做智能视觉分析——按节奏抽帧 →
小模型（YOLO）识别目标 →（可选）大模型（VLM）语义研判 → 事件尾输出
（事件 / 记录 / 素材留存）。**链路由模块自定义拼接**，不预设固定模式。

### 1.1 三条保留链路（开箱即用）

| 链路 | 业务目的 | 流程 | 专篇 |
|---|---|---|---|
| PIPE0001 警用无人机链路 | 无人机画面警情监测 | 帧管理 → YOLO 识人车 → **检测筛选** → 标注 → VLM 研判警情 → **时序判定事件尾** | `chains/pipe0001_police_uav.md` |
| PIPE0002 车牌识别链路 | 车辆车牌识别与存证 | 帧管理 → YOLO 识车 → **检测筛选** → **目标裁剪(裁车)** → 车牌识别（裁车牌→识别→坐标还原→原分辨率提取车牌小图）→ **时序判定事件尾** | `chains/pipe0002_plate.md` |
| PIPE0003 人脸截取链路 | 原分辨率人脸截取归档 | 帧管理 → YOLO 识人 → **检测筛选** → **目标裁剪(裁人)** → 人脸检测（检脸+原图出脸）（无嵌入/检索/事件，截取落盘即目的） | `chains/pipe0003_face_capture.md` |

### 1.2 平台通用能力

- **链路设计器**：在 web 测试台把模块排成阶段、逐阶段调参、存成新链路并运行
- **资源估算（P1）**：每模块/整链路的显存/内存/耗时估算（静态公式 + 实测指纹两层），
  支持多路外推与预算反推
- **可视化测试台**：单模块测试页 + 链路批量运行 + **实时播放叠加**（任务进行时同步播放
  视频，检测/车牌框实时叠在画面上；HEVC 等浏览器不支持的编码按需自动转 H.264 后播放）
- **AI 组装（P4）**：总装/组装工两个 agent + 组装工技术文档，自然语言需求 →
  方案（含资源估算）→ 自动测试

### 1.3 统一输出抽象（业务语义）

多帧视觉分析的输出 = **检测筛选/时序判定等模块往记录上追加的事件（`events`）**——
"过滤后幸存的记录本身就是告警载体，其携带的 events 即产出"。词表/置信/窗口/冷却/
同对象一次由筛选模块的条件与规则表达（如 `label in 治安类|交警类` + 冷却 30s），
详见 `modules/det_filter.md` §2.1、`modules/seq_filter.md`。原「输出处理」七道闸已弃用。

---

## 二、技术结构（怎么实现的）

**技术栈**：Python 3.10（conda 环境 `ai`）＋ ultralytics(YOLO) ＋ insightface(SCRFD/ArcFace)
＋ HyperLPR3(车牌) ＋ onnxruntime/cv2/numpy ＋ stdlib http.server（web 测试台零框架）＋
本地 MySQL（自含 3307，presets/pipelines 暂存）。代码与权重自包含于项目目录，整目录拷贝即可移植。

### 2.1 分层结构

```
web_lab/（可视化测试台）
  ├ index.html 模块测试/链路设计器/数据浏览   ├ db.py MySQL 配置暂存（惰性连接）
  ├ test.html 链路测试台（批量运行 + SSE 同步播放叠加）
  └ server.py 全部 API + 链路运行器（_chain_from_spec 装配 → Pipeline 驱动）
      │
 umvp/pipe/（管道核心）
 ├ composer.py  FrameManager 帧管理(取帧节奏+背压) / YOLODetector 小模型**纯推理** /
 │              VLMAnalyzer 大模型研判 / Pipeline 装配器
 │              （有效性过滤→det_filter，时序门控/事件尾→seq_filter）
 ├ cropper.py   通用裁剪工具 Cropper（按框裁图可缩放/坐标还原/原分辨率提取，纯 cv2+numpy）
 ├ target_crop.py  目标裁剪模块（按上游检测框裁子图，输出分辨率可配；车牌/人脸/VLM 链共用）
 ├ annotate.py  标注模块（在帧上画检测框，供 VLM 看"原图+画框"；纯绘制）
 ├ det_filter.py 检测筛选 DetFilter（有效性：逐条条件 DSL + 成组数量判定）
 ├ seq_filter.py 时序判定 SeqFilter（窗口确认闸 + 事件标注）
 ├ conditions.py 条件 DSL 引擎（与/或/非/括号 + 派生尺寸量，筛选/裁剪/标注共用）
 └ records.py   通用记录 Record 约定 + to_record 对象适配
      │
 专用模块
 ├ umvp/plate_recog/   PlateRecognizer(HyperLPR3) + PlateStage(链路阶段) + extract_plates
 ├ umvp/face_detect/   FaceDetector(SCRFD+5点对齐) + FaceStage(检脸+原图出脸) + providers.py
 ├ umvp/face_embed/    FaceEmbedder(ArcFace 512d) + FaceStore(numpy 线性检索底库)
 └ umvp/resources.py   ResourceEstimator 资源估算（静态公式+实测指纹）
      │
 模型/数据：models/(YOLO 权重池) face_models/(buffalo_l) plate_models/(hyperlpr3)
           out/fingerprints.json(实测指纹)  本地 MySQL(presets/pipelines)
```

### 2.2 关键机制速览

| 机制 | 一句话 | 权威文档 |
|---|---|---|
| 帧管理门控 | frame_skip/墙钟/帧数三式取帧 + EMA 背压自适应放宽 | `modules/frame_manager.md` |
| IntervalGate 统一节流 | yolo→vlm 报送 / yolo→车牌车辆框共用同一段间隔门控（key=类别/track） | `umvp/pipe/composer.py`（IntervalGate） |
| 时序门控 + 事件尾 | 整类节流(原 check_interval)/track 变化(原 track_change_only) + 窗口确认 + 事件标注（`events`） | `modules/seq_filter.md` |
| 目标裁剪 + 原分辨率提取 | 按 YOLO 框裁子图（分辨率可配）→识别模块裁二次目标→坐标还原→从原图直接裁出小图（零缩放） | `modules/target_crop.md`、`modules/plate_recog.md` |
| track 去重 | 车牌按 track_id 只识别一次（冷却可重试）；输出按车牌号只报一次 | `modules/plate_recog.md` §5 |
| 实时播放叠加 | SSE 逐分析帧推框数值，浏览器播视频 + canvas 叠加，播放节流保同步 | `design/live_overlay.md` |
| 参数入库 | 每模块一张参数表，schema 双向收敛（加参数补列/删参数删列），test_only 不落库 | `guides/database_schema.md` §6 |
| 部署台账 | 影响部署的改动只追加记录于 `DEPLOYMENT.md` | 根目录 `DEPLOYMENT.md` |

---

## 三、文档目录分类与索引

> 目录约定（2026-10-16 起）：**目录与文件均英文小写＋下划线**（模块篇文件名 = `module_id`，
> 链路篇 = `pipe编号_英文slug`），与代码标识/PARAMS/DB 一致便于代理索引；各篇 H1 保留中文标题；
> 仅 `README.md` / `INDEX.md` 用大写约定名。

```
doc/
├── README.md        本页：总览（业务架构/技术结构/索引）
├── requirements/    需求（business=业务通俗版 / development=技术开发版，状态只在开发版维护）
├── modules/         模块分册（每模块一篇=module_id，权威参数/输入/输出；入口 INDEX.md，
│                    按「JSON 主干 × 图像/向量旁路」的数据流角色分组）
├── chains/          保留链路专篇（pipe0001_police_uav / pipe0002_plate / pipe0003_face_capture，
│                    六段式：定位→拼接结构→数据流→运行方式→输出→验证）
├── guides/          操作与参考手册（runbook=端口与启停 / database_schema=数据库结构 /
│                    ticketd_service=工单看板常驻服务与多项目复用）
├── design/          专题设计（storage_design / live_overlay＝web 已落地；
│                    video_review_search / wemm_integration＝下期设计稿；
│                    assembly_ports＝拼接层端口协议设计稿，P5；
│                    graph_chain＝图编排设计稿，P2）
├── agent/           挂载 agent 生成物（static_prompts / full_run_record；脚本生成勿手改）
└── research/        历史选型调研（face_search_survey / plate_recog_survey；实现现状以 modules/ 为准）
```

**维护归属**（多代理协作）：`modules/` + `chains/` → pipe-chain；`guides/` + `design/` 前两篇 → web；
`agent/` → agent-builder；`requirements/` + `design/` 后三篇 + `research/` + 本页 → orchestrator。

### 3.1 快速上手

| 文件 | 内容 |
|---|---|
| `../README.md`（项目根） | 项目速览 + 常用验证命令 |
| `../INSTALL.md` | 安装/移植（场景 A 本机继承 / 场景 B 新机从零） |
| `guides/runbook.md` | 启停脚本、端口、web 页面使用（含 §4.1 实时播放叠加）、FAQ |

### 3.2 需求（requirements/）

| 文件 | 内容 |
|---|---|
| `requirements/business.md` | 业务语言的需求/能力/读者（通俗，不贴代码） |
| `requirements/development.md` | 技术版：现状盘点、资源估算设计（P1）、里程碑状态（权威） |

### 3.3 模块分册（modules/，权威参数/输入/输出说明）

入口 `modules/INDEX.md`，按**数据流角色**分组（与链路/业务域无关）：**通用语先读**
（JSON 主干 + 图像/向量旁路契约，records.py）→ **主干生产者**（frame_manager / yolo）→
**旁路加工与识别**（target_crop / annotate / vlm / face_detect / plate_recog）→
**主干筛选与事件**（det_filter / seq_filter）→ **向量旁路**（face_embed / face_store）→
**通用契约与工具**（records / conditions / cropper）。INDEX 附「两平面角色表」
（每模块对 JSON 主干 / 图像旁路 / 向量旁路的接·产·改）。

### 3.4 保留链路（chains/）

| 文件 | 链路 |
|---|---|
| `chains/pipe0001_police_uav.md` | PIPE0001 警用无人机（YOLO→筛选→标注→VLM→事件尾） |
| `chains/pipe0002_plate.md` | PIPE0002 车牌识别（裁车→识牌→坐标还原→出牌→事件尾） |
| `chains/pipe0003_face_capture.md` | PIPE0003 人脸截取（裁人→检脸→原图出原分辨率人脸） |

### 3.5 手册与专题设计（guides/ design/）

| 文件 | 内容 |
|---|---|
| `guides/runbook.md` | 端口、启停命令、web 页面使用、FAQ（操作手册） |
| `guides/database_schema.md` | 表结构/业务 ID 前缀/播种清单/参数表机制/数据快照（参考手册；设计动机见 design/storage_design.md） |
| `guides/ticketd_service.md` | 工单看板常驻服务（systemd user）：多项目复用 / 配置字段 / 常用命令 / 收件箱/部署台账/模型页签 + ledger CLI + agent_model CLI / WSL 注意 / 换机重建 / FAQ |
| `design/storage_design.md` | presets/pipelines 存储设计决策记录（已落地） |
| `design/live_overlay.md` | SSE 流式运行 + 视频叠加框的方案与约束（已落地） |
| `design/video_review_search.md` | 语言询问→定位历史视频片段：离线片段索引（轻/语义/向量三档，PIPE0004）+ 三级检索 + 帧级精定位（设计稿，业务与实现方案总篇） |
| `design/wemm_integration.md` | WeMM-Embedding 嵌入服务对接专篇：模型机制、serving 部署、WEMMClient 请求 schema、成本与联调清单（设计稿） |
| `design/assembly_ports.md` | 拼接层端口协议（P5 设计稿）：产物命名 / 模块自描述输入输出 / 壳内 `inputs` 连线 / 校验断言；把「阶段插槽硬编码连线」升级为「模块自描述端口 + 壳内声明」；分三步实施（WO-0002） |
| `design/graph_chain.md` | 图编排（P2 设计稿）：节点实例身份（产物全名 `{模块}#{位置}.{产物}`）/ 同模块复用 / `pipeline_edges`+`module_ports` 两表 / 迁移与双写；P2-0+P2-1 实施中（WO-0008），拓扑驱动执行与过渡节点（merge/join/to_images/split）为其后续期 |

### 3.6 挂载 agent（agent/ 与 .claude/agents/）

| 文件 | 内容 |
|---|---|
| `../CLAUDE.md` | 项目结构/关键约定/**常用验证命令（断言数权威）**/git 协作 |
| `../.claude/agents/` | 挂载 agent 两角色提示词（总装/组装，自包含）+ `assembler/` 组装工技术文档（链路壳标准唯一权威/工作流/套路与扩展/坑/决策表） |
| `agent/static_prompts.md` | 两角色运行时静态 system 提示词快照（`web_lab/dump_prompts.py` 生成，勿手改） |
| `agent/full_run_record.md` | 一次「需求→总装工→组装工→链路 spec」的完整示例记录（`web_lab/agent_run.py` 生成） |

### 3.7 历史调研与台账（research/ 等）

| 文件 | 内容 |
|---|---|
| `research/face_search_survey.md`、`research/plate_recog_survey.md` | 选型期调研（历史，实现现状以 `modules/` 分册为准） |
| `../DEPLOYMENT.md` | 部署变更台账（只追加） |

---

## 四、按读者场景导航

- **我想了解这套系统是干嘛的** → 本文 §一 + `requirements/business.md`
- **我想把它跑起来/移植到新机器** → `../INSTALL.md` → `guides/runbook.md`
- **我想查某个模块的参数/输入输出** → `modules/INDEX.md`（数据流角色分组）→ 对应分册
- **我想跑/理解三条保留链路** → `chains/pipe000X_*.md`（六段式专篇）
- **我想拼一条新链路 / 改链路** → 挂载 agent 两角色提示词与组装工技术文档（`.claude/agents/`，含链路壳标准）+ `guides/database_schema.md`
- **我想理解某个行为的实现** → 本文 §二机制表 → 对应专题设计 / 模块分册
- **我要改代码并验证** → `../CLAUDE.md` 常用验证命令 + 改完按 `DEPLOYMENT.md` 规则记台账
