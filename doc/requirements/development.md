# 需求文档（开发版）—— 视觉识别链路定制系统

> 面向开发者：技术设计、代码引用、接口定义。配套通俗版见[《业务版》](business.md)。
> 状态（2026-10-16 更新）：P1 资源估算、P2 链路定制（web_lab）、P3 技术文档、P4 挂载 agent
> 均已落地（`.claude/agents/` 两角色提示词 + `assembler/` 组装工技术文档）。各模块参数清单
> 已移出本文，权威来源为 `doc/modules/`（每模块一篇）。

## 一、背景与目标

UMVP 现有零散模块已齐备，但缺少三层能力：

1. **知识层**：原 `.claude/skills/face-extract-pipeline/SKILL.md` 只记录单条人脸提取链路（WIP 状态），需扩为覆盖全部链路的 skill 手册。
2. **智能层**：无 agent。需要一个新的 pipeline 规划/生成/测试 agent，以 skill 文档为知识底座。
3. **交互层**：`web_lab` 只能单模块测试，无"链路定制 + 资源估算"能力。

用户明确点出的缺失能力：**完整链路的 GPU/内存占用自动计算或估计**。本文档把四件事（skill / agent / web 服务 / 资源估算）统一设计，资源估算是贯穿三层的公共能力。

## 二、现状盘点（可复用资产）

| 资产 | 位置 | 说明 |
|---|---|---|
| 管道模块 | `umvp/pipe/composer.py` | FrameManager / YOLODetector（纯推理）/ VLMAnalyzer / Pipeline 装配器（检测筛选 det_filter、时序判定 seq_filter 等阶段编排，事件尾输出） |
| 人脸三件套 | `umvp/face_detect/` `umvp/face_embed/` | FaceDetector(SCRFD) / FaceEmbedder(ArcFace 512d) / FaceStore |
| 人脸监控 | `umvp/face_scan/` | FaceMonitor（取帧复用 `pipe.composer.FrameManager`，含原抽帧调度背压） |
| 通用裁剪 + 目标裁剪 | `umvp/pipe/cropper.py` + `target_crop.py` | Cropper（按框裁图可缩放/坐标还原/原分辨率提取，人脸车牌链共用）/ crop_targets（按上游检测框裁子图，输出分辨率可配） |
| 测试台 | `web_lab/` | 零新增依赖（http.server + 原生 JS），多模块，`/api/schema` + `/api/test/<mod>`，全局 `_TEST_LOCK` 串行 |
| 资源估算 | `umvp/resources.py` | ResourceEstimator（静态公式 + 实测指纹两层） |
| 性能基线 | `out/fingerprints.json` | 实测指纹（`web_lab/calibrate.py` 产出）为唯一权威；换机重跑标定即可 |
| 回归集 | `tests/` + `umvp/pipe/test_composer.py` 等 | 断言式自检，断言数以各脚本运行自报为准；`tests/data/` 有测试图与视频 |
| agent 文档 | `.claude/agents/` | 两角色提示词（自包含）+ `assembler/` 组装工技术文档（链路壳标准/工作流/套路与扩展/坑/决策表）（见 §五） |

## 三、核心设计：资源估算（ResourceEstimator）

> 状态：✅ 已落地（`umvp/resources.py` + `web_lab/calibrate.py` → `out/fingerprints.json`）。

### 3.1 设计原则

- **静态法 + 实测法两层**。静态：不加载模型，秒级粗估；实测：真实推理标定，精确，结果缓存为 JSON。
- **统一资源指纹（ResourceFingerprint）**：每个模块暴露 `{vram_mb, ram_mb, latency_ms_cpu, latency_ms_gpu, fps@device}`。静态层按模型文件大小/参数量+输入分辨率估算；实测层跑基准写入指纹。
- **链路聚合**：链路 = 各模块按"每帧经过路径"聚合。注意部分模块**不是每帧都跑**（如 VLM 受 `check_interval` 节流、人脸嵌入只在检测命中后跑）——聚合必须按"每帧平均成本"而非简单相加。

### 3.2 数据结构

```python
# umvp/resources.py（已落地）
@dataclass
class ResourceFingerprint:
    vram_mb: float          # 显存峰值（含权重+激活+图）
    ram_mb: float
    latency_ms_cpu: float
    latency_ms_gpu: float
    note: str               # 条件说明（分辨率、batch、设备）

@dataclass
class PipelineEstimate:
    per_module: dict[str, ResourceFingerprint]
    vram_mb: float          # 链路显存 = 各模块峰值并集（GPU 上模型常驻取 max，激活取路径 max）
    ram_mb: float
    latency_per_frame_ms: dict[str, float]   # cpu / gpu
    fps_cpu: float; fps_gpu: float
    streams: int            # 多路外推
    vram_mb_N_streams: float
    note: str
```

### 3.3 估算规则（v1 简化，可后续加精）

- **YOLO**：显存 ≈ 权重(fp16) + 激活(分辨率相关经验公式) + 图；帧耗时按实测指纹表（分模型与 imgsz）。
- **SCRFD(FaceDetector)**：onnxruntime session 常驻；按 det_size 与实测（CPU ~100ms@640）估。
- **ArcFace(FaceEmbedder)**：~37ms CPU / 更短 GPU；批处理可并。
- **FaceStore**：内存 ≈ 向量数 × 512 × 4B（numpy float32）；检索 <1ms，可忽略。
- **VLM**：本地只做素材准备（裁剪/缩放，开销小）；真实推理耗时 = 远端延迟，估算标"依赖实测"。
- **帧管理（取帧+背压）/ 检测筛选 / 时序判定**：纯逻辑，开销可忽略（标记为 0）。
- **多路外推**：模型常驻内存部分不随路数增长（共享）；激活/图/推理并发部分随路数线性增长；超过显存给 WARNING 并建议 `frame_skip`。
- **预算反推**：给定 `vram_budget_mb`，求解满足 `estimate.streams == N` 且总占用 ≤ 预算的最大帧率/最小 frame_skip；输出建议 `frame_skip`、`check_interval` 与预期 FPS。

### 3.4 指纹标定命令（实测层）

```bash
PY="conda run -n ai python"
$PY web_lab/calibrate.py --out out/fingerprints.json   # 遍历 models/*.pt + face 模型，分别 CPU/GPU 实测一次
```

产物 `out/fingerprints.json` 由 ResourceEstimator 加载；机器变化（换卡）只需重跑标定，代码零改动。

## 四、核心设计：链路定制 web 服务（扩展 web_lab）

> 状态：✅ 已落地（web_lab 链路设计器 + 已完成链路 + chain 运行器）。

### 4.1 扩展方式

**不新建服务，扩展现有 web_lab**（零依赖架构已被验证）。实际落地的能力：

- **链路设计器**：侧栏入口，把模块排成阶段列表、设各阶段参数与资源预算，存成新链路（走 `/api/pipelines/save`）。
- **已完成链路**：侧栏底部按名运行（PIPE0001 警用无人机 / PIPE0002 车牌识别 / PIPE0003 人脸截取），运行表单按 `spec.run_module` 取 schema、按阶段展开编辑。
- **chain 运行器**（`/api/test/chain`）：按 `spec.stages` 组装模块并逐帧驱动；spec 顶层键 `runner="chain"` + `run_module="chain"`。链路由 pipelines/pipeline_steps 承载（见 `doc/guides/database_schema.md`）。
- **资源估算接口**：`/api/pipeline/estimate`（链路/多路/反推）。
- **pipeline spec（JSON）**：

```json
{
  "name": "巡检链路",
  "stages": [
    {"module_id": "frame_manager", "params": {"sampling": "wall_clock", "interval_sec": 1.0}},
    {"module_id": "yolo", "params": {"model_path": "yolov8n.pt", "classes": "0"}},
    {"module_id": "det_filter", "params": {"filter": "short>=24", "group_limits": "0:1,300"}},
    {"module_id": "seq_filter", "params": {"throttle": 30}}
  ],
  "streams": 16,
  "budget": {"vram_mb": 8192}
}
```

### 4.2 与现有 API 的关系

现有 `POST /api/test/<module>` 单模块测试保持不变（回归安全）。链路各阶段参数校验直接复用各模块 PARAMS 定义，保证"链路定制"与"单模块测试"参数口径一致。

## 五、核心设计：agent 技术文档

> 状态：✅ 已落地（2026-10 起为 `.claude/agents/`：两角色自包含提示词 + `assembler/` 组装工技术文档；
> 原 `.claude/skills/pipeline-assembly/` 技能库已清理移除，web agent 不再依赖）。

### 5.1 目标

从"单链路 WIP 记录"升级为"组装全链路所需手册"，作为 agent 的知识底座。延续诚实风格（"已验证 / 未验证"分列）。

### 5.2 落地结构

- `.claude/agents/pipeline-architect.md`（总装，自包含：项目 / 角色 / 机制与模块地图 / 工作流 / 产出格式 / 文件索引 / 工具）
- `.claude/agents/pipeline-assembler.md`（组装，自包含）+ `.claude/agents/assembler/`：
  - `01-shell-standard.md` 链路壳标准（唯一权威：spec / 硬规则 / 自检清单 / 红线）
  - `02-assembly-workflow.md` 工作流与交付
  - `03-patterns-and-extension.md` 拼装套路 + **新模块 / 新编排扩展**
  - `04-pitfalls-and-checklist.md` 坑与已验证 / 未验证
  - `05-decision-table.md` 参数决策表
- 动态事实（模块 / 参数 / 已存链路）走工具现取：`list_modules` / `validate_spec` / `estimate` / `list_pipelines`。

### 5.3 与 CLAUDE.md 分工

CLAUDE.md = 项目操作约定（环境、命令、协作）。agent 技术文档 = 组装业务知识。两者互补不重复。

## 六、核心设计：pipeline agent

> 状态：✅ 已落地（总装 `pipeline-architect` + 组装工 `pipeline-assembler` 自包含提示词 + `assembler/` 技术文档），见 `.claude/agents/`。

### 6.1 定位

- **总装（pipeline-architect）**：面向用户的接口人 + 编排者 + 质量门。受理分类、需求澄清、方案设计（闸门：用户确认）、任务拆解、质量门、交付汇总；自己不写壳。
- **组装工（pipeline-assembler）**：只接"组装单条链路"，估算 → 写壳 → 测试 全流程。
- **输入**：自然语言需求（设备约束、分辨率、路数、要识别的目标、告警需求）。
- **护栏**：改动任何 `umvp/` 模块代码前必须声明并单独批准；每次交付跑回归集确认不破坏现有链路。

### 6.2 agent 与技术文档

agent 用自包含提示词（`.claude/agents/*.md`）+ 项目内文档（`doc/` 与 `.claude/agents/assembler/`）+ 工具现取，不硬编码业务知识 → 提示词 / 技术文档 / 工具更新即 agent 能力更新。

## 七、分阶段实施计划

| 阶段 | 交付物 | 验收 | 现状 |
|---|---|---|---|
| P1 资源估算 | `umvp/resources.py` + `calibrate.py` + 单模块估算接入测试台 | 每模块表单下显示估算；`/api/pipeline/estimate` 对多路反推给出建议；标定一次后免重复实测 | ✅ 已落地 |
| P2 链路定制 | pipeline spec 校验 + estimate/run 接口 + 前端链路设计器 + chain 运行器 | 网页拼出链路 → 估算 → 生成/运行可跑通 | ✅ 已落地 |
| P3 文档完善 | 组装 agent 技术文档（`.claude/agents/`） | agent 读后能独立规划 | ✅ 已落地 |
| P4 agent | `.claude/agents/pipeline-architect.md` + `pipeline-assembler.md` | 自然语言需求 → 方案（含资源估算）→ 自动生成测试跑通 → 回归集全绿 | ✅ 已落地 |
| P5 拼接层端口协议 | 模块端口声明 + 壳内连线 + 校验器断言（分三步） | 声明的阶段必被装配执行；图像源（原帧/带框图/裁剪图…）可在壳内选择 | 📐 设计稿（[`../design/assembly_ports.md`](../design/assembly_ports.md)，WO-0002） |

## 八、验证与回归策略

- **回归集（每次交付必须全绿）**：
  ```bash
  PY="conda run -n ai python"
  $PY umvp/pipe/test_composer.py          # 管道纯逻辑（断言数以自报为准）
  $PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz
  $PY tests/test_yolo_params.py           # 真实推理
  $PY tests/test_police_uav_pipeline.py   # 警用链路逻辑断言
  ```
- **估算正确性验证**：静态法 vs 实测法在同一指纹 JSON 上对比，误差 > 30% 的条目标记警告并在文档记录原因。
- **前端回归**：现有模块单测接口全部保持 ok=true。

## 九、风险与未决问题

- **显存估算误差**：PyTorch 缓存分配器会让"实测峰值"大于公式值；以实测为准，公式只做粗筛。
- **多路并发估算**：v1 假设"模型常驻共享 + 激活线性"；实际多进程/多线程各有额外开销，需真机标定修正系数。
- **VLM 远端延迟**：无法本地估算，文档标注依赖实测。
- **agent 自主跑真实推理**：耗时（YOLO 秒级、链路分钟级），需在 agent 约束里设超时与限帧约定。
- **参数决策表**：已迁至 `.claude/agents/assembler/05-decision-table.md`，随模块演进持续维护。

## 十、文档结构

> 文档已按目录分类（2026-10-16 重整），**总览与文件索引见 [`doc/README.md`](../README.md)**：
> `requirements/`（需求）、`modules/`（模块分册，权威参数源）、`guides/`（操作/参考手册）、
> `design/`（专题设计）、`agent/`（挂载 agent 生成物）、`research/`（历史选型调研）。

> 模块参数维护规则：改参数先改 `web_lab/server.py` PARAMS（唯一权威源）→ 同步
> `doc/modules/` 对应分册 → 若涉及资源指纹同步 `out/fingerprints.json` 标定。

## 十一、下一阶段需求：拼接层端口协议（P5）

> 状态：📐 需求已确认（2026-10-16），分步实施；详细设计见
> [`../design/assembly_ports.md`](../design/assembly_ports.md)，实施工单 **WO-0002**。

**背景（触发问题）**：PIPE0003 在测试台走专用单帧路径（`_face_step`）绕开通用 `Pipeline.step`，
导致链路 spec 里声明的 `det_filter` **静默不执行**（详见 `../chains/pipe0003_face_capture.md`
§4.1 与 WO-0001）。根因是「执行路径双轨 + 阶段连线硬编码」。

**两条用户诉求**：

1. **模块可复用**：模块不应被位置绑死（现在 crop 只能吃 dets、plate/face 只能吃 crops）；
2. **壳内自由取数**：拼接层能规定模块从上游取哪部分——图像旁路里同时可能有原帧 / 带框图 /
   低清图 / 裁剪图，接收模块（如 VLM）在**链路壳里**选择其一。

**目标与规矩（确认结论）**：

- 所有**逐帧分析阶段**一律走通用插槽（`Pipeline.step`），**禁止**驱动层专用单帧路径；
  此规矩写入链路壳标准（`.claude/agents/assembler/01-shell-standard.md`，唯一权威）。
- `validate_chain` 增加断言：**spec 声明的阶段必须全部被装配进执行路径**，否则报错。
- 方案 = 产物端口协议（产物命名 / 模块自描述 / spec 声明 `inputs` / 校验 / 按表驱动），
  缺省规则 = 现状（零回归）。

**分步实施**：第 0 步消除例外（人脸并入 step + 断言 + 壳标准写规矩）→ 第 1 步旁路命名统一 +
图像源选择（`image_source`）→ 第 2 步完整端口协议 + 设计器连线 UI。

**设计决策（2026-10-16 确认）**：端口类型 `Image/Records/Vector` 三类够用；**每端口单一上游**
（`inputs` 不允许多值）；协议须支持「同一产物被多个下游消费」（**分支链路列入后续路线**，
本版仍线性执行）；**低清图暂不设计**。

**边界**：落盘/展示/批量编排等**非逐帧**逻辑不进 Pipeline；本版不做通用 DAG 执行器。
