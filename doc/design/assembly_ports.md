# 拼接层端口协议（设计稿）

> 状态：✅ **三步全部实施（2026-10-16）**：第 0 步 WO-0002.1、第 1 步 WO-0002.2、第 2 步 WO-0002.3。
> ✅ **WO-0021（2026-10-23）增补**：§9 标注双模式（scope，已实施）、§10 帧级数据通用约定
> （只约定，暂无生产者）、§11 VLM 图像源统一进端口（只定案，实施待启动）。
> 后续跟进：✅ **WO-0003（2026-10-16）图集与送审**——`records.image_set_of`/`image_of`、
> `target_crop.crops` 记录化、`plate_recog`/`face_detect`/`vlm.dets` 纳入 `inputs` 解析、
> `vlm.image_pick` 多图送审（见 §3⑥）。需求来源：
> `../requirements/development.md`「P5」；实施工单 **WO-0002**（分步）。用户已确认：
> ①「逐帧阶段必须走通用插槽 + validate_chain 断言」写入链路壳标准；②**分步实施**。
>
> 结论先行：把当前的「阶段插槽 + 硬编码连线」升级为「**模块自描述端口 + 壳内声明连线**」——
> 模块声明自己消费/产出哪些产物（原帧 / 带框图 / 裁剪图 / 记录 / 向量…），拼接壳（spec）
> 显式指定取哪一个；缺省保持现状规则，零回归。分三步走（第 0 步消例外 → 第 1 步图像源选择 →
> 第 2 步完整端口协议）。

## 1. 现状与缺口

### 1.1 执行路径不统一（**已于 2026-10-16 修复**，见 §4 第 0 步）

- 通用路径：`umvp/pipe/composer.py Pipeline.step()` 逐帧驱动各阶段插槽；
- 曾经的例外：`web_lab/server.py h_pipeline_run` 的 `_face_step`（`is_face_chain` 门控）
  自行串 `yolo → target_crop → face_detect`，**绕开 Pipeline**（**已删除**）；
- 后果与修复：spec 里声明的 `det_filter`（DF0003）在测试台人脸链**静默不执行**
  （`../chains/pipe0003_face_capture.md`「历史差异与修复」、工单 WO-0001 → WO-0002.1）。

### 1.2 连线硬编码（模块位置绑定）

`Pipeline.step()` 里各阶段的输入是写死的调用顺序与形参：

| 阶段 | 现在吃什么 | 代码位置 |
|---|---|---|
| det_filter | dets | `step()` |
| seq_filter | dets | `step()` |
| crop（目标裁剪） | `(frame, dets, ts)` | `step()` |
| annotate（标注） | `(frame, dets, ts)` | `step()` |
| plate（车牌） | `(frame, last_crops, ts)` | `step()` |
| face（人脸，**已并入插槽**） | `(frame, last_crops, ts)` | `step()`（与 plate 对称） |
| vlm（研判） | 图像源由 `image_source` 参数选（annotated/frame/crops，默认 annotated） | 驱动层 `_pick_vlm_images` |

> **2026-10-16 后**：上表各阶段的输入已可由 spec 的 `inputs` 声明改写（`Pipeline._port()`
> 按 `artifacts` 解析；未声明＝现状默认）。`vlm` 的图像源仍统一走 `image_source` 参数，不经 `inputs.image`。

→ 模块**换不了位置**，也接不了不同上游；VLM 想用原帧/裁剪图而不是带框图，spec 里无处声明
（`vlm` 参数里的 `ref` 只表"素材引用/粒度"，不表"哪张图"）。

> **本节是改造前的现状（历史）**：上述缺口已由第 1 / 2 步实施解决——`inputs` 连线可换上游、
> `image_source` 选 VLM 送审图（见 §4）。

### 1.3 旁路产物命名不统一

`umvp/pipe/records.py` 只约定两个旁路名：`image`（ndarray 子图/帧）、`embedding`（向量）；
实际各模块自造：crop 用 `img`、face 用 `face_img`、plate 用 `plate_img`、annotate 返回帧副本
（非记录）。没有"产物"的统一命名与登记，下游无法通用消费。

> （历史缺口）已由第 1 / 2 步解决：`to_record` 统一补旁路键 `image`（旧键保留为兼容别名），
> 产物逐帧登记在 `Pipeline.artifacts`（见 §3① / §4）。

### 1.4 其他缺口

- 校验器（`web_lab/validate_chain.py`）只查阶段是否存在、终段是否合法，**查不了连线**；
- "低清图"这类产物目前**不存在**（没有任何环节产出），只在设计里被需要；
- 新增模块要改 `Pipeline` 核心（加插槽字段 + 在 `step()` 里加一段），接入成本与核心耦合。

## 2. 目标与原则

1. **一条执行路径**：所有逐帧分析阶段一律经 `Pipeline.step` 插槽执行，**不允许**驱动层再写专用单帧路径。
2. **模块自描述**：模块声明自己的输入/输出端口；装配器按声明连线。
3. **壳内可声明**：spec 的每个阶段可显式指定"从上游取哪部分"（如图像源）；**缺省 = 现状规则**。
4. **声明即执行**：`validate_chain` 增加断言——spec 声明的每个阶段必须被装配进执行路径，否则报错
   （这条断言正是本次事故的回归测试）。
5. **核心零依赖**：`Pipeline` 不引入 cv2/onnx 依赖（保持"鸭子类型 + 拼接层注入"）。

## 3. 方案：产物端口协议（五要素）

### ① 产物命名（Artifact）

`{阶段标识}.{产物名}`，逐帧登记；值分三类：

| 类型 | 例 | 说明 |
|---|---|---|
| 记录列表（JSON 主干） | `yolo.dets`、`face_detect.faces` | dict 记录，进 JSON |
| 图像（旁路） | `frame_manager.frame`、`annotate.annotated`、`target_crop.crops[i].image` | ndarray，不进 JSON |
| 向量（旁路） | `face_embed.embeddings` | ndarray |

> **检测产物形态（2026-10-21 定案，WO-0014.3）**：`yolo.dets` 及其下游检测记录是**原生
> dict Record**——`YOLODetector.infer()` 直接产 `{"kind": "det", "track_id", "cls_id",
> "bbox", "score", "model_path"}`，**不含时间上下文**（infer 内无帧上下文），由
> `Pipeline.step` 在登记产物前统一注入 `ts`/`frame_idx`（契约「时间上下文由调用方注入」，
> 见 `pipe.records`）。`Detection` dataclass **保留**供外部构造 / 直传兼容：直传入口
> （`Pipeline.step(dets=...)`）经 `to_record` 逐条归一为 dict 后进链。

（低清图暂无生产者且本阶段**不设计**——用户 2026-10-16：暂不管，见 §8。）

### ② 模块自描述（输入/输出声明）

```python
class Annotator:
    INPUTS  = {"image": "Image", "dets": "Records"}
    OUTPUTS = {"annotated": "Image"}
```

现阶段可先只写声明（供校验/设计器读取），执行仍按鸭子类型 `run(...)`；后续可让装配器按声明
自动适配实参。

### ③ spec 声明连线

```json
{"module_id": "annotate", "params": {...},
 "inputs": {"image": "frame_manager.frame", "dets": "det_filter.dets"}}
```

未声明 `inputs` 时按**现状默认规则**（yolo.image←本帧原帧；det_filter/seq_filter←当前检测流；
target_crop/annotate←frame+dets；plate_recog/face_detect←frame+`last_crops`）→ 零回归。
注意：**`vlm.inputs` 只接受 `dets`**（决定"报送哪些检测"）；VLM 的**图像源用参数 `image_source`**
（annotated/frame/crops），`vlm.inputs.image` 被校验器报错拒绝（避免双机制）。

### ④ 校验（validate_chain）

- 引用的产物存在且在其上游（不能引用下游/自身）；
- **每个端口只接一个上游产物**（`inputs` 不做多值/列表；阶段可有多个具名端口，各自单源）；
- 端口类型匹配（Image 端口不得喂 Records）；
- 无环；**同一产物可被多个下游引用**（为将来的分支链路留路，见 §8）；
- **spec 声明的阶段必须被装配执行**（第 0 步先落这条）。

### ⑤ 驱动（Pipeline.step）

从"按槽位写死实参"改为"按 spec 查产物表解析实参"：每帧维护 `artifacts: {name: value}`，
阶段执行前解析 `inputs`，执行后登记 `outputs`。

### ⑥ 基数约定（记录 ↔ 图像，2026-10-16 用户确认）

**记录数不随图片数膨胀**，规则三条：

1. **识别模块产出**：每个处理帧多条记录，**一条记录可含多个识别对象**（现状保持不变）；
2. **图像生产模块**：当**一条记录**需要产出多张图时，**不把记录拆成图片数量**，而是把**图集**
   放进该记录的旁路图像字段（`image` ＝单张 ndarray 或 ndarray 列表＝"这条记录的图片集合"）；
3. **下游是否接受集合由下游自行处理**（单图/图集各自兜底）。

推论与现状对照：

| 场景 | 现行 | 是否符合约定 |
|---|---|---|
| YOLO 逐框检测记录（dict，`kind=det`；`Detection` dataclass 保留供外部构造/直传） | 一个目标一条记录 | ✅（一条记录一个对象，无需图集） |
| 目标裁剪 `CropResult` | **每个对象一条记录**（各带一张 `image` 子图） | ✅（属逐对象，**不是"一条记录多图"**，不需要图集） |
| 标注 `Annotator`（VLM 上游） | 一条记录 → **一张图**，多框画在同一张 | ✅ 不受影响 |
| VLM 按类报送（一条记录含 N 个对象） | `image_source=crops` 收集该类**全部命中对象**的子图 | ✅ **已改（WO-0003）**：按记录的图像旁路＝**图集**（N 张子图）；送审端按 `image_pick` 决定送 1 张（`first`，默认＝逐字等价）还是 N 张（`all`，多图 payload） |

> **两种形态务必分清**：`target_crop.crops` 是**逐对象一条记录**（记录数＝对象数，每条一张图）；
> **图集**只用于"**一条记录确需多图**"的场景（如 VLM 按类报送时该类全部命中子图）。把前者当图集
> （或反之）都会与基数约定冲突。
>
> 落地含义：`image_source=crops` 在按类报送下产出**该类全部对象的子图集合**（图集），
> 而非"首个命中"；送审端可单图（取集合首张，现状语义）或图集（多图 payload），
> 由下游决定。**✅ 已落地（2026-10-16，WO-0003）**：`records.image_set_of` / `image_of`、
> `artifacts["target_crop.crops"]` 记录化、`target_crop.as_crop` 归一、`plate_recog`/`face_detect`
> 纳入 `inputs`、`vlm.image_pick`（first 默认/all 图集）与 `vlm.inputs.dets`；整单 qa 验收待办。

## 4. 分步实施（每步独立可交付、可回归）

### 第 0 步：消除最后的例外（前置，WO-0002.1）— ✅ 已实施 2026-10-16

- ✅ `face` 并入 `Pipeline` 插槽（与 `plate` 对称：`run(frame, crops, ts)` + `last_faces`/`last_face_input`）；
- ✅ 拼接层按 `face_detect` 阶段构造 `FaceStage` 注入；`h_pipeline_run` 删 `is_face_chain`/`_face_step`，
  改统一 `step` 驱动 + 从 `last_faces` 出图/计数/timeline（落盘命名、images≤36、`max_saved_images` 200
  等展示逻辑仍在驱动层）；`face_embed` 身份去重在驱动层处理；
- ✅ `validate_chain` 新增「**声明即执行**」断言（`CHAIN_ASSEMBLED_MODULES` 未登记即报错）；
- ✅ 链路壳标准（`.claude/agents/assembler/01-shell-standard.md` §0）写入「逐帧阶段一律走插槽，
  禁止驱动层专用单帧路径」，并附本案为反例警示；
- 验证：`test_face_capture_pipeline` **63**（39→63，含 det_filter 生效断言）、`test_validate_chain` **13/13**
  （含「声明但未装配 → 报错」用例）、`test_composer` 36、`test_plate_recog_pipeline` 28、
  `test_police_uav_pipeline`、`test_agent_runtime` 39 全绿。

### 第 1 步：旁路命名统一 + 图像源选择（WO-0002.2）— ✅ 已实施 2026-10-16

- ✅ **旁路命名统一**（`records.py`）：记录旁路图像统一键 **`image`**（`to_record` 自动补；`img`/`face_img`/`plate_img` 保留兼容别名），新增读取器 `image_of(rec)`；各模块对象属性不变；
- ✅ **`vlm.image_source`**（`annotated`（默认＝现状）/ `frame` / `crops`）：驱动层按值取图，缺源一律回退并在 timeline 标注素材来源；`validate_chain` 非法取值报错、上游缺 `annotate`/`target_crop` 给非致命警告；
- ✅ **补漏**：`composer.step()` 原先「存在 `target_crop` 即不产生送审」，改为按「是否有 `vlm`」判定——`target_crop + vlm` 组合由此前**静默失效**变为可用（`crops` 源端到端成立）；
- **验证**：新增 `tests/test_image_source.py` **59** 项（三取值/回退/records 键统一/crop+vlm 端到端/驱动层）；`test_validate_chain` **20/20**；
  `test_face_capture_pipeline` 63、`test_composer` 36、`test_plate_recog_pipeline` 28、`test_police_uav_pipeline`、
  `test_agent_runtime` 39、`test_cropper` 26、`test_target_crop` 35、`test_annotate` 16、
  `test_det_filter` 66、`test_seq_filter` 43 全绿。

### 第 2 步：完整端口协议（WO-0002.3）— ✅ 已实施 2026-10-16

- ✅ **产物表**：`Pipeline.artifacts` 每帧登记 `frame_manager.frame` / `yolo.dets` / `det_filter.dets` /
  `seq_filter.dets` / `target_crop.crops` / `annotate.annotated` / `face_detect.faces` / `plate_recog.plates`；
- ✅ **模块自描述**：9 个模块声明 `INPUTS` / `OUTPUTS`（类型 `Image`/`Records`/`Vector`）；
- ✅ **`inputs` 连线**：spec 阶段可选 `"inputs": {端口: 上游产物名}`，缺省＝现状（零回归）；非法连线
  fail fast（`PortResolutionError`，含阶段/端口/产物/可用清单）；第 2 步解析范围为
  yolo/det_filter/seq_filter/target_crop/annotate 五条端口（`plate_recog`/`face_detect`/`vlm`
  当时未纳入，由 WO-0003 补齐，见下）；
- ✅ **校验器 DAG**：未知产物 / 非法端口 / 非上游或自引用 / 类型不符 / 值非字符串 → 错；同一产物多下游 → 允许；
- ✅ **设计器连线 UI**：每阶段按可连线端口渲染下拉（上游产物 + 类型标注 + 「默认（现状）」），
  存 `meta.stage_inputs`（`pipeline_steps` 表结构未变，旧链路无键保持原样）；
- ✅ **接入即用**：新模块「插槽 + 声明 INPUTS/OUTPUTS + 登记 `CHAIN_ASSEMBLED_MODULES`」后装配器/
  设计器即可用（是否纳入 `inputs` 解析为可选第 4 件）；
- **验证**：`test_composer` 36→**59**（产物表/连线/非法连线）、`test_validate_chain` 20→**30**、
  新增 `tests/test_port_inputs.py` **39**（含端到端间谍证据：`annotate.dets=yolo.dets` 与
  `=det_filter.dets` 实参确实不同）、`test_image_source` 59、`test_face_capture_pipeline` 63、
  `test_plate_recog_pipeline` 28、`test_police_uav_pipeline`、其它模块自检全绿；CLI 非法 spec `exit=1`；
- **本步未覆盖**（遗留，非阻塞）：真实 MySQL 往返、浏览器内设计器交互、低清图。
  （`vlm`/`plate_recog`/`face_detect` 的 `inputs` 接线依赖 crops 产物形态统一，**已由 WO-0003 补齐**：`plate_recog.frame/crops`、`face_detect.frame/crops`、`vlm.dets` 全部纳入解析，唯一来源见 `server.CHAIN_RESOLVED_PORTS`。）

### 第 3 步（延伸）：图集与送审（WO-0003）— ✅ 已实施 2026-10-16

- ✅ **图集契约**（`records.py`）：旁路 `image` 可为单图 ndarray **或图集 list[ndarray]**（不拆记录）；
  读取器 **`image_of`＝单图语义（图集取首张）**、**`image_set_of`＝集合语义（单图→长度 1 列表）**；
  旧别名 `img`/`face_img`/`plate_img` 兼容不变；
- ✅ **`target_crop.crops` 记录化**：产物为**记录 dict 列表**（逐对象一条、各带一张 `image`），
  `target_crop.as_crop` 对"对象或记录"做类型归一（下游消费两种形态逐字等价）；
- ✅ **纳入 `inputs` 解析**：`plate_recog.frame/crops`、`face_detect.frame/crops`、`vlm.dets`
  （驱动层 `_apply_vlm_inputs`）；
- ✅ **`vlm.image_pick`**：`first`（默认＝只送首张，逐字等价）/ `all`（送图集全部，OpenAI 兼容多图 payload）；
- **验证**：`tests/test_image_source.py` 59（后增至 73）、`tests/test_port_inputs.py` 39（后增至 49）、
  `test_composer` 59→85、`test_validate_chain` 30→34；详见各步验证与工单 WO-0003。

## 5. 兼容与迁移

- 旧 spec / 旧 stage params 不改即可运行（缺省规则 = 现状）；
- `Pipeline` 保留现字段名（`crop/annotate/plate/...`）作为缺省映射，新协议是**增量**；
- 模块兼容别名（`CropRestore`→`Cropper` 之类）策略沿用。

## 6. 边界与不做

- **不进 Pipeline 的**：落盘/展示/计数（驱动层）、视频扫描与批量编排、底库批量事务、测试台交互；
- 本版**不做**通用 DAG 执行器：阶段仍为线性序列，端口只解"取上游哪部分"；但**协议不堵分支**——
  同一产物允许被多个下游消费（分支拓扑列入后续路线，见 §8）；
- 低清图**本阶段不设计生产者**（用户 2026-10-16：暂不管）。

## 7. 验证方法（断言式）

| 层 | 验证 |
|---|---|
| 校验器 | `validate_chain.py` 新增断言（声明阶段必被装配 / 端口存在 / 类型匹配 / 无环）逐条有对应用例 |
| 三条保留链路 | PIPE0001/0002/0003 逻辑与端到端回归全绿 |
| 模块 | 既有模块自检（composer/裁剪/标注/筛选/时序/人脸/车牌）全绿 |
| 设计器 | 新字段不破坏旧 spec；连线非法时前端有提示 |
| 事故回归 | 构造"声明但未装配"的 spec，校验器必须报错（WO-0001 的反例） |

## 8. 决策记录（2026-10-16 用户确认）

1. **端口类型粒度**：`Image / Records / Vector` 三类**够用**（不细分原帧/子图）。
2. **多上游**：**不允许**——每个端口只接一个上游产物，`inputs` 不做多值/列表；阶段可有多个
   具名端口，但各自单源（落到校验规则 §3④）。
3. **分支链路**：**有打算**——协议必须支持「同一产物被多个下游消费」；分支拓扑（DAG）列入
   后续路线，本版执行仍为线性。
4. **低清图**：**暂不管**——本阶段不设计 `frame_lowres` 之类生产者，第 1 步 `image_source`
   先只覆盖 `frame / annotated / crops`。
5. **（2026-10-23，WO-0021）标注双模式**：`annotate` 增 `scope` 参数——`frame`（默认，
   现状零回归）/ `record`（逐记录带整帧图）。record 模式**仅登记 Records 产物、主干 dets
   不动**（最小惊讶：与「标注不动主干」现状语义一致；下游显式连线取）。单帧带图记录数上限
   `max_records=32`（依据见 §9）。
6. **（2026-10-23，WO-0021）帧级记录型数据**：单条 `kind="frame"` 记录，作为**独立
   Records 产物**（独立数据流，不混入 dict 主干）；理由与约定见 §10。
7. **（2026-10-23，WO-0021）VLM 图像源统一进端口**：`vlm.inputs.image` 解禁 +
   `image_source` 退役映射（§11）——**本稿只定案，实施拆后续工单**。

**剩余未决**：第 2 步设计器连线 UI 的改动量评估（届时由 web 板块给出）。

## 9. 标注双模式（scope，WO-0021 定案 2026-10-23，已实施）

### 9.1 参数与语义

| 参数 | 取值 | 语义 |
|---|---|---|
| `scope` | `frame`（默认） | **现状**：帧副本上画全部上游框，产帧级 `annotated`（Image）。零回归。 |
| | `record` | **逐记录**：每条记录单独一张**整帧**图、只画该记录自己的框；图挂该记录 `image` 旁路（浅拷贝记录，不污染上游），记录流**透传**输出为 Records 产物 `annotate.annotated_records`。下游 `inputs.dets` 接它即图随记录走（`image_set_of` 读取）。`filter`/`classes` 只决定**画不画图**（不满足的记录原样透传、不带图）——删记录是 `det_filter` 的职责，标注不改记录流内容。 |
| `max_records` | 32（默认，仅 record 模式生效） | 单帧带图记录数上限；超限**截断**（保留前 N 条）并打告警日志（stderr）。 |

### 9.2 产物形态与「主干不动」定案

- **OUTPUTS 双类型处理**：类属性声明双产物——`OUTPUTS = {"annotated": "Image",
  "annotated_records": "Records"}`；**运行时按 scope 只登记其一**
  （frame → `annotated`；record → `annotated_records`）。
- **产物名分立而非同名变型**：端口协议的类型是静态表（`ARTIFACT_TYPES`），同名产物随
  scope 变型会让校验器无法静态判型，故 record 流用独立产物名。
- **主干不动（最小惊讶）**：record 模式只登记产物，**不替换主干 dets**——与「标注不动
  主干」现状语义一致；带图记录流只能被**显式连线**消费（`inputs.dets =
  "annotate[#k].annotated_records"`）。主干继续供事件汇总（`collect_events`）、报送分组、
  plate/face 等既有消费者按原语义使用，不被带图流污染。
- **last_annotated**：仅 frame 模式实例更新（= 最后一个 frame 实例的产物图，无则 None）；
  record 实例不影响（无单一"标注图"概念）。纯 record 链上驱动层 `image_source=annotated`
  走既有回退逻辑（回退原帧并标注来源），无需改。
- **多实例混链**：record 实例不更新 `cur_img`（无单图）；其后实例的默认 `image` 回退
  上一 frame 实例的图 / 本帧原帧；其后实例可经 `inputs.dets` 连接前一 record 实例的
  `annotated_records`（同模块后续实例在运行时顺序上天然位于其上游）。
- **运行时顺序说明**：`Pipeline.step` 内 annotate 是最后执行的视觉阶段，跨模块"运行时
  下游"目前不存在；record 流的现实消费者 = 驱动层（读产物表）与同模块后续实例连线，
  跨模块消费随 §11 VLM 端口统一 / §10 帧级数据生产者落地。

### 9.3 上限依据（`max_records=32`）

1. **对照检出上限**：YOLO `max_det` 默认 300 是**纯推理**检出上限；record 模式消费的是
   经 `det_filter`/`seq_filter` 筛选后的报送级记录流，常态个位数——上限防的是筛选全开的
   退化场景，不必对齐 300。
2. **内存估算**：整帧图 1280×720×3 B ≈ 2.8 MB/张；32 张 ≈ 88 MB 单帧峰值旁路内存
   （图像走旁路不进 JSON），可承受；若 300 张全画 ≈ 830 MB/帧，不可承受。
3. **下游消费量级**：VLM 多图送审一般 ≤ 20 张/请求、驱动层落盘 `max_saved_images=200`
   ——两位数上限与下游量级匹配且留余量。

## 10. 帧级数据通用约定（WO-0021 定案 2026-10-23；**只约定，暂无生产者**）

帧级数据三类载体，**全走产物表 + 端口协议，不新增机制**：

| 载体 | 产物类型 | 形态 | 现状 |
|---|---|---|---|
| 图像 | Image | 原帧（`frame_manager.frame`）、整帧标注图（`annotate.annotated`） | ✅ 现状即如此 |
| 记录型 | Records | **单条 `kind="frame"` 记录，作为独立 Records 产物**（独立数据流，不混入 dict 主干） | 无生产者（首个预期 = 未来 VLM 全帧研判） |
| 向量 | Vector | `face_embed.embeddings` 先例 | ✅ |

- **记录型走独立流的理由**：主干语义 = **逐目标记录**（bbox/track/count；YOLO、
  det_filter、seq_filter、target_crop 全按逐目标假设消费）——帧级数据混入主干会迫使
  全链路模块加 `kind` 过滤、污染统计（计数/成组判定/事件收集都把"帧记录"误当目标）；
  独立流零污染、消费显式（连线声明即用）、与产物表的帧作用域自洽。
- **事件汇总纳入**（约定入稿；行为改动随首个生产者实施）：链尾事件汇总从
  `collect_events(主干)` 扩为**显式纳入帧记录携带的事件**（帧记录可经 `add_event`
  携带 events）。
- **生产者约定**：任何模块可产帧记录——OUTPUTS 声明 Records 产物、内容为单条
  `kind="frame"` 记录；产物命名归属**生产阶段名下**（如 `vlm.frame_assessment`），
  首个预期生产者 = 未来 VLM 全帧研判。

## 11. VLM 图像源统一进端口（第二子项，WO-0021 定案 2026-10-23；**只定案，实施待启动**）

- **解禁 `vlm.inputs.image`**：可接帧级 Image 产物——`annotate.annotated`（带框图）、
  `frame_manager.frame`（原帧）。`image_source=crops` 的裁剪图集**不经** `inputs.image`
  （其为 Records 语义），已由 `inputs.dets` + `image_pick` 覆盖。
- **`image_source` 退役映射表**（退役时 validate_chain 对旧值报错并提示映射）：

| 旧值 | 新 spec 等价写法 | 说明 |
|---|---|---|
| `annotated`（默认） | `inputs.image = "annotate.annotated"` | 需 annotate 在上游；旧语义"无 annotate 回退原帧"→ 端口语义为"未声明 inputs 按默认取原帧"，等价 |
| `frame` | `inputs.image = "frame_manager.frame"` | 直接映射 |
| `crops` | **无需映射** | 已由 `inputs.dets`（选报送检测集）+ `image_pick`（first/all）覆盖；退役时报错提示改连线 |

- **旧 spec 零回归**：`image_source` 未删除前旧 spec 照跑（现状语义不变）。
- **校验断言变更预告**：`vlm.inputs` 白名单从 `{dets}` 放开为 `{dets, image}`；
  `inputs.image` 与 `image_source` **同时声明 → 报错**（避免双机制，与 §3③ 精神一致）；
  终态 `image_source` 从 PARAMS / `VLM_IMAGE_SOURCES` 移除。
- **实施待用户或后续工单启动**（WO-0021 本轮不实施）。
