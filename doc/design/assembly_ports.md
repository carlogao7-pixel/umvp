# 拼接层端口协议（设计稿）

> 状态：📐 设计稿（**第 0 步已实施 2026-10-16**（WO-0002.1）；第 1/2 步待排）。需求来源：
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
| face（人脸，待并入） | `(frame, last_crops, ts)` | 拟同 plate |
| vlm（研判） | 图像**硬编码** `last_annotated`（server.py:1221） | 驱动层 |

→ 模块**换不了位置**，也接不了不同上游；VLM 想用原帧/裁剪图而不是带框图，spec 里无处声明
（`vlm` 参数里的 `ref` 只表"素材引用/粒度"，不表"哪张图"）。

### 1.3 旁路产物命名不统一

`umvp/pipe/records.py` 只约定两个旁路名：`image`（ndarray 子图/帧）、`embedding`（向量）；
实际各模块自造：crop 用 `img`、face 用 `face_img`、plate 用 `plate_img`、annotate 返回帧副本
（非记录）。没有"产物"的统一命名与登记，下游无法通用消费。

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
| 图像（旁路） | `frame_manager.frame`、`annotate.annotated`、`target_crop.crops[i].img` | ndarray，不进 JSON |
| 向量（旁路） | `face_embed.embeddings` | ndarray |

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
{"module_id": "vlm", "params": {...},
 "inputs": {"image": "annotate.annotated", "dets": "det_filter.dets"}}
```

未声明 `inputs` 时按**现硬编码规则**（crop/annotate←dets，plate/face←crops，vlm←annotated）→ 零回归。

### ④ 校验（validate_chain）

- 引用的产物存在且在其上游（不能引用下游/自身）；
- **每个端口只接一个上游产物**（`inputs` 不做多值/列表；阶段可有多个具名端口，各自单源）；
- 端口类型匹配（Image 端口不得喂 Records）；
- 无环；**同一产物可被多个下游引用**（为将来的分支链路留路，见 §8）；
- **spec 声明的阶段必须被装配执行**（第 0 步先落这条）。

### ⑤ 驱动（Pipeline.step）

从"按槽位写死实参"改为"按 spec 查产物表解析实参"：每帧维护 `artifacts: {name: value}`，
阶段执行前解析 `inputs`，执行后登记 `outputs`。

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

### 第 1 步：旁路命名统一 + 图像源选择（WO-0002.2）

- 旁路产物统一命名（`img`/`face_img`/`plate_img` → `image` 语义 + 保留兼容别名）；
- spec 阶段支持 `image_source`（`frame / annotated / crops`），默认 = 现状（低清图暂不管，见 §8）；
- VLM 素材在壳里可选（满足"原帧/带框图/裁剪图任选"的诉求）；
- 回归：三条保留链路 + 全模块自检。

### 第 2 步：完整端口协议（WO-0002.3）

- 产物表 + 模块 `INPUTS/OUTPUTS` 声明 + `inputs` 连线 + 校验器 + 设计器连线 UI（下拉选产物）；
- `docs/modules/INDEX.md` 的「两平面角色表」升级为「产物端口表」；
- 新增模块接入清单：登记声明 → 装配器自动可用（不再改 `Pipeline` 核心）。

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

**剩余未决**：第 2 步设计器连线 UI 的改动量评估（届时由 web 板块给出）。
