# 标注 Annotator

> 模块组：管道通用模块 | 代码：`umvp/pipe/annotate.py`
> 数据库参数：`ANNO0001`「标注·默认」/ `ANNO0002`「警用无人机·标注」（presets 表 + preset_params_annotate 表）

## 1. 模块定位

在帧副本上画上游检测框（可带类别标签），输出**带框图**——典型用于"报送 VLM 时给
大模型看原图 + 画框"。纯绘制：不含模型、不改坐标、不裁图。

与「目标裁剪」分工：目标裁剪出**子图**（聚焦局部）；标注出**整帧 + 框**（保留上下文）。

## 2. 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| scope | select | frame | 标注产物形态：`frame`=帧级（整帧一张标注图，现状）；`record`=逐记录（每条记录一张整帧图、只画自己的框，见 §3.1） |
| max_records | int | 32 | record 模式单帧带图记录上限，超限截断并告警（依据见 [端口协议设计稿](../design/assembly_ports.md) §9.3） |
| thickness | int | 2 | 框线粗细（像素） |
| show_label | bool | true | 在框上方标注类别名（人/车…）+ track# |
| classes | str | "" | 只画这些类别（逗号分隔，空=全部；如 0,2） |
| filter | str | "" | 框筛选条件 DSL（显式字段名；空=全画）；语法与字段见 [检测筛选](det_filter.md) 条件 DSL 节 |

`classes` 与 `filter` 同时生效（AND）。条件**显式声明字段名**，故不绑定 YOLO——
后续接入其它来源（含同名属性）即可复用。非法条件在装配/保存时抛 ValueError 快速失败；
`scope` 非法取值同样装配期 fail fast。

`image`、`bbox` 为模块测试页参数（test_only，不落库、不进链路）。

## 3. 输入 / 输出

`annotate_frame(frame, dets=None, *, thickness=2, show_label=True, classes=None, filter="") -> ndarray`

| 输入 | 说明 |
|---|---|
| frame | 原帧（在其副本上绘制，不改原帧） |
| dets | 上游检测结果（取 bbox/cls_id/track_id） |

输出：标注后的 BGR 帧（与原帧同尺寸）。类别名映射：`{0:人, 2:车, 3:摩托, 5:公交, 7:卡车}`，未收录用 `cls{编号}`。

端口声明（模块自描述，P5 端口协议）：`INPUTS = {"image": "Image", "dets": "Records"}`、
`OUTPUTS = {"annotated": "Image", "annotated_records": "Records"}`——**双模式产物**，
运行时按 scope 只登记其一（设计稿 [assembly_ports](../design/assembly_ports.md) §9）。

### 3.1 record 模式（逐记录带整帧图，WO-0021）

- `annotate_records(frame, dets, ..., max_records=32) -> (records, dropped)`：每条记录单独
  一张**整帧**图、只画该记录自己的框，图挂记录 `image` 旁路（浅拷贝，不污染上游）；
- 记录流**透传**输出为 Records 产物 **`annotate.annotated_records`**：`filter`/`classes`
  只决定画不画图（不满足的记录原样透传、不带图），**删记录是 det_filter 的职责**；
- **主干不动**（最小惊讶）：仅登记产物，主干 dets/`last_annotated` 均不受影响；下游消费
  显式连线 `"inputs": {"dets": "annotate[#k].annotated_records"}`（`image_set_of` 读取）；
- 上限 `max_records`（默认 32）：超限截断（保留前 N 条）+ stderr 告警。

## 4. 上下游衔接

- 上游：YOLO 检测框。
- 下游：VLM（`Pipeline.last_annotated` 作为素材；`web_lab` 的 VLM 素材准备优先用标注图）。
- 装配：`web_lab/server.py` 的 `_chain_from_spec` 构造 `Annotator` 注入 `Pipeline.annotate`；
  `Pipeline.step` 在 YOLO 之后产出 `last_annotated`，并按 scope 登记产物
  `annotate.annotated`（frame）或 `annotate.annotated_records`（record）。
- 端口连线：`annotate.image` / `annotate.dets` 本期已纳入 `inputs` 解析——spec 可显式声明
  `"inputs": {"image": "frame_manager.frame", "dets": "det_filter.dets"}` 换上游（如标注筛选后的框）；
  **不声明＝现状默认**（image←本帧原帧、dets←当前检测流），零回归。
- 与 VLM 的取图关系：`annotate.annotated` 是 `vlm.image_source=annotated`（默认源）的素材来源；
  缺 `annotate` 阶段时 VLM 回退本帧原图。

## 5. 保留链路

- **PIPE0001 警用无人机链路**：帧管理 → YOLO → 检测筛选 → **标注** → VLM研判 → 时序判定事件尾
  （VLM 看"原图 + 画框"）。见 [警用无人机链路](../chains/pipe0001_police_uav.md)。

## 6. 验证

```bash
conda run -n ai python tests/test_annotate.py   # 画框/尺寸不变/classes 过滤/条件 DSL/Pipeline 集成/record 双模式
```
（断言数不写死，以脚本运行自报为准。）
