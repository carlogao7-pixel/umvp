# YOLO 识别 YOLODetector

> 模块组：管道模块 | 代码：`umvp/pipe/composer.py`（`YOLODetector`）
> 数据库参数：`YOLO0001`「警用无人机·YOLO识别」/ `YOLO0002`「车牌识别·YOLO识车」/ `YOLO0003`「人脸截取·YOLO识人」（presets 表 + preset_params_yolo 表）
> 状态：**2026-10 起 YOLO 收敛为「纯推理」**——有效性过滤/成组数量判定归
> [检测筛选.md](检测筛选.md)（链路上独立的 det_filter 阶段），向下传递节流与报送
> 编排归 `Pipeline`（本文件的装配器）。**YOLO preset 不再携带有效性参数**。

## 1. 模块定位

小模型识别——分析管道中"小模型"这一环，干一件事：**用 YOLO 权重对一帧图像做目标检测**，
输出 `Detection` 列表。COCO 常用类别：0=人、2=车。

2026-10 职责外移（原模块的三件事拆开）：

| 原职责 | 现归属 |
|---|---|
| 推理（本模块） | `YOLODetector.infer()`（conf/iou/imgsz/max_det/classes/device） |
| 过滤：置信度/尺寸/成组数量（视为"未识别"） | 「检测筛选 DetFilter」（`Pipeline.det_filter` 阶段） |
| 报送：按类分组 + 节流 + track 变化抑制 + 构造 SubmitRequest | `Pipeline`（`_submits()` / `_pass_dets()`，装配器内） |

这样 YOLO 只产出"画面里有什么"，"哪些算有效""多久往下送一次""怎么组报送"都是
链上后续阶段/编排的事——检测筛选可作几乎所有模块下游，节流与报送是链路编排语义。

## 2. 参数

### 2.1 模块自身参数（推理，落库为 preset）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model_path | file | 空 | YOLO 权重（models/*.pt）；空则不能 `infer()`（供外部检测列表直传的兼容形态） |
| conf | float | 0.35 | 检出阈值：置信度低于它的框丢弃；调低可召回遮挡/重叠目标 |
| iou | float | 0.7 | NMS 阈值：重叠超过它的重复框合并；调低允许高度重叠的框并存 |
| imgsz | int | 640 | 推理分辨率：越大小目标越清楚，算力约平方增长（远角场景决定性参数） |
| max_det | int | 300 | 单帧最多检出目标数 |
| classes | str | "0" | 类别白名单（逗号分隔，空=不限）——推理期过滤（省算力），非"有效性" |
| device | str | 空 | 推理设备：空=自动；`cpu` / `0`（GPU 序号） |

### 2.2 推理分辨率说明（imgsz）

`imgsz` 原样透传 ultralytics，由其在内部做 **letterbox 等比缩放**（长边缩到 imgsz、
灰边补齐、非拉伸），检出框坐标自动映射回**原始帧坐标系**。例：3840×2160 的 4K 帧
送 `imgsz=640` → 缩放为 640×360 + 补边至 640×384 推理，返回框仍是 4K 原坐标；
下游（目标裁剪/人脸/车牌）按原坐标裁原分辨率子图，画质不损失。代价是小目标在
640 下像素少易漏检——调高 `imgsz`（如 1280）即可。

### 2.3 有效性参数已迁至「检测筛选」阶段（2026-10）

YOLO preset **不再有** `filter_conf`/`min_short`/`min_long`/`class_limits`——这些
有效性判定改由链路上独立的 **det_filter 阶段**（[检测筛选.md](检测筛选.md)）承担，
DB 列已随 schema 收敛删除。三要素对应关系：

| 有效性诉求 | det_filter 参数 |
|---|---|
| 置信度下限 / 尺寸门槛 | `filter` 条件 DSL（`score >= N`、`long >= N`…） |
| 每类数量上下限 | `group_limits`（类别:下限,上限） |

仍留在 YOLO preset 的**编排参数**（`Pipeline` 用）：

| preset 参数 | 默认 | 作用 |
|---|---|---|
| check_interval | 3.0 | `Pipeline.check_interval`（向下传递/报送的按类间隔门控，秒/类；0=每分析帧） |
| track_change_only | false | `Pipeline.track_change_only`（同类 track_id 集合无变化不报送） |

### 2.4 当前库内值（YOLO0001，2026-08-24 实调）

`model_path=models/yolov8n.pt, conf=0.25, iou=0.6, imgsz=1280, max_det=300,
classes="0,2", check_interval=3.0, track_change_only=false`

要点：imgsz=1280 是远角无人机小目标检出/稳定追踪的决定性参数；conf 调低到 0.25
提高远角召回。有效性（人≥2 才报、车≥1 就报）现由该链路的 det_filter 阶段
（DF0001 `group_limits="0:2,300;2:1,300"`）承担；车牌/人脸链对应 DF0002
（`filter="long >= 40"`、`group_limits="2:1,300"`）/ DF0003（`filter="long >= 30"`、
`group_limits="0:1,300"`）。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| frame | ndarray (H×W×3, BGR) | 原始帧（`infer()` 推理用；model_path 为空时不需要） |

`Detection` 结构（推理输出）：

| 字段 | 类型 | 内容 |
|---|---|---|
| track_id | int | 跟踪 ID（未启用跟踪时为 0） |
| cls_id | int | 类别编号（COCO：0=人，2=车） |
| bbox | (x1,y1,x2,y2) int | 目标框，**原图像素坐标** |
| score | float | 置信度 [0,1] |
| model_path | str，可选 | 产出该检测的权重名（多模型来源区分用） |

## 4. 输出与下游（谁用什么）

- `infer(frame, track=False) → list[Detection]`：**纯推理结果**，不做有效性过滤
  （不达标目标也在此）；`track=True` 用跟踪器保持 track_id 稳定。
- 有效性 → 链上 `Pipeline.det_filter`（DetFilter，对所有下游一致生效）。
- 向下传递门控（→目标裁剪/车牌）→ `Pipeline._pass_dets`（按类间隔，key=类别）。
- VLM 报送（→`SubmitRequest`）→ `Pipeline._submits`（按类分组 + 节流 + track 抑制）。
  `SubmitRequest` 字段（track_key/gran/ref/det/dets）与 `full:cls{类别}` / `full:global`
  语义见 [VLM研判.md](VLM研判.md)；`ref` 现已固定在编排层构造。
- 驻留告警（dwell）→ `Pipeline.step` 用有效性筛选后的 dets 逐 track 记 `on_dwell`。

## 5. 上下游衔接

- 上游：帧管理判定要图的帧 → 链路把 frame 交给本模块 `infer()`。
- 下游：`Detection` 列表 → 检测筛选阶段（有效性）→ 目标裁剪/标注/VLM 报送/驻留告警。

## 6. 验证

```bash
conda run -n ai python tests/test_yolo_standalone.py   # 单模块：infer（纯推理）+ 检测筛选演示
conda run -n ai python tests/test_yolo_params.py       # 参数对比（真实推理约 1 分钟）
conda run -n ai python umvp/pipe/test_composer.py      # 装配/编排/检测筛选/门控（64 断言）
```

## 7. 调参速查（通俗指南）

- **漏检了重叠/遮挡目标** → 降 `conf`、降 `iou`、提 `imgsz`（参考 `tests/test_yolo_params.py`）。
- **碎框/小目标太多** → 检测筛选阶段 `filter="long >= N"`（视为未识别，对所有下游生效）。
- **误检太多** → 升 `conf`（严格一点）；或检测筛选 `filter="score >= N"`。
- **画面里目标太多、报送负担大** → 提 `check_interval` 拉长复查间隔；或检测筛选
  `group_limits` 上限把"过度拥挤"的类整体跳过。
- **只想关注部分类别** → `classes` 限定推理类别（省算力），检测筛选限定报送白名单。
- **想送裁剪小图给下游** → YOLO 只检测，裁剪由独立的「目标裁剪」模块承担。

## 8. 注意事项

- `model_path` 为空时不能 `infer()`（供外部检测列表直传的兼容形态）。
- 报送粒度**按类不按目标**：一类多目标合并为一条请求（由 Pipeline 编排构造）。
- **行为约定（2026-10 迁移）**：有效性判定（置信/尺寸/成组数量）由链上 det_filter
  阶段承担，对**所有**下游一致生效并**先于**上报；`check_interval`/`track_change_only`
  上移 Pipeline 编排。有效性参数不再属于 YOLO preset（DB 列已删）。
- 推理参数全部透传 ultralytics：`imgsz` 影响算力约为平方关系，调参时留意 CPU 性能基线
  （权威值见 `out/fingerprints.json`）。
