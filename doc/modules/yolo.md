# YOLO 识别 YOLODetector

> 模块组：管道通用模块 | 代码：`umvp/pipe/composer.py`（`YOLODetector`）
> 数据库参数：`YOLO0001`「警用无人机·YOLO识别」/ `YOLO0002`「车牌识别·YOLO识车」/ `YOLO0003`「人脸截取·YOLO识人」（presets 表 + preset_params_yolo 表）
> 状态：**纯推理**——只做目标检测；有效性过滤归 [检测筛选](det_filter.md)，节流 / 报送编排归 `Pipeline`。

## 1. 模块定位

小模型识别——分析管道中"小模型"这一环，干一件事：**用 YOLO 权重对一帧图像做目标检测**，
输出 `Detection` 列表。COCO 常用类别：0=人、2=车。

三个职责中，本模块**只保留推理**（`infer()`：conf/iou/imgsz/max_det/classes/device）；
**有效性过滤**（置信度/尺寸/成组数量，不达标视为"未识别"）归链上独立的
[检测筛选 DetFilter](det_filter.md) 阶段；**按类报送**（构造 `SubmitRequest`）归装配器
`Pipeline`（`_submits()`）。这样"画面里有什么"与"哪些算有效""多久往下送一次"
解耦，检测筛选可作几乎所有模块的下游。向下传递的节流/变化门控由 [时序判定 SeqFilter](seq_filter.md)
（`throttle`/`change`，承接原 `check_interval`/`track_change_only`）承担。

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

### 2.3 有效性 / 编排参数已迁出（2026-10）

YOLO preset **只剩推理参数**（上表）：原 `filter_conf`/`min_short`/`min_long`/`class_limits`
迁入链上 [检测筛选 DetFilter](det_filter.md)（`filter` 条件 DSL / `group_limits`），
原 `check_interval`/`track_change_only` 迁入 [时序判定 SeqFilter](seq_filter.md)
（`throttle`/`change` 门控）；相关 DB 列已从 schema 删除。

### 2.4 当前库内值（YOLO0001，2026-08-24 实调）

`model_path=models/yolov8n.pt, conf=0.25, iou=0.6, imgsz=1280, max_det=300,
classes="0,2"`

要点：imgsz=1280 是远角无人机小目标检出/稳定追踪的决定性参数；conf 调低到 0.25
提高远角召回。该链路的有效性（人≥2 才报、车≥1 就报）由 det_filter（DF0001）承担，
见 [警用无人机链路](../chains/pipe0001_police_uav.md)；车牌/人脸链对应 DF0002 / DF0003。

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
- 端口声明（模块自描述，P5 端口协议）：`INPUTS = {"image": "Image"}`、`OUTPUTS = {"dets": "Records"}`
  ——产物 `yolo.dets`＝本帧纯推理原始检出（筛选前），逐帧登记在 `Pipeline.artifacts`。
  下游可经 `inputs` 连它（如 `det_filter.dets` 端口连 `yolo.dets` 即"筛选前"语义）；
  `yolo.image` 本期已纳入解析，spec 可换送 YOLO 的图（默认＝本帧原帧，零回归）。
- 有效性 → 链上 `Pipeline.det_filter`（DetFilter，对所有下游一致生效）。
- VLM 报送（→`SubmitRequest`）→ `Pipeline._submits`（按类分组；节流/变化门控在上游
  det_filter/seq_filter 已生效）。`SubmitRequest` 字段（track_key/gran/ref/det/dets）与
  `full:cls{类别}` / `full:global` 语义见 [VLM 研判](vlm.md)。

## 5. 上下游衔接

- 上游：帧管理判定要图的帧 → 链路把 frame 交给本模块 `infer()`。
- 下游：`Detection` 列表 → 检测筛选（有效性）→ 时序门控 → 目标裁剪 / 标注 / VLM 报送。

## 6. 验证

```bash
conda run -n ai python tests/test_yolo_standalone.py   # 单模块：infer（纯推理）+ 检测筛选演示
conda run -n ai python tests/test_yolo_params.py       # 参数对比（真实推理，较慢）
conda run -n ai python umvp/pipe/test_composer.py      # 装配/编排/检测筛选/门控（纯逻辑）
```
（断言数不写死，以脚本运行自报为准。）

## 7. 调参速查（通俗指南）

- **漏检了重叠/遮挡目标** → 降 `conf`、降 `iou`、提 `imgsz`（参考 `tests/test_yolo_params.py`）。
- **碎框/小目标太多** → 检测筛选阶段 `filter="long >= N"`（视为未识别，对所有下游生效）。
- **误检太多** → 升 `conf`（严格一点）；或检测筛选 `filter="score >= N"`。
- **画面里目标太多、报送负担大** → 时序判定阶段调大 `throttle` 间隔拉长复查；或检测筛选
  `group_limits` 上限把"过度拥挤"的类整体跳过。
- **只想关注部分类别** → `classes` 限定推理类别（省算力），检测筛选限定报送白名单。
- **想送裁剪小图给下游** → YOLO 只检测，裁剪由独立的「目标裁剪」模块承担。

## 8. 注意事项

- `model_path` 为空时不能 `infer()`（供外部检测列表直传的兼容形态）。
- 报送粒度**按类不按目标**：一类多目标合并为一条请求（由 Pipeline 编排构造）。
- **职责边界**：有效性判定（置信/尺寸/成组数量）由链上 det_filter 承担，对**所有**下游
  一致生效并**先于**上报；节流/变化门控归 seq_filter。有效性参数不属于 YOLO preset。
- 推理参数全部透传 ultralytics：`imgsz` 影响算力约为平方关系，调参时留意 CPU 性能基线
  （权威值见 `out/fingerprints.json`）。
