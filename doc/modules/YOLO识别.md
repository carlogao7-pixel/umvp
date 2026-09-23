# YOLO 识别 YOLODetector

> 模块组：管道模块 | 代码：`umvp/pipe/composer.py`（`YOLODetector`）
> 数据库参数：`YOLO0001`「警用无人机·YOLO识别」/ `YOLO0002`「车牌识别·YOLO识车」/ `YOLO0003`「人脸截取·YOLO识人」（presets 表 + preset_params_yolo 表）

## 1. 模块定位

小模型识别——分析管道中"小模型"这一环，承担三件事：

1. **推理**：用 YOLO 权重对一帧图像做目标检测（可关，见 `model_path` 留空）。
2. **过滤**：按一套可自定义顺序的规则，把不合格的目标从结果里剔掉。
3. **报送**：把画面里"值得送审"的目标整理成请求，交给下一环（VLM 大模型研判或
   输出处理）。报送粒度固定为**按识别类型（类别）**——每出现一类目标（且数量在该类
   上下限内）就发一条请求，同一类多个目标合并成一条；每类独立节流，避免刷屏。
   COCO 常用类别：0=人、2=车。

代码层 `compose()` 的四种兼容模式（small_only / small_crop / small_full / large_only）
中它负责 small 部分；产品链路已改为**模块自定义拼接**（见 CLAUDE.md 保留链路），
模式差异全部收敛为参数配置，代码本身不写分支。

## 2. 参数

### 2.1 模块自身参数（落库为 preset）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model_path | file | 空 | YOLO 权重（models/*.pt）；空则退化为纯后处理（不推理，只过滤+报送） |
| conf | float | 0.35 | 检出阈值：置信度低于它的框丢弃；调低可召回遮挡/重叠目标 |
| iou | float | 0.7 | NMS 阈值：重叠超过它的重复框合并；调低允许高度重叠的框并存 |
| imgsz | int | 640 | 推理分辨率：越大小目标越清楚，算力约平方增长（远角场景决定性参数） |
| max_det | int | 300 | 单帧最多检出目标数 |
| classes | str | "0" | 类别白名单（逗号分隔，空=不限） |
| device | str | 空 | 推理设备：空=自动；`cpu` / `0`（GPU 序号） |
| filter_conf | float | 0.0 | 过滤链·置信度下限（0=不启用）：低于它的检出**视为未识别** |
| min_short | int | 0 | 过滤链·短边最小像素（0=不启用）：目标短边低于它视为未识别（滤碎框/噪声） |
| min_long | int | 0 | 过滤链·长边最小像素（0=不启用）：目标长边低于它视为未识别（滤小目标） |
| class_limits | str | "0:1,300" | 识别对象白名单+数量范围：`类别:下限,上限`，分号分隔；数量按过滤后该类目标数算，不在范围内该类整体不报送；空=不报送 |
| check_interval | float | 3.0 | **向下游传递的统一间隔门控（秒/类）**：同一识别类型每隔几秒才向下一阶段传一次数据——下游是 VLM 时管报送节拍（process），是目标裁剪/车牌阶段时管检测框传递（pass_dets），共用同一门控（IntervalGate，key=类别）。0=每分析帧都传 |
| track_change_only | bool | false | 只在 track_id 集合变化时才报送（需启用跟踪），抑制同类目标不变的重复送审 |

> **过滤链在 `infer()` 输出处生效**：`filter_conf`/`min_short`/`min_long`（及自定义 `filters`）
> 不达标的目标**视为"未识别"**，不出现在 infer 结果里——因此对**所有**下游生效
> （VLM 报送 / 目标裁剪·车牌·人脸的传递 / 标注），且先于 `class_limits` 的按类计数
> （计数只数过滤后的目标）。

### 2.2 测试参数（仅模块测试页运行用，不落库、不进链路）

`image`（测试图片下拉）。`ts`（报送时间戳）由链路运行器自动填充。

### 2.3 当前库内值（YOLO0001，2026-08-24 实调）

`model_path=models/yolov8n.pt, conf=0.25, iou=0.6, imgsz=1280, max_det=300,
classes="0,2", class_limits="0:2,300;2:1,300", check_interval=3.0,
filter_conf=0.0, min_short=0, min_long=0, track_change_only=false`

要点：imgsz=1280 是远角无人机小目标检出/稳定追踪的决定性参数（见知识库
YOLO 参数对比实测）；conf 调低到 0.25 提高远角召回；人≥2 才报、车≥1 就报。
车牌/人脸链的 YOLO 尺寸门槛来自迁移：YOLO0002 `min_long=40`（车）、YOLO0003
`min_long=30`（人）——即"长边 < N 视为未识别"。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| frame | ndarray (H×W×3, BGR) | 原始帧（`infer()` 推理用；model_path 为空时不需要） |
| dets | list[Detection] | 已检出的目标列表（纯后处理模式 / 链路外部推理时传入） |
| frame_idx / ts | int / float | 帧序号与时间戳（`process()` 报送节流计时用） |

`Detection` 结构（输入或输出均为此结构）：

| 字段 | 类型 | 内容 |
|---|---|---|
| track_id | int | 跟踪 ID（未启用跟踪时为 0） |
| cls_id | int | 类别编号（COCO：0=人，2=车） |
| bbox | (x1,y1,x2,y2) int | 目标框，原图像素坐标 |
| score | float | 置信度 [0,1] |
| model_path | str，可选 | 产出该检测的权重名（多模型来源区分用） |

## 4. 输出（字段和内容）

### 4.1 infer() → list[Detection]（同上表）

### 4.2 process() → list[SubmitRequest]（报送请求，交给 VLM 研判）

| 字段 | 类型 | 内容 |
|---|---|---|
| track_key | int | 识别类型=类别编号（如 0=人）；整帧全局报送时为 -1 |
| gran | str | 报送粒度：`class`（按识别类型） / `scene`（周期性整帧） |
| ref | str | 素材引用：`full:cls{类别}` 全帧（拼接层画该类框标注） / `full:global` 整帧全局 |
| det | Detection，可选 | 单目标素材（保留字段，按类报送时为空） |
| dets | list[Detection] | 该类全部有效检测（拼接层画框/送审取材用） |

## 5. 上下游衔接

- 上游：帧管理判定要图的帧 → 链路把 frame 交给本模块 `infer()`。
- 下游：`process()` 产出的 SubmitRequest 交给 VLM 研判；过滤链亦被
  驻留告警（dwell）复用（`filter_only()`）。

## 6. 验证

```bash
conda run -n ai python tests/test_yolo_standalone.py   # 单模块：infer→filter_only→process
conda run -n ai python tests/test_yolo_params.py       # 参数对比（真实推理约 1 分钟）
```

## 7. 调参速查（通俗指南）

- **漏检了重叠/遮挡目标** → 降 `conf`、降 `iou`、提 `imgsz`（按此顺序先温和后激进，参考 `tests/test_yolo_params.py` 的对比参数组）。
- **碎框/小目标太多** → 用 `min_short`/`min_long`（过滤链）把过小的目标视为未识别（对所有下游生效）。
- **误检太多** → 升 `conf`（严格一点）、必要时用 `filter_conf`（过滤链置信度下限）。
- **画面里目标太多、报送负担大** → 提 `check_interval` 拉长复查间隔；或调 `class_limits` 上限把"过度拥挤"的类整体跳过。
- **只想关注部分类别** → `classes` 限定推理类别（省算力），`class_limits` 限定报送白名单。
- **想送裁剪小图给下游** → YOLO 只检测，裁剪由独立的「目标裁剪」模块承担：在 YOLO 之后插目标裁剪阶段（`scale`/`margin`/`classes`），需要送裁剪时下游消费裁剪子图。

## 8. 注意事项

- `model_path` 为空时模块为**纯后处理**模式：只过滤+报送，`infer()` 会抛错。
- 报送粒度**按类不按目标**：同一类多个目标合并为一条请求；驻留告警（dwell）仍按 track 记首次出现，但那是输出处理模块内部逻辑，不产生报送请求。
- 未在 `class_limits` 登记的类别即使检出也不报送。
- 推理参数全部透传 ultralytics：`imgsz` 影响算力约为平方关系，调参时留意 CPU 性能基线（权威值见 `out/fingerprints.json`）。
