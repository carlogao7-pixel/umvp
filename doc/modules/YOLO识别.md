# YOLO 识别 YOLODetector

> 模块组：管道四模块 | 代码：`umvp/pipe/composer.py`（`YOLODetector`）
> 数据库参数：`YOLO0001`「警用无人机·YOLO识别」（presets 表 + preset_params_yolo 表）

## 1. 模块定位

小模型识别：YOLO 推理（可选）+ 可插拔过滤链 + **按识别类型（类别）报送**。
画面里每出现一类目标（且数量在该类上下限内）就产生一条报送请求，一类多个目标
合并成一条；每类独立节流，避免刷屏。COCO 常用类别：0=人、2=车。

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
| filter_conf | float | 0.0 | 过滤链·置信度下限（0=不启用）：低于它的检出不参与报送 |
| min_short | int | 0 | 过滤链·短边最小像素（0=不启用）：目标太小的框不报送 |
| min_long | int | 0 | 过滤链·长边最小像素（0=不启用） |
| class_limits | str | "0:1,300" | 识别对象白名单+数量范围：`类别:下限,上限`，分号分隔；数量按过滤后该类目标数算，不在范围内该类整体不报送；空=不报送 |
| check_interval | float | 3.0 | 报送节拍（秒/类）：同一类别每隔几秒才报一次 |
| roi | bool | false | 送审素材形态：false=全帧 full:cls{N}（拼接层画该类框标注）；true=该类裁剪 crop:cls{N} |
| track_change_only | bool | false | 只在 track_id 集合变化时才报送（需启用跟踪），抑制同类目标不变的重复送审 |

### 2.2 测试参数（仅模块测试页运行用，不落库、不进链路）

`image`（测试图片下拉）。`ts`（报送时间戳）由链路运行器自动填充。

### 2.3 当前库内值（YOLO0001，2026-08-24 实调）

`model_path=models/yolov8n.pt, conf=0.25, iou=0.6, imgsz=1280, max_det=300,
classes="0,2", class_limits="0:2,300;2:1,300", check_interval=3.0, roi=false,
track_change_only=false`

要点：imgsz=1280 是远角无人机小目标检出/稳定追踪的决定性参数（见知识库
YOLO 参数对比实测）；conf 调低到 0.25 提高远角召回；人≥2 才报、车≥1 就报。

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
| ref | str | 素材引用：`full:cls{类别}` 全帧 / `crop:cls{类别}` 该类裁剪 / `full:global` 整帧全局 |
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
