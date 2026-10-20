# 车牌识别 PlateRecognizer

> 模块组：车牌模块 | 代码：`umvp/plate_recog/plate_recognizer.py`（链路阶段 `umvp/plate_recog/plate_pipeline.py` 的 `PlateStage`）
> 数据库参数：`PLATE0001`「车牌识别·默认」/ `PLATE0002`「车牌识别·车牌识别」（PIPE0002）（presets 表 + preset_params_plate_recog 表）

## 1. 模块定位

基于 **HyperLPR3**（onnxruntime + opencv）的中文车牌识别：从**车辆区域图**里定位车牌
（多任务检测出车牌框 + 四角点，透视矫正成正面图，即"裁车牌"）→ CRNN 字符识别 →
颜色分类兜底。

**职责边界**：模块**不含车辆检测、不裁车**——"识车"由上游 `yolo` 模块负责（`classes=2`）；
"裁车"由独立的**目标裁剪**模块（`pipe.target_crop`）按车辆框裁出车辆区域图。本模块只做
"从车辆区域图里裁车牌并识别 + 车牌框还原回原图 + 从原图提取原分辨率车牌小图
（存证/素材，`Cropper.extract`）"。

模型默认从项目内 `plate_models/hyperlpr3/` 加载（代码按 `__file__` 相对定位，
不依赖 `~/.hyperlpr3`）。

## 2. 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| detect_level | select | high | 检测精度档：high=640（远角/小车牌更准，慢）；low=320（快） |
| mode | select | crop | crop=消费上游「目标裁剪」的车图再识别；direct=整帧直接识别 |
| margin | float | 0.1 | 车牌出图外扩比例（从原图裁原分辨率车牌小图时往外多扩） |
| dedup_iou | float | 0.6 | 车牌框去重 IoU |
| track_dedup | bool | true | 按上游 YOLO 的 track_id 去重：同一辆车只识别一次（视频强烈建议开） |
| track_cooldown | float | 5.0 | 同一 track 多少秒后才允许再次识别；0=只识别一次不再重试 |

构造级参数（代码层，不在 web 表单）：`model_root`（模型根目录）、`logger_level`
（onnxruntime 日志级别）。

测试页脚手架参数（`test_only`，不落库、不进链路设计器）：`image`、`yolo_model`、
`yolo_conf`、`yolo_iou`、`yolo_imgsz`、`yolo_max_det`、`yolo_classes`
（模块测试页用 YOLO 造车辆框；生产链路由上游 `yolo` 阶段提供）。

当前库内值（PLATE0001，与 PIPE0002 的 PLATE0002 同，快照 2026-10）：
`detect_level=high, mode=crop, margin=0.1, dedup_iou=0.6, track_dedup=true, track_cooldown=5.0`。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| img | ndarray (H×W×3, BGR) | 待识别图像：**车辆区域裁剪图**（推荐），或整帧（`recognize()` 二者皆可） |

`extract_plates(img, crops, plate_rec, tool=None, dedup_iou=0.6)`（消费上游裁剪产物）额外需要：

| 字段 | 类型 | 内容 |
|---|---|---|
| img | ndarray | 原帧（车牌小图的出处、坐标还原的基准） |
| crops | list[CropResult] | **上游「目标裁剪」模块**产出的车辆子图（含 `to_original` 还原能力） |
| plate_rec | PlateRecognizer | 本模块实例 |
| tool | Cropper，可选 | 通用裁剪工具（`pipe.cropper`；默认 upscale=1.0, margin=0.0） |
| dedup_iou | float | 车牌框去重 IoU（默认 0.6） |

## 4. 输出（字段和内容）

`recognize()` → `list[PlateResult]`；`extract_plates()` 同类型（附加 `car_bbox`）：

| 字段 | 类型 | 内容 |
|---|---|---|
| plate_no | str | 车牌号（如 粤ACJ0710） |
| plate_type | int | HyperLPR3 类型编号（0=蓝,1=黄单,2=白单,3=绿新能源,4=黑港澳,9=黄双层…） |
| type_name | str | 类型中文名（如 绿牌新能源） |
| color | str | 颜色（blue/yellow/green/white/black/unknown） |
| score | float | 识别置信度 [0,1] |
| bbox | (x1,y1,x2,y2) int | 车牌框，**原图坐标系**（`extract_plates` 已还原） |
| car_bbox | (x1,y1,x2,y2)，可选 | 来源车框（原图坐标，仅 `extract_plates`） |
| upscale | float | 车图的缩放倍数（来自目标裁剪 `out_size`；1.0=未放大） |
| plate_img | ndarray，可选 | **原分辨率车牌图**（从原图直接裁出，margin 外扩，无缩放）；记录化（`to_record`）后统一键名 `image`（旧名 `plate_img` 兼容，指向同一 ndarray） |
| plate_crop_bbox | (x1,y1,x2,y2)，可选 | plate_img 对应的原图裁剪框（含 margin） |
| index | int | 序号 |
| width / height | int 属性 | 车牌框宽/高（像素） |

## 5. 上下游衔接

- **模块核心 `recognize(img)`**：对一张车辆区域图（或整帧）做车牌定位裁剪（透视矫正）
  + 识别。整帧直读在 4K 视频上已验证可用。
- **链路裁剪（远角小牌推荐）**：上游 `yolo` 模块识车给出车辆框 → **目标裁剪模块**
  （`pipe.target_crop`）按框裁车放大（`out_size="×2"`，远角建议 2~3）→ 本模块 `recognize()`
  从车图里定位并裁出车牌 + 识别 → 车牌框还原回原图坐标 → **原分辨率提取**：按还原框
  从原图直接裁出车牌小图（`Cropper.extract`，margin 外扩不缩放，存证/素材用）
  （`extract_plates(frame, crops, ...)` 已封装，crops 来自目标裁剪）。
- **视频去重（重要）**：同一辆车在多帧被检出，若每帧都识别则频次过高。两层门控：
  ① **按识别类型**（yolo→车辆框传递）：上游 `seq_filter` 阶段的 `throttle`（整类节流，
  key=类别；间隔内整类扣留）；② **按车辆**
  （车牌阶段自身）：`track_dedup` 按 track_id 去重——同一 track 只识别一次，
  `track_cooldown` 秒后可重试（应对首帧模糊）。新目标入场延迟由帧管理取帧间隔
  （frame_skip）兜底。上游跟踪由 `Pipeline.track_needed()` 自动开启——它读
  `PlateStage.use_track`（＝`track_dedup and mode != "direct"`）：**仅当按车辆去重开启且
  `mode=crop` 时**，`yolo.infer(track=True)` 才启用。
  实测：车牌识别.MP4 20–35s（90 分析帧 / 332 车辆框）→ 开启前逐帧识别，开启后
  仅 3 次（每车一次，5s 冷却刷新）。
- **端口声明（P5）**：`PlateStage.INPUTS = {"frame": "Image", "crops": "Records"}`、
  `OUTPUTS = {"plates": "Records"}`——产物 `plate_recog.plates` 即产出的车牌记录列表
  （`image` 旁路＝原分辨率车牌图）。`plate_recog.frame` / `plate_recog.crops` 本期已纳入 `inputs`
  解析，spec 可显式连线；**不声明＝现状默认**（frame←本帧原帧、crops←上游「目标裁剪」`last_crops`），
  零回归。`PlateStage.run` 入口经 `as_crop` 归一，既收 `CropResult` 对象、也收裁剪记录 dict。
- 下游：`PlateResult` 可接去重 / 送 VLM 复核 / 事件尾；识别到的新车牌进链路事件尾
  （`SeqFilter` 规则 `kind==plate` + `cooldown<0`），按**车牌号**去重——同车牌只出一次
  事件。车牌链路无 VLM 阶段，不构造报送（日志"报送 0 条"属正常）。

## 6. 验证

```bash
conda run -n ai python tests/test_plate_recog_pipeline.py   # 整帧识别 + 目标裁剪裁车→车牌识别坐标还原 + 原分辨率提取 + 链路装配
```
（断言数不写死，以脚本运行自报为准。）
注意：onnxruntime 的 `VerifyOutputSizes` 形状警告为 SCRFD/HyperLPR3 动态尺寸正常现象。

## 7. 模型与自包含（附录）

- 模型目录：`plate_models/hyperlpr3/20230229/onnx/`
  - `y5fu_320x_sim.onnx` / `y5fu_640x_sim.onnx`：检测（对应 detect_level low/high）
  - `rpv3_mdict_160_r3.onnx`：识别（CRNN）
  - `litemodel_cls_96x_r1.onnx`：颜色分类（字符判不出类型时的兜底）
- 重建（新克隆/换机）：
  ```bash
  conda run -n ai python umvp/plate_recog/fetch_models.py   # 从官方源拉取并解压到项目内
  ```
- 离线保障：hyperlpr3 在 `import` 时会检查 `~/.hyperlpr3/<版本>/`，缺失则联网下载；
  本模块在加载前把项目模型同步到该目录（若缺失），因此**项目模型在即可完全离线运行**，
  且不触发下载。`plate_models/` 不入库（见 `.gitignore`）。

## 8. 保留链路

本模块是 PIPE0002 车牌识别链路的识别阶段（链路编排/数据流/验证见
[车牌识别链路 PIPE0002](../chains/pipe0002_plate.md)）；链路装配、preset 引用与运行方式见该专篇。
