# 目标裁剪 TargetCrop

> 模块组：管道通用模块 | 代码：`umvp/pipe/target_crop.py`（复用 `umvp/pipe/cropper.py` 的 `Cropper`）
> 数据库参数：`CROP0001`「目标裁剪·默认」/ `CROP0002`「车牌识别·目标裁剪」/ `CROP0003`「人脸截取·目标裁剪」（presets 表 + preset_params_target_crop 表）

## 1. 模块定位

按坐标裁图的**通用模块**：接收「原帧 + 坐标」（通常来自上游 YOLO 检测框），按框裁出子图；
输出分辨率可配（`out_size`，默认原图）。**不含任何检测器**，不做目标有效性过滤
（置信度/尺寸过滤由上游识别模块在其输出处负责）——`classes`/`filter` 只决定"裁哪些框"。

原「原分辨率提取 extract_faces」已拆分：裁 YOLO 框的机械裁剪归本模块；检脸 + 原图出脸
归**人脸检测**模块（`face_detect.face_pipeline.FaceStage`）。车牌链的"裁车"同样归本模块。

## 2. 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| out_size | str | "" | 输出分辨率：空=原图；`640`=长边像素等比；`640,480`=严格尺寸；`×2`/`2x`/`*2`=缩放倍数 |
| margin | float | 0.0 | 裁剪外扩比例（按框宽高外扩，clamp 到图边界） |
| classes | str | "" | 只裁这些类别（逗号分隔，空=全部；如 2=车，0=人） |
| filter | str | "" | 框筛选条件 DSL（显式字段名；空=全裁）；语法与字段见 [检测筛选](det_filter.md) 条件 DSL 节 |

`classes` 与 `filter` 同时生效（AND）；条件 DSL 与「标注」「检测筛选」共用 `pipe.conditions`。

> `scale`（缩放倍数，默认 1.0）是代码级构造参数，链路上不暴露（`_chain_from_spec` 不传、等效 1.0）；
> 链路里要做缩放请用 `out_size`（如 `×2`）。仅旧 `extract_faces` 兼容路径以它的 `upscale` 传入。

`image`、`bbox` 为模块测试页参数（test_only，不落库、不进链路）。

## 3. 输入 / 输出

端口声明（模块自描述，P5 端口协议）：`INPUTS = {"frame": "Image", "dets": "Records"}`、
`OUTPUTS = {"crops": "Records"}`——即产物 `target_crop.crops`（`image` 旁路＝该对象单张子图）。

`crop_targets(frame, dets=None, boxes=None, *, scale=1.0, margin=0.0, classes=None, filter="", out_size=None)`

| 输入 | 说明 |
|---|---|
| frame | 原帧（一切坐标的基准） |
| dets | 上游检测结果（取 bbox/cls_id/score/track_id） |
| boxes | 裸框列表（无检测元数据）；与 dets 二选一 |

输出 `list[CropResult]`：

| 字段 | 类型 | 内容 |
|---|---|---|
| img | ndarray | 裁剪子图（按 out_size 缩放后） |
| bbox | (x1,y1,x2,y2) int | 实际裁剪框（原图坐标，含 margin，clamp 后） |
| offset | (ox,oy) int | 子图左上角在原图坐标 |
| scale / scale_x / scale_y | float | 输出缩放倍数（1.0=原分辨率；严格尺寸时 X/Y 各自比例） |
| cls_id / score / track_id | int/float/int | 来源检测元数据（无检测时 cls=-1） |
| index | int | 序号 |

> `img` 为对象字段名；记录化（`to_record`）后旁路图像统一键名 `image`（旧名 `img` 兼容，
> 指向同一 ndarray），见 [records 通用语](INDEX.md#〇通用语先读)。

方法：`to_original(bbox)` / `kps_to_original(kps)` —— 子图坐标 → 原图坐标
（除以 scale 加 offset）；下游识别模块据此还原车牌/人脸框并出原分辨率小图。

**产物形态与类型归一（`as_crop`）**：

- `Pipeline.last_crops`：本帧 `CropResult` **对象列表**（驱动层现路径在用）；
- `Pipeline.artifacts["target_crop.crops"]`：同一批裁剪的**记录 dict 列表**——**逐对象一条记录**
  （每对象含 `bbox`/`offset`/`scale`/`scale_x`/`scale_y`/`index` 等主干键 + 一张 `image` 子图），
  不是"一条记录多图"（基数约定见 [拼接层端口协议设计稿](../design/assembly_ports.md) §3⑥）；
- 下游（人脸检测 / 车牌识别）用 `as_crop(x)` 对"对象或记录"做类型归一（对象零拷贝原样返回；
  dict 用其键重建 `CropResult`，子图取 `image`、`img` 兜底），再走统一的 `img`/`bbox`/`to_original`
  访问器——因此既能消费 `last_crops`（对象），也能消费 `inputs` 连线取到的裁剪记录。

## 4. 上下游衔接

- 上游：YOLO 检测框（人/车）或任意坐标。
- 下游：
  - **车牌识别**（`plate_recog.extract_plates`）：在车图里裁车牌 → 还原坐标 → 原图出牌。
  - **人脸检测**（`face_detect.face_pipeline.FaceStage`）：在人图里检脸 → 还原坐标 → 原图出脸。
  - **VLM**：需要送该类 ROI 裁剪时，YOLO 之后插本模块。
- 装配：`web_lab/server.py` 的 `_chain_from_spec` 构造 `TargetCrop` 注入 `Pipeline.crop`；
  `Pipeline.step` 把上游（检测筛选/时序门控后）的 `dets` 交给裁剪产出 `last_crops`，并把
  `to_record` 后的记录列表登记为产物 `target_crop.crops`。
- 端口连线：`target_crop.frame` / `target_crop.dets` 本期已纳入 `inputs` 解析——spec 阶段可显式声明
  `"inputs": {"frame": "frame_manager.frame", "dets": "det_filter.dets"}` 换上游；**不声明＝现状默认**
  （frame←本帧原帧、dets←当前检测流），零回归。

## 5. 保留链路

本模块是 PIPE0002（裁车）与 PIPE0003（裁人）的中间阶段，链路编排/数据流/验证见
[车牌识别链路 PIPE0002](../chains/pipe0002_plate.md)、[人脸截取链路 PIPE0003](../chains/pipe0003_face_capture.md)。

## 6. 验证

```bash
conda run -n ai python tests/test_target_crop.py   # 按坐标裁图/分辨率(长边/严格/倍数)/外扩/过滤(含条件 DSL)/坐标还原
conda run -n ai python tests/test_cropper.py       # Cropper 纯逻辑
```
（断言数不写死，以脚本运行自报为准。）
