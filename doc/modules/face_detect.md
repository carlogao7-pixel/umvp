# 人脸检测 FaceDetector

> 模块组：人脸链模块 | 代码：`umvp/face_detect/face_detector.py`（链路阶段 `umvp/face_detect/face_pipeline.py` 的 `FaceStage`）
> 数据库参数：`FDET0001`「人脸检测·默认」（独立）/ `FDET0002`「人脸截取·人脸检测」（PIPE0003）（presets 表 + preset_params_face_detect 表）

## 1. 模块定位

SCRFD 人脸检测（InsightFace buffalo_l 包 det_10g.onnx）：定位人脸 +
5 点关键点 + ArcFace 标准 112×112 对齐裁剪。只负责"找到脸"，
识别/检索交给下游人脸嵌入与向量底库。

模型默认从项目内 `face_models/buffalo_l/` 加载（代码按 `__file__` 相对定位，不依赖 `~/.insightface`）。
加载逻辑（provider 选择 / CUDA 失败回退 CPU / `is_gpu`）与 `FaceEmbedder` 共用
`umvp/face_detect/providers.py`。

链路阶段 `FaceStage` 在本模块之上再做一步：在上游「[目标裁剪](target_crop.md)」产出的
**目标子图**里检脸 → 把脸框/关键点还原回原图坐标 → 从**原图**裁出原分辨率人脸（去重），
是 **PIPE0003 人脸截取链路**的末段。

## 2. 参数

### 2.1 检测参数（`FaceDetector`，落库为 preset）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| device | select | auto | 计算设备：auto/cuda/cpu；CUDA 初始化失败自动回退 CPU |
| det_thresh | float | 0.5 | 检测阈值：低于它的脸不认；调低检出更多（误检也多） |
| det_size | str | "640,640" | 检测输入分辨率（宽,高）：越高越慢，小脸更清楚；输出坐标已自动还原到原图坐标系 |
| max_num | int | 0 | 单图最多返回几张脸（0=不限） |
| use_crop | bool | false | 是否同时输出 5 点对齐后的 112×112 裁剪图（face_crop，ArcFace 标准输入）；一般不需要，默认关 |

构造级参数（代码层，不在 web 表单）：`model_root`（模型根目录）、
`pack_name`（模型包名）、`det_name`（检测模型文件名）。

### 2.2 链路阶段参数（`FaceStage`）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| margin | float | 0.15 | 从原图出人脸时的裁剪外扩比例（`Cropper.extract`，clamp 到图边界） |
| dedup_iou | float | 0.6 | 人脸框去重 IoU：上游重叠的目标框会让同一张脸进多个子图，超过它的重复脸只留一张 |

`image` 为模块测试页的图片下拉参数（test_only，不落库、不进链路）。

### 2.3 当前库内值（快照 2026-10）

- `FDET0001`（人脸检测·默认，独立）：`device=auto, det_thresh=0.5, det_size="640,640", max_num=0, use_crop=false, margin=0.15, dedup_iou=0.6`。
- `FDET0002`（人脸截取·人脸检测，PIPE0003，2026-09-23 新增）：同上但 `det_thresh=0.4`。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| img | ndarray (H×W×3, BGR) | 待检测图像。可以是整帧，也可以是上游裁出的目标区域图（如目标裁剪链路中的"人裁剪图"） |
| max_num | int，可选 | 单次调用覆盖构造值 |

## 4. 输出（字段和内容）

`detect()` → list[FaceDetection]（按置信度降序）：

| 字段 | 类型 | 内容 |
|---|---|---|
| bbox | (x1,y1,x2,y2) int | 人脸框，**输入图坐标系**（det_size 内部缩放已还原） |
| score | float | 检测置信度 [0,1] |
| kps | [(x,y)×5] float | 5 点关键点：右眼、左眼、鼻尖、右嘴角、左嘴角（输入图坐标） |
| face_crop | ndarray 112×112 BGR | ArcFace 模板对齐裁剪图（use_crop=true 时；喂给人脸嵌入） |
| index | int | 帧内序号 |
| width / height | int 属性 | 脸宽/脸高（像素） |

## 5. 上下游衔接

- 上游：任意 BGR 图（整帧 / YOLO 人框裁剪图）。
- 下游：face_crop 交给人脸嵌入提特征；bbox/kps 供画框或坐标还原
  （目标裁剪/人脸检测链路把子图坐标系的 kps/bbox 还原回原图）。
- `FaceStage`：消费上游「目标裁剪」的 `CropResult` 子图，产出原分辨率人脸
  （`FaceResult`：person_bbox / face_bbox / face_img / score / kps）；链路编排见 [人脸截取链路 PIPE0003](../chains/pipe0003_face_capture.md)。
  记录化（`to_record`）后旁路图像统一键名 `image`（旧名 `face_img` 兼容，指向同一 ndarray）。
- **端口声明（P5）**：`FaceStage.INPUTS = {"frame": "Image", "crops": "Records"}`、
  `OUTPUTS = {"faces": "Records"}`——产物 `face_detect.faces` 即产出的人脸记录列表（`image` 旁路＝
  原分辨率人脸图）。`face_detect.frame` / `face_detect.crops` 本期已纳入 `inputs` 解析，spec 可显式连线；
  **不声明＝现状默认**（frame←本帧原帧、crops←上游「目标裁剪」的 `last_crops`），零回归。
- **crops 元素类型归一**：`FaceStage.run` 入口经 `pipe.target_crop.as_crop` 归一——既收 `CropResult`
  **对象**（现路径 `last_crops`），也收裁剪**记录 dict**（`inputs` 连线取到的 `target_crop.crops`），
  两者行为逐字等价。

## 6. 验证

```bash
conda run -n ai python umvp/face_detect/test_face_detect.py   # 检测+对齐自检
conda run -n ai python tests/test_face_capture_pipeline.py    # PIPE0003 链路（真实模型）
```
注意：onnxruntime 的 `VerifyOutputSizes` 形状警告为 SCRFD 动态 batch 正常现象。
