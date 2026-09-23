# 模块文档索引（doc/modules/）

> 每个模块/链路一篇，体例统一：模块定位 → 参数 → 输入 → 输出 → 上下游衔接 → 验证。
> 参数以本目录为准；链路与播种事实见 `doc/数据库结构.md`，总体约定见 `CLAUDE.md`。

## 管道模块（`umvp/pipe/composer.py`）

| 文档 | 模块 | 代码 | 主要 preset |
|---|---|---|---|
| [帧管理.md](帧管理.md) | FrameManager 取帧节奏 + 背压 | `FrameManager` | FM0001/0002/0003 |
| [YOLO识别.md](YOLO识别.md) | 小模型识别 + 过滤 + 按类报送 | `YOLODetector` | YOLO0001/0002/0003 |
| [目标裁剪.md](目标裁剪.md) | 按上游检测框裁子图（输出分辨率可配） | `crop_targets` / `TargetCrop` | CROP0001+ |
| [标注.md](标注.md) | 在帧上画检测框（供 VLM 看"原图+画框"） | `annotate_frame` / `Annotator` | ANNO0001+ |
| [VLM研判.md](VLM研判.md) | 大模型研判 + 素材准备 | `VLMAnalyzer` | VLM0001 |
| [输出处理.md](输出处理.md) | **输出处理**统一漏斗（七道闸 + 规则 DSL） | `OutputPolicy`（原告警策略；module_id 仍为 `alarm`，preset 名如 AL0001「警用无人机·告警策略」） | AL0001/AL0002 |

## 人脸链路（`umvp/face_detect/` `umvp/face_embed/` `umvp/face_scan/`）

| 文档 | 模块 | 代码 |
|---|---|---|
| [人脸检测.md](人脸检测.md) | SCRFD 检测 + 5 点对齐（FaceStage：目标子图检脸 + 原图出脸） | `FaceDetector` / `FaceStage` |
| [人脸嵌入.md](人脸嵌入.md) | ArcFace 512d 嵌入 | `FaceEmbedder` |
| [向量底库.md](向量底库.md) | numpy 线性检索底库 | `FaceStore` |

## 车牌识别（`umvp/plate_recog/`）

| 文档 | 模块 | 代码 |
|---|---|---|
| [车牌识别.md](车牌识别.md) | HyperLPR3 识别 + 原图出车牌小图（含 PIPE0002 链路） | `PlateRecognizer` / `extract_plates` / `PlateStage` |

## 保留链路专篇

| 文档 | 链路 |
|---|---|
| [警用无人机链路.md](警用无人机链路.md) | PIPE0001 帧管理→YOLO→标注→VLM→输出处理 |
| [车牌识别.md](车牌识别.md) §8 | PIPE0002 帧管理→YOLO识车→目标裁剪→车牌识别→输出处理 |
| [目标裁剪.md](目标裁剪.md) §5 | PIPE0003 帧管理→YOLO识人→目标裁剪→人脸检测 |

## 通用工具（跨链路共用，无独立 preset）

| 工具 | 代码 | 说明 |
|---|---|---|
| `Cropper` 通用裁剪 | `umvp/pipe/cropper.py` | 按框裁图可缩放 / 坐标还原 / 原分辨率提取；人脸、车牌链共用（旧名 `CropRestore`/`PersonCrop` 为兼容别名） |

## 相关文档（本目录之外）

- 资源估算：`umvp/resources.py`（设计见 `doc/需求文档-开发版.md` §三；实测标定 `web_lab/calibrate.py`）
- 数据库结构 / 存储设计：`doc/数据库结构.md`、`doc/存储设计.md`
- 链路运行器与测试台：`doc/端口与启停手册.md`
- 选型调研（历史）：`doc/research/`
