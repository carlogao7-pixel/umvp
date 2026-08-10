# UMVP 独立部署包

**UMVP（Unified Modular Vision Pipeline / 统一模块化视觉分析管道）**的独立部署包：
四模块管道（帧管理 / YOLO 识别 / VLM 研判 / 告警策略）+ 人脸识别检索分支 + 三种分析
模式实机测试。代码库与模型权重自包含于本目录，不依赖 spaz 平台，也不依赖用户目录
下的模型缓存（人脸模型默认读项目内 `face_models/buffalo_l/`）。

## 内容速览

| 目录 | 内容 |
|------|------|
| `umvp/pipe/` | 四模块 + `compose()` 装配器（small_only / small_crop / small_full / large_only） |
| `umvp/face_detect/` | SCRFD 人脸检测 + 5 点对齐裁剪（`FaceDetector`） |
| `umvp/face_embed/` | ArcFace 512d 嵌入（`FaceEmbedder`）+ 向量底库（`FaceStore`） |
| `umvp/face_scan/` | 抽帧决策（`FrameScheduler`）+ 人脸管道编排（`FaceMonitor`） |
| `models/` | YOLO 模型选择池（v8n/s、v9t、v10s、v11n/s/m，COCO person/car，放 .pt 即入选择） |
| `face_models/` | InsightFace 模型包 `buffalo_l/`（已解压，含 det_10g.onnx / w600k_r50.onnx 等） |
| `tests/` | 三模式实机测试 + YOLO 参数对比 + 测试视频/图片 |

## 环境与运行

本机已有 conda 环境 `ai`（python 3.10，含 cv2 / ultralytics / insightface / onnxruntime），
直接继承使用，无需新装依赖。所有运行命令使用该解释器：

```bash
# 在项目根目录下

# 1. 纯逻辑测试（不加载模型，最快）→ 27 项断言
conda run -n ai python umvp/pipe/test_composer.py

# 2. 人脸链路注册+自检（读项目内 face_models/；用 test_imgs 建库并逐条自检 → 7/7 命中）
conda run -n ai python umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz

# 3. YOLO 推理参数对比（重叠场景截图，真实推理）
conda run -n ai python tests/test_yolo_params.py

# 4. 三模式实机测试（YOLO + 三路真实视频 + 大模型研判，需 VLM 端点可达）
conda run -n ai python tests/test_3modes.py
```

> 若 `ai` 环境已激活（`conda activate ai`），上述命令简写为 `python ...`。
> 测试脚本内部按项目根定位模块与数据，请从本项目根目录运行。

## 移植到另一台机器

代码与权重完全自包含，整目录拷贝即可；目标机需具备：
- Python 3.10 环境（`requirements.txt` 列出全部依赖，`pip install -r requirements.txt`）
- 系统库 `ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libgomp1`（open cv / torch 运行必需）

完整步骤与排查见 **INSTALL.md**。

## 与原仓库的对应关系

| 本包 | 原仓库 |
|------|--------|
| `umvp/` | `uav/umvp_standalone/umvp/`（模块根，仅改人脸模型默认路径） |
| `tests/test_3modes.py` | `uav/umvp_standalone/tests/test_3modes.py`（适配路径） |
| `tests/test_yolo_params.py` | `uav/umvp_standalone/tests/test_yolo_params.py`（适配路径） |
| `models/` | `spaz/custom_model/yolov8n.pt` 等 |
| `face_models/buffalo_l.zip` | `tinymodel/buffalo_l.zip` |

本包为独立 Python 包，用 git 做版本管理（仓库根即本目录）；大二进制
（模型权重/测试视频/MySQL 数据）不入库，克隆后按 INSTALL.md 从上游重建。
与原仓库（spaz/uav）的差异同步仍由手工维护。
