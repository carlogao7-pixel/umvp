# UMVP 独立部署包 — 安装指南

> 本指南覆盖两种场景：
> **场景 A（本机现状）**——项目已在项目根（本目录），代码与权重齐全，
> 直接继承现有 conda 环境 `ai` 运行，无需任何安装。
> **场景 B（移植新机器）**——整目录拷贝后从零装依赖，走通验证测试。
>
> 代码本身无需编译（纯 Python）；YOLO/人脸权重已随包携带（`models/` + `face_models/buffalo_l/`），
> 车牌权重需联网拉取一次（`umvp/plate_recog/fetch_models.py`，见 §3.5）。

---

## 一、包结构（代码库清单）

```
umvp/                                ← 项目根（独立部署包）
├── umvp/                            ← 模块根（管道模块 + 人脸管道，纯 Python + 推理封装）
│   ├── pipe/
│   │   ├── composer.py              ← 核心：模块 + 装配器（FrameManager/YOLODetector纯推理/VLM/DetFilter/SeqFilter/Pipeline编排）
│   │   ├── cropper.py               ← 通用裁剪 Cropper（按框裁图可缩放/坐标还原/原分辨率提取，纯 cv2+numpy）
│   │   ├── target_crop.py           ← 目标裁剪 TargetCrop（按上游检测框裁子图，输出分辨率可配）
│   │   ├── annotate.py              ← 标注 Annotator（在帧上画检测框，供 VLM 看"原图+画框"）
│   │   ├── det_filter.py            ← 检测筛选 DetFilter（有效性：逐条条件+成组数量判定）
│   │   ├── seq_filter.py            ← 时序判定 SeqFilter（窗口确认闸 + 事件标注）
│   │   ├── conditions.py            ← 条件 DSL 引擎（与/或/非/括号 + 派生 w/h/short/long/area，筛选/裁剪/标注共用）
│   │   ├── records.py               ← 通用记录 Record 约定 + to_record 对象适配
│   │   ├── crop_restore.py          ← extract_faces 兼容编排（已拆为「目标裁剪+人脸检测」两模块）
│   │   └── test_composer.py         ← 纯逻辑测试（不加载模型），85 项断言
│   ├── face_detect/
│   │   ├── face_detector.py         ← SCRFD 人脸检测 + 5点对齐裁剪（默认读项目内 face_models/）
│   │   ├── test_face_detect.py      ← 人脸检测验证（--image/--video/--camera）
│   │   └── test_imgs/               ← 人脸测试图片（Tom_Hanks / 群像 / 口罩）
│   ├── face_embed/
│   │   ├── face_embedder.py         ← ArcFace 512d 特征提取（L2 归一化，同项目内模型路径）
│   │   ├── face_store.py            ← 向量底库：注册/检索/持久化（numpy 线性检索）
│   │   └── test_face_embed.py       ← 底库注册/查询/自检（--register-dir/--query/--self-check）
│   ├── face_scan/
│   │   ├── face_monitor.py          ← 人脸管道编排（流/视频/摄像头；取帧+背压复用 pipe.composer.FrameManager）
│   │   └── test_face_scan.py        ← 视频人脸扫描验证
│   ├── plate_recog/
│       ├── plate_recognizer.py      ← HyperLPR3 车牌检测+矫正+识别（默认读项目内 plate_models/）
│       ├── plate_pipeline.py        ← 车牌链路阶段 PlateStage（消费上游车框：裁车→识别→坐标还原）
│       └── fetch_models.py          ← 拉取车牌模型到 plate_models/（新克隆/换机）
│   └── resources.py                 ← P1 资源估算（静态公式 + 实测指纹两层）
├── models/                        ← YOLO 模型选择池（web_lab 前端自动列出，COCO person=0/car=2）
│   ├── yolov8n.pt                 ← v8 轻量档（默认，CPU 单帧 ~0.2s）
│   ├── yolov8s.pt                 ← v8 均衡档（精度优先）
│   ├── yolov9t.pt                 ← v9 超轻量档（比 n 更小，检出略少）
│   ├── yolov10s.pt                ← v10 无 NMS 架构（均衡档）
│   ├── yolo11n.pt                 ← v11 轻量档（最新家族）
│   ├── yolo11s.pt                 ← v11 均衡档
│   └── yolo11m.pt                 ← v11 高精度档（CPU 单帧 ~8s，准实时/离线用）
├── face_models/
│   ├── buffalo_l.zip                ← InsightFace 模型包（原始压缩包，存档）
│   └── buffalo_l/                   ← 已解压（det_10g.onnx / w600k_r50.onnx 等，默认加载路径）
├── plate_models/                    ← 车牌模型（不入库；fetch_models.py 拉取重建）
│   └── hyperlpr3/20230229/onnx/     ← 检测 320/640 + 识别 + 颜色分类
├── web_lab/                       ← 可视化测试台（模块测试页 + 链路设计器 + 批量运行 + 资源估算）
│   ├── server.py                  ← stdlib HTTP 服务（REST + SSE；默认 8001）
│   ├── db.py                      ← MySQL 持久化（模块 preset / 链路 spec；惰性连接）
│   ├── validate_chain.py          ← 链路 stages spec 校验 CLI（产物/端口 DAG）
│   ├── selfcheck_vlm_env.py       ← 真实 VLM 端点环境变量注入自检（34 断言，不打网络）
│   ├── calibrate.py               ← 实测标定 → out/fingerprints.json（换机必跑）
│   └── ...
├── tests/
│   ├── test_police_uav_pipeline.py  ← 警用链路（警用无人机）逻辑断言
│   ├── test_police_uav_video.py     ← 警用链路端到端（真实 YOLO + mock VLM）
│   ├── test_yolo_params.py          ← YOLO 推理参数对比（重叠场景截图）
│   ├── test_yolo_standalone.py      ← YOLO 单模块探索性跑图（无断言计数）
│   ├── test_cropper.py              ← 通用裁剪模块纯逻辑断言（含轻量性检查，秒级）
│   ├── test_target_crop.py          ← 目标裁剪模块纯逻辑断言（裁剪/分辨率/条件 DSL）
│   ├── test_annotate.py             ← 标注模块纯逻辑断言（画框/条件 DSL）
│   ├── test_det_filter.py           ← 检测筛选纯逻辑断言（66）
│   ├── test_seq_filter.py           ← 时序判定纯逻辑断言（43）
│   ├── test_image_source.py         ← 送审图像源 + 图集契约（73）
│   ├── test_port_inputs.py          ← 端口连线端到端（49）
│   ├── test_validate_chain.py       ← spec 校验器（34）
│   ├── test_face_capture_pipeline.py← 人脸截取链路（PIPE0003，63）
│   ├── test_face_extract_pipeline.py← 人脸提取链路（YOLO 识人→目标裁剪裁人→SCRFD，28）
│   ├── test_plate_recog_pipeline.py ← 车牌识别链路（整帧 + 目标裁剪裁车→识别→坐标还原，28）
│   ├── test_agent_runtime.py        ← 挂载 agent 运行时（离线，39）
│   ├── test_video_transcode.py      ← 视频转码（18）
│   └── data/                        ← 测试素材：pic/（图片 6 张）、vid/（视频 8 段）、out/（产物）
├── requirements.txt                 ← Python 依赖清单
├── README.md                        ← 包说明
└── INSTALL.md                       ← 本指南
```

依赖关系（模块 → 第三方库）：
- `pipe/composer.py` → 纯 Python（抽帧+背压已内置于 FrameManager，无第三方依赖）
- `pipe/cropper.py` → 仅 numpy + cv2（零检测器依赖，人脸/车牌链共用）
- `pipe/crop_restore.py` → face_detect.face_detector（裁剪复用 cropper，人脸编排专属）
- `face_detect/face_detector.py` → insightface + onnxruntime + numpy + cv2
- `face_embed/face_embedder.py` → insightface + onnxruntime + numpy
- `face_embed/face_store.py` → 仅 numpy
- `face_scan/face_monitor.py` → cv2 + numpy
- `plate_recog/plate_recognizer.py` → hyperlpr3 + onnxruntime + numpy + cv2
- `tests/*.py` → ultralytics(YOLO) + cv2 + requests(VLM)

---

## 二、场景 A：本机直接运行（无需安装）

本机已具备全部运行条件，只查三步确认：

```bash
# 1) conda 环境存在（含 cv2/ultralytics/insightface/onnxruntime；无输出即 ok）
conda run -n ai python -c "import cv2, ultralytics, insightface"

# 2) 人脸模型已解压（默认加载路径=项目内，不读 ~/.insightface）
ls face_models/buffalo_l/
#    应看到 det_10g.onnx w600k_r50.onnx 等 5 个文件

# 3) 纯逻辑测试冒烟（85 项断言，不加载模型，秒级完成）
cd <项目根>
conda run -n ai python umvp/pipe/test_composer.py
```

通过后即可运行完整验证（见第五节），无需任何安装步骤。

> 人脸模型默认路径是**项目内** `face_models/buffalo_l/`（代码按 `__file__` 相对定位）。
> 若需改用其他目录（如 `~/.insightface/models`），仍可通过
> `FaceDetector(model_root=...)` / `FaceEmbedder(model_root=...)` 显式覆盖。

---

## 三、场景 B：移植到新机器（从零安装）

### 3.1 拷贝代码与权重

整目录拷贝即可（模型权重已包含）：

```bash
# 在目标机上，从源机拷贝（示例 scp；也可用 U 盘/共享盘；路径按源机实际位置）
scp -r <源机用户>@<源机IP>:/home/<源机用户>/projects/umvp ./
cd umvp
```

### 3.2 系统依赖（WSL2 Ubuntu，一次 apt）

```bash
sudo apt update
sudo apt install -y ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 unzip curl
```

说明：
- `libgl1 libglib2.0-0 libsm6 libxext6` — opencv-python wheel 运行时必需的图形库（缺 `libGL.so.1` 时 import cv2 直接崩）
- `ffmpeg` — 可选但推荐，视频解码/编码备用（cv2 自带后端也够用）
- `libgomp1` — torch 的 OpenMP 运行库（缺了加载 torch 报 GLIBC/GOMP 错误）

### 3.3 Python 环境

推荐 conda（与开发机一致）或 venv，Python 版本 **3.10**（insightface 1.0.1 对 3.11/3.12 兼容性不佳）。

方式 A：miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh    # 按提示完成，之后重开终端
conda create -n ai python=3.10 -y
conda activate ai
```

方式 B：venv（轻量）
```bash
sudo apt install -y python3.10 python3.10-venv python3-pip
python3.10 -m venv venv
source venv/bin/activate
```

### 3.4 Python 依赖

```bash
cd umvp
pip install -r requirements.txt
```

默认装 CPU 版 torch（约 2GB，含 ultralytics 全家）。若目标机有 NVIDIA GPU 且已装 CUDA 12.4 驱动：

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

> 提示：先确认驱动可用再装 GPU 版——`nvidia-smi` 有输出即可。CPU 版完全够跑通验证。

### 3.5 模型放置

```bash
# 人脸模型包已随包携带，确认解压目录就绪即可（默认加载路径=项目内）
ls face_models/buffalo_l/
#    若缺失（比如拷贝时漏了），从包内 zip 解压：
#    unzip face_models/buffalo_l.zip -d face_models/

# YOLO 模型已在 models/，无需移动

# 车牌模型（约 12MB）不在包内，联网拉取一次到项目内：
$PY umvp/plate_recog/fetch_models.py
#   产物: plate_models/hyperlpr3/20230229/onnx/（拉取后即可离线运行）
```

---

## 四、运行验证（按依赖从小到大）

```bash
cd <项目根>
PY="conda run -n ai python"   # 便携：ai conda 环境解释器（或先 conda activate ai 后用 python）
```

```bash
# 1) 纯逻辑测试（不加载任何模型，最快）→ 应输出「全部通过: 85 项断言」
$PY umvp/pipe/test_composer.py

# 1b) 通用裁剪模块（纯逻辑 + 轻量性检查 → 26 项断言）
$PY tests/test_cropper.py

# 2) 人脸链路注册+自检（读项目内 face_models/；test_imgs 建库并逐条自检 → 35/35 命中）
$PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz

# 3) YOLO 推理参数对比（重叠场景截图，真实推理 yolov8n.pt）
$PY tests/test_yolo_params.py

# 4) 警用链路端到端（真实 YOLO + mock VLM，落 tests/data/out/police_uav）
$PY tests/test_police_uav_video.py

# 5) 车牌识别链路（整帧识别 + YOLO 识车→裁车放大→坐标还原 → 28 项断言）
$PY tests/test_plate_recog_pipeline.py

# 6) 其余逻辑断言（不加载模型，秒级）：目标裁剪 35 / 标注 16 / 检测筛选 66 / 时序判定 43
$PY tests/test_target_crop.py; $PY tests/test_annotate.py
$PY tests/test_det_filter.py; $PY tests/test_seq_filter.py

# 7) 链路/端口契约类：图集与送审图像源 73 / 端口连线 49 / spec 校验 34 / 人脸截取 63 / 人脸提取 28
$PY tests/test_image_source.py; $PY tests/test_port_inputs.py
$PY tests/test_validate_chain.py; $PY tests/test_face_capture_pipeline.py
$PY tests/test_face_extract_pipeline.py

# 8) 挂载 agent 运行时（离线，39 项断言）；视频转码 18 项
$PY tests/test_agent_runtime.py; $PY tests/test_video_transcode.py

# 9) 链路 spec 校验 CLI（stages spec → [OK]/[ERR] + 退出码）
$PY web_lab/validate_chain.py <spec.json>

# 10) 真实 VLM 端点配置自检（环境变量注入机制，34 项断言，不打网络）
$PY web_lab/selfcheck_vlm_env.py
```

### 注意事项

- **VLM 端点可达性**：警用链路测试（`tests/test_police_uav_video.py`）默认用 mock VLM；
  测试台批量运行时的真实端点配置走链路 vlm 阶段参数（vlm_endpoint/vlm_model/vlm_key），
  或**直接导出环境变量**（优先级更高、key 不必入库）：
  `VLM_ENDPOINT` / `VLM_MODEL` / `VLM_API_KEY`（兼容 `GLM_API_KEY`），
  三者优先级＝环境变量 > 链路阶段参数 > 内置默认；端点不可达时测试台记日志继续
  （连续失败≥3 本轮改用素材包，401/403/429 给可读提示，每轮最多一条）。详见
  `doc/guides/runbook.md` §4.3。
- **运行目录**：从项目根运行（测试脚本内部按根目录定位模块与数据）。
- **首次推理慢**：YOLO 首帧加载模型约 1~2 秒；SCRFD/ArcFace 首次加载 onnx 同理，属正常。
- **性能基线（全 CPU，权威值见 `out/fingerprints.json`）**：YOLO @640 各档 40–280ms
  （yolov8n 39ms / yolo11n 48ms / yolo11s 100ms / yolo11m 276ms）、SCRFD 640 全图 ~100ms、
  ArcFace 单张 ~37ms、千级底库检索 <1ms。WSL2 下接近该量级。

---

## 五、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `ImportError: libGL.so.1: cannot open shared object file` | 缺 opencv 图形库 | 装 `libgl1` 等（3.2 节 apt） |
| `ImportError: libgomp.so.1` | torch 缺 OpenMP | 装 `libgomp1` |
| `ModuleNotFoundError: ultralytics/insightface` | 依赖没装或环境没激活 | 确认 `pip install -r requirements.txt` + `conda activate ai` |
| 人脸模块报「未找到 det_10g.onnx」 | `face_models/buffalo_l/` 缺失或未解压 | 解压 `face_models/buffalo_l.zip` 到 `face_models/`；或检查是否用 `model_root` 覆盖到了不存在路径 |
| `test_police_uav_video` 的 VLM 步骤报错 | 端点不可达/超时/鉴权/限额 | 配 `VLM_ENDPOINT`/`VLM_MODEL`/`VLM_API_KEY`（或链路 vlm 阶段参数）；或先跑 1)/2)/3) 跳过 VLM 验证管道 |
| `RuntimeError: Found no NVIDIA driver` | torch 装了 CUDA 版但机器无 GPU | 重装 CPU 版 torch，或用 `device=cpu` |
