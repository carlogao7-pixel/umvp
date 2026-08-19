# UMVP 独立部署包 — 安装指南

> 本指南覆盖两种场景：
> **场景 A（本机现状）**——项目已在项目根（本目录），代码与权重齐全，
> 直接继承现有 conda 环境 `ai` 运行，无需任何安装。
> **场景 B（移植新机器）**——整目录拷贝后从零装依赖，走通验证测试。
>
> 代码本身无需编译（纯 Python）；模型权重已随包携带（`models/` + `face_models/buffalo_l/`）。

---

## 一、包结构（代码库清单）

```
umvp/                                ← 项目根（独立部署包）
├── umvp/                            ← 模块根（四模块 + 人脸管道，纯 Python + 推理封装）
│   ├── pipe/
│   │   ├── composer.py              ← 核心：四模块 + compose() 装配器（Frame/YOLO/VLM/Alarm）
│   │   └── test_composer.py         ← 纯逻辑测试（不加载模型），27 项断言
│   ├── face_detect/
│   │   ├── face_detector.py         ← SCRFD 人脸检测 + 5点对齐裁剪（默认读项目内 face_models/）
│   │   ├── test_face_detect.py      ← 人脸检测验证（--image/--video/--camera）
│   │   └── test_imgs/               ← 人脸测试图片（Tom_Hanks / 群像 / 口罩）
│   ├── face_embed/
│   │   ├── face_embedder.py         ← ArcFace 512d 特征提取（L2 归一化，同项目内模型路径）
│   │   ├── face_store.py            ← 向量底库：注册/检索/持久化（numpy 线性检索）
│   │   └── test_face_embed.py       ← 底库注册/查询/自检（--register-dir/--query/--self-check）
│   └── face_scan/
│       ├── face_monitor.py          ← 人脸管道编排（流/视频/摄像头；取帧+背压复用 pipe.composer.FrameManager）
│       └── test_face_scan.py        ← 视频人脸扫描验证
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
├── tests/
│   ├── test_3modes.py               ← 主实机测试：三模式真实视频 + YOLO + 大模型
│   ├── test_yolo_params.py          ← YOLO 推理参数对比（重叠场景截图）
│   ├── 分析模式测试.md              ← 测试配置说明
│   └── data/                        ← 测试视频（face1 / rescue1 / traffic）+ 重叠截图
├── requirements.txt                 ← Python 依赖清单
├── README.md                        ← 包说明
└── INSTALL.md                       ← 本指南
```

依赖关系（模块 → 第三方库）：
- `pipe/composer.py` → 纯 Python（抽帧+背压已内置于 FrameManager，无第三方依赖）
- `face_detect/face_detector.py` → insightface + onnxruntime + numpy + cv2
- `face_embed/face_embedder.py` → insightface + onnxruntime + numpy
- `face_embed/face_store.py` → 仅 numpy
- `face_scan/face_monitor.py` → cv2 + numpy
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

# 3) 纯逻辑测试冒烟（27 项断言，不加载模型，秒级完成）
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
```

---

## 四、运行验证（按依赖从小到大）

```bash
cd <项目根>
PY="conda run -n ai python"   # 便携：ai conda 环境解释器（或先 conda activate ai 后用 python）
```

```bash
# 1) 纯逻辑测试（不加载任何模型，最快）→ 应输出「全部通过: 27 项断言」
$PY umvp/pipe/test_composer.py

# 2) 人脸链路注册+自检（读项目内 face_models/；test_imgs 建库并逐条自检 → 7/7 命中）
$PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz

# 3) YOLO 推理参数对比（重叠场景截图，真实推理 yolov8n.pt）
$PY tests/test_yolo_params.py

# 4) 三模式实机测试（YOLO + 三路真实视频 + 大模型）
$PY tests/test_3modes.py
```

### 注意事项

- **VLM 端点可达性**：`tests/test_3modes.py` 里 `VLM_URL` 指向开发机上的 qwen3-vl 服务。
  目标机需能访问该地址；否则改 `VLM_URL`/`VLM_MODEL` 指向自己的 OpenAI 兼容大模型端点。
- **运行目录**：从项目根运行（测试脚本内部按根目录定位模块与数据）。
- **首次推理慢**：YOLO 首帧加载模型约 1~2 秒；SCRFD/ArcFace 首次加载 onnx 同理，属正常。
- **性能基线（全 CPU）**：YOLO 单帧 ~1.7s（首载后更低）、SCRFD 640 全图 90–130ms、
  ArcFace 单张 ~10ms、千级底库检索 <1ms。WSL2 下接近该量级。

---

## 五、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `ImportError: libGL.so.1: cannot open shared object file` | 缺 opencv 图形库 | 装 `libgl1` 等（3.2 节 apt） |
| `ImportError: libgomp.so.1` | torch 缺 OpenMP | 装 `libgomp1` |
| `ModuleNotFoundError: ultralytics/insightface` | 依赖没装或环境没激活 | 确认 `pip install -r requirements.txt` + `conda activate ai` |
| 人脸模块报「未找到 det_10g.onnx」 | `face_models/buffalo_l/` 缺失或未解压 | 解压 `face_models/buffalo_l.zip` 到 `face_models/`；或检查是否用 `model_root` 覆盖到了不存在路径 |
| `test_3modes` 的 VLM 步骤报错 | 端点不可达/超时 | 改 `VLM_URL`，或先跑 1)/2)/3) 跳过 VLM 验证管道 |
| `RuntimeError: Found no NVIDIA driver` | torch 装了 CUDA 版但机器无 GPU | 重装 CPU 版 torch，或用 `device=cpu` |
