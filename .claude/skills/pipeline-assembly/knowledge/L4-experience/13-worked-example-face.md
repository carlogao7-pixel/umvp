# 先例：原分辨率人脸提取链路（face-extract-pipeline）

> 层：L4-experience ｜ 读者：总装 / 组装工 ｜ 来源：原 `.claude/skills/face-extract-pipeline/SKILL.md`（2026-08-11 并入知识库 L4）
> 状态：WIP 记录（2026-08-05 首版）。只验证过三张静态测试图，很多场景没试过，下文
> "未验证场景"一节是诚实清单。本文件目的是把已跑通的链路、改造、坑先记下来，别丢。

## 这是什么

把"图片 → 识别人 → 裁人 → 识别人脸 → 原分辨率人脸图"串成一条可复用的链路。
最终产物：每张人脸一张**原生分辨率**图片（像素直接取自原图，零缩放）+ 原图坐标元数据。

```
原图 → YOLODetector(classes=[0] 只识别人)
     → CropRestore.crop_person()          # 按人裁图（可选放大）
     → FaceDetector.detect(裁剪图)         # SCRFD 识别人脸
     → CropRestore.to_original()          # 人脸框坐标还原回原图坐标系
     → CropRestore.extract_face()         # 从原图直接裁出原生分辨率人脸
     → FaceResult（人脸图 + 原图坐标 + 来源人框 + kps + score）
```

**拼装成本（最小代价）**：链路 = 已有模块**零改动**（YOLODetector / FaceDetector 原样复用，
会话记录仅 Read）+ **1 个新胶水文件**（`umvp/pipe/crop_restore.py`，约 200 行）+ 1 个测试壳。
全部新增逻辑收敛在胶水文件的"坐标还原 + 原生分辨率出图"两件事上；剩下的只是参数选择
（upscale / det_thresh）。完整清单见"改造记录"。

## 文件与分工

| 文件 | 作用 |
|---|---|
| `umvp/pipe/crop_restore.py` | **新增**。CropRestore 工具类（裁人/坐标还原/原生分辨率出图）+ `extract_faces()` 编排函数（串 YOLO→裁人→人脸检测→去重） |
| `tests/test_face_extract_pipeline.py` | **新增**。链路测试：三张图跑通 + 落盘人脸图/标注图 + 27 项断言 |
| `tests/data/out/faces/` | **新增**。产物目录：`{图片}/face_{序号}_{conf}.png` + `annotated.png`（绿=人框 红=人脸框 蓝=关键点） |
| `umvp/face_detect/face_detector.py` | 零改造，复用（输入 BGR、输出 bbox/kps/face_crop） |
| `umvp/pipe/composer.py` 的 YOLODetector | 零改造，复用（infer 返回 Detection.bbox） |

## 怎么用

```bash
PY="conda run -n ai python"   # 便携写法（ai conda 环境解释器，不写机器绝对路径）
$PY tests/test_face_extract_pipeline.py           # 跑三张测试图，产物在 tests/data/out/faces/
```

代码骨架（模块拼装，不写分支）：

```python
from face_detect.face_detector import FaceDetector
from pipe.composer import YOLODetector
from pipe.crop_restore import CropRestore, extract_faces

yolo = YOLODetector(model_path="models/yolov8n.pt", conf=0.35, iou=0.7,
                    imgsz=640, max_det=300, classes=[0])
face_det = FaceDetector(det_thresh=0.4)
tool = CropRestore(upscale=1.0, margin=0.15)

faces = extract_faces(img_bgr, yolo, face_det, tool)
for f in faces:                       # f.face_img = 原分辨率人脸图
    print(f.to_dict())                # 原图坐标 + 尺寸 + conf + kps + 来源人框
```

## 关键知识点（设计依据，别改丢）

1. **"原分辨率" = 人脸像素永远取自原图**。放大(upscale)只作用于喂给人脸检测器的裁剪图，
   用来提升 SCRFD 对小脸召回；检出框坐标**除以放大倍数 + 裁剪偏移**还原回原图，再从原图裁。
   不要在放大图上直接存图，那不是原分辨率。
2. **SCRFD 的 bbox 已在"输入图坐标系"内**（face_detector.py 内部处理 det_size 缩放并还原），
   所以 裁剪图坐标 → 原图坐标 只需 `x/scale + offset`，没有别的魔法。
3. **按人裁图再检脸**的意义：整图 SCRFD 会把大图缩到 det_size=640，远距小脸直接丢失；
   裁人后脸在裁剪图里的占比变大，小脸可检出。人重叠2 里约 5px 的脸就是这样被找回来的。
4. **upscale 实测效果**（det_thresh=0.4，三张图）：1.0x→0/4/8 张、2.0x→1/4/8 张、3.0x→1/4/8 张。
   2.0x 已能找回 5px 级小脸，3.0x 无额外收益（受限于底库像素本身太少）。
5. **去重必要**：YOLO 重叠人框会让同一张脸进入多个裁剪，extract_faces 按人脸框 IoU>0.6 去重。
6. **det_thresh 权衡**：0.4 会带出贴近阈值的退化检测（关键点挤成一团的误检）；0.5（默认）更干净。
   结构化校验方法：5 点关键点应满足 眼 y ≈ 眼 y < 鼻 y < 嘴 y，且双眼 y 差 <6px。
7. **设备**：YOLO 走 ultralytics 自动选 GPU（RTX 3060，~0.02s/帧）；SCRFD 走 onnxruntime，
   本机回落 CPUExecutionProvider（CPU，整图 ~0.1s 级）。两者设备选择独立，别混为一谈。
8. **人脸框可能超出人框边缘**（侧脸/低头）→ extract_face 的 margin 外扩 + clamp 到图边界；
   人框本身也可能出图 → crop_person 先 clamp。

## 改造记录（2026-08-05）——拼装成本清单

目的验证：**最小代价把已有模块拼成新链路**。结论：现有模块零改动，额外改造 = 1 个胶水文件 + 1 个测试壳。

**新增物（全部额外改造）**：
1. **`umvp/pipe/crop_restore.py`**（约 200 行，唯一需要写的业务代码）：
   - `PersonCrop` / `FaceResult` 两个 dataclass（FaceResult 带 `to_dict()` 便于落盘/上报）
   - `CropRestore` 类：`crop_person` / `to_original` / `kps_to_original` / `extract_face`
   - `extract_faces()` 编排函数：YOLO infer → 过小人过滤(min_person_short=30) → 逐人裁图 →
     SCRFD detect → 坐标还原 → IoU 去重(dedup_iou=0.6) → 原图出图
2. **`tests/test_face_extract_pipeline.py`**：集成烟测壳——三张图全链路 + 落盘 + 27 项断言。

**未动物（零代码改动，会话转录仅有 Read 记录）**：
- `umvp/pipe/composer.py`（YOLODetector）——原样复用；`extract_faces` 连 import 都不需要，鸭子类型传入
- `umvp/face_detect/face_detector.py`（FaceDetector）——原样复用

**测试覆盖的真实边界（诚实记录）**：27 项断言里 24 项（框在图内 / 尺寸=框尺寸）由构造保证恒真，
3 项去重有实际意义；坐标映射正确性实际依赖 `annotated.png` 目检 + 会话内 ad hoc 几何核验，未固化为自动断言。
要固化需补：`to_original` 带期望值的单元测试、upscale>1 还原路径（1.0x 时 scale=1，除法等效恒等变换，没真测到）、
边界/空输入用例。已列入"未验证场景"跟踪。

**过程中修掉的两个自写脚本 bug（非模块 bug）**：断言里 `all()` 用在单个布尔上、
三张图传相同 ts 导致被 check_interval 抑制报送（改独立时间戳）。

## 已验证结果（2026-08-05，三张图）

| 图片 | 检出人 | 检出人脸(1.0x) | 人脸尺寸 | 备注 |
|---|---|---|---|---|
| 人遮挡1.png | 10 | 8 | 15–22px | 12 张检测里 1 张关键点退化（conf=0.416，疑误检），余 11 张结构有效 |
| 人重叠.png | 4 | 4 | 6–10px | 全部结构有效 |
| 人重叠2.png | 2 | 0（2.0x 时 1） | ~5px | 脸太小，放大后找回 1 张 |

坐标映射核验：全部人脸框在来源人框内、裁剪非空白、关键点在脸框内且结构有序 → 映射正确。

## 未验证场景（WIP，先记录，别当已支持）

- [ ] 视频流/抽帧链路（FrameScheduler 接入，而非静态图）
- [ ] 大规模图片吞吐与内存（当前逐张处理）
- [ ] 超分模型接入（Real-ESRGAN 级）替代 cv2 放大，以及 4x+ 放大的收益边界
- [ ] 批处理 detect_batch 的收益验证
- [ ] 人重叠2 剩余 1 张 5px 级人脸为何放大后仍检不出（像素太少，需超分而非插值）
- [ ] 关键点退化检测的自动过滤策略（结构校验自动剔除 vs 仅调 det_thresh）
- [ ] 真实遮挡（半身/侧脸/低头）对 YOLO 人框和 SCRFD 命中率的影响
- [ ] margin 行为：人框顶到图边缘时人脸被 clamp 截断的对策
- [ ] 移动端 / 纯 CPU 部署的性能预算
- [ ] 与 FaceEmbedder（ArcFace 112x112）衔接：FaceResult.face_img 直接喂 embedding 的可用性
- [ ] 测试强化：`to_original` / upscale>1 带期望值的单元测试（当前 27 断言中 24 项构造恒真，详见"改造记录"）
