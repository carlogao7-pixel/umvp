# 参数决策表（选型初值）

> 读者：组装工 ｜ 用途：落 `params` 初值时的依据。状态：拼接决策矩阵随经验续填；YOLO 参数初值已实测。

## 1. 拼接决策（先看输入，再定模块组合）

输入：有无 VLM / 识别目标 / 输出事件需求 / 成本约束 → 输出：模块组合（阶段清单）+ 关键参数初值。

典型组合：

| 需求形态 | 模块组合（示例） |
|---|---|
| 仅统计 / 计数（无 VLM） | `frame_manager → yolo → det_filter`（`group_limits` / `events`） |
| 行为研判（VLM） | `frame_manager → yolo → det_filter → annotate → vlm → seq_filter`（事件尾） |
| 车牌识别 | `frame_manager → yolo(车) → det_filter → target_crop(车) → plate_recog → seq_filter` |
| 人脸截取 | `frame_manager → yolo(人) → det_filter → target_crop(人) → face_detect` |
| 人脸身份 | 上者 + `face_embed → face_store` |

> 模块与参数白名单以 `list_modules` 现取为准；本表只给组合骨架与方向。

## 2. YOLO 检测 / 跟踪参数初值（实测，2026-08-12）

> 环境：RTX 3060 Laptop GPU、模型固定 `yolov8n.pt`、4K 源素材；数据 `out/yolo_bench{2,3,4,5}/`。
> 适用于任何链路的 YOLO 阶段（与选哪种组合无关）：先看场景再落值。

| 场景 | imgsz | conf | iou | 依据 |
|---|---|---|---|---|
| 近景干净人群（大目标、高置信） | 640 够用 / 1280 更稳 | 0.35 基线；0.50 控误检 | 0.7 | imgsz > conf > iou 区分度 |
| 远角小目标（车 / 人） | **1280 默认开**（4× 召回） | 0.15 找回远角人；0.50 丢目标 | 0.7 | 检出 52→230、track 1→7 |
| 少量行人小目标（4K 源） | **1280 必开**（640 下 track 全灭） | 0.15 提检出量，救不了 640 | 0.7 | 640 全组 0 track |
| 人群密集 / 重叠 | 未实测 | 未实测 | **0.9 待验**（防合并） | iou 仅重叠素材有区分度 |

**通用原则**：
- 区分度排序：**imgsz 最强**、conf 次之（只影响检出量、不影响 track 能力）、iou 仅重叠场景有效。
- GPU 下 1280 vs 640 耗时几乎持平（约 +26ms/帧内），分辨率是"几乎免费"的召回提升。
- **验收看 track 数而非检出数**：检出多 ≠ 跟踪好。
- 远角场景车是主力目标（人极难过检）；检远角人需 conf 0.15 + imgsz 1280，或含车时
  `classes="0,2"` 让大目标撑起 track 骨架。

## 3. 链路形态备忘

- 人脸：YOLO(识人) → 目标裁剪(裁人+坐标还原) → SCRFD(检脸) → ArcFace(嵌入) → FaceStore(检索)。
- 监控：抽帧 + 人脸检索持续扫（`face_scan.FaceMonitor`）。

## 4. 决策原则

先看"有没有 VLM / 要哪种输出事件 / 预算"再定模块组合；参数给范围并说明动机；方案可对比。
