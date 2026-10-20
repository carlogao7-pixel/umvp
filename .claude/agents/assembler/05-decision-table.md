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
| 人脸身份 | 上者 + `face_embed`（**链路终段**；⚠ `face_store` 检索**未纳入链路 spec**，声明会被 `validate_spec` 拒绝） |

> 模块与参数白名单以 `list_modules` 现取为准；本表只给组合骨架与方向。

**VLM 送审素材（`image_source` 选「哪张」× `image_pick` 选「几张」）**：
默认 `image_source=annotated`（带框图，＝现状）；要 VLM 看原帧用 `frame`；要它只看目标子图用 `crops`
（该组合须同时含 `target_crop`，即上表"行为研判"行可加一段 `target_crop`）。`crops` 按类报送时收集
**该类全部命中目标**的子图（图集），`image_pick` 决定送 1 张还是 N 张：

| `image_source` | 场景 → `image_pick` |
|---|---|
| `annotated`（默认） | 一般行为研判 / 场景级判断——送整帧带框图；单图源，`first`（`all` 等价） |
| `frame` | 需看原始画面（无框遮挡）——单图源，`first` |
| `crops` + `first`（默认） | **单主体 / 成本敏感**：只送该类**首张**子图，省 token（＝旧行为，零回归） |
| `crops` + `all` | **多主体同帧需逐一看清**（多车牌 / 多人行为）：送该类**全部**子图（多图 payload） |

> 缺省不变、缺源回退，细节见 `doc/modules/vlm.md` 与 `01-shell-standard.md §2`。**取哪张用
> `image_source`、取几张用 `image_pick`；`vlm.inputs.image` 不支持**（图片源统一走 `image_source`）。

**产物连线（`inputs`）用法**（端口协议，见 `01-shell-standard.md §3`）：默认各阶段已按**固定流水**接好
（`det_filter` / `seq_filter` ← `dets`，`target_crop` / `annotate` ← `frame` + `dets`），
**不写 `inputs` = 现状**；只在想让某阶段**取上游另一环**时才显式连线（每端口一个上游产物，值须字符串）：

- `annotate.dets = det_filter.dets` → 标注画"**同帧筛选后**"的框、**忽略时序门控**（`seq_filter` 的
  节流 / 变化）；不写则 annotate 吃检测流末端（det_filter → seq_filter 之后）。
- `annotate.dets = yolo.dets` → 画**筛选前**的原始检出（调试图省 / 看漏检）。
- `target_crop.dets = yolo.dets` → 按**原始检出**裁剪（跳过 `det_filter` 有效性筛选）。
- **VLM 两旋钮正交**：`image_source` 管**送哪张图**，`vlm.inputs.dets` 管**报送哪些检测**（决定按哪些
  目标报送、从而决定 `crops` 命中谁）——组合示例：
  - `image_source=crops` + `vlm.inputs.dets = yolo.dets` → 对**筛选前全部检出**逐个裁子图送审
    （不写 `dets` 则用筛选后）；配 `image_pick=all` 把该类全部目标子图一次送审。
  - `image_source=crops` + `vlm.inputs.dets = det_filter.dets` → 只对**有效性筛选后**的目标裁子图送审。

> **多数场景无需连线**（默认即现状）；端口与类型以模块 `INPUTS/OUTPUTS` 声明为准。本期可连线端口：
> `yolo/det_filter/seq_filter/target_crop/annotate` ＋ **`plate_recog`/`face_detect`**（消费上游裁剪）
> ＋ `vlm.dets`（**仅此端口**；`vlm.inputs.image` 不支持）；未纳入解析名单现为**空**。详见
> `01-shell-standard.md §3`。

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

- 人脸：YOLO(识人) → 目标裁剪(裁人+坐标还原) → SCRFD(检脸) → ArcFace(嵌入，**链路终段**)；
  `face_store` 检索**不在链路 spec 内**（声明即被 `validate_spec` 拒绝，仅专用运行器 / 自身测试页用）。
- 监控：抽帧 + 人脸检索持续扫（`face_scan.FaceMonitor`）。

## 4. 决策原则

先看"有没有 VLM / 要哪种输出事件 / 预算"再定模块组合；参数给范围并说明动机；方案可对比。
