# 人脸截取链路 PIPE0003

> 类型：模块自定义拼接链路 | 装配：`web_lab/server.py`（`_chain_from_spec` 装配，
> `_seed_pipeline` 播种） | 运行：`web_lab/test.html` 链路测试台（专用 `_face_step` 路径）
> 数据库：pipelines 表 `PIPE0003` + pipeline_steps 5 步（引用 5 条模块 preset）

## 1. 链路定位

保留链路之一（原分辨率人脸截取归档）：**帧管理 → YOLO 识人 → 检测筛选 → 目标裁剪(裁人) →
人脸检测（子图检脸 + 原图出脸）**。面向"把人脸抓出来存档"：截取落盘即是目的，
**不含嵌入 / 底库检索 / 事件尾**（身份去重不在本链路）。

spec 由模块 preset 拼装，链路本身无专属代码；但**测试台的运行路径有人脸专属分支**
（`_face_step`）——它不经过通用 `Pipeline.step`，因而本链路的 `det_filter` 阶段
**声明但不执行**。这是本链路与 PIPE0001 / PIPE0002 最关键的区别，详见
[§4.1 两条执行路径的差异](#41-两条执行路径的差异核心)与文末[已知差异（待决事项）](#已知差异待决事项)。

各阶段参数细节见模块分册（[帧管理](../modules/frame_manager.md)、[YOLO 识别](../modules/yolo.md)、
[检测筛选](../modules/det_filter.md)、[目标裁剪](../modules/target_crop.md)、
[人脸检测](../modules/face_detect.md)）。

## 2. 拼接结构（数据库存储）

```
PIPE0003 人脸截取链路（pipelines 表，streams=1）
 ├─ 步骤0  FM0003    人脸截取·帧管理     （frame_skip=5）
 ├─ 步骤1  YOLO0003  人脸截取·YOLO识人   （classes=0，只识人，纯推理）
 ├─ 步骤2  DF0003    人脸截取·检测筛选   （long>=30；group_limits 人≥1）——测试台路径不执行
 ├─ 步骤3  CROP0003  人脸截取·目标裁剪   （classes=0，按人框裁人，out_size="" 原尺寸）
 └─ 步骤4  FDET0002  人脸截取·人脸检测   （FaceStage：子图检脸 + 原图出原分辨率人脸）
```

| 步骤 | preset | 阶段参数要点 |
|---|---|---|
| frame_manager | FM0003「人脸截取·帧管理」 | `frame_skip=5`，随抽帧节奏 |
| yolo | YOLO0003「人脸截取·YOLO识人」 | `classes=0`，只识人；`conf=0.3`、`imgsz=1280` 兼容远角小目标（纯推理） |
| det_filter | DF0003「人脸截取·检测筛选」 | `filter="long >= 30"`（人长边≥30 才有效）、`group_limits="0:1,300"`（人≥1） |
| target_crop | CROP0003「人脸截取·目标裁剪」 | `classes=0`（按人框裁人）、`out_size=""`、`margin=0.0` |
| face_detect | FDET0002「人脸截取·人脸检测」 | `det_thresh=0.4`、`margin=0.15`（出脸外扩）、`dedup_iou=0.6`（人脸框去重） |

链路只存"步骤引用哪些 preset + 顺序"（pipeline_steps），模块参数本体在各模块参数表；
每模块恰一份参数、链路恰一条，均由服务器启动幂等播种（`web_lab/server.py`
`_SEED_FACE_STAGES`，`_seed_pipeline` → `_seed_chain`）。规则与门控的完整字段见
[数据库结构](../guides/database_schema.md) §7 与各模块分册。

> ⚠ **det_filter 声明但测试台不执行**：DF0003 在 spec / 数据库里存在，但测试台人脸链走
> `_face_step` 直连 YOLO → 目标裁剪，**跳过 `pipe.det_filter`**（详见 §4.1）。PIPE0001 /
> PIPE0002 及不含人脸阶段的链路经 `Pipeline.step` 执行，det_filter 正常生效。

## 3. 数据流（逐阶段记录怎么变）

以下按**测试台实际执行**的路径描述（`_face_step`）；被跳过的 det_filter 以虚框标注。

```
视频帧 ──FM0003 要图决策（每 5 帧取 1，积压自适应放宽）
            │ 要图的帧（驱动层 pipe.frame.wants_frame 门控；帧记录：frame_idx/ts）
            ▼
        YOLO0003 纯推理（_face_step 内 pipe.yolo.infer(frame, track=False)）
            │ → Detection 列表（bbox/score/cls_id=0；track 不启用）
            │  ┌─────────────────────────────────────────────┐
            │  │ DF0003 检测筛选（long>=30）——测试台路径**不执行** │
            │  │ 全部 dets 直接下探（小人框 long<30 也进裁剪）      │
            │  └─────────────────────────────────────────────┘
            ▼
        CROP0003 目标裁剪（f_crop.run(frame, dets, ts)，按人框裁人）→ CropResult 列表
            │ （旁路字段 img=人子图；bbox/offset/scale/cls_id/score 入主干）
            ▼
        FDET0002 人脸检测（fstage.run(frame, crops, ts)，FaceStage）
            │ ① 在人子图里跑 SCRFD 检脸（bbox/kps，子图坐标系）
            │ ② to_original() 把脸框/关键点还原回原图坐标
            │ ③ Cropper.extract 从**原帧**裁原分辨率人脸图（margin 外扩，零缩放）
            │ ④ 人脸框 IoU>dedup_iou(0.6) 去重（上游重叠人框会让同一张脸进多个子图）
            ▼
        FaceResult 列表（person_bbox / face_bbox / face_img / score / kps / upscale / index）
            │ 逐张落盘为原分辨率人脸图（face_{序号}_{ts}s.{格式}）
            ▼
        链路输出 = 原分辨率人脸图 + timeline（仅 frame_manager / target_crop / face_detect 三段）
```

关键机制：

- **两级裁剪**：第一级「目标裁剪」按 YOLO 人框机械裁人子图（旁路字段 `img`）；
  第二级在人子图内检脸（识别模块内部），再把脸框还原回原图、从**原图**裁原分辨率人脸
  （`FaceStage` 归属人脸检测模块，见 [人脸检测](../modules/face_detect.md) §1）。
- **人脸链走专用路径**：测试台的 `_face_step` 直接编排 YOLO → 目标裁剪 → FaceStage，
  不经 `Pipeline.step`；`Pipeline.step` 本身不装配人脸阶段（`FaceStage` 只由 web 人脸路径构造）。
- **无事件尾**：本链路不以 `det_filter`/`seq_filter` 收尾，也不构造报送；输出即落盘人脸图。
  这与 PIPE0001（VLM+事件）、PIPE0002（车牌事件）不同。
- **旁路字段**：人子图与出的人脸图是图像旁路，不进 JSON 主干（记录契约见 `umvp/pipe/records.py`）。

## 4. 运行方式（test.html 链路测试台）

打开 `http://127.0.0.1:8001/test.html` → 视频/图片测试页：
选链路（PIPE0003）+ 选视频/图片 → 运行。表单参数（`/api/test/pipeline_run`）同三条链路通用
（video/image、max_frames、start_time/end_time、output_dir/image_format、max_saved_images 等；
本链路无 `vlm_feedback` 语义）。命令行等价：

```bash
curl -s -X POST http://127.0.0.1:8001/api/test/pipeline_run \
  -H "Content-Type: application/json" \
  -d '{"params":{"pipeline_id":"PIPE0003","video":"face.MP4","max_frames":150}}'
```

### 4.1 执行路径（2026-10-16 起统一）

本链与其它链路一样走**通用 `Pipeline.step`**：装配层（`_chain_from_spec`）按 `face_detect`
阶段参数构造 `FaceStage` 注入 `Pipeline.face` 插槽，逐帧执行

```
frame_manager → yolo → det_filter → target_crop → face_detect
（若链路另配 seq_filter/其它阶段，同样按插槽顺序执行）
```

因此 **`det_filter`（DF0003）真正生效**：「裁出 N 张人子图」与最终人脸数都反映过滤后的检测
（小人框不再进入裁剪与检脸）。驱动层（`h_pipeline_run`）只负责**展示与落盘**：从
`pipe.last_faces` 出人脸图、计数、去重与 timeline。

> **平台规矩**（链路壳标准 §0）：逐帧分析阶段**一律**经 `Pipeline` 插槽执行，**禁止**驱动层
> 专用单帧路径；`validate_chain` 的「**声明即执行**」断言兜底——spec 声明的阶段若未被装配
> 即报错。历史例外见文末「历史差异与修复」。

> 注：`/api/test/chain`（`h_chain`）与 `/api/test/pipeline_run` 共用同一装配器
> （`_chain_from_spec`，`server.py:1677` / `:1589`）；测试台**选链路运行**用人脸链入口
> `/api/test/pipeline_run`（`test.html` 的 `fetch('/api/test/pipeline_run', …)`），人脸图的
> 展示与落盘由它承担。

### 4.2 timeline 阶段标签与落盘

timeline 条目由 `_stage_order`（`server.py:1029`）驱动，人脸链现在**全阶段**可见：

- `frame`（`stage=frame_manager`）：本帧放行；
- `yolo` / **`det_filter`**（`stage=yolo` / `det_filter`）：检出与筛选统计（如
  `有效性筛选 10 → 0（筛掉 10 条）`；仅在确有筛掉时出现）；
- `info`（`stage=target_crop`）：`裁出 N 张人子图`（N 为**过滤后**）；
- `face`（`stage=face_detect`）：每张人脸一条 `人脸 #{n}（WxH）`，外加一条
  `检出人脸 N 张，人脸 M 个` 汇总。

落盘命名 `face_{序号:03d}_{ts:.1f}s.{格式}`（序号为累计检出人脸数）；结果网格 images 上限
**36** 张；全局落盘上限 `max_saved_images` 默认 **200 张**，原图/标注图/人脸图共用计数，
超限跳过写盘。人脸链**只落人脸图**（关键帧原图/标注图落盘对人脸链跳过）。

## 5. 输出

- **原分辨率人脸图（链路输出）**：output_dir 下逐张落盘 `face_{序号}_{秒}s.{格式}`，
  前端结果网格同步显示（人脸图 data URL，最多 36 张）。
- **timeline 结构化时间线**：逐条 `{kind, text, stage, to, stage_label, to_label, data, ts, frame}`，
  kind `face`（检出的人脸，data=FaceResult.to_dict：person_bbox/face_bbox/face_size/score/kps/upscale）
  与 `info`（过程信息，如"裁出 N 张人子图""检出人脸 N 张"）；`frame` 为帧序数。
  **测试台人脸链的 timeline 不含 det_filter 阶段条目**（该阶段未执行）。
- **落盘上限**：`max_saved_images`（默认 200 张）生效，超限跳过写盘。
- 结果头含「处理 N 帧 | 截取人脸 N 张」，与另两条链路的「事件 N 个」不同（本链路无事件尾）。

## 6. 验证

```bash
# 链路核心（帧管理门控 + 逐帧截取 + 同帧去重 + 装配；真实模型）
conda run -n ai python tests/test_face_capture_pipeline.py

# 人脸检测模块自检
conda run -n ai python umvp/face_detect/test_face_detect.py

# web 链路（真实 YOLO）：test.html 选 PIPE0003 → 运行，或命令行
curl -s -X POST http://127.0.0.1:8001/api/test/pipeline_run \
  -H "Content-Type: application/json" \
  -d '{"params":{"pipeline_id":"PIPE0003","video":"face.MP4","max_frames":150}}'
```

（断言数不写死，以脚本运行自报为准。）
注意：onnxruntime 的 `VerifyOutputSizes` 形状警告为 SCRFD 动态 batch 正常现象。

## 历史差异与修复（2026-10-16，已关闭）

**曾经的差异**：2026-10-16 之前，测试台人脸链走 `web_lab/server.py` 的专用单帧路径
`_face_step`（门控 `is_face_chain` = stages 含 `face_detect`/`face_embed`/`extract_faces`），
绕过通用 `Pipeline.step`：

```python
dets  = pipe.yolo.infer(frame, track=False)   # YOLO 只识人（track 关闭）
crops = f_crop.run(frame, dets, ts)           # dets 直接进裁剪（不过 det_filter）
faces = fstage.run(frame, crops, ts)          # FaceStage 检脸 → 原图出脸 → 去重
```

**后果**：spec 声明的 `det_filter`（DF0003，`filter="long >= 30"`）**声明了却不执行**（也不
emit 其阶段日志）——小人框仍被裁剪检脸（不崩溃、截取正常，代价是少量无效算力 / 可能的低质
脸图）。仅**测试台**路径受影响：PIPE0001/0002 及无人脸阶段的自定义链一直走 `Pipeline.step`，
过滤正常。来源：pipe-chain 收件箱 INB-0008 → 工单 **WO-0001**（测试与文档）→ **WO-0002.1**
（修复，P5 第 0 步）。

**修复内容**（2026-10-16）：

- `Pipeline` 新增 `face` 插槽（与 `plate` 对称）→ 拼接层注入 `FaceStage`，人脸链并入
  `Pipeline.step`；
- `server.py` 删除 `_face_step` / `is_face_chain` 专用路径（展示/落盘/计数留驱动层）；
- `validate_chain.py` 新增「**声明即执行**」断言（声明的阶段未装配 → 报错），防止同类问题复现；
- 链路壳标准写入「逐帧阶段一律走插槽」硬规则。

**验证**：`tests/test_face_capture_pipeline.py` **63 项断言**（含 det_filter 生效断言：严格
过滤 → `last_filter=(1,0)`、0 裁剪、0 人脸）；`tests/test_validate_chain.py` 13/13（含
「声明但未装配 → 报错」回归用例）；`test_composer` 36、`test_plate_recog_pipeline` 28、
`test_police_uav_pipeline`、`test_agent_runtime` 39 全绿。

- 相关阅读：[检测筛选](../modules/det_filter.md)（过滤规则本身）、
  [目标裁剪](../modules/target_crop.md) §5（本链路中的裁剪阶段）、
  [人脸检测](../modules/face_detect.md) §5（FaceStage 衔接）、
  [拼接层端口协议设计稿](../design/assembly_ports.md)（P5，本案是其动因）。
