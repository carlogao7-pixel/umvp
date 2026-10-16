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

### 4.1 两条执行路径的差异（核心）

测试台的人脸链**不走通用 `Pipeline.step`**，而由 `web_lab/server.py` 的专用 `_face_step`
（`h_pipeline_run`，路由 `/api/test/pipeline_run`）驱动。门槛是 `is_face_chain`
（`server.py:1057`）：

```python
is_face_chain = any(k in stage_params for k in ("face_embed", "face_detect", "extract_faces"))
```

即**只要链路 stages 含 `face_detect` / `face_embed` / `extract_faces` 任一**（不论保留链路
PIPE0003 还是用户自定义链）即使入该路径。`_face_step`（`server.py:1108-1152`）每帧：

```python
dets  = pipe.yolo.infer(frame, track=False)   # YOLO 只识人（track 关闭）
crops = f_crop.run(frame, dets, ts)           # dets 直接进目标裁剪（不过 det_filter）
faces = fstage.run(frame, crops, ts)          # FaceStage 检脸 → 原图出脸 → 去重
```

图片测试走 `_face_step`（`server.py:1304-1306`）；视频测试同样（`server.py:1385-1390`，
`:1386` 注释「人脸链：extract_faces 内部自跑 YOLO 识人，不走 pipe.step」）。

两条路径对照：

| 环节 | 测试台人脸链（`_face_step`） | 通用 `Pipeline.step`（PIPE0001/0002 及无人脸阶段链） |
|---|---|---|
| 入口 | `h_pipeline_run`（`/api/test/pipeline_run`，video/image） | 同函数非人脸分支（`:1317`/`:1396`）、`h_chain`（`:1758`）、`h_pipeline_stream`（`:1637`） |
| 帧管理 | 驱动层 `pipe.frame.wants_frame` 门控（循环内） | 同（`force=True` 时由调用方先门控） |
| YOLO | `pipe.yolo.infer(frame, track=False)`（`_face_step` 内） | 调用方 `infer(frame, track=pipe.track_needed())` 后传入 |
| **det_filter** | **不执行**（DF0003 被跳过；也不 emit 其阶段日志） | **执行**，并 emit `det_filter` 阶段日志（`:1167-1171`） |
| target_crop | `f_crop.run`（独立实例，读 `target_crop` 阶段参数） | `pipe.step` 内 `self.crop.run`（`:420-423`） |
| face_detect | `fstage.run`（`FaceStage`，`_face_step` 专用） | Pipeline 不装配人脸阶段（face_detect 在 step 中为空操作） |
| seq_filter / 事件尾 | 不执行（无此阶段） | 执行（`:417`/`:454-456`） |
| timeline 阶段标签 | 仅 `frame_manager` / `target_crop` / `face_detect` | 全阶段（含 `det_filter`/`seq_filter`） |

> 注：`/api/test/chain` 的 `h_chain`（模块链路运行器）**没有**人脸分支，恒定走 `pipe.step`；
> 它执行 det_filter，但不驱动 `FaceStage`、也不出人脸图。测试台选链路运行用的是
> `/api/test/pipeline_run`（见 test.html 的 `fetch('/api/test/pipeline_run', …)`）。

### 4.2 timeline 阶段标签与落盘

`_face_step` 只发三类 timeline 条目（阶段标签由 `_stage_order` 驱动，`server.py:1029`）：

- `frame`（`stage=frame_manager`）：本帧放行；
- `info`（`stage=target_crop`）：`裁出 N 张人子图`；
- `face`（`stage=face_detect`）：每张人脸一条 `人脸 #{n}（WxH）`，外加一条
  `检出人脸 N 张，人脸 M 个` 汇总。

**不产生 `yolo` / `det_filter` 阶段条目**。落盘命名 `face_{序号:03d}_{ts:.1f}s.{格式}`
（`server.py:1139`，序号为累计检出人脸数）；结果网格 images 上限 **36** 张
（`server.py:1145`）；全局落盘上限 `max_saved_images` 默认 **200 张**（`server.py:1008`），
原图/标注图/人脸图共用计数，超限跳过写盘（`_save_image`，`:1011-1017`）。

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

## 已知差异（待决事项）

- **测试台人脸链跳过 det_filter**：PIPE0003 的链路 spec 声明了 det_filter（DF0003，
  `filter="long >= 30"`），但测试台走 `_face_step` 时既不调用 `pipe.det_filter`、也不 emit
  其阶段日志。故小人框（人长边 < 30）仍会被裁剪并检脸——**不崩溃、截取正常**，代价是少量
  无效算力 / 可能的低质脸图；`检测筛选` 分册描述的过滤规则在本测试台路径实际不生效。
- 该差异为**当前代码的既成事实**，来源工单 **WO-0001**（pipe-chain 收件箱 INB-0008）。
  修法方向（web 侧人脸链改走 `step` / 提供 step 化的人脸路径入口 / 调整链路定义）由后续工单
  裁定，本篇只如实记录现状，不预设方案。
- 相关阅读：[检测筛选](../modules/det_filter.md)（过滤规则本身）、
  [目标裁剪](../modules/target_crop.md) §5（本链路中的裁剪阶段）、
  [人脸检测](../modules/face_detect.md) §5（FaceStage 衔接）。
