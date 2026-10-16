# 人脸截取链路 PIPE0003

> 类型：模块自定义拼接链路 | 装配：`web_lab/server.py`（`_chain_from_spec` 装配，
> `_seed_pipeline` 播种） | 运行：`web_lab/test.html` 链路测试台
> 数据库：pipelines 表 `PIPE0003` + pipeline_steps 5 步（引用 5 条模块 preset）

## 1. 链路定位

保留链路之一（原分辨率人脸截取归档）：**帧管理 → YOLO 识人 → 检测筛选 → 目标裁剪(裁人) →
人脸检测（子图检脸 + 原图出脸）**。面向"把人脸抓出来存档"：截取落盘即是目的，
**不含嵌入 / 底库检索 / 事件尾**（身份去重不在本链路）。整链由模块 preset 拼装，无专属代码分支。

各阶段参数细节见模块分册（[帧管理](../modules/frame_manager.md)、[YOLO 识别](../modules/yolo.md)、
[检测筛选](../modules/det_filter.md)、[目标裁剪](../modules/target_crop.md)、
[人脸检测](../modules/face_detect.md)）。

## 2. 拼接结构（数据库存储）

```
PIPE0003 人脸截取链路（pipelines 表，streams=1）
 ├─ 步骤0  FM0003    人脸截取·帧管理     （frame_skip=5）
 ├─ 步骤1  YOLO0003  人脸截取·YOLO识人   （classes=0，只识人，纯推理）
 ├─ 步骤2  DF0003    人脸截取·检测筛选   （long>=30；group_limits 人≥1）
 ├─ 步骤3  CROP0003  人脸截取·目标裁剪   （classes=0，按人框裁人，out_size="" 原尺寸）
 └─ 步骤4  FDET0002  人脸截取·人脸检测   （FaceStage：子图检脸 + 原图出原分辨率人脸）
```

| 步骤 | preset | 阶段参数要点 |
|---|---|---|
| frame_manager | FM0003「人脸截取·帧管理」 | `frame_skip=5`，随抽帧节奏 |
| yolo | YOLO0003「人脸截取·YOLO识人」 | `classes=0`，只识人；`imgsz=1280` 兼容远角小目标（纯推理） |
| det_filter | DF0003「人脸截取·检测筛选」 | `filter="long >= 30"`（人长边≥30 才有效，小目标视为未识别） |
| target_crop | CROP0003「人脸截取·目标裁剪」 | `classes=0`（按人框裁人）、`out_size=""`、`margin=0.0` |
| face_detect | FDET0002「人脸截取·人脸检测」 | `det_thresh=0.4`、`margin=0.15`（出脸外扩）、`dedup_iou=0.6`（人脸框去重） |

链路只存"步骤引用哪些 preset + 顺序"（pipeline_steps），模块参数本体在各模块参数表；
每模块恰一份参数、链路恰一条，均由服务器启动幂等播种（`web_lab/server.py` `_SEED_FACE_STAGES`）。

## 3. 数据流（逐阶段记录怎么变）

```
视频帧 ──FM0003 要图决策（每 5 帧取 1，积压自适应放宽）
            │ 要图的帧（帧记录：frame_idx/ts）
            ▼
        YOLO0003 纯推理 → Detection 列表（bbox/score/cls_id=0/track_id）
            │
            ▼
        DF0003 检测筛选（人长边≥30 才有效；不达标视为"未识别"，原元素透传）
            │ 有效人体检测
            ▼
        CROP0003 目标裁剪（按人框裁人）→ CropResult 列表
            │ （旁路字段 img=人子图；bbox/offset/scale/cls_id/score/track_id 入主干）
            ▼
        FDET0002 人脸检测（FaceStage.run(frame, crops, ts)）
            │ ① 在人子图里跑 SCRFD 检脸（bbox/kps，子图坐标系）
            │ ② to_original() 把脸框/关键点还原回原图坐标
            │ ③ Cropper.extract 从**原帧**裁原分辨率人脸图（margin 外扩，零缩放）
            │ ④ 人脸框 IoU>0.6 的去重（上游重叠人框会让同一张脸进多个子图）
            ▼
        FaceResult 列表（person_bbox / face_bbox / face_img / score / kps / upscale / index）
            │ 逐张落盘为原分辨率人脸图（face_0001_..s.jpg）
            ▼
        链路输出 = 原分辨率人脸图 + timeline「face」条目（截取落盘即目的，无事件尾）
```

关键机制：

- **两级裁剪**：第一级「目标裁剪」按 YOLO 人框机械裁人子图（旁路字段 `img`）；
  第二级在人子图内检脸（识别模块内部），再把脸框还原回原图、从**原图**裁原分辨率人脸
  （`FaceStage` 归属人脸检测模块，见 [人脸检测](../modules/face_detect.md) §1）。
- **无事件尾**：本链路不以 `det_filter`/`seq_filter` 收尾，也不构造报送；输出即落盘人脸图。
  这与 PIPE0001（VLM+事件）、PIPE0002（车牌事件）不同。
- **旁路字段**：人子图与出的人脸图是图像旁路，不进 JSON 主干（记录契约见 `umvp/pipe/records.py`）。

## 4. 运行方式（test.html 链路测试台）

打开 `http://127.0.0.1:8001/test.html` → 视频测试页：
选视频 + 选链路（PIPE0003）→ 运行。选视频后信息条按帧管理的 `frame_skip` 与时间段预估
"将分析多少帧"。表单参数（`/api/test/pipeline_run`）同三条链路通用（video/image、max_frames、
start_time/end_time、output_dir/image_format、max_saved_images 等；本链路无 `vlm_feedback` 语义）。
命令行等价：

```bash
curl -s -X POST http://127.0.0.1:8001/api/test/pipeline_run \
  -H "Content-Type: application/json" \
  -d '{"params":{"pipeline_id":"PIPE0003","video":"face.MP4","max_frames":150}}'
```

> 实现注记：test.html 的人脸链由 `web_lab/server.py` 的人脸运行路径驱动
> （YOLO 识人 → 目标裁剪裁人 → `FaceStage` 检脸/出脸 → 落盘），单帧不经过通用 `Pipeline.step`。

## 5. 输出

- **原分辨率人脸图（链路输出）**：output_dir 下逐张落盘 `face_{序号}_{秒}s.{格式}`，
  前端结果网格同步显示（人脸图 data URL）。
- **timeline 结构化时间线**：逐条 `{kind, text, data, ts, frame}`，
  kind `face`（检出的人脸，data=FaceResult.to_dict：person_bbox/face_bbox/face_size/score/kps/upscale）
  与 `info`（过程信息，如"裁出 N 张人子图""检出人脸 N 张"）；`frame` 为帧决策。
- **落盘上限**：`max_saved_images`（默认 200 张）生效，超限跳过写盘。

## 6. 验证

```bash
# 链路核心（帧管理门控 + 逐帧截取 + 同帧去重 + 装配；真实模型）
conda run -n ai python tests/test_face_capture_pipeline.py

# 人脸检测模块自检
conda run -n ai python umvp/face_detect/test_face_detect.py
```

（断言数不写死，以脚本运行自报为准。）
注意：onnxruntime 的 `VerifyOutputSizes` 形状警告为 SCRFD 动态 batch 正常现象。
