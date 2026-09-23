# 模块目录与接口速查

> 层：L1-shared ｜ 读者：总装 / 组装工 ｜ 权威源：umvp/pipe/composer.py、umvp/face_*/、umvp/resources.py

## 管道模块（产品按 stages 自定义拼接；`compose()` 四模式为代码层兼容封装）

> 注：模块测试页的测试参数（测试图片 / 演示回填 / 事件脚本 / 模拟参数，PARAMS 标
> `test_only`）不属于模块参数、不落库、不进链路设计器，下表只列模块自身参数。

| 模块 | 作用 | 关键参数（默认值） |
|---|---|---|
| FrameManager 帧管理 | 取帧节奏 + 背压 | sampling(analysis/wall_clock/frame_count)、frame_skip(3)、interval_sec(3.0)、interval_frames(30)、queue_threshold(32)、backpressure_multiplier(2)、adaptive_relax_ratio(1.2)、adaptive_recover_ratio(0.5) |
| YOLODetector 小模型 | 框出目标（COCO：0=人 2=车）；过滤链在 infer() 输出处生效（视为未识别，对所有下游生效） | model_path、conf(0.35)、iou(0.7)、imgsz(640)、max_det(300)、classes("0")、filter_conf(0)、min_short(0)、min_long(0)、class_limits("0:1,300")、check_interval(3.0)、track_change_only(仅在 track_id 变化时报送) |
| TargetCrop 目标裁剪 | 按上游检测框裁子图（输出分辨率可配，默认原分辨率；只机械裁剪，不做有效性过滤） | out_size(""，640=长边/640,480=严格/×2=倍数)、margin(0.0)、classes("")、filter(""，条件DSL如 track_id == 1) |
| Annotator 标注 | 在帧副本上画检测框（供 VLM 看"原图+画框"） | thickness(2)、show_label(True)、classes("")、filter(""，条件DSL如 track_id == 1) |
| VLMAnalyzer 大模型 | 语义研判 + 素材准备 | max_resolution("1280,720")、prompt、ref(full:cls0/full:global)、use_real_vlm、vlm_endpoint、vlm_model(qwen3-vl-32b) |
| OutputPolicy 输出处理（原告警） | 统一输出漏斗：Finding→七道闸→事件 | kind(window/dwell/inspection)、task_id、target_actions("fire,fight")、window_frames(2，window)、dwell_sec(2.0，dwell)、hit_ratio(1.0)、min_conf(0)、rules(""，字段/操作符/值 DSL)、alarm_cooldown(60)、key_cooldown(-1，inspection 同对象只一次)、cooldown_by_action |

## 人脸链路

| 模块 | 作用 | 关键参数（默认值） |
|---|---|---|
| FaceDetector（SCRFD） | 检脸 + 5 点对齐 | det_thresh(0.5)、det_size(640,640)、max_num |
| FaceEmbedder（ArcFace） | 512 维特征 | device(auto) |
| FaceStore | 向量底库检索 | db_path(out/face_db.npz)、thresh(0.45)、topk(5) |
| face_detect.face_pipeline.FaceStage | 人脸检测阶段：目标子图检脸 + 原图出原分辨率人脸 | margin(0.15)、dedup_iou(0.6) |
| plate_recog.plate_pipeline.PlateStage | 车牌识别阶段：车图裁车牌 + 识别 + 原图出牌 | detect_level(high)、margin(0.1)、dedup_iou(0.6)、track_dedup(True)、track_cooldown(5.0) |

保留链路形态：
- PIPE0002「车牌识别」= 帧管理→YOLO识车→**目标裁剪(裁车)**→PlateStage→输出处理。
- PIPE0003「人脸截取」= 帧管理→YOLO识人→**目标裁剪(裁人)**→FaceStage（无嵌入/检索/告警，
  逐帧截取落盘）；含 face_embed 阶段则为身份去重提取形态（嵌入+相似度去重）。

## 通用工具（跨链路共用）

| 工具 | 作用 | 关键参数（默认值） |
|---|---|---|
| pipe.cropper.Cropper | 通用裁剪：crop(按框裁图可放大) / to_original(坐标还原) / extract(原分辨率提取)；人脸、车牌链共用，纯 cv2+numpy 零检测器依赖（旧名 CropRestore/PersonCrop 为兼容别名） | upscale(1.0)、margin(0.15) |
| pipe.cropper.iou | 框重叠去重（人脸/车牌链共用） | — |

## 资源估算（ResourceEstimator）

| 接口 | 作用 |
|---|---|
| estimate_module | 单模块指纹：显存 / 内存 / CPU / GPU 延迟 |
| estimate_pipeline | 链路聚合（每帧平均成本）+ 多路外推 |
| back_calculate | 预算反推：给显存预算 → frame_skip / check_interval 建议 |

## 监控链路

- face_scan.FaceMonitor：视频抽帧 + 人脸检索持续扫（参数：video、db_path、frame_skip、max_frames）。

## 常量速记（composer.py）

- 告警：ALARM_DWELL="dwell" / ALARM_WINDOW="window" / ALARM_INSPECTION="inspection"
- 采样：SAMPLING_ANALYSIS="analysis" / SAMPLING_WALL_CLOCK="wall_clock" / SAMPLING_FRAME_COUNT="frame_count"
- 报送粒度：GRAN_CLASS="class" / GRAN_SCENE="scene" / GRAN_TRACK="per_track"
