# 模块目录与接口速查

> 层：L1-shared ｜ 读者：总装 / 组装工 ｜ 权威源：umvp/pipe/composer.py、umvp/face_*/、umvp/resources.py

## 管道四模块（产品按 stages 自定义拼接；`compose()` 四模式为代码层兼容封装）

> 注：模块测试页的测试参数（测试图片 / 演示回填 / 事件脚本 / 模拟参数，PARAMS 标
> `test_only`）不属于模块参数、不落库、不进链路设计器，下表只列模块自身参数。

| 模块 | 作用 | 关键参数（默认值） |
|---|---|---|
| FrameManager 帧管理 | 取帧节奏 + 背压 | sampling(analysis/wall_clock/frame_count)、frame_skip(3)、interval_sec(3.0)、interval_frames(30)、queue_threshold(32)、backpressure_multiplier(2)、adaptive_relax_ratio(1.2)、adaptive_recover_ratio(0.5) |
| YOLODetector 小模型 | 框出目标（COCO：0=人 2=车） | model_path、conf(0.35)、iou(0.7)、imgsz(640)、max_det(300)、classes("0")、class_limits("0:1,300")、check_interval(3.0)、roi(裁剪送审)、track_change_only(仅在 track_id 变化时报送) |
| VLMAnalyzer 大模型 | 语义研判 + 素材准备 | crop_padding(0.15)、max_resolution("1280,720")、prompt、ref(crop:cls0/full:cls0/full:global)、use_real_vlm、vlm_endpoint、vlm_model(qwen3-vl-32b) |
| AlarmPolicy 告警 | 判定 + 冷却 | kind(window/dwell/inspection)、target_actions("fire,fight")、smooth_frames、hit_ratio(1.0)、alarm_cooldown(60.0)、cooldown_by_action(按警情类型独立冷却) |

## 人脸链路

| 模块 | 作用 | 关键参数（默认值） |
|---|---|---|
| FaceDetector（SCRFD） | 检脸 + 5 点对齐 | det_thresh(0.5)、det_size(640,640)、max_num |
| FaceEmbedder（ArcFace） | 512 维特征 | device(auto) |
| FaceStore | 向量底库检索 | db_path(out/face_db.npz)、thresh(0.45)、topk(5) |
| crop_restore.extract_faces | 原分辨率人脸提取 | upscale(1.0)、margin(0.15)、min_person_short(30)、dedup_iou(0.6) |

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
