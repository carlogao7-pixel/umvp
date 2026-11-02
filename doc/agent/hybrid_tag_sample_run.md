# 混合检索标签体系真数据演练记录（网络视频 1.5 分钟街景）

> 定位：WO-0033 混合检索标签体系的**端到端真数据演练存档**——向量化 + 三识别
> （YOLO 观察段 / 车牌识别 / 人脸截取）在同一视频上跑通，记录向量以外的全部
> 结构化数据。设计依据 [hybrid_retrieval.md](../design/hybrid_retrieval.md)
> （§2.5 标签 schema / §3 路径 A / §4 存储），标签字段定义以设计稿为准，本文只存实例。
> 日期：2026-11-02 ｜ 执行：orchestrator

## 1. 素材与运行一览

| 项 | 值 |
|---|---|
| 素材 | Pexels #19899275（墨西哥城街景，平视，车流+行人，CC0 许可） |
| 文件 | `tests/data/vid/20261102_170000_city_street.mp4`（62MB，H.264 1920×1080@30，**86.5s**） |
| 命名 | 按采集时间规范命名 `YYYYMMDD_HHMMSS_描述.mp4` → 触发 §2.6 墙钟时间解析 |
| 链路 | PIPE0004（向量化+观察段，mock 向量）/ PIPE0002（车牌）/ PIPE0001（警情）/ PIPE0003（人脸截取） |

| 链路 | 结果 |
|---|---|
| PIPE0004 | 2592 帧抽 173 帧 → **29 片段**、2588 检出、29 标签行落表（`review_index`） |
| PIPE0002 | 519 帧抽检 → **1 条车牌事件**（粤苏866AM2）归档并回填 |
| PIPE0001 | 173 帧 → 0 警情（稀疏抽帧下 seq_filter 阈值未触发，符合预期） |
| PIPE0003 | **201 张人脸截取**落盘（`tests/data/out/test_pipeline/face_*.jpg`）+ 链路尾提炼块 |

## 2. 视频级标签（`review_video`，采集时间首次真实验证）

```json
{"video": "20261102_170000_city_street.mp4",
 "video_time": "2026-11-02T17:00:00+08:00",
 "video_time_src": "filename",
 "indexed_at": "2026-11-02T11:24:00+08:00",
 "vector_model": "", "clip_dim": 1024, "frame_dim": 256,
 "duration": 86.586}
```
`video_time` 由 `label_align.parse_video_time` 从文件名解析（§2.6 来源优先级：文件名 > 标签）。

## 3. 判断层：事件归档 → 片段标签

**事件归档** `out/events/20261102_170000_city_street.mp4.jsonl`（一行一次运行）：

```json
{"run_id": "8f19ef2d6a79", "pipeline": "PIPE0002", "at": "2026-11-02T11:18:18+08:00",
 "video": "20261102_170000_city_street.mp4",
 "events": [{"name": "粤苏866AM2", "kind": "车牌", "ts": 26.693,
             "detail": {"key": "…", "count": 1, "span": 0.0, "window_n": 0},
             "plate_no": "粤苏866AM2"}]}
```

**对齐落表后**（relabel 只重对齐不重嵌入），车牌落判断层 `plates[]`（ts=26.693 → 窗
[24.024, 27.027)）：

```json
{"video": "20261102_170000_city_street.mp4", "t_start": 24.024, "t_end": 27.027,
 "plates": [{"ts": 26.693, "plate_no": "粤苏866AM2"}],
 "events": [], "plate_nos": [], "face_count": 0}
```

> 说明：HyperLPR3 面向中式车牌训练，本素材为墨西哥街景，86 秒仅 1 次命中（车牌样式
> 偶然接近），命中号"粤苏866AM2"为识别输出原样——真数据演练以如实记录为准。

## 4. 观察层：YOLO 检出统计 + 原始记录流

**窗口聚合 `det_stats`（派生缓存）+ `class_names`**——第 0 窗（[0, 3.003)）：

```json
{"det_stats": {"0": {"avg": 9.667, "peak": 15, "tracks": 0},
               "1": {"avg": 0.667, "peak": 1, "tracks": 0},
               "2": {"avg": 3.667, "peak": 4, "tracks": 0},
               "3": {"avg": 1.0,   "peak": 2, "tracks": 0},
               "7": {"avg": 0.333, "peak": 1, "tracks": 0},
               "9": {"avg": 2.0,   "peak": 2, "tracks": 0},
               "24": {"avg": 0.333, "peak": 1, "tracks": 0},
               "26": {"avg": 0.333, "peak": 1, "tracks": 0}},
 "class_names": {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle",
                 "7": "truck", "9": "traffic light", "24": "backpack", "26": "handbag"}}
```
> 街景密度真实体现：person 峰值 15/帧。`tracks=0` = 观察段 YOLO 未开跟踪（§2.5 约定）。

**原始记录流 `frame_dets`**——同窗首帧（dict 主干原样，6 帧 17KB）：

```json
[{"ts": 0.0, "recs": [
    {"ts": 0.0, "bbox": [1510, 764, 1919, 1073], "kind": "det", "label": "car",
     "score": 0.903, "cls_id": 2, "track_id": 0, "frame_idx": 0},
    …共 24 条…]},
 {"ts": 0.6, "recs": […]}, …]
```
体积：本视频 29 窗 `frame_dets` 合计约 **493KB**（繁密街景，约 16KB/窗），与设计估算同量级。

## 5. 人脸识别产出（PIPE0003，截取落盘即目的）

- 201 张人脸截取图：`tests/data/out/test_pipeline/face_<帧号>_<秒>s.jpg`
- 链路尾 `output` 提炼块同步产出（`output_001_face_2.4s_1.jpg` 等，WO-0029 提炼）
- **标签侧说明**：`face_count` 观察列需人脸观察段（帧管理→YOLO(人)→裁人→face_detect→
  video_embed，§2.4 可插拔组合）才会计数，本期播种的是 YOLO 观察段，故 `face_count=0`；
  人脸数据当前以落盘图片形态存在（PIPE0003 即目的），接入人脸观察段后即进标签。

## 6. 检索验证（过滤式，查询向量走 mock）

| 查询 | 结果 |
|---|---|
| `event_name=粤苏866AM2` | **1 命中**：窗 [24.024, 26.527)，why=「车牌 粤苏866AM2 命中；向量相似度 -0.02」 |
| 时间窗 16:00–18:00（墙钟） | 命中含目标视频（why=「视频 … 命中；时间窗命中；向量相似度 …」）；无采集时间的视频按 §2.6 保留并 why=「该视频无采集时间，未参与时间窗过滤」 |
| `video=本视频 + 时间窗` | 3 命中全部落在本视频，why 双信号齐备 |

墙钟时间窗为**首次真实验证**：`ts_abs = video_time + t_start` 判定链路全通。

## 7. 向量化状态与网络诊断（重要）

- 本轮片段向量走 **mock 档**（29 clips 已入库，`out/video_vecs.npz` 共 117 clips）；
- **真实嵌入被网络阻断**，诊断结论：
  - WeMM 服务本身正常（2B，`/v1/models` 200），小请求（文本 ≤500KB）可通；
  - **≥600KB 载荷必断**、持续请求间歇性 ConnectionError/ConnectTimeout，直连备用 IP
    （绕过 MagicDNS）同样复现 → 指向 **Tailscale 数据面**被干扰（代理 TUN 截流或
    Windows 侧 Tailscale 客户端状态异常——`tailscale status` 显示 Logged out 可疑）；
  - 联调当日（10-28）MB 级载荷可通，佐证为环境退化非代码问题。
- **修复建议**：①检查 Windows Tailscale 客户端登录/连接状态；②代理软件将
  `100.64.0.0/10` 与 `ca0.tail952a82.ts.net` 加入直连/绕过规则（TUN 模式必须）。
- **恢复后重跑**：参数表 `preset_params_video_embed`（VEMB0001）当前为
  `use_real_embed=0 / max_resolution=448,448 / jpeg_quality=50 / frames_embed=0`
  （低载荷挡位）；网络修复后改回 `use_real_embed=1`（分辨率/质量可视带宽回调 896/85、
  frames_embed=1），重跑 PIPE0004 即以真实向量同键覆盖，标签自动重算。

## 8. 过程中发现并修复的现场问题

1. **preset 参数表重同步坑**：对 `presets.params_json` 写部分字段后，启动时参数同步会把
   未覆盖列写成 NULL——`store_path/thumb_dir` 被清空导致 PIPE0004 只落标签**不落底库**、
   不出缩略图。修复：参数表行直接补 `store_path='out/video_vecs.npz'`、
   `thumb_dir='out/video_thumbs'`。教训：**改 preset 参数应对参数表整行操作**，
   避免只改 presets 主表触发非对称同步（已记 INB-0139）。
2. **frames_embed 不在参数 schema**（无列，开关不可配）——现场 `ALTER TABLE` 补列
   `frames_embed tinyint(1)` 解决；正式登记（PARAMS schema + 列）待 web 子单收口。
3. 查询侧 `use_real_embed` 走 auto 语义（端点死 → fail fast）——端点不可达期间检索
   报错属预期行为；演示时查询体显式 `"use_real_embed": false` 走 mock 查询向量。

## 9. 数据清点（本演练净增）

| 项 | 量 |
|---|---|
| `review_index` 行 | +29（累计 117） |
| `review_video` 行 | +1（累计 4） |
| 向量库 clips | +29（累计 117，mock；真实向量待网络修复后覆盖） |
| 事件归档 | 1 jsonl / 1 事件 |
| 人脸截取图 | 201 张 + output 提炼块 |
| 缩略图 | 29 张（`out/video_thumbs/20261102_*`） |
| `frame_dets` 原始记录流 | ~493KB（29 窗） |
