# 部署变更台账（DEPLOYMENT.md）

用途：统一记录**影响项目部署**的改动，随代码一起移植，不依赖 git 历史。
维护规则：

- 只追加，不修改历史条目；新条目插在「## 记录」标题正下方（最新在前）。
- AI 在完成部署相关改动并验证后写入；拿不准算不算时，记一条无害。
- 与 CLAUDE.md「协作约定」中的部署台账规则配套使用。

## 判断清单（以下任一命中即为"影响部署"）

- 依赖：requirements.txt / pip 包新增、删除或版本变化
- 模型：models/*.pt、face_models/* 新增/移动/删除，或加载路径约定变化
- 环境变量：MYSQL_* 等新增/改名/默认值变化
- DB：web_lab presets 表结构、连接参数（db.py）变化
- 标定：out/fingerprints.json 重跑（换机/换卡）、性能基线数字变化
- 入口：新增/重命名/删除脚本、server.py 端口、CLI 参数
- 结构：新模块目录、`__file__` 相对路径依赖、包结构变化
- 配置：compose spec 新模式、影响运行结果的默认参数变化

仅代码逻辑/文档改动且不影响运行配置的，不必记录。

## 记录格式

每条以标题块形式插入「## 记录」正下方：

```
### YYYY-MM-DD · 一句话标题
- 变更内容：改了什么
- 影响面：依赖 / 模型 / 环境变量 / DB / 标定 / 入口 / 结构 / 配置（可多选）
- 部署注意：换机/移植/重启时需要做什么
- 验证：使用的命令与结果
```

## 记录

### 2026-09-23(七) · 识别对象有效性过滤统一到 YOLO 输出处；裁剪模块去过滤
- 变更内容：① `YOLODetector.infer()` 返回前套过滤链（`filter_conf`/`min_short`/`min_long`/
  自定义 `filters`）——不达标目标**视为"未识别"**，对**所有下游**（VLM 报送 / 目标裁剪·
  车牌·人脸传递 / 标注）统一生效，且**先于** `class_limits` 按类计数；`pass_dets` 不再过滤
  （入参已是干净输出）。② YOLO 恢复 `min_short`/`min_long` 参数（PARAMS/种子/handler/装配）。
  ③ 目标裁剪删除 `min_short`（参数+代码），回归纯机械裁剪+缩放；`crop_restore.extract_faces`
  兼容层自行过滤人框。④ 种子语义迁移：车牌 YOLO0002 `min_long=40`、人脸 YOLO0003
  `min_long=30`（原目标裁剪 min_short=40/30 的"长边≥N"语义）。
- 影响面：DB（yolo 参数表加回 min_short/min_long、target_crop 删 min_short）、配置（行为）
- 部署注意：重启自动补/删列；**已有库**的 YOLO0001/0002/0003 参数值已在本机直接迁移
  （结构未变的链路不会自动重播种）——换机若已有旧库，需确认 YOLO0002 `min_long=40`、
  YOLO0003 `min_long=30`、目标裁剪 CROP0002 `out_size=×2`。
- 验证：`umvp/pipe/test_composer.py`（60，含 infer 输出处过滤）、`tests/test_target_crop.py`
  （35）、`tests/test_plate_recog_pipeline.py`（28）、`tests/test_face_capture_pipeline.py`（39）、
  `tests/test_police_uav_pipeline.py`、`tests/test_annotate.py`（16）、`tests/test_cropper.py`（26）、
  `umvp/resources.py` 全过；装配抽查：infer 过滤小目标生效、CROP0002 `out_size=×2`、
  YOLO0002 `min_long=40`。

### 2026-09-23(六) · 模块参数优化：清理失效/冗余、背压档位、告警拆字段、目标裁剪合并尺寸
- 变更内容：① 帧管理 5 个背压细节参数（queue_threshold/backpressure_multiplier/max_frame_skip/
  adaptive_relax_ratio/adaptive_recover_ratio）收敛为 1 个「背压档位」`backpressure`
  （off/standard/aggressive），并接上链路装配（此前配了不生效）；composer 新增
  `BACKPRESSURE_LEVELS`/`backpressure_params`。② 删除 VLM「裁剪外扩比例」crop_padding
  （roi 移除后已失效；VLMAnalyzer/_vlm_material 同步）。③ 人脸检测 `use_crop` 默认改 false
  （链路一般不需要对齐裁剪图）。④ 删除 YOLO「过滤链：短边/长边最小像素」min_short/min_long
  （功能归属后续讨论）。⑤ 目标裁剪：`scale` 合并进 `out_size`（支持 `×2`/`2x` 倍数写法）。
  ⑥ 输出处理：`smooth_frames`/`min_dwell` 拆为 `window_frames`（window）/`dwell_sec`（dwell），
  装配按 kind 映射；前端新增 `show_if` 联动显隐。⑦ YOLO 类别筛选统一取消、设备全局默认维持现状。
- 影响面：DB（多模块参数列增删，init_db 自动收敛）、配置（各模块表单参数变化）
- 部署注意：重启 web_lab 自动补/删列并重播种；被删参数的历史值丢弃；各链路阶段结构不变。
- 验证：`tests/test_target_crop.py`（37）、`tests/test_annotate.py`（16）、
  `tests/test_cropper.py`（26）、`umvp/pipe/test_composer.py`（58）、
  `tests/test_police_uav_pipeline.py`、`umvp/resources.py`、
  `tests/test_plate_recog_pipeline.py`（28）、`tests/test_face_capture_pipeline.py`（39）全过。

### 2026-09-23(五) · 条件 DSL 抽公共模块 + 目标裁剪支持框筛选
- 变更内容：① 条件 DSL 抽到 `umvp/pipe/conditions.py`（`parse_condition`/`match_obj`），
  供「标注」与「目标裁剪」共用（原标注内联实现迁移）。② `umvp/pipe/target_crop.py` 新增
  `filter` 参数（条件 DSL，显式字段名），与 `classes` 同时生效（AND）；**默认 `filter=""`
  = 全裁，行为与原来的简单过滤一致**。web_lab 注册 target_crop.filter 参数。
- 影响面：结构（新增 conditions.py）、DB（target_crop 参数表新增 filter 列）、配置（目标裁剪可选筛选）
- 部署注意：重启 web_lab 自动补列；既有目标裁剪 preset 的 filter 为空（=全裁，行为不变）。
- 验证：`tests/test_target_crop.py`（33，含条件 DSL）、`tests/test_annotate.py`（16）、
  `umvp/pipe/test_composer.py`（58）全过。

### 2026-09-23(四) · 标注模块加框筛选条件 DSL
- 变更内容：`umvp/pipe/annotate.py` 新增 `filter` 参数（条件 DSL，只画满足条件的框）——
  显式字段名（`cls_id`/`track_id`/`score`，含中文别名），组内逗号 AND、组间分号 OR，
  操作符 `== != > >= < <= in`（`=`≡`==`，`in` 值用 `|`）；`classes` 与 `filter` 同时生效（AND）。
  条件按对象属性取值，不绑定 YOLO，便于后续接入其它来源。web_lab 注册 annotate.filter 参数。
- 影响面：DB（annotate 参数表新增 filter 列）、配置（标注模块行为）
- 部署注意：重启 web_lab 自动补列；既有标注 preset 的 filter 为空（=全画，行为不变）。
- 验证：`tests/test_annotate.py`（16 断言：画框/classes/条件 DSL 各组/别名/非法报错/Pipeline）。

### 2026-09-23(三) · 目标裁剪加输出分辨率参数 + 新增「标注」模块
- 变更内容：① `umvp/pipe/target_crop.py`：新增 `out_size`（""=原图 / "640"=长边像素等比 /
  "640,480"=严格尺寸），在 scale 之后应用；CropResult 增 `scale_x/scale_y`（严格尺寸非等比时
  坐标还原按 X/Y 各自比例）。② 新增 `umvp/pipe/annotate.py`「标注」模块（`annotate_frame`/
  `Annotator`）：在帧副本上画检测框（可带类别标签），供 VLM 看"原图+画框"；`Pipeline` 增
  `annotate` 阶段与 `last_annotated`，web_lab 的 VLM 素材准备优先用标注图。③ 注册模块 annotate
  （前缀 ANNO）+ 参数；PIPE0001 加「标注」阶段（帧管理→YOLO→标注→VLM→告警）。
- 影响面：结构（新增 annotate 模块/文件）、DB（新增 annotate 参数表与 ANNO 前缀、target_crop
  增 out_size 列）、配置（PIPE0001 阶段变化）
- 部署注意：重启 web_lab 自动重建 PIPE0001（按模块复用旧 preset、新增标注阶段）；
  target_crop 参数表自动补 out_size 列。
- 验证：`tests/test_target_crop.py`（28）、`tests/test_annotate.py`（6）、
  `umvp/pipe/test_composer.py`（58）、`tests/test_plate_recog_pipeline.py`（28）全过；
  DB 迁移后 PIPE0001=5 步（含 annotate）。

### 2026-09-23(二) · 链路设计器防重复模块阶段 + 播种按模块复用 preset
- 变更内容：① `web_lab/index.html`：调色板添加已在链路中的模块时不再新增重复阶段（提示
  "每个模块只能有一个阶段"）；保存链路前按模块去重。② `web_lab/db.py` `save_pipeline`：
  校验步骤模块不重复（重复即报 ValueError，防脏数据）。③ `web_lab/server.py` `_seed_chain`：
  结构变化重建时按模块**复用旧 preset**（保留用户参数），补建缺失阶段、丢弃下线阶段——
  同时自愈历史重复步骤（如 PIPE0001 误加的两条 frame_manager）。
- 影响面：DB（save_pipeline 收紧：同链路重复模块被拒）、配置（种子链路重建逻辑）
- 部署注意：重启 web_lab 自动重建 PIPE0002/0003 到新阶段结构并复用其既有 preset；
  历史 PIPE0001 若因重复添加有冗余步骤，重建时按模块去重自愈。
- 验证：重复模块保存被拒（ValueError，未写入）；PIPE0001=4 步（冗余已清）；PIPE0002=5 步、
  PIPE0003=4 步（含 target_crop）；`tests/test_target_crop.py`/`test_cropper.py`/
  `test_composer.py`/`test_police_uav_pipeline.py` 全过。

### 2026-09-23 · 抽出「目标裁剪」模块：裁剪职责独立，YOLO 去 roi，车牌/人脸链重构
- 变更内容：① 新增目标裁剪模块 `umvp/pipe/target_crop.py`（`crop_targets`/`CropResult`/`TargetCrop`）：
  按上游检测框裁子图，输出分辨率可配（scale，默认 1.0=原分辨率）+ margin/min_short/classes；
  `Cropper.crop` 支持任意缩放（含 <1）。② 车牌链 PIPE0002 改为 帧管理→YOLO识车→目标裁剪(裁车)
  →车牌识别（`extract_plates` 改吃 crops、`PlateStage.run(frame,crops,ts)`；不再裁车）；车牌模块
  参数去掉 car_classes/upscale/min_car_short，新增出牌 margin。③ 人脸链 PIPE0003 改为 帧管理→
  YOLO识人→目标裁剪(裁人)→人脸检测（新增 `face_detect.face_pipeline.FaceStage`：检脸+原图出脸）；
  `extract_faces` 模块下线（进 `_PRUNED_MODULES`），`crop_restore.extract_faces` 降级为兼容编排。
  ④ 移除 YOLO 模块 `roi`（crop:cls 裁剪送审）参数与拼接层裁剪逻辑（YOLODetector 只检测）。
  ⑤ 装配器 `_chain_from_spec` 支持 target_crop 阶段；`_seed_chain` 检测到阶段结构变化时自动重建。
- 影响面：结构（新增模块/文件）、DB（新增 target_crop 参数表与 CROP 前缀；下线 extract_faces）、
  配置（链路阶段结构变化，需重建种子链路）
- 部署注意：重启 web_lab 自动重播种——PIPE0002/PIPE0003 阶段结构变化，`_seed_chain` 检测后
  重建（删除旧步骤、覆盖阶段 preset）；extract_faces 参数表/preset 由 init_db 幂等清理。
  若库内 PIPE0002/0003 曾被手工改过，重建会覆盖其阶段 preset。
- 验证：`tests/test_target_crop.py`（24）、`umvp/pipe/test_composer.py`（58）、
  `tests/test_cropper.py`（26）、`tests/test_face_capture_pipeline.py`（39）、
  `tests/test_face_extract_pipeline.py`（28）、`tests/test_plate_recog_pipeline.py`（28）、
  `tests/test_police_uav_pipeline.py`、`umvp/resources.py` 自检 均通过。

### 2026-09-22 · 视频解码适配：web_lab 按需 H.264 转码（HEVC 素材浏览器可播）
- 变更内容：① `web_lab/server.py` 新增 `GET /api/video/prepare?name=`（ffprobe 探测编码：
  H.264/yuv420p + 常见音轨直接回原文件；HEVC 等则后台 ffmpeg 转 H.264/AAC 并回进度）与
  `GET /media/videos/<name>`（转码缓存文件的 Range 服务）。转码缓存 `out/transcoded/
  <stem>.<hash>.mp4`（键=源名+大小+mtime+参数版本，已 gitignore，可重建）。
  ② `test.html` 选视频改走 `prepare` 并轮询转码进度、转完再播；叠加层坐标改按
  `/api/video/info` 的**原始分辨率**换算（预览可能缩到 1920，故不用 `video.videoWidth`）。
  ③ 新增 `tests/test_video_transcode.py`（18 断言：判定/缓存键/探测/端到端转码）。
- 影响面：依赖（新增 ffmpeg/ffprobe 运行时依赖）、入口（新增两个 API）、结构（新增
  `out/transcoded/` 缓存目录）
- 部署注意：目标机需装 **ffmpeg（含 libx264）**；缺失时 HEVC 素材回错误提示（H.264 素材
  不受影响）。`out/transcoded/` 纯缓存，可随时清空重建；源视频变更会自动重转。
- 验证：`conda run -n ai python tests/test_video_transcode.py` → 18/18 通过；服务重启后
  `curl '/api/video/prepare?name=violent1.mp4'` → `ready/transcoded:false`；
  `curl '/api/video/prepare?name=traffic.MP4'` → `transcoding`；小 HEVC 片端到端转码后
  `/media/videos/_e2e_hevc.mp4` Range 返回 206、输出 ffprobe 为 `h264/yuv420p`；
  `node --check`（提取 script）通过。

### 2026-09-22 · 实时播放叠加（SSE 流式运行 + 视频叠加框）+ 修复中文名视频 404
- 变更内容：① 新增 SSE 流式链路运行 `GET /api/test/pipeline_run/stream`（逐分析帧推
  `{n, ts, dets, plates, alarms}`，事件 `meta/frame/done/error`）与停止接口
  `POST /api/test/pipeline_run/stop {run_id}`（`_STREAMS` 注册表 + 停止事件；复用
  `_TEST_LOCK` 非阻塞抢锁，客户端断开即结束；流结束显式 `close_connection`）。
  ② `test.html` 视频测试台新增「同步播放运行」：canvas 叠加层 + EventSource + 播放节流
  （播到最新结果时间即暂停等结果），帧号用真实 fps（原硬编码 25 已改）。
  ③ `_send_file` 支持 HTTP Range（206）——视频可拖进度条。
  ④ **Bug 修复**：`/files/videos/`、`/files/images/` 路径未 URL 解码，中文名视频/图片
  一直 404（浏览器对非 ASCII 名会百分号编码）；现 `unquote` 后中英文名均可访问。
- 影响面：入口（新增两个 API；test.html 新增控件）、结构（新增 `doc/实时播放叠加设计.md`）
- 部署注意：无依赖/模型/DB 变化；重启 web_lab 生效。流式运行不做真实 VLM 调用
  （避免逐帧长阻塞），需真实研判仍用批量运行入口。
- 验证：SSE 冒烟 `meta→frame×N→done`（含检测框/车牌）；停止接口回 `done.stopped=true`；
  Range 中文/ASCII 名均 206、全量 200；批量入口 PIPE0002 零回归；`test.html` JS
  `node --check` 通过、页面 200；composer 59 / cropper 26 / plate 28 / police_uav 逻辑全过。

### 2026-09-22 · 重复实现重构：人脸加载逻辑抽出 providers.py + 裁剪 clamp 复用
- 变更内容：① 新增 `umvp/face_detect/providers.py`（`build_providers` /
  `load_insightface_model` / `DeviceAware`）——把 `FaceDetector` 与 `FaceEmbedder` 中
  逐字重复的 InsightFace 加载逻辑（provider 列表构造、ctx_id 推导、CUDA 初始化失败
  回退 CPU）与 `is_gpu` 属性收敛为单一实现；`face_detector` 用「相对导入 + 脚本目录
  平级导入兜底」兼容原有 `python umvp/face_detect/test_face_detect.py` 运行方式，
  `face_embedder` 复用同一模块。② `web_lab/server.py` 的 VLM 素材裁剪与车牌测试页
  兜底裁剪改用 `pipe.cropper.clamp_bbox`（与通用裁剪模块同一口径）。
- 影响面：结构（新增 `umvp/face_detect/providers.py`）
- 部署注意：无依赖/模型/DB 变化；行为等价（device=auto 无 CUDA 时仍为 CPU×2 回退链）；
  重启 web_lab 生效。
- 验证：`py_compile` 通过；包导入与脚本目录平级导入两种方式冒烟通过、`is_gpu` 继承自
  DeviceAware；`face_embed/test_face_embed.py --self-check` 28/28 命中；
  `test_face_extract_pipeline.py` 28 / `test_face_capture_pipeline.py` 39 /
  `test_plate_recog_pipeline.py` 28 / `test_composer.py` 59 / `test_cropper.py` 26 全过；
  web_lab 重启后 `/api/test/face_detect`、`/api/test/face_embed` 均 200（device=CPU）。

### 2026-09-22 · 代码/文档冗余清理（死代码、过时事实、调研归档、模块索引）
- 变更内容：① **代码清理**——删除零引用死代码：`web_lab/server.py` 的局部
  `from pipe.composer import Detection`（h_pipeline_run 内未用）、`_file_index` 内死函数
  `names()`、被 `_parse_vlm_result` 取代的 `_parse_vlm_action()`；
  `umvp/plate_recog/plate_recognizer.py` 未用的 `import cv2`；
  `tests/test_police_uav_video.py` 未用 `import numpy`、`tests/test_police_uav_pipeline.py`
  未用 `import json` 与 `filter_min_size`；`web_lab/calibrate.py` 两个标定函数未使用的
  `face_models_dir` 形参与 `FACE_MODELS_DIR` 常量（CLI 参数不变）。
  ② **文档清理**——同步过时事实（断言数、保留链路由一改三、存储设计“删参数不删列”已改自动删列、
  `pipelines` 表现役非空、P4 已落地、性能基线对齐 `out/fingerprints.json`、旧链路名、
  模块表头多 preset 归属）；`doc/yolo识别模块.md` 去重（字段表指向模块分册）；
  `doc/输出处理模块设计.md` 瘦身（闸门/DSL 明细指向 `doc/modules/告警策略.md`）；
  两份选型调研归档至 `doc/research/`（加“实现现状以模块分册为准”指针）；
  新增 `doc/modules/INDEX.md` 模块文档入口。
- 影响面：结构（新增 `doc/research/`、`doc/modules/INDEX.md`；调研文档移动路径）、
  入口（`calibrate.py` 内部函数签名，CLI 不变）
- 部署注意：无依赖/模型/DB 变化；重启 web_lab 生效（已重启）。历史记录类文档
  （DEPLOYMENT.md 历史条目、调研原文、skill L4 先例）按约定只加注不删改。
- 验证：`py_compile` 全部改动文件通过；回归 `test_composer.py` 59 / `test_cropper.py` 26 /
  `test_plate_recog_pipeline.py` 28 / `test_police_uav_pipeline.py` / `test_police_uav_video.py`
  全过；`calibrate.py --help` 正常；web_lab 重启后 `/api/schema` 200。

### 2026-09-22 · 启停脚本修复：web_lab 后台服务改用 setsid 独立会话
- 变更内容：`web_lab/restart.sh` 第 3 步由 `nohup ... &` 改为
  `setsid "$PY" server.py ... < /dev/null >> out/web_lab.log 2>&1 &`——后台服务独立会话/
  进程组，不随调用方（终端/CI/工具）退出被连带杀掉；stdin 接 /dev/null，避免继承调用方
  管道导致调用方等待不返回。
- 影响面：入口（restart.sh 启动机制）
- 部署注意：无依赖/参数变化；重启后服务更稳（此前在非交互调用下可能被连带杀掉）。
- 验证：`timeout 60 setsid ./web_lab/restart.sh < /dev/null` 返回 rc=0 且立即返回；
  `GET /api/schema` 200；随后多次独立请求（/api/db/tables、/api/test/alarm、
  /api/test/pipeline_run）均正常，服务在命令结束后持续存活。

### 2026-09-22 · 告警模块通用化为「输出处理」OutputPolicy（七道闸漏斗 + 规则 DSL）
- 变更内容：`pipe/composer.py` 的 `AlarmPolicy` 升级为 `OutputPolicy`（`AlarmPolicy` 保留为
  兼容别名）——统一入口 `Finding`（key/label/conf/features/gran）→ 七道闸（登记/驻留
  min_dwell/词表/置信度 min_conf/规则 rules/滑窗/去重冷却）→ `AlarmEvent`（新增 conf、
  features 字段）。新增规则 DSL（字段 操作符 值；组内逗号 AND、组间分号 OR；自研解析器
  无 eval，非法表达式快速失败）；`key_cooldown` 支持巡检/车牌「同对象只一次(-1，默认)/
  不去重(0)/冷却间隔(>0)」；车牌 `Pipeline._plate_gate` 退役，改走漏斗按车牌号去重
  （AL0002 行为不变）；`on_vlm_result`/`on_dwell` 保留为薄包装。模块展示名改
  「输出处理（告警）」（module_id 仍为 `alarm`，表名/接口不变）。
- 影响面：DB（preset_params_alarm 自动补 4 列：min_conf/min_dwell DOUBLE、rules TEXT、
  key_cooldown DOUBLE）/ 入口（模块测试页展示）
- 部署注意：重启 web_lab 自动补列；旧 preset 缺值走默认（=旧行为），三条链路零参数改动；
  规则表达式操作符两侧需留空格。
- 验证：`umvp/pipe/test_composer.py` 59 断言（38 旧全过 + 21 新：DSL 解析/求值、七闸、
  巡检去重语义）；`tests/test_plate_recog_pipeline.py` 28 断言；`tests/test_police_uav_pipeline.py`
  通过；web_lab 重启后 alarm 参数表出现 4 新列；PIPE0002 真视频（车牌识别.MP4 20–25s）
  端到端告警 1 条，时间线 `触发 inspection（conf 0.83） — 京ACJ0710 绿牌新能源 0.83`。

### 2026-09-22 · 车牌链路补原分辨率提取 + 新增 PIPE0003「人脸截取链路」
- 变更内容：① `extract_plates` 在坐标还原后新增第三步原分辨率提取——`Cropper.extract`
  按还原车牌框从原图直接裁出车牌小图（margin 外扩不缩放），`PlateResult` 新增
  `plate_img`/`plate_crop_bbox`（整帧直读不带）；链路运行器同步落盘车牌小图
  （`plate_NNN_车牌号_ts.jpg`，受 max_saved_images 上限约束），测试页优先展示 plate_img。
  ② 新链路 PIPE0003「人脸截取链路」：帧管理→YOLO识人→原分辨率提取（FM0003→YOLO0003→
  复用 EXTF0001），无嵌入/检索/告警；运行器 `_face_step` 通用化——仅含 extract_faces
  阶段即逐帧截取落盘，含 face_embed 阶段仍走身份去重提取；`_chain_from_spec` 终段校验
  放行 extract_faces；链路缺 yolo 阶段时按 extract_faces 阶段 yolo_* 参数自建识人检测器。
- 影响面：DB（播种新增 FM0003/YOLO0003 两条 preset + PIPE0003 链路及 3 步引用，
  重启自动播种）/ 入口（链路运行器行为）
- 部署注意：无新增依赖；重启 web_lab 自动播种 PIPE0003；不重播种已存在的同名配置。
- 验证：`tests/test_plate_recog_pipeline.py` 28 断言（含 plate_img=原图切片逐像素断言）；
  新增 `tests/test_face_capture_pipeline.py` 39 断言；cropper 26 / composer 38 /
  face_extract 28 全通过；`_chain_from_spec` 三形态装配冒烟通过（PIPE0003 形态 /
  face_embed 旧形态 / 无终段仍拒绝）。

### 2026-09-22 · 通用裁剪模块 cropper.py 独立（人脸/车牌链共用，车牌链不再引入人脸依赖）
- 变更内容：crop_restore.py 中的通用裁剪能力（按框裁图可放大 / 坐标还原 / 关键点还原 /
  原分辨率提取 / IoU 去重）提炼为新文件 `umvp/pipe/cropper.py`（`Cropper`/`CropPatch`/`iou`，
  纯 cv2+numpy 零检测器依赖）；crop_restore.py 只留 `extract_faces` 人脸编排；
  `plate_recog`（extract_plates/PlateStage，改顶层 import pipe.cropper，并删本地重复 `_iou`）
  与 web_lab/server.py 四处改为直接用 `Cropper`；旧名 `CropRestore`/`PersonCrop`/`_iou`
  保留为兼容别名（同源同实现）。
- 影响面：结构（新增 pipe/cropper.py；车牌链不再经 crop_restore 传递引入
  face_detect/onnxruntime）
- 部署注意：无新增第三方依赖；换机 git pull 即可，无需重跑标定/播种/重启脚本之外的操作
  （web_lab 需重启生效）。
- 验证：`tests/test_cropper.py` 26 断言通过（含子进程轻量性检查：import pipe.cropper
  不拉起 face/onnxruntime/ultralytics/hyperlpr3）；`umvp/pipe/test_composer.py` 38、
  `tests/test_face_extract_pipeline.py` 28、`tests/test_plate_recog_pipeline.py` 21
  断言全部通过；server.py 与改动模块 py_compile 通过。

### 2026-09-21 · 测试台修复真实 VLM 端点不可达导致整轮测试“卡死”
- 变更内容：① `_vlm_real` 超时由单一 120s 改为 `timeout=(连接5s, 读取 vlm_timeout 默认20s)`
  ——端点不可达时快速失败；② `h_pipeline_run` 运行前对 VLM 端点做 TCP 预探测
  （`_endpoint_reachable`，2s），不可达则本轮跳过真实调用、按“素材包”记日志并继续；
  ③ 运行中连续失败 ≥3 次熔断（`vlm_unreachable`），后续不再尝试；④ 结果头 VLM 状态
  显示“端点不可达，已跳过”。
- 影响面：入口（测试台行为）
- 部署注意：无新增依赖；重启 web_lab 生效。VLM 端点仍由链路 vlm 阶段参数配置。
- 验证：警用链路（`use_real_vlm=true`、`vlm_feedback` 开）在端点不可达时 8 帧 15s 完成
  （此前每报送最长等 120s 而近乎卡死），日志逐帧显示“素材包 …（端点不可达，跳过真实调用）”。

### 2026-09-21 · 车牌识别视频去重：按 track_id 只识别一次 + 修复 YOLO 跟踪未生效
- 变更内容：① `YOLODetector.infer` 在 track=True 时**显式传 `tracker="bytetrack.yaml"`**
  ——本机 ultralytics 不显式传 tracker 时不分配 `boxes.id`（静默失效），导致跟踪
  track_id 恒为 0（同时影响 dwell / 帧管理-YOLO 的跟踪）；② `PlateStage` 新增
  `track_dedup`（默认 true）/ `track_cooldown`（默认 5s，落库参数），按上游 YOLO 的
  track_id 去重：同一辆车只识别一次，冷却秒数后才允许重试；③ `Pipeline.track_needed()`
  在链路含车牌阶段时让 yolo 自动开启跟踪。
- 影响面：配置 / DB（plate_recog 新增 track_dedup、track_cooldown 两列）
- 部署注意：重启 web_lab 自动补列；无需重播种（旧 preset 缺值走默认 true / 5.0）。
- 验证：车牌识别.MP4 20–23s 识别 19→**1** 次；20–35s（90 分析帧 / 332 车辆框）
  仅 **3** 次（每车一次，5s 冷却刷新）；`tests/test_plate_recog_pipeline.py` 20 断言通过；
  警用链路与装配器回归通过。

### 2026-09-21 · 车牌识别链路重构：模块只做裁车牌+识别，识车回归 yolo 模块
- 变更内容：原 plate_recog 模块内含 YOLO 识车+裁车；改为职责分离——
  `PlateRecognizer` 只做「从车图裁车牌 + 识别」，`extract_plates()` 改为消费**上游 yolo
  模块的车辆框**（不再自跑 YOLO），`PlateStage` 去掉 yolo 参数。链路 PIPE0002 由
  3 阶段改为 **4 阶段**：帧管理→**YOLO识车**→车牌识别→告警（新增 YOLO0002 preset）。
  plate_recog 参数调整：新增 `car_classes`（落库），`yolo_*` 降为 test_only（仅模块
  测试页脚手架用）。
- 影响面：结构 / DB / 配置
- 部署注意：重启 web_lab 自动建列（plate_recog 新增 car_classes、删除 yolo_* 列）
  并播种 4 阶段 PIPE0002。旧 3 阶段 PIPE0002 与 FM0002/PLATE0002/AL0002 已删除后重建。
- 验证：`tests/test_plate_recog_pipeline.py` 19 断言通过（改用 yolo.infer→dets→extract_plates）；
  `/api/pipelines/load PIPE0002` 返回 4 阶段；`pipeline_run` 跑 车牌识别.MP4（20–23s）→
  检出车辆框 46 个、车牌 19 次、告警 5 条（同车牌去重）。

### 2026-09-21 · 新增车牌识别链路 PIPE0002（链路运行器支持 plate_recog 阶段）
- 变更内容：① `pipe/composer.Pipeline` 新增可选 `plate` 阶段（鸭子类型，不引入
  cv2/onnx 依赖）+ `last_plates`，`step()` 每帧调用并把新出现的车牌去重后触发告警；
  ② 新增 `plate_recog/plate_pipeline.py` 的 `PlateStage`（YOLO 识车→裁车放大→识别）；
  ③ `_chain_from_spec` 支持 `plate_recog` 阶段并放宽校验（终段可为 alarm/face_embed/
  plate_recog）；h_chain / h_pipeline_run 增加车牌日志、标注框与计数；
  ④ plate_recog 的 yolo_*/upscale/margin/min_car_short/dedup_iou 由 test_only 提升为
  落库参数（新增 `yolo_max_det` 列）；⑤ 播种第二条链路 PIPE0002（FM0002/PLATE0002/AL0002）。
- 影响面：结构 / DB / 配置
- 部署注意：重启 web_lab 自动建列（plate_recog 新增 yolo_max_det 等列）并播种 PIPE0002；
  无新增依赖（hyperlpr3 已在前一条记录）。
- 验证：`/api/pipelines` 返回 PIPE0001+PIPE0002；load PIPE0002 得 3 阶段；
  `pipeline_run` 跑 车牌识别.MP4（20–24s）→ 识别 沪/粤ACJ0710、京F07277、川G09U262 等，
  告警 8 条（同车牌跨帧去重）；`tests/test_plate_recog_pipeline.py` 19 断言通过。

### 2026-09-21 · 新增车牌识别模块（HyperLPR3）+ 模型自包含
- 变更内容：新增 `umvp/plate_recog/`（`PlateRecognizer`/`PlateResult`/`extract_plates`
  封装 HyperLPR3 检测+四点矫正+CRNN 识别+颜色分类；`extract_plates` 复用 CropRestore
  做 YOLO 识车→裁车放大→坐标还原；`fetch_models.py` 拉取模型）。web_lab 注册
  `plate_recog` 模块（PARAMS + `h_plate_recog` + MODULES，组「车牌识别链路」）；
  db.py `_ID_PREFIXES` 登记 `PLATE`，播种 `PLATE0001`。requirements.txt 加
  `hyperlpr3==0.1.3`；.gitignore 加 `plate_models/`；新增测试与 `doc/modules/车牌识别.md`。
- 影响面：依赖 / 模型 / 结构 / DB / 入口
- 部署注意：`pip install -r requirements.txt`（新增 hyperlpr3）；新克隆/换机需跑
  `conda run -n ai python umvp/plate_recog/fetch_models.py` 拉模型到
  `plate_models/hyperlpr3/20230229/onnx/`（约 12MB，官方源 `hyperlpr.tunm.top`）。
  模型在项目内即可离线：模块加载前自动同步到 `~/.hyperlpr3/<版本>/` 以跳过
  hyperlpr3 import 时的联网下载。重启 web_lab 后新模块出现在侧栏「车牌识别链路」，
  参数表 `preset_params_plate_recog` 自动建（仅 `detect_level` 落库）。
- 验证：`tests/test_plate_recog_pipeline.py` 16 断言通过（整帧识别 沪/粤ACJ0710 绿牌，
  YOLO 识车→裁车放大→坐标还原框与原图一致）；`/api/test/plate_recog` 的 direct 与 crop
  两种模式均返回车牌；移走 `~/.hyperlpr3` 后仅靠项目模型仍离线识别成功（无下载）；
  项目模型与默认目录均缺失时给出 `fetch_models.py` 提示并报错。

### 2026-09-20 · 测试台新增视频信息预估 + 落盘上限 200 张
- 变更内容：① 新增 GET `/api/video/info?name=` 接口（cv2 读分辨率/帧率/总帧数/时长）；
  test.html 选视频/换链路/改时间段/限帧后，信息条实时显示分辨率与"按链路帧管理
  frame_skip 预计分析 N 帧"（frame_skip 取所选链路 frame_manager 阶段参数，前端缓存）。
  ② h_pipeline_run 落盘上限：单次测试最多保存 200 张图（原图/标注图/人脸图共用
  计数，参数 max_saved_images 可覆盖）；超限跳过写盘、时间线记一条提示、结果头
  汇总"落盘 N 张（超上限跳过 M 张）"。
- 影响面：入口（新增 API 路由）/ 配置（落盘行为变化）
- 部署注意：无新增依赖；重启 web_lab 生效。
- 验证：/api/video/info?name=traffic.MP4 返回 3840×2160@29.97fps/1832 帧/61.1s；
  max_saved_images=3 跑 450 帧链路 → 目录恰 3 张图、汇总"落盘 3 张（超上限跳过
  27 张）"、时间线含上限提示；test.html JS 语法通过。

### 2026-09-20 · 测试参数与模块参数分离：不落库、不进链路设计器
- 变更内容：① server.py PARAMS 为各模块测试参数标记 `"test_only": true`（frame_manager
  的 mode/scenario/fps/n_frames/ts_step、yolo 的 image/ts、vlm 的 image(新增，修真实
  调用无法选图)/ref/gran/track_key/bbox/det_count、alarm 的 timeline、face_detect 的
  image、face_embed 的 image/det_thresh/det_size、face_store 的 action/register_dir/
  image/name、extract_faces 的 image）；② configure_modules 注册 schema 时过滤
  test_only → 这些参数不再有库列；③ db.py 新增 `_prune_stale_columns`：init_db 时
  幂等 DROP 模块参数表中已不在注册 schema 的列（列随 schema 双向收敛，替代原
  "删参数不删列"约定）；④ index.html 模块测试页分区展示（"测试参数（仅本页运行用，
  不存库、不进链路）"分隔块），链路设计器阶段表单过滤 test_only。
- 影响面：DB（模块参数表删列）/ 结构
- 部署注意：换机拉代码重启 web_lab 即自动清理废弃列（幂等）；预设中测试参数不再
  持久化，载入 preset 后测试区回默认值（预期行为）。
- 验证：重启后 SHOW COLUMNS——frame_manager/yolo/vlm/face_store 等表的测试列已删，
  presets 8 行与 PIPE0001 引用不变；保存 preset 带 image/ts 额外键→落库自动剔除
  （load 返回键集仅模块参数）；/api/schema 带 test_only 标记；traffic.MP4 90 帧链路
  端到端无回归；index.html JS 语法检查通过。

### 2026-09-20 · 清理运行器空参数表（chain/face_monitor/frame_yolo 不再落库）
- 变更内容：① server.py 启动注册参数 schema 时按 `hidden` 过滤——只注册 8 个产品模块，
  web 隐藏运行器（chain/face_monitor/frame_yolo，另有无参数的 pipeline_run）是运行入口
  而非产品模块，不再建参数表、参数 preset 不落库（保存报"schema 未注册"）；
  ② db.py `_PRUNED_MODULES` 追加这三个模块，init_db 幂等 DROP 其空表与残留行；
  `_ID_PREFIXES` 移除对应的 FMON/FYLO/CHN。库内 3 张 0 行空表
  （preset_params_chain / preset_params_face_monitor / preset_params_frame_yolo）已删除，
  参数表收敛为 8 张（每产品模块一张、各 1 行）。
- 影响面：DB / 结构
- 部署注意：换机拉代码重启 web_lab 即自动清理（幂等）；运行器模块本身保留
  （/api/schema 仍 12 个模块，test 页运行不受影响）。
- 验证：重启后 SHOW TABLES = 3 主表 + 8 参数表，presets 8 行 / pipelines 1 行不变；
  /api/schema 12 模块正常；/api/db/tables 参数表页签只剩 8 个产品模块；
  对 chain 保存 preset 返回明确报错。

### 2026-09-20 · 工作区整理简化：DB 业务 ID（YOLO0001 式）+ 唯一保留链路 + 播种收敛
- 变更内容：① `web_lab/db.py` schema 变更——presets.id / pipelines.id 从自增整数改为
  "前缀+4 位序号"业务编号（FM/YOLO/VLM/AL/FDET/FEMB/FSTO/EXTF/FMON/FYLO/CHN + PIPE，
  前缀表见 `_ID_PREFIXES`），新建自动按前缀递增；pipeline_steps.preset_id 同步改
  VARCHAR(24)；`init_db()` 检测到旧整数 schema 自动 DROP 全部配置表重建（播种数据可再生）。
  ② 播种收敛到 server.py `_seed_pipeline()` 一处：唯一保留「警用无人机链路」PIPE0001
  （步骤 FM0001→YOLO0001→VLM0001→AL0001，实调参数 imgsz=1280/conf=0.25/use_real_vlm=1）+
  人脸四模块默认 preset（FDET0001/FEMB0001/FSTO0001/EXTF0001）；原 server 双链路种子、
  tests/police_uav_config.py、tests/face_dedup_config.py 移除；删除重复实现
  tests/test_video_face_dedup.py。③ 清理死代码：/db_admin.html 路由、_db_monitor/_db_query
  （另一套连不上库的连接参数）。④ 文档：doc/modules/ 每模块一篇（8 模块+链路，含参数/
  输入/输出字段），删《模块与参数手册.md》，重写《数据库结构.md》。
- 影响面：DB / 入口（删除 2 个测试配置脚本与 2 个死路由）/ 结构（doc/modules/）
- 部署注意：换机拉代码后**重启 web_lab 即自动迁移**（旧库 DROP 重建 + 重新播种，无需
  手工操作；确认 MySQL 3307 在运行）；test.html 链路下拉里 pipeline_id 已是 PIPE0001。
- 验证：重启后 SHOW TABLES + presets 恰 8 行（FM0001/YOLO0001/VLM0001/AL0001/
  FDET0001/FEMB0001/FSTO0001/EXTF0001）、pipelines 恰 1 行 PIPE0001（4 步）；API 实测
  presets save（自动 YOLO0002）/load/delete、定向更新（YOLO0001 参数覆盖 id 不变）、
  pipelines save/load/delete 全通过；traffic.MP4 150 帧端到端跑通（开 vlm_feedback：
  真实 VLM 研判返回"交警类"并成警）；test_composer.py 27 断言 + test_police_uav_pipeline.py
  全过。

### 2026-08-26 · 警用无人机链路 YOLO 参数调整 + 清理空白 frame_manager 孤儿
- 变更内容：① 链路 10（警用无人机）yolo 步骤引用的 preset 85 参数落库更新为
  imgsz=1280、conf=0.25（iou=0.6、roi=false、class_limits="0:2,300;2:1,300" 同步补齐），
  用预定定向更新（preset_id=85，id 稳定，链路引用不漂移）；② 清理空白孤儿 preset 99
  （frame_manager「警用无人机链路 · 帧管理 FrameManager」，参数全空 0/14，未被任何链路
  引用——前次保存 bug 残留）。
- 影响面：DB（preset 85 参数更新、preset 99 删除）、配置（链路 10 yolo 阈值生效）
- 部署注意：web_lab 已重启；链路 10 引用 84/85/86/87，yolo=85（imgsz 1280 / conf 0.25）；
  其余未引用有值 preset 为模块库默认/历史，保留不动。
- 验证：`/api/presets/load {id:85}` imgsz=1280、conf=0.25；`/api/pipelines/load {id:10}`
  步骤 84/85/86/87；全表扫描未引用且全空条 = 无。

### 2026-08-26 · 清理空白配置 + 链路设计器"载入"下拉按模块过滤
- 变更内容：① `web_lab/index.html` `renderStagePresetSel(q, sel, mid)` 加 `mid` 参数，
  阶段级"载入"下拉只列与当前阶段同模块的 preset（此前列出全部模块，误选会填错参数）；
  ② 数据清理：删除 4 条空白无意义 preset——100(yolo 0/16)、101(vlm 0/12)、
  102(alarm 0/8)（前次 bug 生成的未填参孤儿）、76(compose 空 schema 0/0)；
  把链路 10（警用无人机）步骤从 bug 产物 99/100/101/102 重整回完整的 84/85/86/87。
- 影响面：DB（删 4 条空白 preset；链路 10 步骤引用复位）、配置（设计器载入下拉
  只显示本模块配置，避免错选）
- 部署注意：web_lab 已重启生效；未删除"有值但未引用"的（88-91 (配置扩展)、99 等）
  与「*·默认」模块种子——非空白、且部分被模块库"载入默认"使用，保留以免丢配置；
  preset 85 名字仍为"小模型识别 YOLODetector"（模块栏未填名时的默认），功能正常。
- 验证：`node --check` 通过；`/api/pipelines/load {id:10}` 步骤引用 84/85/86/87、
  yolo imgsz=1280（保留你调过的值）已复位；presets 无 id 100/101/102/76。

### 2026-08-26 · 修改链路/模块参数"同步落库"：预设定向更新，不再产生冗余
- 变更内容：修复"链路设计器/模块栏改参数后未落库"且重复生成 preset 的问题。
  根因：`savePipelinePreset` 每次保存都按「链路名·模块名」新建 preset，与链路
  pipeline_steps 实际引用的既有 preset（如 `警用无人机_YOLO识别`=85）名字对不上，
  导致两套 id、引用漂移、旧 preset 变孤儿，改 A 库边引用 B。方案：改参数=更新引用的
  那条 preset（id 稳定），绝不为既有阶段新建。
  - `web_lab/db.py`：`save_preset` 新增可选 `preset_id`——给定则按 id 定向更新
    （校验存在与 kind，UPDATE 该行 + 参数表 upsert，id 不变）；未给则沿用同名覆盖/新建。
  - `web_lab/server.py`：`_preset_save` 读取并透传 `preset_id`。
  - `web_lab/index.html`：设计器 `STAGES` 每阶段带 `presetId`（`addStage`/`renderStages`/
    `loadPipelinePreset` 传递，阶段级"载入"绑定选中 preset id）；`savePipelinePreset`
    有 `presetId` 则定向更新、无则新建并记 id；模块栏 `loadModulePreset`/`saveModulePreset`
    增加 `modulePresetId` 跟踪（载入后改参保存即更新该 preset），切换模块时重置。
- 影响面：DB（preset 更新语义不变更 schema；链路 10 的 yolo 步骤已从缺参孤儿 preset
  100 重整为引用完整 preset 85，参数齐全）、配置（改参数落库到引用那条，链路引用稳定）
- 部署注意：web_lab 已重启；既有冗余孤儿 preset 未删（可后续清理），但不再新增；
  模块栏/设计器载入某 preset 后改参保存，即更新该 preset，不再出现"改了不生效"。
- 验证：`py_compile` + `node --check` 通过；`preset_id=85` 定向更新 imgsz 960→id 不变、
  落库 960→reset 640；yolo preset 总数 6 不变（不新建）；链路 10 载出 yolo 步骤
  引用 preset 85、imgsz=640 参数完整（此前为 import 100 全 None）。

### 2026-08-26 · 修复 web 端编辑模块时偶发"数据库不可用"（DB 连接的线程安全）
- 变更内容：`web_lab/db.py` 修复多线程共享单条 pymysql 连接导致的并发错乱
  （根因：ThreadingHTTPServer 每请求开新线程，前端 loadSchema 一次并发 3 个 presets
  请求打在全局单例 `_conn` 上，连接非线程安全 → `Packet sequence number wrong` /
  `read of closed file` → 接口 ok:false → 前端显示"数据库不可用"）。新增全局
  `_LOCK = threading.RLock()`；`_cursor()` 改为 `@contextlib.contextmanager` 上下文
  管理器（with 存续期间持锁独占连接，退出释放，现有调用点零改动）；`save_preset`/
  `save_pipeline` 两个直连 conn.cursor() 的函数事务体包进 `with _LOCK:`。
- 影响面：DB（访问层串行化，不含锁外长事务；DB 操作均为短查询/短事务，串行开销可忽略）
- 部署注意：web_lab 已重启生效；不影响表结构与连接参数，无迁移。
- 验证：并发回归——修复前 12 并发 3 失败（Packet sequence wrong），修复后 20 并发
  0 失败；`/api/pipelines` 读正常（ok:true，2 条）；`py_compile` 通过。

### 2026-08-26 · 测试台通用化：支持人脸链（无 alarm）+ 结果区 JSON 分行与动态过滤
- 变更内容：`web_lab/server.py` ① `_chain_from_spec` 放宽必需阶段——alarm 不再强制
  （含 face_embed 的人脸链无告警阶段，占位 inspection 策略），frame_manager 仍必需；
  ② `h_pipeline_run` 新增人脸链分支：含 face_embed 阶段时惰性加载 FaceDetector/
  FaceEmbedder/CropRestore，按帧管理节奏走 `extract_faces`（YOLO 识人→SCRFD 检脸→
  坐标还原）→ ArcFace 嵌入 → 与已收集身份余弦去重（阈值取 face_embed 阶段 thresh，
  默认 0.55）→ 新身份记 kind=face 事件（data=FaceResult.to_dict+best_sim）+ 原分辨率
  人脸落盘/出图（≤36 张）；重复记 kind=info。③ timeline 事件全部支持 data 字段
  （submit/vlm/alarm 亦带结构化数据）；④ 字符串参数清洗提升为模块级 `_str_param`。
  `web_lab/test.html` 结果区通用化：日志条目附带 data 时以分行 JSON（pre.log-json）
  展示；过滤器动态生成——按本次 timeline 出现的 kind 决定显示"仅 VLM 报送/仅告警/
  仅新人脸"选项（人脸链不显示告警项，告警链不显示人脸项）。
- 影响面：配置（人脸链可直接在测试台运行；face_detect/face_embed 阶段参数
  det_thresh/det_size/upscale/margin/min_person_short/dedup_iou/thresh 参与运行）
- 部署注意：web_lab 已重启；人脸去重阈值默认 0.55，同人跨帧相似度偏低时可调大
  face_embed 阶段 thresh（阈值高=更倾向判为新身份）。
- 验证：链路 11（视频人脸去重提取）face1.MP4 全视频：178 分析帧、新身份 31、
  重复丢弃 170、31 张人脸图（best_sim 边缘样本可见去重工作正常）；链路 10
  （警用无人机）traffic.MP4 前 120 帧：submit 2 / vlm 2 / alarm 1（描述完整），
  两链路事件分布与过滤器预期一致。

### 2026-08-25 · 链路测试接入真实 VLM 调用（use_real_vlm）+ 解析兼容 alert_type
- 变更内容：`web_lab/server.py` 的 `h_pipeline_run` 支持 VLM 真实研判——从链路 vlm 阶段
  参数读 `use_real_vlm/vlm_endpoint/vlm_model/vlm_key`（vlm 模块 schema 原有字段，此前
  仅模块单测用），开启后每条报送经 `VLMAnalyzer.prepare` + 新增 `_vlm_material()`
  （crop:* 按该类首目标 bbox 外扩裁剪 / full:* 整帧）送真实端点，action 喂
  `on_vlm_result` 判告警；关闭则维持模拟回填（vlm_action 字符串）。配套：
  `_parse_vlm_action` 兼容 `result/alert_type/action` 三字段（警用链路 prompt 返回
  alert_type，原只认 result 必然解析失败）；`_vlm_real` 对 DB 还原的 None 参数防護；
  VLM 调用失败记日志继续不中断测试。DB 侧：链路 10（警用无人机）vlm preset 86
  已更新 `use_real_vlm=true`，并清理该链路重复的 4 个 stages（84-91 → 84-87）。
- 影响面：配置（链路 vlm 阶段 use_real_vlm=true 时测试真实调用端点，网络慢时测试
  时长随报送数增长）、DB（preset 86 覆盖更新、链路 10 steps 重建）
- 部署注意：web_lab 已重启生效；真实调用走 `http://117.42.21.253:8000/v1/chat/completions`
  （vlm 阶段参数可覆盖），无 key；测试时长 = 报送数 × 单次 VLM 耗时，调大 frame_skip/
  check_interval 可减少报送。
- 验证：`py_compile` 过；图片模式冒烟（链路 10 + 人遮挡1.png）：检出 14、报送 2 条
  （cls0×10 / cls2×4）、真实 VLM 返回"无异常"×2、正常场景不告警（行为正确）；
  preset 86 use_real_vlm=true 读回确认；链路 10 steps=84-87 无重复。

### 2026-08-25 · 修复链路测试三连错（stages 解析 / module_id / device "None"）+ 测试台去重复抽帧
- 变更内容：`web_lab/server.py` 修复链路测试（`h_pipeline_run` → `_chain_from_spec`）三处错误：
  ① spec 解析——原从 `pipeline_data.get("spec")` 读取，但 `db.load_pipeline` 返回
  `{id, name, streams, budget, meta, stages}`，改为直接按返回值构建 spec；
  ② stages 字段名——`s.get("module")` 改为 `s.get("module_id")`（db 返回结构）；
  ③ device "None" 报错（根因）——DB 参数表空列经 `_restore_params` 还原为 Python
  `None`，`str(None).strip()` 得字符串 `"None"` 传给 torch 报
  `device string: None`；`_chain_from_spec` 新增 `_s()` 清洗（None/空串 → 默认值），
  覆盖 device/sampling/kind/prompt/model_path/class_limits；`umvp/pipe/composer.py`
  的 `YOLODetector.infer` 同时显式传 device（已配置时）。
  另：测试台 `frame_skip` 与链路内部 FrameManager 双重抽帧，删除测试台侧参数
  （`web_lab/test.html` 输入框与 JS 读取、server 端 h_pipeline_run 读取），
  取图节奏统一由链路设计器帧管理模块控制。
- 影响面：配置（链路测试参数去 frame_skip；DB 空列参数回落默认值不再变成 "None"）
- 部署注意：web_lab 已重启生效；GPU 用户在链路设计器 YOLO 模块 device 填 `0`。
- 验证：`conda run -n ai python /tmp/opencode/test_chain_device.py`（None 参数组装：
  device=None/sampling=analysis/vlm.prompt=""，全断言过）；
  `umvp/pipe/test_composer.py` 27/27；`POST /api/test/pipeline_run {pipeline_id:10}`
  走到"请指定测试视频"（组装成功）；test.html 无 frame_skip 残留。

### 2026-08-19 · OpenClaude 工具优先级调整：启用 GLM MCP 搜索工具，旧工具降级需确认
- 变更内容：`/home/carloga0/.openclaude/settings.json` 权限配置调整——清空 `deny` 列表，将 `WebSearch`、`WebFetch`、`mcp__fetch__fetch` 移至 `ask` 列表（用户确认时才允许），新增 4 个 GLM MCP 服务器（zai-mcp-server/web-search-prime/web-reader/zread）无限制运行。
- 影响面：配置（OpenClaude 全局工具优先级，影响 AI 助手网络搜索能力）
- 部署注意：换机移植时若复用 OpenClaude 配置，需同步 settings.json；GLM MCP 工具提供更精准技术搜索，旧工具降级为备选。
- 验证：OpenClaude MCP 状态显示 6 服务器健康；GLM 搜索测试返回精确技术文档，旧工具触发确认提示。

### 2026-08-18 · 抽帧调度并入帧管理：删除 frame_scheduler 模块，帧决策统一走 FrameManager
- 变更内容：`umvp/face_scan/frame_scheduler.py`（FrameScheduler 抽帧调度）删除，其能力
  并入 `umvp/pipe/composer.py` 的 FrameManager（sampling=analysis 取帧 + queue/adaptive
  两种背压：decide / budget / note_inference / reset / current_skip / relaxed /
  relax_events 均在 FrameManager 上，`face_scan/face_monitor.py` 改为引用 FrameManager）。
  `web_lab` 侧移除 frame_scheduler 模块注册（PARAMS/MODULES/种子），
  `umvp/resources.py` MODULE_TYPE 移除其条目；MySQL 侧 `init_db()` 新增
  `_prune_removed_modules` 幂等清理：先删引用其 preset 的链路步骤（fk_step_preset
  为 RESTRICT），再删 presets 行（参数行级联），最后 DROP `preset_params_frame_scheduler`。
- 影响面：DB（presets 中 frame_scheduler 的 `default1` 行与参数表在重启 init_db 时
  幂等清理）、入口（frame_scheduler 从 web_lab 模块列表移除，只余 12 个模块）、
  结构（face_scan 不再含 frame_scheduler.py，FaceMonitor 引用 FrameManager）、
  配置（frame_manager 参数表新增 sampling/mode/scenario/queue_threshold/
  backpressure_multiplier/max_frame_skip/adaptive_relax_ratio/adaptive_recover_ratio/fps
  等列，由 `_ensure_module_table` 幂等 ALTER 补齐）
- 部署注意：重启 web_lab 触发 `init_db()` 自动清理旧表与旧行，无需手工 SQL；
  已拼链路 spec 中 stages 的 frame_manager 参数名不变（含背压字段），不兼容处仅
  frame_scheduler 模块本身的下线
- 验证：`conda run -n ai python umvp/pipe/test_composer.py`（27 断言）、
  `conda run -n ai python umvp/face_scan/test_face_scan.py`、SQL 查
  `preset_params_frame_scheduler` 不存在且 presets 无 frame_scheduler 行、
  frame_manager 参数表新列齐全（见 tests/test_chain_runner.py 21 断言）

### 2026-08-17 · 删除 3 条四模式链路种子，链路统一走拼接；修复 h_chain 视频驱动双重采样
- 变更内容：`web_lab/server.py` 的 3 条四模式链路种子（small_only 小模型驻留告警 /
  small_yolo_vlm 小模型+大模型研判 / large_only 纯大模型研判）及其模板
  （_SEED_CHAIN_* / _chain_pipeline_spec 等）全部删除，`_SEED_PIPES` 仅剩两条旧链路
  （视频人脸检索 / 帧管理-YOLO识别）；`_chain_from_spec` 装配器、`_ALGO_EXPECT` 校验表、
  `PARAMS["chain"]`、h_chain 视频驱动保留（后续替代链路仍走 runner=chain 拼接）。
  同时修复 h_chain 视频驱动双重采样 bug：帧决策先行 `wants_frame` 门控后调
  `pipe.step()` 会再采样一次（wall_clock/frame_count 等有状态采样第二次返回 False
  导致 large_only 只分析 1 帧/报送 0 条），改为 `pipe.step(..., force=True)` 跳过
  step 内重复采样。`tests/test_chain_runner.py` 相应改写为自拼 spec（仅校验断言 +
  驱动断言，21 断言，不再依赖已删除的种子）。
- 影响面：DB（presets 删除 3 条 kind=pipeline 链路 + 10 条「*·*」module 行，参数表
  由外键级联清理）、配置（模式概念改为统一链路拼接重建）、入口（无新增/删除接口）
- 部署注意：重启 web_lab 不再播种这 3 条链路；旧库中残留行需手工删除（本次执行时
  已删，现 presets 仅剩 default1 + 两条旧链路及其 module 行，无孤儿参数行）；替代
  链路后续用 .claude/skills/pipeline-assembly 拼接重建（runner=chain + stages，
  run_params 用 video/max_frames/vlm_feedback/vlm_action）。
- 验证：`conda run -n ai python tests/test_chain_runner.py`（21 断言通过）；
  `curl /api/presets?kind=pipeline` 仅返回两条旧链路；SQL 查 presets 无 3 条模式链路、
  孤儿参数行 0；h_chain 视频驱动实测 wall_clock 采样报送 1 条（修复前为 0）、
  frame_count 按间隔报送、small_only 驻留告警触发 dwell:0。

### 2026-08-14 · 已完成链路新增四模式运行器（runner=chain）
- 变更内容：`web_lab/server.py` 新增 `PARAMS["chain"]`（feed=script/real 驱动、
  VLM 回填 action）与隐藏模块 `chain`（按 spec.stages 组装四模块实例的通用运行器
  `_chain_from_spec` + `h_chain`，装配前用 `_ALGO_EXPECT` 校验 stages 结构与
  algo_mode 一致）；播种 3 条四模式链路：small_only 驻留告警 / small_yolo_vlm
  小模型+大模型研判（方案A：忽略 ROI，small_crop/small_full 并为一条）/
  large_only 纯大模型研判，spec 显式存 `algo_mode`（compose() 分派键）作顶层键。
  `web_lab/index.html` 运行已完成链路时把完整 spec（含 stages）并入 params 传给
  `/api/test/chain`。新增 `tests/test_chain_runner.py`（51 断言纯逻辑验证）。
- 影响面：入口（新运行器 /api/test/chain）、DB（presets 新增 3 条 kind=pipeline
  链路种子，仅迁移增量）、配置（algo_mode=small_yolo_vlm 新模式，compose() 不识别、
  仅 chain 运行器用）
- 部署注意：重启 web_lab 后种子入库（presets 按 (kind,name) 幂等，已存在的
  small_only/large_only 名称不会覆盖已有同名自定义链路）。compose() 仍只认
  small_only/small_crop/small_full/large_only 四种模式，small_yolo_vlm 不能用于
  compose 链路。
- 验证：`conda run -n ai python tests/test_chain_runner.py`（51 断言通过）；
  `conda run -n ai python umvp/pipe/test_composer.py`（27 断言通过）；seed 装配
  脚本（5 条种子名不重复，3 条 chain 种子全部过 `_chain_from_spec`）。

### 2026-08-12 · 新增已完成链路「帧管理-YOLO识别」+ 链路表单按模块分组折叠
- 变更内容：`web_lab/server.py` 新增 `PARAMS["frame_yolo"]`（参数按「运行控制 / 帧管理
  FrameManager / YOLO 识别 YOLODetector」三组分组，`group` 字段驱动前端折叠）、
  `h_frame_yolo` 处理器（按取图节奏抽帧 → YOLO 检测 + 可选 BOTSORT 跟踪 → 过滤统计，
  输出逐帧时间线、track 汇总与 4 张标注图）与隐藏模块 `frame_yolo`（不占侧栏/模块库）；
  播种新增第二个 pipeline「帧管理-YOLO识别」（runner=frame_yolo，stages 引用
  frame_manager/yolo 两模块，run_params 同 frame_yolo 参数）；pipeline spec 新增
  `run_module` 字段，旧「视频人脸检索」无该字段时前端回退 runner。`web_lab/index.html`
  已完成链路运行表单改为按参数 `group` 分组渲染为可折叠 `<details>`（`renderGroupedForm`），
  运行表单 schema 由 `spec.run_module` 指定。
- 影响面：DB、入口、配置
- 部署注意：重启 server 后首次启动自动播种（幂等，同名校验已存在则跳过，尊重已有编辑）；
  「帧管理-YOLO识别」表单 schema 走隐藏模块 frame_yolo，不占用设计器模块库位置；
  参数分组与折叠纯前端渲染，无需额外部署。
- 验证：`/api/presets?kind=pipeline` 返回两条链路（帧管理-YOLO识别 runner=frame_yolo、
  视频人脸检索 runner=face_monitor）；`/` 页面含 renderGroupedForm（grep 命中 2 处）；
  HTTP 冒烟实测 `POST /api/test/frame_yolo`（横幅1.MP4，读 30 帧/分析 10 帧，
  检出 80 个=平均 8.0/帧，唯一 track 8 条全部稳定跨≥2 帧，avg 652ms/帧，
  逐帧时间线 10 行 + track 汇总 8 行 + 4 张标注图，ok=true，7.3s）。

### 2026-08-11 · web_lab 结构重构：FaceMonitor 转为已完成链路「视频人脸检索」
- 变更内容：`face_monitor` 模块入口改为 `hidden: True`、分组归入「已完成链路」并改名
  「视频人脸检索 FaceMonitor」，不再占用侧栏/设计器模块库位置；顶部 tab 导航删除，链路设计器
  改为侧栏独立槽位，原位置新增「已完成链路」分组（搜索框 + 列表，可选中并运行）。
  新增幂等播种 `_seed_completed_pipeline()`：首次启动时向 `presets` 写入 kind=pipeline 的
  「视频人脸检索」（`params.runner='face_monitor'`，stages 引用 frame_scheduler/face_detect/
  face_embed/face_store 四模块，run_params 含 video/db_path/device/frame_skip 等 8 项），
  并写入 4 条「视频人脸检索·*」模块命名配置；DB 禁用（`--no-db`）时跳过不影响启动。
- 影响面：DB、入口、配置
- 部署注意：重启 server 后首次启动自动播种（幂等，同名校验已存在则跳过，尊重已有编辑）；
  底库/视频依赖同 face_monitor 原配置（`out/face_db.npz`、`tests/data/vid/`）；
  「已完成链路」运行表单走 `/api/test/face_monitor`，参数与运行逻辑不变。
- 验证：JS 括号平衡 + `node --check` 语法 OK；seed 幂等实测（连续播种 pipeline 1 条 /
  module 5 条不变，`_DB_ENABLED=False` 无异常）；`/api/presets?kind=pipeline` 含
  「视频人脸检索」且 runner=face_monitor；`/api/presets?kind=module` 含 4 条
  「视频人脸检索·*」；冒烟实测 `/api/test/face_monitor`（face1.MP4，读 60 帧/分析 10 帧，
  5.58s ok）；侧栏/模块库均不含 face_monitor；空/不存在视频返回优雅错误不崩溃。

### 2026-08-10 · 配置参数列式化存储：每模块一张参数表 + /api/db/tables 数据浏览
- 变更内容：`kind='module'` 的命名配置不再整份 JSON 塞进 `presets.params_json`，改为
  每模块一张参数表 `preset_params_<module_id>`（参数按类型展开为列，int→INT / float→DOUBLE /
  bool→TINYINT(1) / select→VARCHAR(64) / file→VARCHAR(255) / str→VARCHAR(255) / text→TEXT，
  主键 preset_id 外键级联删）；`presets` 降级为注册表，module 行 `params_json` 迁移后置空，
  pipeline 类型仍存 JSON 不受影响。`web_lab/db.py` 新增 `configure_modules()`（启动时注册各模块
  参数 schema）、参数表自动建表/补列（`SHOW COLUMNS` + `ALTER TABLE ADD COLUMN`，删参数不删列）、
  旧 JSON 数据幂等迁移、save/load/list 拆列读写与类型强转；`web_lab/server.py` 新增
  `GET /api/db/tables` 只读快照接口（presets / 各模块参数表 / pipelines / steps，DB 不可用时
  返回 `{ok:false}` 容错）；`web_lab/index.html` 新增"数据浏览"tab 渲染各表。设计文档
  `doc/配置参数列式化存储设计.md` 新建，`doc/数据库结构.md`、`web_lab/设计说明.md` 同步。
- 影响面：DB、入口
- 部署注意：服务器启动 `init_db()` 幂等建参数表/补列/迁移旧数据，无需手工 DDL；模块参数
  清单（server.py PARAMS）加参数自动补列、删参数不删列；module 行 params_json 已置空（仅
  pipeline 行有值）；`/api/db/tables` 只读，增删改仍走 /api/presets* 接口。
- 验证：DB 层断言回归 36/36 ALL PASS（configure_modules 幂等、参数按类型往返一致、同名覆盖
  id 不变、未注册模块容错、删除级联、旧数据迁移等）；curl 实测 `/api/db/tables` 结构正确；
  HTTP save/load/delete 全流程类型往返一致（conf 0.42 float、imgsz 640 int、roi True bool）；
  旧配置 default1 迁移实测（params_json 置空、6 个参数落入 preset_params_frame_manager）；
  测试数据已清理（presets 1 行、pipelines 0 行、pipeline_steps 0 行），AUTO_INCREMENT 实测
  presets=34 / pipelines=10 / pipeline_steps=18。

### 2026-08-10 · 链路拼装存储：新增 pipelines/pipeline_steps 两表与 /api/pipelines* 接口
- 变更内容：链路拼装信息由"整份快照"改为"步骤引用模块 preset id + position 顺序"存储：
  `web_lab/db.py` 新增 `pipelines`（name/streams/budget_json/meta_json）与 `pipeline_steps`
  （pipeline_id/position/preset_id，外键级联与 RESTRICT）两表，新增
  `list/save/load/delete_pipeline` 访问层函数，`delete_preset` 增加被引用拒绝（报错含链路名）；
  `web_lab/server.py` 新增 `/api/pipelines` GET 与 `/api/pipelines/save|load|delete` POST 四接口
  （沿用 DB 不可用时返回 `{ok:false}` 的容错模式）。设计文档 `doc/链路拼装存储设计.md` 新建，
  `doc/数据库结构.md` 同步新增两表结构。
- 影响面：DB、入口
- 部署注意：服务器启动 `init_db()` 幂等自动补建新表，无需手工 DDL；删除被链路引用的模块
  preset 会被拒绝，需先删除/改引用的链路；`pipeline_steps` 依赖 InnoDB 外键（当前引擎即满足）。
- 验证：DB 层断言回归 9/9 ALL PASS（init_db 幂等、save/load 全字段一致、同名覆盖 id 不变、
  引用拒绝含链路名、删链路后级联放行、空/不存在 steps 拒绝、列表倒序、清理干净）；
  `SHOW CREATE TABLE` 实测两表结构同步进 `doc/数据库结构.md`；curl 实测四接口
  （list/save/load/delete 正常，错误 id 与非法 steps 返回明确报错）；测试数据已清理
  （presets 1 行、pipelines 0 行、pipeline_steps 0 行）。

### 2026-08-10 · 新增 web_lab 一键停止/重启脚本与端口操作手册
- 变更内容：新增 `web_lab/stop.sh`（按端口监听 PID 定位，找不到按进程名兜底，停止 web_lab
  server；默认连自含 MySQL 一起停，`--keep-mysql` 保留）与 `web_lab/restart.sh`（一键重启：
  停旧 web 进程→确保自含 MySQL 运行→后台拉起 server，日志写 `out/web_lab.log`，支持
  `--port/--host/--no-db` 透传与 `--stop-mysql` 连库一起重启）；新增操作文档
  `doc/端口与启停手册.md`；`.gitignore` 增加 `out/web_lab.log`。`web_lab/run.sh` 行为不变
  （前台启动）。
- 影响面：入口
- 部署注意：换机后脚本自动定位解释器（`conda run -n ai which python`，可用 `PY=` 覆盖）；
  脚本只按项目 `.mysql/mysql.sock` 特征识别自含 MySQL，不会误停系统 MySQL；
  MySQL 停止优先 `mysqladmin shutdown`，无权限时自动回退 `kill -TERM`（干净落盘）。
- 验证：`bash -n web_lab/stop.sh web_lab/restart.sh` 通过；实机跑通五条路径——① `./web_lab/restart.sh`
  冷启动：MySQL 拉起 + server 后台启动，curl 200，8001/3307 均监听；② `./web_lab/stop.sh` 全停：
  两端口释放；③ `./web_lab/stop.sh --keep-mysql`：web 停、3307 保留；④ `./web_lab/restart.sh
  --stop-mysql`：连库重启成功；⑤ `--port 8080` 换端口启动，且不带 `--port` 的 `stop.sh`
  按进程名定位并停掉 8080 实例。

### 2026-08-10 · 本机远端切 SSH 并首次推送同步
- 变更内容：本机（carloga0）远端地址由 HTTPS `https://github.com/carlogao7-pixel/umvp.git`
  切换为 SSH `git@github.com:carlogao7-pixel/umvp.git`；生成 ed25519 SSH key
  （`~/.ssh/id_ed25519`，GitHub 登记名 `carloga0-wsl2`，Authentication Key）并认证成功；
  首次 `git push -u origin main` 同步两个本地提交（d67019e 适配本机路径 +
  4b9eb6d 路径可移植化），`main` 与 `origin/main` 对齐。CLAUDE.md 新增
  「git 协作（多机同步）」小节（提交习惯 / 换机先 pull / 路径可移植说明）。
- 影响面：配置
- 部署注意：换机移植后需重新生成/登记 SSH key；HTTPS 克隆也可用但每次 push 要认证。
- 验证：`ssh -T git@github.com` 输出 `Hi carlogao7-pixel! You've successfully authenticated`；
  `git status -sb` 显示 `main...origin/main` 无落后超前；工作树干净。

### 2026-08-10 · 解释器与路径可移植化（跨机器同步铺垫）
- 变更内容：文档与脚本中的机器绝对路径统一改为便携写法——解释器一律写
  `conda run -n ai python`（CLAUDE.md / README.md / INSTALL.md / doc/需求文档-开发版.md /
  .claude/agents/pipeline-assembler.md / .claude/skills/face-extract-pipeline/SKILL.md /
  web_lab/calibrate.py 用法）；`web_lab/run.sh` 的 PY 解析改为
  `conda run -n ai which python` 自动定位（失败回退 `command -v python`，仍可用
  `PY=...` 环境变量显式覆盖）；`tests/data/yolo_standalone_output.json` 的 model_path
  改为相对路径 `models/yolov8n.pt`。
- 影响面：配置 / 入口
- 部署注意：换机移植无需再改解释器路径；目标机需有 conda 环境 `ai`
  （与 INSTALL.md 场景 B 一致）。
- 验证：`conda run -n ai python umvp/pipe/test_composer.py` 27/27 断言通过；
  `conda run -n ai python umvp/face_embed/test_face_embed.py --register-dir
  umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz` 21/21 自检命中；
  `bash -n web_lab/run.sh` 语法通过；`web_lab/calibrate.py` py_compile 通过；
  `conda run -n ai which python` 正常解析出解释器路径。

### 2026-08-10 · 本机（carloga0）接入 GitHub 仓库并适配环境路径
- 变更内容：本机 `/home/carloga0/projects/umvp` 接入远端仓库 origin=`https://github.com/carlogao7-pixel/umvp.git`
  （分支 main，首次提交 46 个代码/文档/测试文件，含 `.gitignore`，远端记录见下条）；文档与脚本中
  的解释器/项目根路径由 carl0 机器路径（`/home/carl0/miniconda/envs/ai/bin/python`、
  `/home/carl0/project/umvp`）改回本机 carloga0 路径（`/home/carloga0/miniconda3/envs/ai/bin/python`、
  `/home/carloga0/projects/umvp`）。远端历史条目保留原样不改写。
- 影响面：配置 / 入口
- 部署注意：换机移植后若 conda 路径不同，以实际解释器为准。
- 验证：`git status` 干净；`test_composer.py` 27/27 断言通过。

### 2026-08-09 · git 版本管理引入：首次提交 c742a3f
- 变更内容：`git init`（分支 main），新增 `.gitignore`（忽略 models/*.pt、face_models/、
  tests/data/vid/、.mysql/、out/face_db.*、out/video_db.*、tests/data/out/），首次提交 46 个
  代码/文档/测试文件（7.6M）；README「无 git 关联」说明更新为 git 管理表述。本地身份
  carl0 用户 carlogao7-pixel <carlogao7@gmail.com>，SSH key（ed25519）已认证 GitHub。
- 影响面：结构 / 入口
- 部署注意：克隆后无模型权重与测试视频，需按 INSTALL.md 从上游/zip 重建；`out/fingerprints.json`
  随库保留（标定基线），face_db/video_db 需重跑生成。
- 验证：`git status` 干净；`ssh -T git@github.com` 认证成功。
- 变更内容：ai 环境补装 `insightface==1.0.1`（原环境缺失，人脸链路必需）；全量文档
  （CLAUDE.md / README.md / INSTALL.md / doc/需求文档-开发版.md / .claude/agents/pipeline-assembler.md /
  .claude/skills/face-extract-pipeline/SKILL.md）中的解释器与项目根路径由
  `/home/carloga0/...` 统一修正为 `/home/carl0/miniconda/envs/ai/bin/python` 与
  `/home/carl0/project/umvp`（旧路径在本机不存在）。
- 影响面：依赖 / 配置
- 部署注意：换机移植后若 conda 路径不同，以实际解释器为准；运行命令一律使用
  `/home/carl0/miniconda/envs/ai/bin/python`。
- 验证：`test_composer.py` 27/27 断言通过；`test_face_embed.py --self-check` 28/28 命中；
  `umvp/resources.py` 自检通过。
