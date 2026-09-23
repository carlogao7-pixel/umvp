# UMVP 项目指令（CLAUDE.md）

UMVP（统一模块化视觉分析管道）独立部署包。代码 + 权重自包含，本目录即项目根。
所有 python 命令用 conda 环境 `ai` 的解释器（唯一带 cv2 / ultralytics / insightface / onnxruntime），
统一写 `conda run -n ai python ...`（或 `conda activate ai` 后用 `python`），不写机器绝对路径。

## 项目结构

```
umvp/pipe/         模块 + Pipeline 装配器（FrameManager / YOLODetector / VLMAnalyzer /
                   OutputPolicy 输出处理（原告警策略：Finding 统一进，驻留/词表/置信/规则DSL/
                     滑窗/去重冷却七道闸出；AlarmPolicy 为兼容别名））
                   + cropper.py（通用裁剪工具 Cropper：按框裁图可缩放/坐标还原/原分辨率提取，
                     纯 cv2+numpy 零检测器依赖；旧名 CropRestore/PersonCrop 为兼容别名）
                   + target_crop.py（目标裁剪模块 TargetCrop：按上游检测框裁子图，
                     输出分辨率可配（默认原分辨率）；车牌/人脸/VLM 链共用）
                   + annotate.py（标注模块 Annotator：在帧上画检测框，供 VLM 看"原图+画框"）
                   + crop_restore.py（extract_faces 兼容编排，已拆为「目标裁剪+人脸检测」两模块）
umvp/face_detect/  SCRFD 人脸检测 + 5点对齐（FaceDetector，含 test_imgs/）
                   + face_pipeline.py（FaceStage 人脸检测阶段：在目标子图检脸 + 原图出原分辨率人脸）
                   + providers.py（公共加载：provider 选择/CUDA 回退/is_gpu，与 face_embed 共用）
umvp/face_embed/   ArcFace 512d 嵌入（FaceEmbedder）+ 向量底库（FaceStore，numpy 线性检索）
umvp/face_scan/    人脸管道编排（FaceMonitor，复用 pipe.composer.FrameManager 取帧+背压）
umvp/plate_recog/  车牌识别（PlateRecognizer：HyperLPR3 从车图裁车牌→矫正→CRNN 识别→颜色分类；
                   extract_plates 消费上游「目标裁剪」产出的车辆子图→识别→坐标还原→原图出车牌小图，
                   模块不含车辆检测、不裁车；fetch_models.py 拉取模型到 plate_models/）
umvp/resources.py  P1 资源估算（ResourceEstimator：静态公式 + 实测指纹两层，链路聚合/多路外推/预算反推）
models/            YOLO 选择池（v8n/s、v9t、v10s、v11n/s/m，COCO person=0/car=2；
                     放 .pt 进目录即出现在 web_lab 模型选择中）
face_models/       buffalo_l/ 已解压（det_10g.onnx + w600k_r50.onnx 等，人脸模型默认加载路径）
plate_models/      hyperlpr3/20230229/onnx/ 车牌模型（检测 320/640 + 识别 + 颜色分类；
                     umvp/plate_recog/fetch_models.py 拉取；不入库，离线时自动同步到 ~/.hyperlpr3）
web_lab/           可视化测试台（server.py 表单/测试/presets/资源估算/链路运行；db.py MySQL 配置暂存；
                   calibrate.py 实测标定→out/fingerprints.json；run.sh/stop.sh/restart.sh 启停/重启
                   脚本，操作手册见 doc/端口与启停手册.md；test.html 链路测试台（批量运行 +
                   SSE 流式「同步播放运行」：视频播放时 canvas 实时叠加检测/车牌框；
                   非浏览器友好编码（如 HEVC）按需 ffmpeg 转 H.264 缓存到 out/transcoded/））
.claude/skills/    pipeline-assembly 链路拼接知识库（SKILL.md 技能入口 + INDEX.md 分层索引 +
                    knowledge/ 四层：L1 共享陈述 / L2 总装专属 / L3 组装工专属 / L4 共享经验，
                    含链路决策表【待定稿】、壳代码骨架、常见坑、face-extract 先例等）
doc/               README.md 文档总览（业务架构/技术结构/文件索引，先读它）＋需求文档（业务版/开发版）、
                   端口与启停手册、数据库结构、存储设计、链路组装Agent设计、输出处理模块设计、
                   实时播放叠加设计；doc/research/ 选型调研（历史）
doc/modules/       模块参数文档（每模块一篇：参数/输入/输出字段；入口 INDEX.md；
                   9 个模块 + 通用裁剪 + 三条保留链路）
tests/             警用链路测试 + YOLO 参数对比 + 人脸提取链路 + 车牌识别链路 + 视频/截图数据
```

## 关键约定

- **人脸模型路径**：默认项目内 `face_models/buffalo_l/`（代码按 `__file__` 相对定位），
  不依赖 `~/.insightface`；`FaceDetector(model_root=...)` / `FaceEmbedder(model_root=...)` 可覆盖。
- **车牌模型路径**：默认项目内 `plate_models/hyperlpr3/`（代码按 `__file__` 相对定位），
  不依赖 `~/.hyperlpr3`；`PlateRecognizer(model_root=...)` 可覆盖。hyperlpr3 在 import 时
  会检查 `~/.hyperlpr3/<版本>/` 缺失则联网下载，本模块加载前自动从项目模型同步过去，
  故项目模型在即可完全离线。新克隆用 `umvp/plate_recog/fetch_models.py` 重建。
- **保留链路**：三条——「警用无人机链路」（PIPE0001，帧管理→YOLO→**标注**→VLM研判→告警）、
  「车牌识别链路」（PIPE0002，帧管理→YOLO识车→**目标裁剪(裁车)**→车牌识别→告警）与
  「人脸截取链路」（PIPE0003，帧管理→YOLO识人→**目标裁剪(裁人)**→人脸检测(检脸+原图出脸)；
  无嵌入/检索/告警，截取落盘即目的；运行器同段代码兼容含 face_embed 的身份去重形态）。
  播种收敛在 server.py `_seed_pipeline()`（幂等，同名跳过，经 `_seed_chain()` 逐条播种；
  已存在链路阶段结构变化时自动重建）；各阶段引用自己的模块 preset。
- **识别对象有效性过滤归 YOLO**：置信度/尺寸过滤（`filter_conf`/`min_short`/`min_long`）在
  `YOLODetector.infer()` **输出处**生效——不达标目标视为"未识别"，对**所有**下游
  （VLM 报送 / 目标裁剪·车牌·人脸传递 / 标注）统一生效，且先于 `class_limits` 按类计数。
- **裁剪职责边界**：YOLO 只检测、不裁剪、不做机械缩放；「目标裁剪」按上游检测框裁子图
  （输出分辨率可配，默认原分辨率），只做机械裁剪与缩放、不做有效性过滤；人脸/车牌的
  二次裁剪（检脸裁脸 / 检牌裁牌）留在各自识别模块内
  （人脸检测 `face_detect.face_pipeline.FaceStage`、车牌识别 `plate_recog.extract_plates`）。
- **数据库业务 ID**：presets/pipelines 主键为"前缀+4 位序号"（YOLO0001/FM0001/PIPE0001…），
  新建按前缀自动递增（前缀表见 db.py `_ID_PREFIXES`）；旧整数 schema 由 init_db 检测后
  自动 DROP 重建并重新播种（换机拉代码重启即迁移）。
- **测试参数不落库**：模块测试页专用参数（测试图片/演示回填/事件脚本/模拟参数）在
  server.py PARAMS 标 `"test_only": true`——不注册 DB schema、不落库、不进链路设计器
  （新增此类参数务必打标；机制见 doc/数据库结构.md §3.5）。模块参数表列随 schema
  双向收敛（加参数补列、删参数/转 test_only 自动删列）。
- **测试台落盘上限**：链路运行落盘图片默认上限 200 张（`max_saved_images` 可覆盖），
  超限跳过写盘并在时间线/结果头提示；选视频后 test.html 显示分辨率与预计分析帧数
  （`GET /api/video/info`，按链路 frame_manager.frame_skip 折算）。
- **测试台处理日志通用化**：链路运行日志按「阶段」逐模块展示各自本帧处理内容（不绑定具体
  链路类型，也不要求该模块向下传递）——后端 timeline 每条带 `stage`/`stage_label`（短标签，
  阶段顺序取自链路 stages）、`to`/`to_label`（下一级）、`frame`（帧序数），前端按出现的
  阶段动态生成过滤项；帧管理放行帧 / yolo 检出与报送或车辆框 / VLM 研判或素材包 / 车牌
  识别结果 / 告警判定（无告警也记一条）逐条展示，每条带帧序数。
- **自定义链路**：支持用户通过链路设计器自定义组合模块，不预设分析模式。报送粒度=按识别类型（class_limits），dwell 按 track 记驻留。
- **资源估算（P1）**：`ResourceEstimator` 静态公式估算优先被实测指纹覆盖（`out/fingerprints.json`，
  由 `web_lab/calibrate.py` 生成）；链路按"每帧平均成本"聚合（VLM 按 check_interval 节流折算、
  人脸嵌入只在命中后折算），显存按常驻+激活拆分，多路外推 + 预算反推建议 frame_skip。
  机器变化（换卡/换机）只需重跑 calibrate.py，代码零改动。
- **skill**：`.claude/skills/pipeline-assembly/` 是链路组成环节的单一知识库（user-invocable，
  用 `/pipeline-assembly` 或 Skill 工具调用）——总装 / 组装工两个 agent 的提示词只留基础内容，
  全部所需知识分层存放并由 `INDEX.md` 索引（L1 共享陈述 / L2 总装专属 / L3 组装工专属 /
  L4 共享经验；face-extract 先例已并入 L4）。
- **web_lab presets**：命名配置暂存 MySQL（`web_lab/db.py`，惰性连接，`--no-db` 可禁用；
  连接参数 MYSQL_* 环境变量覆盖，缺失不影响启动）。
- **协作约定**：文档先行（先设计文档后实现）、验证闭环（断言式测试，不接受"看起来对"）；
  用户向文档用通俗语言、不贴代码；技术设计文档可保留代码引用。
- **部署台账**：影响部署的改动（依赖/模型/环境变量/DB/标定/入口/结构/配置，判断清单见
  `DEPLOYMENT.md`）在验证后必须向 `DEPLOYMENT.md`「记录」顶部追加一条，格式见该文件；
  纯代码逻辑/文档改动且不影响运行配置的不记。拿不准算不算时，记一条无害。

## git 协作（多机同步）

- **仓库**：GitHub `carlogao7-pixel/umvp`，分支 `main`。本机远端为 SSH 地址
  `git@github.com:carlogao7-pixel/umvp.git`（2026-08-10 从 HTTPS 切换），SSH key
  （ed25519，GitHub 上名 `carloga0-wsl2`）已认证，push 无需再输密码。
- **提交习惯**：小步提交——一个逻辑改动一个 commit（中文信息，说清"为什么"）；
  按需推送——一个收尾或换机前 `git push` 一次即可，不必改一次推一次。
  零碎改动可在本地攒几个 commit 后一次推送。
- **多机同步**：本仓库在 carl0 与 carloga0 两台机器间共用。换机开发前先
  `git pull` 拉最新，避免两边基于不同状态改产生冲突；改完 `git push`。
- **路径已可移植**：文档/脚本统一 `conda run -n ai python` + 相对路径，两机
  无需再手工改路径。`DEPLOYMENT.md` 历史条目里的旧绝对路径是故意保留的历史记录，
  按"只追加、不修改历史条目"规则不可动。

## 常用验证命令（cd 到项目根执行）

```bash
PY="conda run -n ai python"   # 便携：ai conda 环境解释器（或先 conda activate ai 后用 python）

$PY umvp/pipe/test_composer.py            # 纯逻辑 60 断言（不加载模型，秒级）
$PY tests/test_cropper.py                 # 通用裁剪模块 26 断言（纯逻辑+轻量性检查，秒级）
$PY tests/test_police_uav_pipeline.py     # 警用链路（警用无人机）逻辑断言（秒级）
$PY tests/test_police_uav_video.py        # 警用链路端到端（真实 YOLO + mock VLM，落 tests/data/out/police_uav）
$PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz   # 人脸注册+自检 7/7
$PY tests/test_yolo_params.py             # YOLO 参数对比（真实推理，yolov8n.pt，约 1 分钟）
$PY umvp/resources.py                     # 资源估算自检（不联网；读 out/fingerprints.json 若存在）
$PY web_lab/calibrate.py                  # 实测标定（遍历 models/*.pt 生成 out/fingerprints.json，换机必跑）
$PY tests/test_target_crop.py             # 目标裁剪模块 35 断言（按坐标裁图/分辨率(长边/严格/倍数)/外扩/过滤(含条件DSL)/坐标还原）
$PY tests/test_annotate.py                # 标注模块 16 断言（画框/classes/条件 DSL/Pipeline 集成）
$PY tests/test_face_extract_pipeline.py   # 人脸提取链路 28 断言（YOLO 识人→裁人→SCRFD 检脸→坐标还原）
$PY tests/test_face_capture_pipeline.py   # 人脸截取链路（PIPE0003 形态）39 断言（帧管理门控+逐帧截取）
$PY tests/test_plate_recog_pipeline.py    # 车牌识别链路 28 断言（整帧识别 + 目标裁剪裁车→识别→还原→原分辨率提取）
$PY umvp/plate_recog/fetch_models.py      # 拉取 HyperLPR3 车牌模型到 plate_models/（换机/新克隆必跑一次）
```

## 注意事项

- onnxruntime 会打印大量 `VerifyOutputSizes`/`Expected shape` 形状警告（SCRFD 动态 batch
  正常现象）和 `Duplicate provider` 警告，均不影响结果，验证时可用 `grep -v` 过滤。
- 性能基线（CPU，稳态单帧，权威值见 `out/fingerprints.json`，本机 2026-08-06 实测）：
  YOLO @640 各档 40–280ms（yolov8n 39ms / yolo11n 48ms / yolov9t 56ms / yolov10s 87ms /
  yolov8s 93ms / yolo11s 100ms / yolo11m 276ms；首帧含模型加载额外 1–2s）、
  SCRFD 640 全图 ~100ms、ArcFace 单张 ~37ms、千级底库检索 <1ms。换机后以
  `web_lab/calibrate.py` 重新标定为准。
- 完整安装/移植步骤见 INSTALL.md（场景 A 本机继承 ai 环境 / 场景 B 新机器从零装）。
- 联网检索/抓取优先用 GLM 系 MCP 工具（web-search-prime / web-reader / zread 免确认直用；
  备用 fetch/search 与内置 webfetch/websearch 降级为需确认）——本机配置在全局
  `~/.config/opencode/opencode.jsonc` 与 `~/.config/opencode/AGENTS.md`，换机参考
  `~/openclaude-mcp-export/` 导出包（含 key，禁入 git）。
