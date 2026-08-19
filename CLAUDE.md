# UMVP 项目指令（CLAUDE.md）

UMVP（统一模块化视觉分析管道）独立部署包。代码 + 权重自包含，本目录即项目根。
所有 python 命令用 conda 环境 `ai` 的解释器（唯一带 cv2 / ultralytics / insightface / onnxruntime），
统一写 `conda run -n ai python ...`（或 `conda activate ai` 后用 `python`），不写机器绝对路径。

## 项目结构

```
umvp/pipe/         四模块 + compose() 装配器（FrameManager / YOLODetector / VLMAnalyzer / AlarmPolicy）
                   + crop_restore.py（CropRestore 原分辨率人脸提取链路：YOLO 识人→裁人→SCRFD 检脸→坐标还原→原图出图）
umvp/face_detect/  SCRFD 人脸检测 + 5点对齐（FaceDetector，含 test_imgs/）
umvp/face_embed/   ArcFace 512d 嵌入（FaceEmbedder）+ 向量底库（FaceStore，numpy 线性检索）
umvp/face_scan/    人脸管道编排（FaceMonitor，复用 pipe.composer.FrameManager 取帧+背压）
umvp/resources.py  P1 资源估算（ResourceEstimator：静态公式 + 实测指纹两层，链路聚合/多路外推/预算反推）
models/            YOLO 选择池（v8n/s、v9t、v10s、v11n/s/m，COCO person=0/car=2；
                    放 .pt 进目录即出现在 web_lab 模型选择中）
face_models/       buffalo_l/ 已解压（det_10g.onnx + w600k_r50.onnx 等，人脸模型默认加载路径）
web_lab/           可视化测试台（server.py 表单/测试/presets/资源估算；db.py MySQL 配置暂存；
                    calibrate.py 实测标定→out/fingerprints.json；run.sh/stop.sh/restart.sh 启停/重启
                    脚本，操作手册见 doc/端口与启停手册.md；设计说明.md 通俗文档）
.claude/skills/    pipeline-assembly 链路拼接知识库（SKILL.md 技能入口 + INDEX.md 分层索引 +
                   knowledge/ 四层：L1 共享陈述 / L2 总装专属 / L3 组装工专属 / L4 共享经验，
                   含链路决策表【待定稿】、壳代码骨架、常见坑、face-extract 先例等）
doc/               需求文档（业务版/开发版）、模块与参数手册（13 模块权威清单）、端口与启停手册、
                   数据库结构、存储设计、链路组装Agent设计、yolo识别模块专题
tests/             三模式实机测试 + YOLO 参数对比 + 人脸提取链路 + 视频/截图数据
```

## 关键约定

- **人脸模型路径**：默认项目内 `face_models/buffalo_l/`（代码按 `__file__` 相对定位），
  不依赖 `~/.insightface`；`FaceDetector(model_root=...)` / `FaceEmbedder(model_root=...)` 可覆盖。
- **四种模式收敛为配置**：small_only / small_crop / small_full / large_only 差异全在
  compose spec，不写分支代码。报送粒度=按识别类型（class_limits），dwell 按 track 记驻留。
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

$PY umvp/pipe/test_composer.py            # 纯逻辑 27 断言（不加载模型，秒级）
$PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz   # 人脸注册+自检 7/7
$PY tests/test_yolo_params.py             # YOLO 参数对比（真实推理，yolov8n.pt，约 1 分钟）
$PY tests/test_3modes.py                  # 三模式实机（需 VLM 端点可达，约 3.5 分钟）
$PY umvp/resources.py                     # 资源估算自检（不联网；读 out/fingerprints.json 若存在）
$PY web_lab/calibrate.py                  # 实测标定（遍历 models/*.pt 生成 out/fingerprints.json，换机必跑）
$PY tests/test_face_extract_pipeline.py   # 人脸提取链路 27 断言（YOLO 识人→裁人→SCRFD 检脸→坐标还原）
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
