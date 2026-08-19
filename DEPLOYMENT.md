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
