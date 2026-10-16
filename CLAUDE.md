# UMVP 项目指令（CLAUDE.md · 共享上下文）

> **本文件是共享上下文**：`orchestrator` / `pipe-chain` / `web` / `agent-builder` / `qa`
> 五个开发代理在会话启动时**自动载入本文件全文**。因此这里只放**所有板块都需要、
> 且不常变**的跨切面内容——项目定位、板块划分与对接楔子、设计契约、协作规约、文档导航。
> **板块专属的技术细节、参数、数据、验证数字**放各自 `.opencode/agent/<name>.md`
> （个人上下文）；**具体实现以 `doc/` 为准**（先读 `doc/README.md`）。

UMVP（统一模块化视觉分析管道）独立部署包，代码 + 权重自包含，本目录即项目根。
所有 python 命令用 conda 环境 `ai`（唯一带 cv2 / ultralytics / insightface / onnxruntime），
统一写 `conda run -n ai python ...`（或 `conda activate ai` 后用 `python`），不写机器绝对路径。

## 一、板块划分与对接（全体需知）

项目开发由 5 个 opencode 开发代理分担板块，定义在 `.opencode/agent/`（本机 opencode 用，
与本项目运行的「挂载 agent」无关）。**板块间只通过下列楔子对接；改自己板块内容不越界。**

### `pipe-chain` —— 模块与链路编排
- **负责**：`umvp/pipe/` 各分析模块 + 三条保留链路的编排 + 链路 `stages spec`。
- **技术栈**：Python + OpenCV / numpy + ultralytics(YOLO) + onnxruntime；VLM 走 HTTP。
- **对接楔子**：模块参数经 `server.py` PARAMS / preset 暴露给 web；链路以 `stages spec` 描述；
  模块间通用语 = **dict 记录（Record）+ 事件（events）**（见 §三）。
- **文档**：`doc/modules/`（每模块一篇=module_id，入口 `INDEX.md`，按「JSON 主干×旁路」
  数据流角色分组）；链路专篇 `doc/chains/`（三条保留链路）；链路壳标准
  `.claude/agents/assembler/01-shell-standard.md`（唯一权威）。

### `web` —— 测试台与资源估算
- **负责**：`web_lab/`（server / db / test.html / 启停脚本 / 标定 / spec 校验）+ `umvp/resources.py`。
- **技术栈**：Python 标准库 HTTP 服务 + 原生 JS（test.html）+ MySQL（`db.py`，惰性连接）+ ffmpeg。
- **对接楔子**：消费 pipe 的模块与链路 spec（注册模块测试页/设计器阶段、`_seed_pipeline` 播种）；
  对浏览器暴露 REST + SSE（视频信息、同步播放叠加等）；资源估算读 `out/fingerprints.json`。
- **文档**：`doc/guides/runbook.md`、`doc/guides/database_schema.md`、
  `doc/design/live_overlay.md`、`doc/design/storage_design.md`。

### `agent-builder` —— 运行时挂载 agent 与提示词
- **负责**：`web_lab/agent*.py`（运行时工具 + function-calling 循环）+ `.claude/agents/`
  两角色提示词与组装工技术文档。
- **技术栈**：Python + function-calling LLM（经 `agent_llm.py`）。
- **对接楔子**：`read_doc` 只读项目内 Markdown（`doc/` 与 `.claude/agents/`）；两角色提示词
  **自包含**；`web_lab/` 内**仅** `agent*.py` / `dump_prompts.py` / `agent_run.py` 归本板块。
- **文档**：`.claude/agents/assembler/README.md`（索引与维护规则）。

### `qa` —— 测试
- **负责**：`tests/` 各链路测试 + 各模块自检的运行与回归 + `tests/data/` 素材。
- **技术栈**：Python 断言式测试（纯逻辑秒级 / 真实推理较慢）。
- **对接楔子**：发现被测模块问题向对应板块代理反馈（附复现命令）；不直接改被测模块。
- **文档**：模块行为以 `doc/modules/` 分册为准。

> **上下文分层**：本文件 = 所有代理自动载入的**共享上下文**；各代理
> `.opencode/agent/<name>.md` 的**正文** = 仅该代理载入的**个人上下文**
> （opencode 无"按代理加载的 instructions 文件"，个人上下文只能写在 agent 正文里）。
> 新建会话载入共享上下文 + `orchestrator` 个人上下文；切换代理即换成该代理个人上下文。

## 二、项目结构（高层；细节见 `doc/README.md`）

```
umvp/pipe/        分析模块 + Pipeline 装配器（FrameManager / YOLODetector / VLMAnalyzer /
                  DetFilter / SeqFilter / TargetCrop / Annotator + cropper / conditions / records）
umvp/face_detect/ SCRFD 人脸检测 + 5 点对齐（FaceDetector / FaceStage）
umvp/face_embed/  ArcFace 512d 嵌入 + numpy 向量底库
umvp/face_scan/   人脸管道编排（复用 pipe.FrameManager 取帧+背压）
umvp/plate_recog/ 车牌识别（HyperLPR3；extract_plates 消费目标裁剪产出）
umvp/resources.py P1 资源估算（静态公式 + 实测指纹两层）
web_lab/          可视化测试台（server / db / test.html / 启停脚本 / 标定 / spec 校验）
.claude/agents/   挂载 agent 两角色提示词 + assembler/ 组装工技术文档
doc/              README.md 文档总览（先读）＋ requirements/ guides/ design/ agent/ research/
doc/modules/      模块分册（每篇=module_id，入口 INDEX.md，按数据流角色分组）
doc/chains/       保留链路专篇（pipe0001_police_uav / pipe0002_plate / pipe0003_face_capture）
models/ face_models/ plate_models/   YOLO 选择池 / 人脸模型 / 车牌模型（自包含，不入库）
tests/            链路测试 + YOLO 参数对比 + 测试素材
```

## 三、设计契约（跨板块，必须共同遵守）

- **保留链路**：三条——
  **PIPE0001 警用无人机**（帧管理→YOLO→检测筛选→标注→VLM研判→时序判定事件尾）、
  **PIPE0002 车牌**（帧管理→YOLO识车→检测筛选→目标裁剪(裁车)→车牌识别→时序判定事件尾）、
  **PIPE0003 人脸截取**（帧管理→YOLO识人→检测筛选→目标裁剪(裁人)→人脸检测；截取落盘即目的）。
  链路**输出 = 事件尾产出的记录事件**（原「输出处理 OutputPolicy」已弃用删除）。
  播种幂等收敛在 `web_lab/server.py _seed_pipeline()`；各阶段引用自己的模块 preset。
- **职责边界（改动勿越界）**：YOLO 只**纯推理**；识别有效性筛选的**唯一**来源是链路上
  独立的 `det_filter` 阶段；节流/报送编排在 Pipeline；裁剪只做机械裁剪缩放，二次裁剪
  （检脸 / 检牌）留在各自识别模块内。
- **通用记录与事件（模块对接语）**：模块间以 dict 记录传递（图像 / 向量走旁路字段）；
  记录可携带 `events` 列表（一条记录可多事件）；链路尾以"末段记录的事件汇总"为通用输出，
  不绑定具体模块。老模块对象在边界经适配器统一。API/字段细节见 `umvp/pipe/records.py`
  与 `doc/modules/`。
- **模型 / 权重自包含**：人脸、车牌模型默认项目内路径（代码按 `__file__` 相对定位），
  不依赖 `~/.insightface` / `~/.hyperlpr3`；均可 `model_root=` 覆盖。
- **不预设分析模式**：支持用户经链路设计器自由组合模块（自定义链路）。

## 四、协作规约（多代理）

- **收件箱**：任何代理完成改动 / 发现新坑 / 产生需归档内容 → **先追加**到
  `.opencode/memory/inbox.md`（只追加，格式见文件头）；**不直接改** `CLAUDE.md` / `DEPLOYMENT.md`。
- **工单（任务流）**：`.opencode/memory/tickets.md` ＋ 工具 `.opencode/tool/ticket.py`
  （`new/set/list/show/check`，零依赖，改完必跑 `check`）。用户需求经 orchestrator 审核澄清后
  开**主单**（`WO-xxxx`）再派发；各代理开工前建**子单**（`WO-xxxx.n`，发起人=自己，可
  `--to` 跨板块指派并在回报中 @对方）；状态流转 **进行中 → 待测试（qa 验证）→ 已完成**，
  需求受阻/变更用 **已取消**；主单在全部子单关闭前保持进行中。字段与细则见工单文件头。
- **筛选同步 `/curate`**：用户执行时，主代理翻 inbox「待同步」条目，按需提炼进本文件 /
  `DEPLOYMENT.md`；**涉及 `doc/` 项目文档或新建文档时，必须先列出建议征求用户同意再写入**。
  已处理条目移入「已同步」（注明去向）。
- **代理互调**：会话内用 `@<agent>` 派遣其他板块代理；被派发代理的返回即同步回当前会话。
- **模型配置**：各代理 `model` 写在对应 `.opencode/agent/<name>.md` frontmatter
  （省略则沿用默认主模型）；改完 agent/config 需**重启 opencode** 生效。
- **部署台账**：影响部署的改动（依赖 / 模型 / 环境变量 / DB / 标定 / 入口 / 结构 / 配置）
  验证后向 `DEPLOYMENT.md`「记录」顶部追加一条（格式与判断清单见该文件）。
- **git（多机同步）**：仓库 GitHub `carlogao7-pixel/umvp`，分支 `main`，远端 SSH 已认证。
  小步提交（一逻辑改动一 commit，中文说清"为什么"），按需 push；换机开发前先 `git pull`。
  路径已可移植（统一 `conda run -n ai python` + 相对路径）。
- **协作约定**：文档先行（先设计文档后实现）、验证闭环（断言式测试，不接受"看起来对"）；
  用户向文档用通俗语言、不贴代码，技术设计文档可保留代码引用。

## 五、文档导航

- `doc/README.md` —— **文档总览**（业务架构 / 技术结构 / 目录分类与索引），先读它再决定看哪篇。
- 文档目录（2026-10-16 重整；目录与文件均英文小写＋下划线，各篇 H1 保留中文）：
  `doc/requirements/`（business 业务需求 / development 开发需求）、
  `doc/modules/`（模块分册=module_id，入口 `INDEX.md`，按数据流角色分组）、
  `doc/chains/`（三条保留链路专篇）、
  `doc/guides/`（runbook 端口与启停 / database_schema 数据库结构）、
  `doc/design/`（storage_design / live_overlay 已落地；video_review_search / wemm_integration 下期设计稿）、
  `doc/agent/`（static_prompts / full_run_record，挂载 agent 生成物）、
  `doc/research/`（历史选型调研）。
- 完整安装 / 移植步骤见 `INSTALL.md`；部署台账见 `DEPLOYMENT.md`。

## 六、常用验证命令（cd 到项目根执行）

```bash
PY="conda run -n ai python"   # 便携：ai conda 环境解释器（或先 conda activate ai 后用 python）

$PY umvp/pipe/test_composer.py            # 纯逻辑（不加载模型，秒级）
$PY tests/test_cropper.py                 # 通用裁剪
$PY tests/test_target_crop.py             # 目标裁剪
$PY tests/test_annotate.py                # 标注
$PY tests/test_det_filter.py              # 检测筛选
$PY tests/test_seq_filter.py              # 时序判定
$PY tests/test_police_uav_pipeline.py     # 警用链路逻辑断言
$PY tests/test_police_uav_video.py        # 警用链路端到端（真实 YOLO + mock VLM）
$PY tests/test_face_extract_pipeline.py   # 人脸提取链路
$PY tests/test_face_capture_pipeline.py   # 人脸截取链路（PIPE0003）
$PY tests/test_plate_recog_pipeline.py    # 车牌识别链路
$PY tests/test_agent_runtime.py           # 挂载 agent 运行时（离线）
$PY tests/test_yolo_params.py             # YOLO 参数对比（真实推理，较慢）
$PY umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz
$PY umvp/resources.py                     # 资源估算自检（读 out/fingerprints.json 若存在）
$PY web_lab/calibrate.py                  # 实测标定 → out/fingerprints.json（换机必跑）
$PY web_lab/validate_chain.py <spec.json> # 链路 stages spec 校验
$PY umvp/plate_recog/fetch_models.py      # 拉取车牌模型（换机 / 新克隆必跑）
```

## 七、注意事项

- `onnxruntime` 会打印大量 `VerifyOutputSizes` / `Expected shape` 形状警告（SCRFD 动态 batch
  正常现象）和 `Duplicate provider` 警告，均不影响结果，验证时用 `grep -v` 过滤。
- 性能基线以 `out/fingerprints.json`（`web_lab/calibrate.py` 产出）为唯一权威；机器变化
  （换卡 / 换机）只需重跑 `calibrate.py`，代码零改动。
- 联网检索 / 抓取优先用 GLM 系 MCP 工具（web-search-prime / web-reader / zread 免确认直用），
  完整优先级与配置见全局 `~/.config/opencode/AGENTS.md`。
