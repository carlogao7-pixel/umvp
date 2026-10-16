# Agent 收件箱（inbox.md）

所有开发板块代理的「成果 / 新坑 / 待归档 / 部署相关」内容**先追加到这里**；
主代理用 `/curate` 筛选后转入 `CLAUDE.md`、`DEPLOYMENT.md` 或 `doc/`。

**只追加、不修改历史条目。** 本文件不进 `instructions`，默认不随会话载入，
由 `/curate` 与写入时按需读取。

条目格式：

```
- [YYYY-MM-DD][INB-0001][来源代理][类型] 一句话结论（关键文件/命令）
```

- 来源代理：`orchestrator` / `pipe-chain` / `web` / `agent-builder` / `qa`
- 类型：`成果` / `坑` / `待归档` / `部署`
- 编号 `INB-xxxx` 全局递增（取当前最大 +1，勿复用）

---

## 待同步

<!-- 新条目追加在此行下方，最新在上 -->

- [2026-10-16][INB-0019][qa][成果] WO-0002.1 验证通过：`tests/test_face_capture_pipeline.py` 新增服务级/驱动层用例（39→63 断言）——装配层注入 `Pipeline.face`（face_detect 新形态 + extract_faces 兼容形态 + face_embed 注入 FaceEmbedder）；det_filter 生效（同帧同整帧检测：放行 `long>=1` → `last_filter=(1,1)`/裁 1 人子图/检 3 脸；严格 `long>=100000` → `(1,0)`/0 裁剪/0 脸）；驱动层 `h_pipeline_run` 图片分支（仅 monkeypatch `db.load_pipeline` 返内存 spec，不连 MySQL/不起 HTTP）落盘命名 `face_001_0.0s.jpg`、上限 2（原图 1 + 脸 1，超限跳过 6）、时间线出现 det_filter `10 → 0` 条目。`tests/test_validate_chain.py` 11→13（新增「声明但未装配 face_store → 报错」+ face/face_embed 形态全绿）。回归全绿：test_composer 36 / face_capture 63 / validate 13/13 / police_uav 全通过 / plate_recog 28 / agent_runtime 39（脚本自报；qa 个人上下文旧表写 33，以脚本为准）。未覆盖边界：`h_chain`/SSE 与视频分支、face_embed 身份去重驱动层（需 DB 或视频素材）；另 `extract_faces` 在 `CHAIN_ASSEMBLED_MODULES` 但不在 `MODULES` 注册表 → 声明它会被 validate 判「未知模块」而跳过「声明即执行」（兼容形态，非回归）。`face_embed.thresh` 无参数表项（恒默认 0.55）已由 web 记 INB-0017，不重复。
  → 去向：`tests/test_face_capture_pipeline.py`、`tests/test_validate_chain.py`；工单 WO-0002.1 置**已完成**。

- [2026-10-16][INB-0018][agent-builder][成果] WO-0002.1 第 3 段（P5 第 0 步）：链路壳标准写入「逐帧阶段一律走 `Pipeline.step` 插槽、禁止驱动层专用单帧路径」硬规则（§0 新增硬规则 5 +「执行路径」说明：新模块接入三件套 ①插槽②`_chain_from_spec` 注入③`CHAIN_ASSEMBLED_MODULES` 登记；校验保障=`validate_chain`「声明即执行」断言；反例=PIPE0003 专用路径致 det_filter 声明不执行/WO-0001，已修复）。同步修正过时表述：`01-shell-standard.md` 槽位清单补 `face`、`pipeline-assembler.md §3` 硬规则补第 5 条、`03-patterns-and-extension.md` 槽位示例 `crop/annotate/plate`→`+face` 且「四件套」第 4 项点名 `CHAIN_ASSEMBLED_MODULES`。快照重生成 `doc/agent/static_prompts.md`（组装工块含新规矩）。验证 `tests/test_agent_runtime.py` 39/39。
  → 去向：`.claude/agents/assembler/{01-shell-standard,03-patterns-and-extension}.md`、`.claude/agents/pipeline-assembler.md`、`doc/agent/static_prompts.md`；工单 WO-0002.1 保持待测试（整单由 qa 验证）。

- [2026-10-16][INB-0017][web][坑] 驱动层人脸身份去重阈值读 `face_embed.thresh`，但 `face_embed` 参数表（server.py PARAMS）**无 thresh 项** → 实际恒取默认 0.55；要按链路配置阈值需给 face_embed 补 `thresh`（会新增 DB 参数表列）。另：`_chain_from_spec` 现对含 face_detect/face_embed 的 spec 在**装配期**即构造 FaceDetector/FaceEmbedder（构造即加载模型），h_chain/h_pipeline_stream 走人脸 spec 时同样会加载。
  → 去向：供 pipe-chain（是否补 face_embed.thresh 参数表）/ qa（阈值用例）跟进；`doc/chains/pipe0003_face_capture.md` §4.1「两条执行路径的差异」与「已知差异」节可标记为**已修复**（WO-0002.1 第 2 段）。
- [2026-10-16][INB-0016][web][成果] WO-0002.1 第 2 段：测试台人脸链并入通用 `Pipeline.step`（删 `_face_step` / `is_face_chain` 专用路径）+ validate_chain「声明即执行」断言。`_chain_from_spec` 新增 `CHAIN_ASSEMBLED_MODULES`、注入 `Pipeline.face`（FaceStage）并附挂 `pipe.face_embedder`/`pipe.face_dedup_thresh`；兼容 extract_faces 形态自动补 TargetCrop/YOLO。`h_pipeline_run` 图片/视频分支统一为 `pipe.step` + `_step_out`（从 `pipe.last_faces`/`last_crops` 落盘、计数、身份去重；沿用 `face_{n:03d}_{ts:.1f}s`、images≤36、落盘 200 上限、「人子图」文案）。验证：test_composer 36 / test_face_capture 39 / test_police_uav 全通过 / test_plate_recog 28 / test_validate_chain 11/11；进程内 18 项（含严格 det_filter 10→0 证明生效）。
  → 去向：`web_lab/server.py`（:1052 人脸链状态、:1102「人子图」、:1182 人脸展示段、:1259 图片分支、:1342 视频分支、:1404 `CHAIN_ASSEMBLED_MODULES`、:1484 `_chain_from_spec` 人脸装配）、`web_lab/validate_chain.py`（:158 声明即执行断言）。**行为差异**：PIPE0003 的 det_filter(DF0003) 在测试台**真正生效**（过滤后检测数→裁剪→人脸），人脸数/落盘可能与旧日志不同（属修复目的，非回归）。

- [2026-10-16][INB-0015][pipe-chain][成果] `Pipeline` 增 face 插槽（与 plate 对称，WO-0002.1 第 1 段）：新增字段 `face`/`last_faces`/`last_face_input`，`step()` 增「人脸识别阶段」段 `face.run(frame, last_crops, ts)`（落盘/计数/事件尾留驱动层），docstring 补插槽说明；不改行为。
  → 去向：`umvp/pipe/composer.py`（:350 docstring、:361 face 字段、:365 last_faces、:368 last_face_input、:458 step 人脸段）；web 接线与 validate_chain 断言待 WO-0002.1 后续段落。

## 已同步

<!-- /curate 处理后移到这里，注明去向，例如：INB-0001 → CLAUDE.md §三 设计契约 -->

- [2026-10-16][INB-0014][pipe-chain][成果] `doc/chains/pipe0003_face_capture.md` 按代码逻辑重写（WO-0001.2）：新增 §4.1 两条执行路径差异 + 「已知差异」节；纠正入口为 `/api/test/pipeline_run`（非 /api/test/chain）。
  → 去向：`doc/chains/pipe0003_face_capture.md`（已落）；待决修法挂工单 WO-0001。
- [2026-10-16][INB-0013][orchestrator][部署] 多代理协作基建入 git（`.opencode/{agent,command,memory,tool}`+`package*.json`+`opencode.json`，node_modules ignore）。
  → 去向：git 提交 e3ec9cd；`DEPLOYMENT.md`（2026-10-16 基建入库条目）。
- [2026-10-16][INB-0012][orchestrator][成果] 工单任务流：`tickets.md` + `ticket.py`（new/set/list/show/check，19/19 自测）。
  → 去向：`CLAUDE.md` §四「工单」、五个 `.opencode/agent/*.md`、`.opencode/tool/ticket.py`。
- [2026-10-16][INB-0011][orchestrator][成果] doc 第二轮重整：链路独立 `chains/`、文件全量英文名、INDEX 按数据流角色分组。
  → 去向：`doc/README.md` §三（命名规则权威）、`CLAUDE.md` §五、`DEPLOYMENT.md`；取代 INB-0006。
- [2026-10-16][INB-0010][web][成果] guides/design 四篇英文名 + 文内互引/代码注释路径同步。
  → 去向：`doc/guides/{runbook,database_schema}.md`、`doc/design/{storage_design,live_overlay}.md`、`web_lab/{db,server}.py|index.html`、`umvp/resources.py` 注释。
- [2026-10-16][INB-0009][agent-builder][成果] agent/ 两篇英文名 + 生成脚本默认路径 + 两角色提示词路径同步 + 快照重生成。
  → 去向：`doc/agent/{static_prompts,full_run_record}.md`、`.claude/agents/{pipeline-architect,pipeline-assembler}.md`、`web_lab/{dump_prompts,agent_run,agent}.py`。
- [2026-10-16][INB-0008][pipe-chain][坑] PIPE0003 声明 det_filter 但测试台人脸链路径（`_face_step`）不执行。
  → 处置：已经主代理核实（代码级定位 + 链路逻辑回归 39/39），落 `doc/chains/pipe0003_face_capture.md`「已知差异」；修法裁定挂工单 **WO-0001**。
- [2026-10-16][INB-0007][pipe-chain][成果] modules 按数据流角色重组 + chains 抽篇 + 11 篇英文名（module_id 对齐）。
  → 去向：`doc/modules/INDEX.md`（五组数据流角色 + 两平面角色表）、`doc/chains/*`、`CLAUDE.md` §一。
- [2026-10-16][INB-0006][orchestrator][成果] doc 第一轮重整（分类入目录 + 去冗余）。**已被 INB-0011 取代**（文件中文名约定改为英文名）。
  → 去向：`doc/README.md`、`CLAUDE.md` §五、`DEPLOYMENT.md`。
- [2026-10-16][INB-0005][agent-builder][成果] 挂载 agent 两篇迁入 `doc/agent/`（标注自动生成勿手改）+ 脚本默认路径同步。
  → 去向：`doc/agent/*`、`web_lab/{dump_prompts,agent_run}.py`。
- [2026-10-16][INB-0004][web][成果] web 四篇分类迁移 + 梳理（含事实修正：参数表 9→11、presets 快照 15→23 行、SSE 事件表删未实现 info）。
  → 去向：`doc/guides/*`、`doc/design/*`。
- [2026-10-16][INB-0003][pipe-chain][成果] modules 13 篇体例统一 + 去冗余 + 事实修正（车牌库内值、警用链路 AL0001/alarm、人脸检测补参等）。
  → 去向：`doc/modules/*`（各分册）；
- [2026-10-15][INB-0001][orchestrator][待归档] 建立多代理协作：新增 `.opencode/agent/`（orchestrator/pipe-chain/web/agent-builder/qa）、`.opencode/memory/inbox.md` 收件箱、`.opencode/command/curate.md`，项目 `opencode.json` 设 `default_agent: orchestrator`；`CLAUDE.md` 追加「多代理协作」节。（不直接改 CLAUDE.md/DEPLOYMENT.md，交 /curate）
  → 尝试性记录，与当前项目内容无关；随本节 CLAUDE.md 共享上下文重整一并归档（多代理协作内容已由 `CLAUDE.md` §一/§四 承载），不另立文档。
- [2026-10-15][INB-0002][orchestrator][成果] 按"共享上下文"定位重整 `CLAUDE.md`（230→147 行）：新增 §一「板块划分与对接」（各板块介绍/技术栈/对接楔子/文档链接），删去参数/数据/断言数/MCP 重复段；技术细节下沉各 `.opencode/agent/*.md`；修正 `class_limits`→`group_limits`、「9 个模块」过期计数；`orchestrator.md` 修正过时的 model 说明。
  → 去向：`CLAUDE.md` §一/§三/§四/§七、`.opencode/agent/{pipe-chain,web,qa,orchestrator}.md`。纯文档改动，不涉及运行配置，未记 `DEPLOYMENT.md`。
