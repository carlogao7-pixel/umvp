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

- [2026-10-16][INB-0024][qa][成果] WO-0002.2 验证通过。**新增 `tests/test_image_source.py`（59 断言）**：①`records`——`to_record(CropResult/FaceResult/PlateResult)` 均补 `image` 且旧键 `img/face_img/plate_img` 仍在（同 ndarray 对象）、`plate_img=None` 不凭空造 `image`、`Detection` 不受影响、`image_of` 优先 `image` 回退别名且支持对象属性、dict 输入补键不改源、已有 `image` 不被别名覆盖；②`_pick_vlm_image`——annotated 取标注图（缺则回退原帧）、frame 取原帧且与 annotated 逐像素不同、crops 命中取子图（`_match_crop_for_req` 精确/IoU≥0.5/`req.det.bbox`）、未命中回退且 note 含「回退」、无 last_crops 回退；③crop+vlm 端到端——`_chain_from_spec` 装配真实 `TargetCrop`（`isinstance` 断言），`step()` 产生 1 submit 且 `last_crops` 有值、`image_source=crops` 取到与 `last_crops[0].img` 一致的子图；仅 crop 无 vlm 时 submits=0（既有行为）；④驱动层（monkeypatch `db.load_pipeline`+`_vlm_real`+`_endpoint_reachable`，进程内）——缺省（不填）送出图与显式 annotated **逐像素一致**；frame/crops 与 annotated 确不同（frame 同尺寸不同像素、crops 313x210 < 全帧 779x1391）；timeline 备注「·素材:原帧」「·素材:裁剪子图」，annotated/缺省无备注。**`tests/test_validate_chain.py` 13→20**：非法取值报错（`image_source='lowres'` 不在允许值）、`annotated` 无 annotate / `crops` 无 target_crop 为非致命 warning（errs 空、warnings 有）、旧 spec 未声明不报错不警告、`None` 容忍、旧调用签名 `validate(spec)` 兼容。**回归全绿**：test_composer 36 / test_cropper 26 / test_target_crop 35 / test_annotate 16 / test_det_filter 66 / test_seq_filter 43 / test_police_uav（10 段）全通过 / test_plate_recog 28 / test_face_capture 63 / test_agent_runtime 39。**未覆盖/遗留**：真实 VLM 调用与视频分支、SSE；`h_chain`（mock VLM 路径不下发图，不涉 image_source）；`crops` 按类报送多目标同类仅取首个命中子图（INB-0021/0022 已记）；主单 WO-0002 描述含 `lowres` 取值但本期未实现（WO-0002.2 范围=annotated/frame/crops），校验器按非法值拦截。
  → 去向：`tests/test_image_source.py`（新增）、`tests/test_validate_chain.py`；工单 WO-0002.2 置**已完成**（主单 WO-0002 仍进行中，待后续第 2 步）。

- [2026-10-16][INB-0023][agent-builder][成果] WO-0002.2 文档段（P5 第 1 步）：把 VLM 送审素材 `image_source` 写入链路壳标准与两角色文档。`01-shell-standard.md`：`vlm` 参数速记补 `image_source`；§2 新增「VLM 送审素材」段（三取值/默认 annotated=零回归/crops 须链路含 target_crop/缺源一律回退不报错，指向 `doc/modules/vlm.md`）+ 连线校验注（validate_spec 非法取值报错、「声明 annotated 无 annotate」「声明 crops 无 target_crop」非致命警告；附 crop+vlm 曾静默失效已修的历史注脚）+ §4 清单加一条自检。`05-decision-table.md §1` 加「VLM 送审素材」选择说明；`03-patterns-and-extension.md §3` 既有槽位 `last_crops` 消费者补 VLM(`image_source=crops`)。`pipeline-architect.md §3.2` 修正两处过时表述：`vlm` 输入由「画框帧，上游须有 annotate」改为「送审图：默认画框帧，另有原帧/目标裁剪子图可选」、`target_crop` 由「只喂二次识别，不喂 VLM」改为「也供 VLM 取目标子图（image_source=crops）」。快照重生成 `doc/agent/static_prompts.md`（总装工块含上述两条更新，生成时间 2026-10-16 14:58）。**未改** `pipeline-assembler.md`（§3 只给硬规则、§7 指向技术文档，无过时表述）、`02/04`（无合适位置）。验证 `tests/test_agent_runtime.py` **39/39**。
  → 去向：`.claude/agents/assembler/{01-shell-standard,03-patterns-and-extension,05-decision-table}.md`、`.claude/agents/pipeline-architect.md`、`doc/agent/static_prompts.md`；工单 WO-0002.2 保持**待测试**（整单由 qa 验证）。

- [2026-10-16][INB-0022][pipe-chain][成果] WO-0002.2 补漏段（P5 第 1 步）：修 `composer.Pipeline.step()` —— 此前仅凭「存在 `crop` 阶段」即 `pass # 目标裁剪已产出`、不产生 submits，致 `target_crop`+`vlm` 组合静默不送审、`image_source=crops` 无法端到端。改后判据为 `dets is not None and self.vlm is not None`（删去 `if self.crop is not None: pass` 与 `elif` 分支合并）：**crop 与 vlm 共存时仍 `submits=self._submits(dets)`**（crops 供下游/驱动层取子图的同时按类报送 VLM），无 vlm 时（PIPE0002 车牌/PIPE0003 人脸）行为不变，无 crop 时（PIPE0001）行为不变，仅在「crop 与 vlm 同时存在」这一此前失效组合上生效。`_submits()` 语义/门控（det_filter/seq_filter 已在上游）不变，未引入新参数。注释改为新事实（crop/vlm 可共存）。**验证全绿**：test_composer 36；test_police_uav 9 断言（4 直断言+5 循环，10 段）全通过；test_plate_recog 28；test_face_capture 63；进程内桩验证：`frame_manager+yolo+target_crop(Stub)+vlm` → `step()` 返回 submits=2（cls0/cls2 各 1），`last_crops=2` 并存；对照 `crop` 无 `vlm` → submits=0、last_crops=2。**结论（crops 端到端）**：`image_source=crops` 现已可用；驱动层 `_match_crop_for_req` 以 `req.det.bbox`（按类报送时 `det=None`）**或 `req.dets` 各框**匹配 `last_crops`，故 class 粒度 submits 亦能命中（无需 composer 再携带单目标 bbox）。**剩余建议（非本段改动）**：①按类报送为一条请求、多目标同类时 `crops` 源仅取首个命中子图，若需「多子图同送」需 VLM payload 支持多图（本期不做）；②裁剪含 margin、匹配用 IoU≥0.5，极端 margin 下可能未命中→回退 annotated（有日志，非错误）。
  → 去向：`umvp/pipe/composer.py`（`step()` :434-444）；工单 WO-0002.2 保持**待测试**（整单由 qa 验证）。

- [2026-10-16][INB-0021][web][成果] WO-0002.2 第 2 段（P5 第 1 步）：VLM 送审素材在壳内可选——`vlm` 阶段新增参数 `image_source`（`annotated`（默认=现状）/ `frame` / `crops`）。实现：`server.py` PARAMS["vlm"] 注册（`select`，非 test_only→设计器可见+随 spec 落库）；`h_pipeline_run` 从 vlm 阶段 params 解析（`:1055`）并在送审处用新助手 `_pick_vlm_image(source, req, frame, pipe)` 取图（`:1203`）。取图：`annotated`=last_annotated（缺则回退本帧原图，与改前**逐字节等价**）；`frame`=本帧原图；`crops`=`_match_crop_for_req` 按报送目标 bbox（优先 `req.det.bbox` 否则 `req.dets` 各框，精确或 IoU≥0.5）在 `last_crops` 找对应子图。**回退策略**：任一路径缺源都回退现状（annotated→原帧），绝不抛错；回退/非默认源在研判日志追加 `·素材:<说明>`（如「无匹配裁剪子图，回退标注图」）。校验：`validate_chain` 对 `image_source` 非法取值报错（`[ERR] stages[i] vlm.image_source='lowres' 不在允许值 ['annotated','frame','crops']`）；显式声明 `annotated` 但上游无 annotate / 声明 `crops` 但上游无 target_crop → **非致命 WARN**（新增 `validate(spec, warnings=[...])` 形参，旧调用方零影响；未声明则静默=现状）。顺带令 select 选项检查容忍 `None`（DB 空列还原不误报非法）。**验证全绿**：py_compile；test_validate_chain 13/13；test_face_capture 63；test_plate_recog 28；test_police_uav 全通过（10 段）；test_composer 36；进程内直调 `h_pipeline_run`（monkeypatch `db.load_pipeline`+`_endpoint_reachable`+`_vlm_real`，不连 DB/HTTP）三取值送审图各异（不填==annotated md5 6405a54f1d；frame md5 3ac6ef66d3；crops 命中=裁剪子图 md5 da7f5263ff 且小于全帧；未命中回退 annotated 并有日志）。
  → 去向：`web_lab/server.py`（PARAMS["vlm"] :1983、`VLM_IMAGE_SOURCES`/`_bbox_iou`/`_match_crop_for_req`/`_pick_vlm_image` :233-296、h_pipeline_run 解析 :1055 与取图 :1200-1216）、`web_lab/validate_chain.py`（`validate(...,warnings)` + 步骤 6 + select None 容忍）；工单 WO-0002.2 置**待测试**（整单待 qa 验证）。**遗留（供 qa/pipe-chain）**：①`composer.step` 在存在 `target_crop` 阶段时**不产生 submits**，故 `crops` 源在当前真链路里可得性受限（本次以薄包装补一条 submit 验证驱动层取图逻辑；端到端 crops 有赖 WO-0002.3 / pipe-chain 调整 composer），本单未改 composer（属 pipe-chain）；②`crops` 多目标同类时取首个命中子图（报送为按类一条，无单目标 bbox）；③低清图 `lowres` 本阶段不做（决策4）。 WO-0002.2 第 1 段（P5 第 1 步）：旁路图像记录统一键名 `image` + 旧键兼容。实现=**统一辅助函数**（最小侵入，单文件）：`records.py` 新增 `IMAGE_KEY="image"` / `IMAGE_ALIASES=("img","face_img","plate_img")` / `image_of()` 读取器 / `_unify_image_key()`；`to_record()` 四个分支末尾统一补键（旧键 `img`/`face_img`/`plate_img` 保留，旧键非空且无 `image` 时补 `image`，指向同一 ndarray）。效果：`to_record(CropResult/FaceResult/PlateResult)` 与 dict 输入（含 `img`）均新旧键并存；`Detection` 等无旁路图者原样不变。**对象属性 `c.img`/`f.face_img`/`pl.plate_img` 与 `web_lab/` 全部未动**（消费方零影响）。文档：`records.py` docstring 旁路约定、`doc/modules/INDEX.md`「〇通用语」+「两平面角色表」+ §二行、`target_crop.md`/`face_detect.md`/`plate_recog.md` 键名更新为「`image`（旧名 X 兼容）」。验证全绿：test_composer 36 / test_target_crop 35 / test_det_filter 66 / test_seq_filter 43 / test_annotate 16 / test_cropper 26 / test_face_capture 63 / test_plate_recog 28 / test_face_extract 28。
  → 去向：`umvp/pipe/records.py`（:39-98）、`doc/modules/{INDEX,target_crop,face_detect,plate_recog}.md`；工单 WO-0002.2 保持**进行中**（第 2 段 image_source 归 web，整单待 qa 验证）。

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
