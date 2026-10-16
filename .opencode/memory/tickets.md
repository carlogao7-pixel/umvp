# 工单（tickets.md）—— 多代理任务流

> **用途**：用户需求与各代理改动的在办跟踪（inbox 管「归档沉淀」，工单管「正在办」）。
> **操作一律走工具**：`python3 .opencode/tool/ticket.py`（new / set / list / show / check）；
> 备注更新用 `set --append-note "..."`（追加），`--note` 是**整字段覆盖**（慎用），
> 不要手改表格行；改完必跑 `check`。字段内**禁用竖线 `|`**。
> 完整协作规约见 `CLAUDE.md` §四「工单」。

## 编号与状态规约

- **主单** `WO-0001`（4 位全局递增）：用户需求经 orchestrator 审核澄清后创建；代理发现的
  独立新问题也可自建主单（发起人=该代理）。
- **子单** `WO-0001.1`（所属主单下递增）：各代理开工前创建，记录本板块的具体改动描述；
  跨板块同步需要 → 子单 `--to` 指派目标代理并在回报中 @对方。琐碎修正并入原子单备注，不新开。
- **状态流转**：`进行中`（开工）→ `待测试`（自验通过，等 qa/发起人验证）→ `已完成`；
  需求撤销用 `已取消`。主单在全部子单关闭（已完成/已取消）前保持进行中。
- **验证**：谁置「待测试」谁填验证命令；qa 领取「待测试」节验证，通过置「已完成」
  （备注断言数/命令），不通过退回「进行中」并 @承接方；纯文档单可由发起人直接完成。
- **字段**：编号｜日期｜发起人（orchestrator/pipe-chain/web/agent-builder/qa）｜承接（逗号分隔）｜
  需求/改动描述｜验证（命令或方式）｜备注（产出、关联 INB-xxxx、阻塞原因等）。

## 进行中

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|

## 待测试

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|

## 已完成

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| WO-0002 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 拼接层端口协议 P5（分步实施）：①第0步 人脸并入 Pipeline.step 插槽（删 _face_step 专用路径）+ validate_chain 增「声明阶段必被装配」断言 + 壳标准写「逐帧阶段一律走插槽」规矩；②第1步 旁路命名统一 + spec 支持 image_source（frame/annotated/crops/lowres）；③第2步 完整端口协议（产物表+INPUTS/OUTPUTS 声明+inputs 连线+校验器+设计器连线 UI） | validate_chain 新增断言逐条有用例；PIPE0001/0002/0003 + 模块自检全绿 | 第0步（WO-0002.1）+ 第1步（WO-0002.2）均已实施并验证通过（旁路 image 键统一、vlm.image_source 三取值、crop+vlm 补漏；新增 test_image_source 59 项，validate_chain 20 项，回归全绿）。剩第2步：完整端口协议（产物表+INPUTS/OUTPUTS 声明+inputs 连线+校验器+设计器连线 UI）待派发；三步全部完成并验证（.1 人脸并入step / .2 image_source / .3 端口协议）。遗留（非阻塞，待后续评估）：①审核器"本轮未纳入解析端口"=vlm/plate_recog/face_detect 的 inputs 接线（依赖 crops 产物形态统一）②face_embed.thresh 参数表缺项（INB-0017）③crops 同类多目标仅取首图（INB-0021/0022）④真实 MySQL 往返与浏览器设计器交互未实跑 |
| WO-0002.3 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 第2步（完整端口协议）：①Pipeline 产物表 artifacts + 模块 INPUTS/OUTPUTS 声明 + spec 阶段 inputs 连线解析（默认=现状）；②web 装配器透传 inputs + validate_chain DAG 校验（产物存在/在上游/类型匹配/无环）+ 设计器连线 UI 下拉选产物；③INDEX 两平面角色表升级为产物端口表；④新增模块接入=登记声明即可 | 默认零回归：全套模块/链路测试绿；声明 inputs 后模块收到对应产物；非法连线被校验器拦截 | 两段完成：核心（pipe-chain）artifacts 产物表 + 模块 INPUTS/OUTPUTS + spec 阶段 inputs 解析（默认零回归，test_composer 36→59）；装配/校验/UI（web）——_chain_from_spec 透传 stage_inputs、validate_chain 新增 inputs DAG 校验（未知产物/非法端口/非上游引用/类型不符/多值→错；vlm/plate_recog/face_detect 未纳入解析→非致命警告；旧 spec 零报错零警告）、db.load_pipeline 经 meta.stage_inputs 回填连线、index.html 设计器输入连线下拉（仅对已纳入解析端口开放）。断言数：test_validate_chain 20/20、test_image_source 59、test_face_capture 63、test_composer 59、test_plate_recog 28、test_police_uav（10 段）全通过；web 进程内自检 28 项 + UI 自检 13 项。待 qa 补测（见 inbox INB-0026）。另：端口协议已写入壳标准/技术文档（agent-builder 段）。【qa 验证通过 2026-10-16】新增 tests/test_port_inputs.py（39 断言：模块端口声明/_chain_from_spec 透传形状/端到端间谍捕获 annotate.run 实参/fail-fast/CLI 退出码）+ test_validate_chain 20→30（10 条 DAG 用例：未知产物/非法端口/非上游/自引用/类型不符/列表值/同产物多下游允许/vlm-plate-face 警告不致命/旧 spec 零告警）。端到端证据：同链同入参 dets(0.9/0.2/0.7, det_filter score>=0.5)，annotate.dets=yolo.dets 收到 3 条、默认与=det_filter.dets 收到 2 条（确实不同）。CLI：合法 exit=0 含[OK]、非法 exit=1 含[ERR]。回归全绿：validate 30/30、port_inputs 39、image_source 59、composer 59、target_crop 35、annotate 16、det_filter 66、seq_filter 43、agent_runtime 39、plate_recog 28、face_capture 63、police_uav 全通过(10 段)。无产品缺陷。 |
| WO-0002.2 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 第1步：旁路命名统一（crop img/face face_img/plate plate_img → records 约定的 image 语义，保留兼容别名）+ spec 阶段支持 image_source（frame/annotated/crops，默认=现状 annotated）让 VLM 素材在壳里可选 | 模块自检全绿；三条保留链路回归；image_source 默认时不改变现有输出 | qa 验证通过（2026-10-16）：新增 tests/test_image_source.py（59 断言：records 键统一/补键/旧别名保留/image_of/Detection 不受影响；_pick_vlm_image 三取值与回退；crop+vlm 真实 TargetCrop 端到端 submits+last_crops+crops 命中；驱动层缺省==annotated 逐像素一致、frame/crops 切图确不同、timeline 素材备注）；tests/test_validate_chain.py 13→20（非法取值报错/annotated 无 annotate 与 crops 无 target_crop 非致命警告/旧 spec 不报错不警告/None 容忍/旧签名兼容）。回归全绿：composer36 / cropper26 / target_crop35 / annotate16 / det_filter66 / seq_filter43 / police_uav 全通过 / plate_recog28 / face_capture63 / agent_runtime39。未覆盖：真实 VLM 调用/视频分支/SSE、h_chain（mock VLM 路径不涉 image_source）、crops 多目标同类仅取首个命中。 |
| WO-0002.1 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 第0步：人脸链并入 Pipeline.step（face 插槽，与 plate 对称）+ 删 _face_step/is_face_chain 专用路径（落盘/计数留驱动层）+ validate_chain「声明阶段必被装配」断言 + 01-shell-standard.md 写入插槽规矩 | tests/test_face_capture_pipeline.py + test_composer.py + PIPE0001/0002 回归全绿；构造「声明但未装配」spec 校验器必须报错 | qa 验证通过：test_face_capture 39→63（服务级装配 face 插槽 + det_filter 生效数据点 + 驱动层落盘命名/上限/时间线）；validate_chain 11→13（声明但未装配 face_store 必须报错）；回归全绿 composer36/face63/validate13/13/police全通过/plate28/agent39。未覆盖：h_chain/SSE 与视频分支、face_embed 身份去重驱动层。 |
| WO-0001 | 2026-10-16 | orchestrator | qa | PIPE0003 web 测试台人脸链路径疑似未过 det_filter（来源 pipe-chain INB-0008）：server.py _face_step/is_face_chain 不走 Pipeline.step，DF0003 过滤疑未生效，需 qa 复现后定 web/pipe-chain 修法 | web 测试台跑 PIPE0003 对比 det_filter 前后记录数 | 修法已裁定：并入通用 pipe.step（原选项①）——实施归 WO-0002.1（P5 第0步）；文档已如实记录「已知差异」 |
| WO-0001.2 | 2026-10-16 | orchestrator | pipe-chain | 按代码逻辑重写 doc/chains/pipe0003_face_capture.md：写明测试台人脸链以 _face_step 执行（frame_manager→yolo→target_crop→face_detect），det_filter(DF0003) 声明但未执行；对照 PIPE0001/0002 走 pipe.step 的差异；参数细节指向 modules 分册 | 对照 web_lab/server.py 人脸链路径逐段核实；文档相对链接自检 OK | orchestrator 验收：197 行六段式+已知差异节；server.py 行号引用属实(1057/1108/1386)、h_pipeline_run(:929) 入口更正已采纳；全 doc 链接自检 OK |
| WO-0001.1 | 2026-10-16 | orchestrator | qa | 验证 PIPE0003 端到端通畅度＋确认 det_filter(DF0003) 在 web 测试台人脸链路径是否生效（INB-0008 疑点） | tests/test_face_capture_pipeline.py 全绿；web 链路实测产物与 timeline 对比 det_filter 前后 | qa 会话取消后由 orchestrator 接手完成：逻辑测试 39/39；det_filter 未生效已在代码级定位（server.py:1386 注释 + _face_step(:1108) 无 det_filter 调用；PIPE0001/0002 走 pipe.step(:1396/:1637) 会过滤）。结论：链路可用，DF0003 在测试台人脸链路径不执行 |

## 已取消

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
