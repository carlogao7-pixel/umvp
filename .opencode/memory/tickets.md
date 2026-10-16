# 工单（tickets.md）—— 多代理任务流

> **用途**：用户需求与各代理改动的在办跟踪（inbox 管「归档沉淀」，工单管「正在办」）。
> **操作一律走工具**：`python3 .opencode/tool/ticket.py`（new / set / list / show / check），
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
| WO-0002 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 拼接层端口协议 P5（分步实施）：①第0步 人脸并入 Pipeline.step 插槽（删 _face_step 专用路径）+ validate_chain 增「声明阶段必被装配」断言 + 壳标准写「逐帧阶段一律走插槽」规矩；②第1步 旁路命名统一 + spec 支持 image_source（frame/annotated/crops/lowres）；③第2步 完整端口协议（产物表+INPUTS/OUTPUTS 声明+inputs 连线+校验器+设计器连线 UI） | validate_chain 新增断言逐条有用例；PIPE0001/0002/0003 + 模块自检全绿 | 第0步（WO-0002.1）已实施并验证通过：face 并入 step + 删 _face_step + validate_chain「声明即执行」断言 + 壳标准规矩；test_face_capture 63 项/validate_chain 13 项/回归全绿。第1步（旁路命名统一+image_source）、第2步（完整端口协议）待派发 |

## 待测试

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|

## 已完成

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| WO-0002.1 | 2026-10-16 | orchestrator | pipe-chain,web,agent-builder | 第0步：人脸链并入 Pipeline.step（face 插槽，与 plate 对称）+ 删 _face_step/is_face_chain 专用路径（落盘/计数留驱动层）+ validate_chain「声明阶段必被装配」断言 + 01-shell-standard.md 写入插槽规矩 | tests/test_face_capture_pipeline.py + test_composer.py + PIPE0001/0002 回归全绿；构造「声明但未装配」spec 校验器必须报错 | qa 验证通过：test_face_capture 39→63（服务级装配 face 插槽 + det_filter 生效数据点 + 驱动层落盘命名/上限/时间线）；validate_chain 11→13（声明但未装配 face_store 必须报错）；回归全绿 composer36/face63/validate13/13/police全通过/plate28/agent39。未覆盖：h_chain/SSE 与视频分支、face_embed 身份去重驱动层。 |
| WO-0001 | 2026-10-16 | orchestrator | qa | PIPE0003 web 测试台人脸链路径疑似未过 det_filter（来源 pipe-chain INB-0008）：server.py _face_step/is_face_chain 不走 Pipeline.step，DF0003 过滤疑未生效，需 qa 复现后定 web/pipe-chain 修法 | web 测试台跑 PIPE0003 对比 det_filter 前后记录数 | 修法已裁定：并入通用 pipe.step（原选项①）——实施归 WO-0002.1（P5 第0步）；文档已如实记录「已知差异」 |
| WO-0001.2 | 2026-10-16 | orchestrator | pipe-chain | 按代码逻辑重写 doc/chains/pipe0003_face_capture.md：写明测试台人脸链以 _face_step 执行（frame_manager→yolo→target_crop→face_detect），det_filter(DF0003) 声明但未执行；对照 PIPE0001/0002 走 pipe.step 的差异；参数细节指向 modules 分册 | 对照 web_lab/server.py 人脸链路径逐段核实；文档相对链接自检 OK | orchestrator 验收：197 行六段式+已知差异节；server.py 行号引用属实(1057/1108/1386)、h_pipeline_run(:929) 入口更正已采纳；全 doc 链接自检 OK |
| WO-0001.1 | 2026-10-16 | orchestrator | qa | 验证 PIPE0003 端到端通畅度＋确认 det_filter(DF0003) 在 web 测试台人脸链路径是否生效（INB-0008 疑点） | tests/test_face_capture_pipeline.py 全绿；web 链路实测产物与 timeline 对比 det_filter 前后 | qa 会话取消后由 orchestrator 接手完成：逻辑测试 39/39；det_filter 未生效已在代码级定位（server.py:1386 注释 + _face_step(:1108) 无 det_filter 调用；PIPE0001/0002 走 pipe.step(:1396/:1637) 会过滤）。结论：链路可用，DF0003 在测试台人脸链路径不执行 |

## 已取消

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
