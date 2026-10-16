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
| WO-0001 | 2026-10-16 | orchestrator | qa | PIPE0003 web 测试台人脸链路径疑似未过 det_filter（来源 pipe-chain INB-0008）：server.py _face_step/is_face_chain 不走 Pipeline.step，DF0003 过滤疑未生效，需 qa 复现后定 web/pipe-chain 修法 | web 测试台跑 PIPE0003 对比 det_filter 前后记录数 | 子单全部完成（.1 测试定位 / .2 文档重写）。剩余待用户裁定：DF0003 在测试台人脸链不执行的修法（① web 工具改走 pipe.step ② pipe-chain 提供 step 化人脸入口 ③ 调整 PIPE0003 定义去掉 DF0003）——裁定后开 WO-0002 实施 |

## 待测试

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|

## 已完成

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| WO-0001.2 | 2026-10-16 | orchestrator | pipe-chain | 按代码逻辑重写 doc/chains/pipe0003_face_capture.md：写明测试台人脸链以 _face_step 执行（frame_manager→yolo→target_crop→face_detect），det_filter(DF0003) 声明但未执行；对照 PIPE0001/0002 走 pipe.step 的差异；参数细节指向 modules 分册 | 对照 web_lab/server.py 人脸链路径逐段核实；文档相对链接自检 OK | orchestrator 验收：197 行六段式+已知差异节；server.py 行号引用属实(1057/1108/1386)、h_pipeline_run(:929) 入口更正已采纳；全 doc 链接自检 OK |
| WO-0001.1 | 2026-10-16 | orchestrator | qa | 验证 PIPE0003 端到端通畅度＋确认 det_filter(DF0003) 在 web 测试台人脸链路径是否生效（INB-0008 疑点） | tests/test_face_capture_pipeline.py 全绿；web 链路实测产物与 timeline 对比 det_filter 前后 | qa 会话取消后由 orchestrator 接手完成：逻辑测试 39/39；det_filter 未生效已在代码级定位（server.py:1386 注释 + _face_step(:1108) 无 det_filter 调用；PIPE0001/0002 走 pipe.step(:1396/:1637) 会过滤）。结论：链路可用，DF0003 在测试台人脸链路径不执行 |

## 已取消

| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |
|---|---|---|---|---|---|---|
