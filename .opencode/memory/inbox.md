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
