# 组装工技术文档（.claude/agents/assembler/）

> 读者：**组装工**（`../pipeline-assembler.md`）｜ 定位：把总装工的「语义模块链条」落成合规
> `stages spec` 的**全部技术依据**。**自包含**：不依赖任何外部技能库。
> 用法：组装工用 `read_doc` 按路径取用，例如 `read_doc .claude/agents/assembler/01-shell-standard.md`。

## 文档索引

| 文件 | 内容 | 何时读 |
|---|---|---|
| `01-shell-standard.md` | **链路壳标准（唯一权威）**：spec schema + 硬规则 + 参数白名单来源 + 标准壳 + 装配自检清单 + 红线 | 写 / 改 spec 前必读 |
| `02-assembly-workflow.md` | 组装工作流 + 交付包契约 + 汇报模板 | 接到任务时 |
| `03-patterns-and-extension.md` | 拼装套路 / 胶水 / 槽位判定 + **新模块接入 / 新编排逻辑演进** | 遇新需求 / 扩展时 |
| `04-pitfalls-and-checklist.md` | 常见坑 + 已验证 / 未验证清单 | 选参数 / 交付前 |
| `05-decision-table.md` | 参数初值（YOLO 等实测结论） | 落参数时 |

## 动态事实（不写死在文档里，一律走工具现取）

- **模块清单与精确参数** → `list_modules` 工具：唯一与代码同步的模块目录，**新模块注册后自动出现**。
- **模块 I/O 字段级细节** → 项目文档 `doc/modules/<模块>.md`。
- **已存链路现状** → `list_pipelines` / `get_pipeline`。
- **资源估算** → `estimate` 工具（静态公式 + 本机实测指纹）。

## 维护规则（迭代时更新哪个文件）

- **新增模块**：本文档无需改——`list_modules` 自动纳入；若它引入新的**旁路槽位 / 阶段**，
  更新 `03-patterns-and-extension.md` 的"新模块接入四件套"。
- **新增 / 改变编排逻辑、数据流**：更新 `01-shell-standard.md`（硬规则 / 数据流）与 `03`（扩展点）。
- **参数实测 / 调优结论**：追加 `05-decision-table.md` 与 `04-pitfalls-and-checklist.md`。
- **新坑**：追加 `04-pitfalls-and-checklist.md`。

> 权威源仍是代码（`umvp/`、`web_lab/`）与 `CLAUDE.md`：本文档是面向组装工的**自包含快照**，
> 与代码冲突时以代码 / `list_modules` 为准，并顺手同步本文档。
