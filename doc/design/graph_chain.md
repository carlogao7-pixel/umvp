# 图编排（graph_chain）设计稿

> 状态：📐 **设计已定稿（2026-10-19），P2-0 + P2-1 待实施**。本文是
> `assembly_ports.md`（端口协议 P5）的**拓扑扩展**：端口/产物/类型的既有定义不推翻，只做扩展。
> 用户已确认：①按「先 P2-0 + P2-1，跑稳再上」的节奏实施；②§9 未决点按本稿建议拍定
> （产物全名带 `#position`、引用键用 `position`、连线内联 stage、单上游挪写入期校验、
> `Images`/`fan_in`/`resolved_by` 本期预置）；③新增模块接入成本单列一章（§10）。
> 实施工单：**WO-0008**（分步 P2-0 / P2-1）。
>
> 结论先行：链路模型从「有序 `stages` 列表 + 隐式连线」升级为「**节点图**」——
> 节点（阶段实例）有稳定身份、可复用；连线成为一等公民（可任意 fan-out / fan-in）；
> 端口能力由目录表声明。**本期不改执行顺序**（`Pipeline.step()` 仍按既有顺序跑），
> 只解「实例身份」与「连线存储」两件事，为后续拓扑驱动执行铺路。

## 1. 定位与范围

**本期（P2-0 + P2-1）做**：

1. 节点实例身份：产物全名带 `#position`，`stage_inputs` 键由 `module_id` 换成 `position`；
2. 解除「同模块不可复用」的四道锁；
3. 连线成为一等公民：`pipeline_edges` 表 + `module_ports` 目录表 + 迁移与双写。

**本期明确不做**（后续期，见 §8）：拓扑驱动执行、多终段、过渡节点模块、UI 图编辑画布、
`vlm` 参数迁移。

**兼容底线**：三条保留链路（`pipe0001`/`pipe0002`/`pipe0003`）与全部既有逻辑测试**零回归**；
未声明 `inputs` 的旧 spec 逐字零回归。

## 2. 现状基线

| 维度 | 现状 | 位置 |
|---|---|---|
| 链路模型 | 有序列表 `stages[]`；**执行顺序硬编码**，spec 顺序**不驱动执行** | `umvp/pipe/composer.py` `Pipeline.step()` |
| 节点身份 | 每模块**一个插槽**（`Pipeline` 字段） | `composer.py` `class Pipeline` |
| 产物命名 | 全局扁平 `{module_id}.{artifact}`；8 产物 / 2 类型（`Image`/`Records`） | `composer.py` `ARTIFACT_TYPES`、`RESOLVED_PORTS` |
| 连线存储 | `pipelines.meta_json.stage_inputs`，**键 = `module_id`** | `web_lab/db.py` `load_pipeline` / `_attach_stage_inputs` |
| fan-out | ✅ 允许（校验器注释原文「分支留路」） | `web_lab/validate_chain.py` `_validate_stage_inputs` |
| fan-in | ❌ 禁止（「每端口单一上游，不支持列表/多值」） | 同上 |
| 重复模块 | ❌ 四处硬禁止 | 见下 |

**重复模块的四道锁**（P2-0 全部解除）：

| # | 锁 | 位置 | 现状表现 |
|---|---|---|---|
| ① | 单插槽 | `composer.py` `Pipeline` 字段 | 每种模块只能有一个实例 |
| ② | 装配取第一个 | `web_lab/server.py` `_chain_from_spec._one()` | 重复阶段**静默丢弃**（比报错更危险） |
| ③ | 校验器报错 | `validate_chain.py` | `重复模块阶段（每模块最多一次）: [...]` |
| ④ | DB 拒绝 | `web_lab/db.py` `save_pipeline` | `链路包含重复模块阶段: ...` |
| ⑤ | 附带 | `server.py` 组装 `stage_inputs[mid]` | 键为 `module_id`，重复即**互相覆盖** |

> 说明：现状「执行顺序硬编码、spec 顺序不驱动执行」这条**从未写入文档**，容易被误解
> （把 `annotate` 写在 `target_crop` 之前，运行时仍是 `target_crop` 先跑）。本稿 §4 明确记录。

## 3. 核心模型

### 3.1 节点身份（`position`）

- 节点实例标识 = **`pipeline_steps.position`**（与 `spec.stages` 数组下标天然对齐；导出/导入稳定、
  可移植）。**不用** `pipeline_steps.id`（`AUTO_INCREMENT`，重建即变，不可移植）。
- **产物全名**：`{module_id}#{position}.{artifact}`
  - 例：`target_crop#3.crops`、`annotate#4.annotated`、`frame_manager#0.frame`。
  - **短名兼容**：`{module_id}.{artifact}` 继续可解析——该模块在本链路**只有一个实例**时等效；
    **多实例时必须写全名**，否则报错并要求限定（**不猜**）。
  - 与既有约定同族：旧名即「省略 `#position`」的短名，迁移直觉自然、短名规则统一。
- **`stage_inputs` 键**：`module_id` → 位置串（`"#3"`）。`Pipeline._port()` 按位置取声明。
- **position 重排规则（必须遵守）**：链路在中间插入/删除阶段会使后续 `position` 全部后移。
  保存链路时**以提交的 `stages` 数组顺序为唯一权威**，在**同一事务内重建 `pipeline_steps`
  与 `pipeline_edges`**（沿用 `db.save_pipeline` 现有的「事务内重建 steps」做法）。
  即 `position` 是**派生索引**，权威是数组顺序；不允许出现「steps 与 edges 的位置不一致」。

### 3.2 端口能力目录（`module_ports`）

```sql
CREATE TABLE IF NOT EXISTS module_ports (
  module_id   VARCHAR(32) NOT NULL,
  direction   ENUM('in','out') NOT NULL,
  port_name   VARCHAR(32) NOT NULL,
  io_type     VARCHAR(16) NOT NULL,          -- Image | Images | Records
  fan_in      TINYINT(1) NOT NULL DEFAULT 0, -- 1=允许多条入边（过渡节点用，P2-3 启用）
  resolved_by ENUM('pipe','driver') NOT NULL DEFAULT 'pipe',
  PRIMARY KEY (module_id, direction, port_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

- **真相源仍是代码**（模块类 `INPUTS` / `OUTPUTS` + `composer.ARTIFACT_TYPES`）；本表是启动时
  upsert 的**可查询副本**，与代码不一致即启动告警（避免双真相源漂移）。
- 本期把 `fan_in` / `Images` / `resolved_by` 三个字段**建好但不用**（普通模块全为 0），
  避免 P2-3 再改表结构。
- `resolved_by='driver'` 目前只有 `vlm.image`（已在 `VLMAnalyzer.INPUTS` 声明，但被校验器
  故意挡着「避免双机制」）→ P2-4 通用化后归零。

### 3.3 连线表（`pipeline_edges`）

```sql
CREATE TABLE IF NOT EXISTS pipeline_edges (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  pipeline_id   VARCHAR(24) NOT NULL,
  dst_position  INT NOT NULL,
  dst_module_id VARCHAR(32) NOT NULL,     -- 冗余：便于审计与兼容旧读法
  dst_port      VARCHAR(32) NOT NULL,
  src_position  INT NULL,                 -- NULL = 帧源/驱动层
  artifact      VARCHAR(96) NOT NULL,     -- 产物全名（含 #position）
  edge_seq      INT NOT NULL DEFAULT 0,   -- 同端口多入边排序（fan_in 端口用）
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_edge (pipeline_id, dst_position, dst_port, src_position, artifact),
  KEY idx_pipe (pipeline_id),
  CONSTRAINT fk_edge_pipe FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

**取舍（已拍定）**：

- 「每端口单一上游」**从 DB 约束挪到写入期校验**（读 `module_ports.fan_in` 判定）。
  MySQL 唯一键无法条件化，硬凹会牺牲可读性；写入期校验还能给人话报错。
  `uk_edge` 只防「完全重复的边」（幂等保存）。
- **只存显式声明**：默认取数规则仍属代码契约，**不落行**（避免双真相源）。
- **读取优先级**：`pipeline_edges > pipelines.meta_json.stage_inputs > 代码默认`。
  兼容期**双写**（`save_pipeline` 同时落表与 meta）；停写 meta 的时机在 P2-4 落地后评估。
- **迁移**：`meta.stage_inputs` → `pipeline_edges` 一次性幂等脚本。旧数据含重复模块
  （本就非法）→ **报错停下要人工确认**，不静默猜。

### 3.4 spec 层接口（跨板块契约，勿改语义）

- **spec 里连线仍内联在 stage 上**（`{"module_id": ..., "params": {...}, "inputs": {端口: 产物全名}}`），
  向后兼容旧 spec；**不新增顶层 `edges[]`**（DB 层独立成行即可，导出/导入仍用内联形状）。
- 各阶段 `inputs` 的**键 = 该阶段在 `stages` 数组中的下标**（即 `position`），
  由 `server._chain_from_spec` 组装进 `Pipeline.stage_inputs`。
- 产物名解析（短名 → 全名、多实例歧义报错）集中在 **`composer`**，`validate_chain` 与
  `server` 都调用同一实现（**不各写一份**）。

## 4. 执行语义（本期刻意不变）

- P2-0 / P2-1 **不改执行顺序**：`Pipeline.step()` 仍按既有硬编码顺序跑；图当前的「顺序」
  只用于校验（引用须指向上游阶段）。
- **明文记录**：`spec.stages` 的书写顺序**不驱动执行**（只是声明清单 + 下标来源）。
  例：`annotate` 写在 `target_crop` 之前，运行时 `target_crop` 仍先跑。
- 声明：P2-2 才做拓扑驱动执行（`position` 从「执行顺序」退化为「节点 ID」，边的方向决定
  顺序，拓扑排序，成环拒绝），届时本节将被改写。

## 5. 兼容与零回归

| 场景 | 行为 |
|---|---|
| 旧 spec（无 `inputs`） | 不报错不警告，逐字零回归 |
| 旧短名产物 | 单实例可解析；**多实例报错**要求写全名 |
| 三条保留链路 | `_SEED_STAGES` 全线性，结构零变更，全绿 |
| `db.save_pipeline` 重复检查 | 由「拒绝重复」→「按 `fan_in` 与端口唯一性校验」（对线性链行为零变化） |
| `meta.stage_inputs` | 保留为回退读，双写至 P2-4 后评估 |

## 6. 影响面清单

| 文件 | 改动 |
|---|---|
| `umvp/pipe/composer.py` | 产物名生成/解析（全名 + 短名）、`_port()` 按位置取声明、`stage_inputs` 键、`artifacts` 表键、`ARTIFACT_*` 常量与短名映射 |
| `web_lab/server.py` | `_chain_from_spec` 的 `by_mod`/`_one`（改为按位置装配多实例）、`stage_inputs` 组装、`CHAIN_RESOLVED_PORTS` |
| `web_lab/validate_chain.py` | `_validate_stage_inputs`（位置化引用、`fan_in` 放行、单上游判定）、重复模块校验改写 |
| `web_lab/db.py` | 两张新表 DDL、`save_pipeline`/`load_pipeline`/`_attach_stage_inputs`、迁移脚本、`configure_modules` 后 upsert 目录 |
| `web_lab/test.html` | 阶段列表显示实例序号（图形画布留 P2-4） |
| `doc/guides/database_schema.md` | 两张新表结构 |
| `doc/modules/*`、`doc/design/assembly_ports.md` | 产物名写法（全名/短名）、连线存法、本稿反向链接 |
| `tests/test_port_inputs.py` | 新增「同模块多实例」断言 |

## 7. 验收标准（本期，可执行断言）

1. **同链路两个 `det_filter`**：各自 `filter` 独立生效（变更前必须先复现「第二个被静默丢弃」）。
2. **两个 `det_filter` 各带 `inputs.dets`** 接不同上游，`stage_inputs` 互不覆盖。
3. `validate_chain` **放行重复模块**，但仍拒绝：成环 / 引用下游或同阶段 / 类型不符 /
   `fan_in=0` 端口接第二条 / 多实例下用短名。
4. **零回归**：三条保留链路 + 全量逻辑测试全绿（qa 出具断言数对照，含
   `test_port_inputs` / `test_validate_chain` / `umvp/pipe/test_composer.py`）。
5. **DB 往返**：迁移后 `load_pipeline` 产出的 `stage_inputs` 与迁移前**逐字节一致**；
   双写期表与 meta 数据一致；`pipeline_edges` 随 `save_pipeline` 事务重建（插删阶段后位置正确）。
6. `python3 .opencode/tool/ticket.py check` OK；`DEPLOYMENT.md` 追加 DB schema 变更条目。

## 8. 分期路线

| 期 | 内容 | 本次 |
|---|---|---|
| **P2-0** | 节点实例身份（产物全名 + `stage_inputs` 键换 position）+ 解四道锁 | ✅ 实施 |
| **P2-1** | `pipeline_edges` + `module_ports` 两表 + 迁移 + 双写 + 写入期校验 | ✅ 实施 |
| P2-2 | 拓扑驱动执行 + 多终段 + `events_by_terminal` | 后续 |
| P2-3 | 过渡节点：`merge`（fan-in 汇聚）/ `join`（异类关联）/ `to_images`（取图，`image_pick` 归宿）/ `split`（条件分流） | 后续 |
| P2-4 | `vlm` 通用化（开 `image` 端口、`image_source` 降级为 UI 快捷预设、`image_pick` 迁出、删驱动层硬编码 join）+ 图编辑画布 | 后续 |

## 9. 已拍定的设计决策（原未决问题）

| # | 问题 | 决策 |
|---|---|---|
| 1 | 产物全名格式 | `{module_id}#{position}.{artifact}`；短名（省略 `#position`）单实例兼容，多实例必须全名 |
| 2 | 引用键 | `position`（spec 数组下标，可移植）；不用 `pipeline_steps.id` |
| 3 | 连线位置 | spec 内联 stage `inputs`；不新增顶层 `edges[]`（DB 层独立成行） |
| 4 | 兼容期 | meta 双写延续至 P2-4 落地后评估停写 |
| 5 | 单上游约束 | 挪到写入期校验（读 `module_ports.fan_in`）；DB 只防重复边 |
| 6 | 预置字段 | `Images` 类型 / `fan_in` / `resolved_by` 本期建好不用，避免二次改表 |

## 10. 新增模块接入成本（架构收益的度量）

**问题**：图编排之后，以后新增一个普通的识别模块或处理图模块，系统还有哪些额外改动？

**答**：分三个阶段看。**本期（P2-0/P2-1）几乎不改变新增模块的成本**——它的收益是「同模块可复用」
与「连线可治理」；**真正的减负发生在 P2-2（拓扑驱动执行）**，那时执行顺序由数据（边）决定，
新增模块不再需要动 `Pipeline.step()`。

### 10.1 今天新增一个模块要动的点（基线）

| # | 改动点 | 类型 | 漏了会怎样 |
|---|---|---|---|
| 1 | 模块类（`INPUTS`/`OUTPUTS` 声明 + `run()`） | 领域代码（必写） | — |
| 2 | `web_lab/server.py` `PARAMS["xxx"]`（参数 schema） | 注册 | 参数面板缺失 |
| 3 | `web_lab/server.py` `MODULES["xxx"]` + `h_xxx` 测试页 | 注册 | UI 看不到模块 |
| 4 | `web_lab/server.py` `CHAIN_ASSEMBLED_MODULES` | 同步 | **静默不执行**（WO-0001 事故类型） |
| 5 | `web_lab/server.py` `_chain_from_spec` 加装配分支 | 系统代码 | 装配失败 |
| 6 | `composer.py` `Pipeline` 加插槽字段 | 系统代码 | 装不进去 |
| 7 | `composer.py` `step()` 插一段调用（硬编码顺序） | 系统代码 | **产物不产生** |
| 8 | `composer.py` `ARTIFACT_*` + `ARTIFACT_TYPES` + `RESOLVED_PORTS` 三处同步 | 同步 | 校验/UI 端口清单不同步 |
| 9 | `db.py` `_ID_PREFIXES`（preset 前缀） | 同步 | 前缀退化为前 4 字母 |
| 10 | `validate_chain.py` `TERMINAL_OK`（若可作终段） | 同步 | 不能作终段 |
| 11 | `server.py` `_CHAIN_DRIVER_PORTS`（若素材/图片走驱动层） | 同步 | 端口不可连线 |
| 12 | 测试 + 文档（`tests/`、`doc/modules/`） | 必写 | — |

> 现状约 **11 处代码/注册点**，其中 **#4 / #7 / #8 属「静默失败型」**（不报错、不生效），
> 是最容易出事故的地方。**好消息**：参数面板与参数表的 DB 列已完全自动
> （`/api/schema` 驱动 UI；`configure_modules` + `_ensure_module_table` 自动补列），
> 这一层新增模块零手工。

### 10.2 本期（P2-0/P2-1）之后

- **成本基本持平**，仅 #5 从「加一段装配分支」变成「注册一行阶段工厂」
  （本期的实例身份改造要求把 `_chain_from_spec` 一次性重构为按 `module_id` 查工厂）。
- 新增收益：新模块**天然可复用多实例**（不再被单插槽限制），无需额外工作。
- 诚实结论：**本期是为「图」付基建费**，别指望新增模块立刻变便宜。

### 10.3 P2-2（拓扑驱动执行）之后

| 原改动点 | 之后 |
|---|---|
| #1 模块类 | **仍需写**（领域代码，永远要） |
| #2/#3 参数与 UI | 由参数 schema 自动（现状已如此） |
| #4 `CHAIN_ASSEMBLED_MODULES` | **消失**：装配能力由注册表派生 |
| #5 `_chain_from_spec` 分支 | **消失**：改为 `STAGE_FACTORIES[module_id]` 注册**一行** |
| #6 插槽字段 | **消失**：统一为 `stages: {position → 实例}` 容器 |
| #7 `step()` 插一段 | **消失**：按图拓扑遍历，执行顺序由边决定 |
| #8 三个产物/端口常量 | **消失**：由 `INPUTS`/`OUTPUTS` 声明**派生**并自动 upsert 到 `module_ports` |
| #9 preset 前缀 | 自动派生（首 4 字母） |
| #10 `TERMINAL_OK` | **消失**：由「模块是否声明终端能力」派生 |
| #11 `_CHAIN_DRIVER_PORTS` | **消失**（P2-4 归零） |
| #12 测试 + 文档 | **仍需写**（永远要） |

→ 收敛为 **① 写模块类 ② 注册一行工厂 ③ 测试 ④ 文档**，且**不再有静默失败型同步点**
（漏声明会直接导致 UI 无端口 / 校验报错，而不是静默不执行）。

### 10.4 「处理图模块」（Image→Image）专项

- **现在**：`annotate` 是唯一一个 Image→Image 模块；再加一个会**互挤**（单插槽），
  且要同步 #6/#7/#8 三处。
- **本期后**：产物自动命名 `xxx#2.image`，两个 Image→Image 模块可并存。
- **P2-2 后**：声明 `INPUTS={"image": "Image"}` / `OUTPUTS={"image": "Image"}`
  即可**任意串联**（如超分 → 标注 → 送审），零核心改动；需要「一组图」语义时用 `Images`
  类型（P2-3 的 `to_images` / `merge` 节点提供转换）。
- 若新模块需要**全新数据类型**（如 `Text`/`Embedding` 之外的第三方类型），需扩展类型枚举
  （一处集中，罕见）。

### 10.5 一句话

> 本期为图付基建费；**P2-2 之后新增普通模块只需「写类 + 注册一行工厂 + 测试 + 文档」**，
> 且系统不再有「漏一处就静默失效」的同步点——这才是图编排对长期演进的主要回报。
