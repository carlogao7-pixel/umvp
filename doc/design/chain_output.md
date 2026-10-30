# 链路尾输出提炼声明：spec 顶层 output 段（设计稿 v2）

> 状态：**已过审（2026-10-27 用户拍板：v2 职责划分「收敛归筛选模块、配图出场归 output」；
> Q1 去重不做、取高分由 `source` 承担；Q2 嵌套点路径支持；Q3 limit=100；图集全输出 `image_pick:"all"`）**
> ｜ 提出方：orchestrator（主单 WO-0029）｜ 承接：pipe-chain（筛选扩展+提炼逻辑）、web（驱动/校验/展示/报告）、qa
> 需求：链路数据线（JSON 主干 / 记录级旁路 / 帧级产物）已通用化，在**链路尾部**声明取哪些信息输出提炼——
> 用于存档 / 汇报 / 报告导出。**范围仅链路尾部**，不做中途节点。关联机制：
> 产物全名与类型 `ARTIFACT_TYPES` / `resolve_artifact`（pipe/composer.py）、旁路读取器 `image_of`/`image_set_of`
> （pipe/records.py）、去重键 `SeqRule.key` 组合/回退（[event_dedup_key.md](event_dedup_key.md)）、
> 派生量 `_field_value`（pipe/conditions.py）、det_filter 事件规格 `detail` 投影（pipe/det_filter.py）、
> `image_pick` first/all 语义（web_lab/server.py `VLM_IMAGE_PICKS`）、报告导出 `_build_report_html`（WO-0028）

## 0. 背景与能力盘点（v2 的依据）

链路内数据流动（两平面 × 三类产物，`doc/modules/INDEX.md`）：JSON 主干＝记录流；记录级旁路
（image 单图/图集 + embedding）；帧级产物（`frame_manager#0.frame`、`annotate#N.annotated`）。

**筛选模块已覆盖的能力（output 不重复做）：**

| 能力 | 现状 | 位置 |
|---|---|---|
| 按字段逻辑关系选记录 | 条件 DSL（派生量 w/h/short/long/area、count 聚合、成组 group_limits） | pipe/conditions.py + det_filter/seq_filter `match` |
| 按 key 去重收敛 | `SeqRule.key` 组合/回退键 + `cooldown`（≥0 抑制 / <0 终身一次）；窗口规则达标只放行一条（边沿），主流已收敛 | pipe/seq_filter.py（WO-0025/26） |
| 窗口确认 | span/window_n/min_n/hit_ratio | pipe/seq_filter.py |
| 字段投影（顶层） | det_filter 事件规格 `detail:[字段名]`（det_filter.py:122） | pipe/det_filter.py |

**真缺口（本设计补）：** ① 窗口内**取最优**作事件源（取高分）；② seq_filter 尾**声明式字段投影**与**嵌套取值**；
③ **配图输出 / 图集全输出**（完全空白）；④ limit / 落盘 / 报告消费。

## 1. 定案原则

- **P1** 「链路输出＝事件」定案不变；output 是**配图与出场层**，不替代事件、不改事件结构。
- **P2** 图不进 JSON：提炼块中图一律为**落盘文件相对路径引用**。
- **P3** 仅链路尾部：声明在 spec 顶层 `output` 段，运行收尾执行；不改 `Pipeline.step` 主干、不加中途模块。
- **P4 职责边界（v2 核心）**：**收敛（去重/选优/投影）归筛选模块**（事件尾）；**output 段只管配图与出场**——
  不做自己的去重、不做自己的字段投影。
- **P5** 语义复用单一来源：产物名 `resolve_artifact`；取张 `image_pick`（first/all，同 `VLM_IMAGE_PICKS`）；
  字段取值 `conditions._field_value`（升级点路径后三处受益）。

## 2. 筛选模块扩展（pipe-chain，两个小扩展 + 一处单一来源升级）

### 2.1 `SeqRule.source`：事件源记录选择（Q1「取高分」落点）

```python
source: str = "first"    # "first"（默认=现状，达标记录即源）| "max:<字段>"（窗口内该数值字段最大者）
```

- 仅**窗口规则**（span/window_n>0）有意义；单条规则只有一条候选，`source` 忽略；
- `max:score` 即「窗口达成时，事件（含 detail/配图溯源）落在窗口内分数最高的那条记录上」；
- 字段经 `_field_value` 取值（数值化比较，同 DSL 数值规整）；缺字段/非数值的候选按缺失处理（不参与 max，全缺失回退 first）；
- **放行语义不变**：仍是边沿放行一条；变的只是**哪条记录承载事件**（放行时替换为窗口内最优者，其余照旧扣留——实现细节以单测断言为准）。

### 2.2 `SeqRule.detail_fields`：声明式字段投影进事件 detail（Q2 落点）

```python
detail_fields: tuple = ()    # 字段名列表；支持嵌套点路径（如 "content.result"）
```

- 达成时把 `{字段: 值}` **追加**进事件 detail（与固定统计 `{key,count,span,window_n}` **并存**，向后兼容——
  PIPE0002 等既有 preset 行为逐字不变）；
- 取值经 `_field_value`（点路径 + 派生量可用）；缺字段跳过该键（宽松，对齐 det_filter detail 语义）；
- det_filter 既有 `detail` 同步受益点路径（同一取值函数）。

### 2.3 `conditions._field_value` 升级：嵌套点路径（单一来源）

- 字段名含 `.` 时逐级下钻（仅 dict/Mapping；中途缺失或非容器 → None）；无点路径的既有行为**逐字不变**；
- 条件 DSL（match/filter）自动受益（`_cond_pass` 走同一取值函数）——**设计上允许 DSL 条件引用嵌套字段**，
  文档同步（`doc/modules/conditions` 相关分册与 det_filter/seq_filter 分册）。

## 3. spec 顶层 `output` 段（配图与出场）

```json
"output": {
  "scope":      "events",
  "from":       "seq_filter#6.dets",
  "image":      "annotate#4.annotated_records",
  "image_pick": "all",
  "limit":      100
}
```

| 字段 | 类型 | 缺省 | 语义 |
|---|---|---|---|
| `scope` | `events`/`records` | `events` | **events**＝只取携带事件的记录（收敛已由事件尾完成）；**records**＝全部放行记录（无事件链路如 PIPE0003 用它提炼图清单） |
| `from` | 产物名 | 末段 Records 产物 | 提炼对象记录流；经 `resolve_artifact` 解析；末段无 Records 产物 → 校验报错要求显式声明（不猜） |
| `image` | 产物名 | 无（不配图） | 配图源：Image 产物（整流一张）或 Records 产物（逐记录读旁路图，`image_of`/`image_set_of`） |
| `image_pick` | `first`/`all` | `all` | 图集取首张 / **全部输出**（**输出侧默认 all**，与送审输入侧默认 first 不同——存档场景宁多勿缺，用户拍板） |
| `limit` | 正整数 | 100 | 提炼条数上限，超限截断并标 `truncated: true` |

## 4. 执行语义

1. **提炼单元**＝记录（事件跟记录走，`get_events` 取出即可）；`scope=events` 过滤掉无事件记录。
2. **配图对位**：`image` 为 Records 产物时按 `frame_idx` + 产物注册顺序对位到主流记录；对不上或记录无图 → 该条 `image` 为空列表；`image` 为 Image 产物（帧级图）→ 同帧提炼记录共用该图（各自落盘引用）。
3. **图集全输出**：记录旁路为图集时，`image_pick:"all"` 逐张落盘，提炼项 `image` 为**路径列表**；`first` 取首张。统一列表语义（单图 → 长度 1 列表）。
4. **收敛顺序**：scope 过滤 → 配图 → limit 截断。**不做 key 去重**（P4：收敛是事件尾的职责）。
5. **时序无关**：运行收尾一次执行，不影响逐帧处理与背压。

## 5. 输出形态（提炼块）

`h_pipeline_run` 返回新增 `output` 键；SSE 流式在 `done` 事件附带同结构：

```json
"output": {
  "scope": "events", "from": "seq_filter#6.dets", "count": 2, "truncated": false,
  "items": [
    {"ts": 12.34, "frame_idx": 310, "label": "person", "track_id": 7, "score": 0.91,
     "events": [{"name": "警情", "ts": 12.3, "detail": {"key": "7", "count": 3, "span": 5.0,
                                                          "result": "斗殴", "confidence": 0.9}}],
     "image": ["tests/data/out/test_pipeline/output_001_tid7_12.3s.jpg"]}
  ]
}
```

- 通用字段（ts/frame_idx/label/track_id/score）自动携带（存在才带）；`events` 为该记录事件全明细
  （`detail_fields` 声明的投影字段已在 detail 内——字段声明统一在事件尾规则，单一来源）；
- **test.html**：结果区新增「链路尾输出」区块（字段表 + 缩略图）；未声明 `output` 不显示（向后兼容，静默）；
- **报告导出（WO-0028）**：报告新增「链路尾输出」区块且**优先呈现**（有声明按声明取数，无声明维持现状取数）。

## 6. 落盘

- 提炼图落 `output_dir`，命名 `output_{序号:03d}_{tid或key}_{ts:.1f}s_{图序}.{ext}`；
- **共用** `max_saved_images` 上限计数与 `saved_files` 清单；图源已落盘的场景（如 plate_*）不重复落盘、直接引用（实现取简单者并记录）。

## 7. 校验（validate_chain）

| 项 | 规则 |
|---|---|
| `output` 段缺失 | 合法（行为与现状逐字一致） |
| `from` / `image` | 产物名解析（复用 `resolve_artifact`，歧义/未知报错）；`from` 类型 ∈ {Image, Records}（Image 时 scope 强制 records 单条语义——Image 无事件无主干，提炼块即一张图）；`image` 类型 ∈ {Image, Records} |
| `scope` / `image_pick` | ∈ {events, records} / ∈ {first, all} |
| `limit` | 正整数 |
| `SeqRule.source` | `first` 或 `max:<字段名>` |
| `SeqRule.detail_fields` | 字符串列表 |
| `from` 缺省不可解析 | 报错要求显式声明 |

## 8. 实现拆分（子单规划）

| 板块 | 内容 |
|---|---|
| pipe-chain（WO-0029.1） | ① `conditions._field_value` 点路径；② `SeqRule.source` + `SeqRule.detail_fields`；③ `pipe/chain_output.py` 提炼纯逻辑（零 IO：scope 过滤/配图对位/图集 first-all/limit，图以 `(记录, 图列表)` 对返回，落盘归驱动层）；④ 单测（conditions 点路径、seq_filter source/detail_fields、chain_output 投影配图截断） |
| web（WO-0029.2） | ① server.py 运行收尾调提炼 → 落盘 → 结果附 `output` 块（同步 + SSE done 两路径）；② validate_chain §7 规则；③ test.html 展示区块；④ 报告导出消费 |
| qa（WO-0029.3） | 三条保留链路各配 output 声明端到端断言（字段/图落盘/图集全出/截断/向后兼容）；既有回归 |

## 9. 非目标（本期不做）

- 中途取图 / 图编排过渡节点（`to_images`/`merge`/`join`/`split` 仍归 graph_chain P2-3）；
- 提炼层自身去重/聚合（P4：归事件尾；若未来需要「同 key 聚合多事件」，走 seq_filter 扩展而非 output）；
- 向量旁路提炼（embedding 不出）；提炼块 DB 持久化（首版只进运行结果与报告文件）；
- 事件结构变更（P1）。

## 10. 拍板记录（2026-10-27，用户）

1. **Q1 去重**：output 不做去重——收敛是筛选模块职责；「取高分」→ `SeqRule.source: max:<字段>`；「通用字段逻辑关系」→ 既有条件 DSL。
2. **Q2 嵌套**：支持——`_field_value` 点路径升级，`detail_fields`/det_filter detail/DSL 三处受益。
3. **Q3 limit**：默认 100，定。
4. **图集**：`image_pick:"all"` 全输出（输出侧默认 all）。
5. **职责划分 v2**：「收敛归筛选模块、配图出场归 output」通过。
