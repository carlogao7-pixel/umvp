# 时序判定 SeqFilter

> 模块组：管道通用模块 | 代码：`umvp/pipe/seq_filter.py`（`SeqFilter` / `SeqRule`；条件引擎
> `umvp/pipe/conditions.py`，记录适配 `umvp/pipe/records.py`）
> 数据库参数：`preset_params_seq_filter`（`SEQ0001+`：SEQ0001 警用无人机 / SEQ0002 车牌识别）
> 状态：门控承接原 Pipeline 的 `check_interval`/`track_change_only`；三条保留链路以其 `rules` 作事件尾。

## 1. 模块定位

通用下游**时序判定器**：对有序记录流做三类事——**门控**（整类节流 / track 集合
变化）、**规则确认**（单条命中即放行 / 窗口凑满次数才放行）、**事件标注**（事件作为
字段随证据记录走，不造新对象）。与「检测筛选 DetFilter」（有效性、逐条+成组）互补：

```
上游记录流（YOLO / 车牌 / VLM 结论…）
   │
   ▼
[检测筛选 DetFilter]   有效性：单条条件 + 成组数量（对象/dict 透传）
   │
   ▼
[时序判定 SeqFilter]   本模块：门控（节流/变化）+ 规则确认 + 事件标注
   │
   ▼
裁剪 / 标注 / VLM / 输出物化（下游可按 event 字段筛取）
```

### 1.1 门控（2026-10 自 Pipeline 迁入，与规则可并用）

| 门控 | 参数 | 语义 | 原归属 |
|---|---|---|---|
| 整类节流 | `throttle={"key":"cls_id","interval":3.0}` | 同一 key 每隔 interval 秒才放行**整类**记录，其间扣留 | Pipeline.check_interval |
| 变化门控 | `change={"key":"cls_id","watch":"track_id"}` | 同类 watch 值集合相比上次有变化才放行整类（空集不抑制） | Pipeline.track_change_only |

执行顺序：**throttle → change → 规则**；无规则（纯门控）时门控存活者全部放行。

### 1.2 规则（确认闸 + 事件标注）

- **单条规则**：记录命中 `match` 条件 → 该记录放行并标注事件（电平：每条命中都标）。
- **窗口规则**：同一 `key` 分组内，`span` 秒窗口里命中 `match` ≥ `min_n` 次 →
  第 n 条命中记录放行并标注事件（"m 秒内出现 n 次"；未确认的中间命中扣留）。

## 2. 参数

规则 `SeqRule`（列表传入 `SeqFilter(rules=[...])`）：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| match | str | 必填 | 单条条件 DSL（语法/字段同检测筛选，含派生 w/h/short/long/area 与注入的 ts/frame_idx）；非法表达式装配期报错 |
| event | str | 必填 | 事件名：达成时追加到记录的 `events`；支持 `{字段}` 占位（**动态事件名**，如 "{label}"、"警情-{cls_id}"） |
| key | str | track_id | 分组键（窗口计数、冷却抑制均按它分组；值缺失/None/0 归 `"*"` 组——无跟踪的记录同一窗内一起计数） |
| span | float | 0.0 | 时间窗秒数（与 window_n 二选一；两者皆 0=单条语义） |
| window_n | int | 0 | 计数窗（最近 N 条到达；与 span 二选一） |
| min_n | int | 1 | 窗口内命中 ≥ n 才达成（min_n>1 需 span>0 或 window_n>0） |
| hit_ratio | float | 0.0 | 窗口命中比例阈值（0=关）；>0 时需求 = max(min_n, ceil(分母×比例))，计数窗需先凑满 window_n |
| cooldown | float | 0.0 | 达成后抑制秒；**<0 = 同 key 只触发一次**（不重报，等价旧 inspection 同对象一次）；默认 0=窗口重臂即可再报 |

> 需求判定：`min_n` 为下限；`hit_ratio` 叠加比例需求（计数窗分母=window_n、时间窗分母=窗内到达数）。
> `rules` 可为空（纯门控模式），但 `rules`/`throttle`/`change` 至少给一个，否则构造期报错。

### 事件标注（记录上追加 `events` 列表，支持多个事件）

```
events = [{"name": 事件名, "ts": 达成时刻, "detail": {"key","count","span"}}, ...]
```

- **每条匹配并达成的规则各追加一个事件**；一条记录可命中多条规则 → 多个事件。
- dict 输入原地 append `events`；对象输入挂 `events` 属性（事件随证据元素走，不换类型）。
- 通用事件约定（`records.add_event`/`get_events`/`collect_events`）——不绑定具体模块。

### 语义细则

- **规则各自独立**：每条规则独立判定；任一规则达成则该记录放行并追加其事件；
  未达成（窗口未凑满/冷却期内）仅作该规则证据、不追加事件、不影响其它规则。
- **窗口重臂**：达标后该 key 窗口立即清空重新计数（防帧帧重发）；`cooldown`
  在此之上额外按 key 抑制一段时间。
- **时间取值**：优先记录自带 `ts`（Record 约定），缺省用 `run()` 的 ts。
- **输出保持原类型**：对象进对象出、dict 进 dict 出（与检测筛选一致）。事件标注：
  dict 输入原地 append `events`；对象输入挂 `events` 属性。
- 门控存活者/规则达成者放行；其余（未命中规则、窗口未凑满、冷却期内、门控扣留）扣留。

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| items | list | 有序记录（同帧或跨帧均可）：Detection / dict(Record) / 其它可适配对象 |
| ts / frame_idx | float / int | 时间上下文（记录自带 ts 时以记录为准） |

## 4. 输出（字段和内容）

放行的**原元素**列表：无规则（纯门控）→ 门控存活者；有规则 → 命中单条规则的
记录、或凑满窗口的第 n 条记录（均带 `events` 列表）；其余扣留。

## 5. 上下游衔接

- 上游：任意产出记录列表的模块；链路上紧随 DetFilter（先有效性、后门控/确认）。
- 下游：裁剪/标注/VLM 报送/事件尾——对象透传，下游无感。
- 作为事件尾：外部记录（VLM 结果/车牌）经 `Pipeline.publish()` 走本模块规则出事件；
  链路上本模块用 `rules` 作事件尾、`throttle`/`change` 作检测流门控（一个阶段两用）。
  三条保留链路均以本模块收尾（PIPE0001 词表+冷却 30s、PIPE0002 同车牌一次）。

## 6. 验证

```bash
conda run -n ai python tests/test_seq_filter.py   # 本模块断言（含门控；纯逻辑，秒级）
conda run -n ai python tests/test_det_filter.py   # 姊妹模块回归（共用 conditions/records）
```
（断言数不写死，以脚本运行自报为准。）

## 7. 调参速查（通俗指南）

- **闪断误报多（YOLO 单帧噪声）** → 窗口规则：`span=5, min_n=3`（5 秒见到
  3 次才放行，等价滑窗确认）。
- **VLM 抖动** → key=label，match=词表命中，span/min_n 做连续确认。
- **驻留语义** → key=track_id，match=存在条件（如 `cls_id == 0`），span=驻留
  秒数、min_n≈span×帧率。
- **只想打标不想确认** → 单条规则（span=0），每条命中都带事件字段。
- **同一事件别刷屏** → 配 cooldown（按 key 抑制）；同对象只出一次用 `cooldown<0`。
- 车牌"连续读到才采信"：key=label（车牌号），match=`kind == plate`，
  span=3、min_n=2。

## 8. 边界与路线

**不做**（边界）：不修改事件字段以外的内容；不做"连续 n 次无中断"语义（时间窗已覆盖
主要场景，需要时加 `consecutive: true`）；不做跨帧聚合以外的统计（均值/求和等，出现真实
需求再议）；不构造独立告警对象（本模块产出 `events` 字段即输出）。

**与检测筛选的分工（分层不重复，通用化方向见 [检测筛选](det_filter.md) §8）**：
单帧/批内聚合（count、成组数量判定）归 DetFilter；跨帧时序（本模块：时间窗、连续 N 次、
驻留、track 变化、事件标注）。两者共用 `conditions.py` 条件引擎，登记"时间窗/连续/跨帧"
类能力时只在本模块加，不塞进 DetFilter。
