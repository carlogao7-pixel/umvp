# 门控阶段模块 GateFilter（设计稿）

> 状态：**已落地（2026-10-26，WO-0023 全链闭单）；PIPE0002 灰度对跑通过（INB-0091）后已转正为
> 默认链路形态（2026-10-27，WO-0024，用户拍板）** ｜ 提出方：orchestrator（主单 WO-0023）｜ 承接：pipe-chain
> 背景诊断：INB-0089（车牌链路 track 门控缺口）；关联机制：`IntervalGate`（pipe/composer.py）、
> `SeqFilter.throttle`（pipe/seq_filter.py）、`PlateStage.track_dedup`（plate_recog/plate_pipeline.py）
> 实现/注册/验证见 WO-0023.2（pipe）/.3（web）/.4-.5（qa）；灰度对比数字见 DEPLOYMENT.md 2026-10-26 台账。

## 0. 背景与动机

「按 key + 时间间隔放行」的门控语义目前散落三处、双轨并存：

| # | 位置 | 语义 | 现状问题 |
|---|---|---|---|
| 1 | `PlateStage` 内置 `IntervalGate`（plate_pipeline.py:84-88） | key=track_id，`track_cooldown>0` 周期冷却 / `≤0` 同 track 一次 | **tid=0 绕过**：门控条件 `tid > 0`，track 建立初期/闪断帧（`boxes.id` 未分配）完全绕过门控，同车 0.3s 内重复识别（INB-0089 实测） |
| 2 | Pipeline 编排层 `check_interval` 报送节流（现由 `_seq_filter_of_spec` 译成 `SeqFilter.throttle`） | key=cls_id 整类节流 | 已迁 seq_filter；与 #3 双轨（throttle 自实现 allow/note 等价逻辑） |
| 3 | `SeqFilter._throttle_gate` | key 任意，整类节流 | 与 `IntervalGate` 语义等价但各自实现；且 0/None/缺 key 归 `"*"` 组的规则与 #1 的"tid=0 直接绕过"相悖 |

收敛目标：把这套语义抽成一个**独立、可组合的链路阶段模块**（module_id `gate_filter`），
修掉 tid=0 缺口（匿名桶），拦截可观测（统计旁路），并给三处既有机制的收编路线。

## 1. 模块接口

**文件**：`umvp/pipe/gate_filter.py` ｜ **类名**：`GateFilter` ｜ **module_id**：`gate_filter`
（纯逻辑，不加载模型、不依赖 cv2/onnx——只消费记录 dict）。

### 1.1 端口声明（链路壳标准 §3，鸭子类型对齐 SeqFilter/DetFilter）

```python
class GateFilter:
    # 吃记录列表，放行子集仍是记录（只取舍，不改内容、不加字段、不加事件）
    INPUTS = {"dets": "Records"}
    OUTPUTS = {"dets": "Records"}   # 另登记观测产物 stats（§3，非连线端口）
```

### 1.2 构造参数

```python
GateFilter(key="track_id", interval=5.0, filter="")
```

| 参数 | 类型/默认 | 语义 |
|---|---|---|
| `key` | str，默认 `"track_id"` | **分组键字段名**（记录字段，如 `track_id` / `cls_id` / `label`）；取值经 `conditions._field_value`（dict/对象统一，派生字段 `w/h/short/long/area` 亦可用）。空串构造期抛 `ValueError` |
| `interval` | float，默认 `5.0` | **三值语义，与 `IntervalGate` 完全对齐**：`>0` 同 key 每 interval 秒最多放行一次；`<0` 同 key **只放行一次**（对齐 PlateStage `track_cooldown≤0` → `IntervalGate(-1)`）；`=0` 全放行（no-op） |
| `filter` | str，默认 `""`（**可选，v1 是否带上见 §9-Q1**） | 条件 DSL（与 det_filter/target_crop/annotate 同族）：只对**命中记录**做门控，未命中记录直接放行。空 = 全部记录受门控。非法表达式构造期抛错（`parse_condition` 快速失败） |

内部状态：`self._last: Dict[key值, 上次放行 ts]`（即 `IntervalGate._last` 的等价物）+
`self.last_stats`（§3）。提供 `reset()` 清状态（新视频流复用实例时用；驱动层接线为可选交接项）。

### 1.3 run 签名（与 SeqFilter/DetFilter 同形）

```python
def run(self, dets, ts: float = 0.0, frame_idx: int = 0) -> List:
    """返回放行的**原元素**（dict 进 dict 出、对象进对象出，不拷贝不包裹）。"""
```

- **时间源**：记录自带 `ts` 优先、缺省用 run 注入的 `ts`（同 SeqFilter；`Pipeline.step` 已统一注入，
  两者实际相等）。
- **判定**：`allow = (interval == 0) or (key 首见) or (interval > 0 and rts - last >= interval)`；
  放行即 `note`（更新 `last`）。**同帧同 key 多条**：同帧同 key 的记录要么全放行要么全拦
  （第一条放行后其余命中同一冷却窗口，与 IntervalGate 周期刷新语义一致——一帧内多车同 tid 不存在，
  匿名桶同帧多条取第一条、其余拦，防"匿名桶每帧只进一条"被同帧多条绕过）。
- **顺序稳定**：输出保持输入相对顺序（逐条判定），不重排。

## 2. key 来源与匿名桶（tid=0 缺口的修法）

**key 值归一（铁律，构造期定死、无配置项）**：

```
key 值为 None / "" / 0  →  归匿名桶 "*"
其余（含负数、任意非零值）→  str(值) 各自成桶
```

- **匿名桶统一受门控**：所有"没有 key"的记录（含 `track_id=0` 的 track 未建立帧）共用 `"*"`
  一个桶，按同一 `interval` 冷却——**首条放行、interval 内的后续全拦**。这直接修掉 INB-0089
  缺口①：不再是 PlateStage 式"`tid=0` 绕过门控每帧放行"，而是匿名记录也按节奏冷却放行。
- 与 `SeqFilter._key_of` 的 `"*"` 规则一致（None/""/0 → "*"），两模块对"缺 key"语义自此同律。
- **注意（写入模块分册注意事项）**：若 `key="cls_id"`，COCO 类 0（person）也会落匿名桶——与
  SeqFilter 现行为一致；类级门控建议用 `key="cls_id"` 时接受该语义（匿名桶节奏≈类 0 的节流）或
  改用 `label`。track 级门控（主用例）用 `track_id`，0 恰是"无 track"，语义正合适。
- **首见 key 放行**（`last` 无记录 → True），与 IntervalGate 一致；首次出现不拦。

## 3. 语义铁律与可观测性（统计旁路）

**铁律（用户拍板）**：主路**物理过滤**——被拦记录不传下游、**不加任何标记字段**（不写
`blocked`/`gated` 之类键，返回记录的键集合与输入逐键相同）。拦截的可观测全部走**统计旁路**：

### 3.1 统计产物叫什么、登记在哪

| 层 | 名称 | 形态 | 说明 |
|---|---|---|---|
| 模块属性 | `GateFilter.last_stats` | dict：`{"in": 进入数, "passed": 放行数, "blocked": 拦截数, "blocked_detail": {key值: 条数}}` | 每次 `run` 覆盖更新；`blocked_detail` 按归一化 key（含 `"*"`）分键计数，诊断/时间线直接读 |
| Pipeline 产物 | `gate_filter.dets` | Records（放行子集） | 经 `_register` 登记：多实例全名 `gate_filter#<pos>.dets`、单实例另登记短名别名（现机制零新增） |
| Pipeline 产物 | `gate_filter.stats` | **Records**：单元素列表 `[{"kind": "gate_stats", "ts", "frame_idx", "key", "interval", "in", "passed", "blocked", "blocked_detail"}]` | 同上按位置登记；类型登记进 `ARTIFACT_TYPES`（Records）。**不在** `RESOLVED_PORTS`（无下游端口消费它，观测专用） |
| Pipeline 字段 | `Pipeline.last_gate` | `Optional[Tuple[int, int]]` = (进入, 放行) | **参照 `last_filter=(进入,放行)` 同型字段**：首个门控实例的进入数、全链门控后的放行数；本帧未运行（空 dets）= None。多实例语义与 last_filter 对齐 |

### 3.2 驱动层怎么读

- **时间线/叠加（web）**：与读 `last_filter` 同路——直接读 `pipe.last_gate` 画"放行/拦截"计数；
  要分键明细读 `pipe.artifacts["gate_filter.stats"]`（`blocked_detail` 里 `{"*": 12}` 一眼看出
  匿名拦截量）。web 页面消费为交接项（§8）。
- **诊断脚本**：直接持有 GateFilter 实例读 `last_stats`（灰度对跑 §6 用）。
- **拦截条目 timeline 可见** = stats 的分键计数（不落被拦记录本体——被拦记录已按铁律丢弃，
  不另存样本；如后续要"被拦样本预览"再议，本期不做）。

### 3.3 在 Pipeline 中的执行位置

固定插槽顺序：`det_filter → seq_filter（既有门控）→ gate_filter（新增）→ target_crop → …`。
理由：gate_filter 是**最细粒度的最后一道闸**（逐 key 冷却），紧邻下游消费（裁剪/标注/报送）；
拦截对**所有下游一致生效**（裁剪/标注/VLM 报送/事件尾，与 det_filter 哲学相同）。多实例与
det_filter 同规：按位置升序链式执行、产物全名互不覆盖；`last_gate` 取（首实例进入，链末放行）。

## 4. 与既有机制的关系（本轮边界）与收编路线

**本轮不动**（边界）：

- `SeqFilter` 一行不改（其 throttle/change 语义保持；`_throttle_gate` 与 GateFilter 的双轨
  在本轮容忍）。
- `PlateStage` 内置 `track_dedup` **机制保留不删**（不收编、不删除）；WO-0024 起 PIPE0002 默认
  已改走独立门控（下条路线 1 已执行），`track_dedup=true` 为回切开关。
- Pipeline 的 `check_interval` 旧字段翻译（`_seq_filter_of_spec`）不迁移。
- `IntervalGate` 工具类保留原位（PlateStage 继续用它；composer 自测引用它）。

**未来收编路线（只记录，不实现）**：

1. **近期**（**已执行**：2026-10-27 WO-0024，灰度 B 版转正为 PIPE0002 默认，用户拍板）：
   PIPE0002 默认 spec = `plate_recog.track_dedup=false` + 插入 `gate_filter` 阶段
   （key=track_id, interval=5.0，插 det_filter 与 target_crop 之间；§5、§6）。
   灰度对跑通过依据见 INB-0091。
2. **中期**：`SeqFilter._throttle_gate` 内部改调 GateFilter（或提取公共 allow/note 实现），
   消灭双轨；`throttle` 参数保持不变（对外零感知）。
3. **远期**：PlateStage 删除内置门控（gate_filter 成为唯一 track 门控）；`check_interval` 的
   翻译目标不变（已是 seq_filter.throttle，即收编路线 2 的受益者）。

## 5. spec 声明与装配

### 5.1 stages spec 声明

```json
{"module_id": "gate_filter", "params": {"key": "track_id", "interval": 5.0, "filter": ""}}
```

- 参数白名单以 `list_modules` 现取为准（web 注册后自动可查）。
- `gate_filter` **不是终段**（终段白名单 det_filter/seq_filter/plate_recog/face_detect/face_embed
  不含它，零改动）；每链路可多实例（同模块多实例机制现成）。

PIPE0002 换装后链路片段（原灰度变体 spec，WO-0024 起为默认形态；完整 7 步见
[chains/pipe0002_plate.md](../chains/pipe0002_plate.md) §2）：

```json
{"module_id": "det_filter",   "params": {"...": "..."}},
{"module_id": "gate_filter",  "params": {"key": "track_id", "interval": 5.0}},
{"module_id": "target_crop",  "params": {"classes": "2"}},
{"module_id": "plate_recog",  "params": {"mode": "crop", "track_dedup": false, "track_cooldown": 5.0}}
```

### 5.2 装配（`_chain_from_spec` / Pipeline）

| 改动点 | 内容 | 归属 |
|---|---|---|
| `umvp/pipe/gate_filter.py` | 新模块 | pipe-chain |
| `composer.Pipeline` | 新单槽字段 `gate_filter` + 多实例列表 `gate_filters`；`_stage_instances` pairs 表、`_CLASS_TO_MODULE`、"GateFilter": "gate_filter" 注册 | pipe-chain |
| `composer.step()` | seq_filter 块之后插 gate_filter 执行块（链式 + `_register("gate_filter", pos, "dets"/"stats", …)` + `last_gate`） | pipe-chain |
| `ARTIFACT_TYPES` | += `gate_filter.dets`(Records)、`gate_filter.stats`(Records) | pipe-chain |
| `RESOLVED_PORTS` | += `"gate_filter": ("dets",)` | pipe-chain |
| **`Pipeline.track_needed()`** | **扩展（关键联动，§5.3）**：存在 `key=="track_id"` 的 gate_filter 实例时也返回 True | pipe-chain |
| `web_lab/server.py _build_stage_instance` | `mid == "gate_filter"` 分支：`GateFilter(key=…, interval=…, filter=…)` | **@web** |
| `web_lab/server.py CHAIN_ASSEMBLED_MODULES` | += `"gate_filter"`（「声明即执行」断言依据） | **@web** |
| `list_modules` PARAMS / 设计器「gates」页签 / `validate_chain` 参数白名单 | 注册三参数 | **@web** |
| `_seed_pipeline` 播种灰度 spec（可选） | 另存 PIPE0002 变体或文档示例 | **@web** |

### 5.3 validate_chain「声明即执行」是否现成？——**现成，web 注册即生效**

- 断言机制读 `server.CHAIN_ASSEMBLED_MODULES`（动态导入），**`validate_chain.py` 本体零改动**；
  web 把 `gate_filter` 加进该集合后，spec 声明而未装配的旧报错路径自动覆盖新模块。
- 参数校验走 `list_modules` 白名单（同上，注册即校验）。
- 终段白名单、每模块至多一次、frame_manager 首位等硬性规则对 gate_filter 天然适用，无特例。

### 5.4 关键联动坑：track_needed（设计结论）

`track_id` 的存在前提是 YOLO 开跟踪（`yolo.infer(track=True)`），而开关由
`Pipeline.track_needed()` 决定——目前只看 `plate.use_track`。灰度变体把
`plate_recog.track_dedup` 关掉后若不处理，**track_needed 变 False → 全链无 track_id →
所有记录落匿名桶**，门控退化成全局限流（比现状更差）。

**结论**：`track_needed()` 增加一条——任一 gate_filter 实例 `key == "track_id"`（模块暴露
`needs_track` 属性）→ 返回 True。内置去重与独立门控的组合（都开）也合法：门控先于裁剪，
PlateStage 的二次门控自然短路（同 key 刚放行过，`crops` 到 plate 前已被拦），行为=取更严者。

## 6. 灰度验证方案（PIPE0002 对跑）

**工具**：`/tmp/opencode/diag_plate_chain.py`（已确认存在，真实素材
`tests/data/vid/车牌识别.MP4`）；基线 = INB-0089：内置版（cooldown=5s）同车识别间隔
2.84s→7.84s→13.01s（恰为 cooldown 节奏，但 tid=0 帧绕过——f75 tid=0 与 f85 tid=2 相隔 0.3s
各识别一次）、7 条事件 / 2~3 辆车、tid=2 跨 510 帧稳定。

| 跑法 | 配置 | 目的 |
|---|---|---|
| A（基线） | 现状 PIPE0002：`track_dedup=true, cooldown=5` | 对照组（复测基线数字） |
| B（灰度） | `track_dedup=false` + `gate_filter(key=track_id, interval=5)`（插 det_filter 与 target_crop 之间） | 实验组 |
| C（叠加，可选） | 两者同开 | 验证组合合法性与"取更严者"（§5.4） |

**验收点**（B 对 A）：

1. **同车识别频次不高于内置版**：每 track 识别次数 B ≤ A（总识别次数、每 tid 计数对齐）。
2. **tid=0 帧不再每帧放行**：匿名桶放行次数 ≈ 时长/interval 量级（5s 一条）而非逐帧；
   `blocked_detail["*"]` 显著大于放行数。
3. **拦截统计可见**：`last_gate` / `gate_filter.stats` 非零且分键计数与手工对帧一致。
4. **端到端不劣化**：车牌事件条数 ≤ A（同车异号重复事件随识别频次下降而下降——事件尾聚合
   是另一问题，INB-0089 缺口②，不在本单）；识别结果本身（车牌号/分数）与 A 一致或更少。

真推理跑（YOLO+HyperLPR）较慢，属「待测试」阶段执行；设计阶段仅产出方案与本节验收清单。

## 7. 测试计划

### 7.1 `tests/test_gate_filter.py`（纯逻辑，秒级，不加载模型）

| # | 用例 | 断言要点 |
|---|---|---|
| 1 | 基本节流 | 同 key：interval 内第 2 条拦、`ts-last >= interval` 边界放行 |
| 2 | 多 key 独立 | 不同 key 各自冷却互不影响 |
| 3 | **匿名桶** | tid=0 / None / 字段缺失 → 同落 `"*"`，窗口内只放 1 条；与实名桶互不干扰 |
| 4 | interval<0 | 同 key 只放行一次（对齐 PlateStage `track_cooldown≤0`） |
| 5 | interval=0 | 全放行（no-op，对齐 IntervalGate） |
| 6 | **物理过滤铁律** | 被拦记录不返回；返回元素 `is` 输入元素；**返回记录键集合 == 输入键集合**（无标记字段） |
| 7 | 首见放行 + 时间源 | key 首见放行；记录自带 ts 优先于 run ts |
| 8 | 双形态 | Detection 对象进对象出；dict 进 dict 出 |
| 9 | stats | `last_stats` 的 in/passed/blocked 与 blocked_detail 分键计数正确 |
| 10 | reset() | 清状态后重新首见放行 |
| 11 | 非法构造 | key 空串抛 ValueError；（若带 filter）非法 DSL 构造期抛错 |
| 12 | Pipeline 集成 | gate_filter 阶段插链：被拦记录不达 target_crop（crops 为空/减少）、`last_gate=(进入,放行)`、artifacts 有 `gate_filter.dets`/`gate_filter.stats`、track_needed 联动（§5.4） |

### 7.2 既有回归（改动关联必跑）

```
conda run -n ai python umvp/pipe/test_composer.py        # Pipeline 插槽/artifacts/track_needed 改动
conda run -n ai python tests/test_gate_filter.py         # 新增
conda run -n ai python tests/test_plate_recog_pipeline.py
conda run -n ai python tests/test_seq_filter.py          # seq_filter 不动，防回归
conda run -n ai python tests/test_validate_chain.py
conda run -n ai python tests/test_target_crop.py         # 上游被 gate 截断后 crops=[] 行为
conda run -n ai python web_lab/validate_chain.py <灰度spec>   # web 注册后（交接项完成时）
```

## 8. 交接清单（@web，实现阶段完成后派发）

1. `CHAIN_ASSEMBLED_MODULES` += `gate_filter`；`_build_stage_instance` 加分支。
2. `list_modules` PARAMS 注册（key/interval/filter 三参数 + 默认值）→ 设计器「gates」分组
   出现新节点；`validate_chain` 参数白名单自动生效。
3. 时间线/叠加消费 `pipe.last_gate`（放行/拦截计数）与 `gate_filter.stats`（分键明细，
   匿名桶占比）——具体 UI 形态 web 定。
4. （可选）`_seed_pipeline` 播种 PIPE0002 灰度变体；GateFilter.reset() 接入换流重启路径。

## 9. 待确认问题

| # | 问题 | 倾向 |
|---|---|---|
| Q1 | v1 是否带 `filter`（条件 DSL 限定门控范围）参数？ | **带**——与 det_filter/target_crop/annotate 参数族一致、成本近零；不要则删（接口其余不变） |
| Q2 | 匿名桶语义 v1 定死无配置项（不可配"缺 key 直放"）——确认？ | **定死**。"缺 key 直放"即 INB-0089 缺口①的现状，不该成为可选项 |
| Q3 | `gate_filter.stats` 是否需要被拦记录**本体**留存（样本预览）？ | 本期**不留**（铁律+最小面）；需要时二期加 `last_blocked` 上限 N 条环形缓冲 |
| Q4 | 灰度变体是否播种进 DB（PIPE0002 变体 preset）？ | 倾向先文档示例、设计器手工组链验证，播种等灰度通过后（→ **已收口**：灰度通过，WO-0024 起播种转正为 PIPE0002 默认 7 步） |
