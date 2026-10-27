# 门控 GateFilter

> 模块组：管道通用模块（主干筛选）｜ 代码：`umvp/pipe/gate_filter.py`（`GateFilter`；
> 条件引擎 `umvp/pipe/conditions.py`，间隔语义复用 `pipe.composer.IntervalGate`）
> 数据库参数：preset「车牌识别·门控」（PIPE0002 默认链路引用，WO-0024 起播种；自定义链路
> spec 声明 `gate_filter` 阶段即用）
> 状态：2026-10 WO-0023 新增（设计稿 `doc/design/gate_filter.md`，缺口诊断 INB-0089）；
> 2026-10-27 WO-0024 转正为 PIPE0002 默认链路形态（灰度对跑通过，INB-0091）

## 1. 模块定位

通用下游**门控器**：「按 key + 时间间隔放行」的唯一链路阶段实现（收敛自三处散落语义：
`PlateStage` 内置 track 门控 / Pipeline `check_interval` 报送节流 / `SeqFilter.throttle`）。
主路**物理过滤**——被拦记录不传下游、不加任何标记字段；拦截可观测走**统计旁路**。

```
上游记录流（yolo → det_filter → seq_filter 之后）
   │
   ▼
[门控 GateFilter]   同 key 每 interval 秒最多放行一次（匿名桶统一受门控）
   │                主路：物理过滤（被拦就地丢弃，零标记字段）
   ▼                旁路：last_stats + 产物 gate_filter.stats + Pipeline.last_gate
裁剪 / 标注 / 报送 / 事件尾（拦截对所有下游一致生效）
```

- **与检测筛选 DetFilter 互补**：DetFilter 管"这条记录有没有效"（逐条+成组，帧内判定）；
  GateFilter 管"这个 key 最近放行过没有"（跨帧节奏）。
- **与时序判定 SeqFilter 互补**：SeqFilter 的 throttle/change 是**整类**门控（key=cls_id，
  一放全放）；GateFilter 是**逐 key** 冷却（key=track_id，每个目标各自计节奏），粒度更细、
  是检测流**最后一道闸**（固定插槽在 seq_filter 之后、target_crop 之前）。

### 1.1 三值语义（与 IntervalGate 完全对齐，唯一实现复用）

| interval | 语义 |
|---|---|
| `> 0` | 同 key 每 interval 秒最多放行一次（首见放行，放行即刷新） |
| `< 0` | 同 key **只放行一次**（此后不再放行；对齐 PlateStage `track_cooldown≤0`） |
| `= 0` | 全放行（no-op，不记状态） |

### 1.2 key 归一与匿名桶（铁律，定死无配置项）

```
key 值 None / "" / 0  →  匿名桶 "*"
其余值                →  str(值) 各自成桶（首见放行）
```

**匿名桶统一受门控**——所有"没有 key"的记录共用 `"*"` 一个桶按 interval 冷却（首条放行、
窗口内后续全拦）。这是 INB-0089 缺口①的修法：track 建立初期/闪断帧 `track_id=0` 不再绕过
门控每帧放行，而是与其它匿名记录一起按节奏冷却。同帧同 key 多条 = 同帧同桶，首条放行后
其余拦（防匿名桶被同帧多条绕过）。

## 2. 参数

| 参数 | 类型/默认 | 说明 |
|---|---|---|
| `key` | str，默认 `"track_id"` | 分组键字段名（**必填**，空串构造期抛 `ValueError`）。取值经 `conditions._field_value`（dict/对象统一；派生字段 `w/h/short/long/area` 亦可用；`track_id` 数值规整 None→0→匿名桶） |
| `interval` | float，默认 `5.0` | 三值语义见 §1.1 |
| `filter` | str，默认 `""` | 条件 DSL（与 det_filter/target_crop/annotate 同族）：只对**命中记录**做门控，未命中记录直接放行且**不触碰门控状态**（不耗该桶额度）；空 = 全部记录受门控；非法表达式构造期抛错 |

无 preset；spec 声明示例：

```json
{"module_id": "gate_filter", "params": {"key": "track_id", "interval": 5.0, "filter": ""}}
```

## 3. 输入 / 输出

- **INPUTS**：`dets` → Records（上游任意记录列表；对象进对象出、dict 进 dict 出）。
- **OUTPUTS**：`gate_filter.dets` → Records（放行子集，**原元素**返回）；另登记观测产物
  `gate_filter.stats` → Records（单元素 `[{"kind":"gate_stats","ts","frame_idx","key",
  "interval","in","passed","blocked","blocked_detail":{key值:条数}}]`；**非连线端口**，
  观测专用）。多实例产物全名 `gate_filter#<位置>.{dets,stats}` 互不覆盖。

**统计旁路（驱动层/时间线读取）**：

| 层 | 读法 |
|---|---|
| `Pipeline.last_gate` | `(进入, 放行)`——参照 `last_filter` 同型，本帧未运行 = None；时间线直读 |
| `pipe.artifacts["gate_filter.stats"]` | 分键明细（`blocked_detail` 里 `"*"` 计数 = 匿名拦截量） |
| 模块实例 `last_stats` | 诊断脚本直读（同 stats_record 内容） |

## 4. 上下游衔接

- **上游**：`seq_filter`（既有门控在前）；也接受 `det_filter` / `yolo.dets` 直连。
- **下游**：`target_crop` / `annotate` / 报送 / 事件尾——被拦记录对所有下游一致生效。
- **track_needed 联动（关键）**：`key=="track_id"` 时模块 `needs_track=True`，
  `Pipeline.track_needed()` 据此让 YOLO 开跟踪。**不要**只在链路里放
  `gate_filter(key=track_id)` 而把 `plate_recog.track_dedup` 关掉却不经
  `track_needed`——那是设计稿 §5.4 排掉的坑（track_needed 误判 False → 全链无
  track_id → 全部落匿名桶退化成全局限流）。内置去重与独立门控同开合法：门控在裁剪前
  拦截，PlateStage 二次门控自然短路，行为 = 取更严者。
- **与既有机制的关系**：`SeqFilter` 保留不动；`PlateStage.track_dedup` 机制保留作回切开关
  ——PIPE0002 默认已走本模块（`track_dedup=true` 即回内置门控，取更严者语义见 §4 track_needed
  联动）；收编路线见设计稿 §4。

## 5. 验证

```bash
conda run -n ai python tests/test_gate_filter.py   # 62 断言（纯逻辑秒级）
conda run -n ai python umvp/pipe/test_composer.py  # Pipeline 插槽/last_gate/artifacts/track_needed
```

灰度对跑（PIPE0002 A/B，真实推理）：A=现状内置 `track_dedup=true, cooldown=5`；
B=`track_dedup=false` + `gate_filter(key=track_id, interval=5)`。验收：同车识别频次
B ≤ A、匿名桶放行为 interval 节奏（tid=0 不再逐帧放行）、`last_gate`/stats 可见、
端到端事件不增。基线 INB-0089（内置版同车间隔 2.84s→7.84s→13.01s、7 事件）。
**结果：灰度通过（INB-0091），2026-10-27 WO-0024 起转正为 PIPE0002 默认链路**（B 版形态
入播种 7 步；`plate_recog.track_dedup=true` 即回切内置门控，机制未删）。

## 6. 注意事项与边界

- **`key="cls_id"` 时 COCO 类 0（person）落匿名桶**：0 按铁律归 `"*"`（与 SeqFilter
  `_key_of` 同律）。类级节流若在意此语义，改用 `label` 或接受"类 0 ≈ 匿名桶节奏"。
  track 级门控（主用例）用 `track_id`，0 恰是"无 track"，语义正合适。
- **匿名桶无"缺 key 直放"选项**（设计定死）：那即 INB-0089 缺口①的现状，不作为可配置项回潮。
- **被拦记录本体不留存**（本期）：观测只有计数与分键；需要"被拦样本预览"二期再议
  （如 `last_blocked` 环形缓冲）。
- `interval=0` 是合法 no-op（全放行）；spec 里这么配通常是无意义的占位，建议去掉该阶段。
- `reset()`：换视频流复用实例时清门控状态与统计（驱动层接线为可选交接项）。
