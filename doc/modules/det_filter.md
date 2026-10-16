# 检测筛选 DetFilter

> 模块组：管道通用模块 | 代码：`umvp/pipe/det_filter.py`（条件引擎 `umvp/pipe/conditions.py`，记录适配 `umvp/pipe/records.py`）
> 数据库参数：`preset_params_det_filter`（`DF0001+`：DF0001 警用无人机 / DF0002 车牌识别 / DF0003 人脸截取）
> 状态：**有效性筛选的唯一来源**——三条保留链路均在 YOLO 之后显式插入本阶段；YOLO preset 的有效性参数已删除。

## 1. 模块定位

通用下游筛选器 +（可选）事件标注，可作链路尾。三层能力：

1. **逐条筛选**——条件 DSL：范围比较 + 条件间**与 / 或 / 非**组合，字段名灵活（别名 + 派生宽高尺寸量）。
2. **聚合谓词**——`count(...)` 同帧计数（字段分组 / 子条件计数 / 总数）。
3. **成组数量判定**——`group_limits` 按类别计数，越界或未登记类别整类剔除。

只取舍、不改内容：放行元素**原样返回**（对象进对象出、dict 进 dict 出），链路上透明插入、
对所有下游一致生效。经 `to_record` 统一字段视图后判定，因此可作**几乎所有模块**的下游——
上游是 YOLO（`list[Detection]`）还是产出 dict 记录的模块（车牌/人脸/裁剪结果）都能筛。

## 2. 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| filter | str | 空 | 条件 DSL（见下）；空 = 全放行。非法条件在构造时抛 ValueError（装配期快速失败） |
| group_limits | dict | None | 类别 → (数量下限, 数量上限)：**成组数量判定**，越界或未登记类别整类剔除（视为未识别）；`None`=不判定；显式空 dict=全拦（程序化构造时用；web/链路空值一律 None） |
| predicates | list | 空 | 代码级谓词链（可选，按序 AND 叠加在 DSL 之后）——`compose()` 的自定义过滤链等高级用法；DSL 能表达的场景不必用它 |
| events | list | 空 | **事件规格**（可选，可作链路尾）：`{"name": 事件名, "when": 条件DSL(可选), "detail": [字段名...]}`；放行记录命中即追加事件（**支持一条记录多个事件**），`detail` 从记录字段取值（如 VLM 的 JSON 内容字段） |

### 2.1 事件标注（可作链路尾，通用、不绑定模块）

**链路尾语义**：以筛选模块收尾时，**过滤后幸存的记录本身就是告警载体**（通过所有闸
即命中），其携带的 **`events` 即链路产出**——不再需要"另造告警对象"的物化环节。

`events` 规格让 DetFilter **直接充当链路结尾**：放行记录上按规格追加事件，产物即
携带 `events` 的记录 / 事件列表。通用事件约定见 `records.py`：

```
记录字段：events = [{"name": 事件名, "ts": 时间戳, "detail": {自定义内容}}, ...]
add_event(item, name, ts=..., detail=...)   # 任何模块都可给记录追加
collect_events(items)                        # 链路尾汇总为输出
```

- 幸存但**未带 events** 的记录语义上仍是"命中"，但没有名字/内容——要让产出可读，
  在尾部配 `events` 规格（或规则），或直接以幸存记录字段为产出。
- 多条规格 → 同记录多个事件；dict 输入原地 append，对象输入挂 `events` 属性。
- `detail: ["content"]` 可把 VLM 阶段塞进记录的 JSON 内容字段原样带出。
- Pipeline 每帧 `last_events` = 末段记录的事件汇总（`collect_events`），供驱动层读取；
  `Pipeline.publish(records, ts)` 把外部产生的记录（如 VLM 结果）经本阶段/尾部处理出事件。
- 链路终段可用本模块（`_chain_from_spec` 允许 det_filter/seq_filter 收尾）。

> 装配：链路上本阶段紧随 YOLO，是有效性过滤的**唯一来源**；`web_lab.server._det_filter_of()`
> 由 preset 参数构造 `filter`/`group_limits`。

### 2.2 条件 DSL 语法（conditions.py）

- 条件 = `字段 操作符 值`；操作符 `== != > >= < <= in`（`=` 等价 `==`；`in` 值用 `|` 分隔）
- 组合：组内逗号 `,` = **并且**；组间分号 `;` = **或者**；`not` = **非**（可连用
  `not not`）；`(` `)` 分组（括号内可再写 `,` / `;`，任意嵌套）
- 值本身含 `, ; )` 时用引号 `'` `"` 包住；数值条件按数字比较，否则按字符串比较

字段表（别名 → 标准名；不绑定来源，dict 键或对象属性皆可）：

| 字段 | 别名 | 说明 |
|---|---|---|
| cls_id | class / cls / 类别 | 类别编号（COCO：0=人，2=车） |
| track_id | track / id | 跟踪 ID |
| score | conf / 置信度 | 置信度 |
| w / h | width / 宽；height / 高 | **派生**：从 bbox 算宽 / 高 |
| short / long | 短边 / 长边 | **派生**：框的短边 / 长边像素 |
| area | 面积 | **派生**：宽 × 高 |
| ts | 时间戳 | `run()` 注入的帧时间戳 |
| frame_idx | 帧号 / frame | `run()` 注入的帧序号 |
| 其余字段 | — | 记录自带字段直接引用（label / text / kind / plate_no…） |

**聚合谓词**（同帧计数，与逐条条件同语法、可任意组合）：

| 写法 | 含义 |
|---|---|
| `count(字段)` | 当前记录在**该字段值**上的同帧分组大小（如 `count(cls_id)` = 同帧同类个数） |
| `count(子条件)` | 同帧满足该子条件的记录数（帧级常量，对所有记录一致） |
| `count()` | 同帧记录总数 |

- **两种分组语义**：`count(字段)` 是"按该字段值分组，当前记录所在组的大小"，等价
  `count(字段 == 本记录该字段值)`，**整组同进同出**（适合"每类数量上下限"）；
  `count(子条件)` 是同帧满足该子条件的记录数（帧级常量，对所有记录一致）。
- **"同帧"判定 = `run()` 调用边界**：一次 `run()` 的入参就是该帧全部检测（Pipeline 每
  分析帧调一次），无需从数据推断；若入参跨帧混入且记录自带 `frame_idx`，按 `frame_idx`
  二次分组（各帧独立计数）。
- 对同组记录返回相同值 → **整组同进同出**，保持"只取舍"契约。
- 子条件里**不再支持嵌套 `count`**；无聚合上下文时（目标裁剪/标注的逐框筛选）`count`
  取不到值 → 该条件不通过。

> `group_limits` 与 `count` 的关系：前者是"按类数量上下限"的**便捷参数**（未登记类别
> 剔除），后者是通用聚合谓词；两者可共存（AND）。`group_limits={0:(1,300)}` 近似
> `cls_id == 0, count(cls_id) >= 1, count(cls_id) <= 300`。

示例：

```
"score >= 0.5"                          置信度下限
"cls_id in 0|2, short >= 30"            人/车 且 短边≥30 像素
"not cls_id == 2"                       排除车
"not (cls_id == 0, score < 0.5)"        非（人且置信不足）
"(cls_id == 0; long >= 60), ts >= 10"   (人 或 长边≥60) 且 10 秒后
"count(cls_id) >= 2"                    同帧同类≥2 才留（轻量成组判定）
"count(cls_id == 0, score >= 0.5) >= 2" 同帧"人且置信≥0.5"的≥2 才留
"count(cls_id == 0; track_id == 3) <= 5" 同帧满足该 OR 子条件的≤5 才留
"count() <= 100"                        同帧总量上限（超了整帧都不留）
"not count(cls_id) >= 2"                同帧同类<2 才留
```

## 3. 输入（必要字段）

| 字段 | 类型 | 内容 |
|---|---|---|
| dets | list | 元素为任意对象或 dict（Record）：Detection、车牌/人脸/裁剪产物、通用记录 |
| ts | float | 帧时间戳（注入记录供条件引用，可省略） |
| frame_idx | int | 帧序号（同上） |

> 通用记录（Record）约定：模块逐帧输出 = list[dict]，必备 kind/ts/frame_idx，
> 通用字段 bbox/score/cls_id/track_id/label/text，旁路字段 image/embedding
> （像素与向量不进 JSON）。见 `umvp/pipe/records.py` 模块头注释。

## 4. 输出（字段和内容）

放行元素组成的子集（`list`）——**原元素原样返回**，不构造新对象、不改字段；
上游是 Detection 出来就是 Detection，是 dict 出来就是同一个 dict。

## 5. 上下游衔接

- 上游：任何产出 list[Detection] 或 list[Record] 的模块（YOLO `infer()` 最典型）。
- 下游：目标裁剪 / 标注 / VLM 报送编排 / 事件尾——本模块不改数据形态，下游无感知。
- 与消费模块选材参数的分工：本模块管**有效性**（不达标 = 该检测在链路上不存在，对
  所有下游一致）；TargetCrop/Annotator 自带的 classes/filter 管**选材**（本阶段消费哪些
  存在的检测），两者 AND 叠加、互不取代。

## 6. 验证

```bash
conda run -n ai python tests/test_det_filter.py    # 本模块（纯逻辑，秒级）
# 共用条件引擎，改动 conditions.py 后必跑回归：
conda run -n ai python tests/test_target_crop.py
conda run -n ai python tests/test_annotate.py
conda run -n ai python umvp/pipe/test_composer.py
```
（断言数不写死，以脚本运行自报为准。）

## 7. 调参速查（通俗指南）

- **碎框/小目标太多** → `short >= 30` / `long >= 40`（等价旧 min_short/min_long）。
- **低置信噪声多** → `score >= 0.5`（等价旧 filter_conf）。
- **只想关注部分类别** → `cls_id in 0|2`；**排除某类** → `not cls_id == 2`。
- **多条件组合** → 括号 + 逗号/分号：`(cls_id == 0; long >= 60), area > 100`。
- **成组数量** → `group_limits="0:2,300;2:1,300"`（人≥2、车≥1）或 `count(...)` 谓词。
- **一段时间后才处理** → `ts >= 10`（配合 run 的 ts 注入）。
- 条件写错（缺操作符/括号不闭合/值为空）在**装配时就报错**，不会带到运行时。

## 8. 边界与路线

**明确不做**（边界）：不改记录内容（事件标注除外）、不裁图（目标裁剪）、不画框（标注）、
不构造 VLM 报送请求（SubmitRequest）、不做跨帧时序确认（归 [时序判定](seq_filter.md)）。

**与「时序判定 SeqFilter」分层不重复**：

| 作用域 | 归属 | 典型能力 |
|---|---|---|
| 单帧 / 批内 | **本模块 DetFilter** | count、成组数量判定、坐标/数值/字符串条件 |
| 跨帧时序 | **SeqFilter** | 时间窗计数、连续 N 次、驻留时长、track 集合变化、事件标注 |

两者**共用同一条件/表达式引擎**（`conditions.py`）：DetFilter 每次 `run()` 携带单帧上下文，
SeqFilter 携带跨帧状态；引擎扩展一次实现、两处受益。**不把时间窗/跨帧塞进 DetFilter，
也不把单帧分组计数塞进 SeqFilter。**

**未来通用化方向**（待排期，非现状）：条件两侧放开为表达式（坐标/四则运算）、字符串关系
运算、`avg/max/min/sum` 聚合、ROI/区域命中。
