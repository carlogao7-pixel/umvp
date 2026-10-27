# 事件尾去重键升级：组合/回退键（设计稿）

> 状态：**已过审（2026-10-27，用户拍板：Q1 同车二次入镜再报 / Q2 事件名固定取首号 /
> Q5 tid=0 确认回退按 label）——实现 WO-0026.2（pipe-chain），组合键已落地** ｜
> 提出方：orchestrator（主单 WO-0025）｜ 承接：pipe-chain
> 背景诊断：INB-0089 缺口②（同车异号重复事件；2026-10-26 INB-0095 曾拍板保留，用户重启需求）；关联机制：
> `SeqRule.key` / `SeqFilter._key_of`（pipe/seq_filter.py）、`PlateStage`（plate_recog/plate_pipeline.py）、
> `Pipeline.step` 车牌发布段（pipe/composer.py ~884-890）、`PlateResult`（plate_recog/plate_recognizer.py:56）
> 前序：[gate_filter.md](gate_filter.md)（缺口① 匿名桶已修，WO-0023/0024；本篇处理缺口② 事件归并）

## 0. 背景与动机

### 0.1 问题：同车异号 → 多条事件

PIPE0002 事件尾现状（SEQ0002 preset）：

```json
{"match": "kind == plate", "event": "{label}", "key": "label", "cooldown": -1.0}
```

即**按车牌号终身一次**（`cooldown<0` = 同 key 只触发一次）。同一辆车因模糊/角度在
5s 门控重试时读出**异号**（INB-0089 实测：沪DHH230→沪OHH236→京GMH236，同一 tid=2 车），
号不同即绕过 → 同车出多条事件。

实测数据佐证（横幅1.MP4）：门控节奏下同一 track 每 5s 重识别一次，事件尾把 20 次识别
收敛成 5 条（按号）；同车异号场景下按号去重会误报多条——若按 track 聚合，预期收敛到
track 数量级。

### 0.2 数据缺口（已核实）：track_id 在两步丢失

track_id 本可从 YOLO 一路透传，但现状在「crop→PlateResult→发布记录」断链：

```
YOLO 检测（track_id ✓）
  → 裁剪 CropResult.track_id ✓（target_crop.py:55；to_dict/记录/ as_crop 重建均透传）
  → PlateStage.run 内部门控读 c.track_id ✓（plate_pipeline.py:84，仅用于识别去重门控）
  → PlateResult ✗（plate_recognizer.py:56，无 track_id 字段）——丢失点 1
  → composer 发布记录 ✗（composer.py:887 手工构造只有 kind/label/score/content）——丢失点 2
  → 事件尾 SeqFilter 拿到的记录无 track_id 可用
```

### 0.3 语义缺口：rules.key 仅单字段，无回退

`SeqFilter._key_of` 的归一化 None/""/0 → `"*"` 匿名组是 WO-0023 拍板定死的。若直接把
key 改成 `track_id`：tid=0/字段缺失的帧全部落匿名组，配合 `cooldown=-1` **整个视频终身
只出一条事件**——不同车全并成一条，比现状更糟。必须同时给出「tid 不可用时回退按 label」
的组合语义。

### 0.4 沿革说明

INB-0095（2026-10-26）：缺口②曾拍板「保留现状，同 track 多车牌输出是有意行为，后续作
单独数据处理」——即「事件收敛」推迟、「识别/落盘不收敛」是既定边界。本单是用户重启
聚合需求：**只动事件归并层，识别频次与素材落盘照旧**（见 §3 铁律 4）。

## 1. 数据面方案：track_id 透传链补齐

原则：**只透传，不碰任何门控/筛选逻辑**；全部改动带默认值，老构造代码零回归。

| # | 改动点 | 内容 | 零回归性 |
|---|---|---|---|
| D1 | `PlateResult`（plate_recognizer.py:56） | 增字段 `track_id: int = 0`（放有默认值字段区）；`to_dict()` 增 `"track_id": int(self.track_id)` | dataclass 默认值，老构造不传不炸；to_dict 多一键，消费方读已知键不受影响 |
| D2 | `extract_plates`（plate_recognizer.py:219-223 构造处） | 构造 PlateResult 时 `track_id=int(getattr(c, "track_id", 0) or 0)`（从 CropResult 透传） | CropResult.track_id 本就有（对象属性与记录键、as_crop 重建均已透传）；`getattr` 兜底容忍非标 crop |
| D3 | `_plate_record` adapter（plate_recognizer.py:100） | **零改动**——D1 的 to_dict 带上后自动随记录透传 | — |
| D4 | `PlateStage`（plate_pipeline.py） | **零改动**——内部 track 门控（track_dedup / IntervalGate）读 `c.track_id` 的逻辑不动；本方案只是让**结果**带上来源 track，与门控无关 | `use_track` 语义不变，track_needed 联动不受影响 |
| D5 | composer step 车牌发布段（composer.py:884-890） | 发布记录增 `"track_id": int(getattr(p, "track_id", 0) or 0)`（与 content 平级，进 JSON 主干） | 现有 rules `key: "label"` 不读 track_id → 行为逐字不变；产物表 `plate_recog.plates`（to_record 路径）经 D1/D3 自动带键，两条路同步补齐 |

要点：

- **direct 模式**（`mode="direct"` 整帧直读，无 crop 来源）：track_id 恒 0 → 事件尾回退
  按 label（= 现状行为），语义自洽。
- **发布记录带 `track_id=0` 不是脏数据**：0 即「无 track」，与 records 契约的
  `track_id` 通用字段语义一致；归一化（0→无效）交给事件尾的 key 层（§2/§3）。
- 不新增旁路：track_id 是整数标量，走 JSON 主干（契约：图像/向量才走旁路）。

## 2. 语义面方案：组合/回退键（两候选对比定案）

### 2.1 候选 A：`SeqRule.key` 支持逗号组合字段，按序取第一个有效值

```python
# seq_filter.py _key_of 新实现（单文件一处）
def _key_of(self, rec: dict, key: str) -> str:
    """分组键取值：key 支持逗号组合字段（如 "track_id,label"），按序取第一个
    有效值（非 None/""/0）；全部字段无效 → 匿名桶 "*"。
    单字段 key 与旧实现逐字等价（零回归）。"""
    for f in key.split(","):
        f = f.strip()
        if not f:
            continue
        v = rec.get(f)
        if not (v is None or v == "" or v == 0):
            return str(v)
    return "*"
```

- `key: "label"` → split 后单字段，循环体走一次，与旧实现逐字等价；
- `key: "track_id,label"` → tid>0 按 track 分组；tid=0/字段缺失 → 该字段无效，换下一个
  字段（label）→ **回退按号**（= 现状行为）；
- 全部字段无效（如记录既无 track_id 也无 label）→ `"*"` 匿名桶（与现归一化同律）。

### 2.2 候选 B：不动 seq_filter，发布记录预聚合 `dedup_key` 字段

composer 车牌段构造发布记录时聚合好一个有效键：

```python
tid = int(getattr(p, "track_id", 0) or 0)
rec["dedup_key"] = f"t{tid}" if tid > 0 else f"l:{p.plate_no}"
# rules 仍单字段：key: "dedup_key"
```

### 2.3 对比与定案

| 维度 | 候选 A（组合键） | 候选 B（预聚合 dedup_key） |
|---|---|---|
| **改动面** | seq_filter `_key_of` 一处——**通用能力进通用模块**；数据面透传（D1-D5，两候选都要做） | composer 车牌段（**发布边界专用代码**）；数据面透传同左 |
| **对其它链路/自定义链路** | 任意记录来源（det/plate/vlm/自定义）直接可用；设计器自组链路免费受益 | 仅车牌发布路径有此字段；VLM/自定义记录要同类能力得逐处复制聚合代码 |
| **设计器可理解性** | 参数框填 `"track_id,label"`，一句话语义：「逗号组合，按序取第一个非空(非0/None)字段值做分组键」 | 用户需理解 `dedup_key` 的编码格式（`"t5"` / `"l:沪A"`）——黑盒；想换回退顺序改不了 |
| **回退顺序可配性** | 配置即语义（换序改字符串，如 `"label,track_id"`） | 写死在 composer 代码 |
| **匿名桶语义兼容** | 兼容：值级归一化代码一行不动（§3 铁律 2）；全字段无效才落 `"*"` | 兼容（绕开归一化，但也绕开了机制——匿名组对 dedup_key 永不生效） |
| **碰撞处理** | 无需（字段值原样 str，不同字段不混） | 需前缀 `t`/`l:` 防「track 5」与「label 恰为 "5"」碰撞——字符串拼接是味道 |
| **单字段零回归** | 逐字等价（split 单元素路径） | 天然（不碰 seq_filter） |

**定案：候选 A**。理由：改动面小且位置正确（组合/回退是「分组键取值」的一般语义，
归属 seq_filter 而非某个发布点）；设计器里一句话可解释、可自由组合；单字段路径逐字
等价给出硬零回归保证。候选 B 的唯一优势（seq_filter 零改动）被 A 的「单字段逐字等价」
抵消，而其专用性/黑盒编码/不可复用是长期负债。

## 3. 语义铁律（硬约束逐条落实）

1. **现有单字段 key 配置零回归**：`_key_of` 单字段路径逐字等价（§2.1）；`tests/test_seq_filter.py`
   现有用例一行不改全过。`SeqRule` 其余字段（match/event/span/window_n/min_n/hit_ratio/cooldown）
   不动；key 不加构造期校验（空串/纯逗号 → 全无效 → `"*"`，与现状 `rec.get("")→None→"*"` 同律）。
2. **匿名桶归一化不动**：None/""/0 → `"*"` 的**值级**归一化保留原样；组合键只改「选哪个
   字段的值」这一层。回退不等于匿名：`tid=0 + label 有效` → **按 label 分组**（非 `"*"`）。
   反例论证：若 tid=0 也归 `"*"`，无 track 帧的不同车牌会并成一桶、`cooldown=-1` 时终身
   一条——丢事件，比现状差。故铁律：**回退发生在选字段层，归一化只对「最终选中值」生效**。
3. **PIPE0001 VLM 记录行为不变（兼容性试金石）**：VLM 发布记录为
   `{"kind": "vlm", "label": vlm_action, "text": ""}`（server.py ~2268/~2373），**无 track_id
   字段**。两层保证：
   - 其 rules 保持 `key: "label"` 单字段 → `_key_of` 单字段路径 → 逐字不变（铁律 1 直接覆盖）；
   - 即便有人配成 `key: "track_id,label"`：`rec.get("track_id")` = None → 无效 → 回退
     label → 分组结果与 `key: "label"` 完全一致。**组合键对缺字段的记录自动退化为首个
     有效字段，不会误聚合**——不同警情仍按 label 各自成组、各自冷却 30s，行为不变。
4. **事件收敛 ≠ 识别/存证收敛**：本方案只动事件尾的归并键。PlateStage 的识别频次
   （gate_filter 5s 节奏）、产物表登记、plate_img/关键帧落盘全部照旧——INB-0095 拍板的
   「识别/落盘不收敛」边界不动，异号车牌素材仍完整留存，只是**不再各出一条事件**。
5. **执行序不变**：事件尾仍在链路尾（publish → event_filter），组合键判定发生在事件尾
   内部，与上游 det_filter/gate_filter 的门控层各司其职：门控管「识别频次」（匿名桶
   拦截在裁剪之前），事件尾管「事件归并」（组合键分组在发布之后）。

## 4. PIPE0002 具体改法

### 4.1 换装后 rules（SEQ0002 preset）

```json
[{"match": "kind == plate", "event": "{label}", "key": "track_id,label", "cooldown": -1.0}]
```

仅 `key` 一个词的变化：`"label"` → `"track_id,label"`。

### 4.2 cooldown 语义随键的调整说明

`cooldown<0` = 同 key 只触发一次——「同 key」的含义随组合键自然变为「同（回退后的）组」：

| 场景 | 现状（key=label） | 换装后（key=track_id,label） |
|---|---|---|
| 同车同号重复识别 | 1 条（号去重） | 1 条（track 去重，更早收敛） |
| 同车异号（沪DHH230→沪OHH236） | **N 条（每号一条，缺陷）** | **1 条（track 聚合，事件名=首号）** |
| tid=0 帧的车 | 按号去重 | 回退按号去重（= 现状，不劣化） |
| 二次入镜（新 track） | 按号（首见号新报） | 新 tid 新组 → 再报 1 条（待拍板 Q1） |

- **事件名取首号**：`{label}` 渲染读达成时刻记录的 label，即该 track 首次触发事件的读数；
  后续异号被抑制不产事件。若需「同车全号码」，组内明细可走 detail（Q2，本期不做）。
- **cooldown 改正值的影响**：若从 -1 改成 N 秒，语义 = 同车 N 秒内只报一条、到期同组可
  再报（二次入镜语义不变，新 tid 本就是新组）。默认仍取 -1（终身一次/车）。

### 4.3 行为说明：同车二次入镜（新 track）

车开出画面 → ByteTrack 销毁 track → 车再入镜分配新 tid → 组合键取新 tid → 新组 →
`cooldown=-1` 对新组无效 → **再报一条**。这与现状（新入镜首见号新报）方向一致，仅当
新 track 首帧恰好读出与旧 track 相同的号时，现状报、新案也报（号同但 tid 不同 → 不同组）
——即换装后**二次入镜必再报**（现状下若号相同则不报）。差异点与拍板见 Q1。

## 5. 三条链影响分析

| 链路 | 事件尾 | 影响 |
|---|---|---|
| **PIPE0001 警用无人机** | rules `key: "label"`, cooldown=30.0；VLM 记录无 track_id | **零影响**。rules 不改 → 单字段路径逐字不变（铁律 1）；即便改组合键也回退 label 行为不变（铁律 3 试金石）。数据面 D1-D5 均在车牌路径，VLM 路径不经过 |
| **PIPE0002 车牌** | rules `key: "label"` → `"track_id,label"` | **主场景**。数据面补透传 + rules 换组合键；同车异号收敛（§4.2 表） |
| **PIPE0003 人脸截取** | **无事件尾**（截取落盘即目的；face 阶段只登记产物 `face_detect.faces`，不经 publish） | **不受影响**。无 publish → 无事件归并语义可谈 |

自定义链路：seq_filter 是通用模块，组合键对所有「记录带 track_id 的事件尾」开箱可用
（如 YOLO→det_filter→seq_filter 直接按 track 聚合检测事件）；不带 track_id 的记录自动
回退后续字段，无破坏面。

## 6. 测试计划（纯逻辑单测用例表）

### 6.1 `tests/test_seq_filter.py` 增用例（组合键，不加载模型秒级）

| # | 用例 | 断言要点 |
|---|---|---|
| 1 | 单字段逐字回归 | 现有全部用例不改通过；`key="label"` 路径与旧实现同结果（等价性断言） |
| 2 | 组合键 tid>0 聚合 | 同 tid 异 label × N 条（cooldown=-1）→ 恰 1 事件；事件名 = 首条 label；detail.key = tid 串 |
| 3 | tid=0 回退 label | tid=0 + 异 label 各自成组（= 现状行为）；tid=0 + 同 label → 1 条 |
| 4 | 全字段无效归 "*" | 缺 track_id 且缺 label → `"*"` 组；多条全无效 + cooldown=-1 → 终身 1 条（匿名桶语义不变） |
| 5 | **VLM 记录兼容** | `{"kind":"vlm","label":"治安类","text":""}` + `key="track_id,label"` → 按 label 分组，与 `key="label"` 的事件序列逐条一致 |
| 6 | key 空格容忍 | `"track_id, label"` ≡ `"track_id,label"`；纯逗号/空串 → 全 `"*"`（不抛错，同现状） |
| 7 | 窗口规则 + 组合键 | span/window_n 窗口按组合键分组计数；跨组互不掺窗 |
| 8 | cooldown≥0 + 组合键 | 同组抑制期内扣留、到期可再报；异组不受影响 |
| 9 | 对象形态双兼容 | Detection 对象（to_record 后带 track_id）与 dict 双形态同结果 |

### 6.2 数据面用例（`tests/test_plate_recog_pipeline.py` + `umvp/pipe/test_composer.py`）

| # | 用例 | 断言要点 |
|---|---|---|
| 1 | PlateResult 默认值 | `PlateResult(...)` 不传 track_id → 0（老构造零回归）；to_dict 带 track_id 键 |
| 2 | extract_plates 透传 | 构造带 track_id 的 CropResult → 产出 PlateResult.track_id == 来源 tid；direct 模式恒 0 |
| 3 | 发布记录带键 | composer step 车牌段：publish 记录含 track_id；产物表 `plate_recog.plates` 记录含 track_id |
| 4 | 端到端小链 | fake YOLO（带 tid）→ target_crop → PlateStage(mock) → publish → SeqFilter(rules key="track_id,label") → 同 tid 异号收敛 1 事件 |

### 6.3 既有回归（改动关联必跑）

```bash
conda run -n ai python umvp/pipe/test_composer.py     # D5 发布段改动
conda run -n ai python tests/test_seq_filter.py       # _key_of 改动（核心）
conda run -n ai python tests/test_plate_recog_pipeline.py  # D1/D2 PlateResult/extract_plates
conda run -n ai python tests/test_police_uav_pipeline.py   # PIPE0001 试金石（行为不变）
conda run -n ai python tests/test_det_filter.py       # 同族回归
conda run -n ai python web_lab/validate_chain.py <PIPE0002 spec>  # 换装后 spec 校验
```

## 7. 端到端验收（真实素材对跑）

**工具**：web 链路测试台（`/api/test/pipeline_run`，PIPE0002 + 视频）；原 /tmp 诊断脚本
已清理，对跑前可重建轻量版（组装链路 → 逐帧 step → 收集 last_events）。

| 跑法 | 素材 | 配置 | 目的 |
|---|---|---|---|
| 基线复测 | 横幅1.MP4 / 车牌识别.MP4 | 现状 rules（key=label） | 复测基线（gate_filter 换装后数字或与 INB-0089 的「7 事件」不同，以复测为准；横幅1.MP4 参考基线：20 识别 → 5 事件） |
| 实验组 | 同素材 | rules 换 `key: "track_id,label"` | 对比事件条数与内容 |

**验收点**（实验组 vs 基线）：

1. **同车异号收敛**：车牌识别.MP4 中 tid=2 车的沪DHH230/沪OHH236/京GMH236 类异号事件
   收敛为 1 条（事件名=首号）；总事件数 ≤ 基线（预期 ≈ 有效 track 数）。
2. **横幅1.MP4**：20 次识别产出的 5 条（按号）事件收敛至 track 数量级；无新增误报。
3. **识别/落盘不缩水**（铁律 4）：两次跑的识别次数（gate_filter 放行数）、车牌产物数、
   落盘小图数一致——事件收敛不牺牲存证。
4. **PIPE0001 回归**：警用链路对跑（任一含 VLM 词表命中素材）事件序列与基线一致
   （试金石实测）。

真推理跑较慢（YOLO+HyperLPR），属实现阶段「待测试」执行；设计阶段仅交付本节清单。

## 8. 交接清单（@web，实现阶段完成后派发）

1. `_seed_pipeline` 的 SEQ0002 preset：rules key 改 `"track_id,label"`（**注意播种幂等**：
   已部署 DB 中 preset 已存在，改默认值是否自动迁移需按播种策略确认——见 Q3）。
2. `PARAMS["seq_filter"]["rules"]` 描述文案补组合键说明（「逗号组合，按序取第一个非空
   (非0/None)字段值做分组键」）→ 设计器 seq_filter 节点提示同步。
3. `validate_chain` 零改动预期：rules 走 `_seq_filter_of` 干跑构造，key 组合串不涉及
   白名单校验（实现阶段验证一条含组合键的 spec 即可）。
4. 时间线/事件展示：事件 detail.key 将出现 tid 串（组合键值），展示无需特改；如需
   「按车聚合」视图另议（不在本单）。

## 9. 待确认问题（列给用户拍板）

| # | 问题 | 倾向 |
|---|---|---|
| Q1 | **同车二次入镜（新 track）要不要再报？** 换装后新 tid 必再报（现状下号相同则不报）。告警视角：再次出现再次告警（合理）；存证视角：素材照常落盘不受影响 | **再报**（新 tid 新组是自然语义；不愿再报可把 cooldown 从 -1 改正值 + 接受同组到期重报的组合语义，暂无「跨 track 终身一次」需求） |
| Q2 | 事件名固定取**首号**（后续异号被抑制不出事件），是否够用？ | 够用——异号本就是误读，首号即该车代表号；「组内全部号码」二期需要时往事件 detail 加，不动键语义 |
| Q3 | 已部署环境的 SEQ0002 preset 迁移方式：播种幂等覆盖 or 保留手改？ | **@web 确认播种策略**后定；倾向播种时对「用户未改过的 preset」静默升级、改过的保留并文档说明手工换法（一行 key） |
| Q4 | 组合键在设计器/文档的表述方式 | 「分组键支持逗号组合字段，按序取第一个非空（非 0/None）字段值；全部无效归匿名组」一句话进 PARAMS 文案 + seq_filter 模块分册 + 本篇 |
| Q5 | tid=0 回退 label（而非归匿名组）的边界语义确认 | **确认回退**（§3 铁律 2 反例论证：归匿名组会丢事件）；这是组合键的核心价值，需用户认可该边界 |
