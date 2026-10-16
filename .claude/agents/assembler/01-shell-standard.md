# 链路壳标准（唯一权威）

> 读者：组装工 ｜ **本文件是链路构建的唯一权威规范**。组装工的产出 = 符合本规范的 `stages spec`；
> 装配与运行由平台完成，**不手写"拼接代码"**。照抄 §4 模板、过 §5 清单即算合格。

## 0. 一条链路长什么样（唯一真相）

链路定义 = 一个 `spec`（JSON/dict）：

```json
{
  "name": "链路名",
  "description": "一句话业务目的",
  "streams": 1,
  "stages": [
    {"module_id": "frame_manager", "params": {}},
    {"module_id": "yolo",          "params": {}},
    {"module_id": "det_filter",    "params": {}},
    {"module_id": "seq_filter",    "params": {}}
  ]
}
```

**硬性规则（违反即报错，必须逐条满足）**：

1. `stages` 非空；`frame_manager` **必须**存在且**仅一段**，惯例放**首位**。
2. **每个 `module_id` 在一条链路中最多出现一次**（保存时强校验，重复即拒绝）。
3. **终段**（`stages` 最后一段）必须是 `det_filter` / `seq_filter` / `plate_recog` /
   `face_detect` / `face_embed` 之一。
4. `params` 只用该模块**参数白名单**内的精确键名，不塞未知键——**白名单以 `list_modules` 现取为准**。
5. **逐帧分析阶段一律经 `Pipeline.step` 插槽执行**：帧管理 / 识别（yolo / 车牌 / 人脸）/ 筛选
   （det_filter / seq_filter）/ 时序 / 裁剪 / 标注 / 研判（vlm）等**逐帧**阶段，**必须**由
   `Pipeline` 的固定插槽驱动；**禁止**在驱动层（`web_lab/server.py` 等）为某类链路新增
   **专用单帧路径**。

**三类数据流（运行顺序由壳固定，不由 stages 数组顺序驱动）**：

```
检测流： frame_manager → yolo → det_filter → seq_filter(节流/变化) → target_crop → annotate
报送流： 检测流末端按类别分组 → SubmitRequest → vlm 阶段
事件流： vlm/车牌结果经 Pipeline.publish() → 事件尾 → events（链路输出）
```

> **顺序说明**：装配器 `_chain_from_spec` 按 `module_id` 组装到 `Pipeline` 的**固定槽位**，`Pipeline.step()`
> 按上面固定顺序每帧调用一次——**stages 数组顺序不驱动运行**。数组顺序仅供声明 / 存储（`pipeline_steps.position`）/
> 展示，约束只有：`frame_manager` 在首、终段在白名单、每模块至多一次（中间顺序不校验）。因此 spec 无需
> 额外的 `id/seq` 字段。
>
> **为何每模块至多一次**：每个模块类型只有一个槽位（`Pipeline` 的 `yolo / det_filter / seq_filter / crop /
> annotate / plate / face / vlm / event_filter` 各一个字段）；重复 `module_id` 无法表达（装配器只取第一份参数），
> 故在 `validate_chain` 与 `db.save_pipeline` 处直接拒绝，避免静默丢弃。
>
> **执行路径（逐帧阶段一律走插槽）**：所有逐帧分析阶段**只**经 `Pipeline.step` 的固定插槽执行，
> **不存在**"某类链路的专用单帧路径"。**新模块接入 = 三件套**：① `Pipeline` 提供插槽（鸭子类型
> `run(...)`，与既有槽位对称）② 拼接层 `_chain_from_spec` 按 spec 注入 ③ `CHAIN_ASSEMBLED_MODULES`
> （`web_lab/server.py`）登记；**缺任一件，spec 声明了它也不会执行**。平台以 `web_lab/validate_chain.py`
> 的**「声明即执行」断言**兜底——声明但未被装配的阶段直接报错
> （`声明即执行: 阶段 … 未被 _chain_from_spec 装配进执行路径`）。
>
> **反例警示（本条规矩的由来）**：人脸链（PIPE0003）历史上曾走驱动层**专用单帧路径**，导致 spec 声明的
> `det_filter` **声明了却不执行**（静默失效，WO-0001）；现已并入 `Pipeline.step` 插槽修复。
>
> **产物端口（阶段间连线）**：模块以 `INPUTS` / `OUTPUTS` 自描述端口，spec 阶段可选 `inputs` 显式连线；
> **不写 = 现状固定流水**。详见 §3（完整接入清单见 `03-patterns-and-extension.md`）。

## 1. 标准产物

- **必产**：`stages spec`（§0 的结构）。
- **可选**（仅独立脚本 / 离线验证时）：驱动壳、断言式测试壳。
- **禁止**：新建"拼接 / 胶水模块"、改核心模块（走升级通道）；用已删除 / 改名前的旧概念（见 §6）。

## 2. 模块与参数（来源：`list_modules`）

**参数键名 / 类型 / 默认 / 取值一律以 `list_modules` 现取为准**（唯一与代码同步；新模块自动出现）。
下面是常见阶段的作用与关键参数**速记**（仅为理解，写 spec 时仍以 `list_modules` 为准）：

| module_id | 作用 | 参数（速记，键名以 list_modules 为准） |
|---|---|---|
| `frame_manager` | 取帧节奏 + 背压（必首） | 取帧式、抽帧间隔、时间/帧数间隔、背压档 |
| `yolo` | 小模型**纯推理**（框目标） | 模型、conf、iou、imgsz、max_det、classes、device |
| `det_filter` | 有效性筛选 + 事件标注 | filter(条件 DSL)、group_limits、events |
| `seq_filter` | 时序门控（检测流）+ 事件尾 | throttle、change、rules |
| `target_crop` | 按上游检测框裁子图 | out_size、margin、classes、filter |
| `annotate` | 帧上画检测框（供 VLM） | thickness、show_label、classes、filter |
| `vlm` | 大模型研判（素材 + 推理） | max_resolution、image_source、image_pick、prompt、use_real_vlm、vlm_endpoint/model/key |
| `plate_recog` | 车牌识别（消费车图） | detect_level、mode、margin、dedup_iou、track_dedup、track_cooldown |
| `face_detect` | 人脸检测（消费人图） | device、det_thresh、det_size、max_num、use_crop、margin、dedup_iou |
| `face_embed` | 人脸嵌入 | device |
| `face_store` | 向量底库检索 | db_path、thresh、topk、device |

**VLM 送审素材（`image_source` 选「哪张」× `image_pick` 选「几张」）**：
- `vlm.image_source` ＝送审用**哪一类**图——`annotated`（默认，＝带框图，无上游 `annotate` 时回退
  原帧）/ `frame`（本帧原帧）/ `crops`（与本次报送目标匹配的裁剪子图，**须链路同时含 `target_crop`
  阶段**，按 bbox / IoU≥0.5 命中；按类报送时收集**该类全部命中对象**的子图＝图集）。
- `vlm.image_pick` ＝从所选图源里送**几张**——`first`（默认，只送首张，与旧行为逐字等价）/ `all`
  （送**图集全部**，多图 OpenAI 兼容 payload）。仅 `image_source=crops` 且该类命中多个目标时才有区别；
  单图源下 `first`/`all` 等价。
- **缺省＝现状（零回归）；缺源一律回退、不报错**（`crops` 未命中 → `annotated` → 原帧）；参数细节见 `doc/modules/vlm.md`。

> **连线校验**：`validate_spec` 对非法取值报错，对「声明 `annotated` 但上游无 `annotate`」「声明
> `crops` 但上游无 `target_crop`」给**警告**（非致命，缺源自会回退）。历史注脚：`target_crop + vlm`
> 组合曾因 composer 补漏而**不产生送审**（静默失效），已修复——该组合现按检测按类正常报送 VLM，
> 同时产出 `last_crops` 供 `image_source=crops` 取用。

**条件 DSL**（`det_filter.filter` / `target_crop.filter` / `annotate.filter`）：`字段 操作符 值`，
组内逗号 = 并且、组间分号 = 或者、`not` = 非、`()` 分组；字段 `cls_id/类别`、`track_id/track`、
`score/conf/置信度`、派生 `w/h/short/长边/long/area`、注入 `ts/frame_idx`；
聚合 `count(字段)/count(子条件)/count()`。

**事件规格**：
- `det_filter.events` = `[{"name": 事件名, "when": 条件DSL(可选), "detail": [字段名...]}]`（多事件）。
- `seq_filter.rules` = `[{"match": 条件DSL, "event": 事件名(支持 `{字段}` 动态), "key": 分组键,
  "span"/"window_n"/"min_n"/"hit_ratio"/"cooldown"}]`；`cooldown<0` = 同对象一次。

## 3. 产物端口协议（inputs 连线）

> 目的：让阶段可声明「从上游取哪一部分」——模块自描述端口，壳（spec）按需连线；**缺省 = 现状固定流水**，
> 零回归。连线是**增量**，不改 §0 的固定运行顺序。

**产物与命名（`{阶段标识}.{产物名}`）**：每帧登记在 `Pipeline.artifacts`，值分三类——类型三选一，
连线前先认类型：

| 类型 | 含义 | 本期已登记产物 |
|---|---|---|
| `Image` | 图像（ndarray，旁路，不进 JSON） | `frame_manager.frame`、`annotate.annotated` |
| `Records` | 记录列表（dict / 对象，JSON 主干） | `yolo.dets`、`det_filter.dets`、`seq_filter.dets`、`target_crop.crops`、`face_detect.faces`、`plate_recog.plates` |
| `Vector` | 向量（ndarray，旁路） | （本期无生产者登记，类型保留） |

> 可用产物清单与类型以 `ARTIFACT_TYPES`（`pipe.composer`）为**唯一来源**，`validate_spec` / 设计器现取。

**基数约定（记录 ↔ 图像）**：**记录数不随图片数膨胀**——一条记录需要多张图时，以「图集」（ndarray
列表）放进该记录的 `image` 旁路，**不拆记录**；下游自行处理单图 / 图集（`records.image_of`＝单图语义，
图集取首张、兼容既有调用；`records.image_set_of`＝集合语义，统一返回 list）。例：`target_crop.crops`＝
记录 dict 列表（**每对象一条**，`image` 键存该对象子图）；VLM **按类报送**时，**一条记录携带该类命中
对象的图集**，由送审端 `image_pick` 决定送 1 张还是 N 张。标注仍是一条记录一张图多框。

**模块自描述（`INPUTS` / `OUTPUTS`）**：每个模块用类属性声明自己消费 / 产出的端口及类型
（端口名 → 类型），供装配器、校验器、设计器读取：

```python
class Annotator:
    INPUTS  = {"image": "Image", "dets": "Records"}   # 消费：一张图 + 一批记录
    OUTPUTS = {"annotated": "Image"}                  # 产出一张带框图
```

**spec 声明连线（阶段可选字段 `inputs`）**：`{"端口名": "上游产物名"}`，**每个端口只接一个**上游产物：

```json
{"module_id": "annotate", "params": {...}, "inputs": {"dets": "det_filter.dets"}}
```

- **不写 `inputs` = 现状**（按壳固定流水取数），**零回归**；只在"取上游另一部分 / 绕过固定流水的一环"时才写。
- 同一产物**可被多个下游引用**（为分支留路）；但**一个端口不接多源**（`inputs` 不做列表 / 多值）。

**校验（`validate_spec`，非法连线 fail fast 报错）**：

- 引用的产物名必须存在（在 `ARTIFACT_TYPES`）→ 否则**报错**；
- 端口必须在该模块 `INPUTS` 声明中 → 非法端口**报错**；
- 端口类型须匹配（`Image` 端口不得喂 `Records`）→ 不符**报错**；
- 引用必须是**上游**阶段产物（不能引用下游 / 自身）；值须为字符串（每端口单一上游）→ 违反**报错**。

**可连线端口清单（`RESOLVED_PORTS`，`pipe.composer` 唯一来源；设计器 / `validate_spec` 据此开放）**：

- `yolo.image`；`det_filter.dets`；`seq_filter.dets`；
- `target_crop.frame` / `target_crop.dets`；`annotate.image` / `annotate.dets`；
- **已纳入** `plate_recog.frame` / `plate_recog.crops`、`face_detect.frame` / `face_detect.crops`
  （消费上游裁剪记录）；
- `vlm.dets`（**仅此一个端口**）——决定「**报送哪些检测**」：`yolo.dets`＝筛选前全部检出 /
  `det_filter.dets`＝筛选后（在驱动层解析，见下）。

> **`vlm.inputs.image` 不支持**——送审图片源统一由 `vlm.image_source`（选哪张）+ `image_pick`（送几张）
> 管，避免"连线 + 参数"双机制；校验器对其**报错**并提示改用 `image_source`。

**未纳入 `inputs` 解析的模块 = 本期为空**（所有声明了 `INPUTS` 的模块端口均已解析，含 `plate_recog` /
`face_detect` / `vlm.dets`）。写了 `inputs` 仍按默认取数并给**警告**的机制**保留**给未来新增模块：
推导名单 `CHAIN_UNRESOLVED_INPUT_MODULES`（`web_lab/server.py`）会自动纳入"声明了 `INPUTS` 但未同步
解析"者；新模块接入时同步 `RESOLVED_PORTS` 即可（见 `03-patterns-and-extension.md`）。

> **与参数的关系**：连线只决定"从上游取哪部分"，**不改变模块自身参数**；参数大全以 `list_modules` 现取为准，
> 模块 I/O 字段级细节见 `doc/modules/INDEX.md`。新增模块如何被纳入见 `03-patterns-and-extension.md`。

## 4. 标准壳（照抄填充）

以「警用无人机」等价形态为例（片段）：

```json
{
  "name": "示例·人车+VLM+事件尾",
  "description": "YOLO 识人车 → 有效性 → 标注 → VLM → 事件尾（词表+冷却）",
  "streams": 1,
  "stages": [
    {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 15, "backpressure": "standard"}},
    {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.6, "imgsz": 1280, "classes": "0,2", "device": ""}},
    {"module_id": "det_filter", "params": {"filter": "", "group_limits": "0:2,300;2:1,300"}},
    {"module_id": "annotate", "params": {"thickness": 2, "show_label": true}},
    {"module_id": "vlm", "params": {"max_resolution": "1280,720", "prompt": "<警情四分类提示词>", "use_real_vlm": true}},
    {"module_id": "seq_filter", "params": {
        "throttle": "cls_id:3.0",
        "rules": "[{\"match\": \"label in 治安类|交警类|群体性事件类|救援救助类\", \"event\": \"{label}\", \"key\": \"label\", \"cooldown\": 30.0}]"
    }}
  ]
}
```

## 5. 装配核对清单（逐条自检，全 ✔ 才算完成）

- [ ] `stages` 非空，`frame_manager` 在首位且仅一段。
- [ ] 无重复 `module_id`（一条链路每模块最多一次）。
- [ ] 终段 ∈ {det_filter, seq_filter, plate_recog, face_detect, face_embed}。
- [ ] 每个 `params` 键都在 `list_modules` 白名单内、类型相符、取值在允许范围（select 只取给定选项）。
- [ ] **有效性**（置信 / 尺寸 / 类别数量）→ 用 `det_filter`；不要用 yolo 旧字段
      `filter_conf/min_short/min_long/class_limits`（已删除）。
- [ ] **节流 / track 变化** → 用 `seq_filter` 的 `throttle`/`change`；不要用
      `check_interval`/`track_change_only`（已删除）。
- [ ] **输出事件** → 终段 `seq_filter.rules`（窗口 / 冷却 / 词表 / 动态事件名）或
      `det_filter.events`；VLM 结果由运行器 `publish()` 送入事件尾。**不要用 alarm/OutputPolicy**。
- [ ] VLM 词表：`seq_filter.rules[].match`（如 `label in 治安类|交警类`）必须与 `vlm.prompt` 的输出词对齐。
- [ ] VLM 送审素材：`image_source` 选**哪张**（缺省 `annotated`＝零回归；选 `crops` 须链路含 `target_crop`，
      选 `annotated` 建议含 `annotate`），`image_pick` 选**几张**（缺省 `first`＝只送首张；`all`＝图集全部）；
      缺配套仅警告 / 回退（见 §2）。
- [ ] VLM 报送哪些检测：需要时用 `vlm.inputs.dets`（`yolo.dets`＝筛选前 / `det_filter.dets`＝筛选后）；
      **`vlm.inputs.image` 不支持**，图片源一律用 `image_source`。
- [ ] 车牌链：`yolo.classes="2"` → `det_filter` → `target_crop(classes="2")` → `plate_recog`
      → 终段 `seq_filter(rules: kind==plate, cooldown<0)`。
- [ ] 人脸链：`yolo.classes="0"` → `det_filter` → `target_crop(classes="0")` → `face_detect`（终段）。
- [ ] 写了 `inputs` 连线时（§3）：产物在上游、类型匹配、端口已在模块 `INPUTS` 声明；**不写=现状**，
      非必要不连；本期可连线端口见 §3 清单（`yolo/det_filter/seq_filter/target_crop/annotate` +
      **`plate_recog` / `face_detect`**；`vlm` 仅 `dets`）。
- [ ] 用 `validate_spec` 工具自检，`0 error` 才算过。

## 6. 红线（旧概念，出现即错）

| 旧概念（禁用） | 现状 |
|---|---|
| `alarm` 阶段 / `AlarmPolicy` / `OutputPolicy` | **已删除**；输出改事件尾 |
| `on_vlm_result` / `on_dwell` / `AlarmEvent` | 已删除；改 `publish()` + `events` |
| yolo 参数 `filter_conf/min_short/min_long/class_limits` | 已删除；改 `det_filter` |
| yolo/pipeline 参数 `check_interval/track_change_only` | 已删除；改 `seq_filter.throttle/change` |
| `compose()` 四模式 `small_only/small_crop/small_full/large_only` | 仅代码层兼容 / 回归用，**新链路不用** |
| `target_actions` 词表 | 改 `seq_filter.rules[].match` |
