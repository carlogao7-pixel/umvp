# 链路壳标准（唯一权威）

> 读者：组装工 ｜ **本文件是链路构建的唯一权威规范**。组装工的产出 = 符合本规范的 `stages spec`；
> 装配与运行由平台完成，**不手写"拼接代码"**。照抄 §3 模板、过 §4 清单即算合格。

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
> annotate / plate / vlm / event_filter` 各一个字段）；重复 `module_id` 无法表达（装配器只取第一份参数），
> 故在 `validate_chain` 与 `db.save_pipeline` 处直接拒绝，避免静默丢弃。

## 1. 标准产物

- **必产**：`stages spec`（§0 的结构）。
- **可选**（仅独立脚本 / 离线验证时）：驱动壳、断言式测试壳。
- **禁止**：新建"拼接 / 胶水模块"、改核心模块（走升级通道）；用已删除 / 改名前的旧概念（见 §5）。

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
| `vlm` | 大模型研判（素材 + 推理） | max_resolution、prompt、use_real_vlm、vlm_endpoint/model/key |
| `plate_recog` | 车牌识别（消费车图） | detect_level、mode、margin、dedup_iou、track_dedup、track_cooldown |
| `face_detect` | 人脸检测（消费人图） | device、det_thresh、det_size、max_num、use_crop、margin、dedup_iou |
| `face_embed` | 人脸嵌入 | device |
| `face_store` | 向量底库检索 | db_path、thresh、topk、device |

**条件 DSL**（`det_filter.filter` / `target_crop.filter` / `annotate.filter`）：`字段 操作符 值`，
组内逗号 = 并且、组间分号 = 或者、`not` = 非、`()` 分组；字段 `cls_id/类别`、`track_id/track`、
`score/conf/置信度`、派生 `w/h/short/长边/long/area`、注入 `ts/frame_idx`；
聚合 `count(字段)/count(子条件)/count()`。

**事件规格**：
- `det_filter.events` = `[{"name": 事件名, "when": 条件DSL(可选), "detail": [字段名...]}]`（多事件）。
- `seq_filter.rules` = `[{"match": 条件DSL, "event": 事件名(支持 `{字段}` 动态), "key": 分组键,
  "span"/"window_n"/"min_n"/"hit_ratio"/"cooldown"}]`；`cooldown<0` = 同对象一次。

## 3. 标准壳（照抄填充）

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

## 4. 装配核对清单（逐条自检，全 ✔ 才算完成）

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
- [ ] 车牌链：`yolo.classes="2"` → `det_filter` → `target_crop(classes="2")` → `plate_recog`
      → 终段 `seq_filter(rules: kind==plate, cooldown<0)`。
- [ ] 人脸链：`yolo.classes="0"` → `det_filter` → `target_crop(classes="0")` → `face_detect`（终段）。
- [ ] 用 `validate_spec` 工具自检，`0 error` 才算过。

## 5. 红线（旧概念，出现即错）

| 旧概念（禁用） | 现状 |
|---|---|
| `alarm` 阶段 / `AlarmPolicy` / `OutputPolicy` | **已删除**；输出改事件尾 |
| `on_vlm_result` / `on_dwell` / `AlarmEvent` | 已删除；改 `publish()` + `events` |
| yolo 参数 `filter_conf/min_short/min_long/class_limits` | 已删除；改 `det_filter` |
| yolo/pipeline 参数 `check_interval/track_change_only` | 已删除；改 `seq_filter.throttle/change` |
| `compose()` 四模式 `small_only/small_crop/small_full/large_only` | 仅代码层兼容 / 回归用，**新链路不用** |
| `target_actions` 词表 | 改 `seq_filter.rules[].match` |
