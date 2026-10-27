# 车牌识别链路 PIPE0002

> 类型：模块自定义拼接链路 | 装配：`web_lab/server.py`（`_chain_from_spec` 装配，
> `_seed_pipeline` 播种） | 运行：`web_lab/test.html` 链路测试台
> 数据库：pipelines 表 `PIPE0002` + pipeline_steps 7 步（引用 7 条模块 preset）

## 1. 链路定位

保留链路之一（车牌识别与存证）：**帧管理 → YOLO 识车 → 检测筛选 → 门控 → 目标裁剪(裁车) →
车牌识别（裁车牌→识别→坐标还原→原分辨率出牌）→ 时序判定事件尾**。
面向道路车辆的中文车牌识别：整链由模块 preset 拼装而成，无链路专属代码分支；
输出为「按车牌号去重」的事件（同车牌只出一次）。链路**无 VLM 阶段**，不做报送。

各阶段参数细节见模块分册（[帧管理](../modules/frame_manager.md)、[YOLO 识别](../modules/yolo.md)、
[检测筛选](../modules/det_filter.md)、[门控](../modules/gate_filter.md)、
[目标裁剪](../modules/target_crop.md)、[车牌识别](../modules/plate_recog.md)、
[时序判定](../modules/seq_filter.md)）。

## 2. 拼接结构（数据库存储）

```
PIPE0002 车牌识别链路（pipelines 表，streams=1）
 ├─ 步骤0  FM0002    车牌识别·帧管理     （frame_skip=5）
 ├─ 步骤1  YOLO0002  车牌识别·YOLO识车   （classes=2，只识车，纯推理）
 ├─ 步骤2  DF0002    车牌识别·检测筛选   （long>=40；group_limits 车≥1）
 ├─ 步骤3  gate_filter「车牌识别·门控」    （key=track_id，interval=5.0；匿名桶统一受门控）
 ├─ 步骤4  CROP0002  车牌识别·目标裁剪   （out_size="×2", margin=0.1, classes=2，裁车放大）
 ├─ 步骤5  PLATE0002 车牌识别·车牌识别   （从车图裁车牌 + HyperLPR3 识别；track_dedup=false，去重已移交外部门控）
 └─ 步骤6  SEQ0002   车牌识别·时序判定   （链路尾：rules kind==plate + cooldown<0，同车牌号一次）
```

| 步骤 | preset | 阶段参数要点 |
|---|---|---|
| frame_manager | FM0002「车牌识别·帧管理」 | `frame_skip=5`（比警用链路密，兼顾远角小牌），随抽帧节奏 |
| yolo | YOLO0002「车牌识别·YOLO识车」 | `classes=2`，只识车；为车牌识别提供车辆框（纯推理） |
| det_filter | DF0002「车牌识别·检测筛选」 | `filter="long >= 40"`（车长边≥40 才有效，小目标视为未识别） |
| gate_filter | 「车牌识别·门控」 | `key=track_id, interval=5.0, filter=""`（同车每 5s 最多识别一次；track_id=0/缺失归匿名桶 `*` 统一受门控——修 INB-0089 tid=0 绕过缺口） |
| target_crop | CROP0002「车牌识别·目标裁剪」 | 按车辆框裁车放大 `out_size="×2"`、`margin=0.1`、`classes=2` |
| plate_recog | PLATE0002「车牌识别·车牌识别」 | `mode=crop`（消费车图）、`track_dedup=false`（内置去重已关，`track_cooldown=5.0` 保留作回切开关——`track_dedup=true` 即回内置门控） |
| seq_filter | SEQ0002「车牌识别·时序判定」 | 事件尾：`rules` = `match:kind == plate` + `key:label` + `cooldown:-1`（同车牌号只出一次） |

链路只存"步骤引用哪些 preset + 顺序"（pipeline_steps），模块参数本体在各模块参数表；
每模块恰一份参数、链路恰一条，均由服务器启动幂等播种（`web_lab/server.py` `_SEED_PLATE_STAGES`；
门控 preset「车牌识别·门控」的业务 ID 由播种定，WO-0024）。
规则与时序门控的完整字段见 [时序判定](../modules/seq_filter.md) §2；独立门控阶段见
[门控](../modules/gate_filter.md)。

## 3. 数据流（逐阶段记录怎么变）

```
视频帧 ──FM0002 要图决策（每 5 帧取 1，积压自适应放宽）
            │ 要图的帧（帧记录：frame_idx/ts）
            ▼
        YOLO0002 纯推理 → dict 记录列表（kind=det；bbox/score/cls_id=2/track_id）
            │
            ▼
        DF0002 检测筛选（车长边≥40 才有效；不达标视为"未识别"，原元素透传）
            │ 有效车辆检测（kind=det 记录）
            ▼
        门控 gate_filter（key=track_id，interval=5.0：同车每 5s 最多放行一次）
            │  track_id=0/缺失 → 匿名桶 "*" 统一受门控（track 未建立帧不再绕过，修 INB-0089）
            │  被拦记录物理过滤（不传下游、零标记字段）；统计旁路：Pipeline.last_gate
            │  ＋产物 gate_filter.stats（blocked_detail 按桶分键计数）
            ▼
        CROP0002 目标裁剪（按车辆框裁车 + 放大 2×）→ CropResult 列表
            │ （旁路字段 img=车图；bbox/offset/scale/cls_id/score/track_id 入主干）
            ▼
        PLATE0002 车牌识别（PlateStage.run(frame, crops)）
            │ 子图内裁车牌 → HyperLPR3 识别 → 车牌框还原回原图坐标
            │ → 从原帧裁原分辨率车牌小图（旁路字段 plate_img）
            │ 产出 PlateResult（kind=plate；label=车牌号）
            ▼
        车牌记录经事件尾 publish → SEQ0002 事件尾
            （rules：kind==plate，按 label=车牌号 分组，cooldown<0 = 同车牌号只触发一次）
            │
            ▼
        链路输出 = events（末段记录携带的事件，seq_filter 追加）
```

关键机制：

- **track 去重（已换装独立门控阶段，WO-0024 转正）**：`Pipeline.track_needed()` 检测到车牌阶段
  或 `key=track_id` 的门控阶段即让上游 `yolo.infer(track=True)`，YOLO 的 `track_id` 随目标裁剪
  产物透传。默认门控 = 独立阶段 `gate_filter`（key=track_id, interval=5.0）——同一辆车 5 秒内
  只识别一次；track 建立初期/闪断帧的 `track_id=0` 归匿名桶统一受门控（修内置版 tid=0 绕过
  缺口，INB-0089），且拦截统计可见（时间线「门控 放行 N / 拦截 M」）。
  **回切开关**：`plate_recog.track_dedup=true` 即回内置门控（`PlateStage` 内置 `IntervalGate`，
  key=track_id 冷却 `track_cooldown` 秒）——内置机制保留未删；两版对比与收编路线见
  [门控阶段设计稿](../design/gate_filter.md) §4。
- **已知行为（有意保留）**：同车门控冷却到期重试时若读出**异号**（识别抖动，如
  沪DHH230/沪OHH236），事件尾按车牌号去重会出多条事件——同 track_id 的多个车牌输出为
  **有意保留**（用户拍板 2026-10-26），后续作单独数据处理。
- **坐标还原**：车牌框在车辆子图坐标系里检出，经 `CropResult.to_original()` 还原到原图
  （除以 scale 加 offset）；原分辨率车牌小图由 `Cropper.extract` 从**原帧**直接裁出（零缩放）。
- **旁路字段**：车图（crop `img`，记录化后统一键 `image`）与车牌小图（`plate_img`，记录化后
  `image`）是图像旁路，不进 JSON 主干；主干只带文本/坐标/score（记录契约见 `umvp/pipe/records.py`）。
- **端口与连线（P5）**：本链路未显式声明 `inputs`，各阶段按现状默认取数（零回归）——产物名
  `frame_manager.frame` → `yolo.dets` → `det_filter.dets` → `gate_filter.dets` →
  `target_crop.crops` → `plate_recog.plates` → `seq_filter.dets`，逐帧登记在
  `Pipeline.artifacts`（另有观测产物 `gate_filter.stats`，非连线端口，时间线/诊断读）。
  `plate_recog.crops` 等端口已开放显式连线（`inputs`），默认不写即现状。链路尾
  `Pipeline.last_events` ＝末段记录携带的事件汇总；门控拦截计数另见 `Pipeline.last_gate`。

## 4. 运行方式（test.html 链路测试台）

打开 `http://127.0.0.1:8001/test.html` → 视频测试页：
选视频 + 选链路（PIPE0002）→ 运行。表单参数（`/api/test/pipeline_run`）同三条链路通用
（video/image、max_frames、start_time/end_time、output_dir/image_format、max_saved_images 等；
本链路无 `vlm_feedback` 语义）。命令行等价：

```bash
curl -s -X POST http://127.0.0.1:8001/api/test/pipeline_run \
  -H "Content-Type: application/json" \
  -d '{"params":{"pipeline_id":"PIPE0002","video":"车牌识别.MP4","max_frames":200}}'
```

运行日志按「阶段→下一级」通用渲染，如 `[YOLO识别→车牌识别] 传递车辆框 N 个`、
`[车牌识别→时序判定] 事件 粤ACJ0710 …`；结果头含「检出 N 个（车辆框）｜车牌 N 次」。
链路无 VLM 阶段，日志"报送 0 条"属正常。

## 5. 输出

- **事件（链路输出）**：车牌记录（`kind=plate`，`label`=车牌号）经 `SeqFilter` 规则出事件，
  按车牌号 `cooldown<0` 去重——同一车牌号只出一次。末段记录的事件汇总见 `Pipeline.last_events`。
- **timeline 结构化时间线**：逐条 `{kind, text, data, ts, frame}`（帧决策/传递/事件/过程信息）。
- **结果图网格**：关键帧、带框标注图（车牌框）；`Pipeline.last_plates` 供驱动层读取。
- **落盘**：output_dir 下关键帧与结果图；车牌小图（原分辨率 `plate_img`）可作存证素材。

## 6. 验证

```bash
# 链路核心逻辑（整帧识别 + 裁车放大→识别→坐标还原 + 原分辨率提取 + 装配）
conda run -n ai python tests/test_plate_recog_pipeline.py

# 阶段回归（纯逻辑，秒级）
conda run -n ai python tests/test_det_filter.py
conda run -n ai python tests/test_target_crop.py
conda run -n ai python tests/test_seq_filter.py
```

（断言数不写死，以脚本运行自报为准。）
注意：onnxruntime 的 `VerifyOutputSizes` / `Expected shape` 形状警告为 SCRFD/HyperLPR3
动态尺寸正常现象。
