# 挂载 Agent 全流程记录（需求 → 总装工 → 组装工 → 链路 spec）

> 生成时间：2026-10-14 12:32 ｜ 模型：glm-4-flash
>
> ⚠️ **本文件由脚本自动生成，勿手改**；更新请重跑 `web_lab/agent_run.py`。
>
> 本记录由 `web_lab/agent_run.py` 真跑生成：包含**完整对话、工具调用记录、最终产出**。
> 两角色 system 提示词（静态）见 `doc/agent/static_prompts.md`，本文不重复。
> 运行顺序由壳固定，stages 数组顺序仅供声明 / 存储 / 展示（见组装工 §3）。

## 需求（用户输入）

小区门口单路摄像头识别进出车辆的车牌，1080p，8G 显存，把识别结果存下来。

---

## 第一阶段：总装工（architect）

> 角色 `architect` ｜ ok=True ｜ 轮数 steps=1 ｜ 工具=[]

**用户**

小区门口单路摄像头识别进出车辆的车牌，1080p，8G 显存，把识别结果存下来。

**助手**

好的，您需要构建一个识别进出车辆车牌的链路，以下是设计思路：

1. **帧管理**：单路摄像头，1080p 分辨率，8G 显存，按需抽帧。
2. **车牌识别**：识别进出车辆的车牌，并将结果存下来。

请问您对车牌识别的精度有要求吗？例如识别准确率、识别速度等。

### 本阶段产出

好的，您需要构建一个识别进出车辆车牌的链路，以下是设计思路：

1. **帧管理**：单路摄像头，1080p 分辨率，8G 显存，按需抽帧。
2. **车牌识别**：识别进出车辆的车牌，并将结果存下来。

请问您对车牌识别的精度有要求吗？例如识别准确率、识别速度等。

---

## 第二阶段：组装工（assembler）

> 角色 `assembler` ｜ ok=True ｜ 轮数 steps=4 ｜ 工具=['list_modules', 'validate_spec', 'estimate']

**用户**

总装工已把用户需求整理为下方语义链条 / 需求。请按你的工作流程落成**可校验的 stages spec**并交付（写 spec → 校验 → 估算 → 汇报）。

【用户原话】小区门口单路摄像头识别进出车辆的车牌，1080p，8G 显存，把识别结果存下来。

【总装产出】
好的，您需要构建一个识别进出车辆车牌的链路，以下是设计思路：

1. **帧管理**：单路摄像头，1080p 分辨率，8G 显存，按需抽帧。
2. **车牌识别**：识别进出车辆的车牌，并将结果存下来。

请问您对车牌识别的精度有要求吗？例如识别准确率、识别速度等。

**→ 工具调用** `list_modules`

```json
{}
```

**← 工具结果**

```
{"modules": {"frame_manager": {"name": "帧管理 FrameManager", "group": "管道四模块", "desc": "复用 pipe.composer.FrameManager：三种取图节奏 + 积压/耗时背压（原 frame_scheduler 并入）。", "params": {"sampling": {"label": "取图节奏", "type": "select", "default": "analysis", "options": ["analysis", "wall_clock", "frame_count"]}, "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3}, "interval_sec": {"label": "墙钟间隔（秒）", "type": "float", "default": 3.0}, "interval_frames": {"label": "帧数间隔（帧）", "type": "int", "default": 30}, "backpressure": {"label": "背压档位", "type": "select", "default": "standard", "options": ["off", "standard", "aggressive"]}}}, "yolo": {"name": "小模型识别 YOLODetector", "group": "管道四模块", "desc": "复用 pipe.composer.YOLODetector：**纯推理**（有效性过滤归「检测筛选」阶段，节流/报送编排在 Pipeline）。", "params": {"model_path": {"label": "权重模型", "type": "file", "default": ""}, "conf": {"label": "检出阈值 conf", "type": "float", "default": 0.35}, "iou": {"label": "NMS 阈值 iou", "type": "float", "default": 0.7}, "imgsz": {"label": "推理分辨率 imgsz", "type": "int", "default": 640}, "max_det": {"label": "单帧最多目标 max_det", "type": "int", "default": 300}, "classes": {"label": "类别白名单（逗号分隔，空=不限）", "type": "str", "default": "0"}, "device": {"label": "推理设备（空=自动）", "type": "str", "default": ""}}}, "det_filter": {"name": "检测筛选 DetFilter", "group": "管道四模块", "desc": "复用 pipe.det_filter.DetFilter：有效性筛选（逐条条件 DSL：范围+与/或/非+派生尺寸量+ 聚合 count(字段)/count(子条件)/count()；+ 成组数量判定）。链路里插在 YOLO 之后；有效性过滤的唯一来源（空条件+空上下限=不筛选）。", "params": {"filter": {"label": "筛选条件（条件 DSL）", "type": "text", "default": ""}, "group_limits": {"label": "每类数量上下限（如 0:1,300; 2:1,10）", "type": "str", "default": ""}, "events": {"label": "事件标注（JSON 列表）", "type": "text", "default": ""}}}, "seq_filter": {"name": "时序判定 SeqFilter", "group": "管道四模块", "desc": "复用 pipe.seq_filter.SeqFilter：时序门控（整类节流 throttle / track 集合变化 change）+ 窗口/单条规则确认与事件标注。链路里插在检测筛选之后；承载原 Pipeline 的 check_interval/track_change_only 语义。", "params": {"throttle": {"label": "整类节流（key:秒，如 cls_id:3.0）", "type": "str", "default": ""}, "change": {"label": "变化门控（key:watch，如 cls_id:track_id）", "type": "str", "default": ""}, "rules": {"label": "确认/事件规则（JSON 列表）", "type": "text", "default": ""}}}, "target_crop": {"name": "目标裁剪 TargetCrop", "group": "管道四模块", "desc": "按上游检测框裁子图（输出分辨率可配，默认原分辨率）；供车牌识别/人脸检测/VLM 消费。不含检测器。", "params": {"out_size": {"label": "输出分辨率（空=原图；640=长边；640,480=严格；×2=倍数）", "type": "str", "default": ""}, "margin": {"label": "裁剪外扩比例", "type": "float", "default": 0.0}, "classes": {"label": "只裁这些类别（逗号分隔，空=全部）", "type": "str", "default": ""}, "filter": {"label": "框筛选条件（空=全裁）", "type": "str", "default": ""}}}, "annotate": {"name": "标注 Annotator", "group": "管道四模块", "desc": "在帧副本上画上游检测框（供 VLM 看“原图+画框”）；纯绘制，无模型。", "params": {"thickness": {"label": "框线粗细（像素）", "type": "int", "default": 2}, "show_label": {"label": "显示类别标签", "type": "bool", "default": true}, "classes": {"label": "只画这些类别（逗号分隔，空=全部）", "type": "str", "default": ""}, "filter": {"label": "框筛选条件（空=全画）", "type": "str", "default": ""}}}, "vlm": {"n
…（截断，原长 5549 字符）
```

**助手**

根据用户需求和总装工的设计思路，以下是我为您构建的 stages spec：

```json
{
  "name": "车辆车牌识别链路",
  "description": "识别进出车辆的车牌并存储结果",
  "streams": 1,
  "stages": [
    {
      "module_id": "frame_manager",
      "params": {
        "sampling": "analysis",
        "frame_skip": 3,
        "interval_sec": 3.0,
        "interval_frames": 30,
        "backpressure": "standard"
      }
    },
    {
      "module_id": "yolo",
      "params": {
        "model_path": "",
        "conf": 0.35,
        "iou": 0.7,
        "imgsz": 640,
        "max_det": 300,
        "classes": "0",
        "device": ""
      }
    },
    {
      "module_id": "det_filter",
      "params": {
        "filter": "",
        "group_limits": "",
        "events": ""
      }
    },
    {
      "module_id": "target_crop",
      "params": {
        "out_size": "",
        "margin": 0.0,
        "classes": "",
        "filter": ""
      }
    },
    {
      "module_id": "plate_recog",
      "params": {
        "detect_level": "high",
        "mode": "crop",
        "margin": 0.1,
        "dedup_iou": 0.6,
        "track_dedup": true,
        "track_cooldown": 5.0
      }
    }
  ]
}
```

接下来，我将使用 `validate_spec` 工具来校验这个 spec 是否满足硬性规则和参数白名单。{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant","content":null,"reasoning_content":null,"reasoning":null,"audio":null,"tool_calls":[{"id":"call_2026101412321653395812ab3e41de_0","index":0,"type":"function","function":{"name":"validate_spec","arguments":"{\"spec\": {\"name\": \"车辆车牌识别链路\", \"description\": \"识别进出车辆的车牌并存储结果\", \"streams\": 1, \"stages\": [{\"module_id\": \"frame_manager\", \"params\": {\"sampling\": \"analysis\", \"frame_skip\": 3, \"interval_sec\": 3.0, \"interval_frames\": 30, \"backpressure\": \"standard\"}}, {\"module_id\": \"yolo\", \"params\": {\"model_path\": \"\", \"conf\": 0.35, \"iou\": 0.7, \"imgsz\": 640, \"max_det\": 300, \"classes\": \"0\", \"device\": \"\"}}, {\"module_id\": \"det_filter\", \"params\": {\"filter\": \"\", \"group_limits\": \"\", \"events\": \"\"}}, {\"module_id\": \"target_crop\", \"params\": {\"out_size\": \"\", \"margin\": 0.0, \"classes\": \"\", \"filter\": \"\"}}, {\"module_id\": \"plate_recog\", \"params\": {\"detect_level\": \"high\", \"mode\": \"crop\", \"margin\": 0.1, \"dedup_iou\": 0.6, \"track_dedup\": true, \"track_cooldown\": 5.0}}]}}","outputs":null},"code_interpreter":null,"retrieval":null,"mcp":null,"drawing_tool":null,"web_browser":null,"search_intent":null,"search_result":null}],"tool_call_id":null,"attachments":null,"metadata":null}}

**→ 工具调用** `validate_spec`

```json
{"spec": {"name": "车辆车牌识别链路", "description": "识别进出车辆的车牌并存储结果", "streams": 1, "stages": [{"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 3, "interval_sec": 3.0, "interval_frames": 30, "backpressure": "standard"}}, {"module_id": "yolo", "params": {"model_path": "", "conf": 0.35, "iou": 0.7, "imgsz": 640, "max_det": 300, "classes": "0", "device": ""}}, {"module_id": "det_filter", "params": {"filter": "", "group_limits": "", "events": ""}}, {"module_id": "target_crop", "params": {"out_size": "", "margin": 0.0, "classes": "", "filter": ""}}, {"module_id": "plate_recog", "params": {"detect_level": "high", "mode": "crop", "margin": 0.1, "dedup_iou": 0.6, "track_dedup": true, "track_cooldown": 5.0}}]}}
```

**← 工具结果**

```
{"ok": true, "errors": []}
```

**→ 工具调用** `estimate`

```json
{"spec": {"name": "车辆车牌识别链路", "description": "识别进出车辆的车牌并存储结果", "streams": 1, "stages": [{"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 3, "interval_sec": 3.0, "interval_frames": 30, "backpressure": "standard"}}, {"module_id": "yolo", "params": {"model_path": "", "conf": 0.35, "iou": 0.7, "imgsz": 640, "max_det": 300, "classes": "0", "device": ""}}, {"module_id": "det_filter", "params": {"filter": "", "group_limits": "", "events": ""}}, {"module_id": "target_crop", "params": {"out_size": "", "margin": 0.0, "classes": "", "filter": ""}}, {"module_id": "plate_recog", "params": {"detect_level": "high", "mode": "crop", "margin": 0.1, "dedup_iou": 0.6, "track_dedup": true, "track_cooldown": 5.0}}]}}
```

**← 工具结果**

```
{"per_module": {"frame_manager": {"vram_mb": 0.0, "ram_mb": 0.0, "latency_ms_cpu": 0.0, "latency_ms_gpu": 0.0, "note": "纯逻辑模块，开销可忽略", "source": "static"}, "yolo": {"vram_mb": 126.7, "ram_mb": 193.0, "latency_ms_cpu": 39.2, "latency_ms_gpu": 12.2, "note": "yolo imgsz=640（ram 静态补值）", "source": "measured"}, "det_filter": {"vram_mb": 0.0, "ram_mb": 0.0, "latency_ms_cpu": 0.0, "latency_ms_gpu": 0.0, "note": "纯逻辑模块，开销可忽略", "source": "static"}, "target_crop": {"vram_mb": 0.0, "ram_mb": 0.0, "latency_ms_cpu": 0.0, "latency_ms_gpu": 0.0, "note": "纯逻辑模块，开销可忽略", "source": "static"}, "plate_recog": {"vram_mb": 0.0, "ram_mb": 0.0, "latency_ms_cpu": 0.0, "latency_ms_gpu": 0.0, "note": "纯逻辑模块，开销可忽略", "source": "static"}}, "vram_mb": 126.7, "ram_mb": 193.0, "latency_per_frame_ms": {"cpu": 39.2, "gpu": 12.2}, "fps_cpu": 25.5, "fps_gpu": 82.0, "streams": 1, "vram_mb_N_streams": 126.7, "warning": [], "note": ""}
```

**助手**

**交付包**：

1. **方案**：
   - 阶段清单：帧管理 → YOLO识别 → 检测筛选 → 目标裁剪 → 车牌识别
   - 理由：先进行帧管理，然后使用YOLO进行目标检测，接着进行检测筛选，再进行目标裁剪，最后进行车牌识别。
   - 参数初值：
     - 帧管理：取图节奏为分析，抽帧间隔为3帧，墙钟间隔为3秒，帧数间隔为30帧，背压档位为标准。
     - YOLO识别：检出阈值conf为0.35，NMS阈值iou为0.7，推理分辨率imgsz为640，单帧最多目标max_det为300，类别白名单为0，推理设备为自动。
     - 检测筛选：筛选条件为空，每类数量上下限为空，事件标注为空。
     - 目标裁剪：输出分辨率为空，裁剪外扩比例为0.0，只裁类别为空，框筛选条件为空。
     - 车牌识别：检测精度档为高，识别方式为裁剪，车牌出图外扩比例为0.1，车牌去重IoU为0.6，按车辆跟踪去重为真，同车重试间隔为5秒。
   - 备选对比：无。
2. **完整spec**：
   ```json
   {
       "name": "车辆车牌识别链路",
       "description": "识别进出车辆的车牌并存储结果",
       "streams": 1,
       "stages": [
           {
               "module_id": "frame_manager",
               "params": {
                   "sampling": "analysis",
                   "frame_skip": 3,
                   "interval_sec": 3.0,
                   "interval_frames": 30,
                   "backpressure": "standard"
               }
           },
           {
               "module_id": "yolo",
               "params": {
                   "model_path": "",
                   "conf": 0.35,
                   "iou": 0.7,
                   "imgsz": 640,
                   "max_det": 300,
                   "classes": "0",
                   "device": ""
               }
           },
           {
               "module_id": "det_filter",
               "params": {
                   "filter": "",
                   "group_limits": "",
                   "events": ""
               }
           },
           {
               "module_id": "target_crop",
               "params": {
                   "out_size": "",
                   "margin": 0.0,
                   "classes": "",
                   "filter": ""
               }
           },
           {
               "module_id": "plate_recog",
               "params": {
                   "detect_level": "high",
                   "mode": "crop",
                   "margin": 0.1,
                   "dedup_iou": 0.6,
                   "track_dedup": true,
                   "track_cooldown": 5.0
               }
           }
       ]
   }
   ```
3. **validate_spec校验结果**：无错误。
4. **资源报告**：
   - 单个模块资源消耗：
     - 帧管理：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - YOLO识别：VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms。
     - 检测筛选：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - 目标裁剪：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - 车牌识别：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
   - 总资源消耗：
     - VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms。
     - 每帧CPU延迟 39.2 ms，每帧GPU延迟 12.2 ms。
     - CPU帧率 25.5 fps，GPU帧率 82.0 fps。
5. **已验证 / 未验证清单**：
   - 已验证：spec校验，资源估算。
   - 未验证：测试。
6. **测试结果**：未做。

**汇报**：

1. 方案：帧管理 → YOLO识别 → 检测筛选 → 目标裁剪 → 车牌识别。
2. spec：
   ```json
   {
       "name": "车辆车牌识别链路",
       "description": "识别进出车辆的车牌并存储结果",
       "streams": 1,
       "stages": [
           {
               "module_id": "frame_manager",
               "params": {
                   "sampling": "analysis",
                   "frame_skip": 3,
                   "interval_sec": 3.0,
                   "interval_frames": 30,
                   "backpressure": "standard"
               }
           },
           {
               "module_id": "yolo",
               "params": {
                   "model_path": "",
                   "conf": 0.35,
                   "iou": 0.7,
                   "imgsz": 640,
                   "max_det": 300,
                   "classes": "0",
                   "device": ""
               }
           },
           {
               "module_id": "det_filter",
               "params": {
                   "filter": "",
                   "group_limits": "",
                   "events": ""
               }
           },
           {
               "module_id": "target_crop",
               "params": {
                   "out_size": "",
                   "margin": 0.0,
                   "classes": "",
                   "filter": ""
               }
           },
           {
               "module_id": "plate_recog",
               "params": {
                   "detect_level": "high",
                   "mode": "crop",
                   "margin": 0.1,
                   "dedup_iou": 0.6,
                   "track_dedup": true,
                   "track_cooldown": 5.0
               }
           }
       ]
   }
   ```
3. 校验结果：无错误。
4. 资源报告：VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms，CPU帧率 25.5 fps，GPU帧率 82.0 fps。
5. 测试结果：未做。
6. 已验证 / 未验证清单：已验证spec校验，资源估算；未验证测试。
7. 下一步建议：进行测试。

### 本阶段产出

**交付包**：

1. **方案**：
   - 阶段清单：帧管理 → YOLO识别 → 检测筛选 → 目标裁剪 → 车牌识别
   - 理由：先进行帧管理，然后使用YOLO进行目标检测，接着进行检测筛选，再进行目标裁剪，最后进行车牌识别。
   - 参数初值：
     - 帧管理：取图节奏为分析，抽帧间隔为3帧，墙钟间隔为3秒，帧数间隔为30帧，背压档位为标准。
     - YOLO识别：检出阈值conf为0.35，NMS阈值iou为0.7，推理分辨率imgsz为640，单帧最多目标max_det为300，类别白名单为0，推理设备为自动。
     - 检测筛选：筛选条件为空，每类数量上下限为空，事件标注为空。
     - 目标裁剪：输出分辨率为空，裁剪外扩比例为0.0，只裁类别为空，框筛选条件为空。
     - 车牌识别：检测精度档为高，识别方式为裁剪，车牌出图外扩比例为0.1，车牌去重IoU为0.6，按车辆跟踪去重为真，同车重试间隔为5秒。
   - 备选对比：无。
2. **完整spec**：
   ```json
   {
       "name": "车辆车牌识别链路",
       "description": "识别进出车辆的车牌并存储结果",
       "streams": 1,
       "stages": [
           {
               "module_id": "frame_manager",
               "params": {
                   "sampling": "analysis",
                   "frame_skip": 3,
                   "interval_sec": 3.0,
                   "interval_frames": 30,
                   "backpressure": "standard"
               }
           },
           {
               "module_id": "yolo",
               "params": {
                   "model_path": "",
                   "conf": 0.35,
                   "iou": 0.7,
                   "imgsz": 640,
                   "max_det": 300,
                   "classes": "0",
                   "device": ""
               }
           },
           {
               "module_id": "det_filter",
               "params": {
                   "filter": "",
                   "group_limits": "",
                   "events": ""
               }
           },
           {
               "module_id": "target_crop",
               "params": {
                   "out_size": "",
                   "margin": 0.0,
                   "classes": "",
                   "filter": ""
               }
           },
           {
               "module_id": "plate_recog",
               "params": {
                   "detect_level": "high",
                   "mode": "crop",
                   "margin": 0.1,
                   "dedup_iou": 0.6,
                   "track_dedup": true,
                   "track_cooldown": 5.0
               }
           }
       ]
   }
   ```
3. **validate_spec校验结果**：无错误。
4. **资源报告**：
   - 单个模块资源消耗：
     - 帧管理：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - YOLO识别：VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms。
     - 检测筛选：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - 目标裁剪：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
     - 车牌识别：VRAM 0.0 MB，RAM 0.0 MB，CPU延迟 0.0 ms，GPU延迟 0.0 ms。
   - 总资源消耗：
     - VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms。
     - 每帧CPU延迟 39.2 ms，每帧GPU延迟 12.2 ms。
     - CPU帧率 25.5 fps，GPU帧率 82.0 fps。
5. **已验证 / 未验证清单**：
   - 已验证：spec校验，资源估算。
   - 未验证：测试。
6. **测试结果**：未做。

**汇报**：

1. 方案：帧管理 → YOLO识别 → 检测筛选 → 目标裁剪 → 车牌识别。
2. spec：
   ```json
   {
       "name": "车辆车牌识别链路",
       "description": "识别进出车辆的车牌并存储结果",
       "streams": 1,
       "stages": [
           {
               "module_id": "frame_manager",
               "params": {
                   "sampling": "analysis",
                   "frame_skip": 3,
                   "interval_sec": 3.0,
                   "interval_frames": 30,
                   "backpressure": "standard"
               }
           },
           {
               "module_id": "yolo",
               "params": {
                   "model_path": "",
                   "conf": 0.35,
                   "iou": 0.7,
                   "imgsz": 640,
                   "max_det": 300,
                   "classes": "0",
                   "device": ""
               }
           },
           {
               "module_id": "det_filter",
               "params": {
                   "filter": "",
                   "group_limits": "",
                   "events": ""
               }
           },
           {
               "module_id": "target_crop",
               "params": {
                   "out_size": "",
                   "margin": 0.0,
                   "classes": "",
                   "filter": ""
               }
           },
           {
               "module_id": "plate_recog",
               "params": {
                   "detect_level": "high",
                   "mode": "crop",
                   "margin": 0.1,
                   "dedup_iou": 0.6,
                   "track_dedup": true,
                   "track_cooldown": 5.0
               }
           }
       ]
   }
   ```
3. 校验结果：无错误。
4. 资源报告：VRAM 126.7 MB，RAM 193.0 MB，CPU延迟 39.2 ms，GPU延迟 12.2 ms，CPU帧率 25.5 fps，GPU帧率 82.0 fps。
5. 测试结果：未做。
6. 已验证 / 未验证清单：已验证spec校验，资源估算；未验证测试。
7. 下一步建议：进行测试。

---
