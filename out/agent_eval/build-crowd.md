# build-crowd  (build)

> 场景：广场上人突然聚集很多，要提示。
> 关注点：人群聚集（成组数量 + 确认）
> 采集：2026-10-13 15:41

## 总装工（architect）
_ok=True steps=1 tools=[]_

好的，您希望系统对广场上人突然聚集很多的情况进行提示。为了更好地设计这个链路，我需要确认几个细节：

1. **触发条件**：系统是如何判断“人突然聚集很多”的？是依靠视频分析还是其他方式？
2. **提示方式**：您希望系统通过什么方式发出提示？例如，是通过手机APP、短信还是其他方式？
3. **其他需求**：是否有其他需求，比如对聚集人数的下限要求，或者对提示时间的延迟要求？

请您提供这些信息，以便我为您设计合适的链路。

## 组装工（assembler · plan 模式）
_ok=True steps=1 tools=[]_

若落地，我会：

1) **需求解析**：从总装工提供的需求/语义链条中提取关键信息，包括场景（广场上人聚集）、触发条件（人聚集）、提示方式（未指定）和其他需求（未指定）。
2) **参数选择**：根据提取的信息，从 `list_modules` 中选择合适的模块和参数。例如，选择 `yolo` 模块进行目标检测，并设置相关参数以检测人群聚集。
3) **spec 编写**：根据选定的模块和参数，编写 `spec` 文件，包括 `stages` 和 `params`。例如：
    ```json
    {
        "name": "广场人群聚集提示",
        "stages": [
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
                    "events": "聚集"
                }
            },
            {
                "module_id": "seq_filter",
                "params": {
                    "throttle": "",
                    "change": "",
                    "rules": "聚集"
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
                "module_id": "annotate",
                "params": {
                    "thickness": 2,
                    "show_label": True,
                    "classes": "",
                    "filter": ""
                }
            },
            {
                "module_id": "vlm",
                "params": {
                    "max_resolution": "1280,720",
                    "prompt": "广场上人聚集很多，请及时处理。",
                    "use_real_vlm": False,
                    "vlm_endpoint": "http://117.42.21.253:8000/v1/chat/completions",
                    "vlm_model": "qwen3-vl-32b",
                    "vlm_key": ""
                }
            }
        ]
    }
    ```
4) **自检**：使用 `validate_spec` 函数校验 `spec` 文件，确保其满足硬性规则和参数白名单。
5) **估算**：使用 `estimate` 函数对 `spec` 文件进行资源估算，包括单链路和多路资源消耗。
6) **测试（可选）**：根据需要进行断言式测试或限帧真实推理烟测，以确保链路功能正常。
7) **汇报**：将方案、`spec` 文件、校验结果、资源报告和测试结果（如有）汇报给总装工。
