# build-drone-far  (build)

> 场景：无人机高空俯拍，画面里远处的小目标和车辆，要尽量不漏。
> 关注点：远角小目标：imgsz/conf + 无人机场景
> 采集：2026-10-13 15:41

## 总装工（architect）
_ok=True steps=1 tools=[]_

您好！根据您的需求，您希望无人机高空俯拍时，画面中能尽量不漏掉远处的小目标和车辆。以下是我为您整理的需求：

**需求**：
- 无人机高空俯拍
- 画面中包含远处的小目标和车辆
- 尽量不漏掉目标

**约束**：
- 未提及具体距离和分辨率要求

请问以上需求是否准确？如果需要进一步细化，请告诉我。接下来，我将根据您的需求设计链路雏形。

## 组装工（assembler · plan 模式）
_ok=True steps=1 tools=[]_

若落地，我会：

1) **需求解析**：从总装工提供的需求链条中提取关键信息，包括场景（无人机高空俯拍）、目标（远处的小目标和车辆）、以及约束条件（未提及具体距离和分辨率要求）。
2) **选参数**：根据提取的需求信息，结合 `list_modules` 获取的模块参数信息，初步选择合适的模块和参数。例如，可能需要使用 `frame_manager` 来控制帧率，`yolo` 来检测小目标和车辆，以及 `target_crop` 来裁剪图像以减少漏检。
3) **写 spec**：根据选定的模块和参数，构建 `stages spec`。spec 将包含模块 ID 和相应的参数值，例如：
    ```json
    {
        "name": "drone_surveillance",
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
                "module_id": "target_crop",
                "params": {
                    "out_size": "1024,768",
                    "margin": 0.0,
                    "classes": "",
                    "filter": ""
                }
            }
        ]
    }
    ```
4) **自检**：使用 `validate_spec` 校验构建的 spec 是否满足硬性规则和参数白名单。如果有错误，将根据错误信息进行修改，直到 `validate_spec` 返回 `0 error`。
5) **估算**：使用 `estimate` 函数对构建的 spec 进行资源估算，包括单链路和多路情况下的显存、内存和帧率等资源消耗。
6) **测试（可选）**：根据需要，可以编写断言式测试或进行限帧真实推理烟测，以验证链路的功能和性能。
7) **汇报**：将方案、spec、校验结果、资源报告和测试结果（如有）整理成文档，并提交给总装工。
