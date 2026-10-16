# build-face-identity  (build)

> 场景：在视频监控里找出特定的几个人是谁，我有他们的照片底库。
> 关注点：人脸身份：裁人→检脸→嵌入→底库
> 采集：2026-10-13 15:41

## 总装工（architect）
_ok=True steps=2 tools=['list_modules']_

为了在视频监控中识别特定的几个人，我们可以设计一个链路，包括以下模块：

1. **帧管理（FrameManager）**：用于控制视频帧的抽取节奏，例如每隔3秒抽取一帧。
2. **小模型识别（YOLODetector）**：用于识别视频帧中的人脸。
3. **检测筛选（DetFilter）**：用于筛选出人脸检测模块识别出的人脸。
4. **时序判定（SeqFilter）**：用于控制检测频率，例如每3秒检测一次。
5. **目标裁剪（TargetCrop）**：用于裁剪出人脸区域。
6. **特征提取（FaceEmbedder）**：用于提取人脸特征。
7. **向量底库（FaceStore）**：用于存储人脸特征库和进行人脸识别。

以下是链路设计的示例：

```json
{
  "intent": "build",
  "chain_name": "特定人物识别",
  "description": "识别特定人物",
  "stages": [
    {"id": 1, "module": "frame_manager", "description": "每隔3秒抽取一帧"},
    {"id": 2, "module": "yolo", "description": "识别视频帧中的人脸"},
    {"id": 3, "module": "det_filter", "description": "筛选出人脸"},
    {"id": 4, "module": "seq_filter", "description": "每3秒检测一次"},
    {"id": 5, "module": "target_crop", "description": "裁剪出人脸区域"},
    {"id": 6, "module": "face_embed", "description": "提取人脸特征"},
    {"id": 7, "module": "face_store", "description": "存储人脸特征库和进行人脸识别"}
  ],
  "notes": [],
  "constraints": {}
}
```

请注意，这只是一个初步的设计，具体的参数设置和优化需要根据实际情况进行调整。如果您需要进一步的帮助，请告诉我。

## 组装工（assembler · plan 模式）
_ok=True steps=1 tools=[]_

若落地，我会：

1) **需求解析**：分析总装工提供的需求/语义链条，提取关键信息，包括视频监控场景、识别特定人物的需求、已提供的模块和参数等。

2) **选参数**：
   - **帧管理（FrameManager）**：根据需求，每隔3秒抽取一帧，设置参数 `interval_sec` 为 3.0。
   - **小模型识别（YOLODetector）**：使用默认参数，确保能够识别视频帧中的人脸。
   - **检测筛选（DetFilter）**：使用默认参数，筛选出人脸检测模块识别出的人脸。
   - **时序判定（SeqFilter）**：根据需求，每3秒检测一次，设置参数 `throttle` 为 3.0。
   - **目标裁剪（TargetCrop）**：使用默认参数，裁剪出人脸区域。
   - **特征提取（FaceEmbedder）**：使用默认参数，提取人脸特征。
   - **向量底库（FaceStore）**：使用默认参数，存储人脸特征库和进行人脸识别。

3) **写 spec**：根据选定的参数，构建符合链路壳标准的 spec，确保每个模块的参数键严格使用白名单。

4) **自检**：使用 `validate_spec` 函数校验 spec 是否满足硬性规则和参数白名单，确保没有错误。

5) **估算**：使用 `estimate` 函数对 spec 进行资源估算，包括单链路和多路的情况，以及根据显存预算反推建议。

6) **测试（可选）**：根据需要，进行断言式测试和限帧真实推理烟测，以确保链路能够按预期工作。

7) **汇报**：整理方案、spec、校验结果、资源报告和测试结果，形成完整的交付包。
