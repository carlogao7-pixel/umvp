# 环境与验证命令

> 层：L1-shared ｜ 读者：总装 / 组装工

## 环境

- 唯一解释器：conda 环境 `ai`（命令写 `conda run -n ai python ...`，或先 activate 后用 `python`）。
- 依赖已齐：cv2 / ultralytics / insightface / onnxruntime。零新依赖是组装工的硬边界。

## 回归集

| 命令 | 内容 | 耗时 |
|---|---|---|
| `conda run -n ai python umvp/pipe/test_composer.py` | 纯逻辑 27 断言 | 秒级 |
| `conda run -n ai python umvp/face_embed/test_face_embed.py --register-dir umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz` | 人脸注册 + 自检 7/7 | 秒级 |
| `conda run -n ai python tests/test_yolo_params.py` | YOLO 参数对比（真实推理，yolov8n.pt） | 约 1 分钟 |
| `conda run -n ai python tests/test_police_uav_pipeline.py` | 唯一保留链路（警用无人机）逻辑断言（不加载模型） | 秒级 |
| `conda run -n ai python tests/test_police_uav_video.py` | 警用链路端到端（真实 YOLO + mock VLM） | 约 1 分钟 |
| `conda run -n ai python umvp/resources.py` | 资源估算自检（不联网） | 秒级 |
| `conda run -n ai python tests/test_face_extract_pipeline.py` | 人脸提取链路 27 断言 | 秒级 |

## 标定

- `conda run -n ai python web_lab/calibrate.py` → 遍历 models/*.pt 生成 out/fingerprints.json，**换机必跑**。
- 估算依赖本机指纹，机器变化（换卡 / 换机）只需重跑标定，代码零改动。

## 注意事项

- onnxruntime 会打印大量 `VerifyOutputSizes` / `Expected shape` 形状警告（SCRFD 动态 batch
  正常现象）和 `Duplicate provider` 警告，均不影响结果，验证时可用 `grep -v` 过滤。
- 真实推理 / 长任务前告知预计耗时（YOLO 秒级、端到端链路分钟级）。
