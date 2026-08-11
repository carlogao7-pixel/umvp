# 常见坑

> 层：L4-experience ｜ 读者：总装 / 组装工 ｜ 状态：先列已知条目，随实践补充

- onnxruntime 形状警告（VerifyOutputSizes / Duplicate provider）不影响结果
- check_interval 节流：同 ts 连续报送会被抑制
- 坐标还原：缩放 + 偏移；放大图坐标不能直接存图
- 设备选择独立：YOLO(ultralytics 自动) vs SCRFD(onnxruntime CPU) 别混为一谈
- 重叠框去重：同一目标进多路需 IoU 去重
- VLM 提示词词表必须对齐 target_actions（否则命中率归零）
