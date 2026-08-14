# 常见坑

> 层：L4-experience ｜ 读者：总装 / 组装工 ｜ 状态：先列已知条目，随实践补充

- onnxruntime 形状警告（VerifyOutputSizes / Duplicate provider）不影响结果
- check_interval 节流：同 ts 连续报送会被抑制
- 坐标还原：缩放 + 偏移；放大图坐标不能直接存图
- 设备选择独立：YOLO(ultralytics 自动) vs SCRFD(onnxruntime CPU) 别混为一谈
- 重叠框去重：同一目标进多路需 IoU 去重
- VLM 提示词词表必须对齐 target_actions（否则命中率归零）
- 检出多 ≠ 跟踪好：640 下小目标弱框帧间不可关联，conf 再低也是 0 track（人群1 实测）；验收看 track 数
- 远角小目标 conf 0.50 直接丢目标，0.15 才能找回（远角车+人实测）
- 极端低阈值双阈值组合（conf 0.05 + filter_conf 0.35）无实用价值：弱检测全被下限滤掉（v1 教训，已移除）
- iou 0.9 弱 NMS 保留的是重复/噪点框，不等于多检目标；iou 维度只在重叠素材有区分度
