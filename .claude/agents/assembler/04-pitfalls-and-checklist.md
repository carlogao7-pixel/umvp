# 常见坑 与 已验证 / 未验证清单

> 读者：组装工 ｜ 用途：选参数 / 交付前自查；每次交付把新结论追加进来。

## 常见坑

- onnxruntime 形状警告（`VerifyOutputSizes` / `Duplicate provider`）不影响结果，可忽略。
- **节流**：`seq_filter.throttle`（整类）同 `ts` 连续报送会被抑制；时序门控只管"跨帧"，同帧即时
  命中归 `det_filter`。
- **坐标还原**：缩放 + 偏移；放大图坐标不能直接当原图坐标存图——出原图小图要用坐标还原后的框。
- **设备选择独立**：YOLO（ultralytics 自动）与 SCRFD（onnxruntime CPU）别混为一谈。
- **重叠框去重**：同一目标进多路 / 多框需 IoU 去重（`plate_recog.dedup_iou` 等）。
- **VLM 词表必须对齐事件尾规则** `seq_filter.rules[].match`，否则命中率归零（先定词表再写提示词）。
- **检出多 ≠ 跟踪好**：640 下小目标弱框帧间不可关联，conf 再低也是 0 track（人群1 实测）；
  验收看 track 数。
- **远角小目标** conf 0.50 直接丢目标，0.15 才能找回（远角车+人实测）。
- 极端低阈值双阈值组合（conf 0.05 + 有效性下限）无实用价值：弱检测全被下限滤掉。
- iou 0.9 弱 NMS 保留的是重复 / 噪点框，不等于多检目标；iou 维度只在重叠素材有区分度。
- **旧概念禁用**：`alarm/OutputPolicy`、`on_vlm_result`、`filter_conf/min_short/min_long`、
  `check_interval/track_change_only`、`compose` 四模式（见 `01-shell-standard.md §5`）。

## 已验证 / 未验证清单（持续维护）

沿用诚实风格：`| 场景 | 结果 | 备注 |`。每次交付追加，不做无依据的"已验证"。

| 场景 | 结果 | 备注 |
|---|---|---|
| YOLO 参数·横幅1（近景 8 人） | 8/8 稳定 track；1280 找回遮挡人 12 帧、耗时 +26ms/帧 | 素材过干净区分度有限；数据 `out/yolo_bench2/` |
| YOLO 参数·远角车+人（仅人） | 默认全灭（0 检出）；1280 仅帧 54 检 1 人 | 远角小目标须 imgsz 1280+ |
| YOLO 参数·远角车+人（人+车） | 1280 检出 52→230、track 1→7；conf0.15 找回 1 人；conf0.50 丢目标 | 车撑 track 骨架、人极难过检；`out/yolo_bench4/` |
| YOLO 参数·人群1（仅人） | 640 全组 0 track；1280 4 条稳定 track | 小目标 imgsz 1280 必开；`out/yolo_bench5/` |
| YOLO 模型档位（横幅1） | yolov8n vs yolov8s 检出 386 vs 391，几乎无差 | 固定 `yolov8n.pt` 即可 |
| 警用无人机链路端到端（traffic.MP4） | 150 帧→10 帧分析→报送→真实 qwen3-vl 研判"交警类"→事件尾命中 | 保留链路 PIPE0001 |
| 输出处理删除 + 事件尾（2026-10） | `OutputPolicy` 删除；三条保留链路以 `seq_filter`（事件尾）收尾 | spec 校验 `web_lab/validate_chain.py` |

**未验证**：人群密集 / 重叠素材的 iou 防合并效果；比远角车+人更小目标的 1280+ 极限召回；
长视频（>150 帧）track 稳定性与漂移。
