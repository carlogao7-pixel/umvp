# 链路组装 agent 评测汇总

> 时间：2026-10-13 15:50｜模型：deepseek-chat

| id | kind | 关注点 | 总装ok | 组装ok |
|---|---|---|---|---|
| qa-mechanism | qa | 链路机制与数据流（总装须知） | True | — |
| build-fight | build | 行为研判（VLM）+ 事件尾 + 词表 | True | True |
| build-plate | build | 车牌链：识车→裁车→车牌识别 | True | True |
| build-drone-far | build | 远角小目标：imgsz/conf + 无人机场景 | True | True |
| mod-plate-repeat | modify | 同车牌反复报：时序判定同对象一次/冷却 | True | True |
| prompt-vlm | prompt | VLM 提示词 + 词表对齐 | True | — |
