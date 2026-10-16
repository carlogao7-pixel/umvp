# 链路组装 agent 评测汇总

> 时间：2026-10-13 15:41｜模型：Qwen/Qwen2.5-32B-Instruct

| id | kind | 关注点 | 总装ok | 组装ok |
|---|---|---|---|---|
| qa-capability | qa | 系统能力边界（识别什么、能否判行为） | True | — |
| qa-mechanism | qa | 链路机制与数据流（总装须知） | True | — |
| build-parking | build | 车辆数量判定/拥堵；成组数量或VLM | True | True |
| build-fight | build | 行为研判（VLM）+ 事件尾 + 词表 | True | True |
| build-plate | build | 车牌链：识车→裁车→车牌识别 | True | True |
| build-face-capture | build | 人脸截取：识人→裁人→检脸出原图 | True | True |
| build-face-identity | build | 人脸身份：裁人→检脸→嵌入→底库 | True | True |
| build-drone-far | build | 远角小目标：imgsz/conf + 无人机场景 | True | True |
| build-smoking | build | VLM 行为识别 + 事件尾 | True | True |
| build-firelane | build | 车辆停留/时序判定（持续占用） | True | True |
| build-crowd | build | 人群聚集（成组数量 + 确认） | True | True |
| mod-interval | modify | 省资源：帧管理抽帧间隔 | True | True |
| mod-falsepos | modify | 误报：检测筛选/YOLO置信度 | True | True |
| mod-plate-repeat | modify | 同车牌反复报：时序判定同对象一次/冷却 | True | True |
| mod-face-blur | modify | 人像糊：目标裁剪输出分辨率 | True | True |
| upgrade-core | upgrade | 升级通道：不擅自动核心模块 | True | — |
| ambiguous | ambiguous | 信号不足：先澄清不硬猜 | True | — |
| prompt-vlm | prompt | VLM 提示词 + 词表对齐 | True | — |
