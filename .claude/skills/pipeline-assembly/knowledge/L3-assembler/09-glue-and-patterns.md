# 拼装套路与胶水文件（组装工专属）

> 层：L3-assembler ｜ 读者：组装工

## 拼装套路（模块复用模式）

- **2.1 零改动复用**：鸭子类型传入，模块间不强耦合（如 plate.FaceStage/PlateStage 只需实现约定接口）。
- **2.2 参数即配置**：差异收敛为模块 preset / 链路 stages，不写分支代码。
- **2.3 报送 / 告警粒度约定**：按识别类型（class）报送、dwell 按 track、scene 仅 large_only。
- **2.4 新胶水文件最小化**：新增逻辑只进胶水文件，核心模块只读。

## 胶水文件套路（先例：裁剪/检测的拆分）

- 现有模块零改动；新增逻辑全在胶水/模块里（如坐标还原 + 原分辨率出图）。
- 编排函数串模块（现例：`目标裁剪(crop_targets)` 按 YOLO 框裁子图 → 识别模块
  `FaceStage`/`extract_plates` 在子图里检脸/检牌 → 坐标还原 → 原图出小图）。
  `crop_restore.extract_faces` 为**历史兼容编排**（已被拆成「目标裁剪 + 人脸检测」两模块）。
- 配一个测试壳：断言 + 落盘产物（annotated.png 级）供目检。
- 判断"要新增胶水 vs 改模块"：优先零改动复用；要新增时说明成本（行数级）再动手。
- **通用能力沉淀**：`pipe/cropper.py`（Cropper：按框裁图可缩放 / 坐标还原 / 原分辨率提取，
  纯 cv2+numpy）＋ `pipe/target_crop.py`（目标裁剪模块）＋ `pipe/conditions.py`（框筛选 DSL）；
  新链路凡涉及"按上游框裁图 + 坐标还原"直接用目标裁剪，别再造轮子（车牌/人脸链共用）。

## 资源估算速用

只记"怎么用"：estimate_module / estimate_pipeline / back_calculate 的调用形态与参数；
公式与指纹细节在 `umvp/resources.py`（溯源指针，不复制）。调用模板见 07-shell-templates.md §3.4。

## VLM 提示词辅助撰写要点

- 从业务需求起草 `vlm_prompt` 初稿，给 2–3 版候选（简洁 / 严格 / 带示例）说明取舍。
- **词表对齐（关键）**：AlarmPolicy 按 `target_actions` **精确匹配** VLM 返回的 action 字符串——
  提示词输出词表必须与 `target_actions` 对齐，一并给出 `target_actions` 建议。撰写顺序：先定词表、再写提示词。
- 提示词效果需实测（真 VLM 或模拟 infer），标注"依赖实测"。
