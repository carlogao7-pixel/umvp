---
name: pipeline-assembly
description: 链路拼接与壳代码编写手册 —— 把 UMVP 已有模块拼成可运行管道的经验集：链路决策表、拼装套路、壳代码骨架模板、常见坑、已验证/未验证清单。当需要"把业务需求组装成可运行管道"或"写拼接层/测试壳/胶水文件"时使用。Also for deciding between the four modes (small_only/small_crop/small_full/large_only) and face/monitor chains.
user-invocable: true
---

# 链路拼接与壳代码编写（pipeline-assembly）

> 状态：**框架版 v0.1**（2026-08-07）。只立结构与关键要点，正文标【待定稿】处后续补全。
> **自包含约定**：本 skill 单独交出即可用——经验知识全部内联，不依赖读代码或别的文档；
> 代码文件路径只作溯源。可能被本项目会话 / 其他 agent / 裸 LLM 使用。
> 陈述性知识（系统介绍、模块速查、验证命令）不在此重复，见 agent 提示词。

## 这是什么

把"业务需求 → 可运行管道"的拼装经验沉淀成手册。样板先例：face-extract-pipeline 链路
（现有模块零改动 + 1 个胶水文件 `umvp/pipe/crop_restore.py` + 1 个测试壳）。

## 1. 链路决策表（什么场景选什么链路）【待定稿】

### 1.1 四模式判断矩阵
输入列：有无 VLM / 识别目标 / 告警需求 / 成本约束 → 输出：模式 + 关键参数初值。
逐条填：small_only / small_crop / small_full / large_only 各自适用场景、取舍、参数初值范围。

### 1.2 人脸链路
YOLO(识人) → crop_restore(裁人+坐标还原) → SCRFD(检脸) → ArcFace(嵌入) → FaceStore(检索)。

### 1.3 监控链路
FaceMonitor（视频抽帧 + 人脸检索）。

### 1.4 决策原则
先看"有没有 VLM / 要哪种告警 / 预算"再落模式；参数给范围并说明动机；方案可对比。

## 2. 拼装套路（模块复用模式）

- 2.1 **零改动复用**：鸭子类型传入，模块间不强耦合（extract_faces 连 import 都不需要）。
- 2.2 **参数即配置**：差异收敛为 compose spec，不写分支代码。
- 2.3 **报送/告警粒度约定**：按识别类型（class）报送、dwell 按 track、scene 仅 large_only。
- 2.4 **新胶水文件最小化**：新增逻辑只进胶水文件，核心模块只读。

## 3. 壳代码骨架（模板）【待定稿：填可复用片段】

- 3.1 驱动壳：compose(spec) → Pipeline → 帧循环 step()
- 3.2 接线壳：YOLO dets 喂入、VLM infer 回调（真/模拟）、告警消费落盘
- 3.3 测试壳：断言式 + 限帧 + 落盘可视化核验（annotated.png 级）
- 3.4 估算接入：ResourceEstimator.estimate_pipeline / back_calculate

## 4. 常见坑【待定稿，先列已知条目】

- onnxruntime 形状警告（VerifyOutputSizes / Duplicate provider）不影响结果
- check_interval 节流：同 ts 连续报送会被抑制
- 坐标还原：缩放 + 偏移；放大图坐标不能直接存图
- 设备选择独立：YOLO(ultralytics 自动) vs SCRFD(onnxruntime CPU) 别混为一谈
- 重叠框去重：同一目标进多路需 IoU 去重
- VLM 提示词词表必须对齐 target_actions（否则命中率归零）

## 5. 已验证 / 未验证清单（持续维护）

沿用 face-extract-pipeline 的诚实风格：`| 场景 | 结果 | 备注 |`。每次交付追加。

## 6. 资源估算速用

只记"怎么用"：estimate_module / estimate_pipeline / back_calculate 的调用形态与参数；
公式与指纹细节在 `umvp/resources.py`（溯源指针，不复制）。

## 7. 维护规则

新坑 / 新决策 / 新模板追加到对应节；填写时保持自包含（读者可能没有代码可看）。
