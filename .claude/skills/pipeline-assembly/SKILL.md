---
name: pipeline-assembly
description: 链路拼接与壳代码编写知识库入口 —— UMVP 链路组成环节的全部所需知识（系统介绍、模块速查、链路决策、壳代码骨架、拼装套路、常见坑、已验证/未验证清单）。当需要"把业务需求组装成可运行管道"、"写拼接层/测试壳/胶水文件"、"拼装链路 / 选模块组合"、"给链路建议或资源估算"时使用。总装/组装工两个 agent 的知识来源。
user-invocable: true
---

# 链路拼接知识库（pipeline-assembly）

> 版本：v1.0（2026-08-11）。由旧"内联大段 agent 提示词 + 两个 skill 文件夹"重塑为单一知识库。

## 这是什么

链路组成环节的全部所需知识，按 4 层存于 `knowledge/` 子目录，由 `INDEX.md` 索引。
总装与组装工两个 agent 的提示词均指向本知识库；裸 LLM 由注入方按 INDEX 内联。

## 快速导航

| 层 | 内容 | 路径 |
|---|---|---|
| L1-shared | 系统介绍 / 环境命令 / 模块速查 | `knowledge/L1-shared/` |
| L2-architect | 意图识别 / 工作流 / 质量门与汇报 | `knowledge/L2-architect/` |
| L3-assembler | 壳模板 / 拼装与胶水套路 / 工作流与汇报 | `knowledge/L3-assembler/` |
| L4-experience | 决策表 / 坑 / 验证清单 / 先例 | `knowledge/L4-experience/` |

完整索引、文件清单与加载时机见 `INDEX.md`。
