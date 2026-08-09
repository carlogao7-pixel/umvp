# 部署变更台账（DEPLOYMENT.md）

用途：统一记录**影响项目部署**的改动，随代码一起移植，不依赖 git 历史。
维护规则：

- 只追加，不修改历史条目；新条目插在「## 记录」标题正下方（最新在前）。
- AI 在完成部署相关改动并验证后写入；拿不准算不算时，记一条无害。
- 与 CLAUDE.md「协作约定」中的部署台账规则配套使用。

## 判断清单（以下任一命中即为"影响部署"）

- 依赖：requirements.txt / pip 包新增、删除或版本变化
- 模型：models/*.pt、face_models/* 新增/移动/删除，或加载路径约定变化
- 环境变量：MYSQL_* 等新增/改名/默认值变化
- DB：web_lab presets 表结构、连接参数（db.py）变化
- 标定：out/fingerprints.json 重跑（换机/换卡）、性能基线数字变化
- 入口：新增/重命名/删除脚本、server.py 端口、CLI 参数
- 结构：新模块目录、`__file__` 相对路径依赖、包结构变化
- 配置：compose spec 新模式、影响运行结果的默认参数变化

仅代码逻辑/文档改动且不影响运行配置的，不必记录。

## 记录格式

每条以标题块形式插入「## 记录」正下方：

```
### YYYY-MM-DD · 一句话标题
- 变更内容：改了什么
- 影响面：依赖 / 模型 / 环境变量 / DB / 标定 / 入口 / 结构 / 配置（可多选）
- 部署注意：换机/移植/重启时需要做什么
- 验证：使用的命令与结果
```

## 记录

### 2026-08-09 · git 版本管理引入：首次提交 c742a3f
- 变更内容：`git init`（分支 main），新增 `.gitignore`（忽略 models/*.pt、face_models/、
  tests/data/vid/、.mysql/、out/face_db.*、out/video_db.*、tests/data/out/），首次提交 46 个
  代码/文档/测试文件（7.6M）；README「无 git 关联」说明更新为 git 管理表述。本地身份
  carl0 用户 carlogao7-pixel <carlogao7@gmail.com>，SSH key（ed25519）已认证 GitHub。
- 影响面：结构 / 入口
- 部署注意：克隆后无模型权重与测试视频，需按 INSTALL.md 从上游/zip 重建；`out/fingerprints.json`
  随库保留（标定基线），face_db/video_db 需重跑生成。
- 验证：`git status` 干净；`ssh -T git@github.com` 认证成功。
- 变更内容：ai 环境补装 `insightface==1.0.1`（原环境缺失，人脸链路必需）；全量文档
  （CLAUDE.md / README.md / INSTALL.md / doc/需求文档-开发版.md / .claude/agents/pipeline-assembler.md /
  .claude/skills/face-extract-pipeline/SKILL.md）中的解释器与项目根路径由
  `/home/carloga0/...` 统一修正为 `/home/carl0/miniconda/envs/ai/bin/python` 与
  `/home/carl0/project/umvp`（旧路径在本机不存在）。
- 影响面：依赖 / 配置
- 部署注意：换机移植后若 conda 路径不同，以实际解释器为准；运行命令一律使用
  `/home/carl0/miniconda/envs/ai/bin/python`。
- 验证：`test_composer.py` 27/27 断言通过；`test_face_embed.py --self-check` 28/28 命中；
  `umvp/resources.py` 自检通过。
