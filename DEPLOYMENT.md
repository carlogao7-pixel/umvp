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

### 2026-08-10 · 新增 web_lab 一键停止/重启脚本与端口操作手册
- 变更内容：新增 `web_lab/stop.sh`（按端口监听 PID 定位，找不到按进程名兜底，停止 web_lab
  server；默认连自含 MySQL 一起停，`--keep-mysql` 保留）与 `web_lab/restart.sh`（一键重启：
  停旧 web 进程→确保自含 MySQL 运行→后台拉起 server，日志写 `out/web_lab.log`，支持
  `--port/--host/--no-db` 透传与 `--stop-mysql` 连库一起重启）；新增操作文档
  `doc/端口与启停手册.md`；`.gitignore` 增加 `out/web_lab.log`。`web_lab/run.sh` 行为不变
  （前台启动）。
- 影响面：入口
- 部署注意：换机后脚本自动定位解释器（`conda run -n ai which python`，可用 `PY=` 覆盖）；
  脚本只按项目 `.mysql/mysql.sock` 特征识别自含 MySQL，不会误停系统 MySQL；
  MySQL 停止优先 `mysqladmin shutdown`，无权限时自动回退 `kill -TERM`（干净落盘）。
- 验证：`bash -n web_lab/stop.sh web_lab/restart.sh` 通过；实机跑通五条路径——① `./web_lab/restart.sh`
  冷启动：MySQL 拉起 + server 后台启动，curl 200，8001/3307 均监听；② `./web_lab/stop.sh` 全停：
  两端口释放；③ `./web_lab/stop.sh --keep-mysql`：web 停、3307 保留；④ `./web_lab/restart.sh
  --stop-mysql`：连库重启成功；⑤ `--port 8080` 换端口启动，且不带 `--port` 的 `stop.sh`
  按进程名定位并停掉 8080 实例。

### 2026-08-10 · 本机远端切 SSH 并首次推送同步
- 变更内容：本机（carloga0）远端地址由 HTTPS `https://github.com/carlogao7-pixel/umvp.git`
  切换为 SSH `git@github.com:carlogao7-pixel/umvp.git`；生成 ed25519 SSH key
  （`~/.ssh/id_ed25519`，GitHub 登记名 `carloga0-wsl2`，Authentication Key）并认证成功；
  首次 `git push -u origin main` 同步两个本地提交（d67019e 适配本机路径 +
  4b9eb6d 路径可移植化），`main` 与 `origin/main` 对齐。CLAUDE.md 新增
  「git 协作（多机同步）」小节（提交习惯 / 换机先 pull / 路径可移植说明）。
- 影响面：配置
- 部署注意：换机移植后需重新生成/登记 SSH key；HTTPS 克隆也可用但每次 push 要认证。
- 验证：`ssh -T git@github.com` 输出 `Hi carlogao7-pixel! You've successfully authenticated`；
  `git status -sb` 显示 `main...origin/main` 无落后超前；工作树干净。

### 2026-08-10 · 解释器与路径可移植化（跨机器同步铺垫）
- 变更内容：文档与脚本中的机器绝对路径统一改为便携写法——解释器一律写
  `conda run -n ai python`（CLAUDE.md / README.md / INSTALL.md / doc/需求文档-开发版.md /
  .claude/agents/pipeline-assembler.md / .claude/skills/face-extract-pipeline/SKILL.md /
  web_lab/calibrate.py 用法）；`web_lab/run.sh` 的 PY 解析改为
  `conda run -n ai which python` 自动定位（失败回退 `command -v python`，仍可用
  `PY=...` 环境变量显式覆盖）；`tests/data/yolo_standalone_output.json` 的 model_path
  改为相对路径 `models/yolov8n.pt`。
- 影响面：配置 / 入口
- 部署注意：换机移植无需再改解释器路径；目标机需有 conda 环境 `ai`
  （与 INSTALL.md 场景 B 一致）。
- 验证：`conda run -n ai python umvp/pipe/test_composer.py` 27/27 断言通过；
  `conda run -n ai python umvp/face_embed/test_face_embed.py --register-dir
  umvp/face_detect/test_imgs/ --self-check --db out/face_db.npz` 21/21 自检命中；
  `bash -n web_lab/run.sh` 语法通过；`web_lab/calibrate.py` py_compile 通过；
  `conda run -n ai which python` 正常解析出解释器路径。

### 2026-08-10 · 本机（carloga0）接入 GitHub 仓库并适配环境路径
- 变更内容：本机 `/home/carloga0/projects/umvp` 接入远端仓库 origin=`https://github.com/carlogao7-pixel/umvp.git`
  （分支 main，首次提交 46 个代码/文档/测试文件，含 `.gitignore`，远端记录见下条）；文档与脚本中
  的解释器/项目根路径由 carl0 机器路径（`/home/carl0/miniconda/envs/ai/bin/python`、
  `/home/carl0/project/umvp`）改回本机 carloga0 路径（`/home/carloga0/miniconda3/envs/ai/bin/python`、
  `/home/carloga0/projects/umvp`）。远端历史条目保留原样不改写。
- 影响面：配置 / 入口
- 部署注意：换机移植后若 conda 路径不同，以实际解释器为准。
- 验证：`git status` 干净；`test_composer.py` 27/27 断言通过。

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
