# 工单看板常驻服务（ticketd）

> 操作手册：把 `.opencode/tool/` 下的工单工具（`ticket.py` / `ticket_web.py` / `tickets.html`）
> 装成本机（Ubuntu / WSL）**systemd user 常驻服务**，浏览器随时打开看板、点点鼠标改工单；
> **一份代码可挂多个项目**（每项目一个配置文件 + 一个实例）。
> 文档总览见 [`../README.md`](../README.md)；UMVP 主服务的端口与启停见
> [`runbook.md`](runbook.md)；安装 / 换机步骤见 [`../../INSTALL.md`](../../INSTALL.md)。
> 最后更新：2026-10-20

## 一、这是什么 / 定位

- **一份代码多实例**：代码只有一份（就在本仓库 `.opencode/tool/`），每个项目写一个
  **配置文件**，用 systemd 的**模板服务** `ticketd@<项目名>` 各起一个进程。
  「实例名 = 项目名」，互不干扰：各自读各自的工单文件、各自的标题/状态/代理名单、各自的端口。
- **零依赖**：只用系统 `python3`（标准库），**不需要 conda `ai` 环境、不需要 MySQL**。
  进程常驻约 15–30MB、单线程，看板操作不会让它变胖。
- **常驻 + 开机自启**：装好即 `enable --now`；WSL 里开了 systemd（`/etc/wsl.conf`
  `systemd=true`）且用户 `Linger=yes`，关掉终端服务也不停。
- **只在本机**：默认绑 `127.0.0.1`，不做鉴权，局域网/公网访问不到（后续"B 方案"才考虑
  单端口多项目路由/鉴权，本期不做）。
- **只读实例**：配置文件里 `"readonly": true` 可让某个实例纯查看、禁用所有写操作。

## 二、快速开始（三步）

以 UMVP 项目为例（仓库即 `/home/carloga0/projects/umvp`）：

```bash
# 1) 写配置（下面这行会生成 ~/.config/ticketd/umvp.json）
#    工单文件默认就指向本仓库的 .opencode/memory/tickets.md，端口默认 8010
bash .opencode/tool/install_ticketd.sh --name umvp --port 8010

# 2) 启用并自启（脚本内部已执行；这一步想手动补也行）
systemctl --user enable --now ticketd@umvp

# 3) 打开浏览器
#    WSL：在 Windows 侧浏览器访问 http://localhost:8010/
```

装完脚本会打印访问地址和常用命令；`systemctl --user is-active ticketd@umvp` 应为 `active`。

## 三、配置文件字段

位置：`~/.config/ticketd/<项目名>.json`（尊重 `XDG_CONFIG_HOME`）。

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | **是** | 工单文件路径。支持 `~` 展开；给相对路径时**按配置文件所在目录**解析（建议写绝对路径） |
| `name` | 否 | 实例名（默认 `umvp`） |
| `title` | 否 | 看板标题（默认 `UMVP 工单看板`） |
| `host` | 否 | 监听地址（默认 `127.0.0.1`，仅本机） |
| `port` | 否 | 监听端口（默认 `8010`） |
| `statuses` | 否 | 状态名单（数组）。不给=内置 `进行中/待测试/已完成/已取消`；**给了就覆盖** |
| `agents` | 否 | 承接代理名单（数组）。不给=内置 UMVP 五代理；**给了就覆盖** |
| `readonly` | 否 | 只读实例（`true` 时页面隐藏写入口、后端拒绝写操作），默认 `false` |
| `inbox_file` | 否 | 收件箱文件（`/inbox.md`）。**不给或 `null` = 隐藏「收件箱」页签与相关 API**；给了路径即启用（支持 `~`，相对按配置文件所在目录） |
| `deployment_file` | 否 | 部署台账文件（仓库根 `DEPLOYMENT.md`）。**不给或 `null` = 隐藏「部署台账」页签与相关 API** |
| `models_tab` | 否 | 是否启用「模型」页签（UMVP 专属；默认 `true`）。复用本工具的其它项目可设 `false` 关闭 |

示例（另一个项目的配置；不想启用收件箱/台账就不写这两行，或写 `null`）：

```json
{
  "name": "myproj",
  "title": "MyProj 工单看板",
  "file": "/home/carloga0/projects/myproj/.opencode/memory/tickets.md",
  "host": "127.0.0.1",
  "port": 8011,
  "statuses": ["待办", "处理中", "归档"],
  "agents": ["alice", "bob"],
  "inbox_file": null,
  "deployment_file": null
}
```

> 已有旧配置升级：重跑一次 `install_ticketd.sh`（不带 `--force`）会**只补缺失的**
> `inbox_file` / `deployment_file`（探测到文件就启用，否则写 `null`），其余字段不动；
> 之后 `systemctl --user restart ticketd@<name>` 生效。

**配置来源优先级（从高到低）**：`CLI 参数` > `环境变量` > `配置文件` > `内置默认`。

- 环境变量：`TICKET_CONFIG`（指向配置文件路径；兼容 `TICKETD_CONFIG`）。
- 命令行：`ticket_web.py --config PATH` / `--name NAME` / `--file/--host/--port/--title`；
  `ticket.py --config PATH` / `--name NAME` / `--file PATH`。全局选项写在子命令**之前**：

  ```bash
  python3 .opencode/tool/ticket.py --config ~/.config/ticketd/myproj.json list
  ```
- **状态/代理名单可配置**：新建工单落入**状态名单的第一个状态**；主单收尾提醒把
  **最后两个状态**视为终态。工单文件仍是「每状态一个 `## 节` + 7 列表格」，字段内禁用 `|`
  （格式契约不变）。自定义项目建议沿用 `WO-0001` 形式的编号。
- 模板 unit 的 `ExecStart` 假定配置目录为 `~/.config/ticketd`（即未自定义 `XDG_CONFIG_HOME`）；
  若你改过 `XDG_CONFIG_HOME`，请相应调整 unit 里的 `--config` 路径（或改用 `ticketd --config` 手工启动）。

**配置非法会 fail-fast**：坏 JSON / 缺 `file` / 端口非法 / 状态为空等 → 打印中文错误并以
非 0 退出，**绝不写工单文件**。

## 四、常用命令（systemd user 级）

```bash
systemctl --user status  ticketd@umvp      # 看状态
systemctl --user restart ticketd@umvp      # 重启
systemctl --user stop    ticketd@umvp      # 临时停
systemctl --user disable --now ticketd@umvp # 停止并取消开机自启
systemctl --user enable  --now ticketd@umvp # 启用并自启
journalctl --user -u ticketd@umvp -f       # 看实时日志
```

安装脚本：

```bash
bash .opencode/tool/install_ticketd.sh --help         # 看全部选项
bash .opencode/tool/install_ticketd.sh --name umvp --port 8010
bash .opencode/tool/install_ticketd.sh --name umvp --uninstall   # 停止+取消自启（保留配置文件）
```

脚本是**幂等**的：重复执行不会覆盖已有配置/模板（加 `--force` 才覆盖），包装脚本每次刷新（自愈）。
安装器新增 `--inbox-file PATH` / `--deployment-file PATH`（默认自动探测仓库内
`.opencode/memory/inbox.md` 与 `DEPLOYMENT.md`，存在即启用；传 `none` 显式禁用）。

## 四A、看板页签与统一记录 CLI（ledger.py）

看板除「工单」外还有两个可配置页签（配置了对应文件才出现）：

- **收件箱**：卡片流展示 `inbox.md` 条目（按状态/类型/来源筛选 + 全文搜索），
  支持页面上**追加新条目**（INB 编号自动递增，落「待同步」顶部）与**流转**
  （待同步 → 已同步，可填去向说明，写入/更新 `→ 去向：` 行）。
- **部署台账**：卡片流展示 `DEPLOYMENT.md` 条目（按影响面筛选 + 搜索），
  **登记表单**内嵌判断清单勾选（依赖/模型/环境变量/DB/标定/入口/结构/配置 8 选 N），
  日期与星期自动，新条目插「## 记录」正下方（最新在前）。
- 页头**编号直达**框：输入 `INB-0049` / `WO-0009` / `DEP-0001`（或台账「日期+标题」
  模糊串）直接打开单条详情，避免翻整个文件。

配套 CLI（写测试**务必显式 `--file`/`--deployment-file` 指向临时副本**，INB-0039 教训）：

```bash
python3 .opencode/tool/ledger.py get INB-0049            # 收件箱单条（纯文本全文）
python3 .opencode/tool/ledger.py get WO-0009             # 工单单条（转发 ticket.py 解析）
python3 .opencode/tool/ledger.py get DEP-0001            # 台账单条（解析顺序编号，最新在前）
python3 .opencode/tool/ledger.py get "2026-09-23 抽出目标裁剪"   # 台账模糊（日期+标题）
python3 .opencode/tool/ledger.py get WO-0008 --from-deployment  # 按标题内 WO 号查台账条目
python3 .opencode/tool/ledger.py list [--status 待同步] [--type 成果] [--source web] [--since 日期]
python3 .opencode/tool/ledger.py list --deployment [--impact DB] [--q 关键字]
python3 .opencode/tool/ledger.py append --source web --type 成果 --text "一句话结论" [--followup "→ 去向：…"]
python3 .opencode/tool/ledger.py append --deployment --title "标题" --changes "改了什么" \
                                    --impact DB,入口 [--notes …] [--verify …] [--date …]
python3 .opencode/tool/ledger.py move INB-0049 [--destination "去向说明"]
```

REST API（供脚本/其它工具调用；未配置对应文件的实例返回 404）：
`GET /api/inbox`（筛选参数 status/type/source/since/q）、`POST /api/inbox/append`、
`POST /api/inbox/move`、`GET /api/deployment`（impact/since/q）、
`POST /api/deployment/new`、`GET /api/entry?id=`（统一编号，自动识别 INB/WO/DEP）。

格式契约（**文件是唯一真相源，本工具不迁移不改写历史条目**）：

- `inbox.md`：条目 = `- [日期][INB-xxxx][来源][类型] 结论` + 缩进续行（如 `→ 去向：…`）；
  INB 编号全局递增（两节合取 max+1）；历史存在编号跳空（如 INB-0020 未用），按 max+1 续。
- `DEPLOYMENT.md`：条目 = `### 日期(星期) · 标题` + 四字段块（变更内容/影响面/部署注意/验证）；
  历史条目星期可缺省、文末存在同字段重复的孤儿块——解析器均兼容（多块按序合并展示）。

## 四B、「模型」页签与 agent_model CLI

「模型」页签（`models_tab: true`，默认开启）管理**本机 opencode 的代理模型**：
展示五个开发代理（orchestrator / pipe-chain / web / agent-builder / qa）与项目默认模型，
支持逐代理设置/回默认、改项目默认。数据来源与写操作**全部在 `.opencode/tool/agent_model.py`**
（页签只是它的界面，勿在别处复制第二套逻辑）：

```bash
python3 .opencode/tool/agent_model.py show                  # 五代理 + 项目默认现状
python3 .opencode/tool/agent_model.py set qa zai-coding-plan/glm-5.2   # 该代理独立模型
python3 .opencode/tool/agent_model.py clear qa              # 删 frontmatter model，回项目默认
python3 .opencode/tool/agent_model.py default zai-coding-plan/glm-5.2  # 改项目默认（opencode.json）
python3 .opencode/tool/agent_model.py --self-test           # 52 项断言（临时副本，真实文件只读）
```

**生效语义（重要）**：

- 会话内 `/model` 切换 → **立即生效且被派发的子代理跟随**（零重启；子代理模型 =
  frontmatter `model` ?? 父会话模型）。
- 本页签 / CLI 的改动写的是**配置文件**（agent frontmatter 或 `.opencode/opencode.json`）
  → **新会话生效**；当前会话不受影响，也不会自动重启。
- 想给某个代理固定不同模型 → 用它自己的 `model:` 行；日常整体切换用 `/model` 即可。
- 页签为只读实例（`readonly: true`）时禁用写操作；后端 API：
  `GET /api/models`、`POST /api/models/set|clear|default`（`models_tab=false` 的实例一律 404）。

## 五、多项目与端口规划

- 每个项目一个实例：`ticketd@<项目名>`，名字只允许字母/数字/下划线/连字符。
- 端口逐个分配、**同机不重复**（脚本会检测：与其它实例配置重复、或端口已被占用 → 报错退出）：

| 实例（项目） | 端口 | 工单文件 |
|---|---|---|
| umvp | 8010 | `~/projects/umvp/.opencode/memory/tickets.md` |
| myproj | 8011 | `~/projects/myproj/.opencode/memory/tickets.md` |
| proj2 | 8012 | `~/projects/proj2/.opencode/memory/tickets.md` |

- 共享的**同一份代码**在本仓库；其它项目只要一个 `~/.config/ticketd/<名字>.json`。
- 模板服务 `~/.config/systemd/user/ticketd@.service` 只写一次，所有实例共用。

## 六、WSL 注意事项

- **Windows 侧浏览器访问 `http://localhost:8010/`**（WSL2 会转发到 Linux 回环）。
- **项目务必放 Linux 家目录（如 `~/projects/...`），不要放 `/mnt/c/...`**：
  Windows 盘符挂载（DrvFs）上文件锁（flock）语义不完整，多个写者可能互相覆盖。
- 锁文件放在 `/tmp`（按目标文件绝对路径哈希），属**单机语义**——同一文件不要跨机器同时写。
- 保证 `/etc/wsl.conf` 有 `[boot] systemd=true`，且 `loginctl show-user $USER -p Linger`
  为 `Linger=yes`（终端关掉服务不停）。改完 `/etc/wsl.conf` 需 `wsl --shutdown` 重启 WSL。

## 七、换机 / 新项目重建

换一台机器（或新克隆本项目）后，从零重建只需三步：

```bash
# 1) 打印模板 + 包装脚本 + 配置（路径自动取当前仓库位置）
bash .opencode/tool/install_ticketd.sh --name umvp --port 8010
# 2) 若已有旧配置想沿用，可先把它拷到 ~/.config/ticketd/myproj.json，
#    然后只 enable： systemctl --user enable --now ticketd@myproj
```

脚本内所有路径都基于**脚本自身位置**推导（`--file` 默认指本仓库的工单文件、包装脚本指向本
仓库的 `ticket_web.py`），因此换机后不必改脚本。

## 八、FAQ

- **为什么不用 conda `ai` / MySQL？** 工单工具是纯标准库小工具：只需要 `python3` 读写一个
  Markdown 文件，没有模型推理、也没有数据库依赖。因此它跟 UMVP 主服务（web_lab 用 8001 +
  MySQL 3307）完全解耦，不占用 AI 环境，也不受主服务启停影响。
- **两个实例能同时跑吗？** 能（各自进程、各自端口、各自文件）；写操作走文件锁，互不影响。
- **为什么我在临时目录里跑脚本，工单文件还是被改了？** 因为默认工单文件是按**脚本自身位置**
  解析的（不再依赖当前工作目录），所以在任意 cwd 下都能正确找到本仓库的工单文件——这既是
  "换机/服务化后不用 cd"的原因，也意味着**做测试/自检/一次性脚本时必须显式传
  `--file <临时副本>`**，否则写操作会落到真实工单文件上。（自带的 `--self-test` 已在临时副本上跑，安全。）
- **页面能删工单吗？** 不能（按设计不提供删除）；改状态用下拉选择，不支持拖拽。
- **局域网同事能访问吗？** 默认不能（只绑 `127.0.0.1`、无鉴权）。这是有意为之。
- **`~/.local/bin` 不在 PATH？** 脚本会提示但不改你的 shell 配置；systemd 用绝对路径启动
  不受影响，只有你想在终端直接敲 `ticketd` 时才需要把它加进 PATH。
