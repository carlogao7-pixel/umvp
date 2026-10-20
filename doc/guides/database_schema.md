# 数据库结构

> 参考手册：web_lab 命名配置暂存（presets）所用 MySQL 库的**当前结构事实**——
> 表结构、业务 ID 前缀、播种清单、参数表机制、数据快照、读写语义、与 HTTP 接口的对应。
> 设计动机与决策理由见 [`../design/storage_design.md`](../design/storage_design.md)（本文只记结构，不重复论证）。
> 数据依据：2026-10-16 对运行实例的实测；结构说明 2026-10-19 与代码复核更新。

## 1. 概览

- **用途**：web_lab 前端"命名配置暂存/取用"的后端存储，把各模块/链路的命名配置
  （preset）存下来，供以后按名字取回复用。
- **数据库名**：`umvp`（默认，可用 `MYSQL_DB` 覆盖）
- **表数量**：5 张主表（`presets` / `pipelines` / `pipeline_steps` /
  `pipeline_edges` / `module_ports`，后两张为图编排 P2-1，见 §5.5/§5.6）
  \+ 每模块一张参数表 `preset_params_<module_id>`（当前 11 张，见 §6）
- **ID 规则**：presets.id / pipelines.id 为"前缀+4 位序号"的业务编号
  （如 `YOLO0001`、`PIPE0001`），新建时按前缀自动递增；前缀表见 §3.2
- **播种（seed）**：服务器启动时 `_seed_pipeline()` 幂等写入**三条保留链路**
  （警用无人机 PIPE0001 + 车牌识别 PIPE0002 + 人脸截取 PIPE0003）+ 各阶段模块 preset，
  见 §7
- **引擎/字符集**：InnoDB / utf8mb4 / utf8mb4_unicode_ci
- **访问层**：`web_lab/db.py`（惰性连接 + 全局锁串行化，pymysql），
  HTTP 接口在 `web_lab/server.py`

## 2. 运行实例

| 项 | 值 | 覆盖方式 |
|---|---|---|
| 主机 | 127.0.0.1 | `MYSQL_HOST` |
| 端口 | 3307 | `MYSQL_PORT` |
| 用户 | umvp | `MYSQL_USER` |
| 密码 | 123 | `MYSQL_PASSWORD` |
| 数据库 | umvp | `MYSQL_DB` |

实例自含于项目内（免 sudo 启动）。

MySQL 不可用时不影响 web_lab 启动：`--no-db` 可显式禁用，presets 接口统一返回"未启用"。
启停脚本会自动拉起/关闭该实例，日常无需手动操作（见 [`runbook.md`](runbook.md)）。

### 2.1 运维命令（状态 / 开启 / 关闭）

```bash
# ---- 查看状态 ----
mysqladmin --socket=项目/.mysql/mysql.sock -u root ping   # 输出 mysqld is alive 即正常
ss -ltn | grep 3307                                       # 端口是否在监听
mysql --socket=项目/.mysql/mysql.sock -u umvp -p123 umvp \
  -e "SHOW TABLES; SELECT id,kind,name FROM presets;"     # 看库内容（应用账号）
tail -n 30 项目/.mysql/mysql.log                          # 排错看错误日志

# ---- 开启（日常交给 run.sh/restart.sh 自动完成；此处为单独手动启动）----
mysqld --datadir=项目/.mysql/data --socket=项目/.mysql/mysql.sock \
       --port=3307 --bind-address=127.0.0.1 \
       --log-error=项目/.mysql/mysql.log --daemonize

# ---- 关闭（umvp 应用账号没有 SHUTDOWN 权限，须用 root 走 socket）----
mysqladmin --socket=项目/.mysql/mysql.sock -u root shutdown
```

注意两点：

- `--log-error` 必须指到项目内（默认 `/var/log/mysql/` 普通用户写不了，会导致启动失败）。
- 关库不影响 web_lab 启动，只是 presets 界面显示"数据库不可用"；`./web_lab/run.sh --no-db`
  可显式禁用数据库。数据库故障只影响暂存功能，不影响测试/校准等其它能力。

## 3. 表：`presets`（命名配置注册表）

### 3.1 建表语句

```sql
CREATE TABLE `presets` (
  `id` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL,
  `kind` enum('module','pipeline') COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'module',
  `name` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `module_id` varchar(32) COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '',
  `params_json` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_kind_name` (`kind`,`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

### 3.2 业务 ID 规则

**业务 ID = 模块前缀 + 4 位序号**，由 `db.py` 在新建时按前缀自动递增分配
（查同前缀最大序号 +1，锁内执行）；再存一条 YOLO 配置即得 `YOLO0002`。
前缀登记在 `db.py` `_ID_PREFIXES`：

| 模块 | 前缀 | 模块 | 前缀 |
|---|---|---|---|
| frame_manager | FM | face_store | FSTO |
| yolo | YOLO | target_crop | CROP |
| vlm | VLM | plate_recog | PLATE |
| det_filter | DF | seq_filter | SEQ |
| face_detect | FDET | face_embed | FEMB |
| annotate | ANNO | （链路：pipelines / kind='pipeline'） | PIPE |

> 未登记前缀的模块取 `module_id` 前 4 字符大写兜底；`extract_faces`（EXTF）为已下线模块，
> 前缀仍登记但不再建表/播种。

**旧 schema 自动迁移**：`init_db()` 检测到 `presets.id` 仍是旧的自增整数
（2026-09-20 之前的库）时，自动 DROP 全部配置相关表并按新 schema 重建——
表内均为播种/测试数据，服务器启动随即重新播种保留链路，换机/升级零手工操作。

### 3.3 字段说明

| 字段 | 含义 | 说明 |
|---|---|---|
| id | 业务 ID | 形如 `YOLO0001`；`load`/`delete` 按它定位；定向更新（保 id 改参数）也按它 |
| kind | 配置类型 | `module`＝单模块配置；`pipeline`＝链路 JSON 快照（历史兼容保留，当前已无此类行） |
| name | 配置名 | 同 kind 下必须唯一；重复保存＝覆盖旧配置，不新建 |
| module_id | 模块标识 | 仅 `module` 类型用（如 `yolo`、`frame_manager`） |
| params_json | 配置内容 | 仅 `pipeline` 类型用；`module` 类型迁移后恒为空串，参数改存模块参数表（§6） |
| created_at / updated_at | 时间戳 | 列表按 `updated_at DESC, id DESC` 倒序（最近用的在前） |

### 3.4 索引

| 索引 | 列 | 类型 | 作用 |
|---|---|---|---|
| PRIMARY | id | 主键 | 按 id 精确取用/删除 |
| uk_kind_name | kind, name | 唯一索引 | 保证"同 kind 下名字唯一"，并支撑同名覆盖判断 |

### 3.5 当前数据快照（2026-10-16 实测）

- 行数：**23**，全部 `kind='module'`（**无 `kind='pipeline'` 行**——链路统一走
  `pipelines` + `pipeline_steps` 引用式，见 §4/§5）：

| id | name | module_id | 归属 |
|---|---|---|---|
| FM0001 | 警用无人机·帧管理 | frame_manager | PIPE0001 |
| YOLO0001 | 警用无人机·YOLO识别 | yolo | PIPE0001 |
| DF0001 | 警用无人机·检测筛选 | det_filter | PIPE0001 |
| ANNO0002 | 警用无人机·标注 | annotate | PIPE0001 |
| VLM0001 | 警用无人机·VLM研判 | vlm | PIPE0001 |
| SEQ0001 | 警用无人机·时序判定 | seq_filter | PIPE0001 |
| FM0002 | 车牌识别·帧管理 | frame_manager | PIPE0002 |
| YOLO0002 | 车牌识别·YOLO识车 | yolo | PIPE0002 |
| DF0002 | 车牌识别·检测筛选 | det_filter | PIPE0002 |
| CROP0002 | 车牌识别·目标裁剪 | target_crop | PIPE0002 |
| PLATE0002 | 车牌识别·车牌识别 | plate_recog | PIPE0002 |
| SEQ0002 | 车牌识别·时序判定 | seq_filter | PIPE0002 |
| FM0003 | 人脸截取·帧管理 | frame_manager | PIPE0003 |
| YOLO0003 | 人脸截取·YOLO识人 | yolo | PIPE0003 |
| DF0003 | 人脸截取·检测筛选 | det_filter | PIPE0003 |
| CROP0003 | 人脸截取·目标裁剪 | target_crop | PIPE0003 |
| FDET0002 | 人脸截取·人脸检测 | face_detect | PIPE0003 |
| FDET0001 | 人脸检测·默认 | face_detect | 独立 |
| FEMB0001 | 人脸嵌入·默认 | face_embed | 独立 |
| FSTO0001 | 向量底库·默认 | face_store | 独立 |
| CROP0001 | 目标裁剪·默认 | target_crop | 独立 |
| ANNO0001 | 标注·默认 | annotate | 独立 |
| PLATE0001 | 车牌识别·默认 | plate_recog | 独立 |

> 播种按「种子定义的名称」；在链路设计器里保存过的链路，其阶段 preset 名可能被改为
> 「链路名 · 模块名」，故名称以库内实测为准。
>
> 历史清理：2026-09-20 旧整数 schema DROP 重建时，清除了此前累积的历史 module 行
> 与 pipeline JSON 快照；后续 alarm / extract_faces 等下线模块的 preset 亦已剪除。

## 4. 表：`pipelines`（链路拼装）

链路 = 一组**有序**模块步骤，每步引用一个已入库的模块 preset（`presets.id`，`kind='module'`）。

### 4.1 建表语句

```sql
CREATE TABLE `pipelines` (
  `id` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL,
  `name` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `streams` int NOT NULL DEFAULT '1',
  `budget_json` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `meta_json` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

### 4.2 字段说明

| 字段 | 含义 | 说明 |
|---|---|---|
| id | 业务 ID | `PIPE` 前缀自动递增（PIPE0001→PIPE0002…）；save/load/delete 按它定位 |
| name | 链路名 | 必须唯一；重复保存＝覆盖旧链路（id 不变、步骤重建），不新建 |
| streams | 路数 | 多路外推，默认 1 |
| budget_json | 资源预算 | JSON，如 `{"vram_mb": 8192}`；链路级字段，不属于任何模块 |
| meta_json | 扩展字段 | JSON，链路级字段：场景描述 `description` + **端口连线 `stage_inputs`**（2026-10-19 图编排 P2-1 起：连线的一等公民是 `pipeline_edges` 表（§5.5），保存时**双写** meta；读取优先级 `pipeline_edges > meta.stage_inputs > 代码默认`。meta 里的形状规范为位置键 `{"#<position>": {端口: 产物名}}`，产物名为全名 `{module}#{position}.{artifact}`（短名在唯一实例时亦可）；`load_pipeline` 取回时回填到对应 stage 的 `inputs`，与只读 meta 时逐字节一致；旧链路 meta 无此键则原样（零回归） |
| created_at / updated_at | 时间戳 | 列表按 `updated_at DESC, id DESC` 倒序 |

### 4.3 当前数据快照（2026-10-16 实测）

- 行数：**3**——`PIPE0001` 警用无人机链路、`PIPE0002` 车牌识别链路、`PIPE0003` 人脸截取链路
  （均 streams=1，meta 带 description），步骤引用见 §7。
- 三条保留链路由 `_seed_pipeline()` 播种，meta **仅含 `description`、无 `stage_inputs`**——
  端口连线只在经设计器保存过连线的链路里才有。

## 5. 表：`pipeline_steps`（链路步骤）

### 5.1 建表语句

```sql
CREATE TABLE `pipeline_steps` (
  `id` int NOT NULL AUTO_INCREMENT,
  `pipeline_id` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL,
  `position` int NOT NULL,
  `preset_id` varchar(24) COLLATE utf8mb4_unicode_ci NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_pipeline_position` (`pipeline_id`,`position`),
  KEY `idx_preset` (`preset_id`),
  CONSTRAINT `fk_step_pipeline` FOREIGN KEY (`pipeline_id`) REFERENCES `pipelines` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_step_preset` FOREIGN KEY (`preset_id`) REFERENCES `presets` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

（`id` 为内部自增主键，非业务编号；业务定位用 `(pipeline_id, position)`。）

### 5.2 字段说明

| 字段 | 含义 | 说明 |
|---|---|---|
| id | 自增主键 | 内部行号 |
| pipeline_id | 所属链路 | 外键 → `pipelines.id`（如 PIPE0001） |
| position | 步骤顺序 | 同链路内 0..n-1 唯一；`load` 按它排序还原链路 |
| preset_id | 模块配置引用 | 外键 → `presets.id`（如 YOLO0001），且该行 `kind` 必须为 `module`；"角色"（yolo/vlm…）从 `presets.module_id` 反查，不另存 |

> 阶段间**端口连线**（`inputs = {端口: 上游产物名}`）自 2026-10-19（图编排 P2-1）起
> 独立落 `pipeline_edges`（§5.5），并与本表**同一事务**按提交的 `stages` 数组重建
> （`position` = 数组下标）；`pipelines.meta_json.stage_inputs` 作为双写副本与回退读路径保留。

### 5.3 引用完整性

- 删链路 → 步骤级联删除（`ON DELETE CASCADE`）。
- 删被引用的模块 preset → **拒绝**：`ON DELETE RESTRICT` 兜底 + `delete_preset` 先查引用、
  报错带引用它的链路名。

### 5.4 当前数据快照（2026-10-19 实测）

- 链路步骤共 **17**：PIPE0001 的 0..5 引用 FM0001 / YOLO0001 / DF0001 / ANNO0002 /
  VLM0001 / SEQ0001；PIPE0002 的 0..5 引用 FM0002 / YOLO0002 / DF0002 / CROP0002 /
  PLATE0002 / SEQ0002；PIPE0003 的 0..4 引用 FM0003 / YOLO0003 / DF0003 / CROP0003 /
  FDET0002。
- `pipeline_edges` **0 行**：三条保留链路未显式声明连线（全走代码默认取数规则）；
  只有经设计器保存过显式连线的链路才会落行（设计稿 `design/graph_chain.md` §3.3）。

### 5.5 表：`pipeline_edges`（链路连线，图编排 P2-1）

```sql
CREATE TABLE IF NOT EXISTS pipeline_edges (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  pipeline_id   VARCHAR(24) NOT NULL,
  dst_position  INT NOT NULL,             -- 下游阶段位置（= stages 数组下标）
  dst_module_id VARCHAR(32) NOT NULL,     -- 冗余：便于审计/兼容旧读法
  dst_port      VARCHAR(32) NOT NULL,
  src_position  INT NULL,                 -- 上游阶段位置；NULL=帧源
  artifact      VARCHAR(96) NOT NULL,     -- 产物全名 {module}#{position}.{artifact}
  edge_seq      INT NOT NULL DEFAULT 0,   -- 同端口多入边排序（fan_in 端口，P2-3 启用）
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_edge (pipeline_id, dst_position, dst_port, src_position, artifact),
  KEY idx_pipe (pipeline_id),
  CONSTRAINT fk_edge_pipe FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

| 字段 | 含义 | 说明 |
|---|---|---|
| dst_position / dst_port | 连线终点 | 下游阶段位置 + 输入端口名 |
| src_position | 连线起点 | 上游阶段位置；由产物全名的 `#position` 解析而来 |
| artifact | 产物名 | 规范全名 `{module}#{position}.{artifact}`；短名在保存时**原样保留**（往返一致），解析时按唯一实例归一 |
| edge_seq | 同端口序号 | 仅 `fan_in=1` 的端口会有 `>0`（本期无此类端口） |

- **只存显式声明**：默认取数规则仍属代码契约，不落行（避免双真相源）。
- **单上游不变式**由写入期校验承担（读 `module_ports.fan_in`），`uk_edge` 只防完全重复的边
  ——MySQL 唯一键无法条件化，硬凹会牺牲可读性。
- 保存链路时与本表**同一事务**重建（插删阶段后 `position` 随数组下标重排）。
- 迁移：`web_lab/migrate_stage_inputs.py`（`--dry-run` / `--only <PIPE号>`，幂等；
  旧 `module_id` 键遇多实例时返回 1 并跳过，要求人工确认，不静默猜）。

### 5.6 表：`module_ports`（端口能力目录，图编排 P2-1）

```sql
CREATE TABLE IF NOT EXISTS module_ports (
  module_id   VARCHAR(32) NOT NULL,
  direction   ENUM('in','out') NOT NULL,
  port_name   VARCHAR(32) NOT NULL,
  io_type     VARCHAR(16) NOT NULL,          -- Image | Images | Records
  fan_in      TINYINT(1) NOT NULL DEFAULT 0, -- 1=允许多条入边（过渡节点用）
  resolved_by ENUM('pipe','driver') NOT NULL DEFAULT 'pipe',
  PRIMARY KEY (module_id, direction, port_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
```

- **真相源仍是代码**（模块类 `INPUTS`/`OUTPUTS` + `composer.ARTIFACT_TYPES`）；本表是
  启动时 `db.upsert_module_ports(server.module_ports_catalog())` 幂等 upsert 的**可查询副本**，
  与代码不一致即启动告警并以代码为准收敛。
- 当前 **21 行**（普通模块入端口 `fan_in=0`）；`resolved_by='driver'` 仅 `vlm.dets`
  （`vlm.image` 走 `image_source`，P2-4 才归零）。
- `fan_in` / `Images` / `resolved_by` 为 P2-3/P2-4 预置，本期建好不用。

## 6. 每模块一张参数表（`preset_params_<module_id>`，列式存储）

模块参数**不整份塞进 `params_json`**，而是按模块拆成一张参数表，每参数一列、类型化存储。
设计动机见 [`../design/storage_design.md`](../design/storage_design.md)，这里只记结构事实。

- 表名：`preset_params_<module_id>`，主键 `preset_id`（VARCHAR(24)）引用 `presets.id`，
  外键级联删除（删 presets 行时该行参数自动清理）。
- 列：每个参数一列，类型由 server.py 参数类型映射：

  | 参数 type | 列类型 | 说明 |
  |---|---|---|
  | int | `INT` | 数值输入 |
  | float | `DOUBLE` | 阈值、比例等 |
  | bool | `TINYINT(1)` | 复选框，存 0/1 |
  | select | `VARCHAR(64)` | 下拉选项（值短） |
  | file | `VARCHAR(255)` | 文件/路径选择 |
  | str | `VARCHAR(255)` | 短文本 |
  | text | `TEXT` | 脚本、时间线、提示词等长文本 |

- 当前注册 **11** 张（仅产品模块，server.py 启动时按 `hidden` 过滤后注册）：
  det_filter / seq_filter / annotate / face_detect / face_embed / face_store / frame_manager /
  plate_recog / target_crop / vlm / yolo。
- **隐藏运行器不注册**：chain / face_monitor / frame_yolo（另有 no-params 的 pipeline_run）
  是 web 端运行入口而非产品模块，不建参数表、其参数 preset 不落库（保存时报
  "schema 未注册"）。
- **schema 双向收敛（加参数自动补列）**：`init_db()` 对每个已注册模块调 `_ensure_module_table`——
  建表（`CREATE TABLE IF NOT EXISTS`）后用 `SHOW COLUMNS` 比对现有列，缺列自动
  `ALTER TABLE ... ADD COLUMN ... NULL` 补上（如 2026-10-16 新增的 `vlm.image_pick` 列，
  重启 / 换机拉代码即自动补）；参数删除或标记 `test_only` 后其历史列
  自动 `DROP COLUMN`（`_prune_stale_columns`，2026-09-20 起，原"删参数不删列"约定作废）。
- **test_only 测试参数不落库**：测试图片 / 演示回填字段 / 事件脚本等仅存在于 web 模块
  测试页（PARAMS 里标 `"test_only": true`，前端单独分区展示），不注册 schema、不建列；
  链路设计器阶段表单也不显示。
- **下线模块清理**：`db.py _PRUNED_MODULES`（frame_scheduler / compose / utest / chain /
  face_monitor / frame_yolo / extract_faces / alarm）在启动时幂等清理其参数表与 presets 行。

## 7. 链路播种清单（`_seed_pipeline` → `_seed_chain`）

服务器启动时（`main()` 里 `init_db()` 之后）自动调用 `web_lab/server.py` 的
`_seed_pipeline()`。**幂等**：同名已存在则跳过（尊重用户在 web 端的编辑）；
每条链路经 `_seed_chain()` 按需创建各阶段 preset，**已存在链路阶段结构变化时自动重建**
（按模块复用旧 preset、补建缺失阶段、丢弃下线阶段）。

播种三条保留链路（各阶段 preset 与引用）：

- **警用无人机链路 PIPE0001**（6 步）：帧管理→YOLO→检测筛选→标注→VLM研判→时序判定（尾），
  引用 FM0001→YOLO0001→DF0001→ANNO0002→VLM0001→SEQ0001
  （frame_skip=15；YOLO imgsz=1280 / conf=0.25 / classes=0,2；检测筛选人≥2、车≥1；
  annotate 把检出框画在原帧上供 VLM 看"原图+画框"；VLM 结果经 `publish` 走 seq_filter
  的 rules 事件尾——词表命中 + 按警情类型独立冷却 30s）。
- **车牌识别链路 PIPE0002**（6 步）：帧管理→YOLO识车→检测筛选→目标裁剪(裁车)→车牌识别→时序判定，
  引用 FM0002→YOLO0002→DF0002→CROP0002→PLATE0002→SEQ0002（YOLO 只识车 classes=2 提供车辆框；
  检测筛选车长边≥40；target_crop 按车辆框裁车放大 2×；plate_recog 从车图裁车牌 + 识别 →
  坐标还原 → 原分辨率提取车牌小图；时序判定 rules `kind==plate` + 同车牌一次出事件）。
- **人脸截取链路 PIPE0003**（5 步）：帧管理→YOLO识人→检测筛选→目标裁剪(裁人)→人脸检测，
  引用 FM0003→YOLO0003→DF0003→CROP0003→FDET0002（YOLO 只识人 classes=0；
  检测筛选人长边≥30；target_crop 按人框裁人；face_detect 阶段（FaceStage）在子图里检脸 +
  从原图出原分辨率人脸；无嵌入/检索/告警——逐帧提取原分辨率人脸落盘即目的）。

另有**独立模块默认 preset**（`_SEED_STANDALONE`，不入任何链路步骤）：人脸检测 / 人脸嵌入 /
向量底库 / 目标裁剪 / 标注 / 车牌识别 各一份「·默认」。合计 17（链路阶段）+ 6（独立）= 23。

播种来源已收敛到这一处（原 server.py 双链路种子、tests/police_uav_config.py、
tests/face_dedup_config.py 均已移除）。

## 8. 行为约定（presets 读写语义）

- **命名即身份**：同 kind + name 再次保存 = 覆盖（UPDATE），返回 `created=false`；
  新名字则 INSERT（自动分配前缀 ID），返回 `created=true`。
- **定向更新**：给定 `preset_id` 时按 id 覆盖参数（id 稳定、链路引用不漂移），
  名称留空沿用原名——链路设计器"单模块保存"即走此路径。
- **module 参数读写走模块参数表**：save 事务内 upsert `presets`（params_json 置空）+
  对应 `preset_params_<module_id>`（参数按类型强转后写入）；load/list 从参数表组装回
  `{param: 值}` 对象返回。pipeline 类型仍 JSON。
- **列表**：按 `updated_at DESC, id DESC` 倒序。
- **取用**：按 id 返回完整记录；id 不存在时报错。
- **删除**：幂等，删不存在的 id 不报错；module 行的参数表行由外键级联清理。
- **容错**：未注册模块的 module 行参数按空对象 `{}` 处理；`params_json` 无法解析时按 `{}`
  处理，不抛异常。

## 9. 与 HTTP 接口的对应

| 接口 | 方法 | 调用的 db 函数 | 说明 |
|---|---|---|---|
| /api/presets?kind=module | GET | list_presets | 列出某类配置 |
| /api/presets/save | POST | save_preset | 保存/覆盖（body: kind/name/module_id/params，可选 preset_id 定向更新） |
| /api/presets/load | POST | load_preset | 按 id 取回 |
| /api/presets/delete | POST | delete_preset | 按 id 删除 |
| /api/pipelines | GET | list_pipelines | 链路列表（含每步 preset 引用与模块名） |
| /api/pipelines/save | POST | save_pipeline | 保存/覆盖（body: name/steps:["YOLO0001",…]/streams/budget/meta） |
| /api/pipelines/load | POST | load_pipeline | 按 id 取回完整链路（stages 带参数） |
| /api/pipelines/delete | POST | delete_pipeline | 按 id 删除（级联删步骤） |
| /api/db/tables | GET | table_snapshots | 各表只读快照（presets / 各模块参数表 / pipelines / steps），DB 不可用时 `{ok:false}` |

## 10. 扩展注意事项

- 服务器启动时 `db.configure_modules(...)` 注册各模块参数 schema（来源 server.py 的 PARAMS，
  hidden 运行器与 test_only 测试参数被过滤），随后 `init_db()` 幂等建表（含旧整数 schema
  自动 DROP 重建，见 §3.2）+ 为每个已注册模块补建/补列/清理废弃列参数表 + 迁移旧 JSON
  数据，表缺失会自动补建。
- 新增模块：server.py `MODULES` 注册后，在 `db.py` `_ID_PREFIXES` 登记前缀
  （未登记时按 module_id 前 4 字符大写兜底）。
- 若要新增表（如保存测试记录、校准结果），沿用同一访问层模式：在 `db.py` 增加对应函数，
  惰性连接与 `MYSQL_*` 环境变量覆盖无需改动。
