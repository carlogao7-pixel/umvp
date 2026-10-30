# -*- coding: utf-8 -*-
"""UMVP 配置暂存 —— MySQL 访问层（web_lab 扩展）。

职责：把各模块/链路的命名配置（preset）存进 MySQL，供前端"命名暂存/取用"。

存储模型（2026-08-10 起，详见 doc/design/storage_design.md）:
  - presets 保留为"注册表"：id / kind / name / module_id / 时间戳；
  - 每个模块（module_id，如 yolo/vlm）一张参数表 preset_params_<module_id>，
    参数从 JSON 展开为类型化列（由 server.py 启动时 configure_modules 注册 schema）；
  - kind='pipeline' 仍是整份 JSON 快照（params_json），结构不固定不适合列式。

ID 规则（2026-09-20 起，见 doc/guides/database_schema.md）:
  - presets.id / pipelines.id 不再用自增整数，改"模块前缀 + 4 位序号"的业务编号，
    如 YOLO0001 / FM0001 / PIPE0001；新建时按前缀自动递增；
  - init_db 检测到旧整数 schema 时自动 DROP 全部相关表并重建（播种数据可再生）。

设计约定（与 web_lab 一致）:
  - 惰性连接：首次调用才建立连接，服务器启动不依赖 MySQL；
  - pymysql 缺失或连接失败时函数抛异常，由 server.py 统一转 {ok:false, error}，
    服务器本身照常运行（DB 不可用 ≠ web_lab 不可用）；
  - 命名即身份：同 kind 下 name 重复保存 = 覆盖旧配置（重新保存即更新）；
  - 连接参数可用 MYSQL_* 环境变量覆盖（换机器/换库零改码）。

运行实例（本项目自含，免 sudo，见 doc/guides/database_schema.md）:
  mysqld --datadir=项目/.mysql/data --socket=项目/.mysql/mysql.sock \
         --port=3307 --bind-address=127.0.0.1
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import threading

# pymysql 缺失时不阻断导入（server.py 依旧可启动，DB 接口返回明确错误）
try:
    import pymysql
    _HAS_PYMYSQL = True
except ImportError:  # pragma: no cover
    pymysql = None
    _HAS_PYMYSQL = False

_CFG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3307")),
    "user": os.environ.get("MYSQL_USER", "umvp"),
    "password": os.environ.get("MYSQL_PASSWORD", "123"),
    "db": os.environ.get("MYSQL_DB", "umvp"),
}

_conn = None  # 惰性连接

# DB 访问全局锁：ThreadingHTTPServer 多线程共享单条 pymysql 连接（连接非线程安全，
# 并发读会 Packet sequence wrong / read of closed file），所有 DB 操作在锁内串行执行。
_LOCK = threading.RLock()

# 模块参数 schema：module_id -> {参数名: 参数类型}。由 server.py 启动时 configure_modules 注册，
# db 据此建参数表 / 类型强转（设计见 doc/design/storage_design.md）。
_SCHEMAS: dict[str, dict[str, str]] = {}

# 参数类型 -> 列类型（类型映射见设计文档 §2.3）
_SQL_TYPES = {
    "int": "INT",
    "float": "DOUBLE",
    "bool": "TINYINT(1)",
    "select": "VARCHAR(64)",
    "file": "VARCHAR(255)",
    "str": "VARCHAR(255)",
    "text": "TEXT",
}

# 模块标识 / 参数名白名单（防表名/列名注入）
_MODULE_ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# 业务 ID：模块前缀（前缀 + 4 位序号，如 YOLO0001）。新模块加进 MODULES 后在此登记前缀。
_ID_PREFIXES = {
    "frame_manager": "FM",
    "yolo": "YOLO",
    "det_filter": "DF",
    "seq_filter": "SEQ",
    "target_crop": "CROP",
    "annotate": "ANNO",
    "vlm": "VLM",
    "face_detect": "FDET",
    "face_embed": "FEMB",
    "face_store": "FSTO",
    "gate_filter": "GATE",
    "extract_faces": "EXTF",
    "plate_recog": "PLATE",
    "video_embed": "VEMB",
    # 回顾检索标签表（WO-0033）：片段标签 / 视频级元数据，业务 ID 前缀
    "review_index": "RVIDX",
    "review_video": "RVVID",
}
_PIPELINE_PREFIX = "PIPE"  # 链路（pipelines 表 / kind='pipeline' 的 preset）
# 合法业务 ID 形如 YOLO0001（前缀取 _preset_prefix：显式表或 module_id 前 4 字母大写；
# {2,6} 兼容 5~6 字母前缀——如 plate_recog→PLATE0002，旧 {2,4} 会误拒存量 ID，
# 定向更新（preset_id=）路径首次踩中，WO-0024.2）
_PRESET_ID_RE = re.compile(r"^[A-Z]{2,6}\d{4}$")

# 已下线/不落库模块（不注册参数 schema、不建参数表）：init_db 时幂等清理其参数表
# 与 presets 行。
# 2026-08-18: frame_scheduler 抽帧调度并入 frame_manager（帧管理），模块合并。
# 2026-09-20: compose/utest 为历史遗留空表，一并清理。
# 2026-09-20(二): chain/face_monitor/frame_yolo 为 web 端隐藏运行器（运行入口，
# 非产品模块），参数 preset 统一不落库，空参数表清理。
# 2026-09-23: extract_faces（原分辨率提取）拆分为「目标裁剪 target_crop + 人脸检测
# face_detect」，模块下线，参数表/preset 清理。
# 2026-10: alarm（输出处理 OutputPolicy）弃用删除——输出改由 检测筛选/时序判定 事件尾
# 承担，参数表/preset 清理。
_PRUNED_MODULES = {"frame_scheduler", "compose", "utest",
                   "chain", "face_monitor", "frame_yolo", "extract_faces", "alarm"}


def _preset_prefix(module_id: str) -> str:
    """模块的业务 ID 前缀；未登记的模块取 module_id 前 4 字符大写兜底。"""
    return _ID_PREFIXES.get(module_id) or module_id[:4].upper()


def _next_preset_id(cur, module_id: str) -> str:
    """按模块前缀分配下一个 preset 业务 ID（YOLO0001 → YOLO0002…）。锁内调用。"""
    prefix = _preset_prefix(module_id)
    cur.execute("SELECT id FROM presets WHERE id LIKE %s", (prefix + "%",))
    mx = 0
    for (v,) in cur.fetchall():
        m = re.fullmatch(rf"{prefix}(\d+)", str(v))
        if m:
            mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1:04d}"


def _next_pipeline_id(cur) -> str:
    """分配下一个链路业务 ID（PIPE0001 → PIPE0002…）。锁内调用。"""
    prefix = _PIPELINE_PREFIX
    cur.execute("SELECT id FROM pipelines WHERE id LIKE %s", (prefix + "%",))
    mx = 0
    for (v,) in cur.fetchall():
        m = re.fullmatch(rf"{prefix}(\d+)", str(v))
        if m:
            mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1:04d}"


def _get_conn():
    """惰性建立连接（autocommit）。失败抛异常，由调用方处理。"""
    global _conn
    if _conn is None:
        if not _HAS_PYMYSQL:
            raise RuntimeError("未安装 pymysql（pip install pymysql cryptography）")
        _conn = pymysql.connect(host=_CFG["host"], port=_CFG["port"],
                                user=_CFG["user"], password=_CFG["password"],
                                database=_CFG["db"], charset="utf8mb4",
                                autocommit=True)
    return _conn


@contextlib.contextmanager
def _cursor():
    """持锁借出 cursor：with 存续期间独占连接，退出时释放锁（DB 操作串行化）。"""
    global _conn
    with _LOCK:
        try:
            conn = _get_conn()
            conn.ping(reconnect=True)  # 缓存连接已断开时自动重连
        except Exception:
            # 重连失败（MySQL 未启动等）：重置连接，换全新连接再试；仍失败则抛出由调用方处理
            _conn = None
            conn = _get_conn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            try:
                cur.close()
            except Exception:
                pass


_DDLS = [
    """CREATE TABLE IF NOT EXISTS presets (
  id VARCHAR(24) NOT NULL PRIMARY KEY,
  kind ENUM('module','pipeline') NOT NULL DEFAULT 'module',
  name VARCHAR(64) NOT NULL,
  module_id VARCHAR(32) NOT NULL DEFAULT '',
  params_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_kind_name (kind, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 链路拼装：链路自身字段（streams/budget/meta）+ 有序步骤引用模块 preset
    # （设计见 doc/design/storage_design.md）
    """CREATE TABLE IF NOT EXISTS pipelines (
  id VARCHAR(24) NOT NULL PRIMARY KEY,
  name VARCHAR(64) NOT NULL,
  streams INT NOT NULL DEFAULT 1,
  budget_json TEXT NOT NULL,
  meta_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE IF NOT EXISTS pipeline_steps (
  id INT AUTO_INCREMENT PRIMARY KEY,
  pipeline_id VARCHAR(24) NOT NULL,
  position INT NOT NULL,
  preset_id VARCHAR(24) NOT NULL,
  UNIQUE KEY uk_pipeline_position (pipeline_id, position),
  KEY idx_preset (preset_id),
  CONSTRAINT fk_step_pipeline FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE,
  CONSTRAINT fk_step_preset FOREIGN KEY (preset_id) REFERENCES presets(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 端口能力目录（图编排 P2-1，设计稿 doc/design/graph_chain.md §3.2）：
    # 真相源仍是代码（模块 INPUTS/OUTPUTS + composer.ARTIFACT_TYPES），本表是启动时
    # upsert 的可查询副本；与代码不一致即启动告警（避免双真相源漂移）。
    # 本期 fan_in / Images / resolved_by 建好但不用（普通模块 fan_in=0），避免 P2-3 二次改表。
    """CREATE TABLE IF NOT EXISTS module_ports (
  module_id   VARCHAR(32) NOT NULL,
  direction   ENUM('in','out') NOT NULL,
  port_name   VARCHAR(32) NOT NULL,
  io_type     VARCHAR(16) NOT NULL,
  fan_in      TINYINT(1) NOT NULL DEFAULT 0,
  resolved_by ENUM('pipe','driver') NOT NULL DEFAULT 'pipe',
  PRIMARY KEY (module_id, direction, port_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 连线表（图编排 P2-1，设计稿 doc/design/graph_chain.md §3.3）：连线成为一等公民。
    # 与 pipeline_steps 在同一事务内按提交的 stages 数组重建（position = 数组下标）。
    # uk_edge 只防「完全重复的边」（幂等保存）；「每端口单一上游」由写入期校验
    # （读 module_ports.fan_in）承担，不硬凹唯一键。只存显式声明，默认取数仍属代码契约。
    """CREATE TABLE IF NOT EXISTS pipeline_edges (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  pipeline_id   VARCHAR(24) NOT NULL,
  dst_position  INT NOT NULL,
  dst_module_id VARCHAR(32) NOT NULL,
  dst_port      VARCHAR(32) NOT NULL,
  src_position  INT NULL,
  artifact      VARCHAR(96) NOT NULL,
  edge_seq      INT NOT NULL DEFAULT 0,
  created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_edge (pipeline_id, dst_position, dst_port, src_position, artifact),
  KEY idx_pipe (pipeline_id),
  CONSTRAINT fk_edge_pipe FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 回顾检索·片段标签（WO-0033，doc/design/hybrid_retrieval.md §2.5/§4）：
    # 列名 = 标签 schema 字段名（同名同义）；标签不进向量库 npz meta。
    # (video, t_start) 唯一键与 EmbedStore 的 upsert 键一一对应。
    # frame_dets（§11 定案 6 / §2.5 原则 3）：观察层原始记录流——逐帧
    # {ts, recs:[…]}（过质量闸后的 dict 记录主干，kind=det/plate/face 多态混存，
    # 剥 model_path，单窗 >500 条截断附 truncated）；det_stats 降级为可再生缓存。
    """CREATE TABLE IF NOT EXISTS review_index (
  id VARCHAR(24) NOT NULL PRIMARY KEY,
  video VARCHAR(255) NOT NULL,
  t_start DOUBLE NOT NULL,
  t_end DOUBLE NOT NULL DEFAULT 0,
  det_stats JSON NULL,
  frame_dets JSON NULL,
  class_names JSON NULL,
  plate_nos JSON NULL,
  plates JSON NULL,
  events JSON NULL,
  face_count INT NOT NULL DEFAULT 0,
  description TEXT NULL,
  thumb_path VARCHAR(512) NOT NULL DEFAULT '',
  vector_model VARCHAR(64) NOT NULL DEFAULT '',
  clip_dim INT NULL,
  frame_dim INT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_video_tstart (video, t_start),
  KEY idx_review_video (video)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 回顾检索·视频级元数据（§2.6）：采集时间只存这里（video_time 可空），
    # 另有索引备注（indexed_at / 模型与维度版本 / duration），同一视频重跑覆盖。
    """CREATE TABLE IF NOT EXISTS review_video (
  id VARCHAR(24) NOT NULL PRIMARY KEY,
  video VARCHAR(255) NOT NULL,
  video_time VARCHAR(64) NOT NULL DEFAULT '',
  video_time_src VARCHAR(16) NOT NULL DEFAULT '',
  indexed_at VARCHAR(64) NOT NULL DEFAULT '',
  vector_model VARCHAR(64) NOT NULL DEFAULT '',
  clip_dim INT NULL,
  frame_dim INT NULL,
  duration DOUBLE NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_review_video (video)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]


def _legacy_id_schema(cur) -> bool:
    """检测 presets.id 是否还是旧的自增整数 schema（需 DROP 重建）。"""
    cur.execute("SELECT COLUMN_TYPE FROM information_schema.COLUMNS"
                " WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='presets' AND COLUMN_NAME='id'")
    row = cur.fetchone()
    return bool(row and str(row[0]).lower().startswith("int"))


def _drop_legacy_tables(cur) -> None:
    """DROP 全部配置相关表（presets/pipelines/pipeline_steps/pipeline_edges/
    module_ports/preset_params_*）。

    仅在检测到旧整数 ID schema 时调用：表内均为播种/测试数据（服务器启动会重新
    播种唯一保留链路），重建后即迁移到前缀编号 schema。幂等。
    pipeline_edges 对 pipelines 有外键，DROP 期间临时关闭外键检查以消除顺序依赖。
    """
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    try:
        cur.execute("SHOW TABLES")
        for (t,) in cur.fetchall():
            if t in ("presets", "pipelines", "pipeline_steps", "pipeline_edges",
                     "module_ports") or str(t).startswith("preset_params_"):
                cur.execute(f"DROP TABLE `{t}`")
    finally:
        cur.execute("SET FOREIGN_KEY_CHECKS=1")


def configure_modules(schemas: dict[str, dict[str, str]] | None) -> None:
    """注册各模块参数 schema（module_id -> {参数名: 类型}）。

    server.py 启动时调用；参数名/模块标识做白名单过滤，非法标识忽略（不注册）。
    类型不在映射表内的按 TEXT 兜底，避免建表失败。
    """
    global _SCHEMAS
    _SCHEMAS = {}
    for mid, params in (schemas or {}).items():
        if not _MODULE_ID_RE.match(mid):
            continue
        clean = {k: _SQL_TYPES.get(t, "TEXT") for k, t in (params or {}).items()
                 if _MODULE_ID_RE.match(k)}
        if clean:
            _SCHEMAS[mid] = clean


def _module_table(module_id: str) -> str:
    """模块参数表名 preset_params_<module_id>；标识非法抛错。"""
    if not _MODULE_ID_RE.match(module_id):
        raise ValueError(f"非法模块标识: {module_id!r}")
    return f"preset_params_{module_id}"


def init_db() -> None:
    """建表（幂等）+ 旧 schema 自动重建 + 迁移旧 JSON 数据。服务器启动时调用一次。

    先检测旧整数 ID schema：存在则 DROP 全部配置表（数据可再生）重建为前缀编号
    schema；然后为每个已注册模块 ensure 参数表（缺列自动 ALTER 补列），再把历史
    kind='module' 的 params_json 拆入对应参数表并清空（幂等）。
    review_* 标签表为存量库追加列（只 ADD 不动行，不丢数据）。
    """
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.TABLES"
                    " WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='presets'")
        if cur.fetchone() and _legacy_id_schema(cur):
            _drop_legacy_tables(cur)
            print("[db] 检测到旧整数 ID schema，已 DROP 重建为前缀编号 schema")
        for ddl in _DDLS:
            cur.execute(ddl)
        _ensure_review_columns(cur)
        for module_id, schema in _SCHEMAS.items():
            _ensure_module_table(cur, module_id, schema)
        _prune_stale_columns(cur)
        _migrate_legacy_module_params(cur)
        _prune_removed_modules(cur)


# review_* 表存量库自动补列登记（表名 → (列名, 列声明)…）。新列同时要进
# 上方 CREATE TABLE DDL（新库直接建全）与 _REVIEW_INDEX_FIELDS / upsert cols
# （读写接线）。幂等：缺列才 ALTER ADD COLUMN，存量行新列取 NULL/缺省，不丢数据。
_REVIEW_ENSURE_COLS: dict[str, tuple[tuple[str, str], ...]] = {
    "review_index": (
        # frame_dets（WO-0033 第四轮，§2.5 原则 3 / §11 定案 6）：观察层原始记录流
        ("frame_dets", "JSON NULL"),
    ),
}


def _ensure_review_columns(cur) -> None:
    """存量 review_* 表缺列自动补（幂等；只加列不动行）。

    新库由 _DDLS 建表时已含全部列，此处自然跳过；旧库（建表时还没有该列）
    在此 ALTER 补齐——88 行存量标签数据原样保留。
    """
    for tbl, cols in _REVIEW_ENSURE_COLS.items():
        cur.execute("SELECT 1 FROM information_schema.TABLES"
                    " WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s", (tbl,))
        if not cur.fetchone():
            continue
        cur.execute(f"SHOW COLUMNS FROM {tbl}")
        existing = {r[0] for r in cur.fetchall()}
        for name, decl in cols:
            if name not in existing:
                cur.execute(f"ALTER TABLE `{tbl}` ADD COLUMN `{name}` {decl}")
                print(f"[db] {tbl} 补列: {name}")


def _ensure_module_table(cur, module_id: str, schema: dict[str, str]) -> None:
    """确保模块参数表存在且列齐全（新增参数自动 ALTER 补列）。"""
    tbl = _module_table(module_id)
    cols = ", ".join(f"`{k}` {t}" for k, t in schema.items())
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {tbl} (
  preset_id VARCHAR(24) NOT NULL PRIMARY KEY,
  {cols},
  CONSTRAINT fk_pp_{module_id} FOREIGN KEY (preset_id) REFERENCES presets(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""")
    cur.execute(f"SHOW COLUMNS FROM {tbl}")
    existing = {r[0] for r in cur.fetchall()}
    for k, t in schema.items():
        if k not in existing:
            cur.execute(f"ALTER TABLE {tbl} ADD COLUMN `{k}` {t} NULL")


def _migrate_legacy_module_params(cur) -> None:
    """把历史 kind='module' 行的 params_json 拆入模块参数表，然后清空 params_json。

    幂等：只处理 params_json 非空的行；拆完后置空，下次跳过。模块未注册的行跳过。
    """
    cur.execute("SELECT id, module_id, params_json FROM presets"
                " WHERE kind='module' AND params_json IS NOT NULL AND params_json <> ''")
    for rid, module_id, params_json in cur.fetchall():
        schema = _SCHEMAS.get(module_id)
        if not schema:
            continue
        try:
            params = json.loads(params_json)
        except (TypeError, ValueError):
            continue
        _upsert_module_params(cur, _module_table(module_id), rid,
                              _coerce_params(params, schema))
        cur.execute("UPDATE presets SET params_json='' WHERE id=%s", (rid,))


def _prune_stale_columns(cur) -> None:
    """删掉模块参数表里已不在注册 schema 中的列（幂等）。

    列随 schema 双向收敛：加参数自动补列（_ensure_module_table），参数下线/
    标记 test_only 不落库后，其历史列在此自动删除（2026-09-20 起，原"删参数
    不删列（保守）"约定作废——配置数据可再生，一致性优先）。
    """
    for module_id, schema in _SCHEMAS.items():
        tbl = _module_table(module_id)
        cur.execute(f"SHOW COLUMNS FROM {tbl}")
        stale = {r[0] for r in cur.fetchall()} - set(schema.keys()) - {"preset_id"}
        for k in stale:
            cur.execute(f"ALTER TABLE {tbl} DROP COLUMN `{k}`")
            print(f"[db] {tbl} 清理废弃列: {k}")


def _prune_removed_modules(cur) -> None:
    """清理已下线模块（从产品移除）：删其 presets 行与参数表，幂等。

    顺序：先删引用这些 preset 的链路步骤（fk_step_preset 为 RESTRICT，直接删
    presets 会因被引用而失败），再删 presets 行（参数表行靠 ON DELETE CASCADE
    级联清掉），最后 DROP 参数表。全程幂等：无行/无表时自然跳过。
    """
    if not _PRUNED_MODULES:
        return
    for mid in _PRUNED_MODULES:
        if not _MODULE_ID_RE.match(mid):
            continue
        cur.execute("DELETE ps FROM pipeline_steps ps"
                    " JOIN presets p ON ps.preset_id = p.id"
                    " WHERE p.kind='module' AND p.module_id=%s", (mid,))
        cur.execute("DELETE FROM presets WHERE kind='module' AND module_id=%s", (mid,))
        cur.execute(f"DROP TABLE IF EXISTS `{_module_table(mid)}`")


# ---------------- 端口能力目录（module_ports，图编排 P2-1） ----------------

def upsert_module_ports(ports, warn=None) -> None:
    """把**代码声明**的端口目录 upsert 进 module_ports（启动时调用一次，幂等）。

    ports: 可迭代的 (module_id, direction, port_name, io_type, fan_in, resolved_by)。
    真相源始终是代码（模块 INPUTS/OUTPUTS + composer.ARTIFACT_TYPES）；本表是可查询副本。
    与库内现有副本不一致时经 warn(msg) 告警（默认 print），随后**以代码为准收敛**
    （upsert 缺失/变更行 + 删除陈旧行）——避免双真相源漂移。
    """
    desired: dict = {}
    for row in ports or ():
        mid, direction, port, io_type, fan_in, resolved_by = row
        desired[(mid, direction, port)] = (str(io_type), int(fan_in), str(resolved_by))
    with _cursor() as cur:
        cur.execute("SELECT module_id, direction, port_name, io_type, fan_in, resolved_by"
                    " FROM module_ports")
        existing = {(r[0], r[1], r[2]): (str(r[3]), int(r[4]), str(r[5]))
                    for r in cur.fetchall()}
        if existing != desired:
            (warn or print)(
                f"[db] module_ports 目录与代码声明不一致（库 {len(existing)} 行 vs 代码 "
                f"{len(desired)} 行）——以代码为唯一真相源收敛")
        for key in set(existing) - set(desired):
            cur.execute("DELETE FROM module_ports WHERE module_id=%s AND direction=%s"
                        " AND port_name=%s", key)
        for (mid, direction, port), (io_type, fan_in, resolved_by) in desired.items():
            cur.execute(
                "INSERT INTO module_ports (module_id, direction, port_name, io_type, fan_in,"
                " resolved_by) VALUES (%s,%s,%s,%s,%s,%s)"
                " ON DUPLICATE KEY UPDATE io_type=VALUES(io_type), fan_in=VALUES(fan_in),"
                " resolved_by=VALUES(resolved_by)",
                (mid, direction, port, io_type, fan_in, resolved_by))


def _module_port_fan_in(cur) -> dict:
    """读 module_ports 的入端口 fan_in：{(module_id, port_name): int}。缺行按 0 兜底。"""
    cur.execute("SELECT module_id, port_name, fan_in FROM module_ports WHERE direction='in'")
    return {(r[0], r[1]): int(r[2]) for r in cur.fetchall()}


# ---------------- 链路连线（pipeline_edges，图编排 P2-1） ----------------
# position = stages 数组下标（可移植，不用 pipeline_steps.id）；产物名规范形态为全名
# {module}#{position}.{artifact}（短名 {module}.{artifact} 在唯一实例时亦可，按位置解析）。

def _positions_by_module(module_ids) -> dict:
    """module_id → 该模块在链路中的位置列表（数组下标升序）。"""
    out: dict = {}
    for i, mid in enumerate(module_ids):
        out.setdefault(mid, []).append(i)
    return out


def _parse_artifact_pos(art, positions_by_module) -> tuple:
    """产物名 → (module_id, position)。

    - 全名 ``{module}#{position}.{artifact}`` → 取解析位（module 部分原样）；
    - 短名 ``{module}.{artifact}`` → 该模块在本链路**唯一实例**时取该位置；
      多实例歧义 / 未知模块 / 非法名 → 抛 ValueError（**不猜**，与 composer 短名规则一致）。
    """
    if not isinstance(art, str) or "." not in art:
        raise ValueError(f"非法产物名: {art!r}")
    head, _sep, _tail = art.partition(".")
    mod, hashsep, pos = head.partition("#")
    if hashsep:
        if not mod or not pos.isdigit():
            raise ValueError(f"非法产物全名: {art!r}（应为 {{module}}#{{position}}.{{artifact}}）")
        return mod, int(pos)
    if not mod:
        raise ValueError(f"非法产物名: {art!r}")
    poss = positions_by_module.get(mod) or []
    if not poss:
        raise ValueError(f"未知模块 {mod!r}（产物 {art!r}）")
    if len(poss) > 1:
        raise ValueError(
            f"产物短名 {art!r} 有歧义：模块 {mod!r} 在本链路有 {len(poss)} 个实例"
            f"（位置 {poss}）；多实例须写全名 {mod}#{poss[0]}.{_tail}")
    return mod, poss[0]


def canonical_stage_inputs(stage_inputs, positions_by_module) -> dict:
    """把 stage_inputs 规范为 ``{"#<position>": {端口: 产物名 | [产物名...]}}``。

    键容忍旧式 ``module_id``（该模块唯一实例时映射到其位置；多实例报错不猜）；
    产物名**原样保留**（不解析短/全名）——保证 ``save → load`` 往返逐字节一致。
    非对象 / 空对象 / 非字符串产物名（含空串）跳过。
    """
    if not isinstance(stage_inputs, dict):
        return {}
    out: dict = {}
    for key, ports in stage_inputs.items():
        if not isinstance(ports, dict) or not ports:
            continue
        if isinstance(key, str) and key.startswith("#") and key[1:].isdigit():
            pos = int(key[1:])
        else:
            poss = positions_by_module.get(key) or []
            if len(poss) > 1:
                raise ValueError(
                    f"stage_inputs 键 {key!r} 指向多实例模块（位置 {poss}），"
                    f"须改用位置串 '#<position>'")
            if not poss:
                continue
            pos = poss[0]
        clean: dict = {}
        for p, a in ports.items():
            if not isinstance(p, str):
                continue
            if isinstance(a, str) and a.strip():
                clean[p] = a
            elif isinstance(a, list):
                vals = [x for x in a if isinstance(x, str) and x.strip()]
                if vals:
                    clean[p] = vals
        if clean:
            out[f"#{pos}"] = clean
    return out


def _edges_from_inputs(canonical, module_ids, fan_in) -> list:
    """规范后的 stage_inputs → pipeline_edges 行。

    写入期校验「单上游端口是否被接了第二条」：端口 fan_in=0 时只许单条入边（值须字符串），
    fan_in=1 才允许多条（本期无此类端口）。返回 [(dst_position, dst_module_id, dst_port,
    src_position, artifact, edge_seq)]。
    """
    positions_by_module = _positions_by_module(module_ids)
    n = len(module_ids)
    edges: list = []
    for key, ports in canonical.items():
        dst_position = int(key[1:])
        if dst_position < 0 or dst_position >= n:
            raise ValueError(f"连线目标位置 {dst_position} 超出链路阶段范围（0..{n - 1}）")
        dst_module_id = module_ids[dst_position]
        for port, art in ports.items():
            arts = art if isinstance(art, list) else [art]
            if len(arts) > 1 and not fan_in.get((dst_module_id, port), 0):
                raise ValueError(
                    f"端口 {dst_module_id}.{port} 为单一上游（fan_in=0），不接受 "
                    f"{len(arts)} 条入边: {arts}；如需多上游请改用支持 fan-in 的过渡节点")
            for seq, a in enumerate(arts):
                _mod, src_position = _parse_artifact_pos(a, positions_by_module)
                edges.append((dst_position, dst_module_id, port, src_position, a, seq))
    return edges


# ---------------- 值转换（保存强转 / 读取还原） ----------------

def _coerce_params(params: dict, schema: dict[str, str]) -> dict:
    """保存前把前端参数按列类型强转；空串/缺失 -> NULL。

    前端表单 int/float 输入返回字符串，bool 返回真布尔；这里统一转成列类型。
    """
    out = {}
    for k, t in schema.items():
        v = params.get(k)
        if v is None or v == "":
            out[k] = None
            continue
        if t == "INT":
            out[k] = int(v)
        elif t == "DOUBLE":
            out[k] = float(v)
        elif t == "TINYINT(1)":
            out[k] = 1 if (v if isinstance(v, bool)
                           else str(v).lower() in ("1", "true", "yes", "on")) else 0
        else:
            out[k] = str(v)
    return out


def _restore_params(values: dict, schema: dict[str, str]) -> dict:
    """读取后把列值还原为 Python 类型（bool 从 0/1 还原）。"""
    out = {}
    for k, t in schema.items():
        v = values.get(k)
        if v is None:
            out[k] = None
        elif t == "INT":
            out[k] = int(v)
        elif t == "DOUBLE":
            out[k] = float(v)
        elif t == "TINYINT(1)":
            out[k] = bool(v)
        else:
            out[k] = str(v)
    return out


def _insert_module_params(cur, tbl: str, preset_id: str, values: dict) -> None:
    cols = ", ".join(f"`{k}`" for k in values)
    ph = ", ".join(["%s"] * len(values))
    cur.execute(f"INSERT INTO {tbl} (preset_id, {cols}) VALUES (%s, {ph})",
                [preset_id] + list(values.values()))


def _upsert_module_params(cur, tbl: str, preset_id: str, values: dict) -> None:
    """按 preset_id 更新参数行；不存在则插入（同名覆盖 / 迁移用）。

    先查存在性再决定 UPDATE/INSERT：UPDATE 后靠 rowcount 判断会把"参数与旧值
    完全相同"的覆盖误判为不存在（MySQL 无 CLIENT_FOUND_ROWS 时 UPDATE 相同的
    值 affected=0），导致重复 INSERT 撞主键。
    """
    cur.execute(f"SELECT 1 FROM {tbl} WHERE preset_id=%s", [preset_id])
    if cur.fetchone():
        set_clause = ", ".join(f"`{k}`=%s" for k in values)
        cur.execute(f"UPDATE {tbl} SET {set_clause} WHERE preset_id=%s",
                    list(values.values()) + [preset_id])
    else:
        _insert_module_params(cur, tbl, preset_id, values)


def list_presets(kind: str = "module") -> list[dict]:
    """按 updated_at 倒序列出指定 kind 的配置。

    module 类型从模块参数表组装 params（按 module_id 分组批量取）；
    pipeline 类型仍从 params_json 解析。
    """
    with _cursor() as cur:
        cur.execute(
            "SELECT id, name, module_id, params_json FROM presets"
            " WHERE kind=%s ORDER BY updated_at DESC, id DESC", (kind,))
        rows = cur.fetchall()
        params_map: dict[str, dict] = {}
        if kind == "module":
            by_module: dict[str, list[int]] = {}
            for rid, _n, module_id, _pj in rows:
                by_module.setdefault(module_id, []).append(rid)
            for mid, ids in by_module.items():
                schema = _SCHEMAS.get(mid)
                if not schema:
                    continue
                tbl = _module_table(mid)
                cols = ", ".join(f"`{k}`" for k in schema)
                ph = ", ".join(["%s"] * len(ids))
                cur.execute(f"SELECT preset_id, {cols} FROM {tbl}"
                            f" WHERE preset_id IN ({ph})", tuple(ids))
                for prow in cur.fetchall():
                    params_map[prow[0]] = _restore_params(
                        dict(zip(schema.keys(), prow[1:])), schema)
    out = []
    for rid, name, module_id, params_json in rows:
        if kind == "module":
            if rid in params_map:
                params = params_map[rid]
            elif module_id not in _SCHEMAS:
                params = _loads(params_json)  # 未注册模块：回退旧 JSON
            else:
                params = {}
        else:
            params = _loads(params_json)
        out.append({"id": rid, "name": name, "module_id": module_id,
                    "params": params})
    return out


def save_preset(kind: str, name: str, module_id: str, params: dict,
                preset_id: str | None = None) -> tuple[str, bool]:
    """保存命名配置；同名覆盖旧配置。返回 (业务 id 如 YOLO0001, 是否新建)。

    新建时按模块前缀自动编号（YOLO0001→YOLO0002…；pipeline 类型用 PIPE 前缀）。
    给定 preset_id 时按 id 定向更新（保持 id 稳定，链路上游引用不漂移，不新建）；
    且名称留空时沿用原配置名（连名字都不改，只覆盖参数），供链路设计器"单模块保存"
    复用——改完参数保存回同一 preset，id 不变、链路层引用不受影响。
    否则沿用"同 kind+name 覆盖或新建"（模块库首存场景）。
    module 类型：事务内 upsert 注册表行（params_json 置空）+ 参数表行（拆列）；
    pipeline 类型：整份 params 存 params_json（结构不固定，不适合列式）。
    """
    name = (name or "").strip()
    if len(name) > 64:
        raise ValueError("配置名过长（最多 64 字符）")
    if preset_id is not None and not _PRESET_ID_RE.match(preset_id):
        raise ValueError(f"非法配置 ID（应为前缀+4位序号，如 YOLO0001）: {preset_id!r}")
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            conn.begin()
            if preset_id is not None:
                # 定向更新：保持 id 稳定（链路上游引用不漂移）。先确认存在并校验类型。
                cur.execute("SELECT kind, name FROM presets WHERE id=%s", (preset_id,))
                prow = cur.fetchone()
                if prow is None:
                    raise ValueError(f"配置不存在: id={preset_id}")
                if kind == "module" and prow[0] != "module":
                    raise ValueError(f"id={preset_id} 不是模块配置（kind={prow[0]}）")
                if kind == "pipeline" and prow[0] != "pipeline":
                    raise ValueError(f"id={preset_id} 不是链路配置（kind={prow[0]}）")
                # 没写名称 → 沿用原名：仅覆盖参数，名字与 id 都稳定，不影响链路层
                if not name and prow[1]:
                    name = prow[1]
                row = (preset_id,)
            else:
                if not name:
                    raise ValueError("配置名不能为空")
                cur.execute("SELECT id FROM presets WHERE kind=%s AND name=%s", (kind, name))
                row = cur.fetchone()
            if not name:
                raise ValueError("配置名不能为空")
            if kind == "module":
                schema = _SCHEMAS.get(module_id)
                if not schema:
                    raise ValueError(f"模块参数 schema 未注册: {module_id}")
                values = _coerce_params(params, schema)
                tbl = _module_table(module_id)
                if row:
                    pid = row[0]
                    cur.execute("UPDATE presets SET module_id=%s, name=%s, params_json=''"
                                " WHERE id=%s", (module_id, name, pid))
                    _upsert_module_params(cur, tbl, pid, values)
                    created = False
                else:
                    pid = _next_preset_id(cur, module_id)
                    cur.execute("INSERT INTO presets (id, kind, name, module_id, params_json)"
                                " VALUES (%s, %s, %s, %s, '')", (pid, kind, name, module_id))
                    _insert_module_params(cur, tbl, pid, values)
                    created = True
            else:
                params_json = json.dumps(params, ensure_ascii=False)
                if row:
                    pid = row[0]
                    cur.execute("UPDATE presets SET params_json=%s WHERE id=%s",
                                (params_json, pid))
                    created = False
                else:
                    pid = _next_preset_id(cur, "pipeline")
                    cur.execute("INSERT INTO presets (id, kind, name, module_id, params_json)"
                                " VALUES (%s, %s, %s, '', %s)", (pid, kind, name, params_json))
                    created = True
            conn.commit()
            return pid, created
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def load_preset(preset_id: str) -> dict:
    """按业务 id 取回配置：{id, kind, name, module_id, params}。不存在抛 ValueError。

    module 类型从参数表组装 params；pipeline 类型解析 params_json。
    """
    with _cursor() as cur:
        cur.execute("SELECT id, kind, name, module_id, params_json"
                    " FROM presets WHERE id=%s", (preset_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"配置不存在: id={preset_id}")
        rid, kind, name, module_id, params_json = row
        if kind == "module":
            schema = _SCHEMAS.get(module_id)
            if schema:
                tbl = _module_table(module_id)
                cols = ", ".join(f"`{k}`" for k in schema)
                cur.execute(f"SELECT {cols} FROM {tbl} WHERE preset_id=%s", (rid,))
                prow = cur.fetchone()
                params = (_restore_params(dict(zip(schema.keys(), prow)), schema)
                          if prow else {})
            else:
                params = _loads(params_json)
        else:
            params = _loads(params_json)
    return {"id": rid, "kind": kind, "name": name, "module_id": module_id,
            "params": params}


def delete_preset(preset_id: str) -> None:
    """删除配置；不存在也不报错（幂等）。被链路引用时拒绝（引用完整性）。"""
    with _cursor() as cur:
        cur.execute("SELECT p.name FROM pipeline_steps ps"
                    " JOIN pipelines p ON p.id = ps.pipeline_id"
                    " WHERE ps.preset_id=%s ORDER BY p.name", (preset_id,))
        refs = [r[0] for r in cur.fetchall()]
        if refs:
            shown = "、".join(refs[:3]) + ("…" if len(refs) > 3 else "")
            raise ValueError(f"该配置正被链路引用（{shown}），先移除引用再删除")
        cur.execute("DELETE FROM presets WHERE id=%s", (preset_id,))


def _loads(text: str) -> dict:
    """JSON 文本解析，失败按 {} 处理（沿用 list_presets 的容错）。"""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {}


def list_pipelines() -> list[dict]:
    """列出全部链路（含每步的 preset 引用与模块名），按最近更新倒序。"""
    with _cursor() as cur:
        cur.execute("SELECT id, name, streams, budget_json, meta_json, updated_at"
                    " FROM pipelines ORDER BY updated_at DESC, id DESC")
        rows = cur.fetchall()
        steps: dict[str, list[dict]] = {}
        cur.execute("SELECT ps.pipeline_id, ps.position, ps.preset_id, p.module_id"
                    " FROM pipeline_steps ps JOIN presets p ON p.id = ps.preset_id"
                    " ORDER BY ps.pipeline_id, ps.position")
        for pid, position, preset_id, module_id in cur.fetchall():
            steps.setdefault(pid, []).append(
                {"position": position, "preset_id": preset_id, "module_id": module_id})
    out = []
    for rid, name, streams, budget_json, meta_json, _ in rows:
        out.append({"id": rid, "name": name, "streams": streams,
                    "budget": _loads(budget_json), "meta": _loads(meta_json),
                    "steps": steps.get(rid, [])})
    return out


def save_pipeline(name: str, preset_ids: list[str], streams: int = 1,
                  budget: dict | None = None, meta: dict | None = None) -> tuple[str, bool]:
    """保存链路（步骤引用模块 preset）；同名覆盖。返回 (业务 id 如 PIPE0001, 是否新建)。

    preset_ids 的顺序即链路步骤顺序（position 0..n-1，= stages 数组下标）；事务内**同时重建**
    pipeline_steps 与 pipeline_edges（图编排 P2-1），保证两者位置一致、原子提交。
    步骤只能引用 kind='module' 的配置；**允许同模块多实例**（每模块不再限制一个阶段）。
    连线（meta.stage_inputs）双写：规范为位置键落 pipeline_edges，同时回写 meta_json。
    写入期按 module_ports.fan_in 校验「单上游端口是否被接了第二条」（fan_in=0 拒多入边）。
    新建时按 PIPE 前缀自动编号。
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("链路名不能为空")
    if len(name) > 64:
        raise ValueError("链路名过长（最多 64 字符）")
    if not preset_ids:
        raise ValueError("链路至少需要一个模块步骤")
    budget_json = json.dumps(budget or {}, ensure_ascii=False)
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            # 校验步骤引用的 preset 存在、为 module 类型（重复模块已放行，按位置装配）
            placeholders = ",".join(["%s"] * len(preset_ids))
            cur.execute(f"SELECT id, kind, module_id FROM presets WHERE id IN ({placeholders})",
                        tuple(preset_ids))
            found = {rid: (kind, mid) for rid, kind, mid in cur.fetchall()}
            for rid in preset_ids:
                if rid not in found:
                    raise ValueError(f"步骤引用的配置不存在: id={rid}")
                if found[rid][0] != "module":
                    raise ValueError(f"步骤只能引用模块配置（kind='module'），id={rid} 是 {found[rid][0]}")
            module_ids = [found[rid][1] for rid in preset_ids]
            positions_by_module = _positions_by_module(module_ids)
            # 连线：规范为位置键（产物名原样保留 → save/load 往返逐字节一致），双写 meta + edges
            meta = {**(meta or {})}
            canonical = canonical_stage_inputs(meta.get("stage_inputs"), positions_by_module)
            if canonical or "stage_inputs" in meta:
                meta["stage_inputs"] = canonical
            meta_json = json.dumps(meta, ensure_ascii=False)
            conn.begin()
            cur.execute("SELECT id FROM pipelines WHERE name=%s", (name,))
            row = cur.fetchone()
            if row:
                pid = row[0]
                cur.execute("UPDATE pipelines SET streams=%s, budget_json=%s, meta_json=%s"
                            " WHERE id=%s", (streams, budget_json, meta_json, pid))
                cur.execute("DELETE FROM pipeline_steps WHERE pipeline_id=%s", (pid,))
                created = False
            else:
                pid = _next_pipeline_id(cur)
                cur.execute("INSERT INTO pipelines (id, name, streams, budget_json, meta_json)"
                            " VALUES (%s, %s, %s, %s, %s)",
                            (pid, name, streams, budget_json, meta_json))
                created = True
            for position, preset_id in enumerate(preset_ids):
                cur.execute("INSERT INTO pipeline_steps (pipeline_id, position, preset_id)"
                            " VALUES (%s, %s, %s)", (pid, position, preset_id))
            # 连线随 steps 在同一事务内重建；写入期 fan_in 校验在此触发（DB 唯一键只防重复边）
            fan_in = _module_port_fan_in(cur)
            edges = _edges_from_inputs(canonical, module_ids, fan_in)
            cur.execute("DELETE FROM pipeline_edges WHERE pipeline_id=%s", (pid,))
            for dst_position, dst_module_id, dst_port, src_position, artifact, edge_seq in edges:
                cur.execute(
                    "INSERT INTO pipeline_edges (pipeline_id, dst_position, dst_module_id,"
                    " dst_port, src_position, artifact, edge_seq)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (pid, dst_position, dst_module_id, dst_port, src_position, artifact, edge_seq))
            conn.commit()
            return pid, created
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def _attach_stage_inputs(stages: list[dict], stage_inputs) -> list[dict]:
    """把位置键 stage_inputs（{"#<position>": {端口: 上游产物名}}）回填到各 stage 的 inputs。

    纯函数（无 DB 依赖，便于单测）：按 stages 数组下标取对应位置串；只对连线非空的阶段注入；
    旧链路（无 stage_inputs）保持原样 → 零回归。返回同一 stages 列表（就地更新）。
    """
    if not isinstance(stage_inputs, dict):
        return stages
    for i, st in enumerate(stages):
        ins = stage_inputs.get(f"#{i}")
        if isinstance(ins, dict) and ins:
            st["inputs"] = ins
    return stages


def _edges_to_stage_inputs(edge_rows) -> dict:
    """pipeline_edges 行 → {"#<dst_position>": {dst_port: artifact | [artifact...]}}。

    同端口多条入边（fan_in=1 端口，本期无）→ 值为列表，保序（edge_seq）。
    """
    out: dict = {}
    for dst_position, dst_port, artifact, _seq in edge_rows:
        ports = out.setdefault(f"#{dst_position}", {})
        if dst_port in ports:
            prev = ports[dst_port]
            ports[dst_port] = (prev if isinstance(prev, list) else [prev]) + [artifact]
        else:
            ports[dst_port] = artifact
    return out


def load_pipeline(pipeline_id: str) -> dict:
    """按 id 取回完整链路：{id, name, streams, budget, meta, stages}。

    stages 为有序步骤 [{position, preset_id, module_id, params}]，可直接生成
    P2 pipeline spec 的 stages=[{module, params}]。params 从模块参数表取回。
    连线读取优先级（图编排 P2-1）：pipeline_edges > meta.stage_inputs > 代码默认；
    取回后按位置回填到各 stage 的 inputs → 形状与只读 meta 时逐字节一致。
    """
    with _cursor() as cur:
        cur.execute("SELECT id, name, streams, budget_json, meta_json"
                    " FROM pipelines WHERE id=%s", (pipeline_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"链路不存在: id={pipeline_id}")
        pid, name, streams, budget_json, meta_json = row
        cur.execute("SELECT ps.position, ps.preset_id, p.module_id"
                    " FROM pipeline_steps ps JOIN presets p ON p.id = ps.preset_id"
                    " WHERE ps.pipeline_id=%s ORDER BY ps.position", (pid,))
        steps = cur.fetchall()  # (position, preset_id, module_id)
        # 按 module_id 分组，批量取各步骤的参数（模块参数表）
        params_map: dict[str, dict] = {}
        by_module: dict[str, list[tuple[int, int]]] = {}
        for position, preset_id, module_id in steps:
            by_module.setdefault(module_id, []).append((position, preset_id))
        for mid, items in by_module.items():
            schema = _SCHEMAS.get(mid)
            if not schema:
                continue
            tbl = _module_table(mid)
            ids = [p for _, p in items]
            cols = ", ".join(f"`{k}`" for k in schema)
            ph = ", ".join(["%s"] * len(ids))
            cur.execute(f"SELECT preset_id, {cols} FROM {tbl} WHERE preset_id IN ({ph})",
                        tuple(ids))
            for prow in cur.fetchall():
                params_map[prow[0]] = _restore_params(
                    dict(zip(schema.keys(), prow[1:])), schema)
        cur.execute("SELECT dst_position, dst_port, artifact, edge_seq FROM pipeline_edges"
                    " WHERE pipeline_id=%s ORDER BY dst_position, dst_port, edge_seq", (pid,))
        edge_rows = cur.fetchall()
    stages = [{"position": position, "preset_id": preset_id, "module_id": module_id,
               "params": params_map.get(preset_id, {})}
              for position, preset_id, module_id in steps]
    meta = _loads(meta_json)
    # 端口连线（P5 第 2 步 + 图编排 P2-1）：优先读 pipeline_edges；无连线时回退 meta.stage_inputs
    # （均规范为位置键）；再回填到对应 stage 的 inputs → 直接生成带连线的 P2 spec。
    if edge_rows:
        stage_inputs = _edges_to_stage_inputs(edge_rows)
    else:
        stage_inputs = canonical_stage_inputs(
            meta.get("stage_inputs") if isinstance(meta, dict) else None,
            _positions_by_module([s["module_id"] for s in stages]))
    _attach_stage_inputs(stages, stage_inputs)
    return {"id": pid, "name": name, "streams": streams,
            "budget": _loads(budget_json), "meta": meta,
            "stages": stages}


def delete_pipeline(pipeline_id: str) -> None:
    """删除链路（steps 级联删除）；不存在也不报错（幂等）。"""
    with _cursor() as cur:
        cur.execute("DELETE FROM pipeline_steps WHERE pipeline_id=%s", (pipeline_id,))
        cur.execute("DELETE FROM pipelines WHERE id=%s", (pipeline_id,))


# ---------------- 可视化：各表快照（GET /api/db/tables 数据源） ----------------

def _jsonable(v):
    """datetime/date 转字符串（json.dumps 直接序列化 datetime 会失败）。"""
    if isinstance(v, (datetime.datetime, datetime.date)):
        return str(v)
    return v


def table_snapshots() -> dict:
    """返回各表完整内容快照，供前端"数据浏览"页渲染。

    {presets: {columns, rows}, modules: [{module_id, columns, rows, count}],
     pipelines: {columns, rows}, steps: {columns, rows}}。只读。
    """
    with _cursor() as cur:
        cur.execute("SELECT id, kind, name, module_id, created_at, updated_at"
                    " FROM presets ORDER BY updated_at DESC, id DESC")
        presets_rows = [[_jsonable(v) for v in r] for r in cur.fetchall()]
        cur.execute("SELECT id, name, streams, budget_json, meta_json,"
                    " created_at, updated_at FROM pipelines"
                    " ORDER BY updated_at DESC, id DESC")
        pipelines_rows = [[_jsonable(v) for v in r] for r in cur.fetchall()]
        cur.execute("SELECT s.id, s.pipeline_id, s.position, s.preset_id,"
                    " p.module_id, p.name"
                    " FROM pipeline_steps s LEFT JOIN presets p ON p.id = s.preset_id"
                    " ORDER BY s.pipeline_id, s.position")
        steps_rows = [[_jsonable(v) for v in r] for r in cur.fetchall()]
        modules = []
        for mid, schema in _SCHEMAS.items():
            tbl = _module_table(mid)
            cols = list(schema.keys())
            col_sql = ", ".join(f"t.`{k}`" for k in cols)
            cur.execute(f"SELECT t.preset_id, p.name, p.updated_at, {col_sql}"
                        f" FROM {tbl} t JOIN presets p ON p.id=t.preset_id"
                        f" ORDER BY p.updated_at DESC, p.id DESC")
            rows = []
            for r in cur.fetchall():
                preset_id, pname, updated_at = r[0], r[1], r[2]
                values = _restore_params(dict(zip(cols, r[3:])), schema)
                rows.append([preset_id, pname, _jsonable(updated_at)]
                            + [values.get(k) for k in cols])
            modules.append({"module_id": mid,
                            "columns": ["id", "配置名", "更新时间"] + cols,
                            "rows": rows, "count": len(rows)})
    return {
        "presets": {"columns": ["id", "kind", "name", "module_id",
                                "created_at", "updated_at"],
                    "rows": presets_rows},
        "modules": modules,
        "pipelines": {"columns": ["id", "name", "streams", "budget", "meta",
                                  "created_at", "updated_at"],
                      "rows": pipelines_rows},
        "steps": {"columns": ["id", "pipeline_id", "position", "preset_id",
                              "module_id", "配置名"],
                  "rows": steps_rows},
    }


# ======================================================================
# 回顾检索·标签索引（WO-0033，doc/design/hybrid_retrieval.md §2.5/§2.6/§4/§12-D）
#
# 两张表：review_index（片段标签，列名=§2.5 冻结字段）与 review_video（视频级
# 元数据，采集时间只存这里）。设计约定：
#   - **无 DB 降级**：--no-db 或 MySQL 连接失败时，同结构落
#     out/review_index.json / out/review_video.json，查询走同一套接口（SQL 替身）；
#   - 业务 ID：RVIDX0001 / RVVID0001（_ID_PREFIXES 登记，前缀 + 4 位序号）；
#   - 幂等键：review_index (video, t_start)、review_video video——重跑索引覆盖；
#   - 删除视频两侧同步清理（delete_review_video）。
# ======================================================================

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# None = 尚未探测（首次调用试 DB，失败后永久回落 JSON）；server 启动成功可显式置 True。
_REVIEW_USE_DB: bool | None = None

_REVIEW_JSON_LOCK = threading.RLock()

_REVIEW_INDEX_FIELDS = (
    "video", "t_start", "t_end", "det_stats", "frame_dets", "class_names",
    "plate_nos", "plates", "events", "face_count", "description", "thumb_path",
    "vector_model", "clip_dim", "frame_dim",
)
_REVIEW_VIDEO_FIELDS = (
    "video", "video_time", "video_time_src", "indexed_at", "vector_model",
    "clip_dim", "frame_dim", "duration",
)
# review_index 的 JSON 列（写入序列化 / 读出反序列化共用清单）。
# frame_dets 是 JSON 数组（观察层原始记录流），缺省 []（与 plate_nos 等数组列同组）。
_REVIEW_JSON_COLS = ("det_stats", "frame_dets", "class_names", "plate_nos",
                     "plates", "events")


def set_review_db_enabled(enabled: bool) -> None:
    """server 启动时告知标签索引是否走 MySQL（--no-db / init_db 失败 → False）。"""
    global _REVIEW_USE_DB
    _REVIEW_USE_DB = bool(enabled)


def review_backend_mode() -> str:
    """当前标签索引后端："db"（MySQL）或 "json"（无 DB 降级）。"""
    return "db" if _review_db_available() else "json"


def _review_db_available() -> bool:
    """标签索引是否可用 MySQL；首次探测失败即永久回落 JSON（避免每次请求等连接）。"""
    global _REVIEW_USE_DB
    if _REVIEW_USE_DB is not None:
        return _REVIEW_USE_DB
    try:
        with _cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        _REVIEW_USE_DB = True
    except Exception:
        _REVIEW_USE_DB = False
    return _REVIEW_USE_DB


def _review_json_path(name: str) -> str:
    """JSON 降级文件路径（可用 UMVP_REVIEW_*_JSON 环境变量覆盖，便于测试隔离）。"""
    env = os.environ.get({"review_index": "UMVP_REVIEW_INDEX_JSON",
                          "review_video": "UMVP_REVIEW_VIDEO_JSON"}[name])
    if env:
        return env if os.path.isabs(env) else os.path.join(_ROOT_DIR, env)
    return os.path.join(_ROOT_DIR, "out", f"{name}.json")


def _review_json_read(name: str) -> list[dict]:
    path = _review_json_path(name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):          # 宽容：{"rows": [...]} 形态
        data = data.get("rows") or []
    return [dict(r) for r in data if isinstance(r, dict)]


def _review_json_write(name: str, rows: list[dict]) -> None:
    path = _review_json_path(name)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _jsonish(v, default):
    """列值 → Python 结构（JSON 文本/已是结构/None → default）。"""
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def _num_or_none(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_zero(v):
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _review_index_row(row: dict) -> dict:
    """片段标签行归一（字段缺省补类型；t_start 3 位小数对齐 EmbedStore upsert 键）。

    frame_dets = 观察层原始记录流（§2.5）：缺省 []；存量为 NULL 的旧行也归一为 []。
    """
    return {
        "video": str(row.get("video") or ""),
        "t_start": round(_num_or_none(row.get("t_start")) or 0.0, 3),
        "t_end": round(_num_or_none(row.get("t_end")) or 0.0, 3),
        "det_stats": row.get("det_stats") or {},
        "frame_dets": _jsonish(row.get("frame_dets"), []),
        "class_names": row.get("class_names") or {},
        "plate_nos": row.get("plate_nos") or [],
        "plates": row.get("plates") or [],
        "events": row.get("events") or [],
        "face_count": _int_or_zero(row.get("face_count")),
        "description": str(row.get("description") or ""),
        "thumb_path": str(row.get("thumb_path") or ""),
        "vector_model": str(row.get("vector_model") or ""),
        "clip_dim": _num_or_none(row.get("clip_dim")),
        "frame_dim": _num_or_none(row.get("frame_dim")),
        "id": row.get("id"),
    }


def _review_video_row(row: dict) -> dict:
    """视频级元数据行归一（§2.6 字段）。"""
    return {
        "video": str(row.get("video") or ""),
        "video_time": str(row.get("video_time") or ""),
        "video_time_src": str(row.get("video_time_src") or ""),
        "indexed_at": str(row.get("indexed_at") or ""),
        "vector_model": str(row.get("vector_model") or ""),
        "clip_dim": _num_or_none(row.get("clip_dim")),
        "frame_dim": _num_or_none(row.get("frame_dim")),
        "duration": _num_or_none(row.get("duration")),
        "id": row.get("id"),
    }


def _next_review_id(cur, table: str, prefix: str) -> str:
    """按前缀分配下一个业务 ID（RVIDX0001 → RVIDX0002…）。锁内调用。"""
    if table not in ("review_index", "review_video"):
        raise ValueError(f"非法标签表: {table!r}")
    cur.execute(f"SELECT id FROM {table} WHERE id LIKE %s", (prefix + "%",))
    mx = 0
    for (v,) in cur.fetchall():
        m = re.fullmatch(rf"{prefix}(\d+)", str(v))
        if m:
            mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1:04d}"


def _review_next_json_id(rows: list[dict], prefix: str) -> str:
    mx = 0
    for r in rows:
        m = re.fullmatch(rf"{prefix}(\d+)", str(r.get("id") or ""))
        if m:
            mx = max(mx, int(m.group(1)))
    return f"{prefix}{mx + 1:04d}"


# ---------------- review_index 写入 / 删除 ----------------

def upsert_review_index(rows: list[dict]) -> int:
    """批量 upsert 片段标签（幂等键 (video, t_start)）；返回写入行数。"""
    rows = [_review_index_row(r) for r in (rows or []) if str(r.get("video") or "").strip()]
    if not rows:
        return 0
    if not _review_db_available():
        with _REVIEW_JSON_LOCK:
            cur_rows = _review_json_read("review_index")
            by_key = {(str(r.get("video", "")), round(_num_or_none(r.get("t_start")) or 0.0, 3)): r
                      for r in cur_rows}
            next_id = _review_next_json_id(cur_rows, "RVIDX")
            for r in rows:
                key = (r["video"], r["t_start"])
                old = by_key.get(key)
                if old is not None:
                    r["id"] = old.get("id") or next_id
                else:
                    r["id"] = next_id
                    next_id = _increment_prefix(next_id, "RVIDX")
                by_key[key] = r
            out = sorted(by_key.values(), key=lambda r: (str(r.get("video", "")),
                                                         float(r.get("t_start") or 0.0)))
            _review_json_write("review_index", out)
        return len(rows)

    cols = ("video", "t_start", "t_end", "det_stats", "frame_dets", "class_names",
            "plate_nos", "plates", "events", "face_count", "description", "thumb_path",
            "vector_model", "clip_dim", "frame_dim")
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            conn.begin()
            for r in rows:
                cur.execute("SELECT id FROM review_index WHERE video=%s AND t_start=%s",
                            (r["video"], r["t_start"]))
                hit = cur.fetchone()
                if hit:
                    rid = hit[0]
                    sets = ", ".join(f"`{k}`=%s" for k in cols)
                    vals = [_sql_json(r[k]) if k in _REVIEW_JSON_COLS else r[k]
                            for k in cols]
                    cur.execute(f"UPDATE review_index SET {sets} WHERE id=%s", vals + [rid])
                else:
                    rid = _next_review_id(cur, "review_index", "RVIDX")
                    ph = ", ".join(["%s"] * (len(cols) + 1))
                    vals = [rid] + [_sql_json(r[k]) if k in _REVIEW_JSON_COLS
                                    else r[k] for k in cols]
                    cur.execute(f"INSERT INTO review_index (id, {', '.join(cols)})"
                                f" VALUES ({ph})", vals)
            conn.commit()
            return len(rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def _sql_json(v) -> str:
    """JSON 列写入值：结构 → 紧凑 JSON 文本。"""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(_jsonish(v, {}), ensure_ascii=False)


def _increment_prefix(rid: str, prefix: str) -> str:
    m = re.fullmatch(rf"{prefix}(\d+)", str(rid or ""))
    return f"{prefix}{(int(m.group(1)) + 1) if m else 1:04d}"


def delete_review_video(video: str) -> int:
    """删除某视频的片段标签与视频级元数据（幂等）；返回删除的片段行数。"""
    video = str(video or "").strip()
    if not video:
        return 0
    if not _review_db_available():
        with _REVIEW_JSON_LOCK:
            idx = _review_json_read("review_index")
            keep = [r for r in idx if str(r.get("video", "")) != video]
            n = len(idx) - len(keep)
            if n:
                _review_json_write("review_index", keep)
            vids = _review_json_read("review_video")
            keep_v = [r for r in vids if str(r.get("video", "")) != video]
            if len(keep_v) != len(vids):
                _review_json_write("review_video", keep_v)
        return n
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            conn.begin()
            cur.execute("DELETE FROM review_index WHERE video=%s", (video,))
            n = int(cur.rowcount or 0)
            cur.execute("DELETE FROM review_video WHERE video=%s", (video,))
            conn.commit()
            return n
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


# ---------------- review_video 写入 / 读取 ----------------

def upsert_review_video(rows: list[dict]) -> int:
    """批量 upsert 视频级元数据（幂等键 video）；返回写入行数。"""
    rows = [_review_video_row(r) for r in (rows or []) if str(r.get("video") or "").strip()]
    if not rows:
        return 0
    if not _review_db_available():
        with _REVIEW_JSON_LOCK:
            cur_rows = _review_json_read("review_video")
            by_video = {str(r.get("video", "")): r for r in cur_rows}
            next_id = _review_next_json_id(cur_rows, "RVVID")
            for r in rows:
                old = by_video.get(r["video"])
                if old is not None:
                    r["id"] = old.get("id") or next_id
                else:
                    r["id"] = next_id
                    next_id = _increment_prefix(next_id, "RVVID")
                by_video[r["video"]] = r
            out = sorted(by_video.values(), key=lambda r: str(r.get("video", "")))
            _review_json_write("review_video", out)
        return len(rows)

    cols = ("video", "video_time", "video_time_src", "indexed_at", "vector_model",
            "clip_dim", "frame_dim", "duration")
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            conn.begin()
            for r in rows:
                cur.execute("SELECT id FROM review_video WHERE video=%s", (r["video"],))
                hit = cur.fetchone()
                if hit:
                    rid = hit[0]
                    sets = ", ".join(f"`{k}`=%s" for k in cols)
                    cur.execute(f"UPDATE review_video SET {sets} WHERE id=%s",
                                [r[k] for k in cols] + [rid])
                else:
                    rid = _next_review_id(cur, "review_video", "RVVID")
                    ph = ", ".join(["%s"] * (len(cols) + 1))
                    cur.execute(f"INSERT INTO review_video (id, {', '.join(cols)})"
                                f" VALUES ({ph})", [rid] + [r[k] for k in cols])
            conn.commit()
            return len(rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def list_review_videos(video: str | None = None) -> list[dict]:
    """列出视频级元数据（可按视频过滤），字段名 = §2.6。"""
    videos = None
    if isinstance(video, str) and video.strip():
        videos = [video.strip()]
    if not _review_db_available():
        with _REVIEW_JSON_LOCK:
            rows = _review_json_read("review_video")
        if videos:
            rows = [r for r in rows if str(r.get("video", "")) in videos]
        return [_review_video_row(r) for r in rows]

    cols = ("id", "video", "video_time", "video_time_src", "indexed_at",
            "vector_model", "clip_dim", "frame_dim", "duration")
    sql = f"SELECT {', '.join(cols)} FROM review_video"
    args: list = []
    if videos:
        sql += " WHERE video IN (" + ", ".join(["%s"] * len(videos)) + ")"
        args = videos
    sql += " ORDER BY video"
    with _cursor() as cur:
        cur.execute(sql, args)
        return [_review_video_row(dict(zip(cols, r))) for r in cur.fetchall()]


# ---------------- 过滤查询 ----------------

def _as_dt(v):
    """时间窗参数 → 带时区 datetime（None 原样；数值=epoch；无时区按本地时区）。"""
    if v is None or v == "":
        return None
    if isinstance(v, datetime.datetime):
        return v if v.tzinfo else v.astimezone()
    if isinstance(v, (int, float)):
        return datetime.datetime.fromtimestamp(float(v)).astimezone()
    s = str(v).strip()
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.astimezone()


def _video_time_of(video_row: dict | None):
    """review_video 行 → 采集时间 datetime（空/解析失败 → None）。"""
    if not video_row:
        return None
    return _as_dt(video_row.get("video_time"))


def query_review_index(video: str | None = None,
                       time_start=None, time_end=None,
                       event_name: str | None = None) -> list[dict]:
    """按视频 / 墙钟时间窗 / 事件名（含车牌号）过滤片段标签。

    - time_start/time_end：按 review_video.video_time + t_start 判定；**无采集时间的
      视频不参与该过滤**（保留并在行上标 `_time_unknown=True`，由 why 说明）；
    - event_name：精确匹配 events[].name 与 plates[].plate_no（§9 Q2 定案）；
    - 返回行 = §2.5 字段 + `_time_unknown`（仅时间窗生效时存在）。
    """
    t0, t1 = _as_dt(time_start), _as_dt(time_end)
    ev = str(event_name or "").strip()
    videos = [video.strip()] if isinstance(video, str) and video.strip() else None
    video_map = {v["video"]: v for v in list_review_videos(video)}
    if t0 or t1:
        named = set(videos) if videos else set(video_map)
        # 时间窗可用性：有采集时间的视频按区间粗筛；无采集时间的一律保留（不静默丢弃）
        if videos:
            keep = []
            for v in videos:
                vt = _video_time_of(video_map.get(v))
                if vt is None:
                    keep.append(v)
                    continue
                dur = _num_or_none((video_map.get(v) or {}).get("duration")) or 0.0
                if ((t0 is None or vt + datetime.timedelta(seconds=dur) >= t0)
                        and (t1 is None or vt <= t1)):
                    keep.append(v)
            videos = keep
        else:
            keep = []
            for v in named:
                vt = _video_time_of(video_map.get(v))
                if vt is None:
                    keep.append(v)
                else:
                    dur = _num_or_none((video_map.get(v) or {}).get("duration")) or 0.0
                    if ((t0 is None or vt + datetime.timedelta(seconds=dur) >= t0)
                            and (t1 is None or vt <= t1)):
                        keep.append(v)
            videos = sorted(keep)
        if not videos:
            return []

    if not _review_db_available():
        with _REVIEW_JSON_LOCK:
            rows = [_review_index_row(r) for r in _review_json_read("review_index")]
    else:
        cols = ("id",) + _REVIEW_INDEX_FIELDS
        sql = f"SELECT {', '.join(cols)} FROM review_index"
        args: list = []
        if videos:
            sql += " WHERE video IN (" + ", ".join(["%s"] * len(videos)) + ")"
            args = videos
        sql += " ORDER BY video, t_start"
        with _cursor() as cur:
            cur.execute(sql, args)
            rows = []
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                for k in _REVIEW_JSON_COLS:
                    d[k] = _jsonish(d.get(k), [] if k in ("plate_nos", "plates",
                                                          "events", "frame_dets") else {})
                rows.append(_review_index_row(d))

    out = []
    for r in rows:
        if videos and r["video"] not in videos:
            continue
        if ev:
            names = {str(e.get("name", "")) for e in (r.get("events") or []) if isinstance(e, dict)}
            plates = {str(p.get("plate_no", "")) for p in (r.get("plates") or []) if isinstance(p, dict)}
            if ev not in names and ev not in plates:
                continue
        if t0 or t1:
            vt = _video_time_of(video_map.get(r["video"]))
            if vt is None:
                r["_time_unknown"] = True
            else:
                abs_ts = vt + datetime.timedelta(seconds=float(r["t_start"]))
                if (t0 is not None and abs_ts < t0) or (t1 is not None and abs_ts > t1):
                    continue
        out.append(r)
    return out
