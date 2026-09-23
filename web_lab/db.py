# -*- coding: utf-8 -*-
"""UMVP 配置暂存 —— MySQL 访问层（web_lab 扩展）。

职责：把各模块/链路的命名配置（preset）存进 MySQL，供前端"命名暂存/取用"。

存储模型（2026-08-10 起，详见 doc/存储设计.md）:
  - presets 保留为"注册表"：id / kind / name / module_id / 时间戳；
  - 每个模块（module_id，如 yolo/vlm）一张参数表 preset_params_<module_id>，
    参数从 JSON 展开为类型化列（由 server.py 启动时 configure_modules 注册 schema）；
  - kind='pipeline' 仍是整份 JSON 快照（params_json），结构不固定不适合列式。

ID 规则（2026-09-20 起，见 doc/数据库结构.md）:
  - presets.id / pipelines.id 不再用自增整数，改"模块前缀 + 4 位序号"的业务编号，
    如 YOLO0001 / FM0001 / PIPE0001；新建时按前缀自动递增；
  - init_db 检测到旧整数 schema 时自动 DROP 全部相关表并重建（播种数据可再生）。

设计约定（与 web_lab 一致）:
  - 惰性连接：首次调用才建立连接，服务器启动不依赖 MySQL；
  - pymysql 缺失或连接失败时函数抛异常，由 server.py 统一转 {ok:false, error}，
    服务器本身照常运行（DB 不可用 ≠ web_lab 不可用）；
  - 命名即身份：同 kind 下 name 重复保存 = 覆盖旧配置（重新保存即更新）；
  - 连接参数可用 MYSQL_* 环境变量覆盖（换机器/换库零改码）。

运行实例（本项目自含，免 sudo，见 doc）:
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
# db 据此建参数表 / 类型强转（设计见 doc/存储设计.md）。
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
    "target_crop": "CROP",
    "annotate": "ANNO",
    "vlm": "VLM",
    "alarm": "AL",
    "face_detect": "FDET",
    "face_embed": "FEMB",
    "face_store": "FSTO",
    "extract_faces": "EXTF",
    "plate_recog": "PLATE",
}
_PIPELINE_PREFIX = "PIPE"  # 链路（pipelines 表 / kind='pipeline' 的 preset）
_PRESET_ID_RE = re.compile(r"^[A-Z]{2,4}\d{4}$")  # 合法业务 ID 形如 YOLO0001

# 已下线/不落库模块（不注册参数 schema、不建参数表）：init_db 时幂等清理其参数表
# 与 presets 行。
# 2026-08-18: frame_scheduler 抽帧调度并入 frame_manager（帧管理），模块合并。
# 2026-09-20: compose/utest 为历史遗留空表，一并清理。
# 2026-09-20(二): chain/face_monitor/frame_yolo 为 web 端隐藏运行器（运行入口，
# 非产品模块），参数 preset 统一不落库，空参数表清理。
# 2026-09-23: extract_faces（原分辨率提取）拆分为「目标裁剪 target_crop + 人脸检测
# face_detect」，模块下线，参数表/preset 清理。
_PRUNED_MODULES = {"frame_scheduler", "compose", "utest",
                   "chain", "face_monitor", "frame_yolo", "extract_faces"}


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
    # （设计见 doc/存储设计.md）
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
]


def _legacy_id_schema(cur) -> bool:
    """检测 presets.id 是否还是旧的自增整数 schema（需 DROP 重建）。"""
    cur.execute("SELECT COLUMN_TYPE FROM information_schema.COLUMNS"
                " WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='presets' AND COLUMN_NAME='id'")
    row = cur.fetchone()
    return bool(row and str(row[0]).lower().startswith("int"))


def _drop_legacy_tables(cur) -> None:
    """DROP 全部配置相关表（presets/pipelines/pipeline_steps/preset_params_*）。

    仅在检测到旧整数 ID schema 时调用：表内均为播种/测试数据（服务器启动会重新
    播种唯一保留链路），重建后即迁移到前缀编号 schema。幂等。
    """
    cur.execute("SHOW TABLES")
    for (t,) in cur.fetchall():
        if t in ("presets", "pipelines", "pipeline_steps") or str(t).startswith("preset_params_"):
            cur.execute(f"DROP TABLE `{t}`")


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
    """
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.TABLES"
                    " WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='presets'")
        if cur.fetchone() and _legacy_id_schema(cur):
            _drop_legacy_tables(cur)
            print("[db] 检测到旧整数 ID schema，已 DROP 重建为前缀编号 schema")
        for ddl in _DDLS:
            cur.execute(ddl)
        for module_id, schema in _SCHEMAS.items():
            _ensure_module_table(cur, module_id, schema)
        _prune_stale_columns(cur)
        _migrate_legacy_module_params(cur)
        _prune_removed_modules(cur)


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

    preset_ids 的顺序即链路步骤顺序（position 0..n-1）；事务内重建 steps，
    保证 pipelines 与 pipeline_steps 原子一致。步骤只能引用 kind='module' 的配置。
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
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with _LOCK:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            conn.ping(reconnect=True)
            # 校验步骤引用的 preset 存在、为 module 类型、且模块不重复（一条链路每个模块最多一个阶段）
            placeholders = ",".join(["%s"] * len(preset_ids))
            cur.execute(f"SELECT id, kind, module_id FROM presets WHERE id IN ({placeholders})",
                        tuple(preset_ids))
            found = {rid: (kind, mid) for rid, kind, mid in cur.fetchall()}
            for rid in preset_ids:
                if rid not in found:
                    raise ValueError(f"步骤引用的配置不存在: id={rid}")
                if found[rid][0] != "module":
                    raise ValueError(f"步骤只能引用模块配置（kind='module'），id={rid} 是 {found[rid][0]}")
            seen: dict = {}
            for rid in preset_ids:
                mid = found[rid][1]
                if mid in seen:
                    raise ValueError(
                        f"链路包含重复模块阶段: {mid}（{seen[mid]} 与 {rid}）；"
                        f"每个模块在一条链路中最多出现一次")
                seen[mid] = rid
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
            conn.commit()
            return pid, created
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def load_pipeline(pipeline_id: str) -> dict:
    """按 id 取回完整链路：{id, name, streams, budget, meta, stages}。

    stages 为有序步骤 [{position, preset_id, module_id, params}]，可直接生成
    P2 pipeline spec 的 stages=[{module, params}]。params 从模块参数表取回。
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
    stages = [{"position": position, "preset_id": preset_id, "module_id": module_id,
               "params": params_map.get(preset_id, {})}
              for position, preset_id, module_id in steps]
    return {"id": pid, "name": name, "streams": streams,
            "budget": _loads(budget_json), "meta": _loads(meta_json),
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
