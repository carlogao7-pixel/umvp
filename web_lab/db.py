# -*- coding: utf-8 -*-
"""UMVP 配置暂存 —— MySQL 访问层（web_lab 扩展）。

职责：把各模块/链路的命名配置（preset）存进 MySQL，供前端"命名暂存/取用"。

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

import json
import os

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


def _cursor():
    global _conn
    try:
        conn = _get_conn()
        conn.ping(reconnect=True)  # 缓存连接已断开时自动重连（否则 execute 阶段才报错）
        return conn.cursor()
    except Exception:
        # 重连失败（MySQL 未启动等）：重置连接，换全新连接再试一次；仍失败则抛出由调用方处理
        _conn = None
        return _get_conn().cursor()


_DDLS = [
    """CREATE TABLE IF NOT EXISTS presets (
  id INT AUTO_INCREMENT PRIMARY KEY,
  kind ENUM('module','pipeline') NOT NULL DEFAULT 'module',
  name VARCHAR(64) NOT NULL,
  module_id VARCHAR(32) NOT NULL DEFAULT '',
  params_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_kind_name (kind, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    # 链路拼装：链路自身字段（streams/budget/meta）+ 有序步骤引用模块 preset
    # （设计见 doc/链路拼装存储设计.md）
    """CREATE TABLE IF NOT EXISTS pipelines (
  id INT AUTO_INCREMENT PRIMARY KEY,
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
  pipeline_id INT NOT NULL,
  position INT NOT NULL,
  preset_id INT NOT NULL,
  UNIQUE KEY uk_pipeline_position (pipeline_id, position),
  KEY idx_preset (preset_id),
  CONSTRAINT fk_step_pipeline FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE,
  CONSTRAINT fk_step_preset FOREIGN KEY (preset_id) REFERENCES presets(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]


def init_db() -> None:
    """建表（幂等）。服务器启动时调用一次。"""
    with _cursor() as cur:
        for ddl in _DDLS:
            cur.execute(ddl)


def list_presets(kind: str = "module") -> list[dict]:
    """按 updated_at 倒序列出指定 kind 的配置，params 解析为 dict。"""
    with _cursor() as cur:
        cur.execute(
            "SELECT id, name, module_id, params_json FROM presets"
            " WHERE kind=%s ORDER BY updated_at DESC, id DESC", (kind,))
        rows = cur.fetchall()
    out = []
    for rid, name, module_id, params_json in rows:
        try:
            params = json.loads(params_json)
        except (TypeError, ValueError):
            params = {}
        out.append({"id": rid, "name": name, "module_id": module_id,
                    "params": params})
    return out


def save_preset(kind: str, name: str, module_id: str, params: dict) -> tuple[int, bool]:
    """保存命名配置；同名覆盖旧配置。返回 (id, 是否新建)。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("配置名不能为空")
    if len(name) > 64:
        raise ValueError("配置名过长（最多 64 字符）")
    params_json = json.dumps(params, ensure_ascii=False)
    with _cursor() as cur:
        cur.execute("SELECT id FROM presets WHERE kind=%s AND name=%s", (kind, name))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE presets SET module_id=%s, params_json=%s WHERE id=%s",
                        (module_id, params_json, row[0]))
            return row[0], False
        cur.execute("INSERT INTO presets (kind, name, module_id, params_json)"
                    " VALUES (%s, %s, %s, %s)", (kind, name, module_id, params_json))
        return cur.lastrowid, True


def load_preset(preset_id: int) -> dict:
    """按 id 取回配置：{id, kind, name, module_id, params}。不存在抛 ValueError。"""
    with _cursor() as cur:
        cur.execute("SELECT id, kind, name, module_id, params_json FROM presets WHERE id=%s",
                    (preset_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"配置不存在: id={preset_id}")
    rid, kind, name, module_id, params_json = row
    try:
        params = json.loads(params_json)
    except (TypeError, ValueError):
        params = {}
    return {"id": rid, "kind": kind, "name": name, "module_id": module_id,
            "params": params}


def delete_preset(preset_id: int) -> None:
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
        steps: dict[int, list[dict]] = {}
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


def save_pipeline(name: str, preset_ids: list[int], streams: int = 1,
                  budget: dict | None = None, meta: dict | None = None) -> tuple[int, bool]:
    """保存链路（步骤引用模块 preset）；同名覆盖。返回 (id, 是否新建)。

    preset_ids 的顺序即链路步骤顺序（position 0..n-1）；事务内重建 steps，
    保证 pipelines 与 pipeline_steps 原子一致。步骤只能引用 kind='module' 的配置。
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
    conn = _get_conn()
    cur = conn.cursor()
    try:
        conn.ping(reconnect=True)
        # 校验步骤引用的 preset 存在且为 module 类型（放事务外，避免无用锁占位）
        placeholders = ",".join(["%s"] * len(preset_ids))
        cur.execute(f"SELECT id, kind FROM presets WHERE id IN ({placeholders})",
                    tuple(preset_ids))
        found = {rid: kind for rid, kind in cur.fetchall()}
        for rid in preset_ids:
            if rid not in found:
                raise ValueError(f"步骤引用的配置不存在: id={rid}")
            if found[rid] != "module":
                raise ValueError(f"步骤只能引用模块配置（kind='module'），id={rid} 是 {found[rid]}")
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
            cur.execute("INSERT INTO pipelines (name, streams, budget_json, meta_json)"
                        " VALUES (%s, %s, %s, %s)", (name, streams, budget_json, meta_json))
            pid = cur.lastrowid
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


def load_pipeline(pipeline_id: int) -> dict:
    """按 id 取回完整链路：{id, name, streams, budget, meta, stages}。

    stages 为有序步骤 [{position, preset_id, module_id, params}]，可直接生成
    P2 pipeline spec 的 stages=[{module, params}]。
    """
    with _cursor() as cur:
        cur.execute("SELECT id, name, streams, budget_json, meta_json"
                    " FROM pipelines WHERE id=%s", (pipeline_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"链路不存在: id={pipeline_id}")
        pid, name, streams, budget_json, meta_json = row
        cur.execute("SELECT ps.position, ps.preset_id, p.module_id, p.params_json"
                    " FROM pipeline_steps ps JOIN presets p ON p.id = ps.preset_id"
                    " WHERE ps.pipeline_id=%s ORDER BY ps.position", (pid,))
        stages = [{"position": position, "preset_id": preset_id, "module_id": module_id,
                   "params": _loads(params_json)}
                  for position, preset_id, module_id, params_json in cur.fetchall()]
    return {"id": pid, "name": name, "streams": streams,
            "budget": _loads(budget_json), "meta": _loads(meta_json),
            "stages": stages}


def delete_pipeline(pipeline_id: int) -> None:
    """删除链路（steps 级联删除）；不存在也不报错（幂等）。"""
    with _cursor() as cur:
        cur.execute("DELETE FROM pipeline_steps WHERE pipeline_id=%s", (pipeline_id,))
        cur.execute("DELETE FROM pipelines WHERE id=%s", (pipeline_id,))
