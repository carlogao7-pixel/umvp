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


_DDL = """
CREATE TABLE IF NOT EXISTS presets (
  id INT AUTO_INCREMENT PRIMARY KEY,
  kind ENUM('module','pipeline') NOT NULL DEFAULT 'module',
  name VARCHAR(64) NOT NULL,
  module_id VARCHAR(32) NOT NULL DEFAULT '',
  params_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_kind_name (kind, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def init_db() -> None:
    """建表（幂等）。服务器启动时调用一次。"""
    with _cursor() as cur:
        cur.execute(_DDL)


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
    """删除配置；不存在也不报错（幂等）。"""
    with _cursor() as cur:
        cur.execute("DELETE FROM presets WHERE id=%s", (preset_id,))
