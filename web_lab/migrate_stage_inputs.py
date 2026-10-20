# -*- coding: utf-8 -*-
"""一次性幂等迁移：pipelines.meta_json.stage_inputs → pipeline_edges（图编排 P2-1）。

背景见 doc/design/graph_chain.md §3.3：连线从 meta JSON 内联升级为 pipeline_edges 一等表。
本脚本把历史 meta.stage_inputs（旧式 {module_id: {端口: 产物名}} 或新式 {"#<position>": ...}）
规范为位置键并落 pipeline_edges，同时把 meta.stage_inputs 回写为同一规范形态
（与表一致，读取优先级 edges > meta 时输出逐字节一致）。

原则：
  - **幂等**：再次运行，若表与 meta 均已一致则跳过（不写、不动 updated_at）。
  - **不猜**：旧数据含重复模块且 stage_inputs 仍用 module_id 键（本就非法/歧义）时
    **报错停下要人工确认**，不静默选取。

用法（项目根）:
    conda run -n ai python web_lab/migrate_stage_inputs.py            # 执行迁移
    conda run -n ai python web_lab/migrate_stage_inputs.py --dry-run  # 只报告将改动的链路
    conda run -n ai python web_lab/migrate_stage_inputs.py --only PIPE0001
    conda run -n ai python web_lab/migrate_stage_inputs.py --help

退出码：0=成功（含无改动）；1=有链路需人工确认或连接失败。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import db  # noqa: E402


def _steps_module_ids(cur, pid: str) -> list:
    cur.execute("SELECT p.module_id FROM pipeline_steps ps JOIN presets p ON p.id = ps.preset_id"
                " WHERE ps.pipeline_id=%s ORDER BY ps.position", (pid,))
    return [r[0] for r in cur.fetchall()]


def _existing_edge_keys(cur, pid: str) -> list:
    cur.execute("SELECT dst_position, dst_port, src_position, artifact FROM pipeline_edges"
                " WHERE pipeline_id=%s", (pid,))
    return sorted((r[0], r[1], r[2], r[3]) for r in cur.fetchall())


def migrate(dry_run: bool = False, only: str | None = None) -> int:
    with db._cursor() as cur:
        cur.execute("SELECT id, name, meta_json FROM pipelines ORDER BY id")
        rows = cur.fetchall()

    scanned = changed = skipped = 0
    errors: list = []
    for pid, name, meta_json in rows:
        if only and pid != only:
            continue
        scanned += 1
        meta = db._loads(meta_json)
        raw_si = meta.get("stage_inputs") if isinstance(meta, dict) else None
        with db._cursor() as cur:
            module_ids = _steps_module_ids(cur, pid)
        positions_by_module = db._positions_by_module(module_ids)
        try:
            canonical = db.canonical_stage_inputs(raw_si, positions_by_module)
        except ValueError as e:  # 多实例旧式键 / 歧义 → 停人工确认
            errors.append((pid, name, f"stage_inputs 规范化失败: {e}"))
            continue
        with db._cursor() as cur:
            fan_in = db._module_port_fan_in(cur)
        try:
            edges = db._edges_from_inputs(canonical, module_ids, fan_in)
        except ValueError as e:
            errors.append((pid, name, f"生成连线失败: {e}"))
            continue

        want_keys = sorted((d, port, src, art) for (d, _mid, port, src, art, _seq) in edges)
        with db._cursor() as cur:
            have_keys = _existing_edge_keys(cur, pid)
        meta_new = None
        if canonical != (raw_si if isinstance(raw_si, dict) else {}):
            meta_new = {**meta, "stage_inputs": canonical}

        if want_keys == have_keys and meta_new is None:
            skipped += 1
            continue
        changed += 1
        if dry_run:
            print(f"  [DRY] {pid} {name}: edges {len(have_keys)} → {len(want_keys)} 行"
                  + ("，meta 也将规范化" if meta_new is not None else ""))
            continue

        with db._LOCK:
            conn = db._get_conn()
            cur = conn.cursor()
            try:
                conn.ping(reconnect=True)
                conn.begin()
                cur.execute("DELETE FROM pipeline_edges WHERE pipeline_id=%s", (pid,))
                for d, mid, port, src, art, seq in edges:
                    cur.execute(
                        "INSERT INTO pipeline_edges (pipeline_id, dst_position, dst_module_id,"
                        " dst_port, src_position, artifact, edge_seq)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s)", (pid, d, mid, port, src, art, seq))
                if meta_new is not None:
                    cur.execute("UPDATE pipelines SET meta_json=%s WHERE id=%s",
                                (json.dumps(meta_new, ensure_ascii=False), pid))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
        print(f"  [OK] {pid} {name}: edges → {len(edges)} 行"
              + ("，meta 已规范化" if meta_new is not None else ""))

    print(f"\n扫描 {scanned} 条链路；{'将改动' if dry_run else '已迁移'} {changed} 条，"
          f"跳过 {skipped} 条（已一致）")
    if errors:
        print("\n⚠ 以下链路需人工确认（未改动，不静默猜）：")
        for pid, name, msg in errors:
            print(f"  [ERR] {pid} {name}: {msg}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="meta.stage_inputs → pipeline_edges 幂等迁移")
    ap.add_argument("--dry-run", action="store_true", help="只报告将改动的链路，不写库")
    ap.add_argument("--only", default=None, help="只迁移指定链路 id（如 PIPE0001）")
    args = ap.parse_args()
    try:
        return migrate(dry_run=args.dry_run, only=args.only)
    except Exception as e:  # 连接失败等
        print(f"[ERR] 迁移失败: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
