# -*- coding: utf-8 -*-
"""图编排 web 侧自检（P2-0 装配器/校验器位置化 + P2-1 DB）——WO-0008 子单。

用法（项目根）:
    conda run -n ai python web_lab/test_graph_chain.py            # 全跑（p20 + p21）
    conda run -n ai python web_lab/test_graph_chain.py --section p20
    conda run -n ai python web_lab/test_graph_chain.py --section p21

p20：装配器 / 校验器按位置装配多实例、stage_inputs 位置键 + 产物全名解析、歧义/上游/列表报错。
p21：连线规范化 / 产物名解析 / fan_in 写入期校验（纯函数，无需 DB）；
     MySQL 可用时另跑 save→load 往返、fan_in 拒多入边、迁移幂等（不可用则跳过并提示）。
退出码：0=全过；1=有失败。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402
import db  # noqa: E402
import validate_chain as vc  # noqa: E402
import migrate_stage_inputs as mig  # noqa: E402
from pipe.composer import (  # noqa: E402
    PortResolutionError, resolve_artifact as composer_resolve_artifact)

_FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" | {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def _spec(stages):
    return {"stages": stages}


# ---------------------------------------------------------------- p20
def section_p20() -> None:
    print("== P2-0 装配器 / 校验器位置化 ==")

    # resolve_artifact 单一实现（re-export 自 composer）
    check("resolve_artifact 复用 composer 单一实现",
          server.resolve_artifact is composer_resolve_artifact)
    check("PortResolutionError 复用 composer",
          server.PortResolutionError is PortResolutionError)

    pbm = {"yolo": [1], "det_filter": [2, 4]}
    check("resolve: 短名唯一实例 → 全名",
          server.resolve_artifact("yolo.dets", pbm) == "yolo#1.dets")
    check("resolve: 已是全名 → 原样返回",
          server.resolve_artifact("det_filter#4.dets", pbm) == "det_filter#4.dets")
    try:
        server.resolve_artifact("det_filter.dets", pbm)
        check("resolve: 多实例短名歧义 → 抛错", False, "未抛错")
    except PortResolutionError as e:
        check("resolve: 多实例短名歧义 → 抛错", "全名" in str(e), str(e))

    check("_positions_by_module 位置列表",
          server._positions_by_module(["fm", "yolo", "df", "yolo"]) ==
          {"fm": [0], "yolo": [1, 3], "df": [2]})

    # 按位置装配多实例（det_filter 两实例）
    spec = _spec([
        {"module_id": "frame_manager", "params": {"frame_skip": 3}},
        {"module_id": "yolo", "params": {}},
        {"module_id": "det_filter", "params": {}},
        {"module_id": "target_crop", "params": {}},
        {"module_id": "det_filter", "params": {}},
        {"module_id": "annotate", "params": {},
         "inputs": {"dets": "det_filter#4.dets", "image": "frame_manager.frame"}},
        {"module_id": "det_filter", "params": {}},
    ])
    pipe = server._chain_from_spec(spec)
    check("装配: 同模块多实例各自构造（det_filters 位置）",
          [p for p, _ in pipe.det_filters] == [2, 4, 6], str([p for p, _ in pipe.det_filters]))
    check("装配: stages 容器按位置携带",
          sorted(pipe.stages.keys()) == [0, 1, 2, 3, 4, 5, 6], str(sorted(pipe.stages.keys())))
    check("装配: stage_positions（单插槽位置，int）", pipe.stage_positions.get("yolo") == 1)
    check("装配: stage_inputs 位置键 + 产物全名",
          pipe.stage_inputs == {"#5": {"dets": "det_filter#4.dets",
                                       "image": "frame_manager#0.frame"}},
          str(pipe.stage_inputs))

    # 校验器：多实例放行
    check("校验: 同模块多实例放行（无重复模块报错）",
          not any("重复模块" in e for e in vc.validate(spec)))
    bad_fm = _spec([{"module_id": "frame_manager", "params": {}},
                    {"module_id": "frame_manager", "params": {}},
                    {"module_id": "det_filter", "params": {}}])
    check("校验: frame_manager 多段 → 报错",
          any("frame_manager" in e and "多次" in e for e in vc.validate(bad_fm)))

    amb = _spec([
        {"module_id": "frame_manager", "params": {}},
        {"module_id": "yolo", "params": {}},
        {"module_id": "det_filter", "params": {}},
        {"module_id": "det_filter", "params": {}},
        {"module_id": "annotate", "params": {}, "inputs": {"dets": "det_filter.dets"}},
        {"module_id": "det_filter", "params": {}},
    ])
    errs = vc.validate(amb)
    check("校验: 短名多实例歧义 → 报错要求写全名",
          any("无法解析" in e and "全名" in e for e in errs), str(errs))

    down = _spec([
        {"module_id": "frame_manager", "params": {}},
        {"module_id": "yolo", "params": {}},
        {"module_id": "annotate", "params": {}, "inputs": {"dets": "det_filter#3.dets"}},
        {"module_id": "det_filter", "params": {}},
    ])
    check("校验: 引用下游 → 报错（不是上游）",
          any("不是上游" in e for e in vc.validate(down)))

    lst = _spec([
        {"module_id": "frame_manager", "params": {}},
        {"module_id": "yolo", "params": {}},
        {"module_id": "annotate", "params": {},
         "inputs": {"image": ["frame_manager.frame", "yolo.dets"]}},
        {"module_id": "det_filter", "params": {}},
    ])
    check("校验: 列表值（fan_in=0）→ 报错含 list",
          any("值须为字符串" in e and "list" in e for e in vc.validate(lst)))

    # 端口目录
    cat = server.module_ports_catalog()
    check("目录: module_ports_catalog 非空且全 fan_in=0",
          bool(cat) and all(r[4] == 0 for r in cat), f"{len(cat)} 行")
    check("目录: resolved_by ∈ {pipe,driver}",
          all(r[5] in ("pipe", "driver") for r in cat))
    check("目录: 含 vlm.dets(in,driver) 与 yolo.image(in,pipe)",
          ("vlm", "in", "dets", "Records", 0, "driver") in cat
          and ("yolo", "in", "image", "Image", 0, "pipe") in cat)

    # 驱动层 _apply_vlm_inputs 按位置读
    class _P:
        stage_inputs = {"#3": {"dets": "det_filter#2.dets"}}
        module_positions = {"vlm": [3], "det_filter": [2]}
        artifacts = {"det_filter#2.dets": [{"cls_id": 0, "bbox": [0, 0, 10, 10]}]}

        def _submits(self, dets):
            return [("n", len(dets))]
    out = server._apply_vlm_inputs(_P(), [])
    check("驱动: _apply_vlm_inputs 按位置取 dets",
          out == [("n", 1)], str(out))


# ---------------------------------------------------------------- p21
def section_p21() -> None:
    print("== P2-1 连线规范化 / 产物解析 / fan_in（纯函数） ==")

    pbm = {"frame_manager": [0], "yolo": [1], "det_filter": [2, 3], "annotate": [4]}
    # 旧 module_id 唯一实例键 → 位置键
    can = db.canonical_stage_inputs({"annotate": {"dets": "det_filter#2.dets"}}, pbm)
    check("规范: 旧 module_id 键（唯一实例）→ 位置键", can == {"#4": {"dets": "det_filter#2.dets"}},
          str(can))
    # 位置键原样
    can2 = db.canonical_stage_inputs({"#4": {"dets": "yolo.dets"}}, pbm)
    check("规范: 位置键保留 + 产物名原样（短名不解析）", can2 == {"#4": {"dets": "yolo.dets"}}, str(can2))
    # 多实例 module_id 键 → 抛错（不猜）
    try:
        db.canonical_stage_inputs({"det_filter": {"dets": "yolo.dets"}}, pbm)
        check("规范: 多实例 module_id 键 → 抛错", False, "未抛错")
    except ValueError as e:
        check("规范: 多实例 module_id 键 → 抛错", "位置串" in str(e), str(e))
    # 列表值保留
    can3 = db.canonical_stage_inputs({"#4": {"dets": ["a.b", "c.d"]}}, pbm)
    check("规范: 列表值保留", can3 == {"#4": {"dets": ["a.b", "c.d"]}}, str(can3))

    # 产物名解析
    check("解析: 短名唯一 → (mod, pos)",
          db._parse_artifact_pos("yolo.dets", pbm) == ("yolo", 1))
    check("解析: 全名 → (mod, pos)",
          db._parse_artifact_pos("det_filter#3.dets", pbm) == ("det_filter", 3))
    try:
        db._parse_artifact_pos("det_filter.dets", pbm)
        check("解析: 短名多实例 → 抛错", False, "未抛错")
    except ValueError as e:
        check("解析: 短名多实例 → 抛错", "歧义" in str(e), str(e))
    try:
        db._parse_artifact_pos("nope.dets", pbm)
        check("解析: 未知模块 → 抛错", False, "未抛错")
    except ValueError as e:
        check("解析: 未知模块 → 抛错", "未知模块" in str(e), str(e))

    # fan_in 写入期校验
    module_ids = ["frame_manager", "yolo", "det_filter", "det_filter", "annotate"]
    ok = db._edges_from_inputs({"#4": {"dets": "det_filter#2.dets"}}, module_ids, {})
    check("fan_in: 单上游字符串 → 1 行边",
          ok == [(4, "annotate", "dets", 2, "det_filter#2.dets", 0)], str(ok))
    try:
        db._edges_from_inputs({"#4": {"dets": ["det_filter#2.dets", "det_filter#3.dets"]}},
                              module_ids, {})
        check("fan_in: 单上游端口接两条 → 抛错", False, "未抛错")
    except ValueError as e:
        check("fan_in: 单上游端口接两条 → 抛错", "fan_in=0" in str(e), str(e))
    multi = db._edges_from_inputs({"#4": {"dets": ["det_filter#2.dets", "det_filter#3.dets"]}},
                                  module_ids, {("annotate", "dets"): 1})
    check("fan_in: fan_in=1 端口允许多入边（edge_seq 保序）",
          [e[5] for e in multi] == [0, 1] and len(multi) == 2, str(multi))

    # 回填往返
    stages = [{"module_id": "annotate"}]
    db._attach_stage_inputs(stages, {"#0": {"dets": "yolo.dets"}})
    check("回填: _attach_stage_inputs 位置键 → stage.inputs",
          stages[0].get("inputs") == {"dets": "yolo.dets"}, str(stages))
    rows = [(4, "dets", "det_filter#2.dets", 0), (4, "image", "frame_manager#0.frame", 0)]
    check("回填: _edges_to_stage_inputs 行 → 位置键 dict",
          db._edges_to_stage_inputs(rows) == {
              "#4": {"dets": "det_filter#2.dets", "image": "frame_manager#0.frame"}},
          str(db._edges_to_stage_inputs(rows)))

    _p21_db()


def _p21_db() -> None:
    try:
        with db._cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as e:
        print(f"  [SKIP] MySQL 不可用，跳过 DB 集成自检: {type(e).__name__}: {e}")
        return
    print("  -- MySQL 可用：save→load 往返 / fan_in 拒多入边 / 迁移幂等 --")
    db.configure_modules(
        {mid: {k: spec.get("type", "str") for k, spec in m["params"].items()
               if not spec.get("test_only")}
         for mid, m in server.MODULES.items() if not m.get("hidden")})
    db.init_db()
    db.upsert_module_ports(server.module_ports_catalog(), warn=lambda m: None)

    made: list = []
    pid = None
    try:
        ids = []
        for mid, nm in (("frame_manager", "tgc_fm"), ("yolo", "tgc_yolo"),
                        ("det_filter", "tgc_df1"), ("det_filter", "tgc_df2"),
                        ("annotate", "tgc_anno")):
            i, _ = db.save_preset("module", nm, mid, {})
            ids.append(i)
            made.append(i)
        meta = {"stage_inputs": {"#4": {"dets": "det_filter#3.dets",
                                        "image": "frame_manager#0.frame"}}}
        pid, _ = db.save_pipeline("tgc_pipe", ids, meta=meta)
        loaded = db.load_pipeline(pid)
        st4 = loaded["stages"][4]
        want_inputs = {"dets": "det_filter#3.dets", "image": "frame_manager#0.frame"}
        check("DB: save→load 位置键回填 stage.inputs",
              st4.get("inputs") == want_inputs, str(st4.get("inputs")))

        # fan_in 写入期拒多入边
        try:
            db.save_pipeline("tgc_bad", ids, meta={"stage_inputs": {
                "#4": {"dets": ["det_filter#2.dets", "det_filter#3.dets"]}}})
            check("DB: save_pipeline 拒单端口多入边", False, "未抛错")
        except ValueError:
            check("DB: save_pipeline 拒单端口多入边", True)
        db.delete_pipeline("tgc_bad")

        # 迁移：手工造「只有 meta、无 edges」的旧数据（旧 module_id 键）——先验证
        # **meta 回退读取与 edges 读取逐字节一致**（§7.5），再 migrate 重建 + 规范化 meta
        with db._LOCK:
            conn = db._get_conn()
            cur = conn.cursor()
            try:
                conn.ping(reconnect=True)
                conn.begin()
                cur.execute("DELETE FROM pipeline_edges WHERE pipeline_id=%s", (pid,))
                legacy_meta = json.dumps({"stage_inputs": {
                    "annotate": {"dets": "det_filter#3.dets",
                                 "image": "frame_manager#0.frame"}}}, ensure_ascii=False)
                cur.execute("UPDATE pipelines SET meta_json=%s WHERE id=%s", (legacy_meta, pid))
                conn.commit()
            finally:
                cur.close()
        loaded_meta = db.load_pipeline(pid)  # 无 edges → 回退 meta（规范为位置键再回填）
        got_meta = loaded_meta["stages"][4].get("inputs")
        check("DB: meta 回退读取与 edges 读取逐字节一致",
              json.dumps(got_meta, sort_keys=True) == json.dumps(want_inputs, sort_keys=True),
              str(got_meta))
        rc = mig.migrate(dry_run=False, only=pid)
        check("DB: 迁移执行成功", rc == 0, f"rc={rc}")
        with db._cursor() as cur:
            cur.execute("SELECT dst_position, dst_port, src_position, artifact"
                        " FROM pipeline_edges WHERE pipeline_id=%s"
                        " ORDER BY dst_position, dst_port", (pid,))
            edges = [tuple(r) for r in cur.fetchall()]
            cur.execute("SELECT meta_json FROM pipelines WHERE id=%s", (pid,))
            meta_now = db._loads(cur.fetchone()[0])
        check("DB: 迁移重建 edges（旧 module_id 键 → 位置）",
              edges == [(4, "dets", 3, "det_filter#3.dets"),
                        (4, "image", 0, "frame_manager#0.frame")], str(edges))
        check("DB: 迁移规范化 meta.stage_inputs 为位置键",
              meta_now.get("stage_inputs") ==
              {"#4": {"dets": "det_filter#3.dets", "image": "frame_manager#0.frame"}},
              str(meta_now.get("stage_inputs")))
        loaded_edges = db.load_pipeline(pid)  # 迁移后走 edges 路径
        check("DB: 迁移后 edges 读取仍与迁移前逐字节一致",
              json.dumps(loaded_edges["stages"][4].get("inputs"), sort_keys=True)
              == json.dumps(want_inputs, sort_keys=True),
              str(loaded_edges["stages"][4].get("inputs")))
        # 幂等：再次迁移不改动
        before = json.dumps(meta_now, ensure_ascii=False, sort_keys=True)
        mig.migrate(dry_run=False, only=pid)
        with db._cursor() as cur:
            cur.execute("SELECT meta_json FROM pipelines WHERE id=%s", (pid,))
            after = json.dumps(db._loads(cur.fetchone()[0]), ensure_ascii=False, sort_keys=True)
        check("DB: 迁移幂等（二次运行无改动）", before == after)

        # §7.5 插删阶段后位置重排：删 df2（#3）→ annotate #4→#3，连线键随数组下标重建
        ids2 = ids[:3] + ids[4:]  # fm,yolo,df1,anno
        db.save_pipeline("tgc_pipe", ids2,
                         meta={"stage_inputs": {"#3": {"dets": "det_filter#2.dets"}}})
        loaded2 = db.load_pipeline(pid)
        check("DB: 删阶段后 steps 重排（4 段）",
              [s["module_id"] for s in loaded2["stages"]] ==
              ["frame_manager", "yolo", "det_filter", "annotate"],
              str([s["module_id"] for s in loaded2["stages"]]))
        check("DB: 删阶段后连线随位置重排回填",
              loaded2["stages"][3].get("inputs") == {"dets": "det_filter#2.dets"},
              str(loaded2["stages"][3].get("inputs")))
        with db._cursor() as cur:
            cur.execute("SELECT dst_position, dst_port, src_position, artifact"
                        " FROM pipeline_edges WHERE pipeline_id=%s"
                        " ORDER BY dst_position, dst_port", (pid,))
            remap = [tuple(r) for r in cur.fetchall()]
        check("DB: 删阶段后 pipeline_edges 位置正确（dst=3 src=2）",
              remap == [(3, "dets", 2, "det_filter#2.dets")], str(remap))
        # 插回 df2 → annotate 回 #4，连线位置还原
        db.save_pipeline("tgc_pipe", ids, meta={"stage_inputs": {
            "#4": {"dets": "det_filter#3.dets", "image": "frame_manager#0.frame"}}})
        loaded3 = db.load_pipeline(pid)
        check("DB: 插回阶段后连线位置还原（#4 双端口）",
              json.dumps(loaded3["stages"][4].get("inputs"), sort_keys=True)
              == json.dumps(want_inputs, sort_keys=True),
              str(loaded3["stages"][4].get("inputs")))
    finally:
        if pid:
            try:
                db.delete_pipeline(pid)
            except Exception:
                pass
        for i in made:
            try:
                db.delete_preset(i)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="图编排 web 侧自检")
    ap.add_argument("--section", choices=["p20", "p21"], default=None)
    args = ap.parse_args()
    if args.section in (None, "p20"):
        section_p20()
    if args.section in (None, "p21"):
        section_p21()
    total_note = f"失败 {len(_FAILS)} 项" if _FAILS else "全部通过"
    print(f"\n{'=' * 12} {total_note} {'=' * 12}")
    for f in _FAILS:
        print(f"  [FAIL] {f}")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
