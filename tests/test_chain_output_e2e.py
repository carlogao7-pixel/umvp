# -*- coding: utf-8 -*-
"""链路尾输出提炼声明端到端验收（WO-0029.3，设计 doc/design/chain_output.md §3-§7）。

两层结构（验收语义：筛选侧收敛 × 提炼侧配图出场 × web 驱动层收尾）：

  默认（纯逻辑，秒级；不加载模型权重、不起 HTTP 服务、不连 MySQL——server import 惰性）：
    [A] 筛选侧×提炼侧组合：SeqRule.source="max:score" 窗口选优后，collect_output
        提炼块的事件源记录确为窗口内最高分；detail_fields 投影（顶层+嵌套点路径）
        与固定统计并存进事件 detail；scope 过滤 / image_pick / limit 截断全链组合。
    [B] det_filter 事件规格 detail 点路径 → 事件 detail → collect_output（组合）。
    [C] web 驱动层进程内闭环：_resolve_output_plan → _make_output_accumulator
        （逐帧 _output_pick_from 运行时裁决 + add）→ _finish_output（tmpdir 落盘）→
        提炼块字段/落盘文件/thumbs（缺省 from 运行时解析契约，WO-0029.5）。
    [D] 向后兼容：无 output 段 → 不产计划（web）+ collect_output 缺省语义不变。
    [E] validate_chain CLI（subprocess 真跑）：含 output 段合法 spec 退出 0；
        limit=-1 / scope 非法 / image_pick 非法 / source 非法 / detail_fields 非法
        各自报错退出非 0。

  --live（真实推理，需测试台已运行 http://127.0.0.1:8001/）：
    三条保留链路各配 output 声明端到端（注入 DB → 运行 → 断言 → 恢复并校验一致）：
    PIPE0001（scope=events + Image 帧级配图 + 报告区块 + SSE note + 向后兼容重跑）、
    PIPE0002（records 图清单可达验收 + events 事件提炼：publish 通路 from=pipeline.tail
    显式与缺省各一次——WO-0029.4/0029.5 修复后翻绿，见 live_pipe0002 docstring）、
    PIPE0003（scope=records 图档索引语义）。
    注入均走正式 API（/api/presets/save + /api/pipelines/save），测后 finally 恢复
    留底并逐键校验一致（DF0001 events 规格 / SEQ0002 rules / 各链路 meta.output）。
    基线契约：测试恢复到 **stash 时状态**——跑前请确保 DB 处于播种基线（DF0001/SEQ0002
    未手工改过、各链路 meta 无 output 段），否则恢复的是你自己的改动而非播种值。

运行:
    conda run -n ai python tests/test_chain_output_e2e.py           # 纯逻辑
    conda run -n ai python tests/test_chain_output_e2e.py --live    # + 三链路真实推理
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from pipe.chain_output import collect_output  # noqa: E402
from pipe.det_filter import DetFilter  # noqa: E402
from pipe.records import add_event  # noqa: E402
from pipe.seq_filter import SeqFilter, SeqRule  # noqa: E402

PASS = 0       # 纯逻辑断言计数
PASS_LIVE = 0  # --live 断言计数
GAPS: list = []  # live 已知缺口（定性为被测模块缺陷、不阻断其余验收；回报用）
LIVE = "--live" in sys.argv


def ok(cond, msg: str) -> None:
    global PASS
    assert cond, f"断言失败: {msg}"
    PASS += 1
    print(f"  [ok] {msg}")


def ok_live(cond, msg: str) -> None:
    global PASS_LIVE
    assert cond, f"LIVE 断言失败: {msg}"
    PASS_LIVE += 1
    print(f"  [live-ok] {msg}")


def img(tag=0):
    """假图：不同 tag 形状不同，便于断言对位。"""
    return np.zeros((8 + tag, 16 + tag, 3), dtype=np.uint8)


# ---------------- [A] 筛选侧 × 提炼侧组合语义 ----------------

def test_a_source_and_projection():
    print("[A] SeqRule.source/detail_fields × collect_output 组合")
    # 窗口规则：window_n=3、min_n=2；三条命中 score=0.6/0.95/0.7，最高分**非最后到达**
    # source=max:score → 事件落在 0.95 那条记录上（而非首条/末条）
    recs = [{"kind": "det", "label": "car", "track_id": 7, "cls_id": 2,
             "score": s, "ts": 1.0 + i * 0.5, "frame_idx": 30 + i * 15,
             "bbox": (0, 0, 40, 20)}
            for i, s in enumerate([0.6, 0.95, 0.7])]
    # 投影字段：顶层（score）+ 嵌套点路径（content.result，构造嵌套 dict 记录）
    recs[1]["content"] = {"result": "占道", "conf": 0.88}
    sf = SeqFilter(rules=[SeqRule(match="cls_id == 2", event="{label}", key="track_id",
                                  window_n=3, min_n=2, source="max:score",
                                  detail_fields=("score", "content.result"))])
    out = sf.run(recs)
    ok(len(out) == 1, f"窗口达成边沿放行一条（实际 {len(out)}）")
    src = out[0]
    ok(src["score"] == 0.95 and src["frame_idx"] == 45,
       f"source=max:score 事件源=窗口内最高分记录（score={src.get('score')}）")
    ev = src["events"][0]
    ok(set(ev["detail"]) >= {"key", "count", "span", "window_n", "score", "content.result"},
       f"detail 固定统计与投影字段并存: {sorted(ev['detail'])}")
    ok(ev["detail"]["content.result"] == "占道", "嵌套点路径投影取值正确")
    ok(ev["detail"]["score"] == 0.95, "顶层投影字段取值正确")

    # 提炼块：事件源记录 = 上述最高分记录；通用字段 + events 明细全携带
    blk = collect_output(out, decl={"scope": "events"})
    ok(blk["count"] == 1 and blk["items"][0]["score"] == 0.95,
       "collect_output 事件的承载记录确为窗口最高分（提炼块与筛选选优一致）")
    ok(blk["items"][0]["events"][0]["detail"]["content.result"] == "占道"
       and blk["items"][0]["events"][0]["detail"]["count"] == 3,
       "提炼块 events 明细同时含投影字段（点路径）与固定统计（count=窗口命中数）")

    # 对照：source 缺省 first → 窗口凑满时放行**当前到达**记录（score=0.7，非最高分 0.95），
    # 证明「事件落在 0.95」确由 source 驱动
    sf2 = SeqFilter(rules=[SeqRule(match="cls_id == 2", event="{label}", key="track_id",
                                   window_n=3, min_n=2)])
    out2 = sf2.run([dict(r) for r in recs])
    ok(out2 and out2[0]["score"] == 0.7, f"对照 source 缺省 first（score={out2[0]['score']}）")

    # scope 过滤组合：门控放行的无事件记录不进 events 块；records 模式全量
    extra = [{"kind": "det", "label": "car", "cls_id": 2, "track_id": 99,
              "score": 0.5, "ts": 9.0, "frame_idx": 900},   # 无事件（如纯门控放行）
             {"kind": "det", "label": "car", "cls_id": 2, "track_id": 98,
              "score": 0.4, "ts": 9.5, "frame_idx": 915}]
    blk_ev = collect_output(out + extra, decl={"scope": "events"})
    blk_rec = collect_output(out + extra, decl={"scope": "records"})
    ok(blk_ev["count"] == 1 and blk_rec["count"] == len(out) + len(extra),
       f"scope 过滤: events={blk_ev['count']} / records={blk_rec['count']}（全量）")

    # image_pick：一条记录多图（图集旁路）all 全出 / first 取首张
    recs[1]["image"] = [img(1), img(2), img(3)]
    blk_all = collect_output(out, image_records=out, decl={"scope": "events", "image_pick": "all"})
    blk_first = collect_output(out, image_records=out,
                               decl={"scope": "events", "image_pick": "first"})
    ok(len(blk_all["items"][0]["images"]) == 3, "image_pick=all 图集全出（3 张）")
    ok(len(blk_first["items"][0]["images"]) == 1, "image_pick=first 取首张")

    # limit 截断：多条事件记录超限截断并标 truncated
    many = []
    for i in range(5):
        r = {"kind": "det", "label": "car", "cls_id": 2, "track_id": 100 + i,
             "score": 0.9, "ts": float(i), "frame_idx": i}
        add_event(r, "目标出现", ts=float(i), detail={"key": str(100 + i)})
        many.append(r)
    blk_lim = collect_output(many, decl={"scope": "events", "limit": 3})
    ok(blk_lim["count"] == 3 and blk_lim["truncated"] and len(blk_lim["items"]) == 3,
       "limit 截断 + truncated 标注")


def test_b_det_filter_dotpath():
    print("[B] det_filter detail 点路径 × collect_output")
    recs = [{"kind": "det", "label": "car", "cls_id": 2, "track_id": 7,
             "score": 0.9, "ts": 1.0, "frame_idx": 15,
             "content": {"result": "违停"}}]
    f = DetFilter(events=[{"name": "目标出现", "when": "cls_id == 2",
                           "detail": ["score", "content.result"]}])
    out = f.run(recs, ts=1.0, frame_idx=15)
    ev = out[0]["events"][0]
    ok(ev["detail"].get("content.result") == "违停" and ev["detail"].get("score") == 0.9,
       f"det_filter detail 点路径投影: {ev['detail']}")
    blk = collect_output(out, decl={"scope": "events"})
    ok(blk["items"][0]["events"][0]["detail"]["content.result"] == "违停",
       "提炼块 events 明细含 det_filter 点路径投影字段")


# ---------------- [C] web 驱动层进程内闭环（plan → 累积 → 收尾落盘） ----------------

_UAV_STAGES = [
    {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 15}},
    {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "0,2"}},
    {"module_id": "det_filter", "params": {"group_limits": "0:2,300;2:1,300"}},
    {"module_id": "annotate", "params": {"thickness": 2, "show_label": True}},
    {"module_id": "vlm", "params": {"use_real_vlm": False}},
    {"module_id": "seq_filter", "params": {"throttle": "cls_id:3.0", "rules": ""}},
]

_PLATE_STAGES = [
    {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 5}},
    {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "2"}},
    {"module_id": "det_filter", "params": {"filter": "long >= 40", "group_limits": "2:1,300"}},
    {"module_id": "gate_filter", "params": {"key": "track_id", "interval": 5.0}},
    {"module_id": "target_crop", "params": {"out_size": "×2", "margin": 0.1, "classes": "2"}},
    {"module_id": "plate_recog", "params": {"detect_level": "high", "mode": "crop",
                                            "track_dedup": False}},
    {"module_id": "seq_filter",
     "params": {"rules": json.dumps([{"match": "kind == plate", "event": "{label}",
                                      "key": "track_id,label", "cooldown": -1.0}],
                                     ensure_ascii=False)}},
]


def _plate_rec(plate_no, tid, ts, frame, im=None):
    r = {"kind": "plate", "label": plate_no, "text": plate_no, "plate_no": plate_no,
         "color": "blue", "score": 0.97, "track_id": tid, "bbox": [10, 10, 60, 24],
         "ts": ts, "frame_idx": frame}
    if im is not None:
        r["image"] = im
    return r


def test_c_web_driver_loop():
    print("[C] web 驱动层闭环（PIPE0002 形态：SeqFilter → 累积器 → _finish_output）")
    import server  # noqa: E402  （惰性连接，无 HTTP/DB 副作用）

    spec = {"name": "e2e", "stages": _PLATE_STAGES,
            "output": {"scope": "events", "image": "plate_recog.plates",
                       "image_pick": "all", "limit": 5}}
    plan, err = server._resolve_output_plan(spec)
    # WO-0029.5：缺省 from 转运行时解析（deferred 计划），静态候选只是 ① 级预报
    ok(err is None and plan.get("from_deferred") and plan["from_full"] is None
       and plan["tail_froms"] == ["seq_filter.dets", "seq_filter#6.dets"],
       f"plan 解析（缺省 from=deferred，静态候选 {plan and plan.get('tail_froms')}）")
    ok(plan["image_full"] == "plate_recog#5.plates" and plan["image_type"] == "Records",
       "image=plate_recog.plates 解析为 Records 配图源")

    # 事件尾真跑（与 PIPE0002 播种同形态 + detail_fields 投影）→ 逐帧喂累积器
    # （驱动同款：缺省 from 逐帧 _output_pick_from 运行时裁决——此处模拟 publish
    #   出事件登记了 pipeline.tail，末段 seq_filter rules-only 不登记 dets 产物）
    rules = [{"match": "kind == plate", "event": "{label}", "key": "track_id,label",
              "cooldown": -1.0, "detail_fields": ["plate_no", "color"]}]
    sf = SeqFilter(rules=[SeqRule(**r) for r in rules])
    acc = server._make_output_accumulator(plan)
    frames = [
        (10.0, 50, [("京A12345", 7)]),
        (15.0, 125, [("京B88888", 9)]),
        (20.0, 200, [("京C66666", 11)]),
    ]
    for ts, frame, plates in frames:
        recs = []
        imgs = []
        for pn, tid in plates:
            im = img(tid % 5)
            recs += sf.run([_plate_rec(pn, tid, ts, frame, im)])
            imgs.append({"frame_idx": frame, "image": im})
        frm = server._output_pick_from(plan, {"pipeline.tail": recs})
        acc["add"](recs if frm else [], imgs if frm else None)
    ok(plan.get("_used_publish") is True and plan["_used_from"] == "pipeline.tail",
       "缺省 from 运行时命中 pipeline.tail（publish 通路，WO-0029.5）")

    with tempfile.TemporaryDirectory(prefix="umvp_e2e_") as td:
        out_dir = Path(td)
        saved: list = []
        counters = [0]

        def save_image(path, im):
            if counters[0] >= 200:
                return False
            cv2.imwrite(str(path), im)
            counters[0] += 1
            saved.append(path.name)
            return True

        blk = server._finish_output(plan, acc, out_dir, "jpg", save_image,
                                    start_frame=0, saved_files=saved)
        ok(blk["count"] == 3 and not blk["truncated"],
           f"提炼块 count={blk['count']}（3 条车牌事件）")
        d0 = blk["items"][0]["events"][0]["detail"]
        ok({"key", "count", "span", "window_n"} <= set(d0)
           and {"plate_no", "color"} <= set(d0),
           f"事件 detail 固定统计与声明投影并存: {sorted(d0)}")
        ok(blk["items"][0]["label"] == "京A12345"
           and blk["items"][0]["track_id"] == 7, "item 通用字段（label/track_id）")
        p0 = blk["items"][0]["images"]
        ok(len(p0) == 1 and p0[0].startswith("output_001_") and (out_dir / p0[0]).is_file(),
           f"配图落盘且文件真实存在: {p0}")
        ok(blk["items"][0]["thumbs"][0].startswith("data:image/jpeg;base64,"),
           "thumbs 缩略 data URL 随块返回")
        ok(all(n.startswith("output_") for n in saved) and len(saved) == 3,
           "共用 saved_files 清单（3 张提炼图）")

        # limit=2 → 截断标注（同一累积数据重收尾）
        plan_lim = dict(plan, limit=2)
        blk2 = server._finish_output(plan_lim, acc, None, save_image=None, note="流式")
        ok(blk2["count"] == 2 and blk2["truncated"] and "note" in blk2
           and all(not it["images"] for it in blk2["items"]),
           "limit 截断 + 流式（output_dir=None）不落盘 images 恒空 + note")


# ---------------- [D] 向后兼容（纯逻辑面） ----------------

def test_d_backward_compat():
    print("[D] 向后兼容：无 output 段")
    import server  # noqa: E402

    plan, err = server._resolve_output_plan({"name": "t", "stages": _UAV_STAGES})
    ok(plan is None and err is None, "无 output 段 → 不产计划（驱动层零改动路径）")
    plan, err = server._resolve_output_plan({"name": "t", "stages": _UAV_STAGES,
                                             "meta": {"description": "x"}})
    ok(plan is None and err is None, "meta 无 output（仅 description）→ 同样不产计划")
    # collect_output 缺省语义（=链路无声明时收尾的兜底形态）不受影响
    r = {"kind": "det", "label": "car", "ts": 0.0, "frame_idx": 1}
    add_event(r, "E", ts=0.0)
    ok(collect_output([r])["count"] == 1, "collect_output 缺省 decl 行为不变")


# ---------------- [E] validate_chain CLI（subprocess 真跑） ----------------

def _write_spec(td: Path, output=None, seq_extra=None) -> Path:
    spec = {"name": "e2e-cli", "stages": json.loads(json.dumps(_PLATE_STAGES))}
    if seq_extra:
        spec["stages"][-1]["params"]["rules"] = json.dumps(
            [dict({"match": "kind == plate", "event": "{label}",
                   "key": "track_id,label", "cooldown": -1.0}, **seq_extra)],
            ensure_ascii=False)
    if output is not None:
        spec["output"] = output
    p = Path(td) / "spec.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return p


def _run_cli(spec_path: Path):
    r = subprocess.run([sys.executable, str(ROOT / "web_lab" / "validate_chain.py"),
                        str(spec_path)], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout + r.stderr)


def test_e_validate_cli():
    print("[E] validate_chain CLI：合法通过 / 非法报错（退出非 0）")
    with tempfile.TemporaryDirectory(prefix="umvp_e2eval_") as td:
        tdp = Path(td)
        code, out = _run_cli(_write_spec(tdp, output={
            "scope": "events", "from": "seq_filter.dets",
            "image": "plate_recog.plates", "image_pick": "all", "limit": 10},
            seq_extra={"source": "max:score", "detail_fields": ["plate_no", "color"]}))
        ok(code == 0 and "[OK]" in out,
           f"含 output 段 + source + detail_fields 合法 spec 通过（exit={code}）")

        bad_cases = [
            ("limit=-1", {"limit": -1}, None, "limit"),
            ("scope 非法", {"scope": "all"}, None, "scope"),
            ("image_pick 非法", {"image_pick": "many"}, None, "image_pick"),
            ("source 非法", {}, {"source": "min:score"}, "source"),
            ("detail_fields 非法", {}, {"detail_fields": [123]}, "detail_fields"),
        ]
        for name, output, seq_extra, keyword in bad_cases:
            code, out = _run_cli(_write_spec(tdp, output=output, seq_extra=seq_extra))
            ok(code != 0 and keyword in out,
               f"{name} → 报错退出 {code}（含 {keyword!r}）")


# ---------------- --live：三条保留链路 HTTP 端到端 ----------------

BASE = "http://127.0.0.1:8001"


def _api_get(path, timeout=600):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _api_post(path, body, timeout=600):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class _Snapshot:
    """DB 夹具留底/恢复：presets（params）与 pipelines（steps+meta）原样回写。"""

    def __init__(self):
        self.presets: dict = {}    # preset_id -> {name, module_id, params}
        self.pipelines: dict = {}  # pipeline_id -> {name, steps, streams, budget, meta}

    def stash_pipeline(self, pid):
        pl = _api_post("/api/pipelines/load", {"id": pid})["pipeline"]
        self.pipelines[pid] = {
            "name": pl["name"], "streams": pl.get("streams") or 1,
            "budget": pl.get("budget") or {}, "meta": pl.get("meta") or {},
            "steps": [s["preset_id"] for s in pl["stages"]],
            "stages": pl["stages"]}
        return self.pipelines[pid]

    def stash_preset(self, preset_id):
        row = next(p for p in _api_get("/api/presets?kind=module")["presets"]
                   if p["id"] == preset_id)
        self.presets[preset_id] = {"name": row["name"], "module_id": row["module_id"],
                                   "params": dict(row["params"])}
        return self.presets[preset_id]

    def restore_and_verify(self) -> list:
        """恢复留底并校验一致，返回不一致清单（空=全部恢复一致）。"""
        diffs = []
        for pid, keep in self.presets.items():
            _api_post("/api/presets/save", {"kind": "module", "preset_id": pid,
                                            "name": keep["name"],
                                            "module_id": keep["module_id"],
                                            "params": keep["params"]})
            now = _api_post("/api/presets/load", {"id": pid})["preset"]
            if now["params"] != keep["params"]:
                diffs.append(f"preset {pid} params 未恢复一致")
        for pid, keep in self.pipelines.items():
            _api_post("/api/pipelines/save",
                      {"name": keep["name"], "steps": keep["steps"],
                       "streams": keep["streams"], "budget": keep["budget"],
                       "meta": keep["meta"]})
            now = _api_post("/api/pipelines/load", {"id": pid})["pipeline"]
            if (now.get("meta") or {}) != keep["meta"]:
                diffs.append(f"pipeline {pid} meta 未恢复一致")
            if [s["preset_id"] for s in now["stages"]] != keep["steps"]:
                diffs.append(f"pipeline {pid} steps 未恢复一致")
        return diffs


def _set_meta_output(snap: _Snapshot, pid: str, output: dict) -> None:
    """给链路 meta 注入 output 段（save_pipeline 同名覆盖；steps/preset 不动）。"""
    keep = snap.pipelines[pid]
    meta = dict(keep["meta"])
    meta["output"] = output
    r = _api_post("/api/pipelines/save", {"name": keep["name"], "steps": keep["steps"],
                                          "streams": keep["streams"],
                                          "budget": keep["budget"], "meta": meta})
    assert r.get("ok"), f"注入 {pid} meta.output 失败: {r}"


def _set_preset_params(snap: _Snapshot, preset_id: str, params: dict) -> None:
    keep = snap.presets[preset_id]
    r = _api_post("/api/presets/save", {"kind": "module", "preset_id": preset_id,
                                        "name": keep["name"],
                                        "module_id": keep["module_id"],
                                        "params": params})
    assert r.get("ok"), f"改 preset {preset_id} 失败: {r}"


def _run_sync(pid, video, max_frames):
    # /api/test/pipeline_run 的运行参数嵌在 params 键下（do_POST 分发约定）
    return _api_post("/api/test/pipeline_run", {"params": {
        "pipeline_id": pid, "video": video, "max_frames": max_frames,
        "vlm_feedback": False}})


def _sse_done(pid, video, max_frames, timeout=600):
    """流式运行读到 done 事件 payload（SSE；流式不落盘 → output.images 恒空 + note）。"""
    qs = urllib.parse.urlencode({"pipeline_id": pid, "video": video,
                                 "max_frames": max_frames, "vlm_feedback": "false"})
    done = None
    with urllib.request.urlopen(f"{BASE}/api/test/pipeline_run/stream?{qs}",
                                timeout=timeout) as r:
        event = None
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: ") and event == "done":
                done = json.loads(line[6:])
                break
    assert done is not None, "SSE 未读到 done 事件"
    return done


def _outdir() -> Path:
    return ROOT / "tests/data/out/test_pipeline"


def _clean_output_files():
    """清掉提炼图 output_*（本轮断言文件存在用；历史帧图 frame_* 不动）。"""
    for p in _outdir().glob("output_*"):
        p.unlink(missing_ok=True)


def live_pipe0001():
    print("[live-1] PIPE0001 警用无人机链路（scope=events + Image 帧级配图 + 报告 + SSE）")
    snap = _Snapshot()
    snap.stash_pipeline("PIPE0001")
    pl = snap.pipelines["PIPE0001"]
    df_pid = next(s["preset_id"] for s in pl["stages"] if s["module_id"] == "det_filter")
    snap.stash_preset(df_pid)
    try:
        # 注入：DF0001 加检测事件规格（car 触发；person 会被 group_limits 剔除）
        df_params = dict(snap.presets[df_pid]["params"])
        df_params["events"] = json.dumps([{
            "name": "目标出现", "when": "cls_id == 2",
            "detail": ["score", "track_id"]}], ensure_ascii=False)
        _set_preset_params(snap, df_pid, df_params)
        _set_meta_output(snap, "PIPE0001", {
            "scope": "events", "from": "det_filter.dets",
            "image": "annotate.annotated", "limit": 3})

        _clean_output_files()
        r = _run_sync("PIPE0001", "远角车+人1.MP4", 120)
        ok_live(r.get("ok"), f"运行成功（{r.get('elapsed_sec')}s）")
        out = r.get("output")
        ok_live(isinstance(out, dict) and "error" not in out,
                f"result.output 存在且无 error: {out and {k: out[k] for k in ('scope', 'from') if k in out}}")
        ok_live(out["scope"] == "events" and out["from"] == "det_filter#2.dets",
                f"scope/from 正确: {out['scope']} / {out['from']}")
        ok_live(out["count"] > 0, f"count>0（实际 {out['count']}）")
        it0 = out["items"][0]
        ok_live("score" in it0["events"][0]["detail"]
                and "track_id" in it0["events"][0]["detail"],
                f"events 明细含投影字段 score/track_id: {it0['events'][0]['detail']}")
        ok_live(it0["events"][0]["name"] == "目标出现", "事件名正确")
        imgs = it0.get("images") or []
        ok_live(bool(imgs) and all((_outdir() / f).is_file() for f in imgs),
                f"images 落盘引用且文件真实存在: {imgs}")
        ok_live(out.get("count") == 3 and out.get("truncated") is True,
                f"limit=3 截断生效（count={out.get('count')}, truncated={out.get('truncated')}）")

        # 报告导出：「链路尾输出」区块优先呈现（事件表前）+ 内嵌图
        rep = _api_post("/api/test/report", {})
        ok_live(rep.get("ok"), "报告导出成功")
        html = rep["html"]
        ok_live("链路尾输出" in html, "报告含「链路尾输出」区块")
        if "链路尾输出" in html:
            ok_live(html.index("链路尾输出") < html.index("事件列表"),
                    "区块优先呈现（事件表之前）")
            ok_live("data:image/jpeg;base64," in html, "内嵌提炼图 data URL")

        # SSE 流式：done.output 带 note、items.images 恒空（流式不落盘）
        done = _sse_done("PIPE0001", "远角车+人1.MP4", 45)
        sout = done.get("output")
        ok_live(isinstance(sout, dict) and sout.get("note"),
                f"SSE done.output 带 note: {sout and sout.get('note', '')[:30]}…")
        ok_live(all(not it.get("images") for it in sout.get("items", [])),
                "流式 items.images 恒空")

        # 向后兼容：恢复 DF0001 与 meta 后重跑 → result 无 output 键、报告无区块
        _set_preset_params(snap, df_pid, snap.presets[df_pid]["params"])
        keep_meta = snap.pipelines["PIPE0001"]["meta"]
        _api_post("/api/pipelines/save", {
            "name": pl["name"], "steps": pl["steps"], "streams": pl["streams"],
            "budget": pl["budget"], "meta": keep_meta})
        r2 = _run_sync("PIPE0001", "远角车+人1.MP4", 120)
        ok_live(r2.get("ok") and "output" not in r2,
                "向后兼容：无 output 段重跑 → result 无 output 键")
        rep2 = _api_post("/api/test/report", {})
        ok_live(rep2.get("ok") and "链路尾输出" not in rep2["html"],
                "向后兼容：报告无「链路尾输出」区块")
    finally:
        diffs = snap.restore_and_verify()
        ok_live(not diffs, f"测后恢复校验（PIPE0001 meta + DF0001 params）: {diffs or '一致'}")


def live_pipe0002():
    """PIPE0002 车牌链路。事件尾（SEQ0002 rules）走 publish 通路——WO-0029.4 登记
    publish 承载记录产物 pipeline.tail、WO-0029.5 缺省 from 改运行时解析后，events
    事件提炼可达（原 WO-0029.3 验收 GAP 已翻绿）。注意：rules-only seq_filter 阶段
    运行时从不登记 seq_filter.dets，events 模式 from 须为 pipeline.tail（或缺省）。
    本函数三段：
      ① 可达验收（scope=records + from=plate_recog.plates）：车牌记录提炼 + 车牌图落盘；
      ② 事件提炼·显式 from=pipeline.tail：事件明细（固定统计+投影并存）+ 车牌图配图；
      ③ 事件提炼·缺省 from（去掉 from）：运行时裁决命中 pipeline.tail，同样命中。
    """
    print("[live-2] PIPE0002 车牌链路（records 图清单 + events 事件提炼：publish 通路）")
    snap = _Snapshot()
    snap.stash_pipeline("PIPE0002")
    pl = snap.pipelines["PIPE0002"]
    seq_pid = next(s["preset_id"] for s in pl["stages"] if s["module_id"] == "seq_filter")
    snap.stash_preset(seq_pid)
    try:
        # 注入：SEQ0002 rules 加 detail_fields（投影字段声明在事件尾规则，单一来源）。
        # 车牌 publish 承载记录形态（WO-0029.4）：plate_no/color 在嵌套 content 里，
        # 投影用点路径下钻（WO-0029.1）；score 为顶层字段。
        seq_params = dict(snap.presets[seq_pid]["params"])
        seq_params["rules"] = json.dumps([{
            "match": "kind == plate", "event": "{label}", "key": "track_id,label",
            "cooldown": -1.0,
            "detail_fields": ["content.plate_no", "content.color", "score"]}],
            ensure_ascii=False)
        _set_preset_params(snap, seq_pid, seq_params)

        # ① 可达验收：scope=records + 显式 from（plate_recog.plates 运行时每帧登记）
        _set_meta_output(snap, "PIPE0002", {
            "scope": "records", "from": "plate_recog.plates",
            "image": "plate_recog.plates", "image_pick": "all", "limit": 10})
        _clean_output_files()
        r = _run_sync("PIPE0002", "车牌识别.MP4", 300)
        ok_live(r.get("ok"), f"运行成功（{r.get('elapsed_sec')}s）")
        out = r.get("output")
        ok_live(isinstance(out, dict) and "error" not in out
                and out.get("count", 0) > 0,
                f"records 模式提炼出车牌记录（count={out and out.get('count')}）")
        if out and out.get("count", 0) > 0:
            it0 = out["items"][0]
            ok_live(it0.get("label"), f"车牌记录通用字段（label={it0.get('label')}）")
            imgs = it0.get("images") or []
            ok_live(bool(imgs) and all((_outdir() / f).is_file() for f in imgs),
                    f"车牌图落盘且文件真实存在: {imgs[:2]}")
            ok_live(all(it["events"] == [] for it in out["items"]),
                    "records 直取产物记录：events 恒空（事件在 publish 承载记录上）")

        # ② 事件提炼·显式 from=pipeline.tail（publish 通路承载记录，WO-0029.4/0029.5）
        _set_meta_output(snap, "PIPE0002", {
            "scope": "events", "from": "pipeline.tail",
            "image": "plate_recog.plates", "image_pick": "all", "limit": 10})
        _clean_output_files()
        r2 = _run_sync("PIPE0002", "车牌识别.MP4", 300)
        ok_live(r2.get("ok"), "第二次运行成功")
        out2 = (r2 or {}).get("output") or {}
        ok_live(out2.get("count", 0) > 0 and "error" not in out2,
                f"events 模式提炼出车牌事件（count={out2.get('count')}）")
        ok_live(out2.get("from") == "pipeline.tail", "块头 from=pipeline.tail")
        ok_live("publish" in (out2.get("note") or ""),
                f"publish 通路来源 note: {out2.get('note')}")
        if out2.get("count", 0) > 0:
            d0 = out2["items"][0]["events"][0]["detail"]
            ok_live({"key", "count", "span", "window_n"} <= set(d0),
                    "事件 detail 含固定统计（key/count/span/window_n）")
            ok_live({"content.plate_no", "content.color", "score"} <= set(d0),
                    "事件 detail 含声明投影字段（点路径下钻 content，与固定统计并存）")
            imgs2 = out2["items"][0].get("images") or []
            ok_live(bool(imgs2) and all((_outdir() / f).is_file() for f in imgs2),
                    f"车牌图同帧配图落盘且文件真实存在: {imgs2[:2]}")

        # ③ 事件提炼·缺省 from（去掉 from）：运行时裁决命中 pipeline.tail，同样翻绿
        _set_meta_output(snap, "PIPE0002", {
            "scope": "events", "image": "plate_recog.plates",
            "image_pick": "all", "limit": 10})
        _clean_output_files()
        r3 = _run_sync("PIPE0002", "车牌识别.MP4", 300)
        ok_live(r3.get("ok"), "第三次运行成功（缺省 from）")
        out3 = (r3 or {}).get("output") or {}
        ok_live(out3.get("count", 0) > 0 and "error" not in out3,
                f"缺省 from 同样命中（count={out3.get('count')}，缺省优先级生效）")
        ok_live(out3.get("from") == "pipeline.tail",
                f"缺省 from 块头呈现运行时实际来源: {out3.get('from')}")
    finally:
        diffs = snap.restore_and_verify()
        ok_live(not diffs, f"测后恢复校验（PIPE0002 meta + SEQ0002 rules）: {diffs or '一致'}")


def live_pipe0003():
    print("[live-3] PIPE0003 人脸截取链路（scope=records 图档索引）")
    snap = _Snapshot()
    snap.stash_pipeline("PIPE0003")
    try:
        _set_meta_output(snap, "PIPE0003", {
            "scope": "records", "image": "face_detect.faces",
            "image_pick": "all", "limit": 20})
        _clean_output_files()
        r = _run_sync("PIPE0003", "face1.MP4", 100)
        ok_live(r.get("ok"), f"运行成功（{r.get('elapsed_sec')}s）")
        out = r.get("output")
        ok_live(isinstance(out, dict) and "error" not in out, "提炼块产出且无 error")
        ok_live(out.get("scope") == "records", f"scope=records: {out.get('scope')}")
        ok_live(out.get("count", 0) > 0, f"records 模式产出图清单（count={out.get('count')}）")
        if out.get("count", 0) > 0:
            ok_live(all(it["events"] == [] for it in out["items"]),
                    "无事件链路：items.events 恒空列表")
            with_imgs = [it for it in out["items"] if it.get("images")]
            ok_live(bool(with_imgs)
                    and all((_outdir() / f).is_file()
                            for it in with_imgs for f in it["images"]),
                    f"图档索引：配图落盘且文件真实存在"
                    f"（{sum(len(it.get('images') or []) for it in out['items'])} 张）")
    finally:
        diffs = snap.restore_and_verify()
        ok_live(not diffs, f"测后恢复校验（PIPE0003 meta）: {diffs or '一致'}")


def run_live() -> None:
    # 服务在跑吗（JSON 端点；/ 是 test.html 页面）
    try:
        _api_get("/api/pipelines", timeout=10)
    except Exception as e:
        raise SystemExit(f"测试台未运行（{BASE}）: {e} —— 先 bash web_lab/restart.sh")
    t0 = time.time()
    live_pipe0001()
    live_pipe0002()
    live_pipe0003()
    print(f"\nlive 通过: {PASS_LIVE} 项断言（{time.time() - t0:.0f}s）")
    if GAPS:
        print(f"已知缺口 {len(GAPS)} 项（不阻断，见回报）:")
        for g in GAPS:
            print(f"  - {g}")


def main() -> None:
    test_a_source_and_projection()
    test_b_det_filter_dotpath()
    test_c_web_driver_loop()
    test_d_backward_compat()
    test_e_validate_cli()
    print(f"\n纯逻辑通过: {PASS} 项断言")
    if LIVE:
        print()
        run_live()
        print(f"总计: 纯逻辑 {PASS} + live {PASS_LIVE} = {PASS + PASS_LIVE} 项断言")


if __name__ == "__main__":
    main()
