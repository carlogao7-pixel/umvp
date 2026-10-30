# -*- coding: utf-8 -*-
"""混合检索（WO-0033）标签链路验收（qa 板块）

设计依据：doc/design/hybrid_retrieval.md（§2.5 schema 冻结 / §2.6 采集时间视频级 /
§3 路径 A 归档+重对齐 / §4 review_index/review_video 存储 / §5 查询过滤 / §12 契约）。

验收对象（实现交付 WO-0033.2/.4/.5/.6）：
  - pipe：label_align（经 web 调用链生效）+ VideoEmbedStage dets 观察端口（observation
    旁路键）+ YOLO det 记录 label 字段（class_names 采集链）
  - web：事件归档 out/events/<video>.jsonl（契约 C，run_id 幂等，detail 截断）；
    review_index/review_video 落表（无 DB JSON 降级）；PIPE0004 播种 4 阶段观察段
    （frame_manager→yolo→det_filter→video_embed + stage_inputs 连线）；
    POST /api/review/relabel（只重对齐不动向量库）；/api/review/search 增
    time_start/time_end / event_name / video / mode

分层（全 mock 档、进程内、**JSON 降级后端**——不依赖 MySQL 与真端点，秒~十秒级）:
  [A] 归档契约：run_id 幂等 / detail 截断 / 读路径车牌 name 载体补 plate_no /
      跨运行去重合并 / 非法入参 fail-soft
  [B] 全链路：造档 → PIPE0004（播种 4 阶段 spec + dets 连线真跑）→ 落表断言
      （行数=片段数、(video,t_start) 与底库键集一致、观察层 det_stats/class_names、
      frame_dets 原始记录流结构（WO-0033.13：逐帧 {ts,recs} / kind 域 / 剥键 /
      JSON 可序列化 / ts 升序）、plates/events 路由与三层分工）+ pipe 侧车牌路由
      双保险（直调 label_align）
  [C] relabel：归档增量 → 只重对齐（标签行更新 + **向量库字节/mtime 不变** +
      frame_dets 逐行保留 §11 定案 6）+ 错误路径
  [D] 查询过滤：时间窗（无采集时间不参与过滤且 why 注明 / 有采集时间硬过滤）/
      event_name 精确匹配两路（events.name 与 plates.plate_no）/ video 过滤 /
      mode=hybrid 走过滤式且 note/why 说明 / signals 元信息 / 无过滤 WO-0032 文案兼容 /
      include_frame_dets 开关（缺省不带 / =true tags 附带，WO-0033.13）
  [E] JSON 降级文件形态（§2.5/§2.6 字段冻结同名同义）
  [F] SSE 路径落表（WO-0033.9 扩展修复回归）：h_pipeline_stream（即
      /api/test/pipeline_run/stream 端点处理函数，server.py 直调）跑同一 4 阶段
      播种 spec（独立 npz 临时目录）→ 断言 review_index 行 class_names 非空且
      键=det_stats 键——SSE 采集路径与同步路径同一喂管道契约

环境隔离：UMVP_REVIEW_*_JSON 指向临时文件、server._EVENTS_DIR 指向临时目录、
db.set_review_db_enabled(False) 强制 JSON 后端、WEMM env 屏蔽（测后全部恢复）。
进程内跑链路前按 INB-0109 先例先 db.configure_modules 注册参数 schema。

运行:
    conda run -n ai python tests/test_hybrid_retrieval.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from video_embed.embed_store import EmbedStore  # noqa: E402

PASS = 0
VIDEO = "violent1.mp4"          # tests/data/vid 素材（链路真跑用）
WEMM_ENV_KEYS = ("WEMM_BASE_URL", "WEMM_ENDPOINT", "WEMM_API_KEY", "WEMM_MODEL",
                 "WEMM_BASE_URL_FALLBACK")
REVIEW_ENV_KEYS = ("UMVP_REVIEW_INDEX_JSON", "UMVP_REVIEW_VIDEO_JSON")


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _file_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _jsonl_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _spec(server, store_path: str, thumb_dir: str) -> dict:
    """PIPE0004 播种形态 spec（单一来源 server._SEED_VEMB_STAGES，4 阶段观察段）。

    差异：store_path/thumb_dir 指向临时目录（不碰真实 out/）；video_embed 显式
    内联 inputs 连线（stage_inputs 声明——驱动层 _vem_feed_dets 只认连线，缺声明
    det_stats 恒空，WO-0033.5 冒烟实证）。
    """
    stages = [{"module_id": m, "params": dict(p)}
              for m, _n, p in server._SEED_VEMB_STAGES]
    vemb = stages[-1]
    vemb["params"].update({"store_path": store_path, "thumb_dir": thumb_dir})
    vemb["inputs"] = {"dets": "det_filter.dets"}
    return {"name": "视频向量化-嵌入查询链路", "stages": stages}


def _run_pipe0004(server, spec: dict, td: str, max_frames: int = 200) -> dict:
    orig_load = server.db.load_pipeline
    try:
        server.db.load_pipeline = lambda pid: dict(spec)
        return server.h_pipeline_run({"pipeline_id": "PIPE0004", "video": VIDEO,
                                      "max_frames": max_frames, "output_dir": td})
    finally:
        server.db.load_pipeline = orig_load


# ---------------- [A] 事件归档契约（契约 C） ----------------

def test_a_archive(server) -> Path:
    print("[A] 事件归档（run_id 幂等 / detail 截断 / name 载体补 plate_no / 去重合并）")
    evs = [
        {"name": "斗殴", "kind": "警情", "ts": 1.2, "detail": "x" * 500},
        {"name": "沪A12345", "kind": "车牌", "plate_no": "沪A12345", "ts": 2.0},
        {"name": "沪B88888", "kind": "车牌", "ts": 3.0},   # name 载体（无 plate_no 键）
    ]
    n = server._archive_run_events("qa_unit.mp4", "PIPE0001", evs, "qa-run-1")
    check("首归档返回写入事件数（3）", n == 3, f"n={n}")
    path = server._events_path_for("qa_unit.mp4")
    lines = _jsonl_lines(path)
    check("jsonl 落盘恰 1 行（一行一次运行）", len(lines) == 1, f"{len(lines)} 行")
    row = lines[0]
    check("归档行契约 C 顶层键齐备（run_id/pipeline/at/video/events）",
          all(k in row for k in ("run_id", "pipeline", "at", "video", "events")),
          str(sorted(row)))
    ev_rows = {e.get("name"): e for e in row["events"]}
    check("车牌 name 载体归一：写路径补 plate_no=沪B88888",
          ev_rows["沪B88888"].get("plate_no") == "沪B88888", str(ev_rows["沪B88888"]))
    det_text = json.dumps(ev_rows["斗殴"].get("detail"), ensure_ascii=False)
    check("detail 截断控体积（500 字符 → 摘要远小于原文且注明截断）",
          len(det_text) < 500 and "截断" in det_text, det_text[:60])

    n2 = server._archive_run_events("qa_unit.mp4", "PIPE0001", evs, "qa-run-1")
    check("同 run_id 重归档返回 0（幂等）", n2 == 0, f"n={n2}")
    check("同 run_id 重归档 jsonl 行数不变", len(_jsonl_lines(path)) == 1)
    n3 = server._archive_run_events("qa_unit.mp4", "PIPE0001", evs[:1], "qa-run-2")
    check("新 run_id 追加成功（返回 1、jsonl 2 行）",
          n3 == 1 and len(_jsonl_lines(path)) == 2, f"n={n3}")

    recs = server._read_archived_events("qa_unit.mp4")
    check("读路径合并两次运行共 4 条事件", len(recs) == 4, f"{len(recs)} 条")
    b = [e for e in recs if e.get("name") == "沪B88888"]
    check("读路径车牌 name 载体补 plate_no（web 读归一）",
          len(b) == 1 and b[0].get("plate_no") == "沪B88888", str(b))
    # 手工追加同 run_id 重复行（容灾：档被复制/重放）→ 读仍按 (run_id+事件) 去重
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(lines[0], ensure_ascii=False) + "\n")
    check("同 run_id 重复行重读不重复（run_id+事件去重）",
          len(server._read_archived_events("qa_unit.mp4")) == 4)

    check("空 video → 返回 0（不落盘）",
          server._archive_run_events("", "PIPE0001", evs, "qa-run-x") == 0)
    check("空事件 → 返回 0",
          server._archive_run_events("qa_unit.mp4", "PIPE0001", [], "qa-run-x") == 0)
    return path


# ---------------- [B] 全链路：造档 → PIPE0004 真跑 → 落表 ----------------

def test_b_full_chain(server, webdb, tmp: str, archive_path: Path, proj_tmp: Path) -> dict:
    print("[B] 全链路：PIPE0004（4 阶段播种 spec + dets 连线）→ 归档对齐落表")
    td = os.path.join(tmp, "chain")
    os.makedirs(td, exist_ok=True)
    npz = os.path.join(td, "video_vecs.npz")
    # thumb_dir 放项目根下（out/qa_hybrid_thumbs，测后清理）：thumb_path 契约是
    # 「项目相对」，thumb_dir 在项目根外会退化成穿目录相对路径（../../../..），偏离生产形态
    spec = _spec(server, npz, str(proj_tmp))

    video = ROOT / "tests/data/vid" / VIDEO
    if not video.is_file():
        check("测试视频存在 tests/data/vid/violent1.mp4", False, "素材缺失")
        raise SystemExit(1)
    check("测试视频存在 tests/data/vid/violent1.mp4", True)

    # ---- 第一次运行：无事件档（观察层落表基线）----
    r1 = _run_pipe0004(server, spec, td)
    check("PIPE0004 mock 全流程运行成功（#1）", r1.get("ok") is True,
          str(r1.get("error"))[:160])
    timeline = r1.get("timeline") or []
    stages_seen = {t.get("stage") for t in timeline if t.get("stage")}
    # det_filter 直通（模块默认不筛选）无事件/提交 → 不出 timeline 条目（记录口径）；
    # 观察段连线生效的直接证据 = 下方 det_stats/class_names 非空（_vem_feed_dets 只认
    # stage_inputs 连线声明，缺声明 det_stats 恒空——WO-0033.5 冒烟实证）
    check("timeline 含 yolo 与 video_embed 阶段",
          {"yolo", "video_embed"} <= stages_seen, str(sorted(stages_seen)))

    store = EmbedStore().load(npz)
    n_clips = len(store)
    metas = [store._clip_meta[i] for i in range(n_clips)]
    wins = sorted({(round(float(m["t_start"]), 3), round(float(m["t_end"]), 3))
                   for m in metas})
    check("片段数 ≥2（14 帧 @6 帧/窗 → 2 满窗 + 1 partial）", n_clips >= 2,
          f"{n_clips} 片段, 窗口 {wins}")

    rows = webdb.query_review_index(video=VIDEO)
    check("索引收尾自动落表：review_index 行数 = 片段数", len(rows) == n_clips,
          f"{len(rows)} vs {n_clips}")
    key_db = {(r["video"], round(float(r["t_start"]), 3)) for r in rows}
    key_vec = {(str(m.get("video")), round(float(m.get("t_start") or 0), 3))
               for m in metas}
    check("(video,t_start) 键集与向量底库完全一致（§4 一一对应）", key_db == key_vec,
          f"仅表 {key_db - key_vec} / 仅库 {key_vec - key_db}")
    merged_stats: dict = {}
    merged_names: dict = {}
    for r in rows:
        merged_stats.update(r.get("det_stats") or {})
        merged_names.update(r.get("class_names") or {})
    check("观察层 det_stats 非空且结构 = {cls_id:{peak,avg,tracks}}",
          merged_stats and all(
              isinstance(v, dict) and {"peak", "avg", "tracks"} <= set(v)
              for v in merged_stats.values()),
          json.dumps(merged_stats, ensure_ascii=False)[:100])
    check("class_names 非空（YOLO det 记录 label 采集链，WO-0033.6）",
          bool(merged_names), json.dumps(merged_names, ensure_ascii=False)[:80])
    check("class_names 键 = det_stats 键（cls_id 同域）",
          set(merged_names) == set(merged_stats),
          f"{sorted(merged_names)} vs {sorted(merged_stats)}")
    check("thumb_path 项目相对（§2.5 原则 4，换机不失效）",
          all(r.get("thumb_path") and not os.path.isabs(r["thumb_path"])
              and r["thumb_path"].startswith("out/") for r in rows),
          rows[0].get("thumb_path", "") if rows else "")
    vrows = webdb.list_review_videos(VIDEO)
    check("review_video 恰 1 行且字段齐备", len(vrows) == 1 and all(
        k in vrows[0] for k in ("video", "video_time", "video_time_src",
                                "indexed_at", "duration")))
    check("文件名无时间戳 → video_time/video_time_src 为空（不阻塞索引，§2.6）",
          vrows[0]["video_time"] == "" and vrows[0]["video_time_src"] == "",
          str(vrows[0]))
    check("review_video.duration 取窗口上界", float(vrows[0]["duration"] or 0) > 0,
          str(vrows[0]["duration"]))

    # ---- 造事件档（契约 C 原始形态，含 name 载体车牌）----
    ts_w0 = wins[0][0] + 1.0
    ts_w1 = wins[1][0] + 1.0
    run_a = {"run_id": "qa-run-a", "pipeline": "PIPE0001",
             "at": "2026-10-29T10:00:00+08:00", "video": VIDEO, "events": [
                 {"name": "斗殴", "kind": "警情", "ts": ts_w0,
                  "detail": {"who": "人群", "score": 0.9}},
                 {"name": "沪A12345", "kind": "车牌", "plate_no": "沪A12345",
                  "ts": ts_w0},
                 {"name": "沪B88888", "kind": "车牌", "ts": ts_w1},  # name 载体
             ]}
    ev_path = archive_path.parent / f"{server._safe_events_filename(VIDEO)}.jsonl"
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_a, ensure_ascii=False) + "\n")
    n_lines_before = len(_jsonl_lines(ev_path))
    check("事件档已造（PIPE0001 形态：警情 1 + 车牌 2，其中 1 条 name 载体）",
          n_lines_before == 1, f"{n_lines_before} 行")

    # ---- 第二次运行：收尾自动读档对齐落表 ----
    r2 = _run_pipe0004(server, spec, td)
    check("PIPE0004 重跑成功（#2，upsert 幂等）", r2.get("ok") is True,
          str(r2.get("error"))[:160])
    check("PIPE0004 自身无链路尾事件 → 归档零追加（§9 Q1 口径）",
          len(_jsonl_lines(ev_path)) == n_lines_before)
    rows2 = webdb.query_review_index(video=VIDEO)
    check("重跑后 review_index 行数不变（(video,t_start) upsert 幂等）",
          len(rows2) == n_clips, f"{len(rows2)} vs {n_clips}")

    by_win = {round(float(r["t_start"]), 3): r for r in rows2}
    plates_all = {p["plate_no"] for r in rows2 for p in (r.get("plates") or [])
                  if isinstance(p, dict) and p.get("plate_no")}
    check("plates[] 路由：两车牌（plate_no 键 + name 载体）均进判断层",
          {"沪A12345", "沪B88888"} <= plates_all, str(sorted(plates_all)))
    events_names = {e.get("name") for r in rows2 for e in (r.get("events") or [])
                    if isinstance(e, dict)}
    check("三层分工：车牌号不重复进 events[]（§2.5 原则 1）",
          not ({"沪A12345", "沪B88888"} & events_names), str(sorted(events_names)))
    ev0 = by_win[round(wins[0][0], 3)].get("events") or []
    dou = [e for e in ev0 if e.get("name") == "斗殴"]
    check("events[] 路由：警情「斗殴」落 ts 所属窗口行，kind=警情",
          len(dou) == 1 and dou[0].get("kind") == "警情", str(ev0)[:100])
    check("事件 ts 保留原值（不裁剪）", dou and dou[0].get("ts") == ts_w0,
          str(dou[0].get("ts") if dou else None))
    pl_rows = [p for r in rows2 for p in (r.get("plates") or [])
               if isinstance(p, dict) and p.get("plate_no") == "沪B88888"]
    check("name 载体车牌带 ts 属性进 plates[]（判断层带属性）",
          len(pl_rows) == 1 and float(pl_rows[0].get("ts") or -1) == ts_w1,
          str(pl_rows))

    # ---- 观察层 frame_dets（WO-0033 第四轮 §2.5 原则 3 / §11 定案 6，WO-0033.13）----
    print("[B++] 落表行 frame_dets 结构（观察层原始记录流契约）")
    fds = [r.get("frame_dets") for r in rows2]
    check("落表行 frame_dets 全为 list 且有帧（观察源接线生效）",
          all(isinstance(f, list) for f in fds) and any(fds),
          f"{sum(len(f or []) for f in fds)} 帧/{len(fds)} 行")
    check("frame_dets 逐帧 = {ts, recs}（ts 可数值、recs 非空）",
          all({"ts", "recs"} <= set(w) and isinstance(w.get("ts"), (int, float))
              and w.get("recs") for f in fds for w in (f or [])))
    all_recs = [rc for f in fds for w in (f or []) for rc in (w.get("recs") or [])]
    check("recs 带 kind 且取值域 ⊆ {det, plate, face}（多态混存载体；"
          "本链路观察源仅 YOLO → 实际全 det，三桶混存 pipe 侧已测）",
          bool(all_recs) and all(rc.get("kind") in ("det", "plate", "face")
                                 for rc in all_recs),
          str(sorted({str(rc.get("kind")) for rc in all_recs})))
    check("recs 主干无残留：model_path/embedding/旁路图键（image/img/face_img/"
          "plate_img）全剥（ndarray 不可 JSON 化）",
          all(not ({"model_path", "embedding", "image", "img", "face_img",
                    "plate_img"} & set(rc)) for rc in all_recs))
    check("recs 可 JSON 序列化（落表前提，逐行 dumps 通过）",
          all(json.dumps(f, ensure_ascii=False) is not None for f in fds))
    check("frame_dets 逐帧 ts 升序（按 ts 顺序保留）",
          all([w.get("ts") for w in (f or [])]
              == sorted(w.get("ts") for w in (f or [])) for f in fds))

    vrows2 = webdb.list_review_videos(VIDEO)
    check("重跑后 video_time 仍为空（不臆造采集时间）",
          len(vrows2) == 1 and vrows2[0]["video_time"] == "")

    # ---- pipe 侧车牌路由双保险（直调 label_align，不经 web 读归一）----
    print("[B+] pipe 侧 build_clip_tags 车牌路由双保险（WO-0033.2 接缝②）")
    from video_embed import label_align as la
    tags = la.build_clip_tags("x.mp4", (0.0, 5.0), {}, [
        {"name": "沪B88888", "kind": "车牌", "ts": 1.0},           # 仅 name 载体
        {"name": "沪C99999", "kind": "车牌", "plate_no": "沪C99999", "ts": 2.0},
        {"name": "斗殴", "kind": "警情", "ts": 3.0},
        {"name": "纠纷", "ts": 4.0},                                # 无 kind 泛化警情
    ])
    tp = {p["plate_no"] for p in tags["plates"]}
    check("name 载体（kind=车牌）也进 plates[]（_plate_item 末级兜底）",
          tp == {"沪B88888", "沪C99999"}, str(sorted(tp)))
    te = {e["name"] for e in tags["events"]}
    check("非车牌事件的 name 不被误判为车牌号（斗殴/纠纷只进 events[]）",
          te == {"斗殴", "纠纷"}, str(sorted(te)))
    return {"npz": npz, "wins": wins, "n_clips": n_clips, "td": td}


# ---------------- [C] relabel：只重对齐不动向量库 ----------------

def test_c_relabel(server, webdb, ctx: dict) -> None:
    print("[C] relabel（仅重对齐）：归档增量 → 标签行更新、向量库字节/mtime 不变")
    npz, wins, n_clips = ctx["npz"], ctx["wins"], ctx["n_clips"]
    sha0, mt0 = _file_sha(npz), os.stat(npz).st_mtime_ns
    # frame_dets 保留语义基线（§11 定案 6）：relabel 前逐行快照
    fd_before = {round(float(r["t_start"]), 3): r.get("frame_dets")
                 for r in webdb.query_review_index(video=VIDEO)}

    ev_path = server._events_path_for(VIDEO)
    ts_w2 = wins[-1][0] + 0.5
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "qa-run-b", "pipeline": "PIPE0001",
                            "at": "2026-10-29T11:00:00+08:00", "video": VIDEO,
                            "events": [{"name": "聚集", "kind": "警情", "ts": ts_w2}]},
                           ensure_ascii=False) + "\n")

    r = server.h_review_relabel({"video": VIDEO, "store_path": npz})
    check("relabel 成功：ok/rows/backend 齐备", r.get("ok") is True
          and r.get("rows") == n_clips and r.get("backend") == "json", str(r)[:140])
    check("relabel note 注明「向量库未改动」", "向量库未改动" in (r.get("note") or ""),
          str(r.get("note")))
    rows = webdb.query_review_index(video=VIDEO)
    check("relabel 后行数不变（重对齐非重建）", len(rows) == n_clips, str(len(rows)))
    by_win = {round(float(r_["t_start"]), 3): r_ for r_ in rows}
    gather = by_win.get(round(wins[-1][0], 3), {}).get("events") or []
    check("归档新增事件「聚集」落末窗行", any(e.get("name") == "聚集" for e in gather),
          str(gather)[:80])
    names_all = {e.get("name") for r_ in rows for e in (r_.get("events") or [])}
    plates_all = {p.get("plate_no") for r_ in rows
                  for p in (r_.get("plates") or [])}
    check("重对齐不丢旧标签（斗殴/两车牌仍在）",
          "斗殴" in names_all and {"沪A12345", "沪B88888"} <= plates_all,
          f"events={sorted(n for n in names_all if n)}")
    # 窗缝回归（2026-10-29 真机抓坑）：store meta 的 t_end=窗内末帧 ts，与索引对齐的
    # clip_windows 口径（t_end=下一窗首帧，无缝）不一致——事件 ts 恰在末帧/窗缝时会
    # 两边都不命中而丢。relabel 已改为从 frame_ts 重建无缝窗口，此处锁死该行为。
    ts_seam = wins[0][1]  # 首窗 store t_end（< 次窗 t_start，正在缝上）
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "qa-run-seam", "pipeline": "PIPE0001",
                            "at": "2026-10-29T11:05:00+08:00", "video": VIDEO,
                            "events": [{"name": "缝上事件", "kind": "警情", "ts": ts_seam}]},
                           ensure_ascii=False) + "\n")
    server.h_review_relabel({"video": VIDEO, "store_path": npz})
    rows2 = webdb.query_review_index(video=VIDEO)
    by_win2 = {round(float(r_["t_start"]), 3): r_ for r_ in rows2}
    seam_rows = [r_ for t0, r_ in by_win2.items()
                 if any(e.get("name") == "缝上事件" for e in (r_.get("events") or []))]
    check("窗缝事件不丢：ts=store t_end 归前窗（无缝窗口口径）",
          len(seam_rows) == 1 and round(float(seam_rows[0]["t_start"]), 3) == round(wins[0][0], 3),
          str([(r_["t_start"], r_["events"]) for r_ in seam_rows])[:140])
    check("观察层保留（relabel 复用旧行 det_stats，不因无观察段清零）",
          any(r_.get("det_stats") for r_ in rows))
    fd_after = {round(float(r_["t_start"]), 3): r_.get("frame_dets") for r_ in rows}
    check("frame_dets 保留：relabel 后逐行原值不动（观察层原始数据不随重对齐重算）",
          fd_before.keys() == fd_after.keys()
          and all(fd_before[k] == fd_after[k] for k in fd_before),
          f"{len(fd_after)} 行比对")
    sha1, mt1 = _file_sha(npz), os.stat(npz).st_mtime_ns
    check("向量库字节不变（不重嵌入，§3 定案 4）", sha1 == sha0)
    check("向量库 mtime 不变", mt1 == mt0, f"{mt0} → {mt1}")

    for name, payload in [
        ("video 必填", {"store_path": npz}),
        ("底库无该视频且未显式传 windows", {"video": "nope.mp4", "store_path": npz}),
    ]:
        try:
            server.h_review_relabel(payload)
            check(f"relabel 错误路径：{name} → ValueError", False, "未抛异常")
        except ValueError as e:
            check(f"relabel 错误路径：{name} → ValueError 可读",
                  bool(str(e)), str(e)[:80])
    # 显式 windows（无底库依赖形态）：只重对齐指定窗口
    r2 = server.h_review_relabel({"video": "wins_only.mp4",
                                  "events_path": str(ev_path),
                                  "windows": [[0.0, 5.0]]})
    check("relabel 显式 windows：无底库也能重对齐（windows 语义）",
          r2.get("ok") is True and r2.get("rows") == 1, str(r2)[:120])


# ---------------- [D] 查询过滤（h_review_search） ----------------

def test_d_filters(server, webdb, ctx: dict) -> None:
    print("[D] 查询过滤：时间窗 / event_name 两路 / video / mode=hybrid")
    npz, wins = ctx["npz"], ctx["wins"]
    base = {"query_text": "人群聚集", "store_path": npz}

    # D1 无采集时间：时间窗不静默丢弃，why 注明（§2.6）
    r = server.h_review_search({**base, "time_start": "2026-03-01T10:00:00+08:00",
                                "time_end": "2026-03-01T10:10:00+08:00"})
    check("无采集时间 + 时间窗 → 命中保留（不参与过滤不静默丢弃）",
          r["ok"] is True and len(r["hits"]) >= 1, f"{len(r['hits'])} 段")
    check("why 注明「无采集时间，未参与时间窗过滤」",
          all("无采集时间" in h["why"] for h in r["hits"]),
          r["hits"][0]["why"][:80] if r["hits"] else "")

    # D2 有采集时间：video_time + t_start 硬过滤
    tz = timezone(timedelta(hours=8))
    t0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=tz)
    old_vrow = webdb.list_review_videos(VIDEO)[0]
    webdb.upsert_review_video([{**old_vrow, "video_time": t0.isoformat(),
                                "video_time_src": "manual"}])
    t1 = (t0 + timedelta(seconds=wins[0][1] - 0.01)).isoformat()
    r = server.h_review_search({**base, "time_start": t0.isoformat(),
                                "time_end": t1})
    check("时间窗 [T, T+w0) 硬过滤：只命中窗口 0",
          len(r["hits"]) == 1 and r["hits"][0]["t_start"] == round(wins[0][0], 3),
          str([(h["t_start"], h["why"]) for h in r["hits"]])[:120])
    check("窗口内命中 why 注明「时间窗命中」",
          "时间窗命中" in r["hits"][0]["why"], r["hits"][0]["why"])
    r_far = server.h_review_search({**base, "time_start": (t0 + timedelta(hours=1))
                                    .isoformat(),
                                    "time_end": (t0 + timedelta(hours=2)).isoformat()})
    check("时间窗在视频时长之外 → 0 命中", r_far["ok"] is True
          and r_far["hits"] == [])
    try:
        server.h_review_search({**base, "time_start": t1, "time_end": t0.isoformat()})
        check("time_start 晚于 time_end → ValueError", False, "未抛异常")
    except ValueError:
        check("time_start 晚于 time_end → ValueError", True)

    # D3 event_name 精确匹配两路（§9 Q2）
    r_ev = server.h_review_search({**base, "event_name": "斗殴"})
    check("event_name=斗殴 → 命中 events.name 路",
          r_ev["ok"] is True and len(r_ev["hits"]) >= 1
          and all(any(e.get("name") == "斗殴"
                      for e in (h.get("tags") or {}).get("events") or [])
                  for h in r_ev["hits"]),
          f"{len(r_ev['hits'])} 段")
    check("events 路命中 why 文案「事件 斗殴 命中」",
          all("事件 斗殴 命中" in h["why"] for h in r_ev["hits"]),
          r_ev["hits"][0]["why"][:80])
    r_pl = server.h_review_search({**base, "event_name": "沪A12345"})
    check("event_name=沪A12345 → 命中 plates.plate_no 路",
          r_pl["ok"] is True and len(r_pl["hits"]) >= 1
          and all(any(p.get("plate_no") == "沪A12345"
                      for p in (h.get("tags") or {}).get("plates") or [])
                  for h in r_pl["hits"]),
          f"{len(r_pl['hits'])} 段")
    check("plates 路命中 why 文案「车牌 沪A12345 命中」",
          all("车牌 沪A12345 命中" in h["why"] for h in r_pl["hits"]),
          r_pl["hits"][0]["why"][:80])
    check("event_name 精确匹配（前缀「斗」不命中，非模糊）",
          server.h_review_search({**base, "event_name": "斗"})["hits"] == [])
    check("event_name 不存在 → 0 命中",
          server.h_review_search({**base, "event_name": "不存在的事件"})["hits"] == [])

    # D4 video 过滤
    r_v = server.h_review_search({**base, "video": VIDEO})
    check("video 过滤命中且 why 注明视频",
          r_v["ok"] is True and len(r_v["hits"]) >= 1
          and all(h["video"] == VIDEO for h in r_v["hits"])
          and all(f"视频 {VIDEO} 命中" in h["why"] for h in r_v["hits"]),
          f"{len(r_v['hits'])} 段")
    check("video 不存在 → 0 命中",
          server.h_review_search({**base, "video": "nope.mp4"})["hits"] == [])

    # D5 mode=hybrid：本期走过滤式（§5.1 / Q3 定案），note/why 说明
    r_h = server.h_review_search({**base, "mode": "hybrid"})
    check("mode=hybrid → query_mode 回显 hybrid",
          r_h["ok"] is True and r_h["query_mode"] == "hybrid")
    check("note 说明「hybrid 融合（RRF）本期未实现，已按过滤式执行」",
          "hybrid 融合（RRF）本期未实现" in (r_h.get("note") or "")
          and "过滤式" in (r_h.get("note") or ""), r_h.get("note", "")[:80])
    check("hybrid 仍出命中（过滤式执行，不空转）", len(r_h["hits"]) >= 1)
    check("why 注明「hybrid 融合未实现，按过滤式执行」",
          all("hybrid 融合未实现" in h["why"] for h in r_h["hits"]),
          r_h["hits"][0]["why"][:80])
    try:
        server.h_review_search({**base, "mode": "bad"})
        check("mode 非法 → ValueError", False, "未抛异常")
    except ValueError:
        check("mode 非法 → ValueError", True)

    # D6 signals 元信息 + 无过滤 WO-0032 文案向后兼容
    r_s = server.h_review_search({**base, "video": VIDEO, "mode": "hybrid"})
    sig = r_s["signals"]
    check("signals.filters 回显入参且 active=True",
          sig["active"] is True and sig["filters"]["video"] == VIDEO
          and sig["filters"]["mode"] == "hybrid", str(sig["filters"]))
    check("signals.meta_hits = 标签行数、vec_top_n ≥ 1、tag_backend=json",
          sig["meta_hits"] == ctx["n_clips"] and sig["vec_top_n"] >= 1
          and sig["tag_backend"] == "json",
          f"meta={sig['meta_hits']} vec={sig['vec_top_n']}")
    r0 = server.h_review_search(dict(base))
    check("无过滤查询保持 WO-0032 文案（why=向量相似度，向后兼容）",
          r0["signals"]["active"] is False
          and r0["hits"][0]["why"] == f"向量相似度 {r0['hits'][0]['score']:.2f}",
          r0["hits"][0]["why"])
    check("无过滤命中不带 tags 块（过滤产物随过滤启用）",
          all("tags" not in h for h in r0["hits"]))
    check("带过滤命中带 tags 块（观察+判断层随行返回）",
          all(set((h.get("tags") or {})) >= {"det_stats", "class_names", "plate_nos",
                                             "plates", "events", "face_count",
                                             "description"}
              for h in r_v["hits"]))

    # D7 include_frame_dets 开关（WO-0033.11）：frame_dets 体积大（观察层原始记录流，
    # KB 级/窗），缺省不随 hits[].tags 返回；=true 才附带（时间形态可视化/调试）
    check("缺省（开关关）tags 不含 frame_dets 键",
          all("frame_dets" not in (h.get("tags") or {}) for h in r_v["hits"]))
    r_fd = server.h_review_search({**base, "video": VIDEO,
                                   "include_frame_dets": True})
    fd_tags = [(h.get("tags") or {}).get("frame_dets") for h in r_fd["hits"]]
    check("include_frame_dets=true → hits[].tags 附带 frame_dets 且为 list",
          r_fd["ok"] is True and all(isinstance(f, list) for f in fd_tags),
          f"{len(r_fd['hits'])} 段")
    check("tags.frame_dets 内容与落表行一致（有帧，逐帧 {ts, recs}）",
          any(f for f in fd_tags)
          and all({"ts", "recs"} <= set(w) for f in fd_tags for w in (f or [])))
    check("signals.filters 回显 include_frame_dets=true",
          r_fd["signals"]["filters"]["include_frame_dets"] is True)
    check("显式 false 与缺省同口径（不带）",
          all("frame_dets" not in (h.get("tags") or {}) for h in
              server.h_review_search({**base, "video": VIDEO,
                                      "include_frame_dets": False})["hits"]))


# ---------------- [E] JSON 降级文件形态 ----------------

def test_e_json_shape(server, webdb, tmp: str) -> None:
    print("[E] 无 DB 降级：JSON 文件形态（§2.5/§2.6 字段同名同义）")
    check("后端模式 = json（强制降级生效）", webdb.review_backend_mode() == "json")
    p_idx = webdb._review_json_path("review_index")
    p_vid = webdb._review_json_path("review_video")
    check("review_index.json 落在 env 隔离路径", str(p_idx).startswith(tmp)
          and os.path.isfile(p_idx), p_idx)
    check("review_video.json 落在 env 隔离路径", str(p_vid).startswith(tmp)
          and os.path.isfile(p_vid), p_vid)
    idx = json.loads(Path(p_idx).read_text(encoding="utf-8"))
    vid = json.loads(Path(p_vid).read_text(encoding="utf-8"))
    check("review_index.json 为非空 list[dict]", isinstance(idx, list)
          and idx and all(isinstance(r, dict) for r in idx), f"{len(idx)} 行")
    check("review_video.json 为非空 list[dict]", isinstance(vid, list)
          and vid and all(isinstance(r, dict) for r in vid), f"{len(vid)} 行")
    need_idx = {"video", "t_start", "t_end", "det_stats", "frame_dets",
                "class_names", "plate_nos", "plates", "events", "face_count",
                "description", "thumb_path", "vector_model", "clip_dim", "frame_dim"}
    check("标签行字段 ⊇ §2.5 冻结字段全集（列名同名同义）",
          all(need_idx <= set(r) for r in idx),
          str(sorted(need_idx - set(idx[0]))))
    need_vid = {"video", "video_time", "video_time_src", "indexed_at",
                "vector_model", "clip_dim", "frame_dim", "duration"}
    check("视频行字段 ⊇ §2.6 字段全集", all(need_vid <= set(r) for r in vid),
          str(sorted(need_vid - set(vid[0]))))


# ---------------- [F] SSE 路径落表（WO-0033.9 扩展修复回归） ----------------

def test_f_sse_chain(server, webdb, tmp: str, proj_tmp: Path) -> None:
    print("[F] SSE 路径（h_pipeline_stream）落表：class_names 观察标签链路级可达")
    stages = [{"module_id": m, "params": dict(p)}
              for m, _n, p in server._SEED_VEMB_STAGES]
    npz = os.path.join(tmp, "video_vecs_sse.npz")
    vemb = stages[-1]
    vemb["params"].update({"store_path": npz, "thumb_dir": str(proj_tmp)})
    vemb["inputs"] = {"dets": "det_filter.dets"}   # 连线契约同 _spec（单一来源播种结构）
    spec = {"name": "视频向量化-嵌入查询链路", "stages": stages}

    events: list = []
    orig_load = server.db.load_pipeline
    try:
        server.db.load_pipeline = lambda pid: dict(spec)
        # 直调端点处理函数（server.py /api/test/pipeline_run/stream 即此直调），
        # write 回调收集 SSE 事件流；mock 档全程无网络
        server.h_pipeline_stream(
            lambda ev, payload: events.append((ev, payload)),
            {"pipeline_id": "PIPE0004", "video": VIDEO, "max_frames": 200})
    finally:
        server.db.load_pipeline = orig_load

    kinds = [ev for ev, _p in events]
    errs = [p for ev, p in events if ev == "error"]
    done = next((p for ev, p in reversed(events) if ev == "done"), None)
    check("SSE 运行成功：done 到达、无 error、有帧处理",
          not errs and done is not None and done.get("processed", 0) > 0,
          f"kinds={kinds[:3]}…{kinds[-2:]} processed={(done or {}).get('processed')}"
          + (f" err={str(errs[0])[:80]}" if errs else ""))
    check("SSE 事件序列 = meta/frame/done 且片段产出经 alarms 上报（SSE 无独立 submit 事件）",
          "meta" in kinds and "frame" in kinds
          and (done or {}).get("alarms", 0) > 0,
          f"types={sorted(set(kinds))} alarms={(done or {}).get('alarms')}")

    store = EmbedStore().load(npz)
    n_clips = len(store)
    metas = [store._clip_meta[i] for i in range(n_clips)]
    key_vec = {(str(m.get("video")), round(float(m.get("t_start") or 0), 3))
               for m in metas}
    rows = webdb.query_review_index(video=VIDEO)
    key_db = {(r["video"], round(float(r["t_start"]), 3)) for r in rows}
    check("SSE 落表：review_index 行数 = 片段数且 (video,t_start) 与底库一致",
          len(rows) == n_clips and key_db == key_vec,
          f"{len(rows)} vs {n_clips} 片段; 仅表 {len(key_db - key_vec)} / 仅库 {len(key_vec - key_db)}")
    merged_stats: dict = {}
    merged_names: dict = {}
    for r in rows:
        merged_stats.update(r.get("det_stats") or {})
        merged_names.update(r.get("class_names") or {})
    check("SSE 观察层 det_stats 非空且结构 = {cls_id:{peak,avg,tracks}}",
          merged_stats and all(
              isinstance(v, dict) and {"peak", "avg", "tracks"} <= set(v)
              for v in merged_stats.values()),
          f"{len(merged_stats)} 类")
    check("SSE class_names 非空（h_pipeline_stream 喂管道保留原始 dict，WO-0033.9 扩展修复）",
          bool(merged_names), json.dumps(merged_names, ensure_ascii=False)[:80])
    check("SSE class_names 键 = det_stats 键（cls_id 同域）",
          set(merged_names) == set(merged_stats),
          f"{sorted(merged_names)} vs {sorted(merged_stats)}")


# ---------------- main ----------------

def main() -> None:
    tmp = tempfile.mkdtemp(prefix="umvp_hybrid_qa_")
    proj_tmp = ROOT / "out" / f"qa_hybrid_thumbs_{os.getpid()}"  # 项目根下临时 thumb_dir（测后清理）
    env_backup = {k: os.environ.pop(k)
                  for k in WEMM_ENV_KEYS + REVIEW_ENV_KEYS if k in os.environ}
    os.environ["UMVP_REVIEW_INDEX_JSON"] = os.path.join(tmp, "review_index.json")
    os.environ["UMVP_REVIEW_VIDEO_JSON"] = os.path.join(tmp, "review_video.json")
    import db as webdb
    import server
    orig_events_dir = server._EVENTS_DIR
    orig_use_db = webdb._REVIEW_USE_DB
    server._EVENTS_DIR = Path(tmp) / "events"          # 归档写临时目录（不碰真实 out/）
    webdb.set_review_db_enabled(False)                 # 强制 JSON 降级后端
    webdb.configure_modules(                           # INB-0109：进程内跑链路先注册 schema
        {mid: {k: spec.get("type", "str") for k, spec in m["params"].items()
               if not spec.get("test_only")}
         for mid, m in server.MODULES.items() if not m.get("hidden")})
    server._REVIEW_STORE_CACHE.clear()
    try:
        archive_path = test_a_archive(server)
        ctx = test_b_full_chain(server, webdb, tmp, archive_path, proj_tmp)
        test_c_relabel(server, webdb, ctx)
        test_d_filters(server, webdb, ctx)
        test_e_json_shape(server, webdb, tmp)
        test_f_sse_chain(server, webdb, tmp, proj_tmp)   # 放最后：SSE 落表 upsert 覆盖不影响 [C]/[D]/[E] 前提
    finally:
        server._EVENTS_DIR = orig_events_dir
        webdb._REVIEW_USE_DB = orig_use_db
        os.environ.update(env_backup)
        server._REVIEW_STORE_CACHE.clear()
        shutil.rmtree(proj_tmp, ignore_errors=True)
    print(f"\n[JSON 降级 + mock 档] 全部 {PASS} 项断言通过 ✓")
    print("done.")


if __name__ == "__main__":
    main()
