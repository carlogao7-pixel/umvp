# -*- coding: utf-8 -*-
"""回顾检索·查询层（向量档）验收测试（WO-0032.1，web 板块）

验证对象：
  - web_lab/server.py：POST /api/review/search（h_review_search 进程内 + 真 HTTP 两层）、
    底库 mtime 缓存重载、thumb data URL、pre_roll 播放定位
  - web_lab/validate_chain.py：_semantic_dry_run 补 video_embed 分支——
    embed_dimension/frame_dimension ∈ WEMM_MRL_DIMS、use_real_embed=true 无端点
    校验期即报（对齐 det_filter/seq_filter 干跑先例；装配期 fail fast 保留双保险）

分层（mock 档为主，秒级；LIVE 段真端点在线才跑、不在线优雅 SKIP）:
  [A] handler 单元（进程内直调 h_review_search，不起服务）：
      造底库 = EmbedStore 写入「已知向量」（mock 档查询文本的确定性伪向量——
      查询即命中自身片段 score≈1.0，完全走生产路径）→ 两步漏斗（clip 粗排 +
      帧级精排 ts 对位）/ pre_roll 计算与下限 0 / top_k 与 video 过滤 / why 字段 /
      mock note（显式 false 与无端点自动落两态）/ mock 查询确定性 / mtime 变化重载 /
      空查询 / 底库不存在 / 空库 / 维度非档位 / use_real_embed=true 无端点 fail fast
  [B] validate_chain 校验期拦截（768 / 无端点），种子形态零回归
  [C] 端到端（进程内起 ThreadingHTTPServer(server.Handler) 真 HTTP 一轮）：
      PIPE0004 mock 索引 violent1.mp4（monkeypatch db.load_pipeline 避开 MySQL）→
      POST /api/review/search → 命中含 thumb data URL + 响应结构全键；
      空查询 → HTTP 400；底库不存在 → HTTP 404
  [LIVE] env 有端点且在线时 real 查询一轮（不在线优雅 SKIP，不算失败）

运行:
    conda run -n ai python tests/test_review_search.py           # mock 档全过
    source wemm接口/REMOTE_CLIENT.env && \
    conda run -n ai python tests/test_review_search.py --live    # + real 查询冒烟
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from video_embed.embed_store import EmbedStore  # noqa: E402
from video_embed.wemm_client import WEMMClient, endpoint_reachable, resolve_wemm_cfg  # noqa: E402

LIVE = "--live" in sys.argv
PASS = 0        # mock 档断言计数
PASS_LIVE = 0   # --live 断言计数
WEMM_ENV_KEYS = ("WEMM_BASE_URL", "WEMM_ENDPOINT", "WEMM_API_KEY", "WEMM_MODEL",
                 "WEMM_BASE_URL_FALLBACK")


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def check_live(name: str, cond: bool, detail: str = "") -> None:
    global PASS_LIVE
    assert cond, f"LIVE FAIL: {name} {detail}"
    PASS_LIVE += 1
    print(f"  [live-ok] {name}" + (f" | {detail}" if detail else ""))


# ---------------- 造库工具：mock 档「已知向量」底库 ----------------
# mock 伪向量确定性：embed_text(text, dim) 恒出同一向量。把查询文本的伪向量直接
# 写进底库（clip 层 + 首帧层），则 handler 生产路径 embed_text 同文本 → top1 该片段
# score≈1.0、帧精排 ts 对位——无需任何测试后门。

def _known_store(path: str, thumb_path: str) -> EmbedStore:
    """64/64 维（最小档位）三片段底库：
      #0 a.mp4 0~5s   clip=q("人群聚集")，帧 [q@ts2.5, q2@ts4.0]（查询 top1）
      #1 b.mp4 0~4s   clip=q("车辆违停")，帧 [q2@ts1.0]
      #2 a.mp4 10~14s clip=q("路边纠纷")，帧 [q3@ts10.4]（video 过滤反例基线）
    """
    cli = WEMMClient(use_real_embed=False, clip_dimension=64, frame_dimension=64)
    q = {t: cli.embed_text(t, dimension=64)
         for t in ("人群聚集", "车辆违停", "路边纠纷", "次帧")}
    st = EmbedStore(clip_dim=64, frame_dim=64)
    st.add_clip(q["人群聚集"], [q["人群聚集"], q["次帧"]], frame_ts=[2.5, 4.0],
                meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0,
                      "thumb_path": thumb_path})
    st.add_clip(q["车辆违停"], [q["次帧"]], frame_ts=[1.0],
                meta={"video": "b.mp4", "t_start": 0.0, "t_end": 4.0,
                      "thumb_path": thumb_path})
    st.add_clip(q["路边纠纷"], [q["路边纠纷"]], frame_ts=[10.4],
                meta={"video": "a.mp4", "t_start": 10.0, "t_end": 14.0,
                      "thumb_path": thumb_path})
    st.save(path)
    return st


def _tmp_thumb(td: str) -> str:
    """真缩略图落盘（thumb data URL 断言用）。"""
    p = os.path.join(td, "thumb_0.000.jpg")
    assert cv2.imwrite(p, np.zeros((48, 80, 3), dtype=np.uint8))
    return p


def _pipe0004_spec(store_path: str, thumb_dir: str) -> dict:
    """PIPE0004 播种形态 spec（store_path/thumb_dir 指向临时目录，不碰真实 out/）。"""
    return {"name": "视频向量化-嵌入查询链路", "stages": [
        {"module_id": "frame_manager", "params": {
            "sampling": "analysis", "frame_skip": 15, "interval_sec": 3.0,
            "interval_frames": 30, "backpressure": "standard"}},
        {"module_id": "video_embed", "params": {
            "embed_endpoint": "", "embed_model": "", "embed_dimension": 1024,
            "frame_dimension": 256, "use_real_embed": False, "embed_timeout": 120,
            "max_resolution": "896,896", "jpeg_quality": 85,
            "frames_per_clip": 6, "clip_seconds": 5.0,
            "thumb_dir": thumb_dir, "thumb_long": 320,
            "flush_partial": True, "store_path": store_path}},
    ]}


# ---------------- [A] handler 单元（进程内直调） ----------------

def test_handler_unit(tmpdir: str) -> None:
    import server
    print("[A] h_review_search 单元（两步漏斗 / pre_roll / 过滤 / note / 确定性 / 错误）")
    td = os.path.join(tmpdir, "unit")
    os.makedirs(td, exist_ok=True)
    thumb = _tmp_thumb(td)
    npz = os.path.join(td, "v.npz")
    _known_store(npz, thumb)
    base = {"query_text": "人群聚集", "store_path": npz,
            "top_k_clips": 10, "top_k_frames": 3, "pre_roll_sec": 1.0}

    r = server.h_review_search(dict(base))
    check("查询成功且 mode=mock", r["ok"] is True and r["mode"] == "mock")
    check("store 元信息回显（路径/clips/clip_dim）",
          r["store"]["clips"] == 3 and r["store"]["clip_dim"] == 64
          and r["store"]["path"] == npz, str(r["store"]))
    check("query_dim = 库 clip_dim", r["query_dim"] == 64)
    check("无端点自动落 mock 并在 note 说明",
          "mock" in r["note"] and "WEMM_BASE_URL" in r["note"], r["note"][:80])

    h0 = r["hits"][0]
    check("top1 命中自身片段（score≈1.0）",
          h0["video"] == "a.mp4" and abs(h0["t_start"]) < 1e-6
          and h0["score"] > 0.999, f"score={h0['score']}")
    check("why 口径「向量相似度 0.xx」",
          h0["why"].startswith("向量相似度")
          and h0["why"] == f"向量相似度 {h0['score']:.2f}", h0["why"])
    check("帧级精排 top1 ts 对位（2.5s）且次帧次之",
          len(h0["frames"]) == 2 and abs(h0["frames"][0]["ts"] - 2.5) < 1e-6
          and abs(h0["frames"][1]["ts"] - 4.0) < 1e-6
          and h0["frames"][0]["score"] > h0["frames"][1]["score"],
          str(h0["frames"]))
    check("best_frame_ts = 帧精排 top1", h0["best_frame_ts"] == h0["frames"][0]["ts"])
    check("play_ts = best_frame_ts − pre_roll",
          abs(h0["play_ts"] - 1.5) < 1e-6, f"play_ts={h0['play_ts']}")
    check("thumb 为可展示 data URL（磁盘路径浏览器取不到）",
          h0["thumb"].startswith("data:image/jpeg;base64,")
          and h0["thumb_path"].endswith(".jpg"))

    # pre_roll 下限 0（回退越过片头不取负）
    r2 = server.h_review_search({**base, "pre_roll_sec": 10.0})
    check("pre_roll 回退越过 0 → play_ts 下限 0", r2["hits"][0]["play_ts"] == 0.0)

    # top_k / video 过滤
    r3 = server.h_review_search({**base, "top_k_clips": 2})
    check("top_k_clips 截断生效", len(r3["hits"]) == 2)
    r4 = server.h_review_search({**base, "video": "b.mp4"})
    check("video 过滤只在指定视频内",
          len(r4["hits"]) >= 1 and all(h["video"] == "b.mp4" for h in r4["hits"]))
    check("video 过滤无命中返回空列表（可读，不报错）",
          server.h_review_search({**base, "video": "c.mp4"})["hits"] == [])

    # mock 查询确定性：同查询两次结果逐字段一致
    ra = server.h_review_search(dict(base))
    check("mock 查询确定性（同文本同命中同分数）",
          [h["score"] for h in ra["hits"]] == [h["score"] for h in r["hits"]]
          and [h["clip_id"] for h in ra["hits"]] == [h["clip_id"] for h in r["hits"]])

    # 显式 use_real_embed=false → mock note 换文案
    r5 = server.h_review_search({**base, "use_real_embed": False})
    check("显式 mock note 注明「显式指定」", "显式" in r5["note"], r5["note"])

    # mtime 变化重载：PIPE0004 重跑 upsert 后查询侧无需重启
    st = EmbedStore().load(npz)
    st.add_clip(np.ones(64, dtype=np.float32) / 8, [np.ones(64) / 8], frame_ts=[20.0],
                meta={"video": "c.mp4", "t_start": 20.0, "t_end": 24.0,
                      "thumb_path": thumb})
    st.save(npz)
    r6 = server.h_review_search(dict(base))
    check("底库 mtime 变化自动重载（条数 3→4）",
          r6["store"]["clips"] == 4, f"clips={r6['store']['clips']}")

    # 错误路径（fail fast 可读中文）
    for name, payload, exc, kws in [
        ("空查询报错", {**base, "query_text": "  "}, ValueError, "query_text"),
        ("底库不存在报错", {**base, "store_path": os.path.join(td, "no.npz")},
         FileNotFoundError, "底库不存在"),
        ("use_real_embed=true 无端点 fail fast",
         {**base, "use_real_embed": True}, ValueError, "WEMM_BASE_URL"),
    ]:
        try:
            server.h_review_search(payload)
            check(name, False, "未抛异常")
        except exc as e:
            check(name, all(k in str(e) for k in kws), str(e)[:90])

    # 空库（有文件、0 条目）
    empty = os.path.join(td, "empty.npz")
    EmbedStore(clip_dim=64, frame_dim=64).save(empty)
    try:
        server.h_review_search({**base, "store_path": empty})
        check("空库报错", False, "未抛异常")
    except ValueError as e:
        check("空库报错（指引建库）", "为空" in str(e), str(e)[:90])

    # 维度非档位（手工造 100 维库 → 查询侧 WEMMClient fail fast）
    odd = os.path.join(td, "odd.npz")
    odd_store = EmbedStore(clip_dim=100, frame_dim=50)
    odd_store.add_clip(np.ones(100, dtype=np.float32) / 10, meta={
        "video": "x.mp4", "t_start": 0.0, "thumb_path": thumb})
    odd_store.save(odd)
    try:
        server.h_review_search({**base, "store_path": odd})
        check("维度非档位报错", False, "未抛异常")
    except ValueError as e:
        check("维度非档位报错（列合法档位）",
              "Matryoshka" in str(e) and "1024" in str(e), str(e)[:90])


# ---------------- [B] validate_chain 校验期拦截 ----------------

def test_validate_guard() -> None:
    print("[B] validate_chain：video_embed 干跑分支（768 / 无端点校验期报错，零回归）")
    from validate_chain import validate
    import server
    stages = server._SEED_VEMB_STAGES
    fm = [{"module_id": m, "params": dict(p)} for m, _s, p in stages[:-1]]

    def spec_of(vemb_params: dict) -> dict:
        return {"stages": fm + [{"module_id": "video_embed", "params": vemb_params}]}

    seed_params = dict(stages[-1][2])
    check("种子形态 spec 仍全绿（零回归）", validate(spec_of(dict(seed_params))) == [])

    errs = validate(spec_of({**seed_params, "embed_dimension": 768}))
    check("768 非档位：校验期即报（不再拖到装配期）",
          any("embed_dimension" in e and "768" in e for e in errs), "; ".join(errs)[:120])
    check("报错列出合法档位（64~2048）",
          any("64" in e and "2048" in e for e in errs), "; ".join(errs)[:120])
    errs2 = validate(spec_of({**seed_params, "frame_dimension": 100}))
    check("frame_dimension=100 同样校验期报错",
          any("frame_dimension" in e and "100" in e for e in errs2))

    backup = {k: os.environ.pop(k) for k in WEMM_ENV_KEYS if k in os.environ}
    try:
        errs3 = validate(spec_of({**seed_params, "use_real_embed": True}))
        check("use_real_embed=true 且 spec/env 均无端点：校验期报错",
              any("use_real_embed" in e and "WEMM_BASE_URL" in e for e in errs3),
              "; ".join(errs3)[:120])
        ok_spec = spec_of({**seed_params, "use_real_embed": True,
                           "embed_endpoint": "http://10.0.0.8:9100/v1/embeddings"})
        check("spec 显式端点 → 校验通过（真实联调形态）", validate(ok_spec) == [])
        os.environ["WEMM_BASE_URL"] = "http://10.0.0.8:9100"
        check("env 有端点 → 校验通过（env 优先形态）",
              validate(spec_of({**seed_params, "use_real_embed": True})) == [])
    finally:
        # 先删后还原：backup 只含进入前的旧值，set 新增的 key 需显式清除（防泄漏到后续组）
        for k in WEMM_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(backup)


# ---------------- [C] 端到端（真 HTTP 一轮） ----------------

def test_http_e2e(tmpdir: str) -> None:
    import server
    print("[C] 端到端：PIPE0004 mock 索引 violent1.mp4 → HTTP /api/review/search")
    video = ROOT / "tests/data/vid/violent1.mp4"
    if not video.is_file():
        check("测试视频存在 tests/data/vid/violent1.mp4", False, "素材缺失")
        return
    td = os.path.join(tmpdir, "e2e")
    os.makedirs(td, exist_ok=True)
    npz = os.path.join(td, "video_vecs.npz")
    spec = _pipe0004_spec(npz, os.path.join(td, "thumbs"))

    orig_load = server.db.load_pipeline
    try:
        server.db.load_pipeline = lambda pid: dict(spec)
        run = server.h_pipeline_run({"pipeline_id": "PIPE0004", "video": "violent1.mp4",
                                     "max_frames": 200, "output_dir": td})
    finally:
        server.db.load_pipeline = orig_load
    check("PIPE0004 mock 索引成功", run.get("ok") is True, str(run.get("error"))[:120])
    check("索引落盘（.npz + .json）",
          os.path.isfile(npz) and os.path.isfile(npz[:-4] + ".json"))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def post(payload: dict):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/review/search",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        status, r = post({"query_text": "人群聚集", "store_path": npz})
        check("HTTP 200 且 ok", status == 200 and r.get("ok") is True,
              f"status={status} err={r.get('error')}")
        for k in ("mode", "query_dim", "store", "hits"):
            check(f"响应含 {k}", k in r)
        check("store 元信息 = 索引产物（clip 维 1024）",
              r["store"]["clip_dim"] == 1024 and r["query_dim"] == 1024
              and r["store"]["clips"] >= 1 and r["store"]["path"] == npz, str(r["store"]))
        check("命中片段（mock 档 query_dim 1024 与索引库一致）", len(r["hits"]) >= 1,
              f"{len(r['hits'])} 段")
        h = r["hits"][0]
        for k in ("video", "t_start", "t_end", "score", "clip_id", "frames",
                  "best_frame_ts", "play_ts", "thumb_path", "thumb", "why"):
            check(f"hits[0] 含 {k}", k in h)
        check("命中 video = 索引视频", h["video"] == "violent1.mp4")
        check("thumb data URL 随响应返回", h["thumb"].startswith("data:image/jpeg;base64,"))
        check("best_frame_ts 落在片段内且 play_ts = best − 1.0 ≥ 0",
              (h["t_start"] - 1e-6) <= (h["best_frame_ts"] or 0) <= (h["t_end"] + 1e-6)
              and abs(h["play_ts"] - max(0.0, (h["best_frame_ts"] or 0) - 1.0)) < 1e-6,
              f"best={h['best_frame_ts']} play={h['play_ts']}")
        check("why = 向量相似度口径", str(h["why"]).startswith("向量相似度"), h["why"])

        status2, r2 = post({"query_text": "", "store_path": npz})
        check("空查询 → HTTP 400 + 可读 error",
              status2 == 400 and r2.get("ok") is False and "query_text" in r2.get("error", ""),
              f"status={status2} {r2.get('error')}")
        status3, r3 = post({"query_text": "x", "store_path": os.path.join(td, "no.npz")})
        check("底库不存在 → HTTP 404 + 指引建库",
              status3 == 404 and "底库不存在" in r3.get("error", "")
              and "PIPE0004" in r3.get("error", ""), f"status={status3}")
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------- [LIVE] 真端点查询冒烟 ----------------

def test_live(npz: str) -> None:
    import server
    print("[LIVE] 真端点查询冒烟（在线才跑）")
    if not os.path.isfile(npz):
        print("  [SKIP] 索引库缺失（[C] 未产出）")
        return
    r = server.h_review_search({"query_text": "人群聚集", "store_path": npz,
                                "top_k_clips": 3})
    check_live("real 查询成功且 mode=real",
               r.get("ok") is True and r.get("mode") == "real", str(r.get("error"))[:120])
    check_live("real 查询 note 无 mock 提示", "mock" not in (r.get("note") or ""))
    check_live("real 查询返回命中与 score", len(r.get("hits") or []) >= 1
               and all(0.0 <= h["score"] <= 1.0 for h in r["hits"]))


# ---------------- main ----------------

def main() -> None:
    tmp = tempfile.mkdtemp(prefix="umvp_review_qa_")
    live_cfg = resolve_wemm_cfg()   # 屏蔽前读取（LIVE 段用）
    npz = os.path.join(tmp, "e2e", "video_vecs.npz")
    env_backup = {k: os.environ.pop(k) for k in WEMM_ENV_KEYS if k in os.environ}
    try:
        test_handler_unit(tmp)
        test_validate_guard()
        test_http_e2e(tmp)
    finally:
        os.environ.update(env_backup)
    print(f"\n[mock 档] 全部 {PASS} 项断言通过 ✓")
    if LIVE:
        if live_cfg[0] and endpoint_reachable(live_cfg[0], timeout=5.0):
            test_live(npz)
            print(f"[live] 真端点查询 {PASS_LIVE} 项断言通过 ✓")
        else:
            print("[live] 嵌入端点未配置或不在线，real 冒烟优雅跳过（不算失败）")
    print("done.")


if __name__ == "__main__":
    main()
