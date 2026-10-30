# -*- coding: utf-8 -*-
"""回顾检索·查询层（向量档）验收 e2e（WO-0032.2，qa 板块）

上游自检 tests/test_review_search.py（55+live）之外的验收视角：
  ① 两步漏斗**维度各对**专项——构造 clip_dim≠frame_dim 异构档位库（256/64），
    首版曾拿 clip 维查询向量比 frame 维库向量 matmul 直接维度不匹配（INB-0116），
    本组任何一步拿错维度都会当场异常 / score 对位失败，双保险钉死；
  ② 命中结构**全字段契约**：best_frame_ts∈[t_start,t_end]、play_ts=best−pre_roll
    下限 0、thumb data URL **base64 解码 + cv2 可解**（不止前缀）、why 口径、
    hits/frames 排序；
  ③ validate_chain **CLI 干跑反例**（exit code 维度）：768 报错含档位清单、
    real 无端点报错含 env 提示（exit 1）、种子 spec 全绿（exit 0）；
  ④ 端到端（进程内起 ThreadingHTTPServer 真 HTTP）：PIPE0004 mock 索引
    violent1.mp4 → 命中契约 + **合成命中 score≈1.0**（把查询文本喂 mock
    embed_text 拿向量走 EmbedStore.add_clip 生产 API 写库再查，无测试后门）；
  ⑤ index.html「回顾检索」页签存在性静态断言（元素 / API 调用 / 播放定位）+ script 语法；
  ⑥ 相对 store_path 归一项目根（INB-0114 CWD 坑；自造夹具库断言，
    WO-0033.13 起不依赖现场底库机器状态）。
  [LIVE] 真端点在线才跑 real 查询一轮，不在线优雅 SKIP（exit 0）。

分层:
  [A] handler 单元（进程内直调 h_review_search，不起服务，mock 档秒级）
  [B] validate_chain CLI 干跑反例（subprocess 真 exit code）
  [C] 端到端（ThreadingHTTPServer 真 HTTP + index.html 静态断言）
  [LIVE] real 查询冒烟（在线才跑）

运行:
    conda run -n ai python tests/test_review_search_e2e.py           # mock 档全过
    source wemm接口/REMOTE_CLIENT.env && \
    conda run -n ai python tests/test_review_search_e2e.py --live    # + real 查询冒烟
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
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
HIT_KEYS = ("video", "t_start", "t_end", "score", "clip_id", "frames",
            "best_frame_ts", "play_ts", "thumb_path", "thumb", "why")


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


def _tmp_thumb(td: str) -> str:
    p = os.path.join(td, "thumb_0.000.jpg")
    assert cv2.imwrite(p, np.full((48, 80, 3), 90, dtype=np.uint8))
    return p


def _thumb_decodable(data_url: str) -> bool:
    """data URL → base64 解码 → cv2 解码成功且非空（「可解码」的硬断言）。"""
    head, _, b64 = data_url.partition(",")
    return (head == "data:image/jpeg;base64" and bool(b64)
            and (lambda im: im is not None and im.size > 0)(
                cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                             cv2.IMREAD_COLOR)))


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

def _hetero_store(path: str, thumb: str) -> None:
    """① 专项：clip_dim=256 / frame_dim=64 异构档位底库（两片段）：
      #0 a.mp4 0~5s  clip=q("人群聚集"@256)，帧 [q("人群聚集"@64)@2.5, q("其他干扰"@64)@4.0]
      #1 b.mp4 0~4s  clip=q("路边纠纷"@256)，帧 [q("路边纠纷"@64)@1.0]
    mock 伪向量 = 全维向量前缀截断重归一——同文本**不同档位是不同向量**（高相关但
    不相等），score≈1.0 只在同维度查询时出现 → 粗排/精排任一步拿错维度即暴露。"""
    cli = WEMMClient(use_real_embed=False, clip_dimension=256, frame_dimension=64)
    q = {(t, d): cli.embed_text(t, dimension=d)
         for t, d in [("人群聚集", 256), ("人群聚集", 64), ("路边纠纷", 256),
                      ("路边纠纷", 64), ("其他干扰", 64)]}
    st = EmbedStore(clip_dim=256, frame_dim=64)
    st.add_clip(q["人群聚集", 256], [q["人群聚集", 64], q["其他干扰", 64]],
                frame_ts=[2.5, 4.0],
                meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0,
                      "thumb_path": thumb})
    st.add_clip(q["路边纠纷", 256], [q["路边纠纷", 64]], frame_ts=[1.0],
                meta={"video": "b.mp4", "t_start": 0.0, "t_end": 4.0,
                      "thumb_path": thumb})
    st.save(path)


def test_a_contract(tmpdir: str) -> None:
    import server
    print("[A] API 契约（两步漏斗维度各对 / 全字段 / top_k / 过滤 / 错误 / mtime / 容错）")
    td = os.path.join(tmpdir, "unit")
    os.makedirs(td, exist_ok=True)
    thumb = _tmp_thumb(td)

    # ① 异构档位库：两步漏斗各用各维度（clip@256 粗排、frame@64 精排）
    npz = os.path.join(td, "hetero.npz")
    _hetero_store(npz, thumb)
    base = {"query_text": "人群聚集", "store_path": npz}
    r = server.h_review_search(dict(base))
    check("异构库(256/64)查询不异常且 mode=mock（首版 bug：拿错维度 matmul 直接炸）",
          r["ok"] is True and r["mode"] == "mock")
    check("query_dim = 库 clip_dim(256)", r["query_dim"] == 256, str(r["query_dim"]))
    h0 = r["hits"][0]
    check("粗排用 clip 维查询向量：top1=a.mp4 片段 score≈1.0",
          h0["video"] == "a.mp4" and h0["score"] > 0.999, f"score={h0['score']}")
    check("精排用 frame 维查询向量：top1 帧 ts=2.5（同文本@64 自身）score≈1.0",
          abs(h0["frames"][0]["ts"] - 2.5) < 1e-6 and h0["frames"][0]["score"] > 0.999,
          str(h0["frames"][0]))
    check("次帧（其他干扰@64）ts=4.0 且 score 严格更低（精排可区分非噪声）",
          abs(h0["frames"][1]["ts"] - 4.0) < 1e-6
          and h0["frames"][1]["score"] < h0["frames"][0]["score"])
    r_b = server.h_review_search({"query_text": "路边纠纷", "store_path": npz})
    check("换查询文本 top1 切换到 b.mp4 片段（粗排非恒定）",
          r_b["hits"][0]["video"] == "b.mp4" and r_b["hits"][0]["score"] > 0.999)
    check("b.mp4 帧精排 top1 ts=1.0 对位",
          abs(r_b["hits"][0]["frames"][0]["ts"] - 1.0) < 1e-6)

    # ② 命中结构全字段契约
    check("hit 全键齐备", all(k in h0 for k in HIT_KEYS),
          str([k for k in HIT_KEYS if k not in h0]))
    check("best_frame_ts ∈ [t_start, t_end]",
          (h0["t_start"] - 1e-6) <= (h0["best_frame_ts"] or -1) <= (h0["t_end"] + 1e-6),
          f"best={h0['best_frame_ts']} win=[{h0['t_start']},{h0['t_end']}]")
    check("play_ts = best_frame_ts − pre_roll(1.0)", abs(h0["play_ts"] - 1.5) < 1e-6,
          f"play={h0['play_ts']}")
    check("play_ts ≥ 0（下限）", h0["play_ts"] >= 0.0)
    check("thumb data URL 真可解码（base64→cv2）", _thumb_decodable(h0["thumb"]))
    check("why 口径 = f『向量相似度 {score:.2f}』",
          h0["why"] == f"向量相似度 {h0['score']:.2f}", h0["why"])
    check("hits 按 score 降序",
          all(r["hits"][i]["score"] >= r["hits"][i + 1]["score"]
              for i in range(len(r["hits"]) - 1)))
    check("frames 按 score 降序",
          all(h0["frames"][i]["score"] >= h0["frames"][i + 1]["score"]
              for i in range(len(h0["frames"]) - 1)))
    rp = server.h_review_search({**base, "pre_roll_sec": 99.0})
    check("pre_roll 回退越过片头 → play_ts 下限 0", rp["hits"][0]["play_ts"] == 0.0)

    # ③ top_k 生效
    rk = server.h_review_search({**base, "top_k_clips": 1, "top_k_frames": 1})
    check("top_k_clips=1 截断", len(rk["hits"]) == 1)
    check("top_k_frames=1 截断", len(rk["hits"][0]["frames"]) == 1)

    # ④ video 过滤
    rv = server.h_review_search({**base, "video": "b.mp4"})
    check("video 过滤后全部命中限定视频",
          rv["ok"] is True and len(rv["hits"]) >= 1
          and all(h["video"] == "b.mp4" for h in rv["hits"]))
    check("video 过滤无命中 → 空 hits 不报错",
          server.h_review_search({**base, "video": "c.mp4"})["hits"] == [])

    # ⑤ 错误路径（fail fast 可读）
    for name, payload, exc, kws in [
        ("空查询 → 400 语义(ValueError)含 query_text",
         {**base, "query_text": " \t "}, ValueError, ["query_text"]),
        ("库不存在 → 404 语义(FileNotFoundError)含指引",
         {**base, "store_path": os.path.join(td, "no.npz")},
         FileNotFoundError, ["底库不存在", "PIPE0004"]),
        ("use_real_embed=true 无端点 → 400 语义含 env 提示",
         {**base, "use_real_embed": True}, ValueError, ["WEMM_BASE_URL"]),
        ("top_k_frames=-1 → 400 语义", {**base, "top_k_frames": -1}, ValueError, ["top_k"]),
        ("top_k_clips=-1 → 400 语义", {**base, "top_k_clips": -1}, ValueError, ["top_k"]),
        ("pre_roll_sec<0 → 400 语义", {**base, "pre_roll_sec": -0.5},
         ValueError, ["pre_roll_sec"]),
    ]:
        try:
            server.h_review_search(payload)
            check(name, False, "未抛异常")
        except exc as e:
            check(name, all(k in str(e) for k in kws), str(e)[:90])

    # 宽松容错（web 实现有意为之，or 默认）：显式 0 与缺省不可区分 → 收敛默认不报错
    # （前端 parseInt||10 同款；负数才有 400。验收记录：不算缺陷，回报注明口径）
    r0 = server.h_review_search({**base, "top_k_clips": 0, "top_k_frames": 0})
    check("top_k 显式 0 → 收敛默认（不报错，宽松容错口径）",
          r0["ok"] is True and len(r0["hits"]) >= 1)

    empty = os.path.join(td, "empty.npz")
    EmbedStore(clip_dim=256, frame_dim=64).save(empty)
    try:
        server.h_review_search({**base, "store_path": empty})
        check("空库 → 400 语义", False, "未抛异常")
    except ValueError as e:
        check("空库 → 400 语义（可读指引建库）", "为空" in str(e), str(e)[:90])

    # ⑥ mtime 变化重载（handler 不重启）
    st = EmbedStore().load(npz)
    cli = WEMMClient(use_real_embed=False, clip_dimension=256, frame_dimension=64)
    st.add_clip(cli.embed_text("新增片段", 256),
                [cli.embed_text("新增片段", 64)], frame_ts=[30.0],
                meta={"video": "c.mp4", "t_start": 30.0, "t_end": 34.0,
                      "thumb_path": thumb})
    st.save(npz)
    rm = server.h_review_search(dict(base))
    check("底库 mtime 变化自动重载（条数 2→3，查询侧免重启）",
          rm["store"]["clips"] == 3, f"clips={rm['store']['clips']}")

    # ⑦ thumb 容错：无 thumb_path / 路径不存在 → thumb 空串不崩
    st2 = EmbedStore().load(npz)
    st2.add_clip(cli.embed_text("无图片段", 256),
                 [cli.embed_text("无图片段", 64)], frame_ts=[40.0],
                 meta={"video": "d.mp4", "t_start": 40.0, "t_end": 44.0})  # 无 thumb_path
    st2.add_clip(cli.embed_text("坏图路径段", 256),
                 [cli.embed_text("坏图路径段", 64)], frame_ts=[50.0],
                 meta={"video": "e.mp4", "t_start": 50.0, "t_end": 54.0,
                       "thumb_path": "/nonexistent_dir/x.jpg"})  # 路径不存在
    st2.save(npz)
    st_ns = os.stat(npz).st_mtime_ns + 1_000_000
    os.utime(npz, ns=(st_ns, st_ns))
    rt = server.h_review_search(dict(base))
    d_hit = next(h for h in rt["hits"] if h["video"] == "d.mp4")
    e_hit = next(h for h in rt["hits"] if h["video"] == "e.mp4")
    check("meta 无 thumb_path → hit 正常返回且 thumb 空串",
          d_hit["thumb"] == "" and d_hit["best_frame_ts"] is not None)
    check("thumb_path 文件不存在 → thumb 空串不崩、hit 照常",
          e_hit["thumb"] == "" and e_hit["best_frame_ts"] is not None)
    check("正常片段 thumb 不受容错影响",
          any(_thumb_decodable(h["thumb"]) for h in rt["hits"]))


def test_a_live_store() -> None:
    """相对 store_path 归一项目根（INB-0114 CWD 坑）。

    机器状态解耦（WO-0033.13，替代旧「复用 web 留的现场底库 clips=3」断言——
    现场库条数随索引进度变化，硬编码必挂）：夹具库自造在项目根 out/ 下（相对路径
    解析目标=项目根，归一语义原样覆盖），条数/视频名确定可断言，测后清理。
    """
    import server
    print("[A+] 相对 store_path 归一项目根（自造夹具库，不依赖现场底库状态）")
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    npz_abs = out_dir / f"qa_review_e2e_rel_{os.getpid()}.npz"
    npz_json = npz_abs.with_suffix(".json")
    try:
        cli = WEMMClient(use_real_embed=False, clip_dimension=1024, frame_dimension=256)
        st = EmbedStore(clip_dim=1024, frame_dim=256)
        st.add_clip(cli.embed_text("人群聚集", 1024),
                    [cli.embed_text("人群聚集", 256)], frame_ts=[2.5],
                    meta={"video": "qa_rel.mp4", "t_start": 0.0, "t_end": 5.0})
        st.save(str(npz_abs))
        r = server.h_review_search({"query_text": "人群聚集",
                                    "store_path": f"out/{npz_abs.name}"})
        check("相对 store_path 以项目根解析（不落 web_lab/out CWD 坑）",
              r["store"]["path"] == str(npz_abs), r["store"]["path"])
        check("夹具库 clips=1（自造库条数确定）", r["store"]["clips"] == 1,
              f"clips={r['store']['clips']}")
        check("夹具库查询 hits 非空", len(r["hits"]) >= 1)
        check("score 最高项 video 匹配夹具库视频",
              r["hits"][0]["video"] == "qa_rel.mp4", r["hits"][0]["video"])
    finally:
        npz_abs.unlink(missing_ok=True)
        npz_json.unlink(missing_ok=True)


# ---------------- [B] validate_chain CLI 干跑反例（真 exit code） ----------------

def _seed_spec_params() -> list:
    import server
    stages = server._SEED_VEMB_STAGES
    fm = [{"module_id": m, "params": dict(p)} for m, _s, p in stages[:-1]]
    return fm, dict(stages[-1][2])


def test_b_validate_cli(tmpdir: str) -> None:
    print("[B] validate_chain CLI 干跑反例（exit code / 档位清单 / env 提示）")
    fm, seed_params = _seed_spec_params()
    vpy = str(ROOT / "web_lab" / "validate_chain.py")
    clean_env = {k: v for k, v in os.environ.items() if k not in WEMM_ENV_KEYS}

    def run_cli(vemb_over: dict, extra_env: dict | None = None) -> subprocess.CompletedProcess:
        spec = {"stages": fm + [{"module_id": "video_embed",
                                 "params": {**seed_params, **vemb_over}}]}
        sp = os.path.join(tmpdir, "spec.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False)
        env = dict(clean_env)
        env.update(extra_env or {})
        return subprocess.run([sys.executable, vpy, sp], capture_output=True,
                              text=True, env=env, timeout=120)

    p = run_cli({})
    check("种子形态 spec → exit 0 全绿（零回归）", p.returncode == 0,
          p.stdout[-160:] + p.stderr[-120:])

    p768 = run_cli({"embed_dimension": 768})
    check("embed_dimension=768 → exit 1（校验期拦截）", p768.returncode == 1)
    check("报错含具体值 768 与字段名", "768" in p768.stdout and "embed_dimension" in p768.stdout,
          p768.stdout.strip().splitlines()[-1][:110] if p768.stdout.strip() else "")
    check("报错列出完整合法档位清单（64~2048）",
          all(d in p768.stdout for d in ("64", "128", "256", "512", "1024", "2048")))

    p100 = run_cli({"frame_dimension": 100})
    check("frame_dimension=100 → exit 1 且含值", p100.returncode == 1 and "100" in p100.stdout)

    preal = run_cli({"use_real_embed": True})
    check("use_real_embed=true 且 spec/env 均无端点 → exit 1", preal.returncode == 1)
    check("报错含 env 配置提示（WEMM_BASE_URL）", "WEMM_BASE_URL" in preal.stdout,
          preal.stdout.strip().splitlines()[-1][:110] if preal.stdout.strip() else "")

    pok = run_cli({"use_real_embed": True,
                   "embed_endpoint": "http://10.0.0.8:9100/v1/embeddings"})
    check("spec 显式端点 → exit 0（真实联调形态放行）", pok.returncode == 0,
          pok.stdout[-120:])
    penv = run_cli({"use_real_embed": True},
                   extra_env={"WEMM_BASE_URL": "http://10.0.0.8:9100"})
    check("env 有端点 → exit 0（env 优先形态放行）", penv.returncode == 0,
          penv.stdout[-120:])


# ---------------- [C] 端到端（起服务真 HTTP + 前端静态断言） ----------------

def test_c_http_e2e(tmpdir: str) -> None:
    import server
    print("[C] 端到端：PIPE0004 mock 索引 → 真 HTTP 查询 + 合成命中 score≈1.0 + 前端静态")
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
    check("索引产物落盘（.npz + .json）",
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
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.status, resp.headers.get("Content-Type", ""), json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get("Content-Type", ""), json.loads(e.read())

        status, ctype, r = post({"query_text": "人群聚集", "store_path": npz})
        check("HTTP 200", status == 200, f"status={status}")
        check("Content-Type = application/json", ctype.startswith("application/json"), ctype)
        check("响应顶层全键（ok/mode/note/query_dim/query_text/store/hits/elapsed_sec）",
              all(k in r for k in ("ok", "mode", "note", "query_dim", "query_text",
                                   "store", "hits", "elapsed_sec")),
              str([k for k in r if k not in
                   ("ok", "mode", "note", "query_dim", "query_text", "store", "hits",
                    "elapsed_sec")]))
        check("store 元信息 = 索引产物（clip 维 1024）",
              r["store"]["clip_dim"] == 1024 and r["query_dim"] == 1024
              and r["store"]["clips"] >= 1 and r["store"]["path"] == npz, str(r["store"]))
        check("note 含 mock 说明（端点屏蔽自动落 mock 档）",
              r["mode"] == "mock" and "mock" in (r.get("note") or ""), r.get("note", "")[:60])
        check("命中片段 hits 非空", len(r["hits"]) >= 1, f"{len(r['hits'])} 段")
        check("score 最高项 video = 索引视频 violent1.mp4",
              r["hits"][0]["video"] == "violent1.mp4", r["hits"][0]["video"])
        h = r["hits"][0]
        check("hit 全键齐备", all(k in h for k in HIT_KEYS),
              str([k for k in HIT_KEYS if k not in h]))
        check("best_frame_ts ∈ [t_start, t_end] 且 play_ts = best − 1.0 ≥ 0",
              (h["t_start"] - 1e-6) <= (h["best_frame_ts"] or -1) <= (h["t_end"] + 1e-6)
              and abs(h["play_ts"] - max(0.0, (h["best_frame_ts"] or 0) - 1.0)) < 1e-6,
              f"best={h['best_frame_ts']} play={h['play_ts']}")
        check("thumb data URL 真可解码", _thumb_decodable(h["thumb"]))
        check("why 口径 = 向量相似度", h["why"] == f"向量相似度 {h['score']:.2f}", h["why"])

        # 合成命中：查询文本喂 mock embed_text 拿向量 → EmbedStore.add_clip 生产 API
        # 追加进索引产物再查（(video, t_start) 幂等 upsert 键避开索引片段的 0/5/10s
        # 窗口，用 100.0 明确「新增条目」而非覆盖）——score≈1.0 精确对位
        cli = WEMMClient(use_real_embed=False, clip_dimension=1024, frame_dimension=256)
        st = EmbedStore().load(npz)
        thumbs = sorted((Path(td) / "thumbs").glob("*.jpg"))
        thumb_path = str(thumbs[0]) if thumbs else _tmp_thumb(td)
        prev_clips = r["store"]["clips"]
        st.add_clip(cli.embed_text("人群聚集", 1024),
                    [cli.embed_text("人群聚集", 256), cli.embed_text("其他画面", 256)],
                    frame_ts=[3.0, 5.5],
                    meta={"video": "violent1.mp4", "t_start": 100.0, "t_end": 104.0,
                          "thumb_path": thumb_path})
        st.save(npz)
        status2, _ct2, r2 = post({"query_text": "人群聚集", "store_path": npz})
        check("add_clip 追加后同端口查询免重启即生效（HTTP 层 mtime 重载）",
              status2 == 200 and r2["ok"] is True
              and r2["store"]["clips"] == prev_clips + 1,
              f"clips={r2.get('store', {}).get('clips')}（原 {prev_clips}）")
        h2 = r2["hits"][0]
        check("合成命中 top1 score≈1.0（查询文本伪向量写库→同文本查询自命中）",
              h2["score"] > 0.999, f"score={h2['score']}")
        check("合成片段 video 匹配", h2["video"] == "violent1.mp4")
        check("合成片段帧精排 top1 ts=3.0 对位（frame@256 维查询向量）",
              abs(h2["frames"][0]["ts"] - 3.0) < 1e-6
              and h2["frames"][0]["score"] > 0.999, str(h2["frames"][0]))
        check("原索引片段仍在命中列表（粗排未被合成片段挤掉）",
              len(r2["hits"]) >= prev_clips + 1, f"{len(r2['hits'])} 段")

        s400, ct400, r400 = post({"query_text": "", "store_path": npz})
        check("空查询 → HTTP 400 + json + 可读 error",
              s400 == 400 and ct400.startswith("application/json")
              and r400.get("ok") is False and "query_text" in r400.get("error", ""),
              f"status={s400} ct={ct400}")
        s404, ct404, r404 = post({"query_text": "x",
                                  "store_path": os.path.join(td, "no.npz")})
        check("库不存在 → HTTP 404 + 指引建库",
              s404 == 404 and ct404.startswith("application/json")
              and "底库不存在" in r404.get("error", "")
              and "PIPE0004" in r404.get("error", ""), f"status={s404}")
    finally:
        httpd.shutdown()
        httpd.server_close()

    # index.html 页签存在性静态断言（HTML 元素 / API 调用 / 播放定位 / 语法）
    print("[C+] index.html「回顾检索」页签静态断言")
    html_path = ROOT / "web_lab" / "index.html"
    check("index.html 存在", html_path.is_file())
    if html_path.is_file():
        html = html_path.read_text(encoding="utf-8")
        check("页签入口：nav-review 按钮 + 「回顾检索」文案",
              'id="nav-review"' in html and "回顾检索" in html)
        check("视图区块：view-review + 查询框 rv-query + 查询按钮",
              'id="view-review"' in html and 'id="rv-query"' in html
              and 'id="rv-search-btn"' in html)
        check("参数控件：top_k×2 / pre_roll / video 过滤 / store_path",
              all(f'id="rv-{k}"' in html for k in
                  ("topk-clips", "topk-frames", "preroll", "video", "store")))
        check("调用 POST /api/review/search", 'fetch("/api/review/search"' in html)
        check("点击卡片内嵌播放器定位：currentTime = max(0, play_ts)",
              "Math.max(0, h.play_ts" in html and 'id="rv-player"' in html)
        check("复用 /api/video/prepare 转码基础设施", "/api/video/prepare" in html)
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        check("script 块可提取（≥1）", len(scripts) >= 1, f"{len(scripts)} 块")
        if scripts and shutil.which("node"):
            js = os.path.join(tmpdir, "index_inline.js")
            with open(js, "w", encoding="utf-8") as f:
                f.write("\n;\n".join(scripts))
            pn = subprocess.run(["node", "--check", js], capture_output=True, text=True)
            check("script 语法 node --check 通过", pn.returncode == 0,
                  (pn.stderr or "")[-160:])
        elif not shutil.which("node"):
            print("  [SKIP] node 不可用，script 语法检查跳过")


# ---------------- [LIVE] 真端点查询冒烟 ----------------

def test_live(npz: str) -> None:
    import server
    print("[LIVE] 真端点 real 查询一轮（在线才跑）")
    if not os.path.isfile(npz):
        print("  [SKIP] 索引库缺失（[C] 未产出）")
        return
    r = server.h_review_search({"query_text": "人群聚集", "store_path": npz,
                                "top_k_clips": 3})
    check_live("real 查询 ok 且 mode=real", r.get("ok") is True and r.get("mode") == "real",
               str(r.get("error"))[:120])
    check_live("note 无 mock 提示", "mock" not in (r.get("note") or ""))
    check_live("hits 非空且 score ∈ [0,1]",
               len(r.get("hits") or []) >= 1
               and all(0.0 <= h["score"] <= 1.0 for h in r["hits"]))
    h = r["hits"][0]
    check_live("real 命中结构与 mock 同契约（best/play/why/thumb 键）",
               all(k in h for k in HIT_KEYS)
               and (h["best_frame_ts"] is None or h["play_ts"] >= 0.0))


# ---------------- main ----------------

def main() -> None:
    tmp = tempfile.mkdtemp(prefix="umvp_review_e2e_")
    live_cfg = resolve_wemm_cfg()   # 屏蔽前读取（LIVE 段用）
    npz = os.path.join(tmp, "e2e", "video_vecs.npz")
    env_backup = {k: os.environ.pop(k) for k in WEMM_ENV_KEYS if k in os.environ}
    # WO-0033 起链路收尾会自动对齐落表（_archive_and_align），隔离到临时文件，
    # 避免污染真实 out/review_*.json（无 MySQL 时落 JSON 默认路径）
    env_backup.update({k: os.environ.pop(k) for k in
                       ("UMVP_REVIEW_INDEX_JSON", "UMVP_REVIEW_VIDEO_JSON")
                       if k in os.environ})
    os.environ["UMVP_REVIEW_INDEX_JSON"] = os.path.join(tmp, "review_index.json")
    os.environ["UMVP_REVIEW_VIDEO_JSON"] = os.path.join(tmp, "review_video.json")
    try:
        test_a_contract(tmp)
        test_a_live_store()
        test_b_validate_cli(tmp)
        test_c_http_e2e(tmp)
    finally:
        os.environ.update(env_backup)
    print(f"\n[mock 档] 全部 {PASS} 项断言通过 ✓")
    if LIVE:
        if live_cfg[0] and endpoint_reachable(live_cfg[0], timeout=5.0):
            test_live(npz)
            print(f"[live] 真端点 real 查询 {PASS_LIVE} 项断言通过 ✓")
        else:
            print("[live] 嵌入端点未配置或不在线，real 冒烟优雅跳过（不算失败，exit 0）")
    print("done.")


if __name__ == "__main__":
    main()
