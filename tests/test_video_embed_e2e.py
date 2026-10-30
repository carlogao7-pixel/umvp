# -*- coding: utf-8 -*-
"""WeMM 嵌入基础设施 + PIPE0004 索引链路验收测试（WO-0031.3，qa 板块）

验收对象（主单 WO-0031 验证条件 ①②③④ 的断言落地）：
  - pipe-chain 交付：umvp/video_embed/ 三件套（WEMMClient / EmbedStore / VideoEmbedStage）
  - web 交付：server.py 注册与驱动层接线、db.py VEMB 前缀、PIPE0004 幂等播种、
    validate_chain 终段扩展

分层（mock 档为主，秒级；--live 段真端点、不在线优雅 SKIP）:
  [A] 客户端契约（本地 ThreadingHTTPServer 假 WeMM）：请求体形态（messages 只一组 /
      dimensions 必带 / Bearer 头 / 图片 data URL 与 max_resolution 缩放）/ 文本 input
      形态 / 5xx 退避重试、4xx 不重试 / 不可达快速失败含端点地址 / key 零泄漏（体、
      异常、repr）/ 主端点挂自动切备用 / mock 伪向量确定性
  [B] EmbedStore 底库：写入检索 top_k / 帧级精排 / 删除视频连带（id 重映射）/
      upsert 幂等 / 维度不匹配拒收 / 持久化往返
  [C] 注册与校验契约：validate_chain PIPE0004 种子 spec 通过 / 非终段拒 /
      装配期维度档位 fail fast（768 报错含档位）/ 装配期 use_real_embed=true 无端点
      fail fast / 装配惰性零副作用（store_path 指临时不存在路径）+ 装配加载正向
      （存在库 load / 维度不匹配 ValueError，WO-0033.12）/ PARAMS 15 键 /
      VEMB 前缀 / 终段白名单 / 端口声明登记
  [D] 链路级（进程内 h_pipeline_run，monkeypatch db.load_pipeline 避开 MySQL/HTTP）：
      PIPE0004 mock 档跑真实测试视频 → 片段入库条目 + clip_indexed 事件进 timeline →
      store/thumbs 落盘 → EmbedStore.load 后检索命中 → 重跑 upsert 幂等 →
      无 output 段时结果不带 output 块（WO-0029 零回归）
   [E] 播种幂等（进程内 monkeypatch db 三函数）：_seed_chain 首播建 4 preset + 1 链路、
       复播同结构零写入；_SEED_VEMB_STAGES 结构 = frame_manager→yolo→det_filter→
       video_embed 终段（WO-0033.5 升级，观察段 hybrid_retrieval §12-2）+ 存量 2 阶段
       旧库自动重建升级（复用 FM/VEMB 旧 preset、只补建缺失阶段）
  [LIVE] 真端点冒烟（source wemm接口/REMOTE_CLIENT.env 后 --live）：探活 ≤5s 不在线
      打印 SKIP 原因不算失败；在线断言文本/单图/clip 多图维度与 L2

演进记录（WO-0032.1 已落地校验期前移）：validate_chain._semantic_dry_run 现有
video_embed 分支——维度档位 / use_real_embed 无端点在**校验期即报**（exit 1），
装配期 WEMMClient fail fast 保留为双保险；本测试 [C] 段装配期断言仍有效。

运行:
    conda run -n ai python tests/test_video_embed_e2e.py           # mock 档全过
    source wemm接口/REMOTE_CLIENT.env && \
    conda run -n ai python tests/test_video_embed_e2e.py --live    # + 真端点冒烟
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from video_embed.embed_store import EmbedStore  # noqa: E402
from video_embed.wemm_client import (  # noqa: E402
    WEMM_MRL_DIMS, WemmHttpError, WEMMClient, endpoint_reachable, resolve_wemm_cfg,
)

LIVE = "--live" in sys.argv
TEST_KEY = "sk-qa-e2e-key-000"  # 仅本地假服务用（非真实凭据）
PASS = 0        # mock 档断言计数
PASS_LIVE = 0   # --live 断言计数


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


def img(w: int = 64, h: int = 48, tag: int = 0) -> np.ndarray:
    a = np.full((h, w, 3), tag % 255, dtype=np.uint8)
    a[0, 0] = (tag + 1) % 255
    return a


# ---------------- 本地假 WeMM /v1/embeddings ----------------

class _FakeWemm:
    """记录请求（体+头），按 status_seq 回状态码；响应首维编码 content 图片数。"""

    def __init__(self) -> None:
        self.requests: list = []
        self.status_seq: deque = deque()
        self._lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                srv: _FakeWemm = self.server.fake
                with srv._lock:
                    srv.requests.append({"auth": self.headers.get("Authorization", ""),
                                         "body": body})
                    code = srv.status_seq.popleft() if srv.status_seq else 200
                if 200 <= code < 300:
                    dim = int(body.get("dimensions") or 4)
                    if isinstance(body.get("messages"), list):
                        n_img = sum(1 for c in body["messages"][0].get("content", [])
                                    if isinstance(c, dict) and c.get("type") == "image_url")
                    else:
                        n_img = 0
                    v = np.zeros(max(dim, 2), dtype=np.float32)
                    v[0], v[1] = float(n_img), 1.0
                    v = v[:dim] / float(np.linalg.norm(v[:dim]) or 1.0)
                    payload = {"data": [{"embedding": [round(float(x), 6) for x in v]}]}
                else:
                    payload = {"error": {"message": f"fake http {code}"}}
                out = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):  # 静音
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.fake = self
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/embeddings"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _dead_endpoint() -> str:
    """当前未监听的本地端口（不可达测试用）。"""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


# ---------------- [A] 客户端契约 ----------------

def test_client(srv: _FakeWemm) -> None:
    print("[A] WEMMClient 契约（请求形态 / 重试 / 快速失败 / key 零泄漏 / 回退 / mock 确定性）")
    cli = WEMMClient(endpoint=srv.url, api_key=TEST_KEY, timeout=5,
                     retry_delays=(0.01, 0.01), use_real_embed=True)

    # 文本 input 形态 + dimensions + Bearer
    srv.requests.clear()
    cli.embed_text("人群聚集", dimension=1024)
    r = srv.requests[-1]
    check("文本走 input 标量（非 messages/非数组）",
          r["body"].get("input") == "人群聚集" and "messages" not in r["body"],
          json.dumps(r["body"], ensure_ascii=False)[:80])
    check("请求带 dimensions=1024（服务端 MRL 截维统一走参数）",
          r["body"].get("dimensions") == 1024)
    check("请求带 model 名", r["body"].get("model") == cli.model)
    check("Authorization: Bearer <key> 头", r["auth"] == f"Bearer {TEST_KEY}")

    # 多模态：messages 只一组 / data URL / 缩放
    srv.requests.clear()
    cli.embed_image(img(1600, 900, tag=3), dimension=256)
    r = srv.requests[-1]
    check("多模态不用 input、走 messages", "input" not in r["body"]
          and isinstance(r["body"].get("messages"), list))
    msgs = r["body"]["messages"]
    check("一条请求只放一组 messages（role=user）",
          len(msgs) == 1 and msgs[0].get("role") == "user")
    content = msgs[0]["content"]
    check("单图消息 content = 1 个 image_url 项",
          len(content) == 1 and content[0]["type"] == "image_url")
    url = content[0]["image_url"]["url"]
    ok_decoded = False
    long_edge = -1
    if url.startswith("data:image/jpeg;base64,"):
        raw = base64.b64decode(url.split(",", 1)[1])
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            ok_decoded = True
            long_edge = max(arr.shape[:2])
    check("图片为可解码 jpeg data URL", ok_decoded)
    check("1600x900 按 max_resolution=896 缩放（只缩不放）", 0 < long_edge <= 896,
          f"长边 {long_edge}")

    # embed_frames：1 clip 多图消息 + N 单图消息，全部带 dimensions
    srv.requests.clear()
    clip_vec, frame_vecs = cli.embed_frames([img(320, 240, tag=i) for i in range(4)])
    check("4 帧 → 5 次请求（1 clip 多图 + 4 单图）", len(srv.requests) == 5,
          f"{len(srv.requests)} 次")
    img_counts = sorted(
        sum(1 for x in rq["body"]["messages"][0]["content"]
            if x.get("type") == "image_url")
        for rq in srv.requests)
    check("请求集合 = 1 条 4 图消息 + 4 条单图消息", img_counts == [1] * 4 + [4],
          str(img_counts))
    check("所有请求都带 dimensions", all("dimensions" in rq["body"] for rq in srv.requests))
    check("返回 (clip@1024, 4 个 frame@256) 且顺序对位",
          clip_vec.shape == (1024,) and len(frame_vecs) == 4
          and all(v.shape == (256,) for v in frame_vecs))
    check("clip 向量来自多图消息（首维标记=4 >0.9）、frame 来自单图（首维标记=1 <0.8）",
          float(clip_vec[0]) > 0.9 and all(float(v[0]) < 0.8 for v in frame_vecs))

    # 重试：5xx 退避重试、4xx 不重试
    srv.requests.clear()
    srv.status_seq = deque([500, 502, 200])
    v = cli.embed_text("重试", dimension=64)
    check("5xx×2 后成功：退避重试恰 3 次请求", v.shape == (64,) and len(srv.requests) == 3)
    srv.status_seq = deque([401])
    srv.requests.clear()
    try:
        cli.embed_text("鉴权", dimension=64)
        check("401 应抛 WemmHttpError", False)
    except WemmHttpError as e:
        check("401 不重试（恰 1 次请求）", e.status == 401 and len(srv.requests) == 1)
        check("异常信息不含 key", TEST_KEY not in str(e))
    srv.status_seq = deque([])

    # key 零泄漏：请求体 / 异常 / repr
    check("key 不出现在任何请求体",
          not any(TEST_KEY in json.dumps(x["body"]) for x in srv.requests))
    check("repr(client) 不含 key", TEST_KEY not in repr(cli))

    # 不可达快速失败（显式禁用备用，防 env 兜底切走）
    dead = _dead_endpoint()
    cli_dead = WEMMClient(endpoint=dead, fallback_endpoint="",
                          use_real_embed=True, probe_timeout=4.0)
    t0 = time.time()
    try:
        cli_dead.embed_text("x", dimension=64)
        check("不可达端点应报错", False)
    except RuntimeError as e:
        dt = time.time() - t0
        check("报错含「嵌入端点不可达」与端点地址",
              "嵌入端点不可达" in str(e) and dead in str(e), f"{dt:.2f}s")
        check("快速失败 ≤5s", dt <= 5.0, f"{dt:.2f}s")

    # 双端点回退
    srv.requests.clear()
    cli_fb = WEMMClient(endpoint=dead, fallback_endpoint=srv.url, api_key=TEST_KEY,
                        use_real_embed=True, retry_delays=(0.01,))
    v = cli_fb.embed_text("备用", dimension=64)
    check("主端点不可达自动切备用端点",
          v.shape == (64,) and cli_fb._active_endpoint == srv.url)

    # mock 伪向量确定性（显式 use_real_embed=False，覆盖 env）
    mcli = WEMMClient(use_real_embed=False, clip_dimension=128, frame_dimension=64)
    a1 = mcli.embed_text("人群聚集", dimension=128)
    a2 = mcli.embed_text("人群聚集", dimension=128)
    a3 = mcli.embed_text("车辆违停", dimension=128)
    check("mock: 同输入同向量、异输入异向量",
          bool(np.allclose(a1, a2)) and not np.allclose(a1, a3))
    check("mock: 维度=配置档位且 L2 归一", a1.shape == (128,)
          and abs(float(np.linalg.norm(a1)) - 1.0) < 1e-5)
    c1, f1 = mcli.embed_frames([img(tag=0), img(tag=1)])
    c2, f2 = mcli.embed_frames([img(tag=0), img(tag=1)])
    check("mock: embed_frames 同帧序同向量（确定性）",
          bool(np.allclose(c1, c2)) and all(np.allclose(x, y) for x, y in zip(f1, f2)))
    c3, _ = mcli.embed_frames([img(tag=1), img(tag=0)])
    check("mock: 帧序敏感（同帧异序异向量）", not np.allclose(c1, c3))


# ---------------- [B] EmbedStore 底库 ----------------

def test_store(tmp: str) -> None:
    print("[B] EmbedStore 底库（检索 / 精排 / 删除 / upsert / 维度锁 / 持久化）")
    st = EmbedStore(clip_dim=8, frame_dim=4)

    def vec(seed: int, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    id0 = st.add_clip(vec(0), [vec(10, 4), vec(11, 4)], frame_ts=[0.0, 0.6],
                      meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0})
    st.add_clip(vec(1), [vec(12, 4)], frame_ts=[5.0],
                meta={"video": "a.mp4", "t_start": 5.0, "t_end": 10.0})
    st.add_clip(vec(2), [], meta={"video": "b.mp4", "t_start": 0.0})
    check("写入 3 片段，库级维度锁定 8/4",
          len(st) == 3 and st.clip_dim == 8 and st.frame_dim == 4)

    hits = st.search(vec(0), top_k=2)
    check("检索 top_k：top1 命中自身（点积=余弦）",
          len(hits) == 2 and hits[0].clip_id == id0 and hits[0].rank == 1
          and hits[0].score > 0.999)
    check("命中携带元数据", hits[0].meta.get("video") == "a.mp4")
    check("video 过滤只在指定视频内", [h.clip_id for h in st.search(vec(0), video="b.mp4")] == [2]
          and st.search(vec(0), video="c.mp4") == [])

    fine = st.search_frames(vec(11, 4), id0, top_k=2)
    check("帧级精排：命中帧向量 + ts 对位",
          len(fine) == 2 and fine[0].rank == 1 and fine[0].clip_id == id0
          and abs(fine[0].ts - 0.6) < 1e-6)

    try:
        st.add_clip(np.ones(16, dtype=np.float32), meta={"video": "x", "t_start": 0})
        check("clip 维度不匹配应拒收", False)
    except ValueError as e:
        check("维度不匹配拒收并指明两侧维度", "16" in str(e) and "8" in str(e))
    try:
        st.add_clip(vec(3), [np.ones(9, dtype=np.float32)], meta={"video": "x", "t_start": 0})
        check("frame 维度不匹配应拒收", False)
    except ValueError as e:
        check("frame 维度不匹配拒收并指明", "9" in str(e) and "4" in str(e))

    n_before = (len(st), len(st._frame_vecs))
    st.add_clip(vec(0), [vec(13, 4)], frame_ts=[0.3],
                meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0})
    check("upsert 幂等：同 (video,t_start) 行数不变、旧 frame 清除",
          len(st) == n_before[0] and len(st._frame_vecs) == n_before[1] - 1)
    check("upsert 后新 frame 向量生效",
          abs(st.search_frames(vec(13, 4), id0)[0].ts - 0.3) < 1e-6)

    check("delete_video 删片段并连带删帧向量、重复删除返回 0",
          st.delete_video("a.mp4") == 2 and len(st) == 1 and st.delete_video("a.mp4") == 0)

    p = os.path.join(tmp, "qa_vecs.npz")
    st.save(p)
    st2 = EmbedStore().load(p)
    check("save→load 往返：条数/维度一致", st2.stats() == st.stats())
    check("往返后检索语义不变（删除后 id 重映射：b.mp4 → 0）",
          st2.search(vec(2), top_k=1)[0].clip_id == 0)
    check("往返 frame 层一致", len(st2._frame_meta) == len(st._frame_meta))


# ---------------- [C] 注册与校验契约 ----------------

def _vemb_params(**over) -> dict:
    p = {"embed_endpoint": "", "embed_model": "", "embed_dimension": 1024,
         "frame_dimension": 256, "use_real_embed": False, "embed_timeout": 120,
         "max_resolution": "896,896", "jpeg_quality": 85,
         "frames_per_clip": 6, "clip_seconds": 5.0,
         "thumb_dir": "out/video_thumbs", "thumb_long": 320,
         "flush_partial": True, "store_path": "out/video_vecs.npz"}
    p.update(over)
    return p


def _pipe0004_spec(**vemb_over) -> dict:
    return {"name": "视频向量化-嵌入查询链路", "stages": [
        {"module_id": "frame_manager", "params": {
            "sampling": "analysis", "frame_skip": 15, "interval_sec": 3.0,
            "interval_frames": 30, "backpressure": "standard"}},
        {"module_id": "video_embed", "params": _vemb_params(**vemb_over)},
    ]}


def test_registry_and_guards() -> None:
    print("[C] 注册契约 + validate_chain 护栏 + 装配期 fail fast")
    import db as webdb
    import server
    from validate_chain import TERMINAL_OK, validate

    # 注册契约
    params = server.PARAMS.get("video_embed") or {}
    chain_keys = {k for k, v in params.items() if not v.get("test_only")}
    check("PARAMS[video_embed] 恰 15 键（14 链路键 + 1 test_only）", len(params) == 15,
          f"{len(params)} 键")
    check("test_only 仅 video（测试页专用，不进链路 spec）",
          set(params) - chain_keys == {"video"})
    expected_keys = {"embed_endpoint", "embed_model", "embed_dimension",
                     "frame_dimension", "use_real_embed", "embed_timeout",
                     "max_resolution", "jpeg_quality", "frames_per_clip",
                     "clip_seconds", "thumb_dir", "thumb_long", "flush_partial",
                     "store_path"}
    check("链路参数键与设计稿 §4.5/§3.3 清单一致", chain_keys == expected_keys,
          str(sorted(chain_keys)))
    check("db._ID_PREFIXES 登记 video_embed → VEMB",
          webdb._ID_PREFIXES.get("video_embed") == "VEMB")
    check("CHAIN_ASSEMBLED_MODULES 含 video_embed（声明即执行）",
          "video_embed" in server.CHAIN_ASSEMBLED_MODULES)
    check("validate_chain 终段白名单含 video_embed", "video_embed" in TERMINAL_OK)
    check("_PORT_DECL_SOURCES 登记 video_embed 自描述声明",
          server._PORT_DECL_SOURCES.get("video_embed")
          == ("video_embed.video_embed_stage", "VideoEmbedStage"))
    from video_embed.video_embed_stage import VideoEmbedStage
    # 契约 B（WO-0033.2，hybrid_retrieval.md §12.1）：INPUTS 增第二入端口 dets（Records，可缺省）
    check("VideoEmbedStage INPUTS/OUTPUTS 自描述（吃 Image+Records 产 Records）",
          VideoEmbedStage.INPUTS == {"frame": "Image", "dets": "Records"}
          and VideoEmbedStage.OUTPUTS == {"clips": "Records"})

    # validate_chain：PIPE0004 种子 spec 通过（临时目录路径，不碰真实 out/）
    with tempfile.TemporaryDirectory(prefix="umvp_vemb_guard_") as td:
        spec = _pipe0004_spec(store_path=os.path.join(td, "v.npz"),
                              thumb_dir=os.path.join(td, "thumbs"))
        warns: list = []
        errs = validate(spec, warnings=warns)
        check("validate_chain: PIPE0004 种子形态 spec 通过", not errs, "; ".join(errs))

        bad = {"stages": spec["stages"] + [{"module_id": "annotate", "params": {}}]}
        errs = validate(bad)
        check("validate_chain: video_embed 非终段拒（后接非法终段报错）",
              bool(errs) and any("终段" in e for e in errs), "; ".join(errs)[:120])

    # 装配期护栏（校验期暂不拦，见文件头「已知现状」；装配层 WEMMClient fail fast）
    # 屏蔽 WEMM env（--live 场景可能 source 过 REMOTE_CLIENT.env）：本组断言语义为
    # 「无端点」，须排除 env 兜底（对齐上游 test_video_embed.py 的 env 屏蔽做法）
    env_backup = {k: os.environ.pop(k) for k in
                  ("WEMM_BASE_URL", "WEMM_ENDPOINT", "WEMM_API_KEY", "WEMM_MODEL",
                   "WEMM_BASE_URL_FALLBACK") if k in os.environ}
    try:
        try:
            server._build_stage_instance("video_embed", _vemb_params(embed_dimension=768))
            check("装配期: 维度 768 应 fail fast", False)
        except ValueError as e:
            check("装配期: 维度 768 报错含原值与合法档位",
                  "768" in str(e) and all(str(d) in str(e) for d in WEMM_MRL_DIMS),
                  str(e)[:100])
        try:
            server._build_stage_instance(
                "video_embed", _vemb_params(use_real_embed=True, embed_endpoint=""))
            check("装配期: use_real_embed=true 无端点应 fail fast", False)
        except ValueError as e:
            check("装配期: use_real_embed=true 无端点报错含 env 提示",
                  "WEMM_BASE_URL" in str(e), str(e)[:100])
    finally:
        os.environ.update(env_backup)

    # 装配惰性零副作用：mock 档构造不加载模型/不锁维度/零网络。
    # store_path 必须指临时**不存在**路径（WO-0033.13 机器状态解耦）：新契约下
    # store_path 存在即装配期 load 并锁维度（WO-0033.12），默认 out/video_vecs.npz
    # 在跑过 PIPE0004 的机器上恒存在，惰性断言会假挂。
    with tempfile.TemporaryDirectory(prefix="umvp_vemb_lazy_") as td_lazy:
        stage = server._build_stage_instance(
            "video_embed", _vemb_params(store_path=os.path.join(td_lazy, "absent.npz")))
        check("装配惰性: mock 档构造零副作用（client.mock / store 维度未锁）",
              stage.client.mock is True and stage.store.clip_dim is None
              and stage.store.frame_dim is None)
        check("装配惰性: 默认参数生效（6 帧/窗、thumb_dir 归一项目根）",
              stage.frames_per_clip == 6
              and str(stage.thumb_dir).startswith(str(ROOT))
              and not str(stage.thumb_dir).startswith(str(ROOT / "web_lab")))

        # 装配加载正向契约（WO-0033.12）：store_path 存在 → 装配期 load；
        # 已载库维度与配置不一致 → 装配期 ValueError（禁止静默混维）
        def _vec(seed: int, dim: int) -> np.ndarray:
            v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
            return v / np.linalg.norm(v)

        ok_npz = os.path.join(td_lazy, "match.npz")
        _st_ok = EmbedStore(clip_dim=1024, frame_dim=256)
        _st_ok.add_clip(_vec(0, 1024), [_vec(1, 256)], frame_ts=[1.0],
                        meta={"video": "m.mp4", "t_start": 0.0, "t_end": 5.0})
        _st_ok.save(ok_npz)
        stage_ok = server._build_stage_instance(
            "video_embed", _vemb_params(store_path=ok_npz))
        check("装配加载: store_path 存在 → 装配期 load（维度锁定 + 条数带入）",
              stage_ok.store.clip_dim == 1024 and stage_ok.store.frame_dim == 256
              and len(stage_ok.store) == 1,
              f"clip={stage_ok.store.clip_dim} frame={stage_ok.store.frame_dim} "
              f"n={len(stage_ok.store)}")
        check("装配加载: load 后仍为 mock 档（零网络语义不变）",
              stage_ok.client.mock is True)

        bad_npz = os.path.join(td_lazy, "mismatch.npz")
        _st_bad = EmbedStore(clip_dim=8, frame_dim=4)
        _st_bad.add_clip(_vec(2, 8), [_vec(3, 4)], frame_ts=[0.5],
                         meta={"video": "x.mp4", "t_start": 0.0, "t_end": 5.0})
        _st_bad.save(bad_npz)
        try:
            server._build_stage_instance("video_embed",
                                         _vemb_params(store_path=bad_npz))
            check("装配加载: 维度不匹配应 ValueError", False, "未抛异常")
        except ValueError as e:
            check("装配加载: 维度不匹配装配期 ValueError（两侧维度 + 指引）",
                  "8" in str(e) and "1024" in str(e) and "store_path" in str(e),
                  str(e)[:110])


# ---------------- [D] 链路级（进程内 h_pipeline_run，mock 档全流程） ----------------

def test_chain_e2e() -> None:
    print("[D] PIPE0004 链路级：mock 档跑真实视频 → 入库/事件/落盘/检索/幂等")
    import server

    video = ROOT / "tests/data/vid/violent1.mp4"
    if not video.is_file():
        check("测试视频存在 tests/data/vid/violent1.mp4", False, "素材缺失")
        return
    check("测试视频存在 tests/data/vid/violent1.mp4", True)

    max_frames = 200  # frame_skip=15 → 放行 14 帧 → 2 满窗 + 1 partial = 3 片段
    with tempfile.TemporaryDirectory(prefix="umvp_vemb_e2e_") as td:
        spec = _pipe0004_spec(store_path=os.path.join(td, "video_vecs.npz"),
                              thumb_dir=os.path.join(td, "thumbs"))
        orig_load = server.db.load_pipeline
        try:
            server.db.load_pipeline = lambda pid: dict(spec)
            r = server.h_pipeline_run({"pipeline_id": "PIPE0004",
                                       "video": "violent1.mp4",
                                       "max_frames": max_frames,
                                       "output_dir": td})
        finally:
            server.db.load_pipeline = orig_load

        check("驱动: PIPE0004 mock 全流程运行成功", r.get("ok") is True,
              str(r.get("error"))[:160])
        text = r.get("text") or ""
        check("结果头含向量化片段统计（「向量化片段 N 条」）",
              "向量化片段" in text, text.splitlines()[0][:120] if text else "")
        check("无 output 声明 → 结果不带 output 块（WO-0029 零回归）",
              "output" not in r)

        timeline = r.get("timeline") or []
        stages = {t.get("stage") for t in timeline if t.get("stage")}
        check("timeline 含 frame_manager 与 video_embed 阶段",
              {"frame_manager", "video_embed"} <= stages, str(sorted(stages)))

        subs = [t for t in timeline
                if t.get("stage") == "video_embed" and t.get("kind") == "submit"
                and "入库" in (t.get("text") or "")]
        evs = [t for t in timeline
               if t.get("stage") == "video_embed"
               and isinstance(t.get("data"), dict)
               and t["data"].get("name") == "clip_indexed"]
        check("timeline 有片段入库条目（≥1 条「片段#n 入库」）", len(subs) >= 1,
              f"{len(subs)} 条")
        check("timeline 有 clip_indexed 事件（publish 进事件流）", len(evs) >= 1,
              f"{len(evs)} 条")

        npz = os.path.join(td, "video_vecs.npz")
        check("底库落盘（.npz + 同名 .json）",
              os.path.isfile(npz) and os.path.isfile(npz[:-4] + ".json"))
        check("结果头含底库落盘路径", os.path.join(td, "video_vecs.npz") in text
              or "video_vecs.npz" in text)

        thumbs = sorted(Path(td, "thumbs").glob("*.jpg"))
        n_clips = len(evs)
        check(f"缩略图落盘且数量=片段数（{len(thumbs)}=={n_clips}）",
              len(thumbs) == n_clips >= 1)

        store = EmbedStore().load(npz)
        check("EmbedStore.load 后条数 = timeline 片段数", len(store) == n_clips,
              f"库 {len(store)} / timeline {n_clips}")
        check("片段数 = ⌈放行帧/窗口⌉（14 帧 @6/窗 → 3 片段，尾段 partial）",
              n_clips == 3, f"{n_clips} 片段")
        metas = [store._clip_meta[i] for i in range(len(store))]
        check("元数据 video 正确、partial 尾段恰 1 条",
              all(m.get("video") == "violent1.mp4" for m in metas)
              and sum(1 for m in metas if m.get("partial")) == 1)
        check("缩略图路径写进 store 元数据且存在",
              all(m.get("thumb_path") and os.path.isfile(m["thumb_path"]) for m in metas))

        # 检索命中：库内首片段向量自检（load 后检索通路可用）
        q = store._clip_vecs[0]
        hit = store.search(q, top_k=1)[0]
        check("load 后 clip 粗排命中（top1=自身, score≈1.0）",
              hit.clip_id == 0 and hit.score > 0.999, f"score={hit.score:.4f}")
        fhit = store.search_frames(store._frame_vecs[0], 0, top_k=1)[0]
        check("load 后 frame 精排命中（ts 与元数据 frame_ts 对位）",
              abs(fhit.ts - metas[0]["frame_ts"][0]) < 1e-6,
              f"ts={fhit.ts} vs {metas[0]['frame_ts'][0]}")

        # 重跑 upsert 幂等：同 spec 同底库再跑一遍，条数不变
        try:
            server.db.load_pipeline = lambda pid: dict(spec)
            r2 = server.h_pipeline_run({"pipeline_id": "PIPE0004",
                                        "video": "violent1.mp4",
                                        "max_frames": max_frames,
                                        "output_dir": td})
        finally:
            server.db.load_pipeline = orig_load
        check("重跑成功", r2.get("ok") is True, str(r2.get("error"))[:120])
        store2 = EmbedStore().load(npz)
        check("重跑 upsert 幂等：库条数不变", len(store2) == n_clips,
              f"{len(store2)} vs {n_clips}")
        check("重跑后检索语义不变", store2.search(q, top_k=1)[0].score > 0.999)


# ---------------- [E] 播种幂等 ----------------

def test_seed_idempotent() -> None:
    print("[E] PIPE0004 播种幂等（进程内，monkeypatch db）")
    import server

    stages = server._SEED_VEMB_STAGES
    # WO-0033.5：播种结构升级 4 阶段（hybrid_retrieval §12-2：YOLO 观察源插 det_filter）
    check("_SEED_VEMB_STAGES 结构 = frame_manager→yolo→det_filter→video_embed（终段）",
          [m for m, _n, _p in stages] == ["frame_manager", "yolo", "det_filter",
                                          "video_embed"])
    vemb_p = dict(stages[-1][2])
    check("种子 video_embed 参数为 mock 档（use_real_embed=False）且 15 参数含 store_path",
          vemb_p.get("use_real_embed") is False and "store_path" in vemb_p)

    calls = {"preset": [], "pipeline": []}

    def fake_save_preset(kind, name, mid, params, preset_id=None):
        calls["preset"].append((mid, name))
        return (f"{mid.upper()}-NEW", True)

    def fake_save_pipeline(name, stage_ids, streams=1, budget=None, meta=None):
        calls["pipeline"].append((name, tuple(stage_ids)))
        return ("PIPE0004-NEW", True)

    orig = (server.db.list_pipelines, server.db.save_preset, server.db.save_pipeline)
    try:
        server.db.list_pipelines = lambda: []
        server.db.save_preset = fake_save_preset
        server.db.save_pipeline = fake_save_pipeline
        have: dict = {}
        server._seed_chain(server._SEED_VEMB_NAME, stages, "qa 播种", have)
        mids = [m for m, _n in calls["preset"]]
        check("首播：4 份模块 preset（frame_manager/yolo/det_filter/video_embed 各一）+ 1 条链路",
              mids == ["frame_manager", "yolo", "det_filter", "video_embed"]
              and len(calls["pipeline"]) == 1,
              f"preset={mids} pipeline={len(calls['pipeline'])}")
        check("首播链路引用 4 个阶段 preset",
              len(calls["pipeline"][0][1]) == 4)
        check("have 回填 preset id（复用机制在位）", len(have) == 4)

        # 复播：已有同结构链路 → 幂等零写入
        calls["preset"].clear()
        calls["pipeline"].clear()
        existing = [{"name": server._SEED_VEMB_NAME,
                     "steps": [{"module_id": m} for m in
                               ("frame_manager", "yolo", "det_filter", "video_embed")]}]
        server.db.list_pipelines = lambda: existing
        server._seed_chain(server._SEED_VEMB_NAME, stages, "qa 播种", have)
        check("复播同结构零写入（幂等，不动用户改动）",
              not calls["preset"] and not calls["pipeline"])

        # 存量 2 阶段旧库（WO-0033.5 升级前形态）→ 结构变 → 按模块复用旧 preset 重建：
        # FM/VEMB 保留旧 id，YOLO/DF 补建，步骤顺序 = 播种结构顺序
        calls["preset"].clear()
        calls["pipeline"].clear()
        legacy = [{"name": server._SEED_VEMB_NAME,
                   "steps": [{"module_id": "frame_manager", "preset_id": "FM-OLD"},
                             {"module_id": "video_embed", "preset_id": "VEMB-OLD"}]}]
        server.db.list_pipelines = lambda: legacy
        have2: dict = {}
        server._seed_chain(server._SEED_VEMB_NAME, stages, "qa 播种", have2)
        new_mids = [m for m, _n in calls["preset"]]
        check("存量 2 阶段库 → 结构变自动重建（save_pipeline 恰 1 次）",
              len(calls["pipeline"]) == 1)
        check("重建复用旧 preset：FM/VEMB 保留旧 id、顺序按播种结构",
              calls["pipeline"][0][1] == ("FM-OLD", "YOLO-NEW", "DET_FILTER-NEW",
                                          "VEMB-OLD"),
              str(calls["pipeline"][0][1]))
        check("重建只补建缺失阶段 preset（yolo/det_filter，不重复播种 FM/VEMB）",
              new_mids == ["yolo", "det_filter"] and len(have2) == 2,
              f"preset={new_mids}")
    finally:
        (server.db.list_pipelines, server.db.save_preset,
         server.db.save_pipeline) = orig


# ---------------- [LIVE] 真端点冒烟 ----------------

def test_live() -> None:
    print("[LIVE] 真端点冒烟（不在线优雅 SKIP）")
    ep, _fb, model, key = resolve_wemm_cfg()
    if not ep:
        print("  [SKIP] 未配置嵌入端点（source wemm接口/REMOTE_CLIENT.env 导出 "
              "WEMM_BASE_URL / WEMM_API_KEY）")
        return
    if not endpoint_reachable(ep, timeout=5.0):
        print(f"  [SKIP] 嵌入端点不在线: {ep}（TCP 探测 ≤5s 失败；服务端未启动属合法结果）")
        return
    print(f"  端点在线: {ep} | model={model or '(默认)'} | key={'已配置' if key else '无'}")
    cli = WEMMClient(use_real_embed=True, timeout=120)
    v = cli.embed_text("人群聚集", dimension=1024)
    check_live("文本嵌入: 维度=1024 且 L2≈1", v.shape == (1024,)
               and abs(float(np.linalg.norm(v)) - 1.0) < 0.05)
    v2 = cli.embed_text("人群聚集", dimension=1024)
    check_live("同文本两次嵌入确定性一致", bool(np.allclose(v, v2, atol=1e-4)))
    vi = cli.embed_image(img(640, 360, tag=7), dimension=256)
    check_live("单图嵌入: 维度 256 且 L2≈1", vi.shape == (256,)
               and abs(float(np.linalg.norm(vi)) - 1.0) < 0.05)
    cv_, fvs = cli.embed_frames([img(896, 504, tag=i) for i in range(3)])
    check_live("clip 多图（3 帧→1 向量@1024）+ 逐帧单图（3 向量@256）",
               cv_.shape == (1024,) and len(fvs) == 3
               and all(f.shape == (256,) for f in fvs))


# ---------------- main ----------------

def main() -> None:
    tmp = tempfile.mkdtemp(prefix="umvp_vemb_qa_")
    # WO-0033 起链路收尾会自动对齐落表（_archive_and_align → review_index/review_video），
    # env 隔离到临时文件，避免污染真实 out/review_*.json（无 MySQL 时落 JSON 默认路径）
    review_env_backup = {k: os.environ.pop(k) for k in
                         ("UMVP_REVIEW_INDEX_JSON", "UMVP_REVIEW_VIDEO_JSON")
                         if k in os.environ}
    os.environ["UMVP_REVIEW_INDEX_JSON"] = os.path.join(tmp, "review_index.json")
    os.environ["UMVP_REVIEW_VIDEO_JSON"] = os.path.join(tmp, "review_video.json")
    srv = _FakeWemm()
    try:
        test_client(srv)
        test_store(tmp)
        test_registry_and_guards()
        test_chain_e2e()
        test_seed_idempotent()
    finally:
        srv.close()
        os.environ.update(review_env_backup)
    print(f"\n[mock 档] 全部 {PASS} 项断言通过 ✓")
    if LIVE:
        test_live()
        if PASS_LIVE:
            print(f"[live] 真端点冒烟 {PASS_LIVE} 项断言通过 ✓")
        else:
            print("[live] 真端点冒烟跳过（见上方 SKIP 原因，不算失败）")
    print("done.")


if __name__ == "__main__":
    main()
