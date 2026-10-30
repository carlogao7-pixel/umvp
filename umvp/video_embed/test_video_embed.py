# -*- coding: utf-8 -*-
"""video_embed 三件套断言式单测（WO-0031.1，设计 doc/design/wemm_integration.md §4/§7、
doc/design/video_review_search.md §3.3/§5）。

两层结构：

  默认（纯逻辑 + 本地 mock HTTP，秒级；不连真端点、不加载模型权重）：
    [A] mock 档：确定性伪向量（同输入同向量 / 异输入异向量）、维度 = 配置档位、
        L2 归一、不同档位共享前缀（MRL 同形）、空输入报错、use_real_embed=True
        无端点 fail fast、并发数收敛 [1,4]。
    [B] 档位校验：非法维度 ValueError 且列出合法档位 64..2048。
    [C] 请求 schema（本地 ThreadingHTTPServer 断言请求体/头）：
        文本走 input + dimensions + Bearer 头；多模态走 messages（content = image_url
        data URL，禁 input）；帧按 max_resolution 缩放（1600x900 → 长边 ≤896）；
        embed_frames = 1 条 clip 多图消息 + N 条单图消息（共 N+1 请求）且顺序对位；
        5xx 退避重试（500,502,200 → 成功且恰 3 次）；401 不重试（恰 1 次）且
        key 永不入异常；端点不可达 ≤5s 快速失败（报错含地址）；主端点挂 → 自动切备用。
    [D] EmbedStore：检索 top1 自命中 / 维度不匹配拒收并指明 / upsert 幂等（行数不变）
        / video 过滤 / delete_video 连带帧向量 / frame 精排 / save-load roundtrip / stats。
    [E] VideoEmbedStage：窗口聚合（frames_per_clip 切窗）、片段记录字段与事件、
        partial 尾片段、缩略图落盘（**项目相对路径**）、embedding 走旁路（主干可
        JSON 序列化）、observation 观察标签（dets 输入分支；含 frame_dets 原始
        记录流：三桶混存 / 剥 model_path·旁路键 / 对账 / 空观察源）、frames_embed=False
        分支、store meta 不含标签、store 条数对齐、INPUTS/OUTPUTS 自描述、index_video 集成。
    [F] label_align（契约 A，纯逻辑）：parse_video_time（文件名>tag/失败双空不抛错）、
        clip_windows（嵌套/frames_per_clip/右开左闭）、aggregate_frame_stats（cls_id
        聚合 / track 并集去重 / plate_nos 集合 / face_count 最大 / frame_dets 原始
        记录流混存·剥键·截断 / 空输入）、
        align_events（边界归下窗 / 跨窗各记一条 / ts 原值）、build_clip_tags（§2.5 全字段）。

  --live（真实 WeMM 端点，source wemm接口/REMOTE_CLIENT.env 后运行）：
    探活 ≤5s，不在线 → 打印 SKIP 原因优雅跳过；在线 → 文本 / 单图 / clip 多图
    → 维度 = 请求档位、L2 ≈ 1、同文本向量确定性。

运行:
    conda run -n ai python umvp/video_embed/test_video_embed.py           # mock 档全过
    source wemm接口/REMOTE_CLIENT.env && \
    conda run -n ai python umvp/video_embed/test_video_embed.py --live    # + 真端点冒烟
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "umvp"))

from video_embed.embed_store import EmbedStore  # noqa: E402
from video_embed.label_align import (  # noqa: E402
    aggregate_frame_stats, align_events, build_clip_tags, clip_windows, parse_video_time,
)
from video_embed.video_embed_stage import VideoEmbedStage  # noqa: E402
from video_embed.wemm_client import (  # noqa: E402
    WEMM_MRL_DIMS, WemmHttpError, WEMMClient, check_dimension, endpoint_reachable,
    resolve_wemm_cfg,
)

PASS = 0       # mock 档断言计数
PASS_LIVE = 0  # --live 断言计数
LIVE = "--live" in sys.argv
TEST_KEY = "sk-test-key-000"  # 仅本地 mock 服务用（非真实凭据）


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


def img(w: int = 64, h: int = 48, tag: int = 0) -> np.ndarray:
    """假图：tag 决定像素内容（异 tag 异内容）。"""
    a = np.full((h, w, 3), tag % 255, dtype=np.uint8)
    a[0, 0] = (tag + 1) % 255
    return a


# ---------------- 本地 mock WeMM 服务 ----------------

class _FakeWemmServer:
    """本地假 WeMM /v1/embeddings：记录请求（体+头），按 status_seq 回状态码。

    响应向量 = [content 图片数（文本=0），1, 0...] 截到 dimensions 维后 L2 归一——
    首维可反推 content 项数，用于断言 clip 多图消息与单图消息、请求顺序对位。
    """

    def __init__(self) -> None:
        self.requests: list = []
        self.status_seq: deque = deque()
        self.count = 0
        self._lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.1 keep-alive（带 Content-Length）：避免并发后 urllib3 池里
            # 残留 HTTP/1.0 半死连接，干扰网络错误重试的断言
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                srv: _FakeWemmServer = self.server.fake
                with srv._lock:
                    srv.count += 1
                    try:
                        body = json.loads(raw)
                    except Exception:
                        body = {"_raw": raw[:64].decode("utf-8", "replace")}
                    srv.requests.append({"path": self.path, "idx": srv.count,
                                         "auth": self.headers.get("Authorization", ""),
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
                    payload = {"data": [{"embedding": [round(float(x), 6) for x in v],
                                         "index": 0}]}
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
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/embeddings"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _unused_port() -> int:
    """取一个当前未监听的端口（bind 后立即关闭，用于不可达测试）。"""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------- [A] mock 档 / [B] 档位 ----------------

def test_mock_and_dims(srv: _FakeWemmServer) -> None:
    print("[A/B] mock 档与档位校验")
    cli = WEMMClient(endpoint=srv.url, api_key=TEST_KEY, use_real_embed=False)
    ok(cli.mock, "use_real_embed=False → mock 档")

    v1 = cli.embed_text("人群聚集", dimension=256)
    v2 = cli.embed_text("人群聚集", dimension=256)
    ok(v1.shape == (256,), "embed_text 维度 = 请求档位 256")
    ok(abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5, "embed_text 输出 L2 归一")
    ok(bool(np.allclose(v1, v2)), "同输入同向量（确定性伪向量）")
    v3 = cli.embed_text("车辆违停", dimension=256)
    ok(not np.allclose(v1, v3), "异输入异向量")

    v1024 = cli.embed_text("人群聚集", dimension=1024)
    v64 = cli.embed_text("人群聚集", dimension=64)
    ok(v1024.shape == (1024,) and v64.shape == (64,), "各档位维度 = 配置值")
    prefix = v1024[:64] / float(np.linalg.norm(v1024[:64]))
    ok(bool(np.allclose(v64, prefix, atol=1e-5)),
       "不同档位共享同一前缀方向（MRL 截形同构：各档截后各自归一）")

    iv1 = cli.embed_image(img(tag=1), dimension=128)
    iv2 = cli.embed_image(img(tag=1), dimension=128)
    iv3 = cli.embed_image(img(tag=2), dimension=128)
    ok(iv1.shape == (128,) and bool(np.allclose(iv1, iv2)) and not np.allclose(iv1, iv3),
       "embed_image 确定性：同图同向量、异图异向量")
    ok(not np.allclose(iv1, cli.embed_text("人群聚集", dimension=128)),
       "文本与图像输入空间分离（key 不同族）")

    frames = [img(tag=i) for i in range(3)]
    c1, f1 = cli.embed_frames(frames, clip_dimension=512, frame_dimension=64)
    c2, f2 = cli.embed_frames(frames, clip_dimension=512, frame_dimension=64)
    ok(c1.shape == (512,) and len(f1) == 3 and f1[0].shape == (64,),
       "embed_frames → (clip 向量, 每帧向量组)，维度 = 配置档位")
    ok(bool(np.allclose(c1, c2)) and all(np.allclose(a, b) for a, b in zip(f1, f2)),
       "embed_frames 确定性：同帧序同向量")
    c3, _ = cli.embed_frames(list(reversed(frames)), clip_dimension=512, frame_dimension=64)
    ok(not np.allclose(c1, c3), "embed_frames 帧序敏感（同帧异序异向量）")

    for bad_call, desc in (
        (lambda: cli.embed_text("  "), "空文本报错"),
        (lambda: cli.embed_frames([]), "空帧列表报错"),
        (lambda: cli.embed_image(None), "空图像报错"),
    ):
        try:
            bad_call()
            ok(False, desc)
        except ValueError:
            ok(True, desc)

    # 屏蔽 env（source 过 REMOTE_CLIENT.env 时也成立）：use_real_embed=True 无端点 → fail fast
    env_backup = {k: os.environ.pop(k) for k in
                  ("WEMM_BASE_URL", "WEMM_ENDPOINT", "WEMM_API_KEY", "WEMM_MODEL",
                   "WEMM_BASE_URL_FALLBACK") if k in os.environ}
    try:
        try:
            WEMMClient(use_real_embed=True)  # 无端点
            ok(False, "use_real_embed=True 无端点应 fail fast")
        except ValueError as e:
            ok("WEMM_BASE_URL" in str(e), "use_real_embed=True 无端点 fail fast（提示 env）")
        cli_noenv = WEMMClient(use_real_embed=False)
        ok(cli_noenv.mock and cli_noenv.endpoint == "",
           "屏蔽 env 且未传端点 → mock 档（端点空）")
    finally:
        os.environ.update(env_backup)

    ok(WEMMClient(endpoint=srv.url, use_real_embed=False).concurrency == 2
       and WEMMClient(endpoint=srv.url, concurrency=8, use_real_embed=False).concurrency == 4
       and WEMMClient(endpoint=srv.url, concurrency=0, use_real_embed=False).concurrency == 1,
       "并发数收敛到 [1,4]（服务端建议 1–4）")

    try:
        check_dimension(100)
        ok(False, "非法档位应报错")
    except ValueError as e:
        ok(all(str(d) in str(e) for d in WEMM_MRL_DIMS),
           "非法维度报错列出全部合法档位 64/128/256/512/1024/2048")
    ok(all(check_dimension(d) == d for d in WEMM_MRL_DIMS), "全部合法档位通过校验")


# ---------------- [C] 请求 schema（本地 mock HTTP） ----------------

def test_http_schema(srv: _FakeWemmServer) -> None:
    print("[C] 请求 schema（messages 形态 / dimensions / Bearer / 缩放 / 重试 / 快速失败）")
    cli = WEMMClient(endpoint=srv.url, api_key=TEST_KEY, timeout=5,
                     retry_delays=(0.01, 0.01), use_real_embed=True)
    ok(not cli.mock, "有端点 + use_real_embed=True → real 档")

    # 文本：input + dimensions + Bearer
    srv.requests.clear()
    cli.embed_text("人群聚集", dimension=1024)
    r = srv.requests[-1]
    ok(r["body"].get("input") == "人群聚集", "纯文本走 input 字段（实测 schema）")
    ok(r["body"].get("dimensions") == 1024, "请求带 dimensions（服务端 MRL 截维统一走参数）")
    ok(r["body"].get("model") == cli.model, "请求带 model")
    ok(r["auth"] == f"Bearer {TEST_KEY}", "Authorization: Bearer <key> 头正确")
    ok(not any(TEST_KEY in json.dumps(x["body"]) for x in srv.requests),
       "key 不出现在任何请求体（只走头）")

    # 单图：messages + image_url data URL + 缩放
    srv.requests.clear()
    cli.embed_image(img(1600, 900, tag=3), dimension=256)
    r = srv.requests[-1]
    ok("input" not in r["body"], "多模态不用 input（实测修正）")
    msgs = r["body"].get("messages")
    ok(isinstance(msgs, list) and len(msgs) == 1 and msgs[0]["role"] == "user",
       "一条请求只放一组 messages（role=user）")
    content = msgs[0]["content"]
    ok(len(content) == 1 and content[0]["type"] == "image_url",
       "单图消息 content = 1 个 image_url 项")
    url = content[0]["image_url"]["url"]
    ok(url.startswith("data:image/jpeg;base64,"), "图片为 jpeg data URL 编码")
    raw = base64.b64decode(url.split(",", 1)[1])
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    ok(arr is not None and max(arr.shape[0], arr.shape[1]) <= 896,
       f"1600x900 帧按 max_resolution 缩放（实测长边 {max(arr.shape[:2])} ≤ 896）")

    # embed_frames：1 clip 多图消息 + N 单图消息，顺序对位、并发
    srv.requests.clear()
    frames = [img(320, 240, tag=i) for i in range(6)]
    clip_vec, frame_vecs = cli.embed_frames(frames)  # 默认档 clip=1024 frame=256
    ok(len(srv.requests) == 7, "6 帧 → 7 次请求（1 条 clip 多图 + 6 条逐帧单图）")
    ok(clip_vec.shape == (1024,) and len(frame_vecs) == 6
       and all(v.shape == (256,) for v in frame_vecs),
       "返回 (clip 向量@1024, 6 个 frame 向量@256)")
    # 响应首维编码了 content 图片数（clip=6、单图=1）：顺序对位断言
    ok(float(clip_vec[0]) > 0.9, "clip 向量来自 6 图消息（首维标记=6，归一后 >0.9）")
    ok(all(float(v[0]) < 0.8 for v in frame_vecs),
       "frame 向量逐条来自单图消息（首维标记=1）")
    img_counts = sorted(
        sum(1 for x in r["body"]["messages"][0]["content"] if x.get("type") == "image_url")
        for r in srv.requests)
    ok(img_counts == [1] * 6 + [6],
       "请求集合 = 1 条 6 图 clip 消息 + 6 条单图消息"
       "（并发到达顺序不定，客户端 map 保序对位返回）")
    ok(all("dimensions" in r["body"] for r in srv.requests),
       "每个请求都带 dimensions（截维全链路统一）")

    # frames_embed=False（with_frames=False）：只发 clip 一条请求、不返回 frame 向量
    srv.requests.clear()
    clip_only, no_frame_vecs = cli.embed_frames(frames, with_frames=False)
    ok(clip_only.shape == (1024,) and no_frame_vecs == [] and len(srv.requests) == 1
       and float(clip_only[0]) > 0.9,
       "with_frames=False → 恰 1 次 clip 多图请求、无 frame 向量（成本大头省掉）")

    # 5xx 退避重试
    srv.requests.clear()
    srv.status_seq = deque([500, 502, 200])
    v = cli.embed_text("重试", dimension=64)
    ok(v.shape == (64,) and len(srv.requests) == 3, "5xx×2 后成功：指数退避重试恰 3 次请求")
    srv.status_seq = deque([])

    # 401 不重试 + key 不入异常
    srv.requests.clear()
    srv.status_seq = deque([401])
    try:
        cli.embed_text("鉴权", dimension=64)
        ok(False, "401 应抛 WemmHttpError")
    except WemmHttpError as e:
        ok(e.status == 401 and len(srv.requests) == 1, "401 不重试（恰 1 次请求）")
        ok(TEST_KEY not in str(e), "异常信息不含 key")
    srv.status_seq = deque([])

    # 422 不重试
    srv.requests.clear()
    srv.status_seq = deque([422])
    try:
        cli.embed_text("参数", dimension=64)
        ok(False, "422 应抛 WemmHttpError")
    except WemmHttpError as e:
        ok(e.status == 422 and len(srv.requests) == 1, "422 不重试（参数错误重试无意义）")
    srv.status_seq = deque([])

    # 端点不可达：快速失败（≤5s），报错含地址（显式禁用备用，防 env 兜底切走）
    dead = f"http://127.0.0.1:{_unused_port()}/v1/embeddings"
    cli_dead = WEMMClient(endpoint=dead, fallback_endpoint="",
                          use_real_embed=True, probe_timeout=4.0)
    t0 = time.time()
    try:
        cli_dead.embed_text("x", dimension=64)
        ok(False, "不可达端点应报错")
    except RuntimeError as e:
        dt = time.time() - t0
        ok("嵌入端点不可达" in str(e) and dead.split("/v1")[0] in str(e),
           f"报错含「嵌入端点不可达」与端点地址（耗时 {dt:.2f}s）")
        ok(dt <= 5.0, f"快速失败 ≤5s（实际 {dt:.2f}s）")

    # 主端点不可达 → 自动切备用
    srv.requests.clear()
    cli_fb = WEMMClient(endpoint=dead, fallback_endpoint=srv.url, api_key=TEST_KEY,
                        use_real_embed=True, retry_delays=(0.01,))
    v = cli_fb.embed_text("备用", dimension=64)
    ok(v.shape == (64,) and cli_fb._active_endpoint == srv.url,
       "主端点不可达自动切备用端点")


# ---------------- [D] EmbedStore ----------------

def test_store(tmp: str) -> None:
    print("[D] EmbedStore：检索 / 维度锁 / upsert / 过滤 / 删除 / 持久化")
    st = EmbedStore(clip_dim=8, frame_dim=4)

    def vec(seed: int, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    id0 = st.add_clip(vec(0), [vec(10, 4), vec(11, 4)], frame_ts=[0.0, 0.6],
                      meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0,
                            "thumb_path": "x.jpg"})
    st.add_clip(vec(1), [vec(12, 4)], frame_ts=[5.0],
                meta={"video": "a.mp4", "t_start": 5.0, "t_end": 10.0})
    st.add_clip(vec(2), [], meta={"video": "b.mp4", "t_start": 0.0, "t_end": 5.0})
    ok(len(st) == 3 and st.clip_dim == 8 and st.frame_dim == 4,
       "写入 3 片段，库级维度锁定 8/4")
    ok(st.stats() == {"clips": 3, "frames": 3, "clip_dim": 8, "frame_dim": 4,
                      "videos": ["a.mp4", "b.mp4"]}, "stats 条数/维度/视频清单自检")

    hits = st.search(vec(0), top_k=2)
    ok(len(hits) == 2 and hits[0].clip_id == id0 and hits[0].rank == 1
       and hits[0].score > 0.999, "检索 top1 命中自身（点积=余弦）")
    ok(hits[0].meta.get("video") == "a.mp4", "命中携带入库元数据")
    ok([h.clip_id for h in st.search(vec(0), video="b.mp4")] == [2],
       "video 过滤只在指定视频内检索")
    ok(st.search(vec(0), video="c.mp4") == [], "无命中视频返回空")
    ok(st.search(vec(0), top_k=10, thresh=0.99)[0].clip_id == id0
       and len(st.search(vec(0), top_k=10, thresh=0.999999)) == 1,
       "thresh 截断生效")

    fine = st.search_frames(vec(11, 4), id0, top_k=2)
    ok(len(fine) == 2 and fine[0].rank == 1 and abs(fine[0].ts - 0.6) < 1e-6
       and fine[0].clip_id == id0, "frame 精排命中帧向量 + ts 对位")
    ok(st.search_frames(vec(11, 4), 999) == [], "空片段精排返回空")

    try:
        st.add_clip(np.ones(16, dtype=np.float32), meta={"video": "x", "t_start": 0})
        ok(False, "clip 维度不匹配应拒收")
    except ValueError as e:
        ok("16" in str(e) and "8" in str(e), "维度不匹配拒收并指明两侧维度")
    try:
        st.add_clip(vec(3), [np.ones(9, dtype=np.float32)],
                    meta={"video": "x", "t_start": 0})
        ok(False, "frame 维度不匹配应拒收")
    except ValueError as e:
        ok("9" in str(e) and "4" in str(e), "frame 维度不匹配拒收并指明")

    n_before = (len(st), len(st._frame_vecs))
    st.add_clip(vec(0), [vec(13, 4)], frame_ts=[0.3],
                meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0})
    ok(len(st) == n_before[0] and len(st._frame_vecs) == n_before[1] - 1,
       "upsert 幂等：同 (video,t_start) 行数不变，旧 frame 向量清除")
    ok(abs(st.search_frames(vec(13, 4), id0)[0].ts - 0.3) < 1e-6,
       "upsert 后新 frame 向量生效")

    ok(st.delete_video("a.mp4") == 2 and len(st) == 1
       and all(fm["clip_id"] != id0 for fm in st._frame_meta),
       "delete_video 删片段并连带删帧向量")
    ok(st.delete_video("a.mp4") == 0, "重复删除返回 0")

    p = os.path.join(tmp, "vecs.npz")
    st.save(p)
    st2 = EmbedStore().load(p)
    ok(st2.stats() == st.stats(), "save→load roundtrip：条数/维度一致")
    ok(bool(np.allclose(np.stack(st2._clip_vecs), np.stack(st._clip_vecs))),
       "roundtrip 向量矩阵逐值一致")
    ok(st2.search(vec(2), top_k=1)[0].clip_id == 0,
       "roundtrip 后检索语义不变（删除后 id 重映射：b.mp4 → 0）")
    ok(len(st2._frame_meta) == len(st._frame_meta),
       "roundtrip frame 层一致（重映射后 clip_id 引用同步）")


# ---------------- [E] VideoEmbedStage ----------------

def test_stage(tmp: str) -> None:
    print("[E] VideoEmbedStage：窗口聚合 / 记录与事件 / partial / 缩略图 / 自描述")
    ok(VideoEmbedStage.INPUTS == {"frame": "Image", "dets": "Records"}
       and VideoEmbedStage.OUTPUTS == {"clips": "Records"},
       "INPUTS/OUTPUTS 自描述声明（契约 B：dets 第二入端口可缺省）")

    cli = WEMMClient(use_real_embed=False, clip_dimension=128, frame_dimension=64)
    store = EmbedStore(clip_dim=128, frame_dim=64)
    thumb_dir = "thumbs"  # 项目相对：root=tmp 时落 <tmp>/thumbs，写进路径的也是相对路径
    stage = VideoEmbedStage(cli, store, video="t.mp4", frames_per_clip=3,
                            clip_seconds=1.8, thumb_dir=thumb_dir, root=tmp)

    recs: list = []
    for i in range(7):  # 7 帧 → 2 满窗 + 尾 1 帧
        r = stage.run(img(32, 24, tag=i), ts=round(i * 0.6, 2), frame_idx=i)
        if i == 6:
            ok(r == [], "未满窗返回空（余帧留在缓冲）")
        recs.extend(r)
    ok(len(recs) == 2, "7 帧 @ 3 帧/窗 → 前 6 帧出 2 个片段")
    tail = stage.finish()
    ok(len(tail) == 1 and tail[0]["partial"] is True, "finish 冲刷尾段出 partial 片段")
    recs.extend(tail)
    ok(stage.finish() == [], "finish 幂等（空缓冲不再出片段）")
    ok(len(stage.indexed) == 3 and len(store) == 3, "累计 3 片段记录 / 库 3 条")

    r0, r2 = recs[0], recs[2]
    ok(r0["kind"] == "clip" and r0["video"] == "t.mp4"
       and abs(r0["t_start"]) < 1e-9 and abs(r0["t_end"] - 1.2) < 1e-9
       and r0["frame_count"] == 3 and r0["clip_id"] == 0 and r0["ts"] == r0["t_start"],
       "片段记录字段：kind/video/t_start/t_end/frame_count/clip_id/ts")
    ok(r2["t_start"] == 3.6 and r2["frame_count"] == 1 and r2["partial"] is True
       and abs(r2["t_end"] - 3.6) < 1e-9, "尾片段 partial=True 且 t_start/t_end 取余帧首末")
    ok(isinstance(r0.get("embedding"), np.ndarray) and r0["embedding"].shape == (128,),
       "clip 向量走记录旁路字段 embedding")
    mainline = {k: v for k, v in r0.items() if k != "embedding"}
    ok(json.dumps(mainline, ensure_ascii=False) is not None,
       "去旁路字段后记录主干可 JSON 序列化（像素与向量不进 JSON 主干）")
    evs = r0.get("events") or []
    ok(len(evs) == 1 and evs[0]["name"] == "clip_indexed"
       and evs[0]["detail"]["clip_id"] == 0 and evs[0]["detail"]["video"] == "t.mp4",
       "片段记录附 clip_indexed 事件（链路尾 collect_events 可汇总）")
    ok(os.path.isfile(os.path.join(tmp, r0["thumb_path"]))
       and r0["thumb_path"].startswith("thumbs/")
       and not os.path.isabs(r0["thumb_path"]),
       "缩略图落盘且记录里是项目相对路径（§2.5 原则 4：换机不失效）")
    thumb = cv2.imread(os.path.join(tmp, r0["thumb_path"]))
    ok(thumb is not None and max(thumb.shape[:2]) <= stage.thumb_long,
       "缩略图长边 ≤ thumb_long")
    ok(all(not os.path.isabs(m.get("thumb_path") or "")
           for m in store._clip_meta),
       "store meta 的 thumb_path 同为项目相对路径")
    ok(store.stats()["videos"] == ["t.mp4"], "store 元数据 video 一致")

    # 观察段泛化（契约 B）：dets 输入 → 片段记录旁路 observation，标签不进 store meta
    store4 = EmbedStore(clip_dim=128, frame_dim=64)
    stage4 = VideoEmbedStage(cli, store4, video="o.mp4", frames_per_clip=3)
    # fixture 对齐真实记录形态（composer YOLO det 带 model_path；plate/face 带旁路图）
    _im = np.zeros((2, 2, 3), dtype=np.uint8)
    dets_by_frame = [
        [{"kind": "det", "cls_id": 0, "track_id": 1, "label": "person",
          "bbox": (1, 2, 3, 4), "score": 0.9, "model_path": "yolov8n.pt"}],
        [{"kind": "det", "cls_id": 0, "track_id": 1, "label": "person",
          "bbox": (1, 2, 3, 4), "score": 0.8, "model_path": "yolov8n.pt"},
         {"kind": "det", "cls_id": 0, "track_id": 2, "label": "person",
          "bbox": (5, 6, 7, 8), "score": 0.7, "model_path": "yolov8n.pt"},
         {"kind": "plate", "plate_no": "沪DHH230", "label": "沪DHH230",
          "text": "沪DHH230", "color": "blue", "score": 0.8, "plate_img": _im},
         {"kind": "face", "bbox": (0, 0, 2, 2), "score": 0.9, "image": _im}],
        [{"kind": "det", "cls_id": 2, "track_id": 7, "label": "car",
          "bbox": (9, 9, 9, 9), "score": 0.6, "model_path": "yolov8n.pt"}],
    ]
    recs4: list = []
    for i, dt in enumerate(dets_by_frame):
        recs4.extend(stage4.run(img(32, 24, tag=i), ts=round(i * 0.5, 2),
                                frame_idx=i, dets=dt))
    obs = recs4[0].get("observation") or {}
    ok(obs.get("det_stats") == {"0": {"peak": 2, "avg": 1.0, "tracks": 2},
                                "2": {"peak": 1, "avg": 0.333, "tracks": 1}},
       "observation.det_stats：cls_id 键 + 帧级 count 峰值/均值 + track 并集去重")
    ok(obs.get("class_names") == {"0": "person", "2": "car"},
       "observation.class_names 采集自上游记录 label（cls_id → 名称）")
    ok(obs.get("plate_nos") == ["沪DHH230"] and obs.get("face_count") == 1,
       "observation.plate_nos 去重集合 / face_count 取窗口内最大")
    # frame_dets 原始记录流（WO-0033.10，§2.5 原则 3 第四轮）：三桶混存 / 剥键 / 主干原样
    fd = obs.get("frame_dets")
    ok(isinstance(fd, list) and [w["ts"] for w in fd] == [0.0, 0.5, 1.0]
       and [len(w["recs"]) for w in fd] == [1, 4, 1]
       and all("truncated" not in w for w in fd),
       "observation.frame_dets 逐帧分组：dets/plates/faces 三路原始记录、"
       "只含有记录的帧、ts 对位")
    ok([r["kind"] for w in fd for r in w["recs"]]
       == ["det", "det", "det", "plate", "face", "det"],
       "frame_dets kind 多态混存（三路同流，桶序固定 dets→plates→faces）")
    ok(all("model_path" not in r and "image" not in r and "plate_img" not in r
           and "embedding" not in r for w in fd for r in w["recs"]),
       "frame_dets 剥 model_path 与旁路键（image/plate_img 像素、embedding 向量"
       "不进 JSON 主干，§2.5 原则 3）")
    d0 = fd[0]["recs"][0]
    ok(d0["cls_id"] == 0 and d0["track_id"] == 1 and d0["bbox"] == (1, 2, 3, 4)
       and d0["score"] == 0.9 and d0["label"] == "person",
       "frame_dets 记录主干原样（cls_id/track_id/bbox/score/label 保留）")
    pl = fd[1]["recs"][2]
    ok(pl["plate_no"] == "沪DHH230" and pl["label"] == "沪DHH230"
       and pl["text"] == "沪DHH230",
       "plate 记录进 frame_dets，plate_no 载体 label/text 原样保留")
    # 对账（§2.5 原则 3：det_stats 定位降级为派生缓存）——从 frame_dets 重算与缓存一致
    per: dict = {}
    trk: dict = {}
    for w in fd:
        cnt: dict = {}
        for r in w["recs"]:
            if r.get("kind") != "det":
                continue
            k = str(r["cls_id"])
            cnt[k] = cnt.get(k, 0) + 1
            if r.get("track_id"):
                trk.setdefault(k, set()).add(r["track_id"])
        for k, c in cnt.items():
            per.setdefault(k, []).append(c)
    n_fr = recs4[0]["frame_count"]
    re_stats = {k: {"peak": max(v), "avg": round(sum(v) / n_fr, 3),
                    "tracks": len(trk.get(k, ()))} for k, v in per.items()}
    ok(re_stats == obs["det_stats"],
       "对账：frame_dets 重算 peak/avg/tracks 与 det_stats 缓存一致"
       "（avg 分母=窗口帧数 frame_count）")
    ok(json.dumps(obs, ensure_ascii=False) is not None,
       "observation 可 JSON 序列化（frame_dets 原始记录流已剥 model_path/旁路键）")
    meta4 = store4._clip_meta[0]
    ok("observation" not in meta4 and "det_stats" not in meta4
       and set(meta4) == {"video", "t_start", "t_end", "frame_count", "partial",
                          "clip_dim", "frame_dim", "clip_seconds", "thumb_path",
                          "frame_ts"},
       "契约 B：标签不进 store meta，meta 仅窗口骨架 10 键")
    # dets 可缺省：不传 observation 全零 + frame_dets=[]（不缺键），不报错
    store4b = EmbedStore(clip_dim=128, frame_dim=64)
    stage4b = VideoEmbedStage(cli, store4b, video="n.mp4", frames_per_clip=2)
    recs4b = stage4b.index_video([img(tag=0), img(tag=1)])
    ok(recs4b[0]["observation"] == {"det_stats": {}, "class_names": {},
                                    "plate_nos": [], "face_count": 0,
                                    "frame_dets": []},
       "dets 缺省（未连线）→ observation 全零且 frame_dets=[]（不缺键），不报错")

    # frames_embed=False：只嵌 clip 向量，不请求也不存 frame 向量
    store5 = EmbedStore(clip_dim=128)
    stage5 = VideoEmbedStage(cli, store5, video="f.mp4", frames_per_clip=2,
                             frames_embed=False)
    recs5 = stage5.index_video([img(tag=0), img(tag=1)])
    ok(len(recs5) == 1 and len(store5._clip_vecs) == 1
       and len(store5._frame_vecs) == 0 and store5.stats()["frames"] == 0,
       "frames_embed=False → 只存 clip 向量、frame 层 0 条（查询仍可用）")
    ok(recs5[0]["embedding"].shape == (128,)
       and recs5[0]["thumb_path"] == "",
       "frames_embed=False 片段记录照常产出（无 thumb_dir → 路径空串）")

    # index_video 集成 + flush_partial=False
    store2 = EmbedStore(clip_dim=128, frame_dim=64)
    stage2 = VideoEmbedStage(cli, store2, video="u.mp4", frames_per_clip=4,
                             flush_partial=False)
    recs2 = stage2.index_video([img(tag=i) for i in range(6)],
                               ts_list=[i * 0.5 for i in range(6)])
    ok(len(recs2) == 1 and len(store2) == 1, "index_video 6 帧 @4/窗 + flush_partial=False → 1 片段")
    ok(recs2[0]["t_start"] == 0.0 and recs2[0]["t_end"] == 1.5,
       "ts_list 时间戳承载片段时间（首窗取前 4 帧 ts 0~1.5，余帧丢弃）")
    try:
        VideoEmbedStage(cli, store2, frames_per_clip=0)
        ok(False, "frames_per_clip≤0 应报错")
    except ValueError:
        ok(True, "frames_per_clip≤0 fail fast")

    # 检索闭环（mock 档）：同片段帧序重建查询向量可回检该片段
    frames = [img(32, 24, tag=i) for i in range(3)]
    cv_, fv_ = cli.embed_frames(frames, clip_dimension=128, frame_dimension=64)
    store3 = EmbedStore(clip_dim=128, frame_dim=64)
    cid = store3.add_clip(cv_, fv_, frame_ts=[0.0, 0.6, 1.2],
                          meta={"video": "c.mp4", "t_start": 0.0})
    hit = store3.search(cv_, top_k=1)[0]
    fine = store3.search_frames(fv_[2], cid, top_k=1)[0]
    ok(hit.clip_id == cid and abs(fine.ts - 1.2) < 1e-6,
       "入库→粗排→精排闭环：clip 命中 + 帧级精定位 ts=末帧")
    ok(abs(store3.search(cli.embed_text("任意文本", dimension=128), top_k=1)[0].score)
       < 1.0, "跨源查询（文本 vs 片段）可计算余弦（mock 向量不同族）")


# ---------------- [F] label_align（契约 A 纯逻辑） ----------------

def test_label_align() -> None:
    print("[F] label_align：采集时间 / 窗口 / 聚合 / 事件对齐 / 片段标签")

    # --- parse_video_time：文件名优先 > tag；失败 ("","") 不抛错 ---
    cases = [
        ("cam1_20261029_101500.mp4", "", "2026-10-29T10:15:00+08:00", "filename",
         "紧凑命名 YYYYMMDD_HHMMSS"),
        ("2026-10-29_10-15-00.mp4", "", "2026-10-29T10:15:00+08:00", "filename",
         "分隔符命名 YYYY-MM-DD_HH-MM-SS"),
        ("20261029101500.mp4", "", "2026-10-29T10:15:00+08:00", "filename",
         "纯数字 14 位"),
        ("2026-10-29T10:15:00+08:00_x.mp4", "", "2026-10-29T10:15:00+08:00",
         "filename", "ISO8601 含时区"),
        ("20261029.mp4", "", "2026-10-29T00:00:00+08:00", "filename", "仅日期按 00:00"),
        ("clip.mp4", "", "", "", "文件名解析不出且无 tag → 双空"),
        ("clip.mp4", "2026-10-29T10:15:00+08:00", "2026-10-29T10:15:00+08:00",
         "tag", "文件名解析不出 → 取 tag"),
        ("20261029_101500.mp4", "2020-01-01T00:00:00+08:00",
         "2026-10-29T10:15:00+08:00", "filename", "来源优先级：文件名 > tag"),
        ("/abs/path/2026-13-45_99-99-99.mp4", "", "", "", "非法月日时分 → 双空不抛错"),
    ]
    for fname, tag, exp_t, exp_src, desc in cases:
        got = parse_video_time(fname, tag)
        ok(got == (exp_t, exp_src), f"parse_video_time {desc} → {got}")
    for bad in (None, "", 12345, object()):
        ok(parse_video_time(bad, bad) == ("", ""),
           f"parse_video_time 非法输入 {type(bad).__name__} → ('','') 不抛错")

    # --- clip_windows：嵌套/帧数切窗 + 右开左闭边界 ---
    nested = [[0, 0.6, 1.2], [1.8, 2.4, 3.0]]
    ok(clip_windows(nested) == [(0.0, 1.8), (1.8, 3.6)],
       "clip_windows 嵌套（片段 frame_ts 列表）→ 窗口右开左闭、末窗按帧间隔补端")
    flat = [0, 0.6, 1.2, 1.8, 2.4, 3.0]
    ok(clip_windows(flat, frames_per_clip=3) == [(0.0, 1.8), (1.8, 3.6)],
       "clip_windows 扁平 + frames_per_clip 与索引切窗一致")
    ok(clip_windows([]) == [] and clip_windows([1.0], clip_seconds=2.0) == [(1.0, 3.0)],
       "clip_windows 空输入 → []，单帧窗用 clip_seconds 撑开")
    ok(clip_windows([0.0, 0.5, 1.0]) == [(0.0, 1.5)],
       "clip_windows 连续均匀帧无帧数 → 单窗（需精确切窗请传 frames_per_clip/嵌套）")

    # --- align_events：边界归下一窗、跨窗各记一条、ts 保留原值 ---
    wins = [(0.0, 5.0), (5.0, 10.0)]
    evs = [{"name": "a", "kind": "警情", "ts": 2.5},
           {"name": "b", "kind": "研判", "ts": 5.0},
           {"name": "c", "kind": "警情", "ts": 10.0},
           {"name": "d", "kind": "警情"}]
    rows = align_events(evs, wins)
    ok(len(rows) == 4 and rows[0]["window"] == [0.0, 5.0]
       and rows[1]["window"] == [5.0, 10.0] and rows[1]["ts"] == 5.0,
       "align_events 右开左闭：ts=5.0 落下一窗 [5,10)，ts 原值保留")
    ok(rows[2]["window"] is None and rows[3]["window"] is None,
       "align_events 窗外交界 ts=10.0 / 无 ts 事件保留并标 window=None（不静默丢弃）")
    cross = align_events([{"name": "x", "kind": "警情", "ts": 4.5}],
                         [(0.0, 5.0), (4.0, 6.0)])
    ok(len(cross) == 2 and cross[0]["ts"] == 4.5 and cross[1]["ts"] == 4.5
       and {tuple(r["window"]) for r in cross} == {(0.0, 5.0), (4.0, 6.0)},
       "align_events 跨窗事件按命中窗口各记一条（ts 保留原值）")

    # --- aggregate_frame_stats：cls_id 聚合 / track 并集 / 集合去重 / frame_dets / 空输入 ---
    # 记录带 kind（records 契约必备字段；真实 det/plate/face 记录形态，见 composer/
    # plate_recognizer/face_pipeline 适配器），frame_dets 主干原样透传
    per_frame = [
        {"ts": 0.0,
         "dets": [{"kind": "det", "cls_id": 0, "track_id": 1, "label": "person",
                   "model_path": "yolov8n.pt"},
                  {"kind": "det", "cls_id": 0, "track_id": 2, "label": "person"}],
         "plates": [{"kind": "plate", "plate_no": "沪DHH230",
                     "plate_img": b"\xff\x00"}],
         "faces": [{"kind": "face", "score": 0.9}]},
        {"ts": 0.6,
         "dets": [{"kind": "det", "cls_id": 0, "track_id": 1, "label": "person"},
                  {"kind": "det", "cls_id": 0, "track_id": 0, "label": "person"}],
         "plates": [{"kind": "plate", "plate_no": "沪DHH230"},
                    {"kind": "plate", "label": "沪A11111"}],
         "faces": [{"kind": "face", "score": 0.9}, {"kind": "face", "score": 0.8}]},
        {"ts": 1.2, "dets": [{"kind": "det", "cls_id": 2, "track_id": 7, "label": "car"}]},
    ]
    st = aggregate_frame_stats(per_frame)
    ok(st["det_stats"] == {"0": {"peak": 2, "avg": 1.333, "tracks": 2},
                           "2": {"peak": 1, "avg": 0.333, "tracks": 1}},
       f"aggregate 帧级 count 峰值/均值 + track 并集去重（track_id=0 不计）: {st['det_stats']}")
    ok(st["class_names"] == {"0": "person", "2": "car"}, "aggregate class_names 取 label")
    ok(st["plate_nos"] == ["沪A11111", "沪DHH230"], "aggregate plate_nos 集合去重（含 label 回退）")
    ok(st["face_count"] == 2, "aggregate face_count 取窗口内最大")
    # frame_dets 原始记录流（§2.5 原则 3 第四轮）：混存 / 剥键 / 对账
    fd = st["frame_dets"]
    ok([w["ts"] for w in fd] == [0.0, 0.6, 1.2]
       and [len(w["recs"]) for w in fd] == [4, 6, 1],
       "aggregate frame_dets 逐帧分组：三桶记录全收、ts 对位、recs 数与上游一致")
    ok([r["kind"] for w in fd for r in w["recs"]]
       == ["det", "det", "plate", "face",
           "det", "det", "plate", "plate", "face", "face", "det"],
       "aggregate frame_dets kind 多态混存（det/plate/face 同流，桶序固定）")
    ok(all("model_path" not in r and "plate_img" not in r and "image" not in r
           for w in fd for r in w["recs"]),
       "aggregate frame_dets 剥 model_path 与旁路图键（plate_img 别名同样剥）")
    ok(json.dumps(st, ensure_ascii=False) is not None,
       "aggregate frame_dets 可 JSON 序列化（旁路 ndarray/bytes 已剥）")
    re_per: dict = {}
    re_trk: dict = {}
    for w in fd:
        cnt: dict = {}
        for r in w["recs"]:
            if r.get("kind") != "det":
                continue
            k = str(r["cls_id"])
            cnt[k] = cnt.get(k, 0) + 1
            if r.get("track_id"):
                re_trk.setdefault(k, set()).add(r["track_id"])
        for k, c in cnt.items():
            re_per.setdefault(k, []).append(c)
    ok({k: {"peak": max(v), "avg": round(sum(v) / len(per_frame), 3),
            "tracks": len(re_trk.get(k, ()))} for k, v in re_per.items()}
       == st["det_stats"],
       "对账：frame_dets 重算 peak/avg/tracks 与 det_stats 一致（avg 分母=len(per_frame)）")
    # 空帧不进 frame_dets；乱序入参按 ts 排序产出（稳定，同 ts 保原序）
    gap = aggregate_frame_stats([
        {"ts": 1.0, "dets": [{"cls_id": 0}]},
        {"ts": 0.5},                      # 无记录帧：跳过
        {"ts": 0.0, "dets": [{"cls_id": 0}]},
    ])["frame_dets"]
    ok([w["ts"] for w in gap] == [0.0, 1.0],
       "frame_dets 跳过无记录帧（空观察源/全空窗 → []）且按 ts 顺序产出")
    empty = aggregate_frame_stats([])
    ok(empty == {"det_stats": {}, "class_names": {}, "plate_nos": [],
                 "face_count": 0, "frame_dets": []},
       "aggregate 空输入 → 五键全零（frame_dets=[] 不缺键）")
    # 单窗上限保护：>500 截断到 500，按 ts 顺序保留靠前帧，截断发生帧附 truncated
    big = [
        {"ts": 0.0, "dets": [{"cls_id": 0}] * 2},
        {"ts": 0.5, "dets": [{"cls_id": 0}] * 499},
        {"ts": 1.0, "dets": [{"cls_id": 1}] * 5},
    ]
    big_st = aggregate_frame_stats(big)
    fdt = big_st["frame_dets"]
    ok([len(w["recs"]) for w in fdt] == [2, 498]
       and sum(len(w["recs"]) for w in fdt) == 500
       and fdt[-1].get("truncated") is True
       and all("truncated" not in w for w in fdt[:-1]),
       "frame_dets 单窗 >500 截断到 500：保留靠前帧、截断发生帧附 truncated:true、"
       "后续帧丢弃")
    ok(big_st["det_stats"].get("1", {}).get("peak") == 5,
       "截断只影响 frame_dets：det_stats 口径不动（按全窗输入统计，cls 1 peak=5 仍在）")
    edge = aggregate_frame_stats([{"ts": 0.0, "dets": [{"cls_id": 0}] * 500}])["frame_dets"]
    ok(len(edge) == 1 and len(edge[0]["recs"]) == 500
       and "truncated" not in edge[0],
       "frame_dets 恰 500 条不截断（上限含边界）")

    # --- build_clip_tags：§2.5 全部字段（向量三列/缩略图由调用方补） ---
    obs = aggregate_frame_stats(per_frame)
    aligned = align_events([
        {"name": "斗殴", "kind": "警情", "ts": 0.6, "detail": {"n": 2}},
        {"name": "沪DHH230", "kind": "车牌", "ts": 1.0, "plate_no": "沪DHH230",
         "color": "blue", "score": 0.8},
        {"name": "窗外", "kind": "警情", "ts": 99.0},
    ], [(0.0, 5.0)])
    tags = build_clip_tags("x.mp4", (0.0, 5.0), obs, aligned)
    ok(set(tags) == {"video", "t_start", "t_end", "det_stats", "frame_dets",
                     "class_names", "plate_nos", "plates", "events", "face_count",
                     "description", "thumb_path"},
       f"build_clip_tags 出 §2.5 全部字段（向量三列由调用方补）: {sorted(tags)}")
    ok(tags["frame_dets"] == obs["frame_dets"]
       and tags["frame_dets"] is not obs["frame_dets"],
       "build_clip_tags frame_dets 透传进标签 dict（§2.5 原则 3 第四轮；窗口层浅拷贝隔离）")
    ok(tags["video"] == "x.mp4" and tags["t_start"] == 0.0 and tags["t_end"] == 5.0
       and tags["face_count"] == 2 and tags["thumb_path"] == ""
       and tags["description"] == "" and tags["plate_nos"] == ["沪A11111", "沪DHH230"],
       "build_clip_tags 窗口/观察层字段对位（thumb_path/description 留空待调用方补）")
    ok(tags["plates"] == [{"plate_no": "沪DHH230", "color": "blue", "score": 0.8,
                           "ts": 1.0}],
       "build_clip_tags 判断层车牌进 plates[]（带属性与 ts，不重复进 events）")
    ok(len(tags["events"]) == 1 and tags["events"][0] ==
       {"name": "斗殴", "kind": "警情", "ts": 0.6, "detail": {"n": 2}},
       "build_clip_tags 判断层警情进 events[]（name/kind/ts/detail），窗外事件不混入")

    # --- 接缝②双保险（WO-0033.2）：kind=车牌 但无 plate_no 键 → plates[] ---
    # 车牌号载体回退顺序 plate_no → label → text → name（name 仅 kind=车牌 时兜底，
    # 非车牌事件 name 是事件名不参与，防警情误判成车牌）。
    aligned2 = align_events([
        {"name": "沪A12345", "kind": "车牌", "ts": 1.0},          # web 归档原始形态：只落 name
        {"kind": "车牌", "label": "沪B22222", "ts": 2.0},          # 管道 plate 记录形态：载体 label
        {"kind": "车牌", "text": "沪C33333", "ts": 3.0},           # 载体 text
        {"name": "斗殴", "kind": "警情", "ts": 0.6},               # 非车牌：name 不兜底
    ], [(0.0, 5.0)])
    tags2 = build_clip_tags("x.mp4", (0.0, 5.0), obs, aligned2)
    ok(tags2["plates"] == [{"plate_no": "沪A12345", "ts": 1.0},
                           {"plate_no": "沪B22222", "ts": 2.0},
                           {"plate_no": "沪C33333", "ts": 3.0}],
       "build_clip_tags kind=车牌 无 plate_no 键也进 plates[]（plate_no 依次回退 "
       "name/label/text；name 兜底仅限车牌事件）")
    ok(len(tags2["events"]) == 1 and tags2["events"][0]["name"] == "斗殴",
       "build_clip_tags 车牌事件不重复进 events[]，非车牌事件路由不受接缝②影响")

    ok(json.dumps(tags, ensure_ascii=False) is not None,
       "build_clip_tags 结果可 JSON 序列化（可直接落 review_index 行）")


# ---------------- --live：真端点冒烟（wemm_integration.md §7） ----------------

def test_live() -> None:
    print("[LIVE] 真端点冒烟")
    ep, _fb, model, key = resolve_wemm_cfg()
    if not ep:
        print("  [SKIP] 未配置嵌入端点（请 source wemm接口/REMOTE_CLIENT.env 导出 "
              "WEMM_BASE_URL / WEMM_API_KEY）")
        return
    if not endpoint_reachable(ep if "/v1/" in ep else ep, timeout=5.0):
        print(f"  [SKIP] 嵌入端点不在线: {ep}（TCP 探测 ≤5s 失败；服务端未启动属合法结果）")
        return
    print(f"  端点在线: {ep} | model={model or '(默认)'} | key={'已配置' if key else '无'}")
    cli = WEMMClient(use_real_embed=True, timeout=120)

    v = cli.embed_text("人群聚集", dimension=1024)
    ok_live(v.shape == (1024,) and abs(float(np.linalg.norm(v)) - 1.0) < 0.05,
            "文本嵌入：维度=服务端截维档位 1024 且 L2≈1（点积=余弦）")
    v2 = cli.embed_text("人群聚集", dimension=1024)
    ok_live(bool(np.allclose(v, v2, atol=1e-4)), "同文本两次嵌入确定性一致")

    v256 = cli.embed_text("人群聚集", dimension=256)
    ok_live(v256.shape == (256,) and abs(float(np.linalg.norm(v256)) - 1.0) < 0.05,
            "服务端 MRL 截维 256：维度与 L2 正确（非本地前缀截断）")

    vi = cli.embed_image(img(640, 360, tag=7), dimension=256)
    ok_live(vi.shape == (256,) and abs(float(np.linalg.norm(vi)) - 1.0) < 0.05,
            "单图嵌入：维度 256 且 L2≈1")

    frames = [img(896, 504, tag=i) for i in range(3)]
    cv_, fvs = cli.embed_frames(frames)  # 默认 clip=1024 / frame=256
    ok_live(cv_.shape == (1024,) and len(fvs) == 3
            and all(f.shape == (256,) for f in fvs)
            and abs(float(np.linalg.norm(cv_)) - 1.0) < 0.05,
            "clip 多图消息（3 帧→1 向量@1024）+ 逐帧单图（3 向量@256），L2≈1")

    d_cross = float(cv_ @ cli.embed_text("画面", dimension=1024))
    print(f"  [info] 跨模态 sanity：clip·文本 = {d_cross:.3f}"
          f"（同域文本相似度普遍偏高，按相对排序不按绝对值——实测 §6）")
    ok_live(-1.0 <= d_cross <= 1.0, "跨模态余弦在 [-1,1] 合理域内")


# ---------------- main ----------------

def main() -> None:
    tmp = tempfile.mkdtemp(prefix="video_embed_test_")
    srv = _FakeWemmServer()
    try:
        test_mock_and_dims(srv)
        test_http_schema(srv)
        test_store(tmp)
        test_stage(tmp)
        test_label_align()
    finally:
        srv.close()
    print(f"\n[mock 档] 全部 {PASS} 项断言通过 ✓")
    if LIVE:
        test_live()
        if PASS_LIVE:
            print(f"[live] 真端点冒烟 {PASS_LIVE} 项断言通过 ✓")
    print("done.")


if __name__ == "__main__":
    main()
