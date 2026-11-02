# -*- coding: utf-8 -*-
"""
WeMM 嵌入客户端（WEMMClient）——文 / 图 / 视频帧统一向量化器（PIPE0004 核心依赖）

对接 WeMM-Embedding 服务（vLLM，OpenAI 兼容 /v1/embeddings），设计依据：
  doc/design/wemm_integration.md §4（客户端对接）/ §6（2026-10-28 联调实测）。
WO-0030 实测修正（本实现**必须**遵守的请求形态）：
  - 纯文本：{"model", "input": <str|[str]>}，input 数组批量 → n 向量顺序对位；
  - **多模态走 messages**（不用 input）：{"model", "messages":[{"role","content":[...]}]}，
    content 项 = {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}} /
    {"type":"text","text":...}；
  - **一条请求只放一组 messages**（多组行为未验证，禁止依赖）——clip 多图消息 +
    逐帧单图消息 = 多次请求，客户端线程池并发 1–4；
  - **MRL 截维统一走服务端参数 dimensions**（实测 ≠ 本地前 N 维截断，cos≈0.50，
    混用 = 检索错乱）；输出已 L2 归一（点积=余弦），客户端只做模长复验告警；
  - 鉴权 Authorization: Bearer <key>；401/404/422 等 4xx 不重试，5xx/网络错误
    指数退避重试；timeout 连接 5s + 读取 embed_timeout（含图请求较慢）。
  - Key 只走环境变量 / 显式参数，**永不打印、永不入异常信息、不入预设**。

mock 档（use_real_embed=False 或端点为空）：确定性伪向量（同输入同向量、按维度档截形、
L2 归一），复刻 use_real_vlm 先例——服务不在线也能全流程开发测试。

用法:
    cli = WEMMClient()                                # env 自动解析；无 env → mock 档
    v = cli.embed_text("人群聚集", dimension=1024)     # (1024,) 归一化向量
    c, fs = cli.embed_frames(frames)                  # (clip 向量, [每帧向量])

依赖: numpy, opencv-python, requests（项目已在用：plate_recog/fetch_models、web_lab/server）
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

# ---------------- 常量（wemm_integration.md §2.2 / §4.5 参数表） ----------------

# Matryoshka 截维档位（WeMM-Embedding-2B；换 4B/9B 型号时以 /v1/models 与服务端为准）
WEMM_MRL_DIMS: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
DEFAULT_WEMM_MODEL = "wemm-embedding"
DEFAULT_CLIP_DIM = 1024   # clip 向量维度（§4.5 默认档，2B@256 保留 98.7%，1024 更稳）
DEFAULT_FRAME_DIM = 256   # frame 向量维度（精排只比相对远近，低维足够）
MOCK_FULL_DIM = 2048      # mock 伪向量全维（按档位前缀截形再归一，模拟真实 MRL 行为）
EMBEDDINGS_PATH = "/v1/embeddings"

# 服务端限制（wemm接口/REMOTE_CLIENT.md §6 / 实测 §6）
PROBE_TIMEOUT_S = 4.0     # TCP 探测上限（任务要求 ≤5s 快速失败）
CONNECT_TIMEOUT_S = 5.0   # HTTP 连接阶段超时（快速失败）


class WemmHttpError(RuntimeError):
    """WeMM 端点 HTTP 错误（可读中文提示；status=0 表示网络层错误。Key 永不入消息）。"""

    def __init__(self, status: int, message: str, hint: str = ""):
        super().__init__(message)
        self.status = status
        self.hint = hint


def check_dimension(dim) -> int:
    """Matryoshka 档位校验：非法 → ValueError 并**列出合法档位**（§2.2 约束）。"""
    try:
        d = int(dim)
    except (TypeError, ValueError):
        raise ValueError(f"向量维度非法: {dim!r}（须为整数档位）")
    if d not in WEMM_MRL_DIMS:
        raise ValueError(f"向量维度 {d} 不在 Matryoshka 档位内，合法档位: {list(WEMM_MRL_DIMS)}")
    return d


def resolve_wemm_cfg(endpoint=None, model=None, api_key=None, fallback=None) -> tuple:
    """端点/模型/key 解析：显式参数 → 环境变量 → 默认。

    env：WEMM_BASE_URL（兼容别名 WEMM_ENDPOINT）/ WEMM_BASE_URL_FALLBACK /
    WEMM_MODEL（默认 wemm-embedding）/ WEMM_API_KEY。
    **Key 只经环境变量或显式参数传入**，不落代码/预设/日志。
    显式传参（含空串）覆盖同名 env（如 fallback_endpoint="" 显式禁用备用）；
    传 None / 不传 → 取 env。
    返回 (endpoint, fallback, model, api_key)，endpoint 可为空串（→ mock 档）。
    """
    endpoint = str(endpoint).strip() if endpoint is not None else \
        (os.environ.get("WEMM_BASE_URL")
         or os.environ.get("WEMM_ENDPOINT") or "").strip()
    fallback = str(fallback).strip() if fallback is not None else \
        (os.environ.get("WEMM_BASE_URL_FALLBACK") or "").strip()
    model = str(model).strip() if model is not None else \
        (os.environ.get("WEMM_MODEL") or "").strip() or DEFAULT_WEMM_MODEL
    api_key = str(api_key).strip() if api_key is not None else \
        (os.environ.get("WEMM_API_KEY") or "").strip()
    return endpoint, fallback, model, api_key


def normalize_embeddings_url(endpoint: str) -> str:
    """端点地址归一：base 地址（无路径）→ 拼 /v1/embeddings；完整地址原样。"""
    ep = str(endpoint or "").strip()
    u = urlparse(ep)
    if not u.scheme or not u.netloc:
        raise ValueError(
            f"嵌入端点地址非法: {ep!r}（应形如 http://<ip>:<port> 或 .../v1/embeddings）")
    if not u.path or u.path == "/":
        return ep.rstrip("/") + EMBEDDINGS_PATH
    return ep


def endpoint_reachable(url: str, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """TCP 快速探测（复用 web_lab/server.py `_endpoint_reachable` 逻辑风格）：
    避免长超时把整轮调用拖住；https 默认 443、http 默认 80；无 host 视为不可达。"""
    import socket
    try:
        u = urlparse(url)
        if not u.hostname:
            return False
        port = u.port if u.port is not None else (443 if u.scheme == "https" else 80)
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except Exception:
        return False


# ---------------- 确定性伪向量（mock 档） ----------------

def _mock_full_vec(key: bytes) -> np.ndarray:
    """输入字节 → sha256 种子 → 全维正态向量 → L2 归一（同输入恒同向量）。"""
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(MOCK_FULL_DIM).astype(np.float32)
    return v / (float(np.linalg.norm(v)) or 1.0)


def _mock_vec(key: bytes, dim: int) -> np.ndarray:
    """按档位前缀截形 + 重新归一（不同档位共享同一前缀，与真实 MRL 行为同形）。"""
    v = _mock_full_vec(key)[:dim]
    return (v / (float(np.linalg.norm(v)) or 1.0)).astype(np.float32)


def _img_key(img) -> bytes:
    """图像 → 稳定哈希键（形状 + 像素字节；同内容同布局同向量）。"""
    a = np.asarray(img)
    return b"img\0" + repr(a.shape).encode("utf-8") + b"\0" + a.tobytes()


# ---------------- WEMMClient ----------------

class WEMMClient:
    """WeMM-Embedding 客户端：embed_text / embed_image / embed_frames。

    参数:
        endpoint       : OpenAI 兼容 /v1/embeddings 地址（base 地址自动拼路径）；
                         env WEMM_BASE_URL（别名 WEMM_ENDPOINT）优先。空 → mock 档。
        model          : 模型名（env WEMM_MODEL 优先，默认 wemm-embedding）
        api_key        : Bearer 凭据（env WEMM_API_KEY 优先；**永不打印/入异常**）
        timeout        : 读取超时秒（含图请求慢，默认 120；连接阶段恒 5s）
        clip_dimension : clip 向量默认维度（档位内，默认 1024）
        frame_dimension: frame 向量默认维度（默认 256）
        max_resolution : 帧编码前缩放上限 "宽,高"（默认 "896,896"，§4.5；长边实测 ≤1280）
        jpeg_quality   : 帧 JPEG 质量（默认 85）
        concurrency    : embed_frames 并发请求数（服务端建议 1–4，默认 2，越界收敛）
        use_real_embed : None=按端点有无自动（对齐 use_real_vlm 先例）；True=强制真端点
                         （端点缺失 fail fast）；False=强制 mock
        retry_delays   : 5xx/网络错误的退避间隔秒序列（默认 (1.0, 2.0) 即最多 3 次）
        fallback_endpoint: 备用端点（env WEMM_BASE_URL_FALLBACK）；主端点探测失败自动切换
    """

    def __init__(self, endpoint=None, model=None, api_key=None, timeout=120,
                 clip_dimension: int = DEFAULT_CLIP_DIM,
                 frame_dimension: int = DEFAULT_FRAME_DIM,
                 max_resolution: str = "896,896", jpeg_quality: int = 85,
                 concurrency: int = 2, use_real_embed: Optional[bool] = None,
                 retry_delays: Sequence[float] = (1.0, 2.0, 4.0),
                 fallback_endpoint=None, probe_timeout: float = PROBE_TIMEOUT_S):
        endpoint, fallback, model, api_key = resolve_wemm_cfg(
            endpoint, model, api_key, fallback_endpoint)
        self.endpoint = normalize_embeddings_url(endpoint) if endpoint else ""
        self.fallback_endpoint = normalize_embeddings_url(fallback) if fallback else ""
        self.model = model
        self._api_key = api_key          # 私有持有：任何日志/异常/repr 均不出现
        self.timeout = float(timeout)
        self.max_resolution = str(max_resolution)
        self.jpeg_quality = int(jpeg_quality)
        self.retry_delays = tuple(float(x) for x in retry_delays)
        self.probe_timeout = float(probe_timeout)
        # mock 档：use_real_embed=False **或端点为空**（wemm_integration.md §4.4）
        if use_real_embed is None:
            self.use_real_embed = bool(self.endpoint)
        else:
            self.use_real_embed = bool(use_real_embed)
            if self.use_real_embed and not self.endpoint:
                raise ValueError(
                    "use_real_embed=True 但未配置嵌入端点（传 endpoint 或导出 WEMM_BASE_URL）")
        # 默认维度即档位校验（非法 fail fast，报错含合法档位）
        self.clip_dimension = check_dimension(clip_dimension)
        self.frame_dimension = check_dimension(frame_dimension)
        # 并发 1–4（实测 §6④：多模态批量靠客户端并发；越界收敛不报错）
        self._concurrency = max(1, min(4, int(concurrency)))
        self._reachable_ok = False     # 探测缓存：通过一次后跳过（掉线由 HTTP 重试/报错兜底）
        self._active_endpoint = self.endpoint
        self._lock = threading.Lock()
        mode = (f"real endpoint={self._active_endpoint or '(未配置)'}"
                if self.use_real_embed else "mock（确定性伪向量）")
        print(f"[WEMMClient] mode={mode} | model={self.model} | "
              f"clip_dim={self.clip_dimension} frame_dim={self.frame_dimension} | "
              f"concurrency={self._concurrency}")

    @property
    def concurrency(self) -> int:
        """embed_frames 实际并发（构造值收敛到 [1,4]）。"""
        return self._concurrency

    @property
    def mock(self) -> bool:
        """是否 mock 档（确定性伪向量）。"""
        return not self.use_real_embed

    # ---------------- 公开方法 ----------------

    def embed_text(self, text, dimension: Optional[int] = None) -> np.ndarray:
        """文本 → 归一化向量（查询侧主入口；dimension 缺省 = clip_dimension，
        **必须与入库 clip 向量同维度**——入库查询同参，wemm_integration.md §2.2）。"""
        dim = check_dimension(self.clip_dimension if dimension is None else dimension)
        text = "" if text is None else str(text)
        if not text.strip():
            raise ValueError("embed_text: 输入文本为空")
        if self.mock:
            return _mock_vec(b"text\0" + text.encode("utf-8"), dim)
        vecs = self._post(self._text_body(text, dim))
        if len(vecs) != 1:
            raise WemmHttpError(0, f"embed_text: 响应向量数 {len(vecs)} ≠ 1（input 单条应返回 1 个）")
        self._check_l2(vecs[0], "embed_text")
        return vecs[0]

    def embed_image(self, img, dimension: Optional[int] = None) -> np.ndarray:
        """单图 → 归一化向量（以图查片段的查询侧同款）。"""
        dim = check_dimension(self.clip_dimension if dimension is None else dimension)
        if img is None:
            raise ValueError("embed_image: 输入图像为空")
        if self.mock:
            return _mock_vec(_img_key(img), dim)
        item = self._image_content_item(self._encode_frames([img])[0])
        vecs = self._post(self._multimodal_body([item], dim))
        if len(vecs) != 1:
            raise WemmHttpError(0, f"embed_image: 响应向量数 {len(vecs)} ≠ 1（一条消息一个向量）")
        self._check_l2(vecs[0], "embed_image")
        return vecs[0]

    def embed_frames(self, frames, clip_dimension: Optional[int] = None,
                     frame_dimension: Optional[int] = None,
                     with_frames: bool = True
                     ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """片段多帧 → (clip 向量, [每帧向量])。

        请求形态（实测 §6③④）：clip = N 帧合**一条** messages（一条消息一个向量）；
        每帧再单独一条单图消息出 frame 向量——共 N+1 次请求，线程池并发（默认 2，≤4），
        `ThreadPoolExecutor.map` 保序 → 返回与帧顺序对位。帧编码一次、两类消息共用。

        `with_frames=False`（PIPE0004 `frames_embed` 开关）：**只请求 clip 向量**，
        返回 `(clip 向量, [])`——省掉每片段 N 次请求（索引成本大头），
        查询仍可用（定位到 5s 窗；frame 精排不可用）。
        """
        cd = check_dimension(self.clip_dimension if clip_dimension is None else clip_dimension)
        fd = check_dimension(self.frame_dimension if frame_dimension is None else frame_dimension)
        frames = list(frames)
        if not frames:
            raise ValueError("embed_frames: 需要至少一帧")
        if self.mock:
            h = hashlib.sha256(b"clip\0")
            for f in frames:  # 顺序敏感：同帧序同片段向量
                h.update(hashlib.sha256(_img_key(f)).digest())
            clip_vec = _mock_vec(h.digest(), cd)
            if not with_frames:
                return clip_vec, []
            return (clip_vec,
                    [_mock_vec(_img_key(f), fd) for f in frames])
        jpegs = self._encode_frames(frames)
        clip_body = self._multimodal_body(
            [self._image_content_item(j) for j in jpegs], cd)
        if not with_frames:
            clip_vec = self._post(clip_body)[0]
            self._check_l2(clip_vec, "embed_frames.clip")
            return clip_vec, []
        frame_bodies = [self._multimodal_body([self._image_content_item(j)], fd)
                        for j in jpegs]
        with ThreadPoolExecutor(max_workers=min(self._concurrency, len(frame_bodies) + 1)) as ex:
            results = list(ex.map(self._post, [clip_body] + frame_bodies))
        clip_vec = results[0][0]
        frame_vecs = [r[0] for r in results[1:]]
        self._check_l2(clip_vec, "embed_frames.clip")
        return clip_vec, frame_vecs

    # ---------------- 请求构造（实测 §4.2 schema） ----------------

    @staticmethod
    def _image_content_item(jpeg: bytes) -> dict:
        b64 = base64.b64encode(jpeg).decode("ascii")
        return {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def _multimodal_body(self, items: List[dict], dim: int) -> dict:
        # 多模态走 messages 透传（实测：不用 input）；一条请求只放一组 messages；
        # 服务端 MRL 截维：请求带 dimensions（全链路统一服务端截维）。
        return {"model": self.model,
                "messages": [{"role": "user", "content": items}],
                "dimensions": dim}

    def _text_body(self, text: str, dim: int) -> dict:
        # 纯文本走 input（实测：数组批量 → n 向量顺序对位；查询侧单条用标量）
        return {"model": self.model, "input": text, "dimensions": dim}

    # ---------------- 帧编码 ----------------

    def _parse_max_resolution(self) -> Tuple[int, int]:
        try:
            w, h = (int(x) for x in str(self.max_resolution).split(","))
            return max(1, w), max(1, h)
        except Exception:
            raise ValueError(f'max_resolution 非法: {self.max_resolution!r}（应为 "宽,高"）')

    def _encode_frames(self, frames) -> List[bytes]:
        """帧 → 缩放（max_resolution 上限，等比、只缩不放）+ JPEG 编码字节。"""
        mr_w, mr_h = self._parse_max_resolution()
        out: List[bytes] = []
        for f in frames:
            if f is None:
                raise ValueError("嵌入输入包含空帧（None）")
            h, w = f.shape[:2]
            scale = min(mr_w / w, mr_h / h)
            if scale < 1.0:
                f = cv2.resize(f, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError("帧转 JPEG 失败")
            out.append(buf.tobytes())
        return out

    # ---------------- HTTP 发送（探测 / 重试 / 鉴权） ----------------

    def _ensure_reachable(self) -> None:
        """TCP 快速探测（≤ probe_timeout，默认 4s）：不可达快速报错并含端点地址。
        主端点不可达且有备用 → 自动切换备用（实测 §6：域名/IP 双通备用位）。
        探测通过后缓存跳过（服务中途下线由 HTTP 网络错误路径兜底）。
        2026-11-02 增补：探测自身也按 retry_delays 退避重试——Tailscale 路径实测
        每数秒抖动一次（连续探测 200/200/000），单次探测即判死会把整场运行
        打死在门口（WA-0033 真机演练：预探测失败 → 29 clips 一条未嵌即中止）。"""
        if self._reachable_ok:
            return
        with self._lock:
            if self._reachable_ok:
                return
            attempts = len(self.retry_delays) + 1
            for attempt in range(attempts):
                for ep in (self.endpoint, self.fallback_endpoint):
                    if not ep:
                        continue
                    if endpoint_reachable(ep, timeout=self.probe_timeout):
                        self._active_endpoint = ep
                        self._reachable_ok = True
                        if ep != self.endpoint:
                            print(f"[WEMMClient] 主端点不可达，切换备用端点: {ep}")
                        return
                if attempt < len(self.retry_delays):
                    time.sleep(self.retry_delays[attempt])
            raise RuntimeError(f"嵌入端点不可达: {self.endpoint or '(未配置)'}")

    def _post(self, body: dict) -> List[np.ndarray]:
        """POST /v1/embeddings → data[i].embedding 列表（顺序与 input/content 对位）。

        401/403/404/422 等 4xx 不重试（鉴权/路径/参数问题，重试无意义）；
        5xx / 网络错误按 retry_delays 退避重试。Key 只在 Authorization 头，
        错误消息自构造、不含请求头/凭据。
        """
        self._ensure_reachable()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last_err: Optional[Exception] = None
        for attempt in range(len(self.retry_delays) + 1):
            try:
                resp = requests.post(self._active_endpoint, json=body, headers=headers,
                                     timeout=(CONNECT_TIMEOUT_S, self.timeout))
                status = getattr(resp, "status_code", None)
                if status is not None and status < 400:
                    data = resp.json().get("data") or []
                    return [np.asarray(d["embedding"], dtype=np.float32).reshape(-1)
                            for d in data]
                if status is not None and status >= 500:
                    last_err = WemmHttpError(int(status), f"嵌入端点服务错误(HTTP {status})",
                                             "服务端 5xx（过载/内部错误），稍后重试或检查服务")
                else:
                    detail = ""
                    try:
                        detail = str(resp.json())[:200]
                    except Exception:
                        pass
                    hint = {401: "嵌入端点需鉴权：请导出 WEMM_API_KEY",
                            403: "嵌入端点拒绝访问：请检查 WEMM_API_KEY 权限",
                            404: "端点路径不存在：请检查 embed_endpoint 是否为 /v1/embeddings",
                            422: "请求被服务端拒绝：请检查 embed_model / dimensions 档位 / messages 形态"}
                    raise WemmHttpError(
                        int(status),
                        f"嵌入端点请求失败(HTTP {status})" + (f"：{detail}" if detail else ""),
                        hint.get(int(status), ""))
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = WemmHttpError(0, f"嵌入端点网络错误: {type(e).__name__}",
                                         "网络不可达或读取超时（含图请求较慢可调大 embed_timeout）")
            if attempt < len(self.retry_delays):
                time.sleep(self.retry_delays[attempt])
        raise last_err  # type: ignore[misc]

    @staticmethod
    def _check_l2(vec: np.ndarray, where: str, tol: float = 0.05) -> None:
        """L2 复验：服务端输出应已归一（点积=余弦）；模长偏离 1 超阈值告警（不拦截）。"""
        n = float(np.linalg.norm(vec)) if vec.size else 0.0
        if abs(n - 1.0) > tol:
            warnings.warn(f"[WEMMClient] {where} 向量 L2 模长偏离 1: {n:.4f}"
                          f"（服务端应已归一，请检查 dimensions 档位与服务状态）")
