# -*- coding: utf-8 -*-
"""
UMVP 可视化测试台 —— 后端（零新增依赖，stdlib http.server）

职责:
  1. GET  /api/schema   返回模块参数清单（前端据此动态生成表单）
  2. POST /api/test/<模块>  按表单参数运行该模块的独立测试，返回
                          {ok, text, images:[{title,data(base64)}], tables:[{title,headers,rows}]}
  3. GET  /             返回单页前端 index.html
  4. GET/POST /api/presets*   命名配置暂存（MySQL，见 db.py，--no-db 可禁用）
  5. GET/POST /api/pipelines*  链路拼装暂存（步骤引用模块 preset id + 顺序）
  6. GET  /api/db/tables  数据库数据浏览（只读快照：presets 注册表 + 各模块参数表 + 链路表）
  7. POST /api/pipeline/build | /api/pipeline/agent  链路生成/智能助手（P2/P4 占位壳子）

设计约定（与项目一致）:
  - 模块测试全部复用现有模块代码，不重复实现逻辑；
  - 模型类模块（YOLO/人脸）惰性加载，仅在测试时导入；
  - 同一次只跑一个测试（全局锁），避免多个模型抢占内存；
  - 图片结果编码为 data URL，前端直接显示。

运行: ai 环境 python web_lab/server.py [--port 8001] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

ROOT = Path(__file__).resolve().parents[1]          # 项目根
UMVP = ROOT / "umvp"
STATIC = Path(__file__).resolve().parent            # web_lab/
sys.path.insert(0, str(UMVP))                       # 使 pipe/face_* 可导入

MODELS_DIR = ROOT / "models"
PIC_DIR = ROOT / "tests/data/pic"
VID_DIR = ROOT / "tests/data/vid"
TEST_IMGS = UMVP / "face_detect" / "test_imgs"

import cv2  # noqa: E402  （cv2 起步即用；ultralytics/insightface 保持惰性）
import numpy as np  # noqa: E402

from resources import ResourceEstimator  # noqa: E402  umvp/resources.py（纯标准库，无重依赖）
import db  # noqa: E402  web_lab/db.py（MySQL 配置暂存，惰性连接，缺失不影响启动）

_ESTIMATOR = None  # 惰性创建：首次 /api/schema 或 /api/pipeline/estimate 时初始化
_DB_ENABLED = True  # 启动时 --no-db 或 MySQL 不可用则置 False，presets 接口返回"未启用"

_IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TEST_LOCK = threading.Lock()

# 阶段短标签：链路运行日志按"阶段→下一级"展示（不依赖具体链路类型）
_STAGE_LABELS = {
    "frame_manager": "帧管理", "yolo": "YOLO识别", "target_crop": "目标裁剪",
    "annotate": "标注", "vlm": "VLM研判", "det_filter": "检测筛选", "seq_filter": "时序判定",
    "plate_recog": "车牌识别", "face_detect": "人脸检测", "face_embed": "人脸嵌入",
    "face_store": "向量底库", "extract_faces": "原分辨率提取",
}


def _stage_label(mid) -> str:
    return _STAGE_LABELS.get(mid) or (mid or "")


# ---------------- 通用工具 ----------------

def _img_data_url(img, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def res(ok: bool = True, text: str = "", images: list = None,
        tables: list = None, error: str = "", timeline: list = None) -> dict:
    d = {"ok": ok, "text": text}
    if images:
        d["images"] = images
    if tables:
        d["tables"] = tables
    if timeline:
        d["timeline"] = timeline
    if error:
        d["error"] = error
    return d


def _int(p: dict, k: str, d: int = 0) -> int:
    v = p.get(k)
    try:
        return int(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


def _float(p: dict, k: str, d: float = 0.0) -> float:
    v = p.get(k)
    try:
        return float(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


def _bool(p: dict, k: str) -> bool:
    v = p.get(k)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on") if v else False


def _int_list(p: dict, k: str, d=None):
    v = p.get(k)
    if not v:
        return d
    out = []
    for x in re.split(r"[,，\s]+", str(v).strip()):
        if x:
            out.append(int(x))
    return out


def _str_list(p: dict, k: str) -> list:
    v = p.get(k)
    if not v:
        return []
    return [x.strip() for x in re.split(r"[,，\s]+", str(v)) if x.strip()]


def _int_tuple(p: dict, k: str, n: int):
    v = p.get(k)
    if not v:
        return None
    parts = [int(x) for x in re.split(r"[,\sx×]+", str(v).strip()) if x]
    return tuple(parts[:n]) if parts else None


def _class_limits(s: str) -> dict:
    """'0:1,300; 2:1,10' → {0:(1,300), 2:(1,10)}（类别:下限,上限）"""
    d: dict = {}
    for part in re.split(r"[;；\n]+", s or ""):
        m = re.match(r"(\d+)\s*[:：]\s*(\d+)\s*[,，\-~]+\s*(\d+)", part.strip())
        if m:
            d[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return d


def _str_param(p: dict, k: str, d: str = "") -> str:
    """读字符串参数；DB 空列还原为 None，None/空白一律回落默认值（防 'None' 字符串）。"""
    v = p.get(k)
    return str(v).strip() if isinstance(v, str) and v.strip() else d


def _find_file(name: str, bases):
    """按 basename 在候选目录里找文件；空名/找不到返回 None。"""
    if not name:
        return None
    for b in bases:
        cand = Path(b) / name
        if cand.is_file():
            return cand
    p = Path(name)
    return p if p.is_file() else None


def _load_frame(p: dict) -> Path:
    import cv2 as _cv2
    path = _find_file(p.get("image", ""), [PIC_DIR, TEST_IMGS])
    frame = _cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return path, frame


def _draw_dets(frame, dets) -> None:
    """在原图副本上画检测框（供 YOLO / compose 测试用）。"""
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"cls{d.cls_id} {d.score:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def _draw_tracks(frame, dets) -> None:
    """画带 track_id 的检测框（帧管理-YOLO 链路用；有 track 橙色、无 track 绿色）。"""
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        color = (255, 180, 0) if d.track_id else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"t{d.track_id} c{d.cls_id} {d.score:.2f}" if d.track_id \
            else f"c{d.cls_id} {d.score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _parse_vlm_result(content: str) -> tuple:
    """解析大模型回复 → (action, description)。

    兼容裸 JSON / ```json 包裹 / 带前后废话；action 取
    result/alert_type/action 三字段之一，description 可缺失（兜底空串）。
    """
    txt = content.strip()
    if txt.startswith("```"):
        parts = txt.split("\n", 1)
        txt = parts[1] if len(parts) > 1 else ""
    m = re.search(r'"(?:result|alert_type|action)"\s*:\s*"([^"]+)"', txt)
    action = m.group(1) if m else (txt or "none").strip()[:64]
    d = re.search(r'"description"\s*:\s*"([^"]*)"', txt)
    description = d.group(1) if d else ""
    return action, description


def _vlm_material(frame, payload: dict, req) -> "np.ndarray":
    """按素材 ref 准备送审图：现只有 full:*（整帧/标注图直通）——ROI 裁剪已由
    上游「目标裁剪」阶段承担，VLM 素材不再自行裁剪。"""
    return frame


# VLM 送审图像源（spec 的 vlm 阶段参数 image_source；默认 annotated=现状语义）
VLM_IMAGE_SOURCES = ("annotated", "frame", "crops")
# VLM 送审张数（spec 的 vlm 阶段参数 image_pick；默认 first＝只送首张=现状语义）
VLM_IMAGE_PICKS = ("first", "all")


def _bbox_iou(a, b) -> float:
    """两框 IoU（(x1,y1,x2,y2)，像素坐标）；退化框返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_crops_for_req(req, crops) -> list:
    """在目标裁剪产物 last_crops 里找与本次报送目标对应的**全部**子图。

    匹配依据（下游可核对）：优先 req.det.bbox（单目标素材），否则取 req.dets 各框；
    按 bbox 精确相等或 IoU≥0.5 命中（裁剪含 margin/缩放，用 IoU 而非全等）。
    返回按 crops 顺序命中的对象列表（基数约定：一条报送记录可涉多个对象 → 图集）。
    """
    if not crops:
        return []
    wanted = []
    det = getattr(req, "det", None)
    if det is not None:
        wanted.append(tuple(int(v) for v in det.bbox))
    for d in getattr(req, "dets", None) or []:
        wanted.append(tuple(int(v) for v in d.bbox))
    hits = []
    for c in crops:
        cb = tuple(int(v) for v in c.bbox)
        if any(cb == w or _bbox_iou(cb, w) >= 0.5 for w in wanted):
            hits.append(c)
    return hits


def _match_crop_for_req(req, crops):
    """取首个匹配子图（单图语义，兼容既有调用）；无命中返回 None。"""
    hits = _match_crops_for_req(req, crops)
    return hits[0] if hits else None


def _pick_vlm_images(source: str, req, frame, pipe) -> tuple:
    """按 vlm 阶段声明的 image_source 取本帧送审图——**列表**（图集语义）。

    - annotated / frame → 长度 1（现状语义）；
    - crops → 该类报送记录所涉**全部对象**的裁剪子图（按 bbox 精确或 IoU≥0.5 收集，
      用 records.image_set_of 取图=集合语义，兼容 CropResult 对象与统一记录）；
    无源/未命中一律回退到现状（annotated→本帧原图），不抛错，回退列表取其源长度。
    返回 (图像列表, 说明, 是否回退)；单图/多图由驱动层按 image_pick 决定（默认 first=现状）。
    """
    from pipe.records import image_set_of

    annotated = getattr(pipe, "last_annotated", None)
    if source == "frame":
        if frame is not None:
            return [frame], "原帧", False
        return ([annotated] if annotated is not None else []), "无本帧原图，回退标注图", True
    if source == "crops":
        hits = _match_crops_for_req(req, getattr(pipe, "last_crops", None) or [])
        imgs = [im for c in hits for im in image_set_of(c) if im is not None]
        if imgs:
            return imgs, "裁剪子图", False
        if annotated is not None:
            return [annotated], "无匹配裁剪子图，回退标注图", True
        return ([frame] if frame is not None else []), "无匹配裁剪子图，回退原帧", True
    # annotated（默认，现状语义）
    if annotated is not None:
        return [annotated], "标注图", False
    return ([frame] if frame is not None else []), "无标注图，回退原帧", True


def _apply_vlm_inputs(pipe, subs):
    """vlm 阶段声明 inputs.dets 时，按指定上游产物重构送审请求（决定"报送哪些检测"）。

    - 未声明 vlm.inputs.dets → 原样返回 step() 的默认 submits（零回归）；
    - 声明了 → 从 pipe.artifacts 取该产物（yolo.dets=筛选前 / det_filter.dets=筛选后 /
      seq_filter.dets=门控后），按 cls_id 分组复用 Pipeline._submits（gran=class 语义）；
    - 产物本帧未登记（如上游无产出）→ 返回默认 submits（不抛错，行为与现状一致）。
    Detection 对象与统一记录 dict（含 cls_id/bbox）均可消费；无 cls_id 的项跳过。
    """
    art = (getattr(pipe, "stage_inputs", None) or {}).get("vlm", {}).get("dets")
    if not art:
        return subs
    items = (getattr(pipe, "artifacts", None) or {}).get(art)
    if items is None:
        return subs
    dets = _as_detections(items)
    try:
        return list(pipe._submits(dets))  # 复用 Pipeline 的按类分组（gran=class）语义
    except Exception:
        return subs


def _as_detections(items):
    """把产物项归一成 Detection 列表：Detection 原样；dict 记录带 cls_id/bbox 时转换。"""
    from pipe.composer import Detection
    dets = []
    for it in items or []:
        if isinstance(it, Detection):
            dets.append(it)
        elif isinstance(it, dict) and "cls_id" in it and "bbox" in it:
            dets.append(Detection(track_id=int(it.get("track_id", 0) or 0),
                                  cls_id=int(it["cls_id"]), bbox=tuple(int(v) for v in it["bbox"]),
                                  score=float(it.get("score", 0.0)),
                                  model_path=it.get("model_path")))
    return dets


def _endpoint_reachable(url: str, timeout: float = 2.0) -> bool:
    """快速探测端点是否可连（TCP 连接），避免 requests 长超时把整轮测试拖住。"""
    import socket
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _vlm_real(p: dict, payload: dict, frame) -> tuple:
    """OpenAI 兼容端点调用（返回 (action, description, content)）。

    frame：单张 ndarray（现状，单图消息）或图集列表（image_pick=all，OpenAI 兼容
    **多图**消息：多个 image_url content 项 + 末尾 text）。单图路径与改动前**逐字等价**。
    timeout=(连接, 读取)：连接阶段最多 5s（端点不可达时快速失败），读取按 vlm_timeout（默认 20s）。
    注：端点对多图 content 的支持离线无法验证，**待真实端点联调**。
    """
    import requests
    url = str(p.get("vlm_endpoint") or "").strip() or "http://117.42.21.253:8000/v1/chat/completions"
    model = str(p.get("vlm_model") or "").strip() or "qwen3-vl-32b"
    key = str(p.get("vlm_key") or "").strip()
    read_timeout = int(p.get("vlm_timeout", 20) or 20)
    mr = tuple(payload.get("max_resolution") or (1280, 720))
    frames = list(frame) if isinstance(frame, (list, tuple)) else [frame]
    content = []
    for f in frames:
        h, w = f.shape[:2]
        scale = min(mr[0] / w, mr[1] / h)
        if scale < 1.0:
            f = cv2.resize(f, (max(1, int(w * scale)), max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("帧转 JPEG 失败")
        b64 = base64.b64encode(buf.tobytes()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    prompt = payload.get("prompt") or (
        "你是画面内容分析器。分析这张图像（可能画有目标框），只输出一个 JSON，"
        "不要任何多余文字或代码块：\n"
        "{\"result\": \"画面主要内容/动作，无异常填 none\", "
        "\"description\": \"一句话中文描述\", \"confidence\": 0.0}"
    )
    content.append({"type": "text", "text": prompt})
    body = {
        "model": model, "temperature": 0.1, "max_tokens": 256,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.post(url, json=body, headers=headers, timeout=(5, read_timeout))
    resp.raise_for_status()
    content_text = resp.json()["choices"][0]["message"]["content"]
    action, description = _parse_vlm_result(content_text)
    return action, description, content_text


def _vlm_content(raw: str):
    """VLM 原始回复 → 内容字段（尽量解析为 JSON；失败退回原文；空则 None）。"""
    if not raw:
        return None
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    try:
        return json.loads(t)
    except ValueError:
        return raw


def _db_path(p: dict) -> Path:
    v = str(p.get("db_path", "out/face_db.npz")).strip() or "out/face_db.npz"
    path = Path(v)
    if not path.is_absolute():
        path = ROOT / path
    return path


# ---------------- 各模块测试处理函数 ----------------

def h_frame_manager(p: dict) -> dict:
    from pipe.composer import (
        FrameManager, SAMPLING_ANALYSIS, SAMPLING_WALL_CLOCK, SAMPLING_FRAME_COUNT,
    )
    sm = p.get("sampling", SAMPLING_ANALYSIS)
    from pipe.composer import backpressure_params
    fm = FrameManager(sampling=sm, frame_skip=_int(p, "frame_skip", 3),
                      interval_sec=_float(p, "interval_sec", 3.0),
                      interval_frames=_int(p, "interval_frames", 30),
                      **backpressure_params(p.get("backpressure", "standard")))
    n = _int(p, "n_frames", 20)
    step = _float(p, "ts_step", 1.0)
    fps = _float(p, "fps", 25.0)
    mode = p.get("mode", "queue")  # analysis 模式的背压模拟：queue / adaptive
    scen = p.get("scenario", "平稳")
    takes = 0
    lines = []
    bp_rows = []
    for i in range(n):
        ts = round(i * step, 2)
        pending = None
        want = False
        if sm == SAMPLING_ANALYSIS:
            if mode == "queue":
                base, wave = 5, 40
                pending = wave if (scen == "积压浪涌" and 10 <= i <= 15) else base
            want = fm.wants_frame(i, ts, pending=pending)
            if mode == "adaptive" and want:
                cost = (0.1 + 0.5 * i / max(n - 1, 1)) if scen == "耗时渐增" else 0.08
                fm.note_inference(cost, fm.budget(fps))
            bp_rows.append([i, "" if pending is None else pending, want,
                            fm.current_skip, fm.relaxed])
        else:
            want = fm.wants_frame(i, ts)
        takes += int(want)
        lines.append(f"帧 {i:>3}  ts={ts:>7.2f}s  →  {'● 取图' if want else '○ 跳过'}")
    if sm == SAMPLING_ANALYSIS:
        head = (f"取图节奏: {sm}({mode}) | 场景: {scen} | 共 {n} 帧, 取图 {takes} 帧 "
                f"({takes / max(n, 1) * 100:.0f}%) | 放宽次数: {fm.relax_events}")
        table = {"title": "帧决策时间线（抽帧 + 背压）",
                 "headers": ["帧", "积压深度", "分析?", "当前间隔", "放宽中"],
                 "rows": [[r[0], r[1], "✓" if r[2] else "·", r[3], "✓" if r[4] else ""]
                          for r in bp_rows]}
        return res(text=head + "\n" + "\n".join(lines), tables=[table])
    text = (f"取图节奏: {sm} | 共 {n} 帧, 取图 {takes} 帧 ({takes / max(n, 1) * 100:.0f}%)\n"
            + "\n".join(lines))
    return res(text=text)


def h_yolo(p: dict) -> dict:
    from pipe.composer import YOLODetector
    img_path, frame = _load_frame(p)
    model = _find_file(p.get("model_path", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
    yolo = YOLODetector(model_path=str(model),
                        conf=_float(p, "conf", 0.35), iou=_float(p, "iou", 0.7),
                        imgsz=_int(p, "imgsz", 640), max_det=_int(p, "max_det", 300),
                        classes=_int_list(p, "classes", None),
                        device=str(p.get("device", "")).strip() or None)
    t0 = time.time()
    dets = yolo.infer(frame)  # 纯推理（有效性筛选由「检测筛选」模块承担）
    dt = time.time() - t0
    by_cls: dict = {}
    for d in dets:
        by_cls.setdefault(d.cls_id, []).append(d)

    canvas = frame.copy()
    _draw_dets(canvas, dets)
    cls_desc = "、".join(f"cls{c}×{len(v)}" for c, v in sorted(by_cls.items())) or "无"
    text = [f"模型: {model.name} | 图片: {img_path.name}",
            f"推理耗时: {dt:.2f}s | 检出 {len(dets)} 个（{cls_desc}）",
            "（纯推理输出；有效性筛选请用「检测筛选 DetFilter」模块）"]
    rows = [[i, d.track_id, d.cls_id, f"{d.score:.3f}", str(d.bbox),
             f"{d.bbox[2] - d.bbox[0]}x{d.bbox[3] - d.bbox[1]}"]
            for i, d in enumerate(dets)]
    tables = [{"title": "检出目标", "headers": ["#", "track", "类别", "置信度", "bbox", "尺寸"],
               "rows": rows}] if rows else []
    return res(text="\n".join(text),
               images=[{"title": "检测标注", "data": _img_data_url(canvas)}], tables=tables)


def _det_filter_of(p: dict):
    """det_filter 参数（filter/group_limits/events）→ DetFilter 实例。

    group_limits 空 = 不判定（None）；events 为 JSON 列表（事件规格，可作链路尾）。
    """
    from pipe.det_filter import DetFilter
    gl_text = str(p.get("group_limits", "") or "").strip()
    ev_text = str(p.get("events", "") or "").strip()
    events = None
    if ev_text:
        try:
            events = json.loads(ev_text)
            if not isinstance(events, list):
                raise ValueError("应为 JSON 列表")
        except ValueError as e:
            raise ValueError(f"events 需为 JSON 列表: {e}")
    return DetFilter(filter=_str_param(p, "filter"),
                     group_limits=(_class_limits(gl_text) if gl_text else None),
                     events=events)


def _seq_filter_of(p: dict, what: str = "all"):
    """seq_filter 参数 → SeqFilter 实例（无配置则 None）。

    throttle 形如 'key:秒'（如 'cls_id:3.0'）=整类节流；change 形如 'key:watch'=track 集合
    变化门控；rules 为 JSON 列表（SeqRule 规格）。what 选择用途：
      "gates" → 只用 throttle/change（检测流门控，Pipeline.seq_filter）
      "rules" → 只用 rules（事件尾，Pipeline.event_filter）
      "all"   → 全部（模块测试页）
    """
    from pipe.seq_filter import SeqFilter, SeqRule
    throttle = change = None
    t = str(p.get("throttle", "") or "").strip()
    if t and ":" in t:
        k, v = t.split(":", 1)
        try:
            iv = float(v)
        except ValueError:
            iv = 0.0
        if k.strip() and iv > 0:
            throttle = {"key": k.strip(), "interval": iv}
    c = str(p.get("change", "") or "").strip()
    if c and ":" in c:
        k, w = c.split(":", 1)
        if k.strip():
            change = {"key": k.strip(), "watch": (w.strip() or "track_id")}
    rules = None
    rj = str(p.get("rules", "") or "").strip()
    if rj:
        try:
            specs = json.loads(rj)
            if not isinstance(specs, list):
                raise ValueError("应为 JSON 列表")
            rules = [SeqRule(**s) if isinstance(s, dict) else SeqRule(match=str(s), event="事件")
                     for s in specs]
        except ValueError as e:
            raise ValueError(f"rules 需为 JSON 列表: {e}")
    if what == "gates":
        rules = None
    elif what == "rules":
        throttle = change = None
    if rules is None and throttle is None and change is None:
        return None
    return SeqFilter(rules=rules, throttle=throttle, change=change)


def h_det_filter(p: dict) -> dict:
    """检测筛选模块测试页：模拟检测列表 → 逐条筛选 + 成组判定 → 放行/筛掉明细（含事件）。"""
    from pipe.composer import Detection
    from pipe.records import collect_events
    try:
        df = _det_filter_of(p)
    except ValueError as e:
        return res(ok=False, error=f"条件无法解析：{e}")
    lines = [l.strip() for l in str(p.get("detections", "") or "").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return res(ok=False, error="请输入模拟检测（每行: 类别,track,置信度,x1,y1,x2,y2，"
                                   "如 '0,1,0.9,100,100,200,200'）")
    dets = []
    for i, line in enumerate(lines, 1):
        parts = [x.strip() for x in line.replace("，", ",").split(",") if x]
        try:
            if len(parts) != 7:
                raise ValueError("应为 7 个数字：类别,track,置信度,x1,y1,x2,y2")
            vals = [float(x) for x in parts]
            dets.append(Detection(track_id=int(vals[1]), cls_id=int(vals[0]),
                                  bbox=tuple(int(v) for v in vals[3:7]), score=vals[2]))
        except ValueError as e:
            return res(ok=False, error=f"第 {i} 行无法解析: {line!r}（{e}）")
    ts = _float(p, "ts", 0.0)
    keep = df.run(dets, ts=ts, frame_idx=0)
    keep_ids = {id(d) for d in keep}
    by_cls_in: dict = {}
    by_cls_out: dict = {}
    for d in dets:
        by_cls_in[d.cls_id] = by_cls_in.get(d.cls_id, 0) + 1
    for d in keep:
        by_cls_out[d.cls_id] = by_cls_out.get(d.cls_id, 0) + 1
    text = [f"输入 {len(dets)} 条 | 放行 {len(keep)} 条 | 筛掉 {len(dets) - len(keep)} 条",
            f"条件: {df.filter or '（空=不筛选）'} | 成组判定: "
            + ("；".join(f"cls{c}:{lo},{hi}" for c, (lo, hi) in sorted(df.group_limits.items()))
               if df.group_limits else "（空=不判定）"),
            "按类: " + "、".join(f"cls{c} {by_cls_out.get(c, 0)}/{by_cls_in[c]}"
                                 for c in sorted(by_cls_in))]
    if df.events:
        evs = collect_events(keep)
        text.append(f"事件: {len(evs)} 条"
                    + (" — " + "、".join(e.get("name", "") for e in evs) if evs else ""))
    rows = [[i, d.cls_id, d.track_id, f"{d.score:.3f}",
             f"{d.bbox[2] - d.bbox[0]}x{d.bbox[3] - d.bbox[1]}",
             min(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]),
             "✓ 放行" if id(d) in keep_ids else "✗ 筛掉"]
            for i, d in enumerate(dets)]
    tables = [{"title": "逐条判定", "headers": ["#", "类别", "track", "置信度", "尺寸",
                                                "短边", "结果"],
               "rows": rows}]
    return res(text="\n".join(text), tables=tables)


def _parse_sim_dets(p: dict):
    """测试页模拟检测文本 → (Detection 列表, 错误信息)。每行 7 个数字。"""
    from pipe.composer import Detection
    lines = [l.strip() for l in str(p.get("detections", "") or "").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return None, "请输入模拟检测（每行: 类别,track,置信度,x1,y1,x2,y2）"
    dets = []
    for i, line in enumerate(lines, 1):
        parts = [x.strip() for x in line.replace("，", ",").split(",") if x]
        try:
            if len(parts) != 7:
                raise ValueError("应为 7 个数字：类别,track,置信度,x1,y1,x2,y2")
            v = [float(x) for x in parts]
            dets.append(Detection(track_id=int(v[1]), cls_id=int(v[0]),
                                  bbox=tuple(int(x) for x in v[3:7]), score=v[2]))
        except ValueError as e:
            return None, f"第 {i} 行无法解析: {line!r}（{e}）"
    return dets, None


def h_seq_filter(p: dict) -> dict:
    """时序判定模块测试页：模拟检测 → 节流/变化门控 → 放行/扣留明细。"""
    try:
        sf = _seq_filter_of(p)
    except ValueError as e:
        return res(ok=False, error=f"门控参数无法解析：{e}")
    if sf is None:
        return res(ok=False, error="请配置整类节流（如 cls_id:3.0）或变化门控（如 cls_id:track_id）")
    dets, err = _parse_sim_dets(p)
    if err:
        return res(ok=False, error=err)
    ts = _float(p, "ts", 0.0)
    keep = sf.run(dets, ts=ts, frame_idx=0)
    keep_ids = {id(d) for d in keep}
    text = [f"输入 {len(dets)} 条 | 放行 {len(keep)} 条 | 扣留 {len(dets) - len(keep)} 条（ts={ts}）",
            f"门控: 整类节流={sf.throttle or '关'} | 变化门控={sf.change or '关'}"]
    rows = [[i, d.cls_id, d.track_id, f"{d.score:.3f}",
             "✓ 放行" if id(d) in keep_ids else "✗ 扣留"]
            for i, d in enumerate(dets)]
    tables = [{"title": "逐条判定", "headers": ["#", "类别", "track", "置信度", "结果"], "rows": rows}]
    return res(text="\n".join(text), tables=tables)


def h_vlm(p: dict) -> dict:
    from pipe.composer import VLMAnalyzer, SubmitRequest, Detection
    vlm = VLMAnalyzer(max_resolution=tuple(_int_tuple(p, "max_resolution", 2) or (1280, 720)),
                      prompt=str(p.get("prompt", "")))
    ref = p.get("ref", "crop:cls0")
    gran = p.get("gran", "class")
    req = SubmitRequest(track_key=_int(p, "track_key", 0), gran=gran, ref=ref)
    if ref.startswith("crop:"):
        cls = int(ref.split("cls")[1])
        bb = tuple(_int_tuple(p, "bbox", 4) or (10, 10, 100, 200))
        req.det = Detection(track_id=0, cls_id=cls, bbox=bb, score=0.9)
    if req.det is None:
        req.dets = [Detection(track_id=i, cls_id=0, bbox=(10, 10, 100, 200), score=0.9)
                    for i in range(max(1, _int(p, "det_count", 1)))]
    payload = vlm.prepare(req, None)

    text = ["素材包（prepare 输出）:", json.dumps(payload, ensure_ascii=False, indent=2)]
    if _bool(p, "use_real_vlm"):
        img_path, frame = _load_frame(p)
        action, description, _raw = _vlm_real(p, payload, frame)
        text.append(f"\n真实 VLM 返回 action: {action}"
                    + (f" — {description}" if description else ""))
    else:
        text.append("\n（未勾选真实 VLM，演示 action=fire）")
    return res(text="\n".join(text))


def h_face_detect(p: dict) -> dict:
    from face_detect.face_detector import FaceDetector
    det = FaceDetector(device=str(p.get("device", "auto")),
                       det_thresh=_float(p, "det_thresh", 0.5),
                       det_size=tuple(_int_tuple(p, "det_size", 2) or (640, 640)),
                       max_num=_int(p, "max_num", 0),
                       use_crop=_bool(p, "use_crop"))
    img_path, frame = _load_frame(p)
    faces = det.detect(frame)
    canvas = FaceDetector.draw(frame, faces)
    images = [{"title": "检测标注", "data": _img_data_url(canvas)}]
    for i, f in enumerate(faces):
        if f.face_crop is not None:
            images.append({"title": f"对齐裁剪 #{f.index}", "data": _img_data_url(f.face_crop, 90)})
    rows = [[f.index, f"{f.score:.3f}", str(f.bbox), f.width, f.height, len(f.kps)]
            for f in faces]
    text = f"图片: {img_path.name} | 检出 {len(faces)} 张人脸 | device={det.is_gpu and 'GPU' or 'CPU'}"
    tables = [{"title": "人脸列表", "headers": ["#", "置信度", "bbox", "宽", "高", "关键点数"],
               "rows": rows}] if rows else []
    return res(text=text, images=images, tables=tables)


def h_face_embed(p: dict) -> dict:
    from face_detect.face_detector import FaceDetector
    from face_embed.face_embedder import FaceEmbedder
    det = FaceDetector(device=str(p.get("device", "auto")),
                       det_thresh=_float(p, "det_thresh", 0.5),
                       det_size=tuple(_int_tuple(p, "det_size", 2) or (640, 640)))
    embedder = FaceEmbedder(device=str(p.get("device", "auto")))
    img_path, frame = _load_frame(p)
    faces = det.detect(frame)
    text = [f"图片: {img_path.name} | 检测到 {len(faces)} 张人脸"]
    for f in faces:
        if f.face_crop is None:
            continue
        vec = embedder.embed_face(f.face_crop)
        head = " ".join(f"{x:+.3f}" for x in vec[:6].tolist())
        tail = " ".join(f"{x:+.3f}" for x in vec[-4:].tolist())
        norm = float(np.linalg.norm(vec))
        text.append(f"  人脸#{f.index}: dim={vec.shape[0]} norm={norm:.4f} "
                    f"[{head} … {tail}]")
    return res(text="\n".join(text))


def h_face_store(p: dict) -> dict:
    from face_detect.face_detector import FaceDetector
    from face_embed.face_embedder import FaceEmbedder
    from face_embed.face_store import FaceStore
    import face_embed.test_face_embed as tfe  # 复用已测的注册/自检函数

    db = _db_path(p)
    store = FaceStore(thresh=_float(p, "thresh", 0.45))
    loaded = db.exists()
    if loaded:
        store.load(db)
    det = FaceDetector(device=str(p.get("device", "auto")), det_thresh=0.4)
    embedder = FaceEmbedder(device=str(p.get("device", "auto")))
    action = p.get("action", "register_dir")
    buf = io.StringIO()
    registered = 0
    with contextlib.redirect_stdout(buf):
        if action == "register_dir":
            d = Path(str(p.get("register_dir", "")).strip() or TEST_IMGS)
            if not d.is_absolute():
                d = ROOT / d
            registered += tfe.do_register_dir(store, det, embedder, str(d))
        elif action == "register_single":
            img, name = str(p.get("image", "")), str(p.get("name", "")).strip()
            path = _find_file(img, [PIC_DIR, TEST_IMGS])
            faces = tfe._faces_of(det, embedder, str(path))
            if faces:
                store.register(name or path.stem, faces[0][1], meta={"src": path.name})
                registered += 1
                print(f"  [注册] {name or path.stem} <- {path.name}")
            else:
                print(f"  [跳过] 未检测到人脸: {path.name}")
        elif action == "query":
            img, topk = str(p.get("image", "")), _int(p, "topk", 5)
            path = _find_file(img, [PIC_DIR, TEST_IMGS])
            tfe.do_query(store, det, embedder, str(path), topk, _float(p, "thresh", 0.45))
        elif action == "self_check":
            tfe.do_self_check(store)
        elif action == "list":
            print(f"[底库内容] {store.stats()}")
            for i, (name, meta) in enumerate(zip(store._names, store._metas)):
                print(f"  #{i} {name}  {meta or ''}".rstrip())
    if registered:
        store.save(db)
        buf.write(f"[底库] 已保存: {db} | {store.stats()}\n")
    head = f"操作: {action} | 底库: {db}{'（已加载）' if loaded else '（新建）'} | {store.stats()}"
    return res(text=head + "\n" + buf.getvalue())


def h_plate_recog(p: dict) -> dict:
    from pipe.cropper import clamp_bbox
    from plate_recog.plate_recognizer import PlateRecognizer, extract_plates
    rec = PlateRecognizer(detect_level=str(p.get("detect_level", "high")))
    img_path, frame = _load_frame(p)
    mode = str(p.get("mode", "crop"))
    if mode == "direct":
        plates = rec.recognize(frame)
    else:
        from pipe.composer import YOLODetector
        from pipe.cropper import Cropper
        from pipe.target_crop import crop_targets
        model = _find_file(p.get("yolo_model", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
        yolo = YOLODetector(model_path=str(model), conf=_float(p, "yolo_conf", 0.25),
                            iou=_float(p, "yolo_iou", 0.6), imgsz=_int(p, "yolo_imgsz", 1280),
                            max_det=_int(p, "yolo_max_det", 300),
                            classes=_int_list(p, "yolo_classes", [2]))
        dets = yolo.infer(frame, track=False)  # 测试页脚手架：造车辆框（生产由上游 yolo 阶段提供）
        # 目标裁剪：按车辆框裁车（测试页内联；生产链路由独立「目标裁剪」阶段完成）
        crops = crop_targets(frame, dets, scale=2.0, margin=0.1,
                             classes=_int_list(p, "yolo_classes", [2]))
        tool = Cropper(upscale=1.0, margin=_float(p, "margin", 0.1))
        plates = extract_plates(frame, crops, rec, tool,
                                dedup_iou=_float(p, "dedup_iou", 0.6))
    h, w = frame.shape[:2]
    canvas = frame.copy()
    for pl in plates:
        if pl.car_bbox:
            cx1, cy1, cx2, cy2 = pl.car_bbox
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
        x1, y1, x2, y2 = pl.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{pl.plate_no} {pl.score:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    images = [{"title": "标注原图（蓝=车框, 绿=车牌框）", "data": _img_data_url(canvas)}]
    for pl in plates:
        if getattr(pl, "plate_img", None) is not None:
            # 两级裁剪链路：直接用从原图提取的原分辨率车牌小图
            images.append({"title": f"车牌 #{pl.index} {pl.plate_no} "
                                    f"({pl.plate_img.shape[1]}x{pl.plate_img.shape[0]})",
                           "data": _img_data_url(pl.plate_img, 95)})
            continue
        x1, y1, x2, y2 = pl.bbox
        pad = int(max(x2 - x1, y2 - y1) * 0.15)
        cx1, cy1, cx2, cy2 = clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size:
            images.append({"title": f"车牌 #{pl.index} {pl.plate_no}",
                           "data": _img_data_url(crop, 95)})
    rows = [[str(pl.index), pl.plate_no, pl.type_name, pl.color, f"{pl.score:.3f}",
             str(list(pl.bbox)), f"{pl.width}x{pl.height}"] for pl in plates]
    tables = [{"title": "车牌列表",
               "headers": ["#", "车牌号", "类型", "颜色", "置信度", "bbox", "尺寸"],
               "rows": rows}]
    text = (f"图片: {img_path.name} | 方式: {mode} | detect_level={rec.detect_level} | "
            f"识别到 {len(plates)} 张车牌")
    return res(text=text, images=images, tables=tables)


def h_target_crop(p: dict) -> dict:
    """目标裁剪模块测试页：按测试框从原图裁出子图（验证 out_size/margin）。"""
    from pipe.target_crop import crop_targets
    img_path, frame = _load_frame(p)
    bb = tuple(_int_tuple(p, "bbox", 4) or (10, 10, 200, 200))
    crops = crop_targets(frame, boxes=[bb], margin=_float(p, "margin", 0.0),
                         filter=_str_param(p, "filter"),
                         out_size=_str_param(p, "out_size"))
    canvas = frame.copy()
    cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
    images = [{"title": "标注原图（蓝=裁剪框）", "data": _img_data_url(canvas)}]
    for c in crops:
        images.append({"title": f"裁剪 #{c.index} ({c.img.shape[1]}x{c.img.shape[0]})",
                       "data": _img_data_url(c.img, 95)})
    text = (f"图片: {img_path.name} | scale={_float(p, 'scale', 1.0)} "
            f"margin={_float(p, 'margin', 0.0)} | 裁出 {len(crops)} 张")
    return res(text=text, images=images)


def h_annotate(p: dict) -> dict:
    """标注模块测试页：按测试框在原图画框。"""
    from pipe.annotate import annotate_frame
    from pipe.composer import Detection
    img_path, frame = _load_frame(p)
    bb = tuple(_int_tuple(p, "bbox", 4) or (10, 10, 200, 200))
    cls = (_int_list(p, "classes", [0]) or [0])[0]
    vis = annotate_frame(frame, [Detection(track_id=0, cls_id=cls, bbox=bb, score=0.9)],
                         thickness=_int(p, "thickness", 2), show_label=_bool(p, "show_label"),
                         filter=_str_param(p, "filter"))
    return res(text=f"图片: {img_path.name} | 画框 1 个（cls{cls}）",
               images=[{"title": "标注图", "data": _img_data_url(vis)}])


def h_face_monitor(p: dict) -> dict:
    from face_detect.face_detector import FaceDetector
    from face_embed.face_embedder import FaceEmbedder
    from face_embed.face_store import FaceStore
    from pipe.composer import FrameManager  # 帧管理：抽帧 + 背压（原 frame_scheduler 并入）
    from face_scan.face_monitor import FaceMonitor
    db = _db_path(p)
    if not db.exists():
        return res(ok=False, error=f"底库不存在: {db}（请先在 FaceStore 页注册并保存底库）")
    store = FaceStore().load(db)
    det = FaceDetector(device=str(p.get("device", "auto")))
    embedder = FaceEmbedder(device=str(p.get("device", "auto")))
    sch = FrameManager(frame_skip=_int(p, "frame_skip", 3),
                       queue_threshold=_int(p, "queue_threshold", 32),
                       backpressure_multiplier=_int(p, "backpressure_multiplier", 2))
    save_dir = Path(str(p.get("save_dir", "tests/data/out/faces/monitor")).strip()
                    or "tests/data/out/faces/monitor")
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    mon = FaceMonitor(det, embedder, store, scheduler=sch, save_dir=str(save_dir),
                      verbose=False)
    vid = _find_file(p.get("video", ""), [VID_DIR])
    max_frames = _int(p, "max_frames", 60)
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        cap.release()
        return res(ok=False, error=f"无法打开视频: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames():
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and n >= max_frames:
                break
            yield frame
            n += 1

    stats = mon.run_stream(frames(), fps)
    cap.release()
    text = (f"视频: {vid.name} | 底库: {db.name} ({store.stats()['identities']} 个身份)\n"
            f"读取帧 {stats.frames_read} | 分析帧 {stats.frames_analyzed} | "
            f"人脸 {stats.faces_found} | 命中 {len(stats.matches)} | "
            f"平均分析耗时 {stats.avg_infer_ms:.0f}ms | 总耗时 {stats.elapsed_sec:.1f}s | "
            f"放宽次数 {stats.relax_events}")
    rows = [[m.frame_idx, m.name, f"{m.score:.3f}", str(m.bbox)] for m in stats.matches]
    images = []
    for m in stats.matches[:12]:
        p_shot = save_dir / f"match_{m.frame_idx}_{m.name}_{m.score:.2f}.jpg"
        if p_shot.is_file():
            data = _img_data_url(cv2.imread(str(p_shot)))
            if data:
                images.append({"title": f"帧#{m.frame_idx} {m.name} {m.score:.2f}", "data": data})
    tables = [{"title": "命中记录", "headers": ["帧号", "身份", "分数", "bbox"],
               "rows": rows}] if rows else []
    return res(text=text, images=images, tables=tables)


def h_frame_yolo(p: dict) -> dict:
    """帧管理 FrameManager → YOLO 识别 YOLODetector 链路的视频评测。

    在固定视频上按取图节奏抽帧，每分析帧跑 YOLO 推理（可选 BOTSORT 跟踪），
    输出逐帧时间线、track 汇总与识别/追踪能力评估指标，用于对比不同 YOLO
    模型与参数配置的效果。
    """
    from pipe.composer import (
        FrameManager, YOLODetector, SAMPLING_ANALYSIS, SAMPLING_WALL_CLOCK,
        SAMPLING_FRAME_COUNT,
    )
    model = _find_file(p.get("model_path", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
    fm = FrameManager(sampling=p.get("sampling", SAMPLING_ANALYSIS),
                      frame_skip=_int(p, "frame_skip", 3),
                      interval_sec=_float(p, "interval_sec", 3.0),
                      interval_frames=_int(p, "interval_frames", 30))
    yolo = YOLODetector(model_path=str(model),
                        conf=_float(p, "conf", 0.35), iou=_float(p, "iou", 0.7),
                        imgsz=_int(p, "imgsz", 640), max_det=_int(p, "max_det", 300),
                        classes=_int_list(p, "classes", None),
                        device=str(p.get("device", "")).strip() or None)
    track = _bool(p, "track")
    vid = _find_file(p.get("video", ""), [VID_DIR])
    if not vid:
        return res(ok=False, error="请选择测试视频（视频必选）")
    max_frames = _int(p, "max_frames", 0)
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        cap.release()
        return res(ok=False, error=f"无法打开视频: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    read = analyzed = total_det = 0
    max_det_once = 0
    cls_counts: dict = {}
    track_seen: dict = {}  # track_id -> [出现过的帧号, ...]
    infer_times = []
    rows, images = [], []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if max_frames and n > max_frames:
            break
        read += 1
        ts = n / fps
        if not fm.wants_frame(n, ts):
            continue
        analyzed += 1
        t0 = time.time()
        dets = yolo.infer(frame, track=track)
        infer_times.append(time.time() - t0)
        total_det += len(dets)
        max_det_once = max(max_det_once, len(dets))
        ids = {d.track_id for d in dets if d.track_id}
        for d in dets:
            cls_counts[d.cls_id] = cls_counts.get(d.cls_id, 0) + 1
            if d.track_id:
                track_seen.setdefault(d.track_id, []).append(n)
        per_cls = " ".join(f"c{k}:{sum(1 for d in dets if d.cls_id == k)}"
                           for k in sorted({d.cls_id for d in dets}))
        rows.append([n, len(dets), len(ids), per_cls])
        if len(images) < 4:
            canvas = frame.copy()
            _draw_tracks(canvas, dets)
            if canvas.shape[1] > 1280:
                k = 1280 / canvas.shape[1]
                canvas = cv2.resize(canvas, (1280, int(canvas.shape[0] * k)))
            images.append({"title": f"帧#{n} 检出 {len(dets)} 个（分析第 {analyzed} 帧）",
                           "data": _img_data_url(canvas, 70)})
    cap.release()
    avg_det = total_det / max(analyzed, 1)
    avg_ms = (sum(infer_times) / len(infer_times) * 1000) if infer_times else 0.0
    track_len = [(t, fr) for t, fr in track_seen.items()]
    track_len.sort(key=lambda kv: len(kv[1]), reverse=True)
    stable = sum(1 for _t, fr in track_len if len(fr) >= 2)
    cls_summary = ", ".join(f"cls{k}={v}" for k, v in sorted(cls_counts.items()))
    text = "\n".join([
        f"视频: {vid.name} | 模型: {model.name} | 跟踪: {'BOTSORT 开' if track else '关'}",
        f"读取帧 {read} | 分析帧 {analyzed}（取图率 {analyzed / max(read, 1) * 100:.0f}%）",
        f"检出目标 {total_det}（平均 {avg_det:.1f}/帧，单帧最多 {max_det_once} 个）",
        f"按类累计: {cls_summary}",
        f"唯一 track 数 {len(track_seen)}（跨≥2帧的稳定 track {stable} 个）",
        f"平均推理耗时 {avg_ms:.0f}ms/帧（首帧含模型加载）",
    ])
    tables = []
    if rows:
        tables.append({"title": "逐帧时间线", "headers": ["帧号", "检出",
                        "独立track", "各类别计数"], "rows": rows})
    if track and track_len:
        tables.append({"title": "track 汇总（按出现帧数排序）",
                       "headers": ["track_id", "出现帧数", "首帧", "末帧", "跨度(帧)"],
                       "rows": [[t, len(fr), fr[0], fr[-1], fr[-1] - fr[0] + 1]
                                for t, fr in track_len]})
    return res(text=text, images=images, tables=tables)


def h_pipeline_run(p: dict) -> dict:
    """链路运行测试：支持视频片段截取、图片按帧保存等功能。

    参数：
    - video: 视频文件名
    - image: 图片文件名（单帧测试）
    - max_frames: 最大处理帧数（0=全部）
    - start_time: 起始时间（秒）
    - end_time: 结束时间（秒，0=视频结尾）
    - vlm_feedback: 是否启用VLM回填
    - vlm_action: VLM回填动作
    - output_dir: 输出目录
    - image_format: 图片格式
    - pipeline_id: 链路ID

    注意：取图节奏由链路内部 FrameManager 的 frame_skip 控制（见链路设计器帧管理模块），
    测试台不再提供独立的 frame_skip 参数，避免重复控制。
    """
    # 获取链路配置（业务 ID，如 PIPE0001）
    pipeline_id = str(p.get("pipeline_id", "") or "").strip()
    if not pipeline_id:
        return res(ok=False, error="请指定测试链路（pipeline_id）")
    
    if not _DB_ENABLED:
        return res(ok=False, error="数据库未启用（--no-db）")
    
    try:
        pipeline_data = db.load_pipeline(pipeline_id)
        spec = {
            "name": pipeline_data.get("name"),
            "streams": pipeline_data.get("streams"),
            "budget": pipeline_data.get("budget"),
            "meta": pipeline_data.get("meta"),
            "stages": pipeline_data.get("stages", [])
        }
    except Exception as e:
        return res(ok=False, error=f"加载链路失败: {e}")
    
    if not spec.get("stages"):
        return res(ok=False, error="链路配置缺少 stages")
    
    try:
        pipe = _chain_from_spec(spec)
    except ValueError as e:
        return res(ok=False, error=str(e))
    
    # 测试参数
    feedback = _bool(p, "vlm_feedback")
    vlm_action = str(p.get("vlm_action", "fire")).strip() or "fire"
    max_frames = _int(p, "max_frames", 0)
    start_time = _float(p, "start_time", 0.0)
    end_time = _float(p, "end_time", 0.0)

    # 真实 VLM 配置：来自链路 vlm 阶段参数（use_real_vlm/vlm_endpoint/vlm_model/vlm_key）
    vlm_cfg = {}
    for s in spec.get("stages") or []:
        if s.get("module_id") == "vlm":
            vlm_cfg = s.get("params") or {}
            break
    use_real_vlm = _bool(vlm_cfg, "use_real_vlm")
    # 送审图像源：壳内声明（vlm 阶段 image_source），默认 annotated=现状；
    # 非法值回落默认（validate_chain 会在建链路时拦，这里兜底防手工 spec 绕过）。
    image_source = _str_param(vlm_cfg, "image_source", "annotated").lower()
    if image_source not in VLM_IMAGE_SOURCES:
        image_source = "annotated"
    # 送审张数（image_pick）：first=只送首张（默认，与改动前逐字等价）/ all=送图集全部
    # （仅 crops 源可能多张；端点多图支持待真实联调）。非法值回落默认 first。
    image_pick = _str_param(vlm_cfg, "image_pick", "first").lower()
    if image_pick not in VLM_IMAGE_PICKS:
        image_pick = "first"
    vlm_fail = 0  # 真实调用失败次数（网络等），累计供结果提示
    # 真实 VLM 端点预探测：不可达则本轮直接跳过真实调用，避免每帧长超时把测试拖住
    _vlm_url = str(vlm_cfg.get("vlm_endpoint") or "").strip() \
        or "http://117.42.21.253:8000/v1/chat/completions"
    vlm_unreachable = False
    if feedback and use_real_vlm and not _endpoint_reachable(_vlm_url):
        vlm_unreachable = True
        print(f"[vlm] 端点不可达，本轮跳过真实调用（改用素材包）: {_vlm_url}")

    # 输出目录设置
    output_dir = Path(str(p.get("output_dir", "tests/data/out/test_pipeline")).strip())
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_format = str(p.get("image_format", "jpg")).lower()

    # ---- 落盘上限：单次测试最多保存 200 张图（原图/标注图/人脸图共用计数）----
    # 超限后跳过写盘只记结果，防止长视频把磁盘写爆；可用参数 max_saved_images 覆盖。
    max_saved_images = _int(p, "max_saved_images", 200)
    saved_images, dropped_saves = [0], [0]

    def _save_image(path: Path, img) -> bool:
        if saved_images[0] >= max_saved_images:
            dropped_saves[0] += 1
            return False
        cv2.imwrite(str(path), img)
        saved_images[0] += 1
        return True

    def _save_summary() -> str:
        s = f" | 落盘 {saved_images[0]} 张"
        if dropped_saves[0]:
            s += f"（超上限跳过 {dropped_saves[0]} 张）"
        return s

    log, images = [], []
    timeline = []  # 结构化逐条记录: {kind, stage, to, text, data?, ts, frame}

    # ---- 阶段顺序：日志统一渲染成"阶段→下一级"（不依赖具体链路类型）----
    _stage_order = [s.get("module_id") for s in spec.get("stages") or [] if s.get("module_id")]
    _next_of = {_stage_order[i]: _stage_order[i + 1] for i in range(len(_stage_order) - 1)}

    def _emit(kind, text, ts, stage=None, to="__next__", data=None,
              log_line=False, prefix="", frame_no=0):
        """写一条结构化日志；自动带 stage→to 标签与帧序数，前端据此按阶段过滤与着色。"""
        if to == "__next__":
            to = _next_of.get(stage)
        item = {"kind": kind, "text": text, "ts": ts, "frame": frame_no}
        if stage:
            item["stage"] = stage
            item["stage_label"] = _stage_label(stage)
        if to:
            item["to"] = to
            item["to_label"] = _stage_label(to)
        if data:
            item["data"] = data
        timeline.append(item)
        if log_line:
            tag = f"[{_stage_label(stage)}] " if stage else ""
            log.append(f"{prefix}{tag}帧#{frame_no} {text}")
        return item

    # ---- 人脸链（装配层已注入 Pipeline.face；落盘/计数/身份去留在驱动层）----
    # ① 含 face_embed 阶段 → 视频人脸去重提取模式（嵌入 + 身份去重）；
    # ② 仅 face_detect 阶段（无嵌入/检索）→ 人脸截取模式：逐帧提取原分辨率人脸并落盘。
    # 链路段：帧管理 → YOLO识人 → 目标裁剪（裁人）→ 人脸检测（检脸 + 原图出脸），
    # 逐帧与普通链一致走 pipe.step（_step_out 从 pipe.last_faces 做落盘/计数/去重）。
    is_face_chain = getattr(pipe, "face", None) is not None
    femb = getattr(pipe, "face_embedder", None)
    face_dedup_thresh = getattr(pipe, "face_dedup_thresh", 0.55)
    face_vecs = []       # 已收集身份的归一化嵌入
    face_count = face_dup = 0

    def _step_out(frame_no, ts, tag, dets, subs, al, frame=None):
        """逐模块展示本帧各自处理的内容（不管是否向下传递），每条带帧序数。"""
        nonlocal detection_count, vlm_fail, plate_count, vlm_unreachable, event_count
        nonlocal face_count, face_dup
        plates = getattr(pipe, "last_plates", [])
        plate_in = getattr(pipe, "last_plate_input", [])
        detection_count += len(dets or [])
        plate_count += len(plates)

        # 帧管理：帧决策（放行本帧）
        if "frame_manager" in _stage_order:
            _emit("frame", tag, ts, stage="frame_manager", frame_no=frame_no, log_line=True)

        # 检测筛选：有效性判定（显式阶段时展示筛掉量；翻译路径不额外刷日志）
        if "det_filter" in _stage_order:
            fi = getattr(pipe, "last_filter", None)
            if fi is not None and fi[0] != fi[1]:
                _emit("info", f"有效性筛选 {fi[0]} → {fi[1]}（筛掉 {fi[0] - fi[1]} 条）", ts,
                      stage="det_filter", frame_no=frame_no, log_line=True, prefix="    ")

        # YOLO 识别：检出统计 +（向下游）报送/传递
        if "yolo" in _stage_order:
            by_cls = {}
            for d in dets or []:
                by_cls[d.cls_id] = by_cls.get(d.cls_id, 0) + 1
            cls_desc = "、".join(f"cls{c}×{n}" for c, n in sorted(by_cls.items())) or "无"
            parts = [f"检出 {len(dets or [])}（{cls_desc}）"]
            if subs:
                parts.append("报送 " + "、".join(f"{s.ref} 目标{len(s.dets)}" for s in subs))
            if pipe.plate is not None and "target_crop" not in _stage_order:
                tids = [d.track_id for d in plate_in if getattr(d, "track_id", 0)]
                parts.append(f"传递车辆框 {len(plate_in)} 个" + (f"（track {tids}）" if tids else ""))
            _emit("submit", "；".join(parts), ts, stage="yolo",
                  frame_no=frame_no, log_line=True, prefix="    ")

        # 目标裁剪：本帧按上游 YOLO 框裁出的子图（人脸链裁的是"人子图"，沿用旧文案）
        if "target_crop" in _stage_order:
            crops = getattr(pipe, "last_crops", [])
            tids = [c.track_id for c in crops if getattr(c, "track_id", 0)]
            crop_unit = "人子图" if is_face_chain else "子图"
            _emit("info", f"裁出 {len(crops)} 张{crop_unit}" + (f"（track {tids}）" if tids else ""),
                  ts, stage="target_crop", frame_no=frame_no, log_line=True, prefix="    ")

        # 标注：本帧生成带框图（供 VLM 素材）
        if "annotate" in _stage_order:
            vis = getattr(pipe, "last_annotated", None)
            _emit("info", "已生成标注图" if vis is not None else "无检出，未生成标注图",
                  ts, stage="annotate", frame_no=frame_no, log_line=True, prefix="    ")

        # VLM 研判：本帧处理内容（真实/模拟研判；未开启回填/端点不可达则展示素材准备）
        if "vlm" in _stage_order and pipe.vlm is not None:
            if not subs:
                _emit("info", "无报送，未处理", ts, stage="vlm",
                      frame_no=frame_no, log_line=True, prefix="    ")
            else:
                for s in subs:
                    if not feedback:
                        _emit("vlm", f"素材包 {s.ref}（未开启研判回填）", ts, stage="vlm",
                              frame_no=frame_no, log_line=True, prefix="    ")
                        continue
                    if use_real_vlm:
                        if vlm_unreachable:
                            _emit("vlm", f"素材包 {s.ref}（端点不可达，跳过真实调用）", ts,
                                  stage="vlm", frame_no=frame_no, log_line=True, prefix="    ")
                            continue
                        if frame is None:
                            continue
                        try:
                            payload = pipe.vlm.prepare(s, None)
                            # 送审图按壳内 image_source 选取（默认 annotated=现状；缺源回退）；
                            # 返回**列表**（crops 源可能为该类多个对象子图=图集）。
                            images_list, src_note, src_fallback = _pick_vlm_images(
                                image_source, s, frame, pipe)
                            # image_pick：first=只送首张（默认，现状）；all=送图集全部。
                            sent = images_list[:1] if image_pick == "first" else list(images_list)
                            if not sent:
                                continue
                            # 图集多张时在素材备注里带张数（默认 first 单张不改文案）
                            note = src_note + (f"×{len(sent)}" if len(sent) > 1 else "")
                            # 回退（或非默认源）时在研判日志里给出可读来源说明
                            src_tag = ("" if (image_source == "annotated" and not src_fallback)
                                       else f"·素材:{note}")
                            materials = [_vlm_material(im, payload, s) for im in sent]
                            base = materials[0] if len(materials) == 1 else materials
                            action, description, raw = _vlm_real(vlm_cfg, payload, base)
                            _emit("vlm", f"研判 {action}"
                                  + (f" — {description}" if description else "")
                                  + f"（真实调用{src_tag}）",
                                  ts, stage="vlm", frame_no=frame_no, log_line=True, prefix="    ")
                        except Exception as e:
                            vlm_fail += 1
                            _emit("fail", f"调用失败({vlm_fail}): {e}", ts, stage="vlm",
                                  frame_no=frame_no, log_line=True, prefix="    ")
                            if vlm_fail >= 3 and not vlm_unreachable:
                                vlm_unreachable = True
                                _emit("info", "真实 VLM 连续失败≥3，本轮后续改用素材包", ts,
                                      stage="vlm", frame_no=frame_no, log_line=True, prefix="    ")
                            continue
                    else:
                        action, description, raw = vlm_action, "", ""
                        _emit("vlm", f"研判 {action}（模拟回填）", ts, stage="vlm",
                              frame_no=frame_no, log_line=True, prefix="    ")
                    # VLM 结果作为记录经事件尾（event_filter）出事件；content 落 VLM 原始 JSON
                    content = _vlm_content(raw) if raw else {"result": action, "description": description}
                    rec = {"kind": "vlm", "label": action, "text": description,
                           "content": content}
                    for ev in pipe.publish([rec], ts, frame_no):
                        event_count += 1
                        nm = ev.get("name", "事件")
                        det = ev.get("detail") or {}
                        _emit("submit", f"事件 {nm}" + (f" {det}" if det else ""), ts,
                              stage="seq_filter", frame_no=frame_no, log_line=True, prefix="    ")

        # 车牌识别：本帧识别结果 + 原分辨率车牌小图落盘
        if "plate_recog" in _stage_order:
            if plates:
                for p in plates:
                    _emit("submit",
                          f"识别 {p.plate_no} {p.type_name}/{p.color} {p.score:.2f} bbox={list(p.bbox)}",
                          ts, stage="plate_recog", data=p.to_dict(),
                          frame_no=frame_no, log_line=True, prefix="    ")
                    if getattr(p, "plate_img", None) is not None:
                        plate_saved[0] += 1
                        fname = output_dir / f"plate_{plate_saved[0]:03d}_{p.plate_no}_{ts:.1f}s.{image_format}"
                        if _save_image(fname, p.plate_img) and len(images) < 36:
                            images.append({"title": f"车牌 {p.plate_no} t={ts:.1f}s",
                                           "data": _img_data_url(p.plate_img, 90)})
            else:
                _emit("info", f"识别 0 张（传入车图 {len(plate_in)} 张）", ts,
                      stage="plate_recog", frame_no=frame_no, log_line=True, prefix="    ")

        # 人脸检测：从 pipe.last_faces（Pipeline 人脸插槽产出）做落盘/计数/身份去重
        # （检测本身已在 pipe.step 里跑；此处只做展示层。无 face_embed 时=人脸截取模式）
        if is_face_chain:
            faces = list(getattr(pipe, "last_faces", []) or [])
            face_stage = ("face_embed" if "face_embed" in _stage_order else
                          "face_detect" if "face_detect" in _stage_order else
                          "extract_faces" if "extract_faces" in _stage_order else
                          (_stage_order[-1] if _stage_order else None))
            new_faces = 0
            for f in faces:
                data = f.to_dict()
                if femb is not None:
                    vec = femb.embed_face(f.face_img)
                    vec = vec / (np.linalg.norm(vec) or 1.0)
                    best = max((float(v @ vec) for v in face_vecs), default=0.0)
                    data["best_sim"] = round(best, 3)
                    if best >= face_dedup_thresh:
                        face_dup += 1
                        _emit("info", f"重复人脸（相似 {best:.2f}）", ts, stage=face_stage,
                              frame_no=frame_no, data=data, log_line=True, prefix="    ")
                        continue
                    face_vecs.append(vec)
                face_count += 1
                new_faces += 1
                fname = output_dir / f"face_{face_count:03d}_{ts:.1f}s.{image_format}"
                _save_image(fname, f.face_img)
                _emit("face", f"{'新人脸' if femb is not None else '人脸'} #{face_count}"
                            f"（{f.face_img.shape[1]}x{f.face_img.shape[0]}）",
                      ts, stage=face_stage, frame_no=frame_no, data=data,
                      log_line=True, prefix="    ")
                if len(images) < 36:
                    images.append({"title": f"{'新人脸' if femb is not None else '人脸'} "
                                            f"#{face_count} t={ts:.1f}s",
                                   "data": _img_data_url(f.face_img, 90)})
            unit = "新身份" if femb is not None else "人脸"
            _emit("info", f"检出人脸 {len(faces)} 张，{unit} {new_faces} 个",
                  ts, stage=face_stage, frame_no=frame_no, log_line=True, prefix="    ")

        # 链路尾通用输出：末段记录携带的事件（det_filter/seq_filter 追加；不绑定模块）
        for ev in (getattr(pipe, "last_events", None) or []):
            event_count += 1
            nm = ev.get("name", "事件")
            det = ev.get("detail") or {}
            _emit("submit", f"事件 {nm}" + (f" {det}" if det else ""), ts,
                  stage="seq_filter", frame_no=frame_no, log_line=True, prefix="    ")
        log.append("")  # 帧块之间空行，多换行更易读
    
    # 初始化计数器
    processed_count = 0
    detection_count = 0
    event_count = 0
    plate_count = 0
    plate_saved = [0]  # 已落盘原分辨率车牌小图数（文件名编号用）
    
    # 图片测试模式
    image_name = str(p.get("image", "")).strip()
    if image_name:
        img_path = _find_file(image_name, [PIC_DIR, TEST_IMGS])
        if not img_path:
            return res(ok=False, error=f"找不到图片: {image_name}")

        if pipe.yolo is None and pipe.plate is None:
            return res(ok=False, error="该链路无 YOLO/车牌阶段，图片测试仅适用于含检测的链路")

        img_path, frame = _load_frame({"image": image_name})

        # 保存原图
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_name = output_dir / f"frame_000_{timestamp}_orig.{image_format}"
        _save_image(orig_name, frame)

        dets = pipe.yolo.infer(frame, track=pipe.track_needed()) if pipe.yolo else []

        subs, al = pipe.step(0, 0.0, dets, frame=frame)
        subs = _apply_vlm_inputs(pipe, subs)
        _step_out(0, 0.0, f"图片测试: {img_path.name}", dets, subs, al, frame=frame)

        if is_face_chain:
            # 人脸链图片测试：与普通链同走 pipe.step；结果头沿用「新身份/人脸」文案
            processed_count += 1
            n_faces = len(getattr(pipe, "last_faces", []) or [])
            unit = "新身份" if femb is not None else "人脸"
            head = f"链路: {spec.get('name') or '未命名'} | 图片: {img_path.name}"
            head += f" | 检出人脸 {n_faces} 张 | {unit} {face_count} 个"
            if femb is not None:
                head += f"（阈值 {face_dedup_thresh}）"
            head += f" | 输出到: {output_dir}" + _save_summary()
            return res(text=head + "\n" + "\n".join(log), images=images, timeline=timeline)

        # 保存标注图（检出框 + 车牌框）
        canvas = frame.copy()
        _draw_tracks(canvas, dets)
        for p in getattr(pipe, "last_plates", []):
            px1, py1, px2, py2 = p.bbox
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(canvas, p.plate_no, (px1, max(py1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        anno_name = output_dir / f"frame_000_{timestamp}_annotated.{image_format}"
        _save_image(anno_name, canvas)

        images.append({"title": f"原始帧 {image_name}", "data": _img_data_url(canvas, 70)})
        
        head = f"链路: {spec.get('name') or '未命名'} | 图片: {img_path.name}"
        head += f" | 检出 {detection_count} 个 | 事件 {event_count} 个"
        head += " | 输出到: " + str(output_dir) + _save_summary()
        
        return res(text=head + "\n" + "\n".join(log), images=images, timeline=timeline)

    # 视频测试模式
    video_name = str(p.get("video", "")).strip()
    if not video_name:
        return res(ok=False, error="请指定测试视频或图片")
    
    vid = _find_file(video_name, [VID_DIR])
    if not vid:
        return res(ok=False, error=f"找不到视频: {video_name}")
    
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        cap.release()
        return res(ok=False, error=f"无法打开视频: {vid}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算处理范围
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else total_frames
    end_frame = min(end_frame, total_frames)
    
    # 跳到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    n = start_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n >= end_frame:
            break
        if max_frames and (n - start_frame) >= max_frames:
            break
        
        # 帧管理取图节奏由链路内部 FrameManager 控制
        ts = n / fps
        if not pipe.frame.wants_frame(n, ts):
            n += 1
            continue

        processed_count += 1
        frame_num = n - start_frame
        time_str = f"{ts:.2f}s"
        tag = f"时间{time_str} (处理第{frame_num+1}帧)"

        dets = None
        if pipe.yolo is not None:
            dets = pipe.yolo.infer(frame, track=pipe.track_needed())

        subs, al = pipe.step(n, ts, dets, frame=frame, force=True)
        subs = _apply_vlm_inputs(pipe, subs)
        _step_out(n, ts, tag, dets, subs, al, frame=frame)

        # 保存关键帧图片（检出框 + 车牌框）；人脸链只落人脸图（在 _step_out 里），不回灌关键帧
        if not is_face_chain:
            plates = getattr(pipe, "last_plates", [])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if (dets and len(dets) > 0) or plates:
                canvas = frame.copy()
                _draw_tracks(canvas, dets or [])
                for p in plates:
                    px1, py1, px2, py2 = p.bbox
                    cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(canvas, p.plate_no, (px1, max(py1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                anno_name = output_dir / f"frame_{frame_num:04d}_{timestamp.replace(':', '')}_{time_str.replace('.', '_')}_det.{image_format}"
                if not _save_image(anno_name, canvas) and dropped_saves[0] == 1:
                    timeline.append({"kind": "info",
                                     "text": f"◌ 落盘达上限（{max_saved_images} 张），后续图片不再保存",
                                     "ts": ts, "frame": processed_count})

                if len(images) < 6:  # 限制显示的图片数量
                    images.append({"title": f"帧#{frame_num} 时间{time_str} 检出{len(dets or [])}个/车牌{len(plates)}",
                                   "data": _img_data_url(canvas, 65)})

        n += 1

    cap.release()

    if is_face_chain:
        head = f"链路: {spec.get('name') or '未命名'} | 视频: {vid.name}"
        head += f" | 范围: {start_time}s-{end_time}s ({start_frame}-{end_frame}帧)"
        head += f" | 处理 {processed_count} 帧"
        if femb is not None:
            head += f" | 新身份 {face_count} 个 | 重复丢弃 {face_dup} 次 | 去重阈值 {face_dedup_thresh}"
        else:
            head += f" | 截取人脸 {face_count} 张"
        head += f" | 输出到: {output_dir}" + _save_summary()
        return res(text=head + "\n" + "\n".join(log), images=images, timeline=timeline)

    head = f"链路: {spec.get('name') or '未命名'} | 视频: {vid.name}"
    head += f" | 范围: {start_time}s-{end_time}s ({start_frame}-{end_frame}帧)"
    head += f" | 处理 {processed_count} 帧 | 检出 {detection_count} 个 | 车牌 {plate_count} 次 | 事件 {event_count} 个"
    if vlm_unreachable:
        vlm_state = "端点不可达，已跳过"
    elif use_real_vlm:
        vlm_state = "真实调用"
    else:
        vlm_state = "模拟回填(" + vlm_action + ")"
    head += f" | VLM: {vlm_state}"
    if vlm_fail:
        head += f"（失败 {vlm_fail} 次）"
    head += f" | 输出到: {output_dir}"
    head += _save_summary()

    return res(text=head + "\n" + "\n".join(log), images=images, timeline=timeline)


# ---------------- 模块链路运行器（runner=chain，模块链路模式） ----------------
# 已完成链路 spec 以 stages（模块链路）为真相源：各阶段存 web_lab 模块参数，
# _chain_from_spec 据此组装四模块实例（帧管理→YOLO→VLM→告警，可选阶段）。


# _chain_from_spec 能装配进执行路径的阶段（validate_chain「声明即执行」断言的唯一依据）：
# 新增可装配阶段务必同步登记，否则 spec 声明了它却静默不执行（WO-0001 事故即此类）。
CHAIN_ASSEMBLED_MODULES = {
    "frame_manager", "yolo", "det_filter", "seq_filter", "target_crop",
    "annotate", "vlm", "plate_recog", "face_detect", "extract_faces", "face_embed",
}


# ---- 产物端口协议（P5 第 2 步，设计稿 doc/design/assembly_ports.md §3）----
# 设计器可连线端口 / validate_chain DAG 硬校验端口，**单一来源**：
#   ① Pipeline.step 已纳入 inputs 解析的端口 = pipe.composer.RESOLVED_PORTS（唯一权威，
#      与 step() 中 _port(...) 调用点一一对应；含 yolo/det_filter/seq_filter/target_crop/
#      annotate/plate_recog/face_detect）；
#   ② web 驱动层额外解析的端口 = _CHAIN_DRIVER_PORTS（不经 Pipeline.step）。
# 别在本文件硬编码模块端口清单——改核心时同步 pipe.composer.RESOLVED_PORTS 即可。
from pipe.composer import RESOLVED_PORTS as _COMPOSER_RESOLVED_PORTS  # noqa: E402
_CHAIN_DRIVER_PORTS: dict = {
    # vlm.dets：h_pipeline_run / h_pipeline_stream / h_chain 按 inputs 从 pipe.artifacts
    # 取检测记录构造送审（见 _apply_vlm_inputs）；vlm.image 不开放（图片源统一由
    # image_source 管，避免双机制——校验器对其报错）。
    "vlm": ("dets",),
}
CHAIN_RESOLVED_PORTS: dict = {**{m: tuple(p) for m, p in _COMPOSER_RESOLVED_PORTS.items()},
                              **{m: tuple(p) for m, p in _CHAIN_DRIVER_PORTS.items()}}

# module_id → (模块路径, 类名)：读取自描述 INPUTS/OUTPUTS 声明（惰性导入，纯声明不加载模型）
_PORT_DECL_SOURCES = {
    "frame_manager": ("pipe.composer", "FrameManager"),
    "yolo": ("pipe.composer", "YOLODetector"),
    "vlm": ("pipe.composer", "VLMAnalyzer"),
    "det_filter": ("pipe.det_filter", "DetFilter"),
    "seq_filter": ("pipe.seq_filter", "SeqFilter"),
    "target_crop": ("pipe.target_crop", "TargetCrop"),
    "annotate": ("pipe.annotate", "Annotator"),
    "face_detect": ("face_detect.face_pipeline", "FaceStage"),
    "plate_recog": ("plate_recog.plate_pipeline", "PlateStage"),
}
_PORT_DECL_CACHE: dict = {}


def module_port_decls() -> dict:
    """{module_id: {"inputs": {端口: 类型}, "outputs": {产物: 类型}}}——模块自描述端口。

    惰性导入各模块类读取类属性 INPUTS/OUTPUTS（纯声明，不加载模型），结果缓存。
    供 /api/schema（设计器连线 UI）与 validate_chain（DAG 校验类型匹配）复用，单一来源。
    """
    if _PORT_DECL_CACHE:
        return _PORT_DECL_CACHE
    for mid, (modpath, clsname) in _PORT_DECL_SOURCES.items():
        try:
            cls = getattr(importlib.import_module(modpath), clsname)
            _PORT_DECL_CACHE[mid] = {
                "inputs": dict(getattr(cls, "INPUTS", {}) or {}),
                "outputs": dict(getattr(cls, "OUTPUTS", {}) or {}),
            }
        except Exception:  # 导入失败不阻断 schema/校验
            _PORT_DECL_CACHE[mid] = {"inputs": {}, "outputs": {}}
    return _PORT_DECL_CACHE


def artifact_types() -> dict:
    """产物名 → 类型（唯一来源：pipe.composer.ARTIFACT_TYPES）。"""
    from pipe.composer import ARTIFACT_TYPES
    return dict(ARTIFACT_TYPES)


# 声明了 INPUTS 但未纳入解析的阶段（写了 inputs → 非致命警告，运行按默认取数）。
# **推导**而非硬编码：取各模块自描述 INPUTS（module_port_decls）中不在 CHAIN_RESOLVED_PORTS
# 的模块。本期全部已声明端口的模块均已解析（含 plate_recog/face_detect/vlm.dets）→ 为空元组；
# 未来新增自描述模块而未同步解析时会自动进入此名单并触发警告。
CHAIN_UNRESOLVED_INPUT_MODULES: tuple = tuple(sorted(
    m for m, d in module_port_decls().items()
    if d.get("inputs") and m not in CHAIN_RESOLVED_PORTS))


def _chain_from_spec(spec: dict):
    """按链路 spec 的 stages 组装 Pipeline（模块链路的装配器）。

    YOLO 惰性不加载；含 face_detect / face_embed 阶段时会构造人脸阶段
    （FaceDetector/FaceEmbedder 构造即加载模型）。spec 声明的每个阶段都必须在此被
    装配进执行路径（validate_chain 的「声明即执行」断言依据 CHAIN_ASSEMBLED_MODULES）。
    """
    from pipe.composer import (
        FrameManager, YOLODetector, VLMAnalyzer, Pipeline,
        SAMPLING_ANALYSIS, SAMPLING_WALL_CLOCK, SAMPLING_FRAME_COUNT,
        backpressure_params,
    )
    stages = spec.get("stages") or []
    by_mod: dict = {}
    for s in stages:
        mid = s.get("module_id")
        if mid:
            by_mod.setdefault(mid, []).append(s.get("params") or {})

    def _one(mid: str) -> dict:
        return (by_mod.get(mid) or [{}])[0]

    # 端口连线：解析各 stage 的可选字段 "inputs": {端口: 上游产物名} → Pipeline.stage_inputs。
    # 缺失/空/非对象 → 不传（Pipeline 默认 {} = 现状规则，零回归）。值只收字符串
    # （每端口单一上游；非字符串交给 validate_chain 报错，这里防御性跳过）。
    stage_inputs: dict = {}
    for s in stages:
        mid = s.get("module_id")
        ins = s.get("inputs")
        if not mid or not isinstance(ins, dict) or not ins:
            continue
        clean = {str(port): art for port, art in ins.items()
                 if isinstance(port, str) and isinstance(art, str) and art.strip()}
        if clean:
            stage_inputs[mid] = clean

    if "frame_manager" not in by_mod:
        raise ValueError("链路缺少必需阶段: frame_manager")
    if not any(k in by_mod for k in ("face_embed", "face_detect", "plate_recog",
                                      "det_filter", "seq_filter")):
        raise ValueError("链路缺少必需阶段: 终段需为 det_filter / seq_filter / "
                         "face_embed / face_detect / plate_recog 之一")

    # 字符串参数统一走模块级 _str_param（None/空白 → 默认值）

    fp = _one("frame_manager")
    fm = FrameManager(
        sampling=_str_param(fp, "sampling", SAMPLING_ANALYSIS),
        frame_skip=_int(fp, "frame_skip", 3),
        interval_sec=_float(fp, "interval_sec", 3.0),
        interval_frames=_int(fp, "interval_frames", 30),
        **backpressure_params(_str_param(fp, "backpressure", "standard")))
    yolo = None
    det_filter = None
    if "det_filter" in by_mod:
        # 检测筛选阶段（有效性过滤的唯一来源；group_limits 空=不判定）
        det_filter = _det_filter_of(_one("det_filter"))
    if "yolo" in by_mod:
        yp = _one("yolo")
        model = _find_file(_str_param(yp, "model_path"), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
        # YOLO 纯推理；节流/track 抑制在 Pipeline 编排
        yolo = YOLODetector(
            model_path=str(model), conf=_float(yp, "conf", 0.35),
            iou=_float(yp, "iou", 0.7), imgsz=_int(yp, "imgsz", 640),
            max_det=_int(yp, "max_det", 300), classes=_int_list(yp, "classes", None),
            device=(_str_param(yp, "device") or None))
    vlm = None
    if "vlm" in by_mod:
        vp = _one("vlm")
        vlm = VLMAnalyzer(
            max_resolution=tuple(_int_tuple(vp, "max_resolution", 2) or (1280, 720)),
            prompt=_str_param(vp, "prompt"))
    crop = None
    if "target_crop" in by_mod:
        from pipe.target_crop import TargetCrop
        cp = _one("target_crop")
        crop = TargetCrop(margin=_float(cp, "margin", 0.0),
                          classes=_int_list(cp, "classes", None),
                          filter=_str_param(cp, "filter"),
                          out_size=_str_param(cp, "out_size"))
    annotate = None
    if "annotate" in by_mod:
        from pipe.annotate import Annotator
        anp = _one("annotate")
        annotate = Annotator(thickness=_int(anp, "thickness", 2),
                             show_label=_bool(anp, "show_label"),
                             classes=_int_list(anp, "classes", None),
                             filter=_str_param(anp, "filter"))
    # 人脸阶段（face_detect / 兼容旧 extract_faces）：构造 FaceStage 注入 Pipeline.face
    # （run(frame, crops, ts) -> list[FaceResult]，与 plate 对称）；face_embed 阶段构造
    # FaceEmbedder（身份去重在驱动层读 last_faces 后处理，落盘/计数同样留驱动层）。
    face = None
    face_embedder = None
    face_dedup_thresh = 0.55
    if "face_detect" in by_mod or "extract_faces" in by_mod:
        from face_detect.face_detector import FaceDetector
        from face_detect.face_pipeline import FaceStage
        from pipe.cropper import Cropper
        edp = _one("extract_faces") if "extract_faces" in by_mod else {}  # 兼容旧形态
        fdp = _one("face_detect") if "face_detect" in by_mod else {}
        # 检测器参数：有 face_detect 阶段用它的；否则用 extract_faces 的 face_thresh
        det_src = fdp if "face_detect" in by_mod else edp
        fdet = FaceDetector(
            device=_str_param(det_src, "device", "auto"),
            det_thresh=(_float(det_src, "det_thresh", 0.0)
                        or _float(det_src, "face_thresh", 0.4)),
            det_size=tuple(_int_tuple(det_src, "det_size", 2) or (640, 640)),
            use_crop=_bool(det_src, "use_crop"))
        # 人脸检测阶段：出脸 margin / 去重 IoU（新形态在 face_detect；兼容旧 extract_faces）
        f_margin = _float(fdp, "margin", 0.0) or _float(edp, "margin", 0.15)
        f_dedup_iou = _float(fdp, "dedup_iou", 0.0) or _float(edp, "dedup_iou", 0.6) or 0.6
        face = FaceStage(fdet, Cropper(upscale=1.0, margin=f_margin), dedup_iou=f_dedup_iou)
        if "face_embed" in by_mod:
            from face_embed.face_embedder import FaceEmbedder
            fep = _one("face_embed")
            face_embedder = FaceEmbedder(device=_str_param(fep, "device", "auto"))
            face_dedup_thresh = _float(fep, "thresh", 0.55) or 0.55
        # 兼容旧 extract_faces 形态：链路未配「目标裁剪」/「YOLO」阶段时按 extract_faces 参数补齐
        if crop is None:
            from pipe.target_crop import TargetCrop
            crop = TargetCrop(scale=_float(edp, "upscale", 1.0) or 1.0, margin=0.0,
                              classes=_int_list(edp, "yolo_classes", [0]) or [0])
        if yolo is None:
            model = (_find_file(_str_param(edp, "yolo_model"), [MODELS_DIR])
                     or MODELS_DIR / "yolov8n.pt")
            yolo = YOLODetector(
                model_path=str(model), conf=_float(edp, "yolo_conf", 0.35),
                iou=_float(edp, "yolo_iou", 0.7), imgsz=_int(edp, "yolo_imgsz", 640),
                max_det=_int(edp, "yolo_max_det", 300),
                classes=_int_list(edp, "yolo_classes", [0]))
    plate = None
    if "plate_recog" in by_mod:
        from plate_recog.plate_recognizer import PlateRecognizer
        from plate_recog.plate_pipeline import PlateStage
        from pipe.cropper import Cropper
        pp = _one("plate_recog")
        tool = Cropper(upscale=1.0, margin=_float(pp, "margin", 0.1))
        td = pp.get("track_dedup")
        plate = PlateStage(
            PlateRecognizer(detect_level=_str_param(pp, "detect_level", "high")),
            tool, mode=_str_param(pp, "mode", "crop"),
            dedup_iou=_float(pp, "dedup_iou", 0.6),
            track_dedup=(True if td in (None, "") else _bool(pp, "track_dedup")),
            track_cooldown=_float(pp, "track_cooldown", 5.0))
    # 时序判定阶段（一个 module_id）：参数按用途拆两个实例——
    # gates（throttle/change）→ 检测流门控 Pipeline.seq_filter；
    # rules（SeqRule 规格）→ 事件尾 Pipeline.event_filter（VLM/车牌记录经 publish 走它出事件）。
    seq_filter = None
    event_filter = None
    if "seq_filter" in by_mod:
        sp = _one("seq_filter")
        seq_filter = _seq_filter_of(sp, what="gates")
        event_filter = _seq_filter_of(sp, what="rules")
    pipe = Pipeline(frame=fm, yolo=yolo, vlm=vlm,
                    crop=crop, annotate=annotate, plate=plate, face=face,
                    det_filter=det_filter, seq_filter=seq_filter, event_filter=event_filter,
                    stage_inputs=stage_inputs)
    # face_embed 身份去重所需（驱动层读 pipe.last_faces 后处理）：非 dataclass 字段，装配层附挂
    pipe.face_embedder = face_embedder
    pipe.face_dedup_thresh = face_dedup_thresh
    return pipe


# ---------------- 实时播放叠加：SSE 流式链路运行（方案 B） ----------------
# 逐分析帧推框的数值（不传图，浏览器本地播原视频 + canvas 叠加）；前端按
# currentTime×fps 定位帧号并"播到最新结果时间就暂停等结果"，实现框与画面同步。
_STREAMS: dict = {}  # run_id -> threading.Event（停止信号）


def h_pipeline_stream(write, params: dict) -> None:
    """SSE 流式运行一条链路：write(event, payload) 逐条推送。

    事件：meta（fps/帧范围/阶段）→ frame（n/ts/dets/plates/alarms）→ done。
    说明：为不阻塞流，本模式不做真实 VLM 调用（mock 回填由 vlm_feedback 控制）；
    需要真实研判请用批量运行。停止：POST /api/test/pipeline_run/stop {run_id}。
    """
    if not _DB_ENABLED:
        write("error", {"message": "数据库未启用（--no-db）"})
        return
    pipeline_id = _str_param(params, "pipeline_id")
    if not pipeline_id:
        write("error", {"message": "请指定测试链路（pipeline_id）"})
        return
    try:
        spec = db.load_pipeline(pipeline_id)
    except Exception as e:
        write("error", {"message": f"加载链路失败: {e}"})
        return
    video_name = _str_param(params, "video")
    vid = _find_file(video_name, [VID_DIR])
    if not vid:
        write("error", {"message": f"找不到视频: {video_name}"})
        return
    try:
        pipe = _chain_from_spec(spec)
    except Exception as e:
        write("error", {"message": f"装配链路失败: {e}"})
        return

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        cap.release()
        write("error", {"message": f"无法打开视频: {vid}"})
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = _float(params, "start_time", 0.0)
    end_time = _float(params, "end_time", 0.0)
    max_frames = _int(params, "max_frames", 0)
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps) if end_time > 0 else total, total)
    fm_params = next((s.get("params") or {} for s in spec.get("stages") or []
                      if s.get("module_id") == "frame_manager"), {})
    vlm_feedback = _bool(params, "vlm_feedback")
    vlm_action = _str_param(params, "vlm_action", "fire")

    run_id = uuid.uuid4().hex[:12]
    stop_evt = threading.Event()
    _STREAMS[run_id] = stop_evt
    write("meta", {"run_id": run_id, "name": spec.get("name"), "fps": fps,
                   "total_frames": total, "start_frame": start_frame,
                   "end_frame": end_frame,
                   "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   "frame_skip": _int(fm_params, "frame_skip", 1),
                   "stages": [s.get("module_id") for s in spec.get("stages") or []]})

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = detection_count = plate_count = event_total = 0
    t0 = time.time()
    n = start_frame
    try:
        while True:
            if stop_evt.is_set():
                break
            ok, frame = cap.read()
            if not ok or n >= end_frame:
                break
            if max_frames and processed >= max_frames:
                break
            ts = n / fps
            if not pipe.frame.wants_frame(n, ts):  # 帧管理取图节奏
                n += 1
                continue
            dets = pipe.yolo.infer(frame, track=pipe.track_needed()) if pipe.yolo is not None else []
            subs, evs = pipe.step(n, ts, dets, frame=frame, force=True)
            subs = _apply_vlm_inputs(pipe, subs)
            if vlm_feedback and pipe.vlm is not None:
                for s in subs:  # 模拟回填（真实 VLM 见函数说明）
                    evs += pipe.publish([{"kind": "vlm", "label": vlm_action, "text": ""}], ts, n)
            plates = list(getattr(pipe, "last_plates", []))
            processed += 1
            detection_count += len(dets or [])
            plate_count += len(plates)
            event_total += len(evs)
            write("frame", {
                "n": n, "ts": round(ts, 3),
                "dets": [{"cls": d.cls_id, "bbox": list(d.bbox),
                          "score": round(float(d.score), 3),
                          "track": int(getattr(d, "track_id", 0) or 0)} for d in (dets or [])],
                "plates": [{"plate_no": p.plate_no, "type_name": p.type_name, "color": p.color,
                            "score": round(float(p.score), 3), "bbox": list(p.bbox),
                            "car_bbox": list(p.car_bbox) if p.car_bbox else None}
                           for p in plates],
                "alarms": [{"action": e.get("name", "事件"), "desc": str(e.get("detail") or ""),
                            "conf": 1.0} for e in evs],
            })
            n += 1
    finally:
        cap.release()
        _STREAMS.pop(run_id, None)
    write("done", {"run_id": run_id, "processed": processed, "detections": detection_count,
                   "plates": plate_count, "alarms": event_total,
                   "elapsed": round(time.time() - t0, 2), "stopped": stop_evt.is_set()})


def h_chain(p: dict) -> dict:
    """模块链路运行器：按 spec.stages 组装模块，script/real 驱动逐帧 step()。"""
    from pipe.composer import Detection
    spec = p.get("spec") or {}
    if not spec.get("stages"):
        return res(ok=False, error="链路 spec 缺少 stages（请先载入一条已完成链路）")
    try:
        pipe = _chain_from_spec(spec)
    except ValueError as e:
        return res(ok=False, error=str(e))
    feedback = _bool(p, "vlm_feedback")
    vlm_action = str(p.get("vlm_action", "fire")).strip() or "fire"
    log, images = [], []
    _so = [s.get("module_id") for s in spec.get("stages") or [] if s.get("module_id")]
    _nx = {_so[i]: _so[i + 1] for i in range(len(_so) - 1)}

    def _tag(stage):
        return f"[{_stage_label(stage)}] "

    def _step_out(frame_no, ts, dets, subs, al):
        """逐模块展示本帧处理内容（不管是否向下传递），每条带帧序数。"""
        if "frame_manager" in _so:
            log.append(_tag("frame_manager") + f"帧#{frame_no} 放行 ts={ts:.2f}s")
        if "det_filter" in _so:
            fi = getattr(pipe, "last_filter", None)
            if fi is not None and fi[0] != fi[1]:
                log.append("    " + _tag("det_filter")
                           + f"帧#{frame_no} 有效性筛选 {fi[0]} → {fi[1]}（筛掉 {fi[0] - fi[1]} 条）")
        if "yolo" in _so:
            by_cls = {}
            for d in dets or []:
                by_cls[d.cls_id] = by_cls.get(d.cls_id, 0) + 1
            desc = "、".join(f"cls{c}×{n}" for c, n in sorted(by_cls.items())) or "无"
            parts = [f"检出 {len(dets or [])}（{desc}）"]
            if subs:
                parts.append("报送 " + "、".join(f"{s.ref} 目标{len(s.dets)}" for s in subs))
            log.append("    " + _tag("yolo") + f"帧#{frame_no} " + "；".join(parts))
        if "target_crop" in _so:
            crops = getattr(pipe, "last_crops", [])
            tids = [c.track_id for c in crops if getattr(c, "track_id", 0)]
            log.append("    " + _tag("target_crop") + f"帧#{frame_no} 裁出 {len(crops)} 张子图"
                       + (f"（track {tids}）" if tids else ""))
        if "vlm" in _so and pipe.vlm is not None:
            if not subs:
                log.append("    " + _tag("vlm") + f"帧#{frame_no} 无报送，未处理")
            else:
                for s in subs:
                    if feedback:
                        log.append("    " + _tag("vlm") + f"帧#{frame_no} 研判 {vlm_action}（模拟回填）")
                        for ev in pipe.publish([{"kind": "vlm", "label": vlm_action, "text": ""}],
                                               ts, frame_no):
                            log.append("    " + _tag("seq_filter")
                                       + f"帧#{frame_no} 事件 {ev.get('name')}")
                    else:
                        log.append("    " + _tag("vlm") + f"帧#{frame_no} 素材包 {s.ref}（未开启研判回填）")
        if "plate_recog" in _so:
            plates = getattr(pipe, "last_plates", [])
            if plates:
                for p in plates:
                    log.append("    " + _tag("plate_recog")
                               + f"帧#{frame_no} 识别 {p.plate_no} {p.type_name}/{p.color} {p.score:.2f}")
            else:
                log.append("    " + _tag("plate_recog") + f"帧#{frame_no} 识别 0 张")

    if p.get("feed") == "video":
        # 真实视频驱动：按帧管理节奏抽帧 → YOLO 推理（dwell 模式 track=True 保
        # track_id 稳定）→ 逐步 step()
        vid = _find_file(p.get("video", ""), [VID_DIR])
        if not vid:
            return res(ok=False, error="请选择测试视频（视频必选）")
        max_frames = _int(p, "max_frames", 0)
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            cap.release()
            return res(ok=False, error=f"无法打开视频: {vid}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = analyzed = total_det = total_plate = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if max_frames and n > max_frames:
                break
            ts = n / fps
            if not pipe.frame.wants_frame(n, ts):
                continue
            analyzed += 1
            dets = None
            if pipe.yolo is not None:
                dets = pipe.yolo.infer(frame, track=pipe.track_needed())
            subs, al = pipe.step(n, ts, dets, frame=frame, force=True)
            subs = _apply_vlm_inputs(pipe, subs)
            total_det += len(dets or [])
            total_plate += len(getattr(pipe, "last_plates", []))
            _step_out(n, ts, dets, subs, al)
            if len(images) < 4:
                canvas = frame.copy()
                _draw_tracks(canvas, dets or [])
                for p in getattr(pipe, "last_plates", []):
                    px1, py1, px2, py2 = p.bbox
                    cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(canvas, p.plate_no, (px1, max(py1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if canvas.shape[1] > 1280:
                    k = 1280 / canvas.shape[1]
                    canvas = cv2.resize(canvas, (1280, int(canvas.shape[0] * k)))
                images.append({"title": f"帧#{n} 检出 {len(dets or [])} 个（分析第 {analyzed} 帧）",
                               "data": _img_data_url(canvas, 70)})
        cap.release()
        head = f"链路: {spec.get('name') or spec.get('description') or '未命名'} | 视频: {vid.name}"
        head += f" | 读取帧 {n} | 分析帧 {analyzed} | 检出 {total_det} 个 | 车牌 {total_plate} 次"
        return res(text=head + "\n" + "\n".join(log), images=images)
    elif p.get("feed") == "real":
        if pipe.yolo is None and pipe.plate is None:
            return res(ok=False, error="该链路无 YOLO/车牌阶段，真实图片驱动仅适用于含检测的链路")
        img_path, frame = _load_frame(p)
        dets = pipe.yolo.infer(frame, track=pipe.track_needed()) if pipe.yolo else None
        canvas = frame.copy()
        _draw_tracks(canvas, dets or [])
        subs, al = pipe.step(0, 0.0, dets, frame=frame)
        subs = _apply_vlm_inputs(pipe, subs)
        for pl in getattr(pipe, "last_plates", []):
            px1, py1, px2, py2 = pl.bbox
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(canvas, pl.plate_no, (px1, max(py1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        images.append({"title": "检测标注（蓝=目标框, 绿=车牌框）", "data": _img_data_url(canvas)})
        _step_out(0, 0.0, dets, subs, al)
    else:
        script = str(p.get("script", ""))
        lines = [l for l in script.splitlines() if l.strip() and not l.strip().startswith("#")]
        for l in lines:
            parts = [x.strip() for x in re.split(r"[,，\s]+", l) if x.strip()]
            if len(parts) < 4:
                continue
            frame_idx, ts = int(parts[0]), float(parts[1])
            cls, count = int(parts[2]), int(parts[3])
            start_track = int(parts[4]) if len(parts) > 4 else 0
            dets = [Detection(track_id=start_track + t, cls_id=cls, bbox=(10, 10, 100, 200),
                              score=0.9, model_path="script") for t in range(count)]
            subs, al = pipe.step(frame_idx, ts, dets)
            subs = _apply_vlm_inputs(pipe, subs)
            _step_out(frame_idx, ts, dets, subs, al)
    head = f"链路: {spec.get('name') or spec.get('description') or '未命名'}"
    return res(text=head + "\n" + "\n".join(log), images=images)


# ---------------- 模块清单（前端表单的"参数即表单"来源） ----------------

PARAMS = {
    "frame_manager": {
        "sampling": {"label": "取图节奏", "type": "select", "default": "analysis",
                     "options": ["analysis", "wall_clock", "frame_count"],
                     "help": "analysis=随抽帧节奏(含背压); wall_clock=按墙钟秒; frame_count=按帧数"},
        "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                       "help": "每 N 帧取一张（analysis 模式用，积压时自动放宽）"},
        "interval_sec": {"label": "墙钟间隔（秒）", "type": "float", "default": 3.0,
                         "help": "每隔多少秒取一张（wall_clock 模式用）"},
        "interval_frames": {"label": "帧数间隔（帧）", "type": "int", "default": 30,
                            "help": "每隔多少帧取一张（frame_count 模式用）"},
        "mode": {"label": "背压模式（analysis）", "type": "select", "default": "queue",
                 "options": ["queue", "adaptive"], "test_only": True,
                 "help": "queue=按积压队列信号放宽; adaptive=按推理耗时自动放宽"},
        "scenario": {"label": "模拟场景", "type": "select", "default": "平稳",
                     "options": ["平稳", "积压浪涌", "耗时渐增"], "test_only": True,
                     "help": "queue 模式用「积压浪涌」造高峰；adaptive 用「耗时渐增」"},
        "backpressure": {"label": "背压档位", "type": "select", "default": "standard",
                         "options": ["off", "standard", "aggressive"],
                         "help": "off=不放宽; standard=标准; aggressive=更激进（积压/耗时高时更快放宽）"},
        "fps": {"label": "视频帧率（预算用）", "type": "float", "default": 25.0,
                "test_only": True,
                "help": "用来估算每帧的推理时间预算（adaptive 模式用）"},
        "n_frames": {"label": "模拟帧数", "type": "int", "default": 20,
                     "test_only": True,
                     "help": "模拟播放多少帧，用来观察逐帧取/跳决策"},
        "ts_step": {"label": "每帧时间步长（秒）", "type": "float", "default": 1.0,
                    "test_only": True,
                    "help": "模拟时间轴：相邻两帧相差几秒"},
    },
    "yolo": {
        "model_path": {"label": "权重模型", "type": "file", "src": "model",
                       "help": "检测权重；不选则用默认 yolov8n.pt"},
        "image": {"label": "测试图片", "type": "file", "src": "img", "test_only": True,
                  "help": "要检测的图片，从下拉选"},
        "conf": {"label": "检出阈值 conf", "type": "float", "default": 0.35,
                 "help": "置信度低于它的框会被丢弃；调高更严格"},
        "iou": {"label": "NMS 阈值 iou", "type": "float", "default": 0.7,
                "help": "重叠超过它的重复框会合并成一个"},
        "imgsz": {"label": "推理分辨率 imgsz", "type": "int", "default": 640,
                  "help": "送进模型的分辨率；越大越慢，小目标更清楚"},
        "max_det": {"label": "单帧最多目标 max_det", "type": "int", "default": 300,
                    "help": "一帧最多保留多少个目标"},
        "classes": {"label": "类别白名单（逗号分隔，空=不限）", "type": "str", "default": "0",
                    "help": "COCO: 0=人, 2=车"},
        "device": {"label": "推理设备（空=自动）", "type": "str", "default": "",
                   "help": "如 cpu 或 0"},
    },
    "det_filter": {
        "filter": {"label": "筛选条件（条件 DSL）", "type": "text", "default": "",
                   "help": "如 'cls_id in 0|2, short >= 30'；组内逗号=并且、分号=或者、not=非、括号分组；"
                            "字段 cls_id/track_id/score + 派生 w/h/short/long/area + ts/帧号；"
                            "聚合 count(字段)/count(子条件)/count()（同帧计数，如 'count(cls_id) >= 2'、"
                            "'count(cls_id == 0, score >= 0.5) >= 2'）；空=不筛选"},
        "group_limits": {"label": "每类数量上下限（如 0:1,300; 2:1,10）", "type": "str",
                         "default": "",
                         "help": "成组数量判定：类别:下限,上限，分号分隔；数量越界/未登记的类别整类视为未识别；"
                                 "空=不判定（等价 count 聚合谓词，见 doc/modules/det_filter.md）"},
        "events": {"label": "事件标注（JSON 列表）", "type": "text", "default": "",
                   "help": "如 [{\"name\":\"警情\",\"when\":\"label in 治安类|交警类\",\"detail\":[\"label\",\"content\"]}]；"
                           "放行记录命中即追加事件（可多事件）；空=不标注。可作链路尾输出"},
        "detections": {"label": "模拟检测（每行一条）", "type": "text", "default": "",
                       "test_only": True,
                       "help": "每行: 类别,track,置信度,x1,y1,x2,y2（如 '0,1,0.9,100,100,200,200'）"},
        "ts": {"label": "帧时间戳（秒）", "type": "float", "default": 0.0,
               "test_only": True,
               "help": "条件里可用 ts（如 'ts >= 10' 表示 10 秒后才开始放行）"},
    },
    "seq_filter": {
        "throttle": {"label": "整类节流（key:秒，如 cls_id:3.0）", "type": "str", "default": "",
                     "help": "同一 key（一般 cls_id）每隔 N 秒才放行整类记录一次；空=不节流。"
                             "等价原 YOLO/Pipeline 的 check_interval"},
        "change": {"label": "变化门控（key:watch，如 cls_id:track_id）", "type": "str", "default": "",
                   "help": "同类 watch 值集合（默认 track_id）相比上次有变化才放行；空=不门控。"
                           "等价原 track_change_only"},
        "rules": {"label": "确认/事件规则（JSON 列表）", "type": "text", "default": "",
                  "help": "SeqRule 规格，如 [{\"match\":\"label in 治安类|交警类\",\"event\":\"{label}\","
                          "\"key\":\"label\",\"cooldown\":30}]；event 支持 {字段} 动态占位，"
                          "cooldown<0=同对象一次，window_n/min_n/hit_ratio 为窗口；空=无规则"},
        "detections": {"label": "模拟检测（每行一条）", "type": "text", "default": "",
                       "test_only": True,
                       "help": "每行: 类别,track,置信度,x1,y1,x2,y2（如 '0,1,0.9,100,100,200,200'）"},
        "ts": {"label": "帧时间戳（秒）", "type": "float", "default": 0.0,
               "test_only": True,
               "help": "门控/规则计时用（同一帧多次调用可改 ts 观察节流）"},
    },
    "vlm": {
        "max_resolution": {"label": "送审缩放上限（宽,高）", "type": "str", "default": "1280,720",
                           "help": "送审图片最大尺寸，超出会缩小"},
        "prompt": {"label": "提示词", "type": "text", "default": "",
                   "help": "留空用默认分类提示词"},
        # 送审图像源（壳内声明"用哪张图"；默认 annotated=现状，低清图本阶段不做）
        "image_source": {"label": "送审图像源", "type": "select", "default": "annotated",
                         "options": ["annotated", "frame", "crops"],
                         "help": "annotated=带框图（默认；无上游标注阶段时回退本帧原图）；"
                                 "frame=本帧原图；crops=与本次报送目标匹配的裁剪子图"
                                 "（匹配不到时回退 annotated/原帧，并记日志）"},
        "image_pick": {"label": "送审张数（图集源）", "type": "select", "default": "first",
                       "options": ["first", "all"],
                       "help": "image_source=crops 且该类命中多个目标时：first=只送首张"
                               "（默认，与改动前一致）；all=送该类全部对象的裁剪子图"
                               "（OpenAI 兼容多图消息；端点支持待联调）"},
        "image": {"label": "测试图片（真实调用用）", "type": "file", "src": "img",
                  "test_only": True,
                  "help": "勾选真实大模型时，从这张图取素材发请求"},
        "ref": {"label": "素材引用", "type": "select", "default": "crop:cls0",
                "options": ["crop:cls0", "full:cls0", "full:global"], "test_only": True,
                "help": "crop:cls0=裁剪第 0 类目标送审；full:cls0=全帧按第 0 类；full:global=全帧全局"},
        "gran": {"label": "报送粒度", "type": "select", "default": "class",
                 "options": ["class", "scene"], "test_only": True,
                 "help": "class=按类别一次; scene=整场景一次"},
        "track_key": {"label": "track_key（类别编号，scene=-1）", "type": "int", "default": 0,
                      "test_only": True,
                      "help": "素材对应的目标类别编号"},
        "bbox": {"label": "单目标 bbox（x1,y1,x2,y2）", "type": "str", "default": "10,10,100,200",
                 "test_only": True,
                 "help": "单个目标的位置，选裁剪送审时用"},
        "det_count": {"label": "该类目标数（无单目标时）", "type": "int", "default": 2,
                      "test_only": True,
                      "help": "没填 bbox 时，用它代表该类目标数量"},
        "use_real_vlm": {"label": "调用真实大模型", "type": "bool", "default": False,
                         "help": "勾选才真发请求；不勾只演示生成素材包"},
        "vlm_endpoint": {"label": "VLM 端点（留空用默认）", "type": "str",
                         "default": "http://117.42.21.253:8000/v1/chat/completions",
                         "help": "大模型接口地址"},
        "vlm_model": {"label": "VLM 模型名", "type": "str", "default": "qwen3-vl-32b",
                      "help": "调用的模型名称"},
        "vlm_key": {"label": "API Key（无 key 留空）", "type": "str", "default": "",
                    "help": "接口密钥，免鉴权可留空"},
    },
    "chain": {
        # 模块链路的通用运行器（runner=chain）：运行表单=「视频输入 + 各阶段参数」，
        # 链路本体（各阶段模块参数）由已完成链路 spec 的 stages 提供，前端运行前并入表单。
        "video": {"label": "测试视频", "type": "file", "src": "vid", "group": "运行控制",
                  "help": "要扫描的视频文件"},
        "max_frames": {"label": "限帧数（0=全部）", "type": "int", "default": 300,
                       "group": "运行控制", "help": "长视频建议先限帧；0=读到结尾"},
        "vlm_feedback": {"label": "VLM 回填（把 action 喂回告警）", "type": "bool", "default": True,
                         "group": "运行控制",
                         "help": "把大模型判定的动作喂给告警策略"},
        "vlm_action": {"label": "回填的 VLM action", "type": "str", "default": "fire",
                       "group": "运行控制", "help": "回填时使用的动作名"},
    },
    "face_detect": {
        "image": {"label": "测试图片", "type": "file", "src": "img", "test_only": True,
                  "help": "要检测的图片，从下拉选"},
        "device": {"label": "设备", "type": "select", "default": "auto",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "det_thresh": {"label": "检测阈值", "type": "float", "default": 0.5,
                       "help": "置信度低于它的脸不认；调低检出更多（也更多误检）"},
        "det_size": {"label": "检测分辨率（宽,高）", "type": "str", "default": "640,640",
                     "help": "越高越慢，小脸更清楚"},
        "max_num": {"label": "最多人脸数（0=不限）", "type": "int", "default": 0,
                    "help": "最多返回几张脸"},
        "use_crop": {"label": "生成对齐裁剪图", "type": "bool", "default": False,
                     "help": "勾选=额外产出 5 点对齐的 112×112 正面脸图（给人脸嵌入用；一般不需要）"},
        "margin": {"label": "人脸出图外扩比例", "type": "float", "default": 0.15,
                   "help": "从原图裁原分辨率人脸时往外多扩多少"},
        "dedup_iou": {"label": "人脸去重 IoU", "type": "float", "default": 0.6,
                      "help": "两张脸重叠超过它算同一张，只留一张"},
    },
    "face_embed": {
        "image": {"label": "测试图片（自动先检测再嵌入）", "type": "file", "src": "img",
                  "test_only": True,
                  "help": "会自动先检测人脸，再对每张脸提取向量"},
        "device": {"label": "设备", "type": "select", "default": "auto",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "det_thresh": {"label": "检测阈值", "type": "float", "default": 0.5,
                       "test_only": True,
                       "help": "置信度低于它的脸不认"},
        "det_size": {"label": "检测分辨率（宽,高）", "type": "str", "default": "640,640",
                     "test_only": True,
                     "help": "越高越慢，小脸更清楚"},
    },
    "face_store": {
        "action": {"label": "操作", "type": "select", "default": "register_dir",
                   "options": ["register_dir", "register_single", "query", "self_check", "list"],
                   "test_only": True,
                   "help": "注册目录 / 注册单张 / 查询相似 / 自检 / 列清单"},
        "db_path": {"label": "底库文件（.npz）", "type": "str", "default": "out/face_db.npz",
                    "help": "底库存到哪；不存在会自动新建"},
        "thresh": {"label": "检索阈值", "type": "float", "default": 0.45,
                   "help": "相似度低于它的不算命中（0~1，越大越严）"},
        "topk": {"label": "Top-K", "type": "int", "default": 5,
                 "help": "查询返回最像的前几个"},
        "device": {"label": "设备", "type": "select", "default": "auto",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "register_dir": {"label": "注册目录（每张图=一个身份）", "type": "str",
                         "default": "umvp/face_detect/test_imgs", "test_only": True,
                         "help": "目录里每张图注册成一个身份，文件名即身份名"},
        "image": {"label": "图片（register_single / query 用）", "type": "file", "src": "img",
                  "test_only": True,
                  "help": "单张注册或查询用的图片"},
        "name": {"label": "身份名（register_single 用）", "type": "str", "default": "",
                 "test_only": True,
                 "help": "注册到库里的身份名字"},
    },
    "target_crop": {
        "out_size": {"label": "输出分辨率（空=原图；640=长边；640,480=严格；×2=倍数）",
                     "type": "str", "default": "",
                     "help": "空=原图；单个数=长边像素等比；宽,高=严格尺寸；×2/2x=缩放倍数"},
        "margin": {"label": "裁剪外扩比例", "type": "float", "default": 0.0,
                   "help": "按框宽高往外多扩多少，避免目标贴边被切"},
        "classes": {"label": "只裁这些类别（逗号分隔，空=全部）", "type": "str", "default": "",
                    "help": "对哪些类别的检测做裁剪（如 2=车，0=人）"},
        "filter": {"label": "框筛选条件（空=全裁）", "type": "str", "default": "",
                   "help": "显式字段名条件：track_id == 1 / cls_id == 2 / "
                           "cls_id in 0|2, score >= 0.5；组内逗号AND、组间分号OR"},
        "image": {"label": "测试图片", "type": "file", "src": "img", "test_only": True,
                  "help": "按框裁图的测试图片"},
        "bbox": {"label": "测试框（x1,y1,x2,y2）", "type": "str", "default": "10,10,200,200",
                 "test_only": True, "help": "测试页按该框裁剪"},
    },
    "annotate": {
        "thickness": {"label": "框线粗细（像素）", "type": "int", "default": 2,
                      "help": "画框的线宽"},
        "show_label": {"label": "显示类别标签", "type": "bool", "default": True,
                       "help": "在框上方标注类别名（人/车…）"},
        "classes": {"label": "只画这些类别（逗号分隔，空=全部）", "type": "str", "default": "",
                    "help": "对哪些类别的检测画框（如 0,2）"},
        "filter": {"label": "框筛选条件（空=全画）", "type": "str", "default": "",
                   "help": "显式字段名条件：track_id == 1 / cls_id == 2 / "
                           "cls_id in 0|2, score >= 0.5；组内逗号AND、组间分号OR"},
        "image": {"label": "测试图片", "type": "file", "src": "img", "test_only": True,
                  "help": "画框测试图片"},
        "bbox": {"label": "测试框（x1,y1,x2,y2）", "type": "str", "default": "10,10,200,200",
                 "test_only": True, "help": "测试页画该框"},
    },
    "plate_recog": {
        "detect_level": {"label": "检测精度档", "type": "select", "default": "high",
                         "options": ["high", "low"],
                         "help": "high=640（远角/小车牌更准，慢）；low=320（快）"},
        "mode": {"label": "识别方式", "type": "select", "default": "crop",
                 "options": ["crop", "direct"],
                 "help": "crop=消费上游目标裁剪的车图再识别（远角小牌）；direct=整帧直接识别"},
        "margin": {"label": "车牌出图外扩比例", "type": "float", "default": 0.1,
                   "help": "从原图裁原分辨率车牌小图时往外多扩多少"},
        "dedup_iou": {"label": "车牌去重 IoU", "type": "float", "default": 0.6,
                      "help": "两张车牌重叠超过它算同一张，只留一张"},
        "track_dedup": {"label": "按车辆跟踪去重", "type": "bool", "default": True,
                        "help": "同一辆车（同一 track）只识别一次，避免逐帧重复识别"},
        "track_cooldown": {"label": "同车重试间隔（秒）", "type": "float", "default": 5.0,
                           "help": "同一 track 多少秒后才允许再次识别；0=只识别一次"},
        # ---- 测试页脚手架（test_only，不落库、不进链路）----
        "image": {"label": "测试图片", "type": "file", "src": "img", "test_only": True,
                  "help": "含车辆/车牌的图片"},
        "yolo_model": {"label": "测试用 YOLO 权重", "type": "file", "src": "model",
                       "test_only": True, "help": "测试页用 YOLO 造车辆框；不选默认 yolov8n.pt"},
        "yolo_conf": {"label": "测试 YOLO conf", "type": "float", "default": 0.25,
                      "test_only": True, "help": "测试页车辆检测置信度"},
        "yolo_iou": {"label": "测试 YOLO iou", "type": "float", "default": 0.6,
                     "test_only": True, "help": "测试页车框合并阈值"},
        "yolo_imgsz": {"label": "测试 YOLO imgsz", "type": "int", "default": 1280,
                       "test_only": True, "help": "测试页推理分辨率"},
        "yolo_max_det": {"label": "测试 YOLO max_det", "type": "int", "default": 300,
                         "test_only": True, "help": "测试页单帧最多车辆数"},
        "yolo_classes": {"label": "测试 YOLO 类别", "type": "str", "default": "2",
                         "test_only": True, "help": "测试页识别哪类目标（2=车）"},
    },
    "face_monitor": {
        "video": {"label": "测试视频", "type": "file", "src": "vid", "group": "运行控制",
                  "help": "要扫描的视频文件"},
        "db_path": {"label": "底库文件（.npz）", "type": "str", "default": "out/face_db.npz",
                    "group": "运行控制", "help": "拿视频里的人脸去比对的底库"},
        "device": {"label": "设备", "type": "select", "default": "auto", "group": "运行控制",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "max_frames": {"label": "限帧数（0=全部）", "type": "int", "default": 60,
                       "group": "运行控制", "help": "测试视频为 4K 长视频，建议先限帧"},
        "save_dir": {"label": "命中帧保存目录", "type": "str", "group": "运行控制",
                     "default": "tests/data/out/faces/monitor",
                     "help": "命中（在底库中找到的人脸）的帧图存到这里"},
        "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                       "group": "帧管理", "help": "每隔几帧检测一次"},
        "queue_threshold": {"label": "积压阈值", "type": "int", "default": 32,
                            "group": "帧管理", "help": "积压超过它放宽抽帧"},
        "backpressure_multiplier": {"label": "放宽倍数", "type": "int", "default": 2,
                                    "group": "帧管理", "help": "放宽时抽帧间隔乘以几倍"},
    },
    "frame_yolo": {
        # 运行控制
        "video": {"label": "测试视频", "type": "file", "src": "vid", "group": "运行控制",
                  "help": "要扫描的视频文件"},
        "max_frames": {"label": "限帧数（0=全部）", "type": "int", "default": 300,
                       "group": "运行控制", "help": "长视频建议先限帧；0=读到结尾"},
        "device": {"label": "推理设备（空=自动）", "type": "str", "default": "",
                   "group": "运行控制", "help": "如 cpu 或 0"},
        "track": {"label": "启用跟踪（BOTSORT）", "type": "bool", "default": True,
                  "group": "运行控制",
                  "help": "开=track_id 跨帧稳定，可评估追踪能力；关=每帧独立检测"},
        # 帧管理 FrameManager
        "sampling": {"label": "取图节奏", "type": "select", "default": "analysis",
                     "group": "帧管理 FrameManager",
                     "options": ["analysis", "wall_clock", "frame_count"],
                     "help": "analysis=随抽帧节奏; wall_clock=按墙钟秒; frame_count=按帧数"},
        "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                       "group": "帧管理 FrameManager",
                       "help": "每 N 帧取一张（analysis 模式用）"},
        "interval_sec": {"label": "墙钟间隔（秒）", "type": "float", "default": 3.0,
                         "group": "帧管理 FrameManager",
                         "help": "每隔多少秒取一张（wall_clock 模式用）"},
        "interval_frames": {"label": "帧数间隔（帧）", "type": "int", "default": 30,
                            "group": "帧管理 FrameManager",
                            "help": "每隔多少帧取一张（frame_count 模式用）"},
        # YOLO 识别 YOLODetector
        "model_path": {"label": "权重模型", "type": "file", "src": "model",
                       "group": "YOLO 识别 YOLODetector",
                       "help": "检测权重；不选则用默认 yolov8n.pt"},
        "conf": {"label": "检出阈值 conf", "type": "float", "default": 0.35,
                 "group": "YOLO 识别 YOLODetector",
                 "help": "置信度低于它的框会被丢弃；调高更严格"},
        "iou": {"label": "NMS 阈值 iou", "type": "float", "default": 0.7,
                "group": "YOLO 识别 YOLODetector",
                "help": "重叠超过它的重复框会合并成一个"},
        "imgsz": {"label": "推理分辨率 imgsz", "type": "int", "default": 640,
                  "group": "YOLO 识别 YOLODetector",
                  "help": "送进模型的分辨率；越大越慢，小目标更清楚"},
        "max_det": {"label": "单帧最多目标 max_det", "type": "int", "default": 300,
                    "group": "YOLO 识别 YOLODetector",
                    "help": "一帧最多保留多少个目标"},
        "classes": {"label": "类别白名单（逗号分隔，空=不限）", "type": "str", "default": "0",
                    "group": "YOLO 识别 YOLODetector",
                    "help": "COCO: 0=人, 2=车"},
    },
}

MODULES = {
    "frame_manager": {
        "name": "帧管理 FrameManager", "group": "管道四模块",
        "desc": "复用 pipe.composer.FrameManager：三种取图节奏 + 积压/耗时背压（原 frame_scheduler 并入）。",
        "params": PARAMS["frame_manager"], "handler": h_frame_manager},
    "yolo": {
        "name": "小模型识别 YOLODetector", "group": "管道四模块",
        "desc": "复用 pipe.composer.YOLODetector：**纯推理**（有效性过滤归「检测筛选」阶段，节流/报送编排在 Pipeline）。",
        "params": PARAMS["yolo"], "handler": h_yolo},
    "det_filter": {
        "name": "检测筛选 DetFilter", "group": "管道四模块",
        "desc": "复用 pipe.det_filter.DetFilter：有效性筛选（逐条条件 DSL：范围+与/或/非+派生尺寸量"
                "+ 聚合 count(字段)/count(子条件)/count()；+ 成组数量判定）。"
                "链路里插在 YOLO 之后；有效性过滤的唯一来源（空条件+空上下限=不筛选）。",
        "params": PARAMS["det_filter"], "handler": h_det_filter},
    "seq_filter": {
        "name": "时序判定 SeqFilter", "group": "管道四模块",
        "desc": "复用 pipe.seq_filter.SeqFilter：时序门控（整类节流 throttle / track 集合变化 change）"
                "+ 窗口/单条规则确认与事件标注。链路里插在检测筛选之后；"
                "承载原 Pipeline 的 check_interval/track_change_only 语义。",
        "params": PARAMS["seq_filter"], "handler": h_seq_filter},
    "target_crop": {
        "name": "目标裁剪 TargetCrop", "group": "管道四模块",
        "desc": "按上游检测框裁子图（输出分辨率可配，默认原分辨率）；"
                "供车牌识别/人脸检测/VLM 消费。不含检测器。",
        "params": PARAMS["target_crop"], "handler": h_target_crop},
    "annotate": {
        "name": "标注 Annotator", "group": "管道四模块",
        "desc": "在帧副本上画上游检测框（供 VLM 看“原图+画框”）；纯绘制，无模型。",
        "params": PARAMS["annotate"], "handler": h_annotate},
    "vlm": {
        "name": "大模型研判 VLMAnalyzer", "group": "管道四模块",
        "desc": "复用 pipe.composer.VLMAnalyzer：素材包准备，可接真实 VLM。",
        "params": PARAMS["vlm"], "handler": h_vlm},
    "face_detect": {
        "name": "人脸检测 FaceDetector", "group": "人脸识别链路",
        "desc": "复用 face_detect.face_detector：SCRFD 检测 + 5 点对齐。",
        "params": PARAMS["face_detect"], "handler": h_face_detect},
    "face_embed": {
        "name": "特征提取 FaceEmbedder", "group": "人脸识别链路",
        "desc": "复用 face_embed.face_embedder：对齐裁剪 → 512 维向量。",
        "params": PARAMS["face_embed"], "handler": h_face_embed},
    "face_store": {
        "name": "向量底库 FaceStore", "group": "人脸识别链路",
        "desc": "复用 face_embed.face_store：注册 / 检索 / 自检 / 持久化。",
        "params": PARAMS["face_store"], "handler": h_face_store},
    "plate_recog": {
        "name": "车牌识别 PlateRecognizer", "group": "车牌识别链路",
        "desc": "复用 plate_recog.plate_recognizer：HyperLPR3 在车辆子图里裁车牌 + 识别；"
                "车辆子图由上游「目标裁剪」提供，或整帧直读。",
        "params": PARAMS["plate_recog"], "handler": h_plate_recog},
    "face_monitor": {
        "name": "视频人脸检索 FaceMonitor", "group": "已完成链路", "hidden": True,
        "desc": "视频/摄像头人脸监控链路：抽帧 + 检测 + 嵌入 + 检索（在「已完成链路」中选择并运行）。",
        "params": PARAMS["face_monitor"], "handler": h_face_monitor},
    "frame_yolo": {
        "name": "帧管理-YOLO识别链路", "group": "已完成链路", "hidden": True,
        "desc": "视频帧管理 + YOLO 识别链路：抽帧决策 → YOLO 检测/跟踪 → 过滤统计，"
               "输出逐帧时间线与追踪能力评估指标（在「已完成链路」中选择并运行）。",
        "params": PARAMS["frame_yolo"], "handler": h_frame_yolo},
    "chain": {
        "name": "四模式链路 chain", "group": "已完成链路", "hidden": True,
        "desc": "四模式链路的通用运行器：按 spec.stages 组装帧管理/YOLO/VLM/告警并"
               "逐帧驱动（script 模拟 / real 真实推理）。",
        "params": PARAMS["chain"], "handler": h_chain},
    "pipeline_run": {
        "name": "链路测试台", "group": "已完成链路", "hidden": True,
        "desc": "专用测试台：支持视频片段截取、图片按帧保存、实时结果展示等功能。",
        "params": {}, "handler": h_pipeline_run},
}


def _file_index() -> dict:
    return {
        "img": sorted({p.name for p in PIC_DIR.glob("*") if p.suffix.lower() in _IMG_SUFFIX}
                      | {p.name for p in TEST_IMGS.glob("*") if p.suffix.lower() in _IMG_SUFFIX}),
        "vid": sorted({p.name for p in VID_DIR.glob("*")
                       if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")}),
        "model": sorted({p.name for p in MODELS_DIR.glob("*.pt")}),
    }


def _video_info(name: str) -> dict:
    """读视频元信息（分辨率/帧率/总帧数/时长），供测试台选视频后预估处理量。"""
    if not name:
        return {"ok": False, "error": "缺少视频名"}
    path = _find_file(name, [VID_DIR])
    if not path:
        return {"ok": False, "error": f"找不到视频: {name}"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": f"无法打开视频: {name}"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"ok": True, "name": name, "width": w, "height": h,
            "fps": round(fps, 2), "total_frames": total,
            "duration_sec": round(total / fps, 1) if fps > 0 else 0.0}


# ---------------- 视频解码适配：非浏览器友好编码按需转码为 H.264 ----------------
# 浏览器 <video> 只解 H.264/VP8/VP9/AV1；项目素材多为 H.265(HEVC)/4K，直接播是黑屏。
# 策略：ffprobe 探测编码 → 浏览器友好则直接回原文件；否则后台 ffmpeg 转 H.264/AAC，
# 缓存到 out/transcoded/（源文件不变），前端轮询进度、转完再播。分析仍用原视频，
# 转码只影响预览：预览宽 >1920 时等比缩到 1920，框坐标在前端按原分辨率还原。
TRANS_DIR = ROOT / "out/transcoded"
TRANS_DIR.mkdir(parents=True, exist_ok=True)
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_BROWSER_VCODECS = {"h264", "vp8", "vp9", "av1"}
_BROWSER_ACODECS = {None, "", "aac", "mp3", "opus", "vorbis"}
_TRANS_LOCK = threading.Lock()
_TRANS: dict = {}  # cache_path(str) -> {status, progress, error, src, proc}


def _probe_video(path: Path) -> dict:
    """ffprobe 读视频/音频编码、像素格式、时长（缺 ffprobe 返回空）。"""
    if not _FFPROBE:
        return {}
    cmd = [_FFPROBE, "-v", "error", "-show_streams", "-show_format",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    dur = 0.0
    with contextlib.suppress(Exception):
        dur = float(data.get("format", {}).get("duration") or 0.0)
    return {"vcodec": v.get("codec_name"), "acodec": a.get("codec_name"),
            "pix_fmt": v.get("pix_fmt"), "width": v.get("width"),
            "height": v.get("height"), "duration": dur}


def _is_browser_playable(info: dict) -> bool:
    """H.264(yuv420p)+常见音轨可直接播；HEVC/非 420 像素格式/未知音轨需转码。"""
    if not info or info.get("vcodec") not in _BROWSER_VCODECS:
        return False
    if info.get("vcodec") == "h264" and info.get("pix_fmt") not in ("yuv420p", "yuvj420p"):
        return False
    return info.get("acodec") in _BROWSER_ACODECS


def _trans_cache_path(src: Path) -> Path:
    """转码缓存路径：源名+大小+mtime+参数版本 哈希，源变则重转。"""
    st = src.stat()
    key = f"{src.name}|{st.st_size}|{st.st_mtime_ns}|v1|max1920"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return TRANS_DIR / f"{src.stem}.{h}.mp4"


def _trans_start(src: Path, out: Path, duration: float) -> None:
    """启动后台转码线程（同 key 已在跑则跳过）。"""
    key = str(out)
    with _TRANS_LOCK:
        st = _TRANS.get(key)
        if st and st.get("status") == "transcoding":
            return
        _TRANS[key] = {"status": "transcoding", "progress": 0.0,
                       "error": "", "src": str(src), "proc": None}

    def _worker():
        tmp = out.with_suffix(".part.mp4")
        vf = "scale='min(1920,iw)':-2"  # 宽 >1920 等比缩到 1920（预览够用，体积/解码成本大降）
        cmd = [_FFMPEG, "-y", "-v", "error", "-i", str(src), "-vf", vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(tmp)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            with _TRANS_LOCK:
                _TRANS[key]["proc"] = proc
            for line in proc.stdout:
                line = line.strip()
                if line.startswith(("out_time_us=", "out_time_ms=")):
                    val = line.split("=", 1)[1]
                    if val.isdigit() and duration > 0:
                        pct = min(99.0, int(val) / 1e6 / duration * 100)
                        with _TRANS_LOCK:
                            _TRANS[key]["progress"] = round(pct, 1)
            proc.wait()
            if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                tmp.replace(out)
                with _TRANS_LOCK:
                    _TRANS[key].update(status="ready", progress=100.0)
            else:
                err = (proc.stderr.read() or "").strip()[-300:] if proc.stderr else ""
                with _TRANS_LOCK:
                    _TRANS[key].update(status="error", error=err or "转码失败")
        except Exception as e:
            with _TRANS_LOCK:
                _TRANS[key].update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            with contextlib.suppress(Exception):
                if tmp.is_file() and not out.is_file():
                    tmp.unlink()

    threading.Thread(target=_worker, daemon=True).start()


def _video_prepare(name: str) -> dict:
    """确保视频可在浏览器播放：友好则回原文件，否则触发/查询 H.264 转码进度。"""
    path = _find_file(name, [VID_DIR])
    if not path:
        return {"ok": False, "error": f"找不到视频: {name}"}
    info = _probe_video(path)
    if _is_browser_playable(info):
        return {"ok": True, "status": "ready", "transcoded": False,
                "url": f"/files/videos/{quote(name)}"}
    if not _FFMPEG:
        return {"ok": False, "error": "该视频编码浏览器不支持，且未安装 ffmpeg，无法转码"}
    out = _trans_cache_path(path)
    if out.is_file() and out.stat().st_size > 0:
        return {"ok": True, "status": "ready", "transcoded": True,
                "url": f"/media/videos/{quote(name)}"}
    with _TRANS_LOCK:
        st = dict(_TRANS.get(str(out), {}))
    if st.get("status") == "error":  # 已失败：不自动重试，回错误让前端提示
        return {"ok": True, "status": "error", "error": st.get("error", "转码失败"),
                "transcoded": True, "url": f"/media/videos/{quote(name)}"}
    _trans_start(path, out, float(info.get("duration") or 0.0))
    with _TRANS_LOCK:
        st = dict(_TRANS.get(str(out), {}))
    return {"ok": True, "status": st.get("status", "transcoding"),
            "progress": st.get("progress", 0.0), "transcoded": True,
            "error": st.get("error", ""), "url": f"/media/videos/{quote(name)}"}


def _media_path(name: str) -> Path | None:
    """已转码缓存文件路径；不存在或空文件返回 None。"""
    src = _find_file(name, [VID_DIR])
    if not src:
        return None
    out = _trans_cache_path(src)
    return out if out.is_file() and out.stat().st_size > 0 else None


def _get_estimator() -> ResourceEstimator:
    """惰性创建全局估算器（首次调用时读 out/fingerprints.json）。"""
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = ResourceEstimator(project_root=ROOT)
    return _ESTIMATOR


def _module_defaults(mid: str) -> dict:
    """从 PARAMS 提取某模块参数的默认值（用于静态估算与前端展示）。"""
    return {k: spec.get("default") for k, spec in MODULES[mid]["params"].items()}


def _module_estimate(mid: str) -> dict:
    """某模块在默认参数下的资源估算（静态法为主，实测指纹命中则优先）。"""
    try:
        fp = _get_estimator().estimate_module(mid, _module_defaults(mid))
        return fp.to_dict()
    except Exception as e:  # 估算失败不阻断 schema
        return {"error": f"{type(e).__name__}: {e}"}


def _schema() -> dict:
    ports = module_port_decls()
    return {"modules": {mid: {"name": m["name"], "group": m["group"],
                              "desc": m["desc"], "params": m["params"],
                              "hidden": m.get("hidden", False),
                              # 端口协议（P5 第 2 步）：自描述输入/输出 + 本期可连线端口，
                              # 供设计器「输入连线」区渲染（只读附加字段，兼容旧前端）。
                              "inputs": ports.get(mid, {}).get("inputs", {}),
                              "outputs": ports.get(mid, {}).get("outputs", {}),
                              "resolved_ports": list(CHAIN_RESOLVED_PORTS.get(mid, ())),
                              "estimate": _module_estimate(mid)}
                        for mid, m in MODULES.items()},
            # 可用产物清单（产物名 → 类型）：连线下拉选项的唯一来源
            "artifacts": artifact_types(),
            "files": _file_index(),
            "estimator": {"fingerprint_count": len(_get_estimator().fingerprints)}}


# ---------------- 命名配置暂存（MySQL，web_lab/db.py） ----------------

def _read_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(length) or b"{}")


def _preset_list(kind: str) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        return {"ok": True, "presets": db.list_presets(kind)}
    except Exception as e:  # DB 不可用不影响服务器运行
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _preset_save(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        kind = body.get("kind", "module")
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体必须是合法 JSON 对象"}
    name = str(body.get("name", "")).strip()
    module_id = str(body.get("module_id", ""))
    params = body.get("params", {})
    preset_id = body.get("preset_id")
    preset_id = str(preset_id).strip() if preset_id not in (None, "") else None
    if kind not in ("module", "pipeline"):
        return {"ok": False, "error": f"未知配置类型: {kind}"}
    if kind == "module" and module_id not in MODULES:
        return {"ok": False, "error": f"未知模块: {module_id}"}
    if kind == "pipeline":
        module_id = ""  # 链路配置整体存于 params（完整 pipeline spec）
    if not isinstance(params, dict):
        return {"ok": False, "error": "params 必须是对象"}
    try:
        rid, created = db.save_preset(kind, name, module_id, params, preset_id=preset_id)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "id": rid, "name": name, "created": created,
            "note": "已保存" if created else "同名配置已覆盖"}


def _preset_load(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        rid = str(body.get("id", "")).strip()
        if not rid:
            raise ValueError("id 为空")
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体缺少合法 id"}
    try:
        return {"ok": True, "preset": db.load_preset(rid)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _preset_delete(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        rid = str(body.get("id", "")).strip()
        if not rid:
            raise ValueError("id 为空")
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体缺少合法 id"}
    try:
        db.delete_preset(rid)
        return {"ok": True, "deleted": rid}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------- 链路拼装暂存（pipelines + pipeline_steps，见 db.py） ----------------

def _pipelines_list() -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        return {"ok": True, "pipelines": db.list_pipelines()}
    except Exception as e:  # DB 不可用不影响服务器运行
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------- 数据库数据浏览（只读快照，见 doc/design/storage_design.md 第一部分） ----------------

def _db_tables() -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        return {"ok": True, **db.table_snapshots()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _pipeline_save(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        name = str(body.get("name", "")).strip()
        steps = body.get("steps")
        streams = int(body.get("streams") or 1)
        budget = body.get("budget") or {}
        meta = body.get("meta") or {}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体必须是合法 JSON 对象"}
    if not name:
        return {"ok": False, "error": "链路名不能为空"}
    if not isinstance(steps, list) or not all(
            isinstance(x, str) and x.strip() for x in steps):
        return {"ok": False, "error": "steps 必须是 preset id（如 YOLO0001）数组"}
    if not isinstance(budget, dict) or not isinstance(meta, dict):
        return {"ok": False, "error": "budget/meta 必须是对象"}
    try:
        pid, created = db.save_pipeline(name, steps, streams=streams,
                                        budget=budget, meta=meta)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "id": pid, "name": name, "created": created,
            "note": "链路已保存" if created else "同名链路已覆盖"}


def _pipeline_load(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        pid = str(body.get("id", "")).strip()
        if not pid:
            raise ValueError("id 为空")
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体缺少合法 id"}
    try:
        return {"ok": True, "pipeline": db.load_pipeline(pid)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _pipeline_delete(handler) -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        pid = str(body.get("id", "")).strip()
        if not pid:
            raise ValueError("id 为空")
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体缺少合法 id"}
    try:
        db.delete_pipeline(pid)
        return {"ok": True, "deleted": pid}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------- 挂载 agent（智能助手：答疑 / 组装 / 修改链路） ----------------

def _agent_chat(handler) -> dict:
    """POST /api/pipeline/agent —— 挂载 agent 对话。

    请求体：{"role":"architect|assembler", "message": "...", "history": [...]}
    返回：  {ok, role, text, tool_calls:[{name,arguments}], steps, error?}

    LLM 端点/密钥经环境变量提供（LLM_BASE_URL / LLM_API_KEY 或 SILICONFLOW_API_KEY /
    LLM_MODEL）；未配置时返回 configured=False 提示。模型经 function calling 调用平台工具
    （读知识库 / 列模块 / 校验 spec / 资源估算 / 读已存链路）。
    """
    try:
        body = _read_json_body(handler)
        if not isinstance(body, dict):
            raise TypeError("请求体必须是对象")
        role = str(body.get("role") or "architect").strip() or "architect"
        message = str(body.get("message") or "").strip()
        history = body.get("history") or []
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体必须是合法 JSON 对象"}
    if role not in ("architect", "assembler"):
        return {"ok": False, "error": f"未知角色: {role}（可用 architect / assembler）"}
    if not message:
        return {"ok": False, "error": "message 不能为空"}
    if not isinstance(history, list):
        return {"ok": False, "error": "history 必须是消息数组"}

    try:
        import agent_llm  # 惰性：运行时依赖仅在调用时引入
    except Exception as e:
        return {"ok": False, "error": f"agent 运行时不可用: {type(e).__name__}: {e}"}
    if not agent_llm._api_key():
        return {"ok": False, "configured": False,
                "error": "未配置 LLM：请设置环境变量 LLM_API_KEY（或 SILICONFLOW_API_KEY）后重启服务"}
    try:
        res = agent_llm.run(role, message, history=history)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": res.get("ok", False), "role": res.get("role"),
            "text": res.get("text", ""), "tool_calls": res.get("tool_calls", []),
            "steps": res.get("steps", 0), "error": res.get("error", "")}


# ---------------- HTTP 服务 ----------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _stream(self, params: dict) -> None:
        """SSE 流式运行（实时播放叠加）：抢 _TEST_LOCK（忙则回 error 事件）后逐事件写出。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def write(event: str, payload: dict) -> None:
            body = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(body.encode("utf-8"))
            self.wfile.flush()

        if not _TEST_LOCK.acquire(blocking=False):
            with contextlib.suppress(Exception):
                write("error", {"message": "另一测试正在运行，请稍后再试"})
            return
        try:
            h_pipeline_stream(write, params)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开：循环随写失败结束
        except Exception as e:
            print(traceback.format_exc())
            with contextlib.suppress(Exception):
                write("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            self.close_connection = True  # 流结束即断开（HTTP/1.1 keep-alive 下否则客户端会一直等）
            _TEST_LOCK.release()

    def _send_file(self, path: Path, ctype: str) -> None:
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if rng:  # 支持 HTTP Range（206）：视频拖动进度条必需
            m = re.match(r"bytes=(\d*)-(\d*)", str(rng).strip())
            if m and (m.group(1) or m.group(2)):
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start <= end and start < size:
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.end_headers()
                    with open(path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(end - start + 1))
                    return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/test.html":
            html = (STATIC / "test.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path.startswith("/files/videos/"):
            video_name = unquote(path[len("/files/videos/"):])  # 浏览器对中文名会百分号编码
            video_path = VID_DIR / video_name
            if video_path.is_file():
                self._send_file(video_path, "video/mp4")
            else:
                self._send(404, b"video not found", "text/plain")
        elif path.startswith("/files/images/"):
            image_name = unquote(path[len("/files/images/"):])
            image_path = PIC_DIR / image_name
            if not image_path.is_file():
                image_path = TEST_IMGS / image_name
            if image_path.is_file():
                self._send_file(image_path, "image/jpeg")
            else:
                self._send(404, b"image not found", "text/plain")
        elif path == "/api/schema":
            self._json(_schema())
        elif path == "/api/video/info":
            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            self._json(_video_info((qs.get("name") or [""])[0]))
        elif path == "/api/video/prepare":  # 解码适配：确保浏览器可播（必要时转码）
            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            self._json(_video_prepare((qs.get("name") or [""])[0]))
        elif path.startswith("/media/videos/"):  # 转码后的 H.264 预览文件
            media_name = unquote(path[len("/media/videos/"):])
            media_path = _media_path(media_name)
            if media_path:
                self._send_file(media_path, "video/mp4")
            else:
                self._send(404, b"media not ready", "text/plain")
        elif path == "/api/test/pipeline_run/stream":  # 实时播放叠加（SSE）
            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            self._stream({k: v[0] for k, v in qs.items() if v})
        elif path == "/api/presets":
            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            kind = (qs.get("kind") or ["module"])[0]
            self._json(_preset_list(kind))
        elif path == "/api/pipelines":
            self._json(_pipelines_list())
        elif path == "/api/db/tables":
            self._json(_db_tables())
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        # ---- 链路资源估算（P1）----
        if self.path == "/api/pipeline/estimate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                spec = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"})
                return
            try:
                streams = int(spec.get("streams") or 1)
                budget = spec.get("budget") or {}
                vram_budget = budget.get("vram_mb") if isinstance(budget, dict) else None
                pe = _get_estimator().estimate_pipeline(
                    spec, streams=streams, vram_budget_mb=vram_budget)
                reco = _get_estimator().back_calculate(
                    spec, streams=streams, vram_budget_mb=vram_budget)
                self._json({"ok": True, "estimate": pe.to_dict(),
                            "back_calc": reco})
            except Exception as e:  # spec 不合法时给出明确报错
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})
            return

        # ---- 实时播放叠加：停止流式运行 ----
        if self.path == "/api/test/pipeline_run/stop":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"})
                return
            rid = str(body.get("run_id") or "").strip()
            evt = _STREAMS.get(rid)
            if evt is None:
                self._json({"ok": False, "error": f"无此运行: {rid}"})
                return
            evt.set()
            self._json({"ok": True, "stopped": rid})
            return

        # ---- 命名配置暂存（MySQL）----
        if self.path == "/api/presets/save":
            self._json(_preset_save(self))
            return
        if self.path == "/api/presets/load":
            self._json(_preset_load(self))
            return
        if self.path == "/api/presets/delete":
            self._json(_preset_delete(self))
            return

        # ---- 链路拼装暂存（pipelines + pipeline_steps）----
        if self.path == "/api/pipelines/save":
            self._json(_pipeline_save(self))
            return
        if self.path == "/api/pipelines/load":
            self._json(_pipeline_load(self))
            return
        if self.path == "/api/pipelines/delete":
            self._json(_pipeline_delete(self))
            return

        # ---- 链路生成 / 智能助手 ----
        if self.path == "/api/pipeline/build":
            self._json({"ok": True, "stub": True,
                        "note": "链路生成（P2 占位）尚未实现"})
            return
        if self.path == "/api/pipeline/agent":
            t0 = time.time()
            result = _agent_chat(self)
            result.setdefault("elapsed_sec", round(time.time() - t0, 2))
            print(f"[agent] ok={result.get('ok')} {time.time() - t0:.1f}s")
            self._json(result)
            return

        if not self.path.startswith("/api/test/"):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        
        mod = self.path.rsplit("/", 1)[-1]
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            params = body.get("params", {})
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "请求体不是合法 JSON"})
            return
        
        # 特殊处理 pipeline_run（不用 MODULES 里的 schema）
        if mod == "pipeline_run":
            t0 = time.time()
            with _TEST_LOCK:
                try:
                    result = h_pipeline_run(params)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                              "traceback": tb[-1800:]}
            result.setdefault("elapsed_sec", round(time.time() - t0, 2))
            print(f"[test] {mod} {time.time() - t0:.1f}s ok={result.get('ok')}")
            self._json(result)
            return
        
        if mod not in MODULES:
            self._json({"ok": False, "error": f"未知模块: {mod}"})
            return
        t0 = time.time()
        with _TEST_LOCK:
            try:
                result = MODULES[mod]["handler"](params)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "traceback": tb[-1800:]}
        result.setdefault("elapsed_sec", round(time.time() - t0, 2))
        print(f"[test] {mod} {time.time() - t0:.1f}s ok={result.get('ok')}")
        self._json(result)


# ---------------- 唯一保留链路播种（警用无人机链路，模块自定义拼接） ----------------
# 项目仅保留一条链路：帧管理→YOLO识人车→VLM研判警情→告警策略。播种收敛到此处
# （原 tests/police_uav_config.py / face_dedup_config.py / _SEED_PIPES 均已移除）：
# 4 条模块 preset（kind='module'，拆列存入各模块参数表）+ 1 条链路行
# （pipelines + pipeline_steps 引用上述 preset）。幂等：同名已存在则跳过
# （尊重用户在 web 端的编辑），每模块恰一份、链路恰一条。

# 警情类型提示词（VLM 研判阶段用；返回 JSON：alert_type + description + confidence）
ALARM_PROMPT = """你是一个警情分析专家。分析画面中的情况，只输出一个 JSON（不要多余文字或代码块）：
{
  "alert_type": "治安类/交警类/群体性事件类/救援救助类/无异常",
  "description": "简要描述观察到的异常情况（无异常时描述正常场景）",
  "confidence": 0.0
}

警情类型说明：
- 治安类：打架斗殴等治安事件
- 交警类：车辆明显形变、车尾有三角警示牌等车辆抛锚情况
- 群体性事件类：多人聚集进行不正常活动（持械或有条幅拦路）
- 救援救助类：有人溺水、倒地并挥手呼救等情况
- 无异常：画面中未发现上述异常情况

请根据实际情况准确判断，严格输出上述字段的 JSON（confidence 为 0~1 的置信度）。"""

_SEED_PIPELINE_NAME = "警用无人机链路"
_SEED_STAGES = [
    # 帧管理：远角无人机视频按 15 帧抽 1（2026-08-24 实调）
    ("frame_manager", "警用无人机·帧管理", {
        "sampling": "analysis", "frame_skip": 15, "interval_sec": 3.0,
        "interval_frames": 30, "backpressure": "standard"}),
    # YOLO：imgsz=1280 为远角小目标决定性参数（2026-08-24 实调，见知识库决策表）
    ("yolo", "警用无人机·YOLO识别", {
        "model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.6,
        "imgsz": 1280, "max_det": 300, "classes": "0,2", "device": ""}),
    # 检测筛选：人≥2 才报、车≥1 才报（有效性对全下游生效）
    ("det_filter", "警用无人机·检测筛选", {
        "filter": "", "group_limits": "0:2,300;2:1,300"}),
    # 标注：把检出框画在原帧上，供 VLM 看"原图 + 画框"
    ("annotate", "警用无人机·标注", {
        "thickness": 2, "show_label": True, "classes": "", "filter": ""}),
    # VLM：接真实端点（use_real_vlm=true；不可达时测试台记日志继续，不中断）
    ("vlm", "警用无人机·VLM研判", {
        "max_resolution": "1280,720", "prompt": ALARM_PROMPT,
        "use_real_vlm": True,
        "vlm_endpoint": "http://117.42.21.253:8000/v1/chat/completions",
        "vlm_model": "qwen3-vl-32b", "vlm_key": ""}),
    # 时序判定（链路尾，一个阶段两种用途）——
    #   throttle=整类节流 3 秒/类（原 check_interval）→ 检测流门控；
    #   rules=事件尾（原「告警策略」window 等价）：VLM 结果经 publish 走此规则，
    #   词表命中 + 按警情类型独立冷却 30s，事件名取 VLM 的 action（{label}）。
    ("seq_filter", "警用无人机·时序判定", {
        "throttle": "cls_id:3.0", "change": "",
        "rules": json.dumps([{
            "match": "label in 治安类|交警类|群体性事件类|救援救助类",
            "event": "{label}", "key": "label", "cooldown": 30.0,
        }], ensure_ascii=False)}),
]

# 第二条链路：车牌识别（帧管理 → YOLO 识车 → 车牌识别(裁车牌+识别) → 告警）
_SEED_PLATE_NAME = "车牌识别链路"
_SEED_PLATE_STAGES = [
    # 帧管理：车牌视频按 5 帧抽 1（比警用链路密，兼顾远角小牌）
    ("frame_manager", "车牌识别·帧管理", {
        "sampling": "analysis", "frame_skip": 5, "interval_sec": 3.0,
        "interval_frames": 30, "backpressure": "standard"}),
    # YOLO：只识车（classes=2），为车牌识别提供车辆框（复用原有 yolo 模块）
    ("yolo", "车牌识别·YOLO识车", {
        "model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.6,
        "imgsz": 1280, "max_det": 300, "classes": "2", "device": ""}),
    # 检测筛选：车长边≥40 才有效（小目标视为未识别）
    ("det_filter", "车牌识别·检测筛选", {
        "filter": "long >= 40", "group_limits": "2:1,300"}),
    # 目标裁剪：按 YOLO 车辆框裁车放大 2×（供车牌识别在车图里裁车牌）
    ("target_crop", "车牌识别·目标裁剪", {
        "out_size": "×2", "margin": 0.1,
        "classes": "2", "filter": ""}),
    # 车牌识别：从车图裁车牌 + 识别 + 坐标还原 + 原图出原分辨率车牌小图
    # （track_dedup：同一辆车只识别一次，5 秒后可重试）
    ("plate_recog", "车牌识别·车牌识别", {
        "detect_level": "high", "mode": "crop", "margin": 0.1,
        "dedup_iou": 0.6, "track_dedup": True, "track_cooldown": 5.0}),
    # 事件尾（原「告警策略」inspection 的等价）：车牌记录经 publish 出事件，同车牌号只一次
    ("seq_filter", "车牌识别·时序判定", {
        "throttle": "", "change": "",
        "rules": json.dumps([{
            "match": "kind == plate", "event": "{label}", "key": "label", "cooldown": -1.0,
        }], ensure_ascii=False)}),
]


# 第三条链路：人脸截取（帧管理 → YOLO 识人 → 原分辨率人脸提取；无嵌入/检索/告警）
_SEED_FACE_NAME = "人脸截取链路"
_SEED_FACE_STAGES = [
    # 帧管理：按 5 帧抽 1（近景人脸视频节奏；远角可调大 frame_skip）
    ("frame_manager", "人脸截取·帧管理", {
        "sampling": "analysis", "frame_skip": 5, "interval_sec": 3.0,
        "interval_frames": 30, "backpressure": "standard"}),
    # YOLO：只识人（classes=0），imgsz=1280 兼容远角小目标（分辨率近乎免费的召回）
    ("yolo", "人脸截取·YOLO识人", {
        "model_path": "models/yolov8n.pt", "conf": 0.3, "iou": 0.6,
        "imgsz": 1280, "max_det": 300, "classes": "0", "device": ""}),
    # 检测筛选：人长边≥30 才有效（小目标视为未识别）
    ("det_filter", "人脸截取·检测筛选", {
        "filter": "long >= 30", "group_limits": "0:1,300"}),
    # 目标裁剪：按 YOLO 人框裁人（供人脸检测在子图里检脸）
    ("target_crop", "人脸截取·目标裁剪", {
        "out_size": "", "margin": 0.0,
        "classes": "0", "filter": ""}),
    # 人脸检测：在目标子图里检脸 + 从原图出原分辨率人脸
    ("face_detect", "人脸截取·人脸检测", {
        "device": "auto", "det_thresh": 0.4, "det_size": "640,640",
        "max_num": 0, "use_crop": False, "margin": 0.15, "dedup_iou": 0.6}),
]

# 链路未使用、但产品保留的模块（人脸识别链）：每模块同样恰播种一份参数（不入链路步骤）
_SEED_STANDALONE = [
    ("face_detect", "人脸检测·默认", {
        "device": "auto", "det_thresh": 0.5, "det_size": "640,640",
        "max_num": 0, "use_crop": False, "margin": 0.15, "dedup_iou": 0.6}),
    ("face_embed", "人脸嵌入·默认", {
        "device": "auto", "det_thresh": 0.5, "det_size": "640,640"}),
    ("face_store", "向量底库·默认", {
        "db_path": "out/face_db.npz", "thresh": 0.45, "topk": 5,
        "device": "auto"}),
    ("target_crop", "目标裁剪·默认", {
        "out_size": "", "margin": 0.0,
        "classes": "", "filter": ""}),
    ("annotate", "标注·默认", {
        "thickness": 2, "show_label": True, "classes": "", "filter": ""}),
    ("plate_recog", "车牌识别·默认", {
        "detect_level": "high", "mode": "crop", "margin": 0.1,
        "dedup_iou": 0.6, "track_dedup": True, "track_cooldown": 5.0}),
]


def _seed_chain(name: str, stages: list, description: str, have: dict) -> None:
    """幂等播种一条链路：先保各阶段模块 preset，再建链路引用。

    已存在同名链路时：阶段结构一致则原样保留（不动用户改动）；结构变化则按模块复用
    旧 preset（保留其参数）、补建缺失阶段、丢弃下线阶段，重建步骤（save_pipeline 同名
    覆盖）。这样既迁移结构，又不产生重复 preset。
    """
    expected = [mid for mid, _s, _p in stages]
    existing = next((p for p in db.list_pipelines() if p["name"] == name), None)
    reuse: dict = {}
    if existing is not None:
        got = [s.get("module_id") for s in (existing.get("steps") or [])]
        if got == expected:
            return
        print(f"[seed] 链路「{name}」阶段已变更，重建…")
        for s in (existing.get("steps") or []):
            reuse.setdefault(s["module_id"], s["preset_id"])  # 按模块复用旧 preset
    stage_ids = []
    for mid, sname, params in stages:
        pid = reuse.get(mid)
        if pid is None:
            key = (mid, sname)
            if key in have:
                pid = have[key]
            else:
                pid, _created = db.save_preset("module", sname, mid, params)
                print(f"[seed] 模块配置「{sname}」已播种（{pid}）")
                have[key] = pid
        stage_ids.append(pid)
    pid, _created = db.save_pipeline(name, stage_ids, streams=1, budget={},
                                     meta={"description": description})
    print(f"[seed] 链路「{name}」已播种（{pid}）")


def _seed_pipeline() -> None:
    """幂等播种链路与模块 preset（同名跳过）。失败不影响运行。"""
    if not _DB_ENABLED:
        return
    try:
        have = {(p["module_id"], p["name"]): p["id"]
                for p in db.list_presets("module")}
        for mid, name, params in _SEED_STANDALONE:
            if (mid, name) not in have:
                pid, _created = db.save_preset("module", name, mid, params)
                print(f"[seed] 模块配置「{name}」已播种（{pid}）")
                have[(mid, name)] = pid
        _seed_chain(_SEED_PIPELINE_NAME, _SEED_STAGES,
                    "警用无人机监测链路：帧管理→YOLO识人车→VLM研判警情→告警策略"
                    "（模块自定义拼接）", have)
        _seed_chain(_SEED_PLATE_NAME, _SEED_PLATE_STAGES,
                    "车牌识别链路：帧管理→YOLO识车→车牌识别(HyperLPR3)→告警"
                    "（模块自定义拼接）", have)
        _seed_chain(_SEED_FACE_NAME, _SEED_FACE_STAGES,
                    "人脸截取链路：帧管理→YOLO识人→原分辨率人脸提取"
                    "（无嵌入/检索/告警，截取落盘即目的）", have)
    except Exception as e:
        print(f"[seed] 播种链路失败（不影响运行）: {e}")


def main() -> None:
    global _DB_ENABLED
    ap = argparse.ArgumentParser(description="UMVP 可视化测试台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-db", action="store_true",
                    help="不连接 MySQL（presets 接口返回未启用）")
    args = ap.parse_args()
    if args.no_db:
        _DB_ENABLED = False
        print("[db] MySQL 已禁用（--no-db），presets 接口不可用")
    else:
        try:
            # 注册各模块参数 schema（参数名+类型 → 列式参数表），须在 init_db 前。
            # 只注册产品模块（hidden 运行器是 web 端运行入口，不存参数 preset，
            # 不建参数表——其历史空表由 init_db 的 _PRUNED_MODULES 幂等清理）；
            # test_only 测试参数（测试图片/演示回填/事件脚本等）不落库。
            db.configure_modules(
                {mid: {k: spec.get("type", "str") for k, spec in m["params"].items()
                       if not spec.get("test_only")}
                 for mid, m in MODULES.items() if not m.get("hidden")})
            db.init_db()
            _seed_pipeline()  # 幂等：播种唯一保留链路（警用无人机链路）
            print("[db] MySQL 配置暂存已就绪（presets 注册表 + 模块参数表就绪）")
        except Exception as e:
            _DB_ENABLED = False
            print(f"[db] MySQL 不可用，presets 接口将返回错误: {e}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nUMVP 可视化测试台已启动: http://{args.host}:{args.port}/")
    print(f"  模块数: {len(MODULES)} | 按 Ctrl+C 停止\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        srv.server_close()


if __name__ == "__main__":
    main()
