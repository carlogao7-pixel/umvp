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
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]          # 项目根
UMVP = ROOT / "umvp"
STATIC = Path(__file__).resolve().parent            # web_lab/
sys.path.insert(0, str(UMVP))                       # 使 pipe/face_* 可导入

MODELS_DIR = ROOT / "models"
PIC_DIR = ROOT / "tests/data/pic"
VID_DIR = ROOT / "tests/data/vid"
TEST_IMGS = UMVP / "face_detect" / "test_imgs"
OUT_DIR = ROOT / "tests/data/out"

import cv2  # noqa: E402  （cv2 起步即用；ultralytics/insightface 保持惰性）
import numpy as np  # noqa: E402

from resources import ResourceEstimator  # noqa: E402  umvp/resources.py（纯标准库，无重依赖）
import db  # noqa: E402  web_lab/db.py（MySQL 配置暂存，惰性连接，缺失不影响启动）

_ESTIMATOR = None  # 惰性创建：首次 /api/schema 或 /api/pipeline/estimate 时初始化
_DB_ENABLED = True  # 启动时 --no-db 或 MySQL 不可用则置 False，presets 接口返回"未启用"

_IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TEST_LOCK = threading.Lock()

# ---------------- 通用工具 ----------------

def _img_data_url(img, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def res(ok: bool = True, text: str = "", images: list = None,
        tables: list = None, error: str = "") -> dict:
    d = {"ok": ok, "text": text}
    if images:
        d["images"] = images
    if tables:
        d["tables"] = tables
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


def _parse_vlm_action(content: str) -> str:
    """大模型可能回纯 JSON 或 ```json 包裹，取 result 字段；失败兜底正则。"""
    txt = content.strip()
    if txt.startswith("```"):
        parts = txt.split("\n", 1)
        txt = parts[1] if len(parts) > 1 else ""
    m = re.search(r'"result"\s*:\s*"([^"]+)"', txt)
    if m:
        return m.group(1)
    return (txt or "none").strip()[:64]


def _vlm_real(p: dict, payload: dict, frame) -> str:
    """OpenAI 兼容端点调用（格式与 tests/test_3modes.py 一致）。"""
    import requests
    url = str(p.get("vlm_endpoint", "")).strip() or "http://117.42.21.253:8000/v1/chat/completions"
    model = str(p.get("vlm_model", "")).strip() or "qwen3-vl-32b"
    key = str(p.get("vlm_key", "")).strip()
    mr = tuple(payload.get("max_resolution") or (1280, 720))
    h, w = frame.shape[:2]
    scale = min(mr[0] / w, mr[1] / h)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("帧转 JPEG 失败")
    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = payload.get("prompt") or (
        "你是一个视觉分类器，分析画面后以 JSON 输出：{\"result\": \"动作名或 none\", "
        "\"description\": \"简要中文描述\"}。无异常时 result=\"none\"。"
    )
    body = {
        "model": model, "temperature": 0.1, "max_tokens": 256,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = requests.post(url, json=body, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_vlm_action(content)


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
    fm = FrameManager(sampling=sm, frame_skip=_int(p, "frame_skip", 3),
                      interval_sec=_float(p, "interval_sec", 3.0),
                      interval_frames=_int(p, "interval_frames", 30))
    n = _int(p, "n_frames", 20)
    step = _float(p, "ts_step", 1.0)
    takes = 0
    lines = []
    for i in range(n):
        ts = round(i * step, 2)
        want = fm.wants_frame(i, ts)
        takes += int(want)
        lines.append(f"帧 {i:>3}  ts={ts:>7.2f}s  →  {'● 取图' if want else '○ 跳过'}")
    text = (f"取图节奏: {sm} | 共 {n} 帧, 取图 {takes} 帧 ({takes / max(n, 1) * 100:.0f}%)\n"
            + "\n".join(lines))
    return res(text=text)


def h_frame_scheduler(p: dict) -> dict:
    from face_scan.frame_scheduler import FrameScheduler
    mode = p.get("mode", "queue")
    sch = FrameScheduler(frame_skip=_int(p, "frame_skip", 3),
                         queue_threshold=_int(p, "queue_threshold", 32),
                         backpressure_multiplier=_int(p, "backpressure_multiplier", 2),
                         max_frame_skip=_int(p, "max_frame_skip", 0) or None,
                         adaptive_relax_ratio=_float(p, "adaptive_relax_ratio", 1.2),
                         adaptive_recover_ratio=_float(p, "adaptive_recover_ratio", 0.5))
    n = _int(p, "n_frames", 30)
    fps = _float(p, "fps", 25.0)
    scen = p.get("scenario", "平稳")
    rows = []
    for i in range(n):
        if mode == "queue":
            base, wave = 5, 40
            pending = wave if (scen == "积压浪涌" and 10 <= i <= 15) else base
            decided = sch.decide(i, pending=pending)
        else:  # 自适应模式
            decided = sch.decide(i)
            cost = (0.1 + 0.5 * i / max(n - 1, 1)) if scen == "耗时渐增" else 0.08
            if decided:
                sch.note_inference(cost, sch.budget(fps))
            pending = None
        rows.append([i, "" if pending is None else pending, decided,
                     sch.current_skip, sch.relaxed])
    table = {"title": "帧决策时间线",
             "headers": ["帧", "积压深度", "分析?", "当前间隔", "放宽中"],
             "rows": [[r[0], r[1], "✓" if r[2] else "·", r[3], "✓" if r[4] else ""] for r in rows]}
    summary = (f"模式: {mode} | 场景: {scen} | 基础间隔: {sch.base_skip} | "
               f"放宽次数: {sch.relax_events} | 结束间隔: {sch.current_skip}")
    return res(text=summary, tables=[table])


def h_yolo(p: dict) -> dict:
    from pipe.composer import YOLODetector, filter_confidence, filter_min_size
    img_path, frame = _load_frame(p)
    model = _find_file(p.get("model_path", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
    filters = []
    fc = _float(p, "filter_conf", 0.0)
    if fc > 0:
        filters.append(filter_confidence(fc))
    ms, ml = _int(p, "min_short", 0), _int(p, "min_long", 0)
    if ms > 0 or ml > 0:
        filters.append(filter_min_size(ms, ml))
    yolo = YOLODetector(filters=filters,
                        class_limits=_class_limits(p.get("class_limits", "")),
                        check_interval=_float(p, "check_interval", 3.0),
                        roi=_bool(p, "roi"),
                        model_path=str(model),
                        conf=_float(p, "conf", 0.35), iou=_float(p, "iou", 0.7),
                        imgsz=_int(p, "imgsz", 640), max_det=_int(p, "max_det", 300),
                        classes=_int_list(p, "classes", None),
                        device=str(p.get("device", "")).strip() or None)
    t0 = time.time()
    dets = yolo.infer(frame)
    dt = time.time() - t0
    valid = yolo.filter_only(dets)
    submits = yolo.process(0, _float(p, "ts", 0.0), dets)

    canvas = frame.copy()
    _draw_dets(canvas, dets)
    text = [f"模型: {model.name} | 图片: {img_path.name}",
            f"推理耗时: {dt:.2f}s | 检出 {len(dets)} 个",
            f"过滤链后有效: {len(valid)} | 报送请求: {len(submits)}"]
    for s in submits:
        text.append(f"  → {s.ref} (gran={s.gran}, 该类目标数={len(s.dets)})")
    rows = [[i, d.track_id, d.cls_id, f"{d.score:.3f}", str(d.bbox),
             f"{d.bbox[2] - d.bbox[0]}x{d.bbox[3] - d.bbox[1]}"]
            for i, d in enumerate(dets)]
    tables = [{"title": "检出目标", "headers": ["#", "track", "类别", "置信度", "bbox", "尺寸"],
               "rows": rows}] if rows else []
    return res(text="\n".join(text),
               images=[{"title": "检测标注", "data": _img_data_url(canvas)}], tables=tables)


def h_vlm(p: dict) -> dict:
    from pipe.composer import VLMAnalyzer, SubmitRequest, Detection
    vlm = VLMAnalyzer(crop_padding=_float(p, "crop_padding", 0.15),
                      max_resolution=tuple(_int_tuple(p, "max_resolution", 2) or (1280, 720)),
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
        action = _vlm_real(p, payload, frame)
        text.append(f"\n真实 VLM 返回 action: {action}")
    else:
        text.append("\n（未勾选真实 VLM，演示 action=fire）")
    return res(text="\n".join(text))


def h_alarm(p: dict) -> dict:
    from pipe.composer import AlarmPolicy
    kind = p.get("kind", "window")
    ap = AlarmPolicy(task_id=_int(p, "task_id", 1), kind=kind,
                     target_actions=_str_list(p, "target_actions"),
                     smooth_frames=_int(p, "smooth_frames", 1),
                     hit_ratio=_float(p, "hit_ratio", 1.0),
                     alarm_cooldown=_float(p, "alarm_cooldown", 60.0))
    script = str(p.get("timeline", ""))
    lines = [l for l in script.splitlines() if l.strip() and not l.strip().startswith("#")]
    events, log = [], []
    if kind == "dwell":
        for l in lines:
            parts = [x.strip() for x in re.split(r"[,，\s]+", l) if x.strip()]
            if len(parts) < 3:
                continue
            ts, tid, cls = float(parts[0]), int(parts[1]), int(parts[2])
            ev = ap.on_dwell(tid, cls, ts)
            tag = f"t={ts:>6.1f} track={tid} cls={cls}"
            log.append(f"  {'⚠ 告警 ' + ev[0].action if ev else '○ 无'}")
            events += ev
    else:
        for l in lines:
            parts = [x.strip() for x in re.split(r"[,，\s]+", l) if x.strip()]
            if len(parts) < 3:
                continue
            ts, tk = float(parts[0]), int(parts[1])
            action = " ".join(parts[2:])
            ev = ap.on_vlm_result(tk, action, ts)
            tag = f"t={ts:>6.1f} track_key={tk} action={action}"
            log.append(f"  {'⚠ 告警 ' + ev[0].action if ev else '○ 无'}")
            events += ev
    text = (f"判定: {kind} | 目标动作: {ap.target_actions or '(无)'} | "
            f"窗口/驻留: {ap.smooth_frames} | 命中比例: {ap.hit_ratio} | "
            f"冷却: {ap.alarm_cooldown}s\n告警 {len(events)} 次\n" + "\n".join(log))
    table = {"title": "告警事件", "headers": ["ts", "gran", "track_key", "action"],
             "rows": [[f"{e.ts:.2f}", e.gran, e.track_key, e.action] for e in events]}
    return res(text=text, tables=[table] if events else [])


def h_compose(p: dict) -> dict:
    from pipe.composer import compose, Detection
    mode = p.get("algo_mode", "small_crop")
    spec = {"task_id": _int(p, "task_id", 1), "algo_mode": mode,
            "extract_frame_rate": _int(p, "extract_frame_rate", 3),
            "smooth_frames": _int(p, "smooth_frames", 1),
            "target_actions": _str_list(p, "target_actions"),
            "alarm_cooldown": _float(p, "alarm_cooldown", 60.0),
            "class_limits": _class_limits(p.get("class_limits", "")),
            "check_interval": _float(p, "check_interval", 3.0),
            "crop_padding": _float(p, "crop_padding", 0.15),
            "vlm_max_resolution": tuple(_int_tuple(p, "vlm_max_resolution", 2) or (1280, 720)),
            "vlm_prompt": str(p.get("vlm_prompt", "")),
            "vlm_hit_ratio": _float(p, "vlm_hit_ratio", 1.0),
            "roi_output": _bool(p, "roi_output")}
    if mode in ("small_only", "small_crop", "small_full"):
        model = _find_file(p.get("yolo_model", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
        spec.update(yolo_model=str(model), yolo_conf=_float(p, "yolo_conf", 0.35),
                    yolo_iou=_float(p, "yolo_iou", 0.7), yolo_imgsz=_int(p, "yolo_imgsz", 640),
                    yolo_max_det=_int(p, "yolo_max_det", 300),
                    yolo_classes=_int_list(p, "yolo_classes", None))
        dev = str(p.get("yolo_device", "")).strip()
        if dev:
            spec["yolo_device"] = dev
    pipe = compose(spec)

    vlm_action = str(p.get("vlm_action", "fire")).strip() or "fire"
    feedback = _bool(p, "vlm_feedback")
    log, tables, images = [], [], []
    cur_ts = 0.0

    def _step_out(tag: str, subs, al, ts: float) -> None:
        nonlocal cur_ts
        cur_ts = ts
        log.append(f"{tag} → 报送 {len(subs)} 条, 告警 {len(al)} 条")
        for s in subs:
            log.append(f"    · {s.ref} (gran={s.gran}, track_key={s.track_key}, 目标数={len(s.dets)})")
        for e in al:
            log.append(f"    ⚠ {e.action}")
        if feedback and subs and pipe.vlm is not None and mode != "small_only":
            for s in subs:
                ev = pipe.on_vlm_result(s.track_key, vlm_action, ts)
                for e in ev:
                    log.append(f"    ★ VLM({vlm_action}) 命中 → {e.action}")

    if p.get("feed") == "real":
        img_path, frame = _load_frame(p)
        if pipe.yolo is None:
            raise RuntimeError("该模式无 YOLO，真实图片驱动仅适用于 small_* / 有 YOLO 的模式")
        dets = pipe.yolo.infer(frame)
        canvas = frame.copy()
        _draw_dets(canvas, dets)
        images.append({"title": "YOLO 标注", "data": _img_data_url(canvas)})
        subs, al = pipe.step(0, 0.0, dets, frame=frame)
        _step_out(f"真实图片: {img_path.name} (检出 {len(dets)} 个)", subs, al, 0.0)
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
            _step_out(f"步: 帧{frame_idx} ts={ts:.1f} cls{cls}×{count}", subs, al, ts)
    text = f"模式: {mode} | " + "; ".join(f"{k}={v}" for k, v in spec.items()
                                          if k not in ("vlm_prompt", "class_limits")) + "\n" + "\n".join(log)
    return res(text=text, images=images, tables=tables)


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


def h_extract_faces(p: dict) -> dict:
    from pipe.composer import YOLODetector
    from face_detect.face_detector import FaceDetector
    from pipe.crop_restore import CropRestore, extract_faces
    model = _find_file(p.get("yolo_model", ""), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
    yolo = YOLODetector(model_path=str(model), conf=_float(p, "yolo_conf", 0.35),
                        iou=_float(p, "yolo_iou", 0.7), imgsz=_int(p, "yolo_imgsz", 640),
                        max_det=_int(p, "yolo_max_det", 300),
                        classes=_int_list(p, "yolo_classes", [0]))
    fd = FaceDetector(det_thresh=_float(p, "face_thresh", 0.4))
    tool = CropRestore(upscale=_float(p, "upscale", 1.0), margin=_float(p, "margin", 0.15))
    img_path, frame = _load_frame(p)
    faces = extract_faces(frame, yolo, fd, tool,
                          min_person_short=_int(p, "min_person_short", 30),
                          dedup_iou=_float(p, "dedup_iou", 0.6))
    canvas = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = f.person_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        fx1, fy1, fx2, fy2 = f.face_bbox
        cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.putText(canvas, f"face#{f.index} {f.score:.2f}", (fx1, max(fy1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    images = [{"title": "标注原图（蓝=人框, 绿=人脸框）", "data": _img_data_url(canvas)}]
    for f in faces:
        images.append({"title": f"原分辨率人脸 #{f.index} ({f.face_img.shape[1]}x{f.face_img.shape[0]})",
                       "data": _img_data_url(f.face_img, 92)})
    text = (f"图片: {img_path.name} | 提取到 {len(faces)} 张原分辨率人脸 | "
            f"upscale={tool.upscale} margin={tool.margin}")
    return res(text=text, images=images)


def h_face_monitor(p: dict) -> dict:
    from face_detect.face_detector import FaceDetector
    from face_embed.face_embedder import FaceEmbedder
    from face_embed.face_store import FaceStore
    from face_scan.frame_scheduler import FrameScheduler
    from face_scan.face_monitor import FaceMonitor
    db = _db_path(p)
    if not db.exists():
        return res(ok=False, error=f"底库不存在: {db}（请先在 FaceStore 页注册并保存底库）")
    store = FaceStore().load(db)
    det = FaceDetector(device=str(p.get("device", "auto")))
    embedder = FaceEmbedder(device=str(p.get("device", "auto")))
    sch = FrameScheduler(frame_skip=_int(p, "frame_skip", 3),
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


# ---------------- 模块清单（前端表单的"参数即表单"来源） ----------------

PARAMS = {
    "frame_manager": {
        "sampling": {"label": "取图节奏", "type": "select", "default": "analysis",
                     "options": ["analysis", "wall_clock", "frame_count"],
                     "help": "analysis=随抽帧节奏; wall_clock=按墙钟秒; frame_count=按帧数"},
        "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                       "help": "每 N 帧取一张（analysis 模式用）"},
        "interval_sec": {"label": "墙钟间隔（秒）", "type": "float", "default": 3.0,
                         "help": "每隔多少秒取一张（wall_clock 模式用）"},
        "interval_frames": {"label": "帧数间隔（帧）", "type": "int", "default": 30,
                            "help": "每隔多少帧取一张（frame_count 模式用）"},
        "n_frames": {"label": "模拟帧数", "type": "int", "default": 20,
                     "help": "模拟播放多少帧，用来观察逐帧取/跳决策"},
        "ts_step": {"label": "每帧时间步长（秒）", "type": "float", "default": 1.0,
                    "help": "模拟时间轴：相邻两帧相差几秒"},
    },
    "frame_scheduler": {
        "mode": {"label": "模式", "type": "select", "default": "queue",
                 "options": ["queue", "adaptive"],
                 "help": "queue=按积压队列信号放宽; adaptive=按推理耗时自动放宽"},
        "scenario": {"label": "模拟场景", "type": "select", "default": "平稳",
                     "options": ["平稳", "积压浪涌", "耗时渐增"],
                     "help": "queue 模式用「积压浪涌」造高峰；adaptive 用「耗时渐增」"},
        "frame_skip": {"label": "基础抽帧间隔（帧）", "type": "int", "default": 3,
                       "help": "默认每几帧取一张；积压时会自动加大"},
        "queue_threshold": {"label": "积压阈值（队列）", "type": "int", "default": 32,
                            "help": "积压超过它就开始放宽抽帧"},
        "backpressure_multiplier": {"label": "放宽倍数", "type": "int", "default": 2,
                                    "help": "放宽时抽帧间隔乘以几倍"},
        "max_frame_skip": {"label": "放宽上限（0=自动）", "type": "int", "default": 0,
                           "help": "抽帧间隔最多放宽到多少；0=不限"},
        "adaptive_relax_ratio": {"label": "放宽比例（自适应）", "type": "float", "default": 1.2,
                                 "help": "耗时变长时，间隔乘多少"},
        "adaptive_recover_ratio": {"label": "恢复比例（自适应）", "type": "float", "default": 0.5,
                                   "help": "耗时恢复后，间隔按什么比例回落"},
        "fps": {"label": "视频帧率（预算用）", "type": "float", "default": 25.0,
                "help": "用来估算每帧的推理时间预算"},
        "n_frames": {"label": "模拟帧数", "type": "int", "default": 30,
                     "help": "模拟多少帧的调度过程"},
    },
    "yolo": {
        "model_path": {"label": "权重模型", "type": "file", "src": "model",
                       "help": "检测权重；不选则用默认 yolov8n.pt"},
        "image": {"label": "测试图片", "type": "file", "src": "img",
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
        "filter_conf": {"label": "过滤链：置信度下限（0=不启用）", "type": "float", "default": 0.0,
                        "help": "低于它的检出直接丢弃，不参与报送"},
        "min_short": {"label": "过滤链：短边最小像素（0=不启用）", "type": "int", "default": 0,
                      "help": "目标短边小于它就不报送"},
        "min_long": {"label": "过滤链：长边最小像素（0=不启用）", "type": "int", "default": 0,
                     "help": "目标长边小于它就不报送"},
        "class_limits": {"label": "每类数量上下限（如 0:1,300; 2:1,10）", "type": "str",
                         "default": "0:1,300", "help": "类别:下限,上限，分号分隔；空=不报送"},
        "check_interval": {"label": "报送节拍（秒/类）", "type": "float", "default": 3.0,
                           "help": "同一类别每隔几秒才报一次，避免刷屏"},
        "roi": {"label": "裁剪送审（crop:cls）", "type": "bool", "default": False,
                "help": "开=送该类裁剪图; 关=送全帧"},
        "ts": {"label": "报送时间戳（秒）", "type": "float", "default": 0.0,
               "help": "本帧的时间戳，随报送结果一起带出"},
    },
    "vlm": {
        "crop_padding": {"label": "裁剪外扩比例", "type": "float", "default": 0.15,
                         "help": "裁剪目标时往外多扩多少，避免贴边"},
        "max_resolution": {"label": "送审缩放上限（宽,高）", "type": "str", "default": "1280,720",
                           "help": "送审图片最大尺寸，超出会缩小"},
        "prompt": {"label": "提示词", "type": "text", "default": "",
                   "help": "留空用默认分类提示词"},
        "ref": {"label": "素材引用", "type": "select", "default": "crop:cls0",
                "options": ["crop:cls0", "full:cls0", "full:global"],
                "help": "crop:cls0=裁剪第 0 类目标送审；full:cls0=全帧按第 0 类；full:global=全帧全局"},
        "gran": {"label": "报送粒度", "type": "select", "default": "class",
                 "options": ["class", "scene"],
                 "help": "class=按类别一次; scene=整场景一次"},
        "track_key": {"label": "track_key（类别编号，scene=-1）", "type": "int", "default": 0,
                      "help": "素材对应的目标类别编号"},
        "bbox": {"label": "单目标 bbox（x1,y1,x2,y2）", "type": "str", "default": "10,10,100,200",
                 "help": "单个目标的位置，选裁剪送审时用"},
        "det_count": {"label": "该类目标数（无单目标时）", "type": "int", "default": 2,
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
    "alarm": {
        "kind": {"label": "判定方式", "type": "select", "default": "window",
                 "options": ["window", "dwell", "inspection"],
                 "help": "window=VLM滑动窗口; dwell=驻留秒数; inspection=直接落盘"},
        "task_id": {"label": "任务编号", "type": "int", "default": 1,
                    "help": "告警会带上这个任务编号"},
        "target_actions": {"label": "目标动作（逗号分隔，空=巡检）", "type": "str",
                           "default": "fire,fight",
                           "help": "哪些动作算告警，多个用逗号分隔"},
        "smooth_frames": {"label": "窗口长度 / 驻留秒数", "type": "int", "default": 2,
                          "help": "window=连续几帧看; dwell=驻留几秒才报"},
        "hit_ratio": {"label": "命中比例", "type": "float", "default": 1.0,
                      "help": "窗口内命中比例到多少才告警（1.0=全部命中）"},
        "alarm_cooldown": {"label": "冷却时间（秒）", "type": "float", "default": 60.0,
                           "help": "同一目标告警后多久内不再重复报"},
        "timeline": {"label": "事件时间线（每行一条）", "type": "text",
                     "default": "# window/inspection: ts, track_key, action\n"
                                "1.0, 0, fire\n2.0, 0, fire\n3.0, 0, fire\n"
                                "# dwell: ts, track_id, cls_id\n10.0, 1, 0\n11.0, 1, 0\n12.0, 1, 0",
                     "help": "window/inspection 用「ts, track_key, action」；dwell 用「ts, track_id, cls_id」"},
    },
    "compose": {
        "algo_mode": {"label": "分析模式", "type": "select", "default": "small_crop",
                      "options": ["small_only", "small_crop", "small_full", "large_only"],
                      "help": "只用小模型 / 小模型裁剪送大模型 / 小模型全帧送大模型 / 只用大模型"},
        "task_id": {"label": "任务编号", "type": "int", "default": 1,
                    "help": "告警与报送会带上这个编号"},
        "feed": {"label": "驱动方式", "type": "select", "default": "script",
                 "options": ["script", "real"],
                 "help": "script=脚本化检测输入; real=真实图片+YOLO 推理"},
        "script": {"label": "脚本（每行: 帧,秒,类别,数量[,起始track]）", "type": "text",
                   "default": "0,0.0,0,2\n3,1.0,0,1\n6,3.0,0,1",
                   "help": "small_only 驻留告警用同一起始 track 连续几行模拟驻留"},
        "image": {"label": "测试图片（real 模式）", "type": "file", "src": "img",
                  "help": "real 模式用的输入图片"},
        "extract_frame_rate": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                               "help": "每隔几帧取一张送检"},
        "smooth_frames": {"label": "窗口长度 / 驻留秒数", "type": "int", "default": 1,
                          "help": "告警判定：连续几帧 / 驻留几秒"},
        "target_actions": {"label": "目标动作", "type": "str", "default": "fire,fight",
                           "help": "哪些动作算告警，逗号分隔"},
        "alarm_cooldown": {"label": "冷却（秒）", "type": "float", "default": 60.0,
                           "help": "同一目标告警后多久不重复报"},
        "class_limits": {"label": "每类数量上下限", "type": "str", "default": "0:1,300",
                         "help": "类别:下限,上限；决定按类报送"},
        "check_interval": {"label": "报送节拍 / 墙钟间隔（秒）", "type": "float", "default": 3.0,
                           "help": "同一类每隔几秒报一次"},
        "roi_output": {"label": "裁剪送审（crop:cls）", "type": "bool", "default": False,
                       "help": "勾选=目标裁剪图送大模型；不勾=送全帧"},
        "crop_padding": {"label": "裁剪外扩比例", "type": "float", "default": 0.15,
                         "help": "裁剪目标时往外多扩多少"},
        "vlm_max_resolution": {"label": "VLM 缩放上限", "type": "str", "default": "1280,720",
                               "help": "送大模型的图最大尺寸"},
        "vlm_prompt": {"label": "VLM 提示词", "type": "text", "default": "",
                       "help": "留空用默认"},
        "vlm_hit_ratio": {"label": "VLM 命中比例", "type": "float", "default": 1.0,
                          "help": "大模型判定命中的比例阈值（1.0=都算命中）"},
        "yolo_model": {"label": "YOLO 权重", "type": "file", "src": "model",
                       "help": "检测权重；不选默认 yolov8n.pt"},
        "yolo_conf": {"label": "YOLO conf", "type": "float", "default": 0.35,
                      "help": "置信度阈值"},
        "yolo_iou": {"label": "YOLO iou", "type": "float", "default": 0.7,
                     "help": "去重阈值"},
        "yolo_imgsz": {"label": "YOLO imgsz", "type": "int", "default": 640,
                       "help": "推理分辨率"},
        "yolo_max_det": {"label": "YOLO max_det", "type": "int", "default": 300,
                         "help": "单帧最多目标数"},
        "yolo_classes": {"label": "YOLO 类别（空=不限）", "type": "str", "default": "0",
                         "help": "默认 0=人"},
        "yolo_device": {"label": "YOLO 设备（空=自动）", "type": "str", "default": "",
                        "help": "如 cpu 或 0"},
        "vlm_feedback": {"label": "VLM 回填（把 action 喂回告警）", "type": "bool", "default": True,
                         "help": "把大模型判定的动作喂给告警策略"},
        "vlm_action": {"label": "回填的 VLM action", "type": "str", "default": "fire",
                       "help": "回填时使用的动作名"},
    },
    "face_detect": {
        "image": {"label": "测试图片", "type": "file", "src": "img",
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
        "use_crop": {"label": "生成对齐裁剪图", "type": "bool", "default": True,
                     "help": "勾选=同时输出 5 点对齐后的脸部裁剪图"},
    },
    "face_embed": {
        "image": {"label": "测试图片（自动先检测再嵌入）", "type": "file", "src": "img",
                  "help": "会自动先检测人脸，再对每张脸提取向量"},
        "device": {"label": "设备", "type": "select", "default": "auto",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "det_thresh": {"label": "检测阈值", "type": "float", "default": 0.5,
                       "help": "置信度低于它的脸不认"},
        "det_size": {"label": "检测分辨率（宽,高）", "type": "str", "default": "640,640",
                     "help": "越高越慢，小脸更清楚"},
    },
    "face_store": {
        "action": {"label": "操作", "type": "select", "default": "register_dir",
                   "options": ["register_dir", "register_single", "query", "self_check", "list"],
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
                         "default": "umvp/face_detect/test_imgs",
                         "help": "目录里每张图注册成一个身份，文件名即身份名"},
        "image": {"label": "图片（register_single / query 用）", "type": "file", "src": "img",
                  "help": "单张注册或查询用的图片"},
        "name": {"label": "身份名（register_single 用）", "type": "str", "default": "",
                 "help": "注册到库里的身份名字"},
    },
    "extract_faces": {
        "image": {"label": "测试图片", "type": "file", "src": "img",
                  "help": "含人的原图，从中提取原生分辨率人脸"},
        "yolo_model": {"label": "YOLO 权重", "type": "file", "src": "model",
                       "help": "检测权重；不选默认 yolov8n.pt"},
        "yolo_conf": {"label": "YOLO conf", "type": "float", "default": 0.35,
                      "help": "先识别人的置信度阈值"},
        "yolo_iou": {"label": "YOLO iou", "type": "float", "default": 0.7,
                     "help": "重叠超过它的人框合并"},
        "yolo_imgsz": {"label": "YOLO imgsz", "type": "int", "default": 640,
                       "help": "推理分辨率"},
        "yolo_max_det": {"label": "YOLO max_det", "type": "int", "default": 300,
                         "help": "单帧最多目标数"},
        "yolo_classes": {"label": "YOLO 类别（默认人）", "type": "str", "default": "0",
                         "help": "识别哪类目标（0=人）"},
        "face_thresh": {"label": "人脸检测阈值", "type": "float", "default": 0.4,
                        "help": "人框内再找脸的人脸阈值"},
        "upscale": {"label": "人裁剪图放大倍数", "type": "float", "default": 1.0,
                    "help": "人框先放大几倍再找脸（1.0=原样）"},
        "margin": {"label": "人脸出图外扩比例", "type": "float", "default": 0.15,
                   "help": "输出人脸时往外多扩多少"},
        "min_person_short": {"label": "最小人框长边（像素）", "type": "int", "default": 30,
                             "help": "人框长边小于它直接跳过"},
        "dedup_iou": {"label": "人脸去重 IoU", "type": "float", "default": 0.6,
                      "help": "两张脸重叠超过它算同一张，只留一张"},
    },
    "face_monitor": {
        "video": {"label": "测试视频", "type": "file", "src": "vid",
                  "help": "要扫描的视频文件"},
        "db_path": {"label": "底库文件（.npz）", "type": "str", "default": "out/face_db.npz",
                    "help": "拿视频里的人脸去比对的底库"},
        "device": {"label": "设备", "type": "select", "default": "auto",
                   "options": ["auto", "cuda", "cpu"],
                   "help": "auto=自动; cuda=显卡; cpu=CPU"},
        "frame_skip": {"label": "抽帧间隔（帧）", "type": "int", "default": 3,
                       "help": "每隔几帧检测一次"},
        "queue_threshold": {"label": "积压阈值", "type": "int", "default": 32,
                            "help": "积压超过它放宽抽帧"},
        "backpressure_multiplier": {"label": "放宽倍数", "type": "int", "default": 2,
                                    "help": "放宽时抽帧间隔乘以几倍"},
        "max_frames": {"label": "限帧数（0=全部）", "type": "int", "default": 60,
                       "help": "测试视频为 4K 长视频，建议先限帧"},
        "save_dir": {"label": "命中帧保存目录", "type": "str",
                     "default": "tests/data/out/faces/monitor",
                     "help": "命中（在底库中找到的人脸）的帧图存到这里"},
    },
}

MODULES = {
    "frame_scheduler": {
        "name": "抽帧调度 FrameScheduler", "group": "帧调度",
        "desc": "复用 face_scan.frame_scheduler：抽帧间隔 + 积压/耗时自动放宽。",
        "params": PARAMS["frame_scheduler"], "handler": h_frame_scheduler},
    "frame_manager": {
        "name": "帧管理 FrameManager", "group": "管道四模块",
        "desc": "复用 pipe.composer.FrameManager：三种取图节奏决策。",
        "params": PARAMS["frame_manager"], "handler": h_frame_manager},
    "yolo": {
        "name": "小模型识别 YOLODetector", "group": "管道四模块",
        "desc": "复用 pipe.composer.YOLODetector：推理 + 过滤链 + 按类报送。",
        "params": PARAMS["yolo"], "handler": h_yolo},
    "vlm": {
        "name": "大模型研判 VLMAnalyzer", "group": "管道四模块",
        "desc": "复用 pipe.composer.VLMAnalyzer：素材包准备，可接真实 VLM。",
        "params": PARAMS["vlm"], "handler": h_vlm},
    "alarm": {
        "name": "告警策略 AlarmPolicy", "group": "管道四模块",
        "desc": "复用 pipe.composer.AlarmPolicy：window/dwell/inspection + 冷却。",
        "params": PARAMS["alarm"], "handler": h_alarm},
    "compose": {
        "name": "四模式集成 compose", "group": "管道四模块",
        "desc": "复用 pipe.composer.compose：四种分析模式端到端跑通。",
        "params": PARAMS["compose"], "handler": h_compose},
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
    "extract_faces": {
        "name": "原分辨率提取 extract_faces", "group": "人脸识别链路",
        "desc": "复用 pipe.crop_restore：YOLO 识人 → 裁人 → SCRFD → 原图出脸。",
        "params": PARAMS["extract_faces"], "handler": h_extract_faces},
    "face_monitor": {
        "name": "视频扫描 FaceMonitor", "group": "视频监控",
        "desc": "复用 face_scan.face_monitor：抽帧 + 检测 + 嵌入 + 检索全链路。",
        "params": PARAMS["face_monitor"], "handler": h_face_monitor},
}


def _file_index() -> dict:
    def names(d: Path):
        return sorted({p.name for p in d.glob("*")
                       if p.is_file() and p.suffix.lower() in _IMG_SUFFIX})
    return {
        "img": sorted({p.name for p in PIC_DIR.glob("*") if p.suffix.lower() in _IMG_SUFFIX}
                      | {p.name for p in TEST_IMGS.glob("*") if p.suffix.lower() in _IMG_SUFFIX}),
        "vid": sorted({p.name for p in VID_DIR.glob("*")
                       if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")}),
        "model": sorted({p.name for p in MODELS_DIR.glob("*.pt")}),
    }


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
    return {"modules": {mid: {"name": m["name"], "group": m["group"],
                              "desc": m["desc"], "params": m["params"],
                              "estimate": _module_estimate(mid)}
                        for mid, m in MODULES.items()},
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
    if kind not in ("module", "pipeline"):
        return {"ok": False, "error": f"未知配置类型: {kind}"}
    if kind == "module" and module_id not in MODULES:
        return {"ok": False, "error": f"未知模块: {module_id}"}
    if kind == "pipeline":
        module_id = ""  # 链路配置整体存于 params（完整 pipeline spec）
    if not isinstance(params, dict):
        return {"ok": False, "error": "params 必须是对象"}
    try:
        rid, created = db.save_preset(kind, name, module_id, params)
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
        rid = int(body.get("id"))
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
        rid = int(body.get("id"))
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


# ---------------- 数据库数据浏览（只读快照，见 doc/配置参数列式化存储设计.md §4） ----------------

def _db_tables() -> dict:
    if not _DB_ENABLED:
        return {"ok": False, "error": "数据库未启用（--no-db）"}
    try:
        return {"ok": True, **db.table_snapshots()}
    except Exception as e:  # DB 不可用不影响服务器运行
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
            isinstance(x, int) and not isinstance(x, bool) for x in steps):
        return {"ok": False, "error": "steps 必须是 preset id 数组"}
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
        pid = int(body.get("id"))
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
        pid = int(body.get("id"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "error": "请求体缺少合法 id"}
    try:
        db.delete_pipeline(pid)
        return {"ok": True, "deleted": pid}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


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

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/schema":
            self._json(_schema())
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

        # ---- 链路生成 / 智能助手（占位壳子，功能在 P2/P4 落地）----
        if self.path == "/api/pipeline/build":
            self._json({"ok": True, "stub": True,
                        "note": "链路生成（P2 占位）尚未实现"})
            return
        if self.path == "/api/pipeline/agent":
            self._json({"ok": True, "stub": True,
                        "note": "智能助手（P4 占位）尚未实现"})
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
        if mod not in MODULES:
            self._json({"ok": False, "error": f"未知模块: {mod}"})
            return
        t0 = time.time()
        with _TEST_LOCK:  # 一次只跑一个测试，避免模型抢占内存
            try:
                result = MODULES[mod]["handler"](params)
            except Exception as e:  # 测试失败也要给前端明确报错
                tb = traceback.format_exc()
                print(tb)
                result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "traceback": tb[-1800:]}
        result.setdefault("elapsed_sec", round(time.time() - t0, 2))
        print(f"[test] {mod} {time.time() - t0:.1f}s ok={result.get('ok')}")
        self._json(result)


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
            # 注册各模块参数 schema（参数名+类型 → 列式参数表），须在 init_db 前
            db.configure_modules(
                {mid: {k: spec.get("type", "str") for k, spec in m["params"].items()}
                 for mid, m in MODULES.items()})
            db.init_db()
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
