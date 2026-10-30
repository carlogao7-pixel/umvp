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
import html
import importlib
import inspect
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
from collections.abc import Mapping
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
    "gate_filter": "门控",
    "plate_recog": "车牌识别", "face_detect": "人脸检测", "face_embed": "人脸嵌入",
    "face_store": "向量底库", "extract_faces": "原分辨率提取",
    "video_embed": "向量化",
}


def _stage_label(mid) -> str:
    return _STAGE_LABELS.get(mid) or (mid or "")


def _gate_stats_text(pipe) -> str:
    """门控分键拦截明细（产物 gate_filter.stats）→ 时间线/日志附加文案。

    多实例逐实例收集 blocked_detail（短名别名仅单实例存在，全名恒可读）；"*" 键 =
    匿名桶（key 值 None/空/0 归一），排序置顶一眼看出匿名拦截量。无明细返回空串。
    """
    arts = getattr(pipe, "artifacts", None) or {}
    poss = (getattr(pipe, "module_positions", None) or {}).get("gate_filter") or []
    names = [f"gate_filter#{p}.stats" for p in poss] or ["gate_filter.stats"]
    parts = []
    for nm in names:
        for rec in (arts.get(nm) or []):
            detail = rec.get("blocked_detail") if isinstance(rec, dict) else None
            for k, n in sorted((detail or {}).items(),
                               key=lambda kv: (kv[0] != "*", str(kv[0]))):
                if n:
                    parts.append(f"*（匿名）×{n}" if k == "*" else f"{k}×{n}")
    return "（拦截明细：" + "、".join(parts) + "）" if parts else ""


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


def _bbox_of(d):
    """检测项 bbox 双形态读取（Detection 对象 / dict 统一记录）；缺 bbox 返回 None。"""
    bb = d.get("bbox") if isinstance(d, Mapping) else getattr(d, "bbox", None)
    return None if bb is None else tuple(int(v) for v in bb)


def _match_crops_for_req(req, crops) -> list:
    """在目标裁剪产物 last_crops 里找与本次报送目标对应的**全部**子图。

    匹配依据（下游可核对）：优先 req.det.bbox（单目标素材），否则取 req.dets 各框；
    按 bbox 精确相等或 IoU≥0.5 命中（裁剪含 margin/缩放，用 IoU 而非全等）。
    det/dets 双形态读取（Detection 对象或 dict 记录），缺 bbox 的项跳过不抛错。
    返回按 crops 顺序命中的对象列表（基数约定：一条报送记录可涉多个对象 → 图集）。
    """
    if not crops:
        return []
    wanted = []
    det = getattr(req, "det", None)
    if det is not None:
        bb = _bbox_of(det)
        if bb is not None:
            wanted.append(bb)
    for d in getattr(req, "dets", None) or []:
        bb = _bbox_of(d)
        if bb is not None:
            wanted.append(bb)
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


def _annotate_stats(pipe, dets_arg=None, frame=None):
    """annotate 阶段本帧展示统计（驱动层 timeline/日志用，WO-0022）。

    record 模式（scope=record）产物为 annotate.annotated_records（带图记录流），
    last_annotated 不更新——展示取数零跨板块：
      - 出图张数/截断数：用 annotate 模块同一套画框判定（pipe.annotate._visible）对
        **本帧输入记录**干跑计数——drawn = min(可画数, 上限)、dropped = max(0, 可画数-上限)，
        与模块实际行为逐条一致（_visible 正是 annotate_frame/annotate_records 共用的
        唯一画框判定实现）；上限 = 实例公开属性 max_records。
      - 输入记录解析（与 composer._port_at 默认语义一致）：该实例显式连线产物
        （stage_inputs.dets）→ 主干在**该实例位置之前**最近一次改写产物（yolo →
        det_filter → seq_filter.dets，位置即处理序，取位置最大者）→ 驱动层传入的
        dets（链上无改写阶段时）；任一步产物缺失向下回退。
    返回 (rec_stats, has_frame_inst, ran)：
      rec_stats      —— [(drawn, cap, dropped), ...]（record 实例按位置升序）；
      has_frame_inst —— 存在 frame 模式实例（或无实例）→ 驱动层保留原「标注图」文案行；
      ran            —— 本帧 annotate 是否实际执行（原图与检出俱备才跑，同 composer 门）。
    """
    insts = [(int(pos), obj) for pos, obj in (getattr(pipe, "annotators", None) or [])]
    if not insts and getattr(pipe, "annotate", None) is not None:
        insts = [(None, pipe.annotate)]  # 单插槽直挂（测试/旧装配）回退
    rec_insts = []
    has_frame = not insts  # 无实例（异常 spec）→ 视同有 frame 实例，保持旧文案行为
    for _pos, obj in insts:
        if getattr(obj, "scope", "frame") == "record":
            rec_insts.append((_pos, obj))
        else:
            has_frame = True
    ran = frame is not None and dets_arg is not None
    if not ran or not rec_insts:
        return [], has_frame, ran
    from pipe.records import to_record as _to_record
    from pipe.annotate import _visible as _anno_visible
    from pipe.conditions import parse_condition as _parse_cond
    arts = getattr(pipe, "artifacts", None) or {}
    si = getattr(pipe, "stage_inputs", None) or {}
    positions = getattr(pipe, "module_positions", None) or {}

    def _trunk_inputs(anno_pos):
        """主干检出记录（annotate 位置的默认输入）：只统计位置在 annotate 之前的改写
        （位置即处理序），取其中位置最大者的 .dets 产物——annotate 之后的阶段
        （如事件尾 seq_filter）不改写 annotate 的输入。"""
        best, best_pos = None, -1
        for mid in ("yolo", "det_filter", "seq_filter"):
            for p_ in (positions.get(mid) or []):
                if anno_pos is not None and p_ >= anno_pos:
                    continue
                a = arts.get(f"{mid}#{p_}.dets")
                if a is not None and p_ > best_pos:
                    best, best_pos = a, p_
        if best is None:  # 位置未知（单插槽直挂回退）→ 短名别名按处理序兜底
            for mid in ("yolo", "det_filter", "seq_filter"):
                a = arts.get(f"{mid}.dets")
                if a is not None:
                    best = a
        return best

    stats = []
    for pos, inst in rec_insts:
        cap = max(1, int(getattr(inst, "max_records", 32)))
        inputs = None
        mapping = (si.get(f"#{pos}") if pos is not None else None) or si.get("annotate") or {}
        if isinstance(mapping, dict) and mapping.get("dets"):
            up = arts.get(mapping["dets"])
            if up is not None:
                inputs = up
        if inputs is None:
            inputs = _trunk_inputs(pos)
        if inputs is None:
            inputs = [_to_record(d) for d in (dets_arg or [])]
        groups = _parse_cond(getattr(inst, "filter", "") or "")
        matching = sum(1 for _ in _anno_visible(inputs, getattr(inst, "classes", None), groups))
        stats.append((min(matching, cap), cap, max(0, matching - cap)))
    return stats, has_frame, ran


def _apply_vlm_inputs(pipe, subs):
    """vlm 阶段声明 inputs.dets 时，按指定上游产物重构送审请求（决定"报送哪些检测"）。

    - 未声明 vlm.inputs.dets → 原样返回 step() 的默认 submits（零回归）；
    - 声明了 → 从 pipe.artifacts 取该产物（yolo.dets=筛选前 / det_filter.dets=筛选后 /
      seq_filter.dets=门控后），按 cls_id 分组复用 Pipeline._submits（gran=class 语义）；
    - 产物本帧未登记（如上游无产出）→ 返回默认 submits（不抛错，行为与现状一致）。
    Detection 对象与统一记录 dict（含 cls_id/bbox）均可消费；无 cls_id 的项跳过。

    P2-0：stage_inputs 键为位置串（"#<position>"），取该链路中 **vlm 阶段的位置** 对应项
    （多实例时取第一个声明了 dets 的 vlm 实例）。
    """
    si = getattr(pipe, "stage_inputs", None) or {}
    art = None
    for pos in ((getattr(pipe, "module_positions", None) or {}).get("vlm") or []):
        d = si.get(f"#{pos}") or {}
        if isinstance(d, dict) and d.get("dets"):
            art = d["dets"]
            break
    if not art:
        return subs
    items = (getattr(pipe, "artifacts", None) or {}).get(art)
    if items is None:
        return subs
    from pipe.records import to_record
    try:
        # WO-0014.6：喂 _submits 前经 to_record 逐条归一（链路通用语 = dict Record）——
        # dict 直通、Detection/dataclass 转同名字段 dict；缺 cls_id 的项由 _submits 跳过
        # （与旧 _as_detections「无 cls_id 跳过」语义对齐）。不再把 dict 产物强转 Detection
        # 对象（WO-0014.3 后 _submits 只吃 dict，强转会 AttributeError 被吞 → 静默回落默认报送集）。
        return list(pipe._submits([to_record(d) for d in items]))  # 复用 Pipeline 的按类分组（gran=class）语义
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
    """快速探测端点是否可连（TCP 连接），避免 requests 长超时把整轮测试拖住。

    https 端点默认端口 443、http 默认 80（与 requests 一致）；无 scheme/host 视为不可达。
    """
    import socket
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
        host = u.hostname
        if not host:
            return False
        if u.port is not None:
            port = u.port
        else:
            port = 443 if u.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# VLM 端点/模型/key 解析：环境变量（部署级强制覆盖）→ 提供方插槽（vlm_provider）→
# 链路阶段参数 → 内置默认。导出 env 即全链路切到该端点、无需改 DB；
# 不设任何 env 且未选提供方时逐字回落到改动前行为（零回归）。
VLM_DEFAULT_ENDPOINT = "http://117.42.21.253:8000/v1/chat/completions"
VLM_DEFAULT_MODEL = "qwen3-vl-32b"

# VLM 提供方插槽表（WO-0019，仅 OpenAI 兼容范式，_vlm_real 调用形态不变）：
# 新增提供方 = 在此加一个条目（label + endpoint + model；无 endpoint 的条目视为
# "自定义"，回落手填端点/模型）。PARAMS["vlm"].vlm_provider 的 options 由此动态生成。
VLM_PROVIDERS = {
    "qwen_selfhost": {"label": "自建端点 qwen3-vl-32b（默认）",
                      "endpoint": VLM_DEFAULT_ENDPOINT, "model": VLM_DEFAULT_MODEL},
    "glm": {"label": "智谱 glm-4v-flash",
            "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            "model": "glm-4v-flash"},
    "custom": {"label": "自定义（手填端点/模型）"},
}


class VlmHttpError(RuntimeError):
    """VLM 端点 HTTP 错误（携带可读中文提示 hint，供上层每轮最多提示一次）。"""

    def __init__(self, status: int, message: str, hint: str = ""):
        super().__init__(message)
        self.status = status
        self.hint = hint


def _resolve_vlm_cfg(p: dict) -> tuple:
    """解析 (url, model, key)：环境变量优先 → 提供方插槽 → spec 阶段参数 → 内置默认。

    env：VLM_ENDPOINT / VLM_MODEL / VLM_API_KEY（key 兼容别名 GLM_API_KEY，
    两者都设时 VLM_API_KEY 优先）。env 为部署级强制覆盖；未设的项回落后续优先级。
    vlm_provider（提供方插槽，WO-0019）：值在 VLM_PROVIDERS 且条目含 endpoint 时
    endpoint/model 取插槽（覆盖 spec 手填 vlm_endpoint/vlm_model）；provider 为
    custom/缺省/未知时回落 spec 手填、再回落内置默认（零回归）。key 不受 provider 影响。
    """
    slot = VLM_PROVIDERS.get(str(p.get("vlm_provider") or "").strip()) or {}
    if "endpoint" not in slot:  # custom（无 endpoint 条目）/缺省/未知 → 插槽不生效
        slot = {}
    url = (os.environ.get("VLM_ENDPOINT") or "").strip() \
        or slot.get("endpoint") \
        or str(p.get("vlm_endpoint") or "").strip() or VLM_DEFAULT_ENDPOINT
    model = (os.environ.get("VLM_MODEL") or "").strip() \
        or slot.get("model") \
        or str(p.get("vlm_model") or "").strip() or VLM_DEFAULT_MODEL
    key = ((os.environ.get("VLM_API_KEY") or os.environ.get("GLM_API_KEY") or "").strip()
           or str(p.get("vlm_key") or "").strip())
    return url, model, key


def _vlm_real(p: dict, payload: dict, frame) -> tuple:
    """OpenAI 兼容端点调用（返回 (action, description, content)）。

    frame：单张 ndarray（现状，单图消息）或图集列表（image_pick=all，OpenAI 兼容
    **多图**消息：多个 image_url content 项 + 末尾 text）。单图路径与改动前**逐字等价**。
    端点/模型/key 三项经 _resolve_vlm_cfg：env 优先 → 提供方插槽（vlm_provider）→
    spec → 默认。
    timeout=(连接, 读取)：连接阶段最多 5s（端点不可达时快速失败），读取按 vlm_timeout（默认 20s）。
    4xx（401/403 鉴权、429 限额）抛 VlmHttpError（携带可读中文提示）。
    注：多图 content 已用智谱 glm-4v-flash 的 OpenAI 兼容 /chat/completions 实证支持
    （多个 image_url 项 + 末尾 text，按序返回；见 INB-0040）；端点为第三方实现时仍建议自查多图支持。
    """
    import requests
    url, model, key = _resolve_vlm_cfg(p)
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
    status = getattr(resp, "status_code", None)
    if status in (401, 403):
        raise VlmHttpError(status, f"VLM 端点需鉴权(HTTP {status})",
                           "VLM 端点需鉴权：请导出 VLM_API_KEY（或 GLM_API_KEY）")
    if status == 429:
        raise VlmHttpError(status, "VLM 端点限额/过载(HTTP 429)",
                           "VLM 端点限额/过载：429（请稍后重试或更换端点/模型）")
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
    dets = _as_detections(yolo.infer(frame))  # 纯推理（有效性筛选由「检测筛选」模块承担）
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


def h_gate_filter(p: dict) -> dict:
    """门控模块测试页：模拟检测 → 按 key+间隔放行 → 放行/拦截明细（含分键统计）。

    单次运行只体现"首见放行/同帧同桶"；冷却节奏跨帧生效，链路行为用链路测试台看。
    """
    from pipe.gate_filter import GateFilter
    try:
        gf = GateFilter(key=(_str_param(p, "key") or "track_id"),
                        interval=_float(p, "interval", 5.0),
                        filter=_str_param(p, "filter"))
    except ValueError as e:
        return res(ok=False, error=f"门控参数无法解析：{e}")
    dets, err = _parse_sim_dets(p)
    if err:
        return res(ok=False, error=err)
    ts = _float(p, "ts", 0.0)
    keep = gf.run(dets, ts=ts, frame_idx=0)
    keep_ids = {id(d) for d in keep}
    st = gf.last_stats or {}
    detail = "、".join((f"*（匿名）×{n}" if k == "*" else f"{k}×{n}")
                      for k, n in sorted((st.get("blocked_detail") or {}).items(),
                                         key=lambda kv: (kv[0] != "*", str(kv[0])))) or "无"
    text = [f"输入 {len(dets)} 条 | 放行 {len(keep)} 条 | 拦截 {len(dets) - len(keep)} 条（ts={ts}）",
            f"门控: key={gf.key} | interval={gf.interval} | "
            + (f"filter={gf.filter}" if gf.filter else "filter=（空=全部受门控）"),
            f"拦截分键: {detail}"]
    rows = [[i, d.cls_id, getattr(d, gf.key, "-"), f"{d.score:.3f}",
             "✓ 放行" if id(d) in keep_ids else "✗ 拦截"]
            for i, d in enumerate(dets)]
    tables = [{"title": "逐条判定", "headers": ["#", "类别", f"key({gf.key})", "置信度", "结果"],
               "rows": rows}]
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


# 视频嵌入模块试跑喂帧上限（≈10 片段@6 帧/片段）：mock 档瞬时完成；真档防长视频
# 把整轮试跑拖死（真端点联调请先小视频验证连通性）
_VEMB_TEST_MAX_FRAMES = 60


def h_video_embed(p: dict) -> dict:
    """视频嵌入试跑（模块测试页）：选视频 → 按标称节奏取样帧 → 片段窗口聚合 →
    嵌入（mock/真档）→ 底库入库并落盘。对齐 face_embed 的页面形态（参数表单+试跑）；
    查询/检索 UI 本期不做（查询层另行开单）。"""
    video_name = str(p.get("video", "")).strip()
    if not video_name:
        return res(ok=False, error="请选择要索引的视频")
    vid = _find_file(video_name, [VID_DIR])
    if not vid:
        return res(ok=False, error=f"找不到视频: {video_name}")
    stage = _build_video_embed_stage(p, video=vid.name)
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        cap.release()
        return res(ok=False, error=f"无法打开视频: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # 取样节奏：frames_per_clip/clip_seconds 帧每秒（窗口标称时间轴；等价帧管理 frame_skip）
    stride = max(1, int(round(fps * stage.clip_seconds / stage.frames_per_clip)))
    recs, fed, n = [], 0, 0
    while fed < _VEMB_TEST_MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        recs.extend(stage.run(frame, ts=n / fps, frame_idx=n))
        fed += 1
        n += 1
        for _ in range(stride - 1):  # 跳帧只解码不取图
            if not cap.grab():
                break
            n += 1
    cap.release()
    recs.extend(stage.finish())
    store_path = _str_param(p, "store_path") or "out/video_vecs.npz"
    sp = Path(store_path)
    if not sp.is_absolute():
        sp = ROOT / store_path
    saved = stage.store.save(str(sp))
    mode = "mock（确定性伪向量）" if stage.client.mock else \
        f"真实端点 {stage.client._active_endpoint or '(未配置)'}"
    lines = [f"视频: {vid.name} | 模式: {mode} | clip 维 {stage.client.clip_dimension} / "
             f"frame 维 {stage.client.frame_dimension} | 取样 stride={stride} 喂 {fed} 帧"
             f"（上限 {_VEMB_TEST_MAX_FRAMES}） | 片段 {len(recs)} 条 | 底库: {saved} | "
             f"{stage.store.stats()}"]
    for r in recs[:10]:
        lines.append(f"  片段#{r['clip_id']} t={r['t_start']:.2f}~{r['t_end']:.2f}s "
                     f"{r['frame_count']} 帧{'（尾段）' if r['partial'] else ''} "
                     f"→ {r['thumb_path'] or '无缩略图'}")
    if len(recs) > 10:
        lines.append(f"  …（其余 {len(recs) - 10} 条见底库）")
    images = []
    for r in recs[:3]:
        tp = r.get("thumb_path")
        if tp and Path(tp).exists():
            img = cv2.imread(tp)
            if img is not None:
                images.append({"title": f"片段#{r['clip_id']} 缩略图 t={r['t_start']:.1f}s",
                               "data": _img_data_url(img, 80)})
    return res(text="\n".join(lines), images=images)


# ---------------- 回顾检索·查询层（WO-0032 向量档 + WO-0033 混合检索·过滤式） ----------------
# 设计依据 doc/design/video_review_search.md §4.1（两步漏斗）/ §6（前端交互）与
# doc/design/hybrid_retrieval.md（§2.5 标签 schema / §3 事件归档 / §4 落表 / §5 查询扩展）：
#   query_text → WeMM embed_text（维度取底库 clip_dim，入库查询同维度）→
#   ① store.search clip 粗排（有过滤器时先 pre-filter 幸存集）→
#   ② 逐片段 search_frames 帧级精排 → play_ts = best_frame_ts − pre_roll_sec（下限 0）。
# 过滤式本期落地（时间窗 / 事件名+车牌 / 视频），hybrid（RRF）只预留 mode 字段。

# ---- 事件留存归档（§3 采集路径 A / §9 Q1 / §12-C）----
# 分析链路收尾把**链路尾事件**追加到 out/events/<视频名>.jsonl，一行一次运行；
# run_id 去重保证幂等。detail 摘要控体积：单字符串 200 字符、列表 10 项、对象 20 键，
# 序列化后超过 800 字符退化为 {"_digest": "…"}（阈值见常量，长 VLM 文本不落全量）。
_EVENTS_DIR = ROOT / "out" / "events"
_EVENT_DETAIL_STR_MAX = 200
_EVENT_DETAIL_ITEMS_MAX = 10
_EVENT_DETAIL_KEYS_MAX = 20
_EVENT_DETAIL_JSON_MAX = 800
_EVENTS_LOCK = threading.Lock()


def _event_detail_digest(detail):
    """事件 detail → 控体积摘要（截断阈值见上方常量）。"""
    if detail is None:
        return {}
    if not isinstance(detail, dict):
        detail = {"value": detail}

    def _cut(v, depth=0):
        if isinstance(v, str):
            return v if len(v) <= _EVENT_DETAIL_STR_MAX else \
                v[:_EVENT_DETAIL_STR_MAX] + "…（截断）"
        if isinstance(v, (list, tuple)):
            items = [_cut(x, depth + 1) for x in list(v)[:_EVENT_DETAIL_ITEMS_MAX]]
            if len(v) > _EVENT_DETAIL_ITEMS_MAX:
                items.append(f"…（共 {len(v)} 项）")
            return items
        if isinstance(v, dict):
            keys = list(v)[:_EVENT_DETAIL_KEYS_MAX]
            out = {str(k): _cut(v[k], depth + 1) for k in keys}
            if len(v) > _EVENT_DETAIL_KEYS_MAX:
                out["_truncated_keys"] = len(v) - _EVENT_DETAIL_KEYS_MAX
            return out
        return v

    d = _cut(detail)
    try:
        text = json.dumps(d, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(d)
    if len(text) > _EVENT_DETAIL_JSON_MAX:
        return {"_digest": text[:_EVENT_DETAIL_JSON_MAX] + "…（截断）"}
    return d


def _event_record(ev: dict, kind: str) -> dict:
    """链路尾事件 → 归档条目（§12-C：name/kind/ts/detail；车牌事件附 plate_no）。

    plate_no 为契约 C 的兼容扩展字段：pipe 侧 build_clip_tags 按 plate_no 把
    车牌事件路由进 plates[]（§2.5 三层分工：车牌号不重复进 events）。
    """
    name = str(ev.get("name") or "").strip()
    rec = {"name": name, "kind": str(ev.get("kind") or kind or "警情")}
    ts = ev.get("ts")
    if ts is not None:
        try:
            rec["ts"] = round(float(ts), 3)
        except (TypeError, ValueError):
            pass
    rec["detail"] = _event_detail_digest(ev.get("detail"))
    pn = ev.get("plate_no")
    if kind == "车牌" and pn in (None, ""):
        pn = name  # PIPE0002 车牌事件 name=车牌号（§2.2）
    if pn not in (None, ""):
        rec["plate_no"] = str(pn)
    return rec


def _event_record_key(rec: dict) -> tuple:
    return (rec.get("name", ""), rec.get("kind", ""), rec.get("ts"),
            json.dumps(rec.get("detail") or {}, ensure_ascii=False, sort_keys=True))


def _archive_add_many(acc: list, events, kind: str = "警情", plate_nos=None) -> None:
    """把一批事件追加进本次运行的归档累加器（同运行内去重）。

    车牌分类（name 命中 plate_nos）**覆写**事件尾的泛化 kind（警情）——
    §2.5 三层分工：车牌号归 plates[]，不与警情混口径。
    """
    for ev in events or []:
        if not isinstance(ev, dict) or not str(ev.get("name") or "").strip():
            continue
        k = kind
        if plate_nos is not None and str(ev.get("name")) in plate_nos:
            k = "车牌"
        ev2 = dict(ev)
        if k == "车牌":
            ev2["kind"] = "车牌"  # 覆写 seq_filter 的泛化 kind（警情）
        rec = _event_record(ev2, k)
        key = _event_record_key(rec)
        if not any(_event_record_key(x) == key for x in acc):
            acc.append(rec)


def _safe_events_filename(video: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(video or "video")).strip("_") or "video"


def _events_path_for(video: str) -> Path:
    return _EVENTS_DIR / f"{_safe_events_filename(video)}.jsonl"


def _archive_run_events(video: str, pipeline_id: str, events: list,
                        run_id: str) -> int:
    """把一次运行的链路尾事件追加到 out/events/<视频名>.jsonl（按 run_id 幂等）。

    返回本次实际写入的事件数（0 = 已归档过 / 无事件 / 无视频名）。
    """
    video = str(video or "").strip()
    if not video or not events:
        return 0
    path = _events_path_for(video)
    with _EVENTS_LOCK:
        seen = set()
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seen.add(str(json.loads(line).get("run_id") or ""))
                    except ValueError:
                        continue
            except OSError:
                pass
        if run_id and run_id in seen:
            return 0
        # 归档前统一规范化（契约 C + detail 摘要控体积 + 车牌事件附 plate_no）：
        # 生产路径 _tail_events_of_timeline 已规范化（此处幂等重入）；直调路径
        # （qa mock / 工具脚本）在此兜底，保证落盘格式一致。
        norm: list = []
        _pn = {str(e.get("name")) for e in events
               if isinstance(e, dict) and (e.get("plate_no") or e.get("kind") == "车牌")}
        _archive_add_many(norm, events, kind="警情", plate_nos=_pn or None)
        payload = {
            "run_id": str(run_id or uuid.uuid4().hex[:12]),
            "pipeline": str(pipeline_id or ""),
            "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "video": video,
            "events": norm,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[events] 归档失败（{path}）: {e}")
            return 0
    return len(norm)


def _read_archived_events(video: str, events_path=None) -> list[dict]:
    """读取某视频的事件归档（所有运行合并，按 run_id+事件去重）。"""
    path = Path(events_path) if events_path else _events_path_for(video)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return []
    out, seen = [], set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        run_id = str(row.get("run_id") or "")
        for ev in row.get("events") or []:
            if not isinstance(ev, dict):
                continue
            rec = _event_record(ev, str(ev.get("kind") or "警情"))
            key = (run_id,) + _event_record_key(rec)
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
    return out


def _tail_events_of_timeline(timeline: list) -> list:
    """从运行时间线提取**链路尾事件**（§9 Q1：只归档链路尾事件）。

    口径：kind=submit 且 data 为含非空 name 的事件 dict——seq_filter 事件尾产出
    （警情名/车牌号/VLM 研判结论，含 publish 通路事件）。两类 submit 不算：
      - plate_recog 提交（data 形态为 plate_no/type_name/…，无 name）；
      - video_embed 入库回执（事件名 clip_indexed，索引产物非分析事件）。
    同运行内重复事件经 _archive_add_many 去重；车牌号命中 plate 提交时 kind 标「车牌」
    （PIPE0002 车牌事件 name=车牌号，§2.2）。
    """
    evs: list = []
    plate_nos: set = set()
    for it in timeline or []:
        if not isinstance(it, dict) or it.get("kind") != "submit":
            continue
        d = it.get("data")
        if not isinstance(d, dict):
            continue
        pn = d.get("plate_no")
        if pn not in (None, ""):
            plate_nos.add(str(pn))
            continue  # 车牌识别提交不是事件
        name = str(d.get("name") or "").strip()
        if not name or name == "clip_indexed":
            continue
        evs.append(d)
    acc: list = []
    _archive_add_many(acc, evs, kind="警情", plate_nos=plate_nos or None)
    return acc


# ---- 索引收尾对齐（§3 / §4 / §12-A）----

def _project_rel(path) -> str:
    """磁盘路径 → 项目相对路径（§2.5 原则 4：换机不失效）。"""
    s = str(path or "").strip()
    if not s:
        return ""
    p = Path(s)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(ROOT))
        except (ValueError, OSError):
            return str(p)
    if s.startswith("../"):  # 历史 CWD 相对形态（web_lab 基准）→ 归一项目根基准
        try:
            rp = (ROOT / s).resolve()
            return str(rp.relative_to(ROOT))
        except (ValueError, OSError):
            pass
    return s


def _video_duration_of(clip_records: list) -> float:
    ends = []
    for r in clip_records or []:
        try:
            ends.append(float(r.get("t_end") or 0.0))
        except (TypeError, ValueError):
            pass
    return round(max(ends), 3) if ends else 0.0


def _window_key(win) -> tuple:
    if isinstance(win, (tuple, list)) and len(win) >= 2:
        return (round(float(win[0]), 3), round(float(win[1]), 3))
    if isinstance(win, dict):
        return (round(float(win.get("t_start") or 0.0), 3),
                round(float(win.get("t_end") or 0.0), 3))
    return (0.0, 0.0)


def _build_review_rows(video: str, windows: list, observations: dict,
                       events: list, extra: dict | None = None) -> list[dict]:
    """调用 pipe 侧 label_align（契约 A）产 §2.5 标签行；模块缺失时返回 []。"""
    try:
        from video_embed import label_align as la
    except Exception as e:  # pipe 侧未落地/导入失败：不阻断索引，记录提示
        print(f"[review] label_align 不可用，跳过标签落表: {e}")
        return []
    aligned = []
    if events:
        try:
            aligned = la.align_events(events, windows) or []
        except Exception as e:
            print(f"[review] align_events 失败: {e}")
            aligned = []
    grouped: dict = {}
    for item in aligned:
        if not isinstance(item, dict):
            continue
        key = None
        for k in ("window", "t_start", "clip_window"):
            if k in item:
                if k == "t_start":
                    key = (round(float(item.get("t_start") or 0.0), 3),
                           round(float(item.get("t_end") or 0.0), 3))
                else:
                    key = _window_key(item.get(k))
                break
        if key is None and item.get("ts") is not None:
            ts = float(item["ts"])
            key = next((w for w in windows if w[0] <= ts < w[1]), None)
        if key is not None:
            grouped.setdefault(key, []).append(item)

    rows = []
    for win in windows:
        key = _window_key(win)
        obs = observations.get(key) or {}
        win_events = grouped.get(key) or []
        try:
            tags = la.build_clip_tags(video, key, obs, win_events) or {}
        except Exception as e:
            print(f"[review] build_clip_tags 失败（{video} {key}）: {e}")
            tags = {}
        # frame_dets（§2.5 观察层原始记录流，§11 定案 6）：pipe 侧 build_clip_tags 透传
        # （契约 A）；pipe 未交付/字段缺失时从 observation 旁路键兜底取（aggregate_frame_stats
        # 产物同含 frame_dets），再缺省 []——观察源未接线不报错，行照常落表。
        fd = tags.get("frame_dets") or obs.get("frame_dets") or []
        row = {
            "video": video, "t_start": key[0], "t_end": key[1],
            "det_stats": tags.get("det_stats") or obs.get("det_stats") or {},
            "frame_dets": fd if isinstance(fd, list) else [],
            "class_names": tags.get("class_names") or obs.get("class_names") or {},
            "plate_nos": tags.get("plate_nos") or obs.get("plate_nos") or [],
            "plates": tags.get("plates") or [],
            "events": tags.get("events") or [],
            "face_count": int(tags.get("face_count") or obs.get("face_count") or 0),
            "description": str(tags.get("description") or ""),
            "thumb_path": _project_rel(tags.get("thumb_path") or ""),
            "vector_model": str(tags.get("vector_model") or (extra or {}).get("vector_model") or ""),
            "clip_dim": tags.get("clip_dim") if tags.get("clip_dim") is not None
            else (extra or {}).get("clip_dim"),
            "frame_dim": tags.get("frame_dim") if tags.get("frame_dim") is not None
            else (extra or {}).get("frame_dim"),
        }
        rows.append(row)
    return rows


def _align_and_store(video: str, clip_records: list, events: list,
                     vector_model: str = "", clip_dim=None, frame_dim=None) -> dict:
    """PIPE0004 收尾对齐落表（review_index + review_video）；返回处理摘要。

    仅落表，不动向量库（向量落盘由调用方完成）。label_align 尚不可用时 fail-soft。
    """
    try:
        from video_embed import label_align as la
    except Exception as e:
        return {"ok": False, "rows": 0, "error": f"label_align 不可用: {e}"}
    windows, observations = [], {}
    frame_ts_all: list = []
    for r in clip_records or []:
        if not isinstance(r, dict):
            continue
        try:
            win = (round(float(r.get("t_start") or 0.0), 3),
                   round(float(r.get("t_end") or 0.0), 3))
        except (TypeError, ValueError):
            continue
        windows.append(win)
        obs = r.get("observation") or {}
        if not obs and isinstance(r.get("frame_stats"), list):
            try:
                obs = la.aggregate_frame_stats(r["frame_stats"]) or {}
            except Exception:
                obs = {}
        observations[win] = obs if isinstance(obs, dict) else {}
        frame_ts_all.extend(r.get("frame_ts") or [])
    if not windows and frame_ts_all:
        try:
            windows = [_window_key(w) for w in la.clip_windows(frame_ts_all)]
        except Exception as e:
            print(f"[review] clip_windows 失败: {e}")
    if not windows:
        return {"ok": True, "rows": 0, "events": len(events),
                "note": "无可对齐的片段窗口"}

    extra = {"vector_model": vector_model, "clip_dim": clip_dim, "frame_dim": frame_dim}
    rows = _build_review_rows(video, windows, observations, events, extra=extra)
    if not rows:
        return {"ok": False, "rows": 0, "error": "标签构建失败（build_clip_tags 无产出）"}
    # 缩略图/维度兜底：以片段记录（EmbedStore meta 同源）为准
    by_win = {_window_key(w): r for w, r in zip(windows, rows)}
    for rec in clip_records or []:
        if not isinstance(rec, dict):
            continue
        key = _window_key((rec.get("t_start"), rec.get("t_end")))
        row = by_win.get(key)
        if row is None:
            continue
        row["thumb_path"] = _project_rel(rec.get("thumb_path") or row["thumb_path"])
        for k in ("clip_dim", "frame_dim"):
            if rec.get(k) is not None:
                row[k] = rec.get(k)
    n = db.upsert_review_index(rows)

    try:
        video_time, src = la.parse_video_time(video)
    except Exception:
        video_time, src = "", ""
    db.upsert_review_video([{
        "video": video,
        "video_time": video_time, "video_time_src": src,
        "indexed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "vector_model": vector_model,
        "clip_dim": clip_dim, "frame_dim": frame_dim,
        "duration": _video_duration_of(clip_records),
    }])
    return {"ok": True, "rows": n, "events": len(events),
            "backend": db.review_backend_mode()}


def _vem_vector_meta(vem, params: dict) -> tuple:
    """视频嵌入阶段 → (vector_model, clip_dim, frame_dim)。"""
    client = getattr(vem, "client", None)
    model = str(getattr(client, "model", "") or params.get("embed_model") or "")
    cd = getattr(client, "clip_dimension", None) or params.get("embed_dimension")
    fd = getattr(client, "frame_dimension", None) or params.get("frame_dimension")
    return model, cd, fd


def _vem_feed_dets(pipe, vem, position, frame, ts, n):
    """向 video_embed 喂入本帧 + 观察源 Records（dets 第二入端口，§12-1）。

    pipe-chain 的 VideoEmbedStage 契约未固定 run 的附加参数形态，这里做兼容适配：
    优先 `run(frame, ts, n, records=...)`；否则若提供 `observe(...)` / `feed_dets(...)`
    先喂观察记录再按原签名 run；均不支持时退化为纯帧索引入口（不报错、无观察标签）。
    """
    records = None
    try:
        records = pipe._port_at(position, "video_embed", vem, "dets", None)
    except Exception:
        records = None
    if records is not None and not isinstance(records, list):
        records = [records] if isinstance(records, dict) else None
    run = getattr(vem, "run", None)
    try:
        sig = inspect.signature(run)
    except (TypeError, ValueError):
        sig = None
    if records and sig is not None and "records" in sig.parameters:
        return run(frame=frame, ts=ts, frame_idx=n, records=records)
    if records and sig is not None and "dets" in sig.parameters:
        return run(frame=frame, ts=ts, frame_idx=n, dets=records)
    if records:
        for name in ("observe", "feed_dets", "add_dets"):
            fn = getattr(vem, name, None)
            if fn is None:
                continue
            try:
                fn(records, ts=ts, frame_idx=n)
            except TypeError:
                try:
                    fn(records, ts, n)
                except TypeError:
                    fn(records)
            break
    return run(frame, ts, n)


def _last_clip_records(vem) -> list:
    """取视频嵌入阶段累计的片段记录（indexed，驱动层可直接读全量）。"""
    recs = getattr(vem, "indexed", None)
    return list(recs) if isinstance(recs, list) else []


def _archive_and_align(video, pipeline_id, events, run_id, vem, params,
                       store_path: str = ""):
    """收尾：事件归档 → （有索引时）标签对齐落表。任一步失败不影响运行主流程。"""
    saved = 0
    try:
        saved = _archive_run_events(video, pipeline_id, events, run_id)
    except Exception as e:
        print(f"[events] 归档异常: {e}")
    if vem is None:
        return {"events": saved}
    try:
        model, cd, fd = _vem_vector_meta(vem, params or {})
        res = _align_and_store(video, _last_clip_records(vem),
                               _read_archived_events(video),
                               vector_model=model, clip_dim=cd, frame_dim=fd)
        res["events_saved"] = saved
        print(f"[review] 标签落表 {video}: {res}")
        return res
    except Exception as e:
        print(f"[review] 标签对齐失败（不阻断索引）: {e}")
        return {"events": saved, "error": str(e)}


_REVIEW_STORE_CACHE: dict = {}   # 库绝对路径 -> (mtime_ns, EmbedStore)；mtime 变化自动重载
_REVIEW_STORE_LOCK = threading.Lock()


def _review_store_path(store_path) -> Path:
    """store_path 归一：相对路径以项目根为基准。

    server 进程 CWD 常为 web_lab/（run.sh），不归一会找成 web_lab/out/…（INB-0114
    同款 CWD 坑）；前端传回的绝对路径原样保留。
    """
    sp = Path(str(store_path or "").strip() or "out/video_vecs.npz")
    return sp if sp.is_absolute() else ROOT / sp


def _review_store(store_path):
    """加载向量底库（缓存复用，文件 mtime 变化自动重载——PIPE0004 重跑 upsert 后
    查询侧无需重启服务）。不存在 / 为空抛可读异常（FileNotFoundError / ValueError）。"""
    from video_embed.embed_store import EmbedStore
    sp = _review_store_path(store_path)
    if not sp.is_file():
        raise FileNotFoundError(
            f"向量底库不存在: {sp}"
            f"（请先运行 PIPE0004「视频向量化-嵌入查询链路」索引视频建库）")
    mtime = sp.stat().st_mtime_ns
    key = str(sp)
    with _REVIEW_STORE_LOCK:
        hit = _REVIEW_STORE_CACHE.get(key)
        if not hit or hit[0] != mtime:
            hit = (mtime, EmbedStore().load(key))
            _REVIEW_STORE_CACHE[key] = hit
    if len(hit[1]) == 0:
        raise ValueError(f"向量底库为空: {sp}（请先运行 PIPE0004 建库后再查询）")
    return hit[1], key


def _review_parse_dt(v):
    """时间窗参数 → 带时区 datetime（None/空 → None；非法 → ValueError）。"""
    if v is None or v == "":
        return None
    if isinstance(v, datetime.datetime):
        return v if v.tzinfo else v.astimezone()
    if isinstance(v, (int, float)):
        return datetime.datetime.fromtimestamp(float(v)).astimezone()
    s = str(v).strip()
    if not s:
        return None
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"时间窗参数非法: {v!r}（应为 ISO8601，如 2026-10-29T10:00:00+08:00）")
    return dt if dt.tzinfo else dt.astimezone()


def _review_video_dt(vrow: dict | None):
    """review_video 行 → 采集时间 datetime（空/非法 → None）。"""
    if not vrow:
        return None
    s = str(vrow.get("video_time") or "").strip()
    if not s:
        return None
    try:
        return _review_parse_dt(s)
    except ValueError:
        return None


def h_review_search(body: dict) -> dict:
    """文搜视频检索 handler（POST /api/review/search 的进程内入口，单测直调）。

    主链路（无过滤器）与 WO-0032 逐字一致：向量粗排 + 帧精排 + pre_roll 定位。
    WO-0033 增量：`time_start/time_end`（墙钟，按 review_video.video_time + t_start）、
    `event_name`（精确匹配 events.name 与 plates.plate_no）、`video`、`mode`
    （vector/hybrid——**本期两者都走过滤式**，hybrid 未实现在 note/why 说明）、
    `include_frame_dets`（缺省 false；true 时 hits[].tags 附带观察层原始记录流
    frame_dets——体积大，仅时间形态可视化/调试时开）。
    `why` 逐条给命中来源；无过滤时保持旧文案「向量相似度 0.xx」向后兼容。

    use_real_embed 语义对齐 WEMMClient：缺省/None = 按端点有无自动（env 无端点自动
    落 mock，响应 note 说明）；显式 true 且无端点 → ValueError fail fast（HTTP 层
    映射 400，与装配期 WEMMClient 行为双保险）；显式 false = mock。
    错误约定：空查询/参数非法/维度非档位/空库 → ValueError（HTTP 400）；
    底库不存在 → FileNotFoundError（HTTP 404）；其余异常由路由层兜底。
    """
    from video_embed.wemm_client import WEMMClient
    query_text = str(body.get("query_text") or "").strip()
    if not query_text:
        raise ValueError("查询文本为空：query_text 必填（如「人群聚集」）")
    top_k_clips = _int(body, "top_k_clips", 10) or 10
    top_k_frames = _int(body, "top_k_frames", 3) or 3
    pre_roll = _float(body, "pre_roll_sec", 1.0)
    if top_k_clips < 1 or top_k_frames < 1:
        raise ValueError(f"top_k 非法：top_k_clips={top_k_clips} top_k_frames="
                         f"{top_k_frames}（均应 ≥1）")
    if pre_roll < 0:
        raise ValueError(f"pre_roll_sec={pre_roll} 非法（应 ≥0）")
    _ure = body.get("use_real_embed")
    use_real = None if _ure in (None, "") else _bool(body, "use_real_embed")
    mode = str(body.get("mode") or "vector").strip().lower() or "vector"
    if mode not in ("vector", "hybrid"):
        raise ValueError(f"mode 非法: {mode!r}（应为 vector / hybrid）")
    t_start_req = _review_parse_dt(body.get("time_start"))
    t_end_req = _review_parse_dt(body.get("time_end"))
    if t_start_req and t_end_req and t_start_req > t_end_req:
        raise ValueError("时间窗非法：time_start 晚于 time_end")
    event_name = str(body.get("event_name") or "").strip()
    video = str(body.get("video") or "").strip()
    # frame_dets（§2.5 观察层原始记录流）默认**不随 hits[].tags 返回**——单窗可达
    # 数十 KB（KB 级行），全量拖回浪费带宽；include_frame_dets=true 时才附带，
    # 供前端时间形态可视化（逐帧检出分布/框重放）与调试用。
    include_frame_dets = _bool(body, "include_frame_dets")
    # filters_active 决定 why 是否逐条说明命中来源：时间窗/事件名之外，video 过滤与
    # mode=hybrid 也纳入（hybrid 本期走过滤式，why 须注明；video-only 命中须注明
    # 视频与有无标签）。纯 vector 无过滤查询保持 WO-0032 文案（向后兼容）。
    filters_active = bool(t_start_req or t_end_req or event_name or video
                          or mode == "hybrid")

    store, store_key = _review_store(body.get("store_path"))
    clip_dim = int(store.clip_dim or 0)
    frame_dim = int(store.frame_dim or clip_dim)
    # 维度非档位（如手工造库 100 维）→ WEMMClient fail fast（ValueError 含合法档位）
    client = WEMMClient(clip_dimension=clip_dim, frame_dimension=frame_dim,
                        use_real_embed=use_real)
    note = ""
    if client.mock:
        note = ("已按 mock 档（确定性伪向量）查询（use_real_embed=false 显式指定）"
                if use_real is False else
                "未配置嵌入端点（env WEMM_BASE_URL / WEMM_ENDPOINT），本次为 mock 档"
                "确定性伪向量查询——命中仅供流程演示；配置真实端点并重建索引后才有语义检索效果")
    if mode == "hybrid":
        note = (note + "；" if note else "") + \
            "hybrid 融合（RRF）本期未实现，已按过滤式执行（mode 字段预留）"

    # 两步漏斗各用各维度的查询向量（WeMM 统一向量空间，MRL 同档可点积）：
    # ① clip 粗排 = embed_text@clip_dim；② 帧精排 = embed_text@frame_dim 与
    #   逐帧向量（入库时按 frame_dimension 嵌出）点积——维度必须两两对齐，
    #   不能拿 clip 维查询向量比 frame 维库向量（matmul 直接维度不匹配）。
    q_clip = client.embed_text(query_text, dimension=clip_dim)
    q_frame = q_clip if frame_dim == clip_dim else \
        client.embed_text(query_text, dimension=frame_dim)

    # 有过滤器时先取全量（漏斗纪律：pre-filter 在排序之前，避免 Top-K 被过滤掏空）
    search_k = max(len(store), top_k_clips) if filters_active else top_k_clips
    raw_hits = list(store.search(q_clip, top_k=search_k, video=video or None))

    tag_map: dict = {}
    video_meta: dict = {}
    backend = "none"
    if filters_active:
        rows = db.query_review_index(video=video or None, time_start=t_start_req,
                                     time_end=t_end_req, event_name=event_name or None)
        tag_map = {(r["video"], round(float(r["t_start"]), 3)): r for r in rows}
        video_meta = {v["video"]: v for v in db.list_review_videos(video or None)}
        backend = db.review_backend_mode()
        if backend == "json":
            note = (note + "；" if note else "") + \
                "标签索引走无 DB 降级（out/review_index.json / review_video.json）"

    hits = []
    for h in raw_hits:
        meta = h.meta or {}
        vname = str(meta.get("video") or "")
        t_start = float(meta.get("t_start") or 0.0)
        tag = tag_map.get((vname, round(t_start, 3))) if filters_active else None
        time_unknown = False
        if filters_active and (t_start_req or t_end_req):
            vt = _review_video_dt(video_meta.get(vname))
            if vt is None:
                time_unknown = True          # §2.6：无采集时间不参与过滤，why 注明
            else:
                abs_ts = vt + datetime.timedelta(seconds=t_start)
                if ((t_start_req is not None and abs_ts < t_start_req)
                        or (t_end_req is not None and abs_ts > t_end_req)):
                    continue
        if event_name:
            # 精确匹配 events.name 与 plates.plate_no（§9 Q2）；无标签片段不参与该过滤
            if tag is None:
                continue
        if filters_active and len(hits) >= top_k_clips:
            break

        frames = [{"frame_index": int(f.index), "ts": round(float(f.ts), 3),
                   "score": round(float(f.score), 4)}
                  for f in store.search_frames(q_frame, h.clip_id, top_k=top_k_frames)]
        best_ts = frames[0]["ts"] if frames else None
        # 播放定位（§4.1）：从最高相似帧回退 pre_roll 秒起播，避免从窗口头看起
        play_ts = max(0.0, best_ts - pre_roll) if best_ts is not None else t_start
        thumb_path = _project_rel(meta.get("thumb_path") or "")
        thumb = ""
        if thumb_path:
            tp = Path(thumb_path)
            if not tp.is_absolute():
                tp = ROOT / tp
            img = cv2.imread(str(tp)) if tp.is_file() else None
            if img is not None:
                thumb = _thumb_data_url(img)  # 浏览器取不到磁盘路径，随响应附 data URL
        if filters_active:
            why = _review_why(tag, video=vname if video else "", t_start=t_start,
                              time_filter=bool(t_start_req or t_end_req),
                              time_unknown=time_unknown, event_name=event_name,
                              mode=mode, score=float(h.score))
        else:
            # 无过滤器：保持 WO-0032 文案逐字一致（向后兼容）
            why = f"向量相似度 {h.score:.2f}"
        hit = {
            "video": vname,
            "t_start": round(t_start, 3),
            "t_end": round(float(meta.get("t_end") or t_start), 3),
            "score": round(float(h.score), 4),
            "clip_id": int(h.clip_id),
            "frames": frames,
            "best_frame_ts": best_ts,
            "play_ts": round(float(play_ts), 3),
            "thumb_path": thumb_path,
            "thumb": thumb,
            "why": why,
        }
        if tag is not None:
            # 标签随命中返回（前端展示/调试；不进原有字段语义）。
            # frame_dets 体积大（观察层原始记录流），默认不带；include_frame_dets=true
            # 时才附带（时间形态可视化/调试）。
            tag_keys = ("det_stats", "class_names", "plate_nos", "plates",
                        "events", "face_count", "description")
            if include_frame_dets:
                tag_keys += ("frame_dets",)
            hit["tags"] = {k: tag.get(k) for k in tag_keys}
        hits.append(hit)

    return {"ok": True, "mode": "mock" if client.mock else "real", "note": note,
            "query_mode": mode, "query_dim": clip_dim, "query_text": query_text,
            "store": {"path": store_key, "clips": len(store), "clip_dim": clip_dim},
            "signals": {"filters": {"time_start": str(body.get("time_start") or ""),
                                    "time_end": str(body.get("time_end") or ""),
                                    "event_name": event_name, "video": video,
                                    "mode": mode,
                                    "include_frame_dets": include_frame_dets},
                        "active": filters_active,
                        "meta_hits": len(tag_map),
                        "vec_top_n": len(raw_hits),
                        "tag_backend": backend},
            "hits": hits}


def _review_why(tag: dict | None, *, video: str, t_start: float,
                time_filter: bool, time_unknown: bool, event_name: str,
                mode: str, score: float) -> str:
    """过滤式 why：逐条说明命中来源（无标签/无采集时间也注明，不静默）。"""
    bits = []
    if video:
        bits.append(f"视频 {video} 命中")
    if time_filter:
        bits.append("该视频无采集时间，未参与时间窗过滤" if time_unknown
                    else "时间窗命中")
    if event_name:
        names = {str(e.get("name", "")) for e in (tag or {}).get("events") or []
                 if isinstance(e, dict)}
        plates = {str(p.get("plate_no", "")) for p in (tag or {}).get("plates") or []
                  if isinstance(p, dict)}
        if event_name in plates:
            bits.append(f"车牌 {event_name} 命中")
        elif event_name in names:
            bits.append(f"事件 {event_name} 命中")
        else:
            bits.append(f"未命中事件/车牌 {event_name}")
    if tag is None:
        bits.append("该片段无标签（未参与标签过滤）")
    if mode == "hybrid":
        bits.append("hybrid 融合未实现，按过滤式执行")
    bits.append(f"向量相似度 {score:.2f}")
    return "；".join(bits)



def _override_obs_get(override_obs: dict, t_start: float, t_end: float) -> dict:
    """relabel observations 覆盖参数的宽松键匹配。

    HTTP body JSON 键恒为字符串（"0.0" / "0.0,5.0"），进程内直调可用浮点键
    （0.0 / (0.0, 5.0)）；按窗口元组 → 组合字符串 → 起点 单独键的顺序逐一尝试。
    """
    if not isinstance(override_obs, dict):
        return {}
    for k in ((t_start, t_end), f"{t_start},{t_end}", t_start, f"{t_start}"):
        v = override_obs.get(k)
        if isinstance(v, dict):
            return v
    return {}


def h_review_relabel(body: dict) -> dict:
    """「仅重对齐不重嵌入」入口（POST /api/review/relabel，§3 定案点）。

    参数：video（必填）/ events_path（可选，缺省 out/events/<视频名>.jsonl）/
    windows（可选 [[t_start,t_end], ...]，缺省从向量底库该视频片段窗口取）/
    store_path（可选，缺省 out/video_vecs.npz）/
    observations（可选 {窗口键: obs}，显式新观察数据——本期无调用入口，仅 frame_dets
    会在传入时覆盖旧行；窗口键 "t_start,t_end" 或 "t_start" 均可）。
    **只 upsert review_index / review_video，绝不动向量库**（不重嵌入）；复用索引收尾
    同一套 label_align 对齐函数。返回处理行数。
    """
    video = str(body.get("video") or "").strip()
    if not video:
        raise ValueError("video 必填（要重对齐的视频名）")
    events = _read_archived_events(video, body.get("events_path"))
    existing = {round(float(r["t_start"]), 3): r
                for r in db.query_review_index(video=video)}
    windows = []
    for w in body.get("windows") or []:
        if isinstance(w, (list, tuple)) and len(w) >= 2:
            windows.append((round(float(w[0]), 3), round(float(w[1]), 3)))
        elif isinstance(w, dict):
            windows.append((round(float(w.get("t_start") or 0.0), 3),
                            round(float(w.get("t_end") or 0.0), 3)))
    store_meta: dict = {}
    vector_model, clip_dim, frame_dim = "", None, None
    if not windows:
        store, _key = _review_store(body.get("store_path"))
        clip_dim, frame_dim = store.clip_dim, store.frame_dim
        client = getattr(store, "client", None)
        try:
            q = np.ones(int(store.clip_dim or 1), dtype=np.float32)
            _fts_nested = []  # (t_start, [帧ts...])：重建无缝窗口用（对齐索引时 clip_windows 口径）
            for h in store.search(q, top_k=len(store), video=video):
                m = h.meta or {}
                w = (round(float(m.get("t_start") or 0.0), 3),
                     round(float(m.get("t_end") or 0.0), 3))
                windows.append(w)
                store_meta[w[0]] = m  # 键=t_start：无缝窗口 t_end≠store t_end，避免查不到兜底
                _fts_nested.append((w[0], [float(t) for t in (m.get("frame_ts") or [])]))
            # 窗口口径统一（坑：store meta t_end=窗内末帧 ts，直接当对齐窗口会在
            # 「末帧 ts ~ 下一窗首帧」的窗缝里丢事件——如 ts=2.502 落在 (0,2.502) 外、
            # (3.003,…) 前。索引时对齐走 clip_windows（t_end=下一窗首帧，无缝），
            # relabel 必须同口径：从各片段 frame_ts 重建；frame_ts 缺失（旧库）才退 store t_end。
            _fts_nested.sort(key=lambda x: x[0])
            try:
                from video_embed import label_align as _la
                _seamless = _la.clip_windows([f for _, f in _fts_nested if f])
                if _seamless:
                    windows = _seamless
            except Exception:
                pass  # label_align 不可用：退回 store t_end 口径（缝上事件可能丢，不阻断）
        except Exception as e:
            raise ValueError(f"从向量底库取片段窗口失败: {e}")
        if not windows:
            raise ValueError(f"向量底库无视频 {video!r} 的片段——请先运行 PIPE0004 建索引，"
                             f"或显式传 windows 参数")
    else:
        # 只给 windows 时仍尽力从底库/旧标签取维度与缩略图（缺省不报错）
        try:
            store, _key = _review_store(body.get("store_path"))
            clip_dim, frame_dim = store.clip_dim, store.frame_dim
        except Exception:
            pass
    if not windows:
        raise ValueError("没有可对齐的片段窗口（windows 为空且底库无该视频）")

    # relabel 观察层重建：只回填聚合缓存（det_stats/class_names/plate_nos/face_count 旧值
    # 保留语义）。**frame_dets 刻意不进这里**——它属观察层原始数据（分析产物），relabel
    # 只重算判断层与聚合缓存，旧行 frame_dets 在下方落表前原样保留；仅 body.observations
    # 显式传入新观察数据（window → obs，本期无调用入口，留参数位）时才覆盖。
    override_obs: dict = body.get("observations") or {}
    observations = {}
    for w in windows:
        old = existing.get(w[0]) or {}
        ovr = _override_obs_get(override_obs, w[0], w[1])
        observations[w] = {
            "det_stats": old.get("det_stats") or {},
            "class_names": old.get("class_names") or {},
            "plate_nos": old.get("plate_nos") or [],
            "face_count": old.get("face_count") or 0,
            # 显式新观察才带 frame_dets；缺省不带（避免被误当成新原始数据）
            "frame_dets": ovr.get("frame_dets") or [],
        }
    extra = {"vector_model": vector_model, "clip_dim": clip_dim, "frame_dim": frame_dim}
    rows = _build_review_rows(video, windows, observations, events, extra=extra)
    if not rows:
        return {"ok": False, "video": video, "rows": 0,
                "error": "label_align 不可用或标签构建失败（见服务日志）"}
    for r in rows:
        old = existing.get(round(float(r["t_start"]), 3)) or {}
        m = store_meta.get(round(float(r["t_start"]), 3)) or {}
        r["thumb_path"] = _project_rel(r.get("thumb_path") or old.get("thumb_path")
                                       or m.get("thumb_path") or "")
        r["description"] = r.get("description") or old.get("description") or ""
        r["vector_model"] = r.get("vector_model") or old.get("vector_model") or ""
        if r.get("clip_dim") is None:
            r["clip_dim"] = old.get("clip_dim") or m.get("clip_dim")
        if r.get("frame_dim") is None:
            r["frame_dim"] = old.get("frame_dim") or m.get("frame_dim")
        # frame_dets 保留语义（§11 定案 6）：观察层原始数据不随 relabel 重算——
        # 新构建行无 frame_dets（relabel 的 obs 不带观察流）时旧行值原样保留；
        # 仅 body.observations 显式传入新观察数据才覆盖（本期无此入口）。
        ovr = _override_obs_get(override_obs, r["t_start"], r["t_end"])
        if ovr.get("frame_dets"):
            r["frame_dets"] = ovr["frame_dets"]
        elif not r.get("frame_dets"):
            r["frame_dets"] = old.get("frame_dets") or []
    n = db.upsert_review_index(rows)

    old_video = next(iter(db.list_review_videos(video)), {})
    try:
        from video_embed import label_align as la
        video_time, src = la.parse_video_time(video)
    except Exception:
        video_time, src = "", ""
    if not video_time and old_video.get("video_time"):
        video_time, src = old_video["video_time"], old_video.get("video_time_src") or ""
    db.upsert_review_video([{
        "video": video,
        "video_time": video_time, "video_time_src": src,
        "indexed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "vector_model": vector_model or old_video.get("vector_model") or "",
        "clip_dim": clip_dim if clip_dim is not None else old_video.get("clip_dim"),
        "frame_dim": frame_dim if frame_dim is not None else old_video.get("frame_dim"),
        "duration": max((w[1] for w in windows), default=0.0),
    }])
    return {"ok": True, "video": video, "rows": n, "events": len(events),
            "backend": db.review_backend_mode(), "note": "仅重对齐：向量库未改动"}


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
        dets = _as_detections(yolo.infer(frame, track=False))  # 测试页脚手架：造车辆框（生产由上游 yolo 阶段提供）
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
        dets = _as_detections(yolo.infer(frame, track=track))
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


# ---------------- 链路运行报告导出（自包含单文件 HTML，WO-0028） ----------------
# 最近一次链路运行快照：同步（h_pipeline_run）/流式（h_pipeline_stream）两条路径收尾时
# 经 _store_last_run 覆盖写入；POST /api/test/report 据此构建报告。服务重启即清空
# （报告本身已落 output_dir/report_*.html，持久保存）。
_LAST_RUN: dict = {}

# 报告表格列的中文标签（已知键优先按此展示，其余键直接用键名——通用不绑定链路）
_REPORT_COL_LABELS = {
    "plate_no": "车牌号", "type_name": "类型", "color": "颜色", "score": "置信度",
    "bbox": "车牌框", "car_bbox": "车辆框", "track_id": "track", "index": "序号",
    "size": "尺寸", "upscale": "放大倍数", "plate_img_size": "小图尺寸",
    "person_bbox": "人体框", "face_bbox": "人脸框", "face_size": "人脸尺寸",
    "kps": "关键点", "best_sim": "相似度", "name": "事件名", "detail": "detail",
}


def _store_last_run(snap: dict) -> None:
    """记录最近一次链路运行快照（整体替换：报告端点只认最近一次运行）。"""
    _LAST_RUN.clear()
    _LAST_RUN.update(snap)


def _esc(v) -> str:
    """HTML 转义（事件/VLM 文本可能含 <、& 等；None → 空串）。"""
    return html.escape(str(v if v is not None else ""), quote=True)


def _report_cell(v) -> str:
    """报告表格单元格文本：dict/list → 紧凑 JSON，float 去 尾 0，None/空 → 空串。"""
    if v is None or v == "":
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def _report_table(headers: list, rows: list, kv: bool = False) -> str:
    """HTML 表格（所有单元格经 _esc 转义；kv=True 时首列作键名窄列样式）。"""
    cls = ' class="kv"' if kv else ""
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
                  for row in rows)
    return (f"<table{cls}><thead><tr>{th}</tr></thead>"
            f"<tbody>{trs}</tbody></table>")


def _tl_rows(snap: dict, pred) -> list:
    """按谓词筛 timeline 条目 → [(ts, frame, data), ...]（data 非 dict 的条目跳过）。"""
    out = []
    for it in snap.get("timeline") or []:
        d = it.get("data")
        if isinstance(d, dict) and pred(it, d):
            out.append((it.get("ts"), it.get("frame"), d))
    return out


def _union_columns(rows_data: list, preferred: list) -> list:
    """列 = preferred 中实际出现的键 + 数据里其余键（按首现顺序，通用不绑定字段集）。"""
    cols = [k for k in preferred if any(k in d for _t, _f, d in rows_data)]
    for _t, _f, d in rows_data:
        for k in d:
            if k not in cols:
                cols.append(k)
    return cols


def _report_read_img(path: Path, max_edge: int, quality: int) -> str:
    """读盘图 → 限最大边长 → JPEG data URL（读失败/编码失败返回空串，调用方跳过）。"""
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if max_edge > 0 and max(h, w) > max_edge:
        k = max_edge / max(h, w)
        img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))))
    return _img_data_url(img, quality)


def _build_report_html(snap: dict, max_anno: int = 60, max_plate: int = 36,
                       max_face: int = 36, max_output: int = 60, img_edge: int = 960,
                       quality: int = 80) -> str:
    """链路运行快照 → 自包含单文件 HTML 报告。

    纯逻辑函数（不碰服务状态，输入快照 dict 输出 HTML 字符串，可单测）。
    通用实现不绑定链路类型：字段表全部取自 timeline 条目的 data（按数据形状识别
    事件/VLM/车牌/人脸），图片按落盘文件名前缀分组（frame_/plate_/face_）；
    某类内容缺失时对应小节整体省略；图片内嵌 data URL，离线双击可开。
    快照含 output 提炼块（WO-0029 链路尾输出声明）时，新增「链路尾输出」区块且
    **优先呈现**（事件表之前）：提炼表 + items.images 落盘图内嵌（读文件转 data URL，
    上限 max_output）；无声明维持现状取数。
    """
    name = snap.get("name") or "未命名"
    stats = snap.get("stats") or {}
    media = snap.get("media") or {}
    timeline = snap.get("timeline") or []

    # ---- 摘要 ----
    stage_names = " → ".join(_stage_label(s) for s in (snap.get("stages") or [])) or "—"
    if media.get("type") == "image":
        media_txt = f"图片 {media.get('name') or '—'}"
        range_txt = "单帧"
    else:
        media_txt = f"视频 {media.get('name') or '—'}"
        range_txt = (f"{_report_cell(media.get('start_time', 0))}s – "
                     f"{_report_cell(media.get('end_time', 0))}s"
                     f"（帧 {media.get('start_frame', 0)}–{media.get('end_frame', 0)}）")
    saved = stats.get("saved_images", 0)
    dropped = stats.get("dropped_saves", 0)
    save_txt = (f"{saved} 张" if saved else "无（流式运行不落盘）")
    if dropped:
        save_txt += f"（超上限跳过 {dropped} 张）"
    sum_rows = [
        ("链路", f"{name}（{snap.get('pipeline_id') or '-'}）"),
        ("阶段序列", stage_names),
        ("媒体", media_txt),
        ("时间范围", range_txt),
        ("处理帧数", stats.get("processed", 0)),
        ("检出目标", stats.get("detections", 0)),
        ("车牌识别", stats.get("plates", 0)),
        ("事件", stats.get("events", 0)),
        ("VLM 状态", stats.get("vlm_state") or "—"),
        ("图片落盘", save_txt),
        ("运行方式", "实时播放叠加（流式）" if snap.get("mode") == "stream" else "批量运行"),
        ("运行耗时", f"{_report_cell(stats.get('elapsed_sec', snap.get('elapsed_sec', 0)))} s"),
        ("生成时间", snap.get("generated_at") or "—"),
    ]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh"><head><meta charset="utf-8">',
        f"<title>UMVP 链路运行报告 · {_esc(name)}</title>",
        "<style>",
        "body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;margin:24px auto;"
        "max-width:1080px;padding:0 16px;color:#1f2430;line-height:1.55}",
        "h1{font-size:20px;border-bottom:2px solid #4f8cff;padding-bottom:8px}",
        "h2{font-size:16px;color:#2c5cc5;margin:28px 0 8px}",
        "h3{font-size:14px;margin:16px 0 6px}",
        "table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}",
        "th,td{border:1px solid #ccd4e0;padding:5px 8px;text-align:left;vertical-align:top;"
        "white-space:pre-wrap;word-break:break-all}",
        "th{background:#eef3fc}",
        "table.kv td:first-child{width:130px;background:#f7f9fd;font-weight:600;white-space:nowrap}",
        ".meta{color:#5a6478;font-size:12px}",
        ".imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}",
        "figure{margin:0;border:1px solid #ccd4e0;border-radius:6px;overflow:hidden}",
        "figure img{width:100%;display:block}",
        "figcaption{font-size:10.5px;color:#5a6478;padding:4px 6px;word-break:break-all}",
        "</style></head><body>",
        "<h1>UMVP 链路运行报告</h1>",
        f'<div class="meta">链路 {_esc(name)} ｜ 生成时间 '
        f'{_esc(snap.get("generated_at") or "—")} ｜ UMVP 测试台导出'
        f'（自包含单文件，离线可开）</div>',
        "<h2>运行摘要</h2>",
        _report_table(["项目", "内容"], sum_rows, kv=True),
    ]

    # ---- 链路尾输出（spec 顶层 output 声明的提炼块，WO-0029；**优先呈现**于事件表前）----
    # items.images 为相对 output_dir 的磁盘路径 → 报告构建时读文件转 data URL 内嵌
    # （复用 _report_read_img 与 img_edge/quality，上限 max_output；无声明无此区块）。
    ob = snap.get("output")
    if isinstance(ob, dict) and (ob.get("items") or ob.get("error") or ob.get("note")):
        n_out = len(ob.get("items") or [])
        parts.append(f"<h2>链路尾输出（{_esc(ob.get('count', n_out))} 条）</h2>")
        if ob.get("error"):
            parts.append(f'<div class="meta">输出块错误：{_esc(ob["error"])}</div>')
        else:
            bits = [f'scope={ob.get("scope", "events")}', f'from={ob.get("from") or "—"}']
            if ob.get("truncated"):
                bits.append("已截断（超出 limit）")
            if ob.get("note"):
                bits.append(str(ob["note"]))
            parts.append(f'<div class="meta">{_esc(" ｜ ".join(bits))}</div>')
            out_rows = []
            for i, it in enumerate(ob.get("items") or [], 1):
                out_rows.append([
                    i, _report_cell(it.get("ts")), it.get("frame_idx"),
                    _report_cell(it.get("label")), _report_cell(it.get("track_id")),
                    _report_cell(it.get("score")),
                    _report_cell(it.get("events")) or "—",
                ])
            parts.append(_report_table(
                ["#", "时间(s)", "帧", "label", "track", "score", "events（明细）"],
                out_rows))
            out_dir_o = snap.get("output_dir")
            if out_dir_o and n_out:
                all_imgs = [str(p) for it in ob.get("items") or []
                            for p in (it.get("images") or [])]
                figs = []
                for p in all_imgs[:max_output] if max_output > 0 else all_imgs:
                    url = _report_read_img(Path(out_dir_o) / p, img_edge, quality)
                    if not url:
                        continue
                    figs.append(f'<figure><img src="{url}" alt="{_esc(p)}">'
                                f"<figcaption>{_esc(p)}</figcaption></figure>")
                if figs:
                    note = (f"（内嵌 {len(figs)}/{len(all_imgs)} 张，超出上限未嵌入）"
                            if len(all_imgs) > len(figs) else f"（{len(figs)} 张）")
                    parts.append(f"<h3>提炼配图{note}</h3>")
                    parts.append(f'<div class="imgs">{"".join(figs)}</div>')

    # ---- 事件列表（timeline 条目 data 含 name 键 = 事件；通用按数据形状识别）----
    ev_rows = _tl_rows(snap, lambda it, d: "name" in d)
    parts.append(f"<h2>事件列表（{len(ev_rows)} 条）</h2>")
    if ev_rows:
        rows = []
        for ts, fr, d in ev_rows:
            extra = {k: v for k, v in d.items() if k not in ("name", "detail")}
            rows.append([
                _report_cell(ts), fr, d.get("name"),
                _report_cell(d.get("detail")) or "—",
                _report_cell(extra) if extra else "",
            ])
        parts.append(_report_table(
            ["时间(s)", "帧", "事件名", "detail", "其他字段"], rows))
    else:
        parts.append('<div class="meta">（本次运行无事件）</div>')

    # ---- VLM 研判（kind=vlm 且 data 含 action 键）----
    vlm_rows = _tl_rows(snap, lambda it, d: it.get("kind") == "vlm" and "action" in d)
    parts.append(f"<h2>VLM 研判（{len(vlm_rows)} 条）</h2>")
    if vlm_rows:
        rows = []
        for ts, fr, d in vlm_rows:
            content = d.get("content")
            content_txt = (json.dumps(content, ensure_ascii=False, indent=2)
                           if isinstance(content, (dict, list)) else _report_cell(content))
            rows.append([_report_cell(ts), fr, d.get("action"),
                         _report_cell(d.get("description")), content_txt])
        parts.append(_report_table(
            ["时间(s)", "帧", "action", "description", "content（原始 JSON）"], rows))
    else:
        parts.append('<div class="meta">（本次运行无 VLM 研判记录）</div>')

    # ---- 车牌识别（data 含 plate_no 键）----
    plate_rows = _tl_rows(snap, lambda it, d: "plate_no" in d)
    parts.append(f"<h2>车牌识别（{len(plate_rows)} 条）</h2>")
    if plate_rows:
        cols = _union_columns(plate_rows, ["plate_no", "type_name", "color",
                                           "score", "bbox", "car_bbox"])
        rows = [[_report_cell(ts), fr] + [_report_cell(d.get(c)) for c in cols]
                for ts, fr, d in plate_rows]
        headers = ["时间(s)", "帧"] + [_REPORT_COL_LABELS.get(c, c) for c in cols]
        parts.append(_report_table(headers, rows))
    else:
        parts.append('<div class="meta">（本次运行无车牌识别记录）</div>')

    # ---- 人脸截取（kind=face 条目；含嵌入链路的重复人脸行不计入）----
    face_rows = _tl_rows(snap, lambda it, d: it.get("kind") == "face")
    parts.append(f"<h2>人脸截取（{len(face_rows)} 张）</h2>")
    if face_rows:
        cols = _union_columns(face_rows, ["index", "face_bbox", "person_bbox",
                                          "face_size", "score", "best_sim"])
        rows = [[_report_cell(ts), fr] + [_report_cell(d.get(c)) for c in cols]
                for ts, fr, d in face_rows]
        headers = ["时间(s)", "帧"] + [_REPORT_COL_LABELS.get(c, c) for c in cols]
        parts.append(_report_table(headers, rows))
    else:
        parts.append('<div class="meta">（本次运行无人脸截取记录）</div>')

    # ---- 重要图片：从落盘目录按前缀分组内嵌（限量防文件超长；无图省略该分组）----
    # saved_files 清单存在时只嵌本次运行落盘的文件（目录里常累积多次运行的历史图，
    # 不按清单过滤会把旧的车牌/人脸/标注帧全嵌进来，报告虚胖）；键缺失（旧快照/
    # 手工构造）回退扫目录全量，保持兼容。
    parts.append("<h2>重要图片</h2>")
    out_dir = snap.get("output_dir")
    n_groups = 0
    if out_dir:
        out_path = Path(out_dir)
        groups = [
            ("标注帧", "frame_", ("_det", "_annotated"), max_anno),
            ("车牌小图", "plate_", None, max_plate),
            ("人脸截图", "face_", None, max_face),
        ]
        own = snap.get("saved_files")
        own_set = set(own) if own is not None else None
        for gtitle, prefix, keywords, limit in groups:
            if limit <= 0 or not out_path.is_dir():
                continue
            files = sorted(
                f for f in out_path.iterdir()
                if f.is_file() and f.suffix.lower() in _IMG_SUFFIX
                and f.name.startswith(prefix)
                and (keywords is None or any(k in f.name for k in keywords))
                and (own_set is None or f.name in own_set))
            if not files:
                continue
            figs = []
            for fp in files[:limit]:
                url = _report_read_img(fp, img_edge, quality)
                if not url:
                    continue
                figs.append(f'<figure><img src="{url}" alt="{_esc(fp.name)}">'
                            f"<figcaption>{_esc(fp.name)}</figcaption></figure>")
            if not figs:
                continue
            n_groups += 1
            if len(files) > limit:
                note = f"（共 {len(files)} 张，超出上限仅嵌入前 {limit} 张）"
            else:
                note = f"（{len(figs)} 张）"  # 按实际嵌入数（不可读的坏图已跳过）
            parts.append(f"<h3>{_esc(gtitle)}{note}</h3>")
            parts.append(f'<div class="imgs">{"".join(figs)}</div>')
    if not n_groups:
        parts.append('<div class="meta">（本次运行无落盘图片）</div>')

    parts.append("</body></html>")
    return "\n".join(parts)


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

    链路尾输出（WO-0029.2，doc/design/chain_output.md §3-§6）：spec 顶层（或 meta）声明
    output 段时，逐帧按 scope 粗筛累积末段记录流 + 配图预对位（内存安全 MVP），收尾
    collect_output 统一提炼、配图落盘（命名 output_*，共用 max_saved_images 计数与
    saved_files 清单），结果与 _LAST_RUN 快照附 "output" 提炼块；Image 帧级产物（MVP）
    从 saved_files 按帧号引用不新落盘；无声明时行为与现状逐字一致。
    """
    t0 = time.time()
    # 获取链路配置（业务 ID，如 PIPE0001）
    pipeline_id = str(p.get("pipeline_id", "") or "").strip()
    if not pipeline_id:
        return res(ok=False, error="请指定测试链路（pipeline_id）")
    run_id = uuid.uuid4().hex[:12]  # 本次运行标识（事件归档 jsonl 按 run_id 幂等，§12-C）
    
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

    # ---- 链路尾输出声明（WO-0029.2，doc/design/chain_output.md §3-§4）----
    # 解析失败 fail-soft：运行照常，收尾附 {"output": {"error": ...}}（validate_chain 硬拦）；
    # 无声明（out_plan/out_err 均 None）→ 行为与现状逐字一致（不产生 output 键）。
    out_plan, out_err = _resolve_output_plan(spec)
    if out_plan is not None and out_plan.get("from_deferred"):
        # 链路有事件尾（seq_filter rules → Pipeline.event_filter）= publish 通路在位：
        # 缺省 from 空结果视为「通路跑过但无事件」（count=0 + note），非「取不到」。
        out_plan["publish_wired"] = getattr(pipe, "event_filter", None) is not None
    out_acc = _make_output_accumulator(out_plan) if out_plan else None
    out_block: list = [None]  # _done() 收尾填充；各 return 经 _res_with_output 附带

    # 测试参数
    feedback = _bool(p, "vlm_feedback")
    vlm_action = str(p.get("vlm_action", "fire")).strip() or "fire"
    max_frames = _int(p, "max_frames", 0)
    start_time = _float(p, "start_time", 0.0)
    end_time = _float(p, "end_time", 0.0)

    # 真实 VLM 配置：来自链路 vlm 阶段参数（use_real_vlm/vlm_provider/vlm_endpoint/
    # vlm_model/vlm_key）
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
    # （仅 crops 源可能多张；多图 payload 已由 glm-4v-flash 实证支持，见 INB-0040）。
    # 非法值回落默认 first。
    image_pick = _str_param(vlm_cfg, "image_pick", "first").lower()
    if image_pick not in VLM_IMAGE_PICKS:
        image_pick = "first"
    vlm_fail = 0  # 真实调用失败次数（网络等），累计供结果提示
    vlm_hint_shown = False  # 4xx 可读提示每轮最多一条（避免逐帧刷屏）
    # 真实 VLM 端点预探测：不可达则本轮直接跳过真实调用，避免每帧长超时把测试拖住
    # （URL 经 _resolve_vlm_cfg：env VLM_ENDPOINT 优先，其次 vlm_provider 插槽，
    # 与 _vlm_real 保持一致）
    _vlm_url = _resolve_vlm_cfg(vlm_cfg)[0]
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
    saved_files: list = []  # 本次运行实际落盘的文件名（报告只嵌这些，不掺历史运行）

    def _save_image(path: Path, img) -> bool:
        if saved_images[0] >= max_saved_images:
            dropped_saves[0] += 1
            return False
        cv2.imwrite(str(path), img)
        saved_images[0] += 1
        saved_files.append(path.name)
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

    # ---- 运行快照（报告导出）：各分支 return 前调 _done()，统一收尾避免两处漂移 ----
    media_info: dict = {}   # 各测试分支填（image/video 名与范围）

    def _done():
        if feedback:
            if use_real_vlm:
                vs = "端点不可达，已跳过" if vlm_unreachable else "真实调用"
                if vlm_fail:
                    vs += f"（失败 {vlm_fail} 次）"
            else:
                vs = f"模拟回填({vlm_action})"
        else:
            vs = "未开启研判回填"
        # 链路尾输出提炼块（WO-0029.2）：收尾统一在此产出（res 附带与快照共用同块）
        if out_plan is not None:
            out_block[0] = _finish_output(
                out_plan, out_acc, output_dir, image_format, _save_image,
                start_frame=start_frame, saved_files=saved_files)
        elif out_err:
            out_block[0] = {"error": out_err}
        _store_last_run({
            "name": spec.get("name") or "未命名",
            "pipeline_id": pipeline_id,
            "stages": list(_stage_order),
            "media": dict(media_info),
            "stats": {
                "processed": processed_count, "detections": detection_count,
                "plates": plate_count, "events": event_count,
                "vlm_state": vs, "vlm_fail": vlm_fail,
                "saved_images": saved_images[0], "dropped_saves": dropped_saves[0],
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "timeline": timeline,
            "output_dir": str(output_dir),
            "saved_files": list(saved_files),
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "sync",
            **({"output": out_block[0]} if out_block[0] is not None else {}),
        })
        # 事件留存归档（WO-0033 §3，契约 C）：只归档链路尾事件 → 追加
        # out/events/<视频名>.jsonl（run_id 幂等）；链路含视频嵌入阶段时顺带标签
        # 对齐落表（PIPE0004 收尾对齐，§3/§4）。任一步失败不影响运行主流程。
        _vid = str(media_info.get("name") or "") if media_info.get("type") == "video" else ""
        if _vid:
            _archive_and_align(_vid, pipeline_id, _tail_events_of_timeline(timeline),
                               run_id, _vem, _vem_params, store_path=_vem_store_path)

    def _res_with_output(**kw) -> dict:
        """res(...) 结果附带链路尾输出块（无声明/未产出时不带 output 键，零回归）。"""
        r = res(**kw)
        if out_block[0] is not None:
            r["output"] = out_block[0]
        return r

    def _output_accumulate_frame():
        """本帧尾记录流 → 累积器（scope 粗筛 + 同帧配图预对位；内存安全见工厂 docstring）。

        from 产物本帧未登记（如 rules-only 事件尾无 dets 产物、无事件帧无 pipeline.tail）
        → 该帧跳过；缺省 from 每帧运行时裁决（_output_pick_from），空块解释收尾统一给。
        """
        if out_acc is None:
            return
        frm = _output_pick_from(out_plan, pipe.artifacts)
        if frm is None:
            return
        recs = pipe.artifacts.get(frm)
        if not isinstance(recs, list):
            return
        imgs = None
        if out_plan.get("image_type") == "Records":
            v = pipe.artifacts.get(out_plan["image_full"])
            imgs = v if isinstance(v, list) else None
        out_acc["add"](recs, imgs)

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

    # ---- 视频嵌入（PIPE0004 向量档索引；驱动层逐帧 run，收尾 finish + 底库落盘）----
    _vem = getattr(pipe, "video_embed_stage", None)
    _vem_pos = next((i for i, s in enumerate(spec.get("stages") or [])
                     if s.get("module_id") == "video_embed"), -1)  # dets 端口连线解析位
    _vem_params = next((s.get("params") or {} for s in spec.get("stages") or []
                        if s.get("module_id") == "video_embed"), {})
    _vem_store_path = str(_vem_params.get("store_path") or "").strip()
    _vem_saved = ""  # 收尾落盘后的底库路径（结果头展示用）

    def _step_out(frame_no, ts, tag, dets, subs, al, frame=None):
        """逐模块展示本帧各自处理的内容（不管是否向下传递），每条带帧序数。"""
        nonlocal detection_count, vlm_fail, plate_count, vlm_unreachable, event_count
        nonlocal face_count, face_dup, vlm_hint_shown
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

        # 门控：按 key 节流放行（仅在确有拦截时刷日志，参照 det_filter 形态；
        # 附加产物 gate_filter.stats 的分键明细，"*（匿名）" 即匿名桶拦截量）
        if "gate_filter" in _stage_order:
            gi = getattr(pipe, "last_gate", None)
            if gi is not None and gi[0] != gi[1]:
                _emit("info", f"门控 放行 {gi[1]} / 拦截 {gi[0] - gi[1]}"
                              + _gate_stats_text(pipe), ts,
                      stage="gate_filter", frame_no=frame_no, log_line=True, prefix="    ")

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

        # 标注：本帧生成带框图（供 VLM 素材）；record 模式逐实例展示出图张数/截断告警
        # （WO-0022：record 模式 last_annotated 不更新，旧文案会误报「无检出」）
        if "annotate" in _stage_order:
            _anno_recs, _anno_has_frame, _anno_ran = _annotate_stats(pipe, dets, frame)
            for _drawn, _cap, _dropped in _anno_recs:
                _emit("info", f"逐记录出图 {_drawn} 张（上限 {_cap}）", ts,
                      stage="annotate", frame_no=frame_no, log_line=True, prefix="    ")
                if _dropped:
                    _emit("alarm", f"带图记录超上限 {_cap}，截断 {_dropped} 条"
                                   "（可调高 annotate.max_records 或收紧上游筛选）", ts,
                          stage="annotate", frame_no=frame_no, log_line=True, prefix="    ")
            if _anno_has_frame or not _anno_ran:
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
                        except Exception as e:
                            vlm_fail += 1
                            _emit("fail", f"调用失败({vlm_fail}): {e}", ts, stage="vlm",
                                  frame_no=frame_no, log_line=True, prefix="    ")
                            hint = getattr(e, "hint", "")
                            if hint and not vlm_hint_shown:
                                vlm_hint_shown = True
                                _emit("info", hint, ts, stage="vlm",
                                      frame_no=frame_no, log_line=True, prefix="    ")
                            if vlm_fail >= 3 and not vlm_unreachable:
                                vlm_unreachable = True
                                _emit("info", "真实 VLM 连续失败≥3，本轮后续改用素材包", ts,
                                      stage="vlm", frame_no=frame_no, log_line=True, prefix="    ")
                            continue
                        # content（VLM 原始 JSON）先算好：timeline 条目挂 data 供报告导出
                        # （except 分支已 continue，走到这里必是调用成功）
                        content = _vlm_content(raw) if raw else {"result": action, "description": description}
                        _emit("vlm", f"研判 {action}"
                              + (f" — {description}" if description else "")
                              + f"（真实调用{src_tag}）",
                              ts, stage="vlm",
                              data={"action": action, "description": description,
                                    "content": content},
                              frame_no=frame_no, log_line=True, prefix="    ")
                    else:
                        action, description = vlm_action, ""
                        content = {"result": action, "description": description}
                        _emit("vlm", f"研判 {action}（模拟回填）", ts, stage="vlm",
                              data={"action": action, "description": description,
                                    "content": content},
                              frame_no=frame_no, log_line=True, prefix="    ")
                    # VLM 结果作为记录经事件尾（event_filter）出事件（content 已在上方计算）
                    rec = {"kind": "vlm", "label": action, "text": description,
                           "content": content}
                    for ev in pipe.publish([rec], ts, frame_no):
                        event_count += 1
                        nm = ev.get("name", "事件")
                        det = ev.get("detail") or {}
                        _emit("submit", f"事件 {nm}" + (f" {det}" if det else ""), ts,
                              stage="seq_filter", frame_no=frame_no, data=ev,
                              log_line=True, prefix="    ")

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

        # 视频嵌入（向量化）：本帧片段产出入库明细与缓冲进度（展示层；记录已 publish 进事件流）
        if "video_embed" in _stage_order:
            vem = getattr(pipe, "video_embed_stage", None)
            if vem is not None:
                _clips = getattr(pipe, "last_clips", None) or []
                for r in _clips:
                    _emit("submit", f"片段#{r.get('clip_id')} 入库 "
                                    f"t={float(r.get('t_start') or 0):.2f}"
                                    f"~{float(r.get('t_end') or 0):.2f}s"
                                    f"（{r.get('frame_count')} 帧"
                                    f"{'，尾段' if r.get('partial') else ''}）",
                          ts, stage="video_embed", frame_no=frame_no,
                          data={k: v for k, v in r.items() if k != "embedding"},
                          log_line=True, prefix="    ")
                if not _clips:
                    _emit("info", f"片段缓冲 {len(vem._buf)}/{vem.frames_per_clip} 帧",
                          ts, stage="video_embed", frame_no=frame_no,
                          log_line=True, prefix="    ")

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
                  stage="seq_filter", frame_no=frame_no, data=ev,
                  log_line=True, prefix="    ")
        log.append("")  # 帧块之间空行，多换行更易读
    
    # 初始化计数器
    processed_count = 0
    detection_count = 0
    event_count = 0
    plate_count = 0
    plate_saved = [0]  # 已落盘原分辨率车牌小图数（文件名编号用）
    start_frame = 0    # 视频分支按起始时间重算；图片分支恒 0（_done 收尾帧号映射用）
    
    # 图片测试模式
    image_name = str(p.get("image", "")).strip()
    if image_name:
        if "video_embed" in _stage_order:
            return res(ok=False, error="视频向量化链路仅支持视频输入"
                                       "（索引按片段窗口切分，请选择视频测试）")
        img_path = _find_file(image_name, [PIC_DIR, TEST_IMGS])
        if not img_path:
            return res(ok=False, error=f"找不到图片: {image_name}")

        if pipe.yolo is None and pipe.plate is None:
            return res(ok=False, error="该链路无 YOLO/车牌阶段，图片测试仅适用于含检测的链路")

        img_path, frame = _load_frame({"image": image_name})
        media_info = {"type": "image", "name": img_path.name}

        # 保存原图
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_name = output_dir / f"frame_000_{timestamp}_orig.{image_format}"
        _save_image(orig_name, frame)

        dets = _as_detections(pipe.yolo.infer(frame, track=pipe.track_needed())) if pipe.yolo else []

        subs, al = pipe.step(0, 0.0, dets, frame=frame)
        subs = _apply_vlm_inputs(pipe, subs)
        _step_out(0, 0.0, f"图片测试: {img_path.name}", dets, subs, al, frame=frame)
        _output_accumulate_frame()  # 链路尾输出：单帧图片测试同样参与提炼累积

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
            _done()
            return _res_with_output(text=head + "\n" + "\n".join(log),
                                    images=images, timeline=timeline)

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
        _done()
        return _res_with_output(text=head + "\n" + "\n".join(log),
                                images=images, timeline=timeline)

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
    media_info = {"type": "video", "name": vid.name, "start_time": start_time,
                  "end_time": end_time, "start_frame": start_frame,
                  "end_frame": end_frame}

    # 视频嵌入：片段 upsert 键 (video, t_start) 用真实文件名（未显式配置时），重跑幂等覆盖
    if _vem is not None and str(_vem.video) in ("", "video"):
        _vem.video = vid.name

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
            # WO-0033.9：喂管道保留 infer() 原始 dict 记录（带 label）——曾用
            # _as_detections 强转 Detection（无 label 字段，见 composer），label 在
            # 进 pipe.step 前即被剥，det_filter/video_embed（dets 端口连线源）拿到
            # 的记录无 label → review_index.class_names 恒空。step 入口 to_record
            # 对 dict 直通；下方展示消费点（_step_out/_draw_tracks 读对象属性）各自
            # 局部转换，不再在喂管道前统一强转。
            dets = list(pipe.yolo.infer(frame, track=pipe.track_needed()))

        subs, al = pipe.step(n, ts, dets, frame=frame, force=True)
        subs = _apply_vlm_inputs(pipe, subs)

        # 视频嵌入：逐帧推入片段缓冲（含观察源 dets 第二入端口，§12-1）；满窗产出的
        # 片段记录经 publish 进事件流（clip_indexed 事件 → 事件计数 / 链路尾输出提炼
        # pipeline.tail 可取）
        if _vem is not None:
            pipe.last_clips = _vem_feed_dets(pipe, _vem, _vem_pos, frame, ts, n)
            for ev in pipe.publish(pipe.last_clips, ts, n):
                event_count += 1
                nm = ev.get("name", "事件")
                det = ev.get("detail") or {}
                _emit("submit", f"事件 {nm}" + (f" {det}" if det else ""), ts,
                      stage="video_embed", frame_no=n, data=ev,
                      log_line=True, prefix="    ")

        # 展示消费点局部转 Detection（WO-0033.9：dets 已是原始 dict 记录）
        _step_out(n, ts, tag, _as_detections(dets), subs, al, frame=frame)
        _output_accumulate_frame()  # 链路尾输出：本帧尾记录流按 scope 粗筛+配图预对位累积

        # 保存关键帧图片（检出框 + 车牌框）；人脸链只落人脸图（在 _step_out 里），不回灌关键帧
        if not is_face_chain:
            plates = getattr(pipe, "last_plates", [])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if (dets and len(dets) > 0) or plates:
                canvas = frame.copy()
                _draw_tracks(canvas, _as_detections(dets or []))
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

    # 视频嵌入收尾：冲刷尾段（partial）→ publish → 底库落盘（PIPE0004「入库即目的」）
    if _vem is not None:
        _tail_ts = (n / fps) if fps else 0.0
        pipe.last_clips = _vem.finish()
        for ev in pipe.publish(pipe.last_clips, _tail_ts, n):
            event_count += 1
            nm = ev.get("name", "事件")
            det = ev.get("detail") or {}
            _emit("submit", f"事件 {nm}" + (f" {det}" if det else ""), _tail_ts,
                  stage="video_embed", frame_no=n, data=ev, log_line=True, prefix="    ")
        if _vem_store_path:
            _sp = Path(_vem_store_path)
            if not _sp.is_absolute():
                _sp = ROOT / _vem_store_path
            _vem_saved = _vem.store.save(str(_sp))

    if is_face_chain:
        head = f"链路: {spec.get('name') or '未命名'} | 视频: {vid.name}"
        head += f" | 范围: {start_time}s-{end_time}s ({start_frame}-{end_frame}帧)"
        head += f" | 处理 {processed_count} 帧"
        if femb is not None:
            head += f" | 新身份 {face_count} 个 | 重复丢弃 {face_dup} 次 | 去重阈值 {face_dedup_thresh}"
        else:
            head += f" | 截取人脸 {face_count} 张"
        head += f" | 输出到: {output_dir}" + _save_summary()
        _done()
        return _res_with_output(text=head + "\n" + "\n".join(log),
                                images=images, timeline=timeline)

    head = f"链路: {spec.get('name') or '未命名'} | 视频: {vid.name}"
    head += f" | 范围: {start_time}s-{end_time}s ({start_frame}-{end_frame}帧)"
    head += f" | 处理 {processed_count} 帧 | 检出 {detection_count} 个 | 车牌 {plate_count} 次 | 事件 {event_count} 个"
    if _vem is not None:
        head += f" | 向量化片段 {len(_vem.indexed)} 条"
        if _vem_saved:
            head += f"（底库 {_vem_saved}）"
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

    _done()
    return _res_with_output(text=head + "\n" + "\n".join(log),
                            images=images, timeline=timeline)


# ---------------- 模块链路运行器（runner=chain，模块链路模式） ----------------
# 已完成链路 spec 以 stages（模块链路）为真相源：各阶段存 web_lab 模块参数，
# _chain_from_spec 据此组装四模块实例（帧管理→YOLO→VLM→告警，可选阶段）。


# _chain_from_spec 能装配进执行路径的阶段（validate_chain「声明即执行」断言的唯一依据）：
# 新增可装配阶段务必同步登记，否则 spec 声明了它却静默不执行（WO-0001 事故即此类）。
CHAIN_ASSEMBLED_MODULES = {
    "frame_manager", "yolo", "det_filter", "seq_filter", "gate_filter", "target_crop",
    "annotate", "vlm", "plate_recog", "face_detect", "extract_faces", "face_embed",
    "video_embed",
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
    # video_embed.frame：web 驱动层逐帧喂入（stage.run(frame, ts, frame_idx)，不经
    # Pipeline.step 解析——对齐 vlm.dets 的 driver 解析定位）；
    # video_embed.dets（WO-0033 §12-1）：第二入端口（观察源 Records，可缺省），
    # 驱动层经 _vem_feed_dets 按 stage_inputs 连线取上游产物（yolo.dets /
    # det_filter.dets / plate_recog.plates / face_detect.faces，单观察源直连）
    "video_embed": ("frame", "dets"),
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
    "gate_filter": ("pipe.gate_filter", "GateFilter"),
    "target_crop": ("pipe.target_crop", "TargetCrop"),
    "annotate": ("pipe.annotate", "Annotator"),
    "face_detect": ("face_detect.face_pipeline", "FaceStage"),
    "plate_recog": ("plate_recog.plate_pipeline", "PlateStage"),
    "video_embed": ("video_embed.video_embed_stage", "VideoEmbedStage"),
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


# ---- 节点实例身份（图编排 P2-0，设计稿 doc/design/graph_chain.md §3.1）----
# 产物名解析（短名 → 全名、多实例歧义报错）**单一实现**在 pipe.composer（WO-0008.1）；
# 本处 re-export 同一实现，供 validate_chain / db 复用，避免各写一份。
from pipe.composer import PortResolutionError as PortResolutionError  # noqa: E402, F401

try:  # pragma: no cover - 取决于 composer 是否已提供（WO-0008.1）
    from pipe.composer import resolve_artifact as resolve_artifact  # noqa: E402
    _RESOLVE_ARTIFACT_FROM_COMPOSER = True
except ImportError:  # composer 未提供时的过渡兜底：语义与契约一致，落地后自动改用 composer
    _RESOLVE_ARTIFACT_FROM_COMPOSER = False

    def resolve_artifact(name, positions_by_module):  # type: ignore[misc]
        """产物名 → 全名 ``{module}#{position}.{artifact}``（短名单实例兼容、多实例报错）。

        与 pipe.composer.resolve_artifact 契约一致（WO-0008.1）：已是全名原样返回；
        短名唯一实例才允许；多实例歧义 / 未知模块 / 空名抛 PortResolutionError。
        """
        name = (name or "").strip()
        if not name:
            raise PortResolutionError("产物名为空")
        head, _sep, _tail = name.partition(".")
        if "#" in head:
            mod, _h, pos = head.partition("#")
            if not mod or not pos.isdigit() or not _tail:
                raise PortResolutionError(f"非法产物全名: {name!r}")
            return name
        if "." not in name or not head:
            raise PortResolutionError(f"非法产物名: {name!r}")
        poss = list(positions_by_module.get(head) or [])
        if not poss:
            raise PortResolutionError(f"未知模块 {head!r}（产物 {name!r}）")
        if len(poss) > 1:
            raise PortResolutionError(
                f"产物短名 {name!r} 有歧义：模块 {head!r} 在本链路有 {len(poss)} 个实例"
                f"（位置 {poss}），请写全名（如 {head}#{poss[0]}.{_tail}）")
        return f"{head}#{poss[0]}.{_tail}"


def _positions_by_module(module_ids) -> dict:
    """module_id → 该模块在链路中的位置（数组下标）列表。"""
    out: dict = {}
    for i, mid in enumerate(module_ids):
        out.setdefault(mid, []).append(i)
    return out


# ---------------- 链路尾输出提炼（spec 顶层 output 段，WO-0029.2） ----------------
# 设计：doc/design/chain_output.md（v2）§3-§7。驱动层在每帧 pipe.step 后按 scope 粗筛 +
# 同帧配图预对位累积（内存安全 MVP），运行收尾调 pipe.chain_output.collect_output 统一
# 提炼、配图落盘并产出提炼块（同步 res/_LAST_RUN 快照 + SSE done 三处消费；
# 报告导出 _build_report_html 优先呈现）。

def _output_decl_of(spec: dict) -> dict | None:
    """spec 的 output 声明段：顶层优先，回落 meta.output（DB 往返只持久化 meta，
    直传 spec / validate_chain 校验的是顶层段）。非 dict 或空 dict → None
    （=未声明，行为与现状逐字一致）。"""
    if not isinstance(spec, dict):
        return None
    out = spec.get("output")
    if not isinstance(out, dict) or not out:
        meta = spec.get("meta")
        out = meta.get("output") if isinstance(meta, dict) else None
    return out if isinstance(out, dict) and out else None


def _default_from_candidates(spec_stages: list) -> list:
    """缺省 from 的**静态候选**（按声明序）：末段阶段的所有 Records 产物短名。

    静态声明只是「预报」——rules-only 事件尾等形态运行时并不登记该产物
    （INB-0109 根因②），故缺省 from 的真取值由 _output_pick_from 逐帧按
    「运行时实际登记 → pipeline.tail」裁决（WO-0029.5），本清单只作 ① 级候选。
    末段无 Records 产物声明（如 face_embed）→ 空列表（只剩 ② pipeline.tail 一条路）。
    """
    from pipe.composer import ARTIFACT_TYPES
    last = next((s.get("module_id") for s in reversed(spec_stages or [])
                 if s.get("module_id")), None)
    outs = (module_port_decls().get(last) or {}).get("outputs") or {}
    return [f"{last}.{art}" for art, typ in outs.items()
            if typ == "Records" and f"{last}.{art}" in ARTIFACT_TYPES]


def _resolve_output_plan(spec: dict) -> tuple:
    """解析 output 声明 → (执行计划, 错误)。计划含解析后的产物全名/类型与规整 decl。

    显式 from 非法（产物名未知或歧义 / 类型非 Image|Records / scope/image_pick/limit
    非法）→ (None, 原因)：驱动层 **fail-soft**——运行照常完成，结果附
    {"output": {"error": ...}} 而非中断测试（spec 层另有 validate_chain 硬拦）。
    from 缺省 → **运行时解析**（WO-0029.5）：静态候选（末段 Records 产物）只是
    ① 级预报，逐帧由 _output_pick_from 按「实际登记 → pipeline.tail」裁决，
    全程取不到才在收尾提炼块报 error（不再静默空块），此路径不再于计划期报错。
    """
    decl = _output_decl_of(spec)
    if decl is None:
        return None, None
    from pipe.composer import ARTIFACT_TYPES, artifact_short
    stages = spec.get("stages") or []
    positions = _positions_by_module([s.get("module_id") for s in stages])
    plan: dict = {}
    # from：显式声明优先；缺省 = 运行时解析（tail_froms 为 ① 级候选全名展开，
    # ② 级 pipeline.tail 在 _output_pick_from 内置）
    frm = str(decl.get("from") or "").strip()
    if not frm:
        plan["from_deferred"] = True
        plan["from_full"] = None
        last = next((s.get("module_id") for s in reversed(stages)
                     if s.get("module_id")), None)
        last_pos = positions.get(last) or []
        tail_froms: list = []
        for short in _default_from_candidates(stages):
            tail_froms.append(short)  # 短名（单实例别名 / 位置未知形态）
            tail_froms += [f"{last}#{p}.{short.partition('.')[2]}" for p in last_pos]
        plan["tail_froms"] = tail_froms
    else:
        plan["from_deferred"] = False
        try:
            from_full = resolve_artifact(frm, positions)
        except PortResolutionError as e:
            return None, f"output.from={frm!r} 无法解析: {e}"
        from_type = ARTIFACT_TYPES.get(artifact_short(from_full))
        if from_type is None:
            return None, (f"output.from={frm!r} 未知产物"
                          f"（可用产物: {sorted(ARTIFACT_TYPES)}）")
        if from_type not in ("Image", "Records"):
            return None, f"output.from={frm!r} 类型 {from_type} 非法（须为 Image/Records）"
        plan["from_full"], plan["from_type"] = from_full, from_type
    # image：可选；Records=逐记录读旁路图（collect_output 对位），
    # Image=帧级图（MVP：收尾从 saved_files 落盘清单按帧号引用，见 _output_frame_image_index）
    img = str(decl.get("image") or "").strip()
    if img:
        try:
            img_full = resolve_artifact(img, positions)
        except PortResolutionError as e:
            return None, f"output.image={img!r} 无法解析: {e}"
        img_type = ARTIFACT_TYPES.get(artifact_short(img_full))
        if img_type is None:
            return None, (f"output.image={img!r} 未知产物"
                          f"（可用产物: {sorted(ARTIFACT_TYPES)}）")
        if img_type not in ("Image", "Records"):
            return None, f"output.image={img!r} 类型 {img_type} 非法（须为 Image/Records）"
        plan["image_full"], plan["image_type"] = img_full, img_type
    # scope / image_pick / limit：此处规整校验（fail-soft 早报），收尾传干净 decl
    plan["scope"] = str(decl.get("scope") or "events").strip().lower() or "events"
    if plan["scope"] not in ("events", "records"):
        return None, f'output.scope={decl.get("scope")!r} 非法（应为 "events" 或 "records"）'
    plan["image_pick"] = str(decl.get("image_pick") or "all").strip().lower() or "all"
    if plan["image_pick"] not in ("first", "all"):
        return None, f'output.image_pick={decl.get("image_pick")!r} 非法（应为 "first" 或 "all"）'
    try:
        limit = int(decl.get("limit") if decl.get("limit") is not None else 100)
    except (TypeError, ValueError):
        return None, f"output.limit={decl.get('limit')!r} 需为正整数"
    if limit <= 0:
        return None, f"output.limit={decl.get('limit')!r} 需为正整数"
    plan["limit"] = limit
    return plan, None


def _output_pick_from(plan: dict, artifacts: dict) -> str | None:
    """本帧 from 产物名裁决（缺省 from 运行时解析，WO-0029.5；同步/流式共用）。

    显式 from → 原样返回 plan["from_full"]（运行时缺该产物即本帧取不到，现状语义）。
    缺省 from → 逐帧优先级：
      ① 末段阶段**实际登记**的 Records 产物（tail_froms，运行时真相——静态声明
        ≠ 运行时登记，rules-only 事件尾从不登记 seq_filter.dets 即此类）；
      ② pipeline.tail（publish 通路承载记录，事件尾链路主路径；无事件帧该键不存在）；
      都没有 → None（本帧无可提炼，跳过不报错；是否「取不到」由收尾统一裁决）。
    命中即回写 plan["_used_from"]（最近一次实际来源，收尾块头呈现）与
    plan["_used_publish"]（用过 publish 通路，收尾加来源 note）。
    """
    if not plan.get("from_deferred"):
        return plan.get("from_full")
    for name in plan.get("tail_froms") or ():
        if isinstance(artifacts.get(name), list):
            plan["_used_from"] = name
            return name
    if isinstance(artifacts.get("pipeline.tail"), list):
        plan["_used_from"] = "pipeline.tail"
        plan["_used_publish"] = True
        return "pipeline.tail"
    return None


def _make_output_accumulator(plan: dict) -> dict:
    """逐帧累积器工厂（内存安全 MVP，实现决策见 WO-0029.2 回报）。

    add(records, image_records) 在每帧 pipe.step 之后调用：
      - 先按 scope 粗筛（events=只留带事件记录——事件是边沿语义天然少量）；
      - 再做**同帧 consume-on-use 配图预对位**（与 collect_output 同策略：一条配图记录
        只配一条主流记录）；对不上的主流记录占位 {"frame_idx": f}——占位无旁路图，
        收尾 collect_output 再对位时结果与本帧预对位逐条一致（同帧桶 consume-on-use
        保序，占位吸收配额且产出空图）；
      - 攒「已配对 (记录, 配图记录)」**对齐列表**（imgs[i] 与 recs[i] 一一对应）。
    Image 帧级产物**不在循环内累积**：帧级图历史不留内存，收尾经 saved_files 落盘
    清单按帧号建引用映射（_output_frame_image_index）。
    records 硬上限 = max(10×limit, 200) 条：超出置 truncated 并停止累积（MVP 语义：
    截断点之后的内容不进提炼块，count 只反映已累积部分；事件边沿语义下正常远达不到）。
    """
    from pipe.records import get_events
    cap = max(10 * plan["limit"], 200)
    st: dict = {"recs": [], "imgs": [], "truncated": False, "cap": cap}

    def add(records, image_records=None):
        if st["truncated"] or not records:
            return
        kept = [r for r in records
                if not (plan["scope"] == "events" and not get_events(r))]
        if not kept:
            return  # 本帧无贡献：其配图流一并丢弃（省内存关键——图不进无贡献帧）
        img_iter = iter(image_records or [])
        for r in kept:
            if len(st["recs"]) >= cap:
                st["truncated"] = True
                return
            st["recs"].append(r)
            ir = next(img_iter, None)
            st["imgs"].append(ir if ir is not None
                              else {"frame_idx": r.get("frame_idx")})
    st["add"] = add
    return st


def _output_frame_image_index(saved_files: list, start_frame: int = 0) -> dict:
    """saved_files 落盘清单 → {frame_idx: 文件名}（Image 帧级产物 MVP 引用映射）。

    Image 帧级产物（frame_manager.frame / annotate.annotated）每帧至多一张且随帧处理
    即时落盘（文件名 frame_{frame_num:04d}_...，同步路径 frame_num = 帧号 - start_frame）。
    **只引用已落盘文件**——不新落盘、不攒内存（同帧共用依赖帧级图历史留存，当前驱动层
    只有 last_annotated 一张在内存，此为 MVP 约束，见设计稿 §4.2）；同帧多张取清单中
    **最后**一张（落盘顺序 orig → annotated/det，标注图优先）。解析不到的帧无图。
    """
    out: dict = {}
    for name in saved_files or []:
        m = re.match(r"^frame_(\d{1,8})_", str(name))
        if not m:
            continue
        out[int(m.group(1)) + int(start_frame or 0)] = name  # 后者覆盖前者 = 最后一张
    return out


def _thumb_data_url(img, max_edge: int = 360, quality: int = 70) -> str:
    """缩略 data URL（提炼块随块返回，test.html 免读盘展示缩略图；失败返回空串）。"""
    try:
        h, w = img.shape[:2]
    except Exception:
        return ""
    if max_edge > 0 and max(h, w) > max_edge:
        k = max_edge / max(h, w)
        img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))))
    return _img_data_url(img, quality)


def _finish_output(plan: dict, acc: dict, output_dir, image_format: str = "jpg",
                   save_image=None, start_frame: int = 0, saved_files: list | None = None,
                   note: str | None = None) -> dict:
    """运行收尾产出提炼块：collect_output 统一提炼 → 配图落盘 → 相对路径 + thumbs。

    - 落盘命名 output_{序号:03d}_{tid或label}_{ts:.1f}s_{图序}.{ext}，落 output_dir，
      **共用** max_saved_images 计数（save_image 闭包）与 saved_files 清单；落盘成功
      才进 item.images（相对 output_dir 的文件名），并同步生成缩略 data URL 附
      item.thumbs（浏览器展示用，磁盘路径浏览器取不到）；
    - Image 帧级产物：不新落盘，从 saved_files 按帧号引用（_output_frame_image_index），
      解析不到的帧 images 为空列表；
    - 流式（output_dir=None）：不落盘，items.images 恒空列表，块附 note；
    - truncated = collect_output 的 limit 截断 或 累积器硬上限截断（两者取或）。
    """
    from pipe.chain_output import collect_output
    out = collect_output(
        acc["recs"], acc["imgs"] if plan.get("image_type") == "Records" else None,
        {"scope": plan["scope"], "image_pick": plan["image_pick"], "limit": plan["limit"]})
    items = out["items"]
    if plan.get("image_type") == "Image":
        idx = _output_frame_image_index(saved_files, start_frame) if saved_files else {}
        for it in items:
            ref = idx.get(it.get("frame_idx"))
            it["images"] = [ref] if ref else []
            if ref and output_dir is not None:
                # 引用文件读盘出缩略（条目数 ≤ limit，代价可控）；读失败无缩略不影响块
                thumb = _report_read_img(Path(output_dir) / ref, 360, 70)
                if thumb:
                    it["thumbs"] = [thumb]
    elif output_dir is not None and save_image is not None:
        for i, it in enumerate(items, 1):
            tid = it.get("track_id")
            if isinstance(tid, (int, float)) and not isinstance(tid, bool) and tid:
                tag = f"tid{int(tid)}"
            else:
                tag = re.sub(r'[\\/:*?"<>|\s]+', "", str(it.get("label") or ""))[:24] or "item"
            ts = float(it.get("ts") or 0.0)
            paths, thumbs = [], []
            for j, img in enumerate(it.get("images") or [], 1):
                fname = f"output_{i:03d}_{tag}_{ts:.1f}s_{j}.{image_format}"
                if save_image(Path(output_dir) / fname, img):
                    paths.append(fname)
                    thumb = _thumb_data_url(img)
                    if thumb:
                        thumbs.append(thumb)
            it["images"] = paths
            if thumbs:
                it["thumbs"] = thumbs
    else:
        for it in items:
            it["images"] = []
    # 块头/空块解释（WO-0029.5）：from 呈现运行时实际来源（缺省裁决 _used_from 优先）；
    # 来源为 pipeline.tail（publish 通路，显式或缺省裁决命中均算）加来源 note——有事件
    # 说明出处，无事件区分「没有」（count=0 不报 error）；缺省 from 全程取不到（末段
    # 无实际登记产物且链路无事件尾通路）→ error，不再静默空块。
    used_from = plan.get("_used_from")
    # 是否真用过 publish 通路：缺省裁决命中 pipeline.tail，或（显式 from=pipeline.tail）
    # 累积器实际取到承载记录——两者任一即「用到了」，来源 note 才不失真
    used_tail = bool(used_from) or bool(acc["recs"])
    shown_from = used_from or plan.get("from_full")
    if plan.get("from_deferred") and not used_from and plan.get("publish_wired"):
        shown_from = "pipeline.tail"  # 通路在位（事件尾已装配）：空结果按 publish 通路解释
    block = {"scope": plan["scope"], "from": shown_from, "count": len(items),
             "truncated": bool(out["truncated"] or acc["truncated"]), "items": items}
    notes = [note] if note else []
    if shown_from == "pipeline.tail":
        src_note = ("事件由链路尾 publish 产出，来源产物 pipeline.tail"
                    if plan["scope"] == "events"
                    else "记录由链路尾 publish 通路产出（来源产物 pipeline.tail）")
        if used_tail:
            notes.append(src_note)
        elif plan.get("publish_wired"):
            notes.append(src_note + "（本次运行无事件）")
    if plan.get("from_deferred") and not used_from and shown_from is None:
        block["error"] = ("末段无可用记录产物：缺省 from 运行时末段无实际登记的 "
                          "Records 产物，链路也无 publish 事件尾通路；"
                          "请显式声明 from（如 plate_recog.plates / "
                          "face_detect.faces / pipeline.tail）")
    if notes:
        block["note"] = "；".join(notes)
    return block


# 本期普通模块端口 fan_in 均为 0（fan-in 端口留 P2-3 过渡节点）；新增 fan-in 端口在此登记。
PORT_FAN_IN: dict = {}


def port_fan_in(module_id: str, port: str) -> int:
    """入端口是否允许多条入边（0=单上游，1=fan-in）。"""
    return int(PORT_FAN_IN.get((module_id, port), 0))


def module_ports_catalog() -> list:
    """**代码声明**的端口能力目录（module_ports 表唯一真相源）。

    返回 (module_id, direction, port_name, io_type, fan_in, resolved_by) 列表：
      - 入端口取自模块 INPUTS；resolved_by='pipe' 若该端口在 Pipeline.step 解析
        （composer.RESOLVED_PORTS），否则 'driver'（如 vlm 的 driver 层解析端口）；
      - 出端口取自模块 OUTPUTS，resolved_by='pipe'。
    """
    from pipe.composer import RESOLVED_PORTS as _RP
    rows = []
    for mid, d in module_port_decls().items():
        pipe_ports = set(_RP.get(mid, ()))
        for port, io_type in (d.get("inputs") or {}).items():
            rows.append((mid, "in", port, str(io_type), port_fan_in(mid, port),
                         "pipe" if port in pipe_ports else "driver"))
        for port, io_type in (d.get("outputs") or {}).items():
            rows.append((mid, "out", port, str(io_type), 0, "pipe"))
    return rows


def _build_face_stage(p: dict):
    """构造人脸检测阶段（FaceStage）；参数取自本阶段（兼容旧 extract_faces 字段）。"""
    from face_detect.face_detector import FaceDetector
    from face_detect.face_pipeline import FaceStage
    from pipe.cropper import Cropper
    fdet = FaceDetector(
        device=_str_param(p, "device", "auto"),
        det_thresh=(_float(p, "det_thresh", 0.0) or _float(p, "face_thresh", 0.4)),
        det_size=tuple(_int_tuple(p, "det_size", 2) or (640, 640)),
        use_crop=_bool(p, "use_crop"))
    margin = _float(p, "margin", 0.0) or 0.15
    dedup_iou = _float(p, "dedup_iou", 0.0) or 0.6
    return FaceStage(fdet, Cropper(upscale=1.0, margin=margin), dedup_iou=dedup_iou)


def _build_video_embed_stage(p: dict, video: str = "video"):
    """按参数构造视频嵌入阶段（WEMMClient + EmbedStore + VideoEmbedStage）。

    **惰性/零副作用**（INB-0109 装配期约束）：mock 档（use_real_embed=false 或无端点）
    构造零网络零模型加载；real 档构造也只解析 env/URL（探测与请求推迟到首帧嵌入）。
    key 只走 env（WEMM_API_KEY），不落 preset/DB/日志。维度非法由 WEMMClient
    fail fast（ValueError 含合法档位清单）。

    store_path 已有向量库 → 装配期先 load（WO-0033.12）：运行收尾 save() 是整库覆盖，
    空构造不加载 = last-writer-wins，会把库里其它视频的向量清掉（违背 hybrid_retrieval
    §4「多视频共存 + (video,t_start) upsert」契约）。对齐 FaceStore 先例
    （h_face_store：db.exists() → store.load）。已载库的 clip_dim/frame_dim 与本次
    preset 配置不一致 → 装配期明确报错（禁止静默混维，运行中才炸难定位）。
    """
    from video_embed.wemm_client import WEMMClient
    from video_embed.embed_store import EmbedStore
    from video_embed.video_embed_stage import VideoEmbedStage
    client = WEMMClient(
        endpoint=_str_param(p, "embed_endpoint") or None,
        model=_str_param(p, "embed_model") or None,
        timeout=_float(p, "embed_timeout", 120.0) or 120.0,
        clip_dimension=_int(p, "embed_dimension", 1024),
        frame_dimension=_int(p, "frame_dimension", 256),
        max_resolution=_str_param(p, "max_resolution") or "896,896",
        jpeg_quality=_int(p, "jpeg_quality", 85),
        use_real_embed=_bool(p, "use_real_embed"))
    store = EmbedStore()
    # store_path 归一（与运行收尾 save 点 L~3232/4171 同规则；补 .npz 后缀对齐
    # EmbedStore.save/load 的内部补全，避免「存 xxx.npz、查 xxx」漏检）
    _sp = Path(_str_param(p, "store_path") or "out/video_vecs.npz")
    if not _sp.is_absolute():
        _sp = ROOT / _sp
    if _sp.suffix != ".npz":
        _sp = _sp.with_name(_sp.name + ".npz")
    if _sp.exists():
        store.load(str(_sp))
        for _what, _lib, _cfg in (("clip", store.clip_dim, client.clip_dimension),
                                  ("frame", store.frame_dim, client.frame_dimension)):
            if _lib is not None and _cfg is not None and int(_lib) != int(_cfg):
                raise ValueError(
                    f"video_embed 装配失败: 已有向量库 {_sp} 的 {_what} 维度为 {_lib}，"
                    f"与本次配置 {_cfg} 不一致（入库与查询必须同维度，禁止混维）——"
                    f"请换 store_path 或删除旧库重建")
    # thumb_dir 相对路径 → 项目根（server 进程 CWD=web_lab，不归一会落进 web_lab/out/；
    # 对齐 face_store db_path 的归一约定。构造期纯字符串处理，零副作用）
    thumb_dir = _str_param(p, "thumb_dir") or None
    if thumb_dir and not Path(thumb_dir).is_absolute():
        thumb_dir = str(ROOT / thumb_dir)
    # flush_partial 缺省=True（PARAMS 默认档；显式空值回落默认，False 才关）
    _fp = p.get("flush_partial")
    flush_partial = True if _fp is None or _fp == "" else _bool(p, "flush_partial")
    return VideoEmbedStage(
        client, store, video=str(video),
        frames_per_clip=_int(p, "frames_per_clip", 6),
        clip_seconds=_float(p, "clip_seconds", 5.0) or 5.0,
        thumb_dir=thumb_dir,
        thumb_long=_int(p, "thumb_long", 320),
        flush_partial=flush_partial,
        root=str(ROOT))  # thumb_path 转项目相对的基准（缺省退 CWD=web_lab 会出 ../out/...）


def _build_stage_instance(mid: str, p: dict):
    """按模块类型构造**单个**阶段实例（按位置；同模块多实例各自构造，不再"取第一个"）。

    未知 / 非装配模块返回 None（validate_chain「声明即执行」会报错）。
    """
    if mid == "frame_manager":
        from pipe.composer import FrameManager, SAMPLING_ANALYSIS, backpressure_params
        return FrameManager(
            sampling=_str_param(p, "sampling", SAMPLING_ANALYSIS),
            frame_skip=_int(p, "frame_skip", 3),
            interval_sec=_float(p, "interval_sec", 3.0),
            interval_frames=_int(p, "interval_frames", 30),
            **backpressure_params(_str_param(p, "backpressure", "standard")))
    if mid == "yolo":
        from pipe.composer import YOLODetector
        model = _find_file(_str_param(p, "model_path"), [MODELS_DIR]) or MODELS_DIR / "yolov8n.pt"
        return YOLODetector(
            model_path=str(model), conf=_float(p, "conf", 0.35),
            iou=_float(p, "iou", 0.7), imgsz=_int(p, "imgsz", 640),
            max_det=_int(p, "max_det", 300), classes=_int_list(p, "classes", None),
            device=(_str_param(p, "device") or None))
    if mid == "vlm":
        from pipe.composer import VLMAnalyzer
        return VLMAnalyzer(
            max_resolution=tuple(_int_tuple(p, "max_resolution", 2) or (1280, 720)),
            prompt=_str_param(p, "prompt"))
    if mid == "det_filter":
        return _det_filter_of(p)
    if mid == "seq_filter":
        return _seq_filter_of(p, what="gates")
    if mid == "gate_filter":
        from pipe.gate_filter import GateFilter
        return GateFilter(key=(_str_param(p, "key") or "track_id"),
                          interval=_float(p, "interval", 5.0),
                          filter=_str_param(p, "filter"))
    if mid == "target_crop":
        from pipe.target_crop import TargetCrop
        return TargetCrop(margin=_float(p, "margin", 0.0),
                          classes=_int_list(p, "classes", None),
                          filter=_str_param(p, "filter"),
                          out_size=_str_param(p, "out_size"))
    if mid == "annotate":
        from pipe.annotate import Annotator
        return Annotator(thickness=_int(p, "thickness", 2),
                         show_label=_bool(p, "show_label"),
                         classes=_int_list(p, "classes", None),
                         filter=_str_param(p, "filter"),
                         scope=_str_param(p, "scope") or "frame",
                         max_records=_int(p, "max_records", 32))
    if mid in ("face_detect", "extract_faces"):
        return _build_face_stage(p)
    if mid == "plate_recog":
        from plate_recog.plate_recognizer import PlateRecognizer
        from plate_recog.plate_pipeline import PlateStage
        from pipe.cropper import Cropper
        tool = Cropper(upscale=1.0, margin=_float(p, "margin", 0.1))
        td = p.get("track_dedup")
        return PlateStage(
            PlateRecognizer(detect_level=_str_param(p, "detect_level", "high")),
            tool, mode=_str_param(p, "mode", "crop"),
            dedup_iou=_float(p, "dedup_iou", 0.6),
            track_dedup=(True if td in (None, "") else _bool(p, "track_dedup")),
            track_cooldown=_float(p, "track_cooldown", 5.0))
    if mid == "face_embed":
        from face_embed.face_embedder import FaceEmbedder
        return FaceEmbedder(device=_str_param(p, "device", "auto"))
    if mid == "video_embed":
        return _build_video_embed_stage(p, video=(_str_param(p, "video") or "video"))
    return None


def _chain_from_spec(spec: dict):
    """按链路 spec 的 stages 组装 Pipeline（模块链路的装配器）。

    节点实例身份（P2-0，设计稿 §3.1）：position = **stages 数组下标**；同模块多实例各自
    构造（不再"取第一个"静默丢弃），装入 Pipeline（容器 key=位置）。本函数面向真实执行路径，
    不依赖 DB —— 必须能只凭 stages 数组装配，故装配能力声明在此由代码给出、
    而非从 DB 读取。

    端口连线：stage 的 "inputs"（端口→上游产物名）解析为 `Pipeline.stage_inputs`，键为
    位置串 "#<position>"、值为**产物全名** `{module}#{position}.{artifact}`
    （短名若唯一实例经 resolve_artifact 归一，多实例歧义报错）。spec 声明的每个阶段都必须
    在此被装配进执行路径（validate_chain 的「声明即执行」断言依据 CHAIN_ASSEMBLED_MODULES）。

    YOLO 惰性不加载；含 face_detect / face_embed 阶段时会构造人脸阶段
    （FaceDetector/FaceEmbedder 构造即加载模型）。
    """
    from pipe.composer import (
        FrameManager, YOLODetector, VLMAnalyzer, Pipeline,
        SAMPLING_ANALYSIS, SAMPLING_WALL_CLOCK, SAMPLING_FRAME_COUNT,
        backpressure_params,
    )
    stages = spec.get("stages") or []
    module_ids = [s.get("module_id") for s in stages]
    ids = {m for m in module_ids if m}
    positions_by_module = _positions_by_module(module_ids)

    def _params_at(mid: str) -> dict:
        return next((s.get("params") or {} for s in stages
                     if s.get("module_id") == mid), {})

    if "frame_manager" not in ids:
        raise ValueError("链路缺少必需阶段: frame_manager")
    if not any(k in ids for k in ("face_embed", "face_detect", "plate_recog",
                                  "det_filter", "seq_filter", "video_embed")):
        raise ValueError("链路缺少必需阶段: 终段需为 det_filter / seq_filter / "
                         "face_embed / face_detect / plate_recog / video_embed 之一")

    # ---- 每阶段按位置构造实例（同模块多实例各自构造，装配进 Pipeline 容器）----
    instances: dict = {}
    for i, s in enumerate(stages):
        mid = s.get("module_id")
        if not mid:
            continue
        obj = _build_stage_instance(mid, s.get("params") or {})
        if obj is not None:
            instances[i] = obj

    def _first(mid: str):
        """该模块**第一个**实例（兼容 Pipeline 单槽字段；多实例时另有容器键）。"""
        return next((instances[i] for i, m in enumerate(module_ids) if m == mid), None)

    fm = _first("frame_manager")
    yolo = _first("yolo")
    vlm = _first("vlm")
    det_filter = _first("det_filter")
    gate_filter = _first("gate_filter")
    crop = _first("target_crop")
    annotate = _first("annotate")
    plate = _first("plate_recog")
    face = _first("face_detect") or _first("extract_faces")

    # 时序判定阶段（一个 module_id）：参数按用途拆两个实例——
    # gates（throttle/change）→ 检测流门控 Pipeline.seq_filter；
    # rules（SeqRule 规格）→ 事件尾 Pipeline.event_filter（VLM/车牌记录经 publish 走它出事件）。
    seq_filter = None
    event_filter = None
    if "seq_filter" in ids:
        sp = _params_at("seq_filter")
        seq_filter = _seq_filter_of(sp, what="gates")
        event_filter = _seq_filter_of(sp, what="rules")

    # face_embed 身份去重（驱动层读 last_faces 后处理）；人脸去重阈值同源。
    face_embedder = _first("face_embed")
    face_dedup_thresh = _float(_params_at("face_embed"), "thresh", 0.55) or 0.55

    # video_embed（PIPE0004 向量档索引）：驱动层逐帧 run/finish 消费，
    # 片段记录经 publish 进事件流（clip_indexed 事件）；非 dataclass 字段附挂
    video_embed_stage = _first("video_embed")

    # 兼容旧 extract_faces 形态：链路未配「目标裁剪」/「YOLO」阶段时按 extract_faces 参数补齐
    if face is not None and "extract_faces" in ids:
        edp = _params_at("extract_faces")
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

    # ---- 端口连线：inputs 内联在各 stage；键 = 该阶段下标 → stage_inputs 键 = "#<position>" ----
    # 值归一为产物全名（短名唯一实例兼容、多实例歧义报错）。非字符串/空防御性跳过；
    # 非法连线由 validate_chain 报错。
    stage_inputs: dict = {}
    for i, s in enumerate(stages):
        ins = s.get("inputs")
        if not isinstance(ins, dict) or not ins:
            continue
        clean: dict = {}
        for port, art in ins.items():
            if not isinstance(port, str) or not isinstance(art, str) or not art.strip():
                continue
            clean[port] = resolve_artifact(art, positions_by_module)
        if clean:
            stage_inputs[f"#{i}"] = clean

    # ---- 按位置装配同模块多实例（P2-0）：多实例列表 [(position, obj), ...] ----
    # composer.Pipeline.step() 按这些列表**位置升序链式**执行各模块实例；单插槽字段保留
    # （取第一实例）兼容旧读法与仅支持单实例的模块（yolo/vlm）。
    _PLURAL = {"det_filter": [], "seq_filter": [], "gate_filter": [], "target_crop": [],
               "annotate": [], "plate_recog": [], "face_detect": []}
    for i, s in enumerate(stages):
        mid = s.get("module_id")
        obj = instances.get(i)
        if obj is not None and mid in _PLURAL:
            _PLURAL[mid].append((i, obj))

    # stage_positions：{module_id: 位置(int)}——单插槽实例的位置声明（step 据此登记全名/解析
    # "#<position>" 连线）；多实例模块的位置由其多实例列表承载。
    stage_positions = {mid: poss[0] for mid, poss in positions_by_module.items()}

    pipe_kwargs = dict(
        frame=fm, yolo=yolo, vlm=vlm,
        crop=crop, annotate=annotate, plate=plate, face=face,
        det_filter=det_filter, seq_filter=seq_filter, event_filter=event_filter,
        gate_filter=gate_filter,
        det_filters=_PLURAL["det_filter"], seq_filters=_PLURAL["seq_filter"],
        gate_filters=_PLURAL["gate_filter"],
        target_crops=_PLURAL["target_crop"], annotators=_PLURAL["annotate"],
        plate_stages=_PLURAL["plate_recog"], face_stages=_PLURAL["face_detect"],
        stage_positions=stage_positions, stage_inputs=stage_inputs,
    )
    if Pipeline.__dataclass_fields__ and "stages" in Pipeline.__dataclass_fields__:
        # 按位置装配的实例容器（key=stage 位置下标），声明式携带（step 现由多实例列表驱动）。
        pipe_kwargs["stages"] = instances
    pipe = Pipeline(**{k: v for k, v in pipe_kwargs.items()
                       if k in Pipeline.__dataclass_fields__})
    # face_embed 身份去重所需（驱动层读 pipe.last_faces 后处理）：非 dataclass 字段，装配层附挂
    pipe.face_embedder = face_embedder
    pipe.face_dedup_thresh = face_dedup_thresh
    # video_embed（PIPE0004）：驱动层逐帧喂帧 + 收尾冲刷/落盘（face_embedder 同款附挂方式）
    pipe.video_embed_stage = video_embed_stage
    # 完整位置映射（供驱动层按位置读 stage_inputs，如 _apply_vlm_inputs；调试同样有用）
    pipe.module_positions = positions_by_module
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

    # ---- 链路尾输出声明（WO-0029.2）：fail-soft 同步路径；流式不落盘 → items.images 恒空
    out_plan, out_err = _resolve_output_plan(spec)
    if out_plan is not None and out_plan.get("from_deferred"):
        # 同步路径：事件尾在位 = publish 通路在位（空结果「没有」≠「取不到」，WO-0029.5）
        out_plan["publish_wired"] = getattr(pipe, "event_filter", None) is not None
    out_acc = _make_output_accumulator(out_plan) if out_plan else None

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
    # 视频嵌入（PIPE0004）：驱动层逐帧 run（含观察源 dets 端口），收尾 finish + 底库落盘
    vem = getattr(pipe, "video_embed_stage", None)
    vem_pos = next((i for i, s in enumerate(spec.get("stages") or [])
                    if s.get("module_id") == "video_embed"), -1)  # dets 端口连线解析位
    vem_params = next((s.get("params") or {} for s in spec.get("stages") or []
                       if s.get("module_id") == "video_embed"), {})
    vem_store_path = str(vem_params.get("store_path") or "").strip()
    if vem is not None and str(vem.video) in ("", "video"):
        vem.video = vid.name  # upsert 键 (video, t_start) 用真实文件名
    tl_stream: list = []  # 轻量时间线（车牌/事件条目带 data，报告导出用；流式不落盘）
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
            # WO-0033.9：同 h_pipeline_run——保留 infer() 原始 dict 记录（带 label）
            # 喂管道，SSE 路径 video_embed 落表 class_names 才有值；下方 write("frame")
            # 叠加快照处对象形态消费点局部转换（本路径 dets 仅 step/快照两用）。
            dets = list(pipe.yolo.infer(frame, track=pipe.track_needed())) \
                if pipe.yolo is not None else []
            subs, evs = pipe.step(n, ts, dets, frame=frame, force=True)
            subs = _apply_vlm_inputs(pipe, subs)
            if vlm_feedback and pipe.vlm is not None:
                for s in subs:  # 模拟回填（真实 VLM 见函数说明）
                    evs += pipe.publish([{"kind": "vlm", "label": vlm_action, "text": ""}], ts, n)
            if vem is not None:  # 视频嵌入：满窗片段记录经 publish 进事件流（叠加侧 alarms）
                evs += pipe.publish(_vem_feed_dets(pipe, vem, vem_pos, frame, ts, n),
                                    ts, n)
            if out_acc is not None:
                # 链路尾输出：流式不落盘 → 配图记录不累积（配额用占位，images 恒空）。
                # 置于 VLM publish **之后**：publish 登记的 pipeline.tail 承载记录须本帧
                # 取（产物表逐帧重建，publish 前读会漏本帧事件，WO-0029.5）；缺省 from
                # 同样逐帧运行时裁决（_output_pick_from），空块解释收尾统一给。
                _of = _output_pick_from(out_plan, pipe.artifacts)
                _os_recs = pipe.artifacts.get(_of) if _of else None
                if isinstance(_os_recs, list):
                    out_acc["add"](_os_recs, None)
            plates = list(getattr(pipe, "last_plates", []))
            processed += 1
            detection_count += len(dets or [])
            plate_count += len(plates)
            event_total += len(evs)
            # 轻量时间线：只记带结构化数据的条目（车牌/事件），报告导出消费
            for _pl in plates:
                tl_stream.append({
                    "kind": "submit", "stage": "plate_recog",
                    "stage_label": _stage_label("plate_recog"),
                    "text": f"识别 {_pl.plate_no} {_pl.type_name}/{_pl.color}",
                    "ts": ts, "frame": n, "data": _pl.to_dict()})
            for _ev in evs:
                tl_stream.append({
                    "kind": "submit", "stage": "seq_filter",
                    "stage_label": _stage_label("seq_filter"),
                    "text": f"事件 {_ev.get('name', '事件')}",
                    "ts": ts, "frame": n, "data": _ev})
            _gi = getattr(pipe, "last_gate", None)  # 门控统计旁路（叠加展示；确有拦截才带）
            write("frame", {
                "n": n, "ts": round(ts, 3),
                "dets": [{"cls": d.cls_id, "bbox": list(d.bbox),
                          "score": round(float(d.score), 3),
                          "track": int(getattr(d, "track_id", 0) or 0)}
                         for d in _as_detections(dets or [])],  # 快照消费点局部转换（WO-0033.9）
                "plates": [{"plate_no": p.plate_no, "type_name": p.type_name, "color": p.color,
                            "score": round(float(p.score), 3), "bbox": list(p.bbox),
                            "car_bbox": list(p.car_bbox) if p.car_bbox else None}
                           for p in plates],
                "alarms": [{"action": e.get("name", "事件"), "desc": str(e.get("detail") or ""),
                            "conf": 1.0} for e in evs],
                "gate": ({"passed": _gi[1], "blocked": _gi[0] - _gi[1]}
                         if _gi and _gi[0] != _gi[1] else None),
            })
            n += 1
    finally:
        cap.release()
        _STREAMS.pop(run_id, None)
    # 视频嵌入收尾：冲刷尾段（partial）publish + 底库落盘（事件计入 done.alarms 与报告时间线）
    if vem is not None:
        _tail_ts = (n / fps) if fps else 0.0
        for _ev in pipe.publish(vem.finish(), _tail_ts, n):
            event_total += 1
            tl_stream.append({"kind": "submit", "stage": "video_embed",
                              "stage_label": _stage_label("video_embed"),
                              "text": f"事件 {_ev.get('name', '事件')}",
                              "ts": _tail_ts, "frame": n, "data": _ev})
        if vem_store_path:
            _p = Path(vem_store_path)
            if not _p.is_absolute():
                _p = ROOT / vem_store_path
            vem.store.save(str(_p))
    # 链路尾输出提炼块（WO-0029.2）：流式无落盘 → items.images 恒空 + note（与快照一致）
    out_block = None
    if out_plan is not None:
        out_block = _finish_output(out_plan, out_acc, None,
                                   note="流式运行不落盘，提炼项不产出配图（需配图请用批量运行）")
    elif out_err:
        out_block = {"error": out_err}
    # 运行快照（报告导出）：流式无落盘图片，时间线仅车牌/事件条目
    _store_last_run({
        "name": spec.get("name") or "未命名",
        "pipeline_id": pipeline_id,
        "stages": [s.get("module_id") for s in spec.get("stages") or []],
        "media": {"type": "video", "name": vid.name, "start_time": start_time,
                  "end_time": end_time, "start_frame": start_frame,
                  "end_frame": end_frame},
        "stats": {"processed": processed, "detections": detection_count,
                  "plates": plate_count, "events": event_total,
                  "vlm_state": (f"模拟回填({vlm_action})"
                                if (vlm_feedback and pipe.vlm is not None)
                                else "未开启研判回填"),
                  "vlm_fail": 0, "saved_images": 0, "dropped_saves": 0,
                  "elapsed_sec": round(time.time() - t0, 2)},
        "timeline": tl_stream,
        "output_dir": None,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "stream",
        **({"output": out_block} if out_block is not None else {}),
    })
    # 事件留存归档（WO-0033 §3，契约 C；流式路径与同步路径同语义）：
    # 只归档链路尾事件；含视频嵌入阶段时顺带标签对齐落表。失败不影响 done 推送。
    try:
        _archive_and_align(vid.name, pipeline_id, _tail_events_of_timeline(tl_stream),
                           run_id, vem, vem_params, store_path=vem_store_path)
    except Exception as e:
        print(f"[events] 流式归档/对齐异常（不阻断推送）: {e}")
    done_payload = {"run_id": run_id, "processed": processed,
                    "detections": detection_count, "plates": plate_count,
                    "alarms": event_total, "elapsed": round(time.time() - t0, 2),
                    "stopped": stop_evt.is_set()}
    if out_block is not None:
        done_payload["output"] = out_block
    write("done", done_payload)


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

    def _step_out(frame_no, ts, dets, subs, al, frame=None):
        """逐模块展示本帧处理内容（不管是否向下传递），每条带帧序数。"""
        if "frame_manager" in _so:
            log.append(_tag("frame_manager") + f"帧#{frame_no} 放行 ts={ts:.2f}s")
        if "det_filter" in _so:
            fi = getattr(pipe, "last_filter", None)
            if fi is not None and fi[0] != fi[1]:
                log.append("    " + _tag("det_filter")
                           + f"帧#{frame_no} 有效性筛选 {fi[0]} → {fi[1]}（筛掉 {fi[0] - fi[1]} 条）")
        if "gate_filter" in _so:
            gi = getattr(pipe, "last_gate", None)
            if gi is not None and gi[0] != gi[1]:
                log.append("    " + _tag("gate_filter")
                           + f"帧#{frame_no} 门控 放行 {gi[1]} / 拦截 {gi[0] - gi[1]}"
                           + _gate_stats_text(pipe))
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
        # 标注：本帧生成带框图；record 模式逐实例展示出图张数/截断告警（WO-0022，
        # 此前本 runner 无标注展示；脚本报文无原图帧 → 标注两种模式都不执行）
        if "annotate" in _so:
            _anno_recs, _anno_has_frame, _anno_ran = _annotate_stats(pipe, dets, frame)
            if not _anno_ran:
                _why = "无原图帧，未处理" if frame is None else "无检出输入，未处理"
                log.append("    " + _tag("annotate") + f"帧#{frame_no} {_why}")
            else:
                for _drawn, _cap, _dropped in _anno_recs:
                    log.append("    " + _tag("annotate")
                               + f"帧#{frame_no} 逐记录出图 {_drawn} 张（上限 {_cap}）")
                    if _dropped:
                        log.append("    " + _tag("annotate")
                                   + f"帧#{frame_no} 带图记录超上限 {_cap}，截断 {_dropped} 条"
                                     "（可调高 annotate.max_records 或收紧上游筛选）")
                if _anno_has_frame:
                    _vis = getattr(pipe, "last_annotated", None)
                    log.append("    " + _tag("annotate")
                               + (f"帧#{frame_no} 已生成标注图" if _vis is not None
                                  else f"帧#{frame_no} 无检出，未生成标注图"))
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
                dets = _as_detections(pipe.yolo.infer(frame, track=pipe.track_needed()))
            subs, al = pipe.step(n, ts, dets, frame=frame, force=True)
            subs = _apply_vlm_inputs(pipe, subs)
            total_det += len(dets or [])
            total_plate += len(getattr(pipe, "last_plates", []))
            _step_out(n, ts, dets, subs, al, frame=frame)
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
        dets = _as_detections(pipe.yolo.infer(frame, track=pipe.track_needed())) if pipe.yolo else None
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
        _step_out(0, 0.0, dets, subs, al, frame=frame)
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
                          "\"key\":\"label\",\"cooldown\":30}]；分组键 key 支持逗号组合字段"
                          "（如 \"track_id,label\"），按序取第一个非空（非 0/None）字段值做分组键，"
                          "全部无效归匿名组 *；event 支持 {字段} 动态占位，"
                          "cooldown<0=同组一次，window_n/min_n/hit_ratio 为窗口；空=无规则"},
        "detections": {"label": "模拟检测（每行一条）", "type": "text", "default": "",
                       "test_only": True,
                       "help": "每行: 类别,track,置信度,x1,y1,x2,y2（如 '0,1,0.9,100,100,200,200'）"},
        "ts": {"label": "帧时间戳（秒）", "type": "float", "default": 0.0,
               "test_only": True,
               "help": "门控/规则计时用（同一帧多次调用可改 ts 观察节流）"},
    },
    "gate_filter": {
        "key": {"label": "分组键字段 key", "type": "str", "default": "track_id",
                "help": "按该记录字段分桶冷却（track_id/cls_id/label 等，可用派生字段 w/h/short/long/area）；"
                        "取值 None/空/0 归匿名桶 *（统一受门控）；key=cls_id 时 COCO 类 0（person）也落匿名桶"},
        "interval": {"label": "放行间隔 interval（秒，三值）", "type": "float", "default": 5.0,
                     "help": ">0=同 key 每 N 秒最多放行一次（首见放行）；<0=同 key 只放行一次；"
                             "0=全放行（no-op）"},
        "filter": {"label": "门控范围（条件 DSL，空=全部）", "type": "text", "default": "",
                   "help": "只对命中记录做门控（如 'cls_id == 2'）；未命中记录直接放行且不耗该桶冷却额度；"
                           "语法同检测筛选；空=全部记录受门控"},
        "detections": {"label": "模拟检测（每行一条）", "type": "text", "default": "",
                       "test_only": True,
                       "help": "每行: 类别,track,置信度,x1,y1,x2,y2（如 '2,1,0.9,100,100,200,200'）"},
        "ts": {"label": "帧时间戳（秒）", "type": "float", "default": 0.0,
               "test_only": True,
               "help": "门控计时用（改 ts 重跑观察冷却窗口）"},
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
                               "（OpenAI 兼容多图消息，已由 glm-4v-flash 实证支持；"
                               "第三方端点建议自查多图支持）"},
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
        "vlm_provider": {"label": "提供方", "type": "select", "default": "qwen_selfhost",
                         "options": [{"value": k, "label": v["label"]}
                                     for k, v in VLM_PROVIDERS.items()],
                         "help": "选中即覆盖下方手填端点/模型；环境变量优先于本参数"},
        "vlm_endpoint": {"label": "VLM 端点（留空用默认）", "type": "str",
                         "default": "http://117.42.21.253:8000/v1/chat/completions",
                         "help": "大模型接口地址；环境变量 VLM_ENDPOINT 优先（导出即全链路切换）"},
        "vlm_model": {"label": "VLM 模型名", "type": "str", "default": "qwen3-vl-32b",
                      "help": "调用的模型名称；环境变量 VLM_MODEL 优先"},
        "vlm_key": {"label": "API Key（无 key 留空）", "type": "str", "default": "",
                    "help": "接口密钥，免鉴权可留空；环境变量 VLM_API_KEY（兼容 GLM_API_KEY）优先，"
                            "此处填写的 key 会入库，建议改用环境变量"},
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
    "video_embed": {
        # ---- 嵌入端点（wemm_integration.md §4.5 参数表；key 只走 env WEMM_API_KEY，勿入库）----
        "embed_endpoint": {"label": "嵌入端点（/v1/embeddings）", "type": "str", "default": "",
                           "help": "OpenAI 兼容 /v1/embeddings 地址（base 地址自动拼路径）；"
                                   "环境变量 WEMM_BASE_URL（别名 WEMM_ENDPOINT）优先；"
                                   "留空且未导出 env = mock 档，无端点也能跑全流程"},
        "embed_model": {"label": "嵌入模型名", "type": "str", "default": "",
                        "help": "服务端 /v1/models 返回的模型名（env WEMM_MODEL 优先，默认 wemm-embedding）"},
        "embed_dimension": {"label": "clip 向量维度", "type": "int", "default": 1024,
                            "help": "片段向量维度（Matryoshka 档位 64~2048，默认 1024）；"
                                    "入库与查询必须同维度，换维度=新建库"},
        "frame_dimension": {"label": "frame 向量维度", "type": "int", "default": 256,
                            "help": "帧向量维度（精排只比相对远近，低维足够，默认 256）"},
        "use_real_embed": {"label": "真实嵌入调用", "type": "bool", "default": False,
                           "help": "true=真发 HTTP（需端点在线，无端点直接报错）；"
                                   "false=确定性伪向量（mock 档，零网络零模型，服务不在线也能全流程）"},
        "embed_timeout": {"label": "读取超时（秒）", "type": "int", "default": 120,
                          "help": "含图请求较慢，默认 120；连接阶段恒 5s 快速失败"},
        "max_resolution": {"label": "送审帧缩放上限（宽,高）", "type": "str", "default": "896,896",
                           "help": "嵌入不需原分辨率，压带宽；跨公网可降到 640,640"},
        "jpeg_quality": {"label": "帧 JPEG 质量", "type": "int", "default": 85,
                         "help": "帧编码质量（1~100）"},
        # ---- 片段窗口 / 底库 / 缩略图（video_review_search.md §3.3/§5）----
        "frames_per_clip": {"label": "每片段帧数", "type": "int", "default": 6,
                            "help": "片段窗口切分唯一依据（默认 6 帧；时间由帧 ts 承载）"},
        "clip_seconds": {"label": "标称窗口秒数", "type": "float", "default": 5.0,
                         "help": "仅写入片段元数据供展示/排查，不影响切分（默认 5s 窗）"},
        "thumb_dir": {"label": "缩略图目录（空=不落盘）", "type": "str",
                      "default": "out/video_thumbs",
                      "help": "窗口首帧缩略图落盘目录；路径写记录与底库元数据"},
        "thumb_long": {"label": "缩略图长边（像素）", "type": "int", "default": 320,
                       "help": "等比缩放、只缩不放（默认 320）"},
        "flush_partial": {"label": "尾段余帧出片段", "type": "bool", "default": True,
                          "help": "true=结束时不足一窗的余帧也出一个 partial 片段；false=丢弃"},
        "store_path": {"label": "向量底库路径（.npz）", "type": "str",
                       "default": "out/video_vecs.npz",
                       "help": "片段向量底库存到哪（npz+json）；同视频重跑按 (video,t_start) 幂等覆盖"},
        # ---- 测试页脚手架（test_only，不落库、不进链路）----
        "video": {"label": "测试视频（试跑索引用）", "type": "file", "src": "vid",
                  "test_only": True,
                  "help": "模块试跑：从下拉选视频，按标称节奏取样喂入片段窗口（上限 60 帧≈10 片段）"},
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
        "scope": {"label": "标注出图方式", "type": "select", "default": "frame",
                  "options": [{"value": "frame", "label": "整帧一张"},
                              {"value": "record", "label": "逐记录出图"}],
                  "help": "整帧一张：本帧所有框画在同一张图上（默认，原有行为）；"
                          "逐记录出图：每个目标单独一张整帧图、只画自己的框，"
                          "下游阶段连线取 annotate.annotated_records 才能收到这些带图记录"},
        "max_records": {"label": "单帧出图上限（逐记录模式）", "type": "int", "default": 32,
                        "show_if": {"key": "scope", "eq": "record"},
                        "help": "逐记录出图时单帧最多出多少条带图记录，超出截断并写告警日志"
                                "（每张整帧图约 2.8MB@720p，32 张约 88MB 内存峰值）"},
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

# 模块清单：定义顺序 = 业务数据流序（/api/schema 输出序即前端展示序，组内序同此）。
# 可见模块五组口径（WO-0017）：帧管理 → 图像旁路 → 识别研判 → 筛选判定 → 向量比对。
MODULES = {
    "frame_manager": {
        "name": "帧管理 FrameManager", "group": "帧管理",
        "desc": "复用 pipe.composer.FrameManager：三种取图节奏 + 积压/耗时背压（原 frame_scheduler 并入）。",
        "params": PARAMS["frame_manager"], "handler": h_frame_manager},
    "target_crop": {
        "name": "目标裁剪 TargetCrop", "group": "图像旁路",
        "desc": "按上游检测框裁子图（输出分辨率可配，默认原分辨率）；"
                "供车牌识别/人脸检测/VLM 消费。不含检测器。",
        "params": PARAMS["target_crop"], "handler": h_target_crop},
    "annotate": {
        "name": "标注 Annotator", "group": "图像旁路",
        "desc": "在帧副本上画上游检测框（供 VLM 看“原图+画框”）；纯绘制，无模型。",
        "params": PARAMS["annotate"], "handler": h_annotate},
    "yolo": {
        "name": "小模型识别 YOLODetector", "group": "识别研判",
        "desc": "复用 pipe.composer.YOLODetector：**纯推理**（有效性过滤归「检测筛选」阶段，节流/报送编排在 Pipeline）。",
        "params": PARAMS["yolo"], "handler": h_yolo},
    "face_detect": {
        "name": "人脸检测 FaceDetector", "group": "识别研判",
        "desc": "复用 face_detect.face_detector：SCRFD 检测 + 5 点对齐。",
        "params": PARAMS["face_detect"], "handler": h_face_detect},
    "plate_recog": {
        "name": "车牌识别 PlateRecognizer", "group": "识别研判",
        "desc": "复用 plate_recog.plate_recognizer：HyperLPR3 在车辆子图里裁车牌 + 识别；"
                "车辆子图由上游「目标裁剪」提供，或整帧直读。",
        "params": PARAMS["plate_recog"], "handler": h_plate_recog},
    "vlm": {
        "name": "大模型研判 VLMAnalyzer", "group": "识别研判",
        "desc": "复用 pipe.composer.VLMAnalyzer：素材包准备，可接真实 VLM。",
        "params": PARAMS["vlm"], "handler": h_vlm},
    "det_filter": {
        "name": "检测筛选 DetFilter", "group": "筛选判定",
        "desc": "复用 pipe.det_filter.DetFilter：有效性筛选（逐条条件 DSL：范围+与/或/非+派生尺寸量"
                "+ 聚合 count(字段)/count(子条件)/count()；+ 成组数量判定）。"
                "链路里插在 YOLO 之后；有效性过滤的唯一来源（空条件+空上下限=不筛选）。",
        "params": PARAMS["det_filter"], "handler": h_det_filter},
    "seq_filter": {
        "name": "时序判定 SeqFilter", "group": "筛选判定",
        "desc": "复用 pipe.seq_filter.SeqFilter：时序门控（整类节流 throttle / track 集合变化 change）"
                "+ 窗口/单条规则确认与事件标注。链路里插在检测筛选之后；"
                "承载原 Pipeline 的 check_interval/track_change_only 语义。",
        "params": PARAMS["seq_filter"], "handler": h_seq_filter},
    "gate_filter": {
        "name": "门控 GateFilter", "group": "门控",
        "desc": "复用 pipe.gate_filter.GateFilter：按 key+时间间隔放行（物理过滤：被拦记录不传下游、"
                "零标记字段；None/空/0 归匿名桶 * 统一受门控）。链路里插在时序判定之后、目标裁剪之前，"
                "是检测流最后一道闸；拦截可观测走统计旁路（Pipeline.last_gate / 产物 gate_filter.stats）。",
        "params": PARAMS["gate_filter"], "handler": h_gate_filter},
    "face_embed": {
        "name": "特征提取 FaceEmbedder", "group": "向量比对",
        "desc": "复用 face_embed.face_embedder：对齐裁剪 → 512 维向量。",
        "params": PARAMS["face_embed"], "handler": h_face_embed},
    "face_store": {
        "name": "向量底库 FaceStore", "group": "向量比对",
        "desc": "复用 face_embed.face_store：注册 / 检索 / 自检 / 持久化。",
        "params": PARAMS["face_store"], "handler": h_face_store},
    "video_embed": {
        "name": "视频嵌入 VideoEmbedStage", "group": "向量比对",
        "desc": "PIPE0004「视频向量化-嵌入查询」核心阶段：片段聚帧 → WeMM 嵌入"
                "（clip+frame 双层向量）→ 底库入库（入库即目的，无 YOLO 无告警）。"
                "mock 档（use_real_embed=false 或无端点）零网络零模型，全流程可跑。",
        "params": PARAMS["video_embed"], "handler": h_video_embed},
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
        # ---- 回顾检索·查询层（向量档，WO-0032；轻查询不占 _TEST_LOCK，可与索引并行）----
        if self.path == "/api/review/search":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"})
                return
            t0 = time.time()
            try:
                result = h_review_search(body)
            except FileNotFoundError as e:   # 底库不存在 → 404（可读指引建库）
                self._send(404, json.dumps(
                    {"ok": False, "error": str(e)}, ensure_ascii=False
                ).encode("utf-8"), "application/json; charset=utf-8")
                return
            except ValueError as e:          # 空查询/参数非法/维度非档位/空库 → 400
                self._send(400, json.dumps(
                    {"ok": False, "error": str(e)}, ensure_ascii=False
                ).encode("utf-8"), "application/json; charset=utf-8")
                return
            except Exception as e:           # 其余（如 real 档端点错误）→ 200 + ok:false
                tb = traceback.format_exc()
                print(tb)
                result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "traceback": tb[-1800:]}
            result.setdefault("elapsed_sec", round(time.time() - t0, 2))
            print(f"[review] search {time.time() - t0:.1f}s ok={result.get('ok')}")
            self._json(result)
            return

        # ---- 回顾检索·仅重对齐不重嵌入（WO-0033 §3 定案点；轻操作不占 _TEST_LOCK）----
        if self.path == "/api/review/relabel":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"ok": False, "error": "请求体不是合法 JSON"})
                return
            try:
                result = h_review_relabel(body)
            except FileNotFoundError as e:   # 底库不存在（需从底库取窗口）→ 404
                self._send(404, json.dumps(
                    {"ok": False, "error": str(e)}, ensure_ascii=False
                ).encode("utf-8"), "application/json; charset=utf-8")
                return
            except ValueError as e:          # video 缺失 / 底库无该视频 → 400
                self._send(400, json.dumps(
                    {"ok": False, "error": str(e)}, ensure_ascii=False
                ).encode("utf-8"), "application/json; charset=utf-8")
                return
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "traceback": tb[-1800:]}
            print(f"[review] relabel ok={result.get('ok')} rows={result.get('rows')}")
            self._json(result)
            return

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

        # ---- 链路运行报告导出（自包含单文件 HTML，WO-0028）----
        if self.path == "/api/test/report":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                body = {}
            if not _LAST_RUN:
                self._json({"ok": False,
                            "error": "暂无可导出的运行，请先完成一次链路测试"})
                return
            # 可选参数：各分组图片上限（默认标注帧 60 / 车牌 36 / 人脸 36 / 链路尾输出 60）
            html_doc = _build_report_html(
                _LAST_RUN, max_anno=_int(body, "max_anno", 60),
                max_plate=_int(body, "max_plate", 36), max_face=_int(body, "max_face", 36),
                max_output=_int(body, "max_output", 60))
            out_dir = Path(_LAST_RUN.get("output_dir")
                           or (ROOT / "tests/data/out/test_pipeline"))
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"report_{ts}.html"
            path.write_text(html_doc, encoding="utf-8")
            self._json({"ok": True, "path": str(path), "html": html_doc})
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

# 第二条链路：车牌识别（帧管理 → YOLO 识车 → 检测筛选 → 门控 → 目标裁剪(裁车)
# → 车牌识别(裁车牌+识别) → 事件尾告警；WO-0024 换装独立门控阶段）
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
    # 独立门控（WO-0024 方案A）：按 track_id 每 5 秒放行一次，物理过滤在裁剪/识别之前
    # （tid 缺失的记录归匿名桶 * 统一受门控，杜绝绕过）
    ("gate_filter", "车牌识别·门控", {
        "key": "track_id", "interval": 5.0, "filter": ""}),
    # 目标裁剪：按 YOLO 车辆框裁车放大 2×（供车牌识别在车图里裁车牌）
    ("target_crop", "车牌识别·目标裁剪", {
        "out_size": "×2", "margin": 0.1,
        "classes": "2", "filter": ""}),
    # 车牌识别：从车图裁车牌 + 识别 + 坐标还原 + 原图出原分辨率车牌小图
    # （track_dedup=False：内置去重关，放行节流已由上游门控阶段承担；
    #   track_cooldown 保留作回切开关——置 True 即回旧行为）
    ("plate_recog", "车牌识别·车牌识别", {
        "detect_level": "high", "mode": "crop", "margin": 0.1,
        "dedup_iou": 0.6, "track_dedup": False, "track_cooldown": 5.0}),
    # 事件尾（原「告警策略」inspection 的等价）：车牌记录经 publish 出事件；
    # key=组合键（设计稿 event_dedup_key.md，WO-0026）：tid>0 按 track 分组（同车异号
    # 收敛一条），tid 缺失/0 回退按号（=旧行为），全字段无效归匿名组
    ("seq_filter", "车牌识别·时序判定", {
        "throttle": "", "change": "",
        "rules": json.dumps([{
            "match": "kind == plate", "event": "{label}", "key": "track_id,label", "cooldown": -1.0,
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

# 第四条链路：视频向量化-嵌入查询（帧管理 → YOLO观察 → 检测筛选 → 视频嵌入；
# 观察段定案 doc/design/hybrid_retrieval.md §12-2：YOLO 观察源插 det_filter，
# 只过有效性筛选、不过时序收敛，det_stats 口径 = det_filter 放行后的检出统计；
# 观察记录经 dets 第二入端口进 video_embed，纯产观察不影响索引，入库即目的）
_SEED_VEMB_NAME = "视频向量化-嵌入查询链路"
_SEED_VEMB_STAGES = [
    # 帧管理：稀疏抽帧（设计稿建议 frame_skip≈15；5s 窗 × 6 帧 → 每 15 帧取 1 的节奏下
    # 25fps 视频约 10s 一窗，窗口时间由帧 ts 承载，可按素材密度调整）
    ("frame_manager", "视频向量化·帧管理", {
        "sampling": "analysis", "frame_skip": 15, "interval_sec": 3.0,
        "interval_frames": 30, "backpressure": "standard"}),
    # YOLO 观察源（§12-2）：为 video_embed 的 dets 端口产观察记录（经驱动层
    # _vem_feed_dets 连线）。classes 不限（""=全类别都进 det_stats，观察场景默认
    # 最宽口径；只想统计特定类别改此一词，如 "0,2"）；模型与 conf 用现有播种默认
    ("yolo", "视频向量化·YOLO观察", {
        "model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.6,
        "imgsz": 1280, "max_det": 300, "classes": "", "device": ""}),
    # 检测筛选（§12-2）：观察层质量闸——只过有效性筛选、不过时序收敛。
    # 播种用模块默认（空条件+空上下限=不筛选，det_stats 即 YOLO 原始检出统计）；
    # 素材有明确尺寸下限时可加有效性条件（如 filter="long >= 30"，参考人脸/车牌链路）
    ("det_filter", "视频向量化·检测筛选", {
        "filter": "", "group_limits": ""}),
    # 视频嵌入：默认参数 + use_real_embed=false（mock 档确定性伪向量，无端点也能全流程；
    # 真端点联调时改 use_real_embed=true 并配 embed_endpoint 或导出 WEMM_BASE_URL——
    # key 只走 env WEMM_API_KEY，不入 preset）
    ("video_embed", "视频向量化·视频嵌入", {
        "embed_endpoint": "", "embed_model": "", "embed_dimension": 1024,
        "frame_dimension": 256, "use_real_embed": False, "embed_timeout": 120,
        "max_resolution": "896,896", "jpeg_quality": 85,
        "frames_per_clip": 6, "clip_seconds": 5.0,
        "thumb_dir": "out/video_thumbs", "thumb_long": 320,
        "flush_partial": True, "store_path": "out/video_vecs.npz"}),
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
    # 标注双模式示范（WO-0021）：逐记录出图——每个目标单独一张整帧图只画自己框，
    # 下游连线取 annotate.annotated_records；给设计器用户一个可直接选用的起点
    ("annotate", "标注·逐记录示范", {
        "scope": "record", "max_records": 32,
        "thickness": 2, "show_label": True, "classes": "", "filter": ""}),
    ("plate_recog", "车牌识别·默认", {
        "detect_level": "high", "mode": "crop", "margin": 0.1,
        "dedup_iou": 0.6, "track_dedup": True, "track_cooldown": 5.0}),
]


def _seed_chain(name: str, stages: list, description: str, have: dict,
                converge: dict | None = None,
                stage_inputs: dict | None = None) -> None:
    """幂等播种一条链路：先保各阶段模块 preset，再建链路引用。

    已存在同名链路时：阶段结构一致则原样保留（不动用户改动）；结构变化则按模块复用
    旧 preset（保留其参数）、补建缺失阶段、丢弃下线阶段，重建步骤（save_pipeline 同名
    覆盖）。这样既迁移结构，又不产生重复 preset。

    converge（WO-0024）：{module_id: 强制参数}——结构变更重建时，对被复用的旧 preset
    强制收敛指定参数（其余用户参数保留；定向更新 id 不变）。用于「形态升级后旧参数
    语义反转」的场景：如 PIPE0002 换装独立门控后 plate_recog 的 track_dedup 必须关，
    否则旧库复用 preset 仍 True，生效形态变成「叠加」而非目标形态。

    stage_inputs（WO-0033.5）：链路级连线声明（save_pipeline meta 双写 pipeline_edges，
    位置键）。驱动层解析的端口（如 video_embed.dets）不经 Pipeline.step、设计器默认
    不连线，播种形态要生效观察段必须显式声明（§12-1 单观察源直连）。
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
    if converge and reuse:
        forced = {mid: patch for mid, patch in converge.items() if mid in reuse}
        if forced:
            cur = {p["id"]: p["params"] for p in db.list_presets("module")}
            for mid, patch in forced.items():
                pid = reuse[mid]
                old = cur.get(pid)
                if old is None:
                    print(f"[seed] 警告：模块配置 {pid}（{mid}）参数读取失败，"
                          f"跳过强制收敛（链路「{name}」）")
                    continue
                changed = {k: v for k, v in patch.items() if old.get(k) != v}
                if not changed:
                    continue
                db.save_preset("module", "", mid, {**old, **patch}, preset_id=pid)
                print(f"[seed] 模块配置 {pid}（{mid}）强制收敛 {changed}"
                      f"（链路「{name}」结构变更迁移，其余参数保留）")
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
                                     meta={"description": description,
                                           **({"stage_inputs": stage_inputs}
                                              if stage_inputs else {})})
    print(f"[seed] 链路「{name}」已播种（{pid}）")


# SEQ0002 事件尾 rules 旧默认（组合键升级前，WO-0026）：存量库启动播种时，
# rules 恰好等于此值（JSON 语义等价，键序无关）→ 定向升级为组合键版；
# 用户改过（≠此值）→ 保留不动并提示手工换法
_LEGACY_SEQ_RULES = [{
    "match": "kind == plate", "event": "{label}", "key": "label", "cooldown": -1.0,
}]
_SEQ_GROUP_KEY_NEW = "track_id,label"


def _converge_seq_group_key() -> None:
    """SEQ0002 事件尾分组键定向收敛（设计稿 event_dedup_key.md Q3，WO-0026.1）。

    组合键升级只改 rules 一个词，链路阶段结构未变——`_seed_chain` 对结构一致的
    已存在链路提前 return、不迁移 preset 参数，故单独立此函数：启动播种时按
    链路引用找到车牌链路的 seq_filter preset，按「是否恰好等于旧默认」分派：
      - 等于旧默认（key="label" 那份 JSON，语义等价比较）→ 定向更新为组合键版
        （id 与其余参数不变，打 [seed] 日志）；
      - 已是新键 → 静默（幂等重入）；
      - 用户改过（≠旧默认）→ 保留不动，打一行提示告知手工换法（改 rules 的 key 一词）。
    """
    pl = next((p for p in db.list_pipelines() if p["name"] == _SEED_PLATE_NAME), None)
    if pl is None:
        return
    sid = next((s.get("preset_id") for s in (pl.get("steps") or [])
                if s.get("module_id") == "seq_filter"), None)
    cur = next((p for p in db.list_presets("module") if p["id"] == sid), None) if sid else None
    if cur is None:
        return
    try:
        rules = json.loads(str(cur["params"].get("rules") or ""))
    except (ValueError, TypeError):
        rules = None
    if not isinstance(rules, list):
        rules = None
    if rules != _LEGACY_SEQ_RULES:
        if rules != [{"match": "kind == plate", "event": "{label}",
                      "key": _SEQ_GROUP_KEY_NEW, "cooldown": -1.0}]:
            print(f"[seed] 模块配置 {sid}（seq_filter）rules 非旧默认值（用户已改），"
                  f"保留不动；如需组合键去重请把 rules 的 key 手工改为 "
                  f"\"{_SEQ_GROUP_KEY_NEW}\"（链路「{_SEED_PLATE_NAME}」）")
        return
    new_rules = [{**r, "key": _SEQ_GROUP_KEY_NEW} for r in rules]
    db.save_preset("module", "", "seq_filter",
                   {**cur["params"], "rules": json.dumps(new_rules, ensure_ascii=False)},
                   preset_id=sid)
    print(f"[seed] 模块配置 {sid}（seq_filter）rules key 定向收敛 "
          f"label → {_SEQ_GROUP_KEY_NEW}（组合键升级，其余参数保留）")


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
                    "车牌识别链路：帧管理→YOLO识车→检测筛选→门控→目标裁剪(裁车)"
                    "→车牌识别(HyperLPR3)→事件尾告警"
                    "（模块自定义拼接；独立门控按 track_id 节流放行）", have,
                    converge={"plate_recog": {"track_dedup": False}})
        _converge_seq_group_key()  # 存量 SEQ0002 rules 旧默认 → 组合键（结构未变，_seed_chain 不迁移）
        _seed_chain(_SEED_FACE_NAME, _SEED_FACE_STAGES,
                    "人脸截取链路：帧管理→YOLO识人→原分辨率人脸提取"
                    "（无嵌入/检索/告警，截取落盘即目的）", have)
        # 观察段连线（hybrid_retrieval §12-1 单观察源直连）：video_embed.dets ←
        # 紧邻上游 det_filter.dets（位置键 stage_inputs，随 save_pipeline 双写
        # pipeline_edges；帧主入口 frame 由驱动层逐帧喂入，不经连线声明）。
        # 不声明则驱动层 _vem_feed_dets 取不到观察记录、det_stats 恒空——
        # 观察段结构（§12-2）与连线必须成对播种。
        _vemb_pos = [mid for mid, _n, _p in _SEED_VEMB_STAGES].index("video_embed")
        _vemb_src = _SEED_VEMB_STAGES[_vemb_pos - 1][0]
        _seed_chain(_SEED_VEMB_NAME, _SEED_VEMB_STAGES,
                    "视频向量化-嵌入查询（离线索引，向量档）", have,
                    stage_inputs={f"#{_vemb_pos}": {"dets": f"{_vemb_src}.dets"}})
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
            # 端口能力目录（图编排 P2-1）：代码声明为真相源，启动时幂等 upsert 到
            # module_ports；陈旧端口行删除、声明与库不一致告警。端口表问题不阻塞启动。
            try:
                db.upsert_module_ports(module_ports_catalog(),
                                       warn=lambda msg: print(f"[db] {msg}"))
            except Exception as e:
                print(f"[db] module_ports 同步告警（不阻塞启动）: {e}")
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
