# -*- coding: utf-8 -*-
"""
片段标签对齐纯逻辑（WO-0033.2）——混合检索「标签数据」的采集/对齐层

设计依据：doc/design/hybrid_retrieval.md
  §2.3 对齐规则（事件 ts → 片段窗口，右开左闭）
  §2.4 双层标签模型（观察层 = 记录流降维成帧级统计，不过时序收敛）
  §2.5 标签 schema（冻结字段名 = 未来 review_index 列名）
  §2.6 采集时间（视频级一行，来源优先级 文件名 > tag）
  §12.1 契约 A（本模块 5 个入口的签名/语义）

**纯逻辑**：零网络、零模型、零 IO——可直接单测，web 侧也直接调用。
观察层产物（帧级统计 + `frame_dets` 原始记录流）只在内存 dict 里存在，随 obs
交由调用方落 `review_index`（§2.5 原则 3 第四轮改定：帧级原始记录流落盘，
原「帧级不落盘」废止）；本模块自身不做持久化。

入口（契约 A）:
    parse_video_time(filename, tag="")        -> (video_time, src)   # 视频级采集时间
    clip_windows(frame_ts, frames_per_clip=…) -> [(t_start, t_end)]  # 片段窗口
    aggregate_frame_stats(per_frame)          -> {det_stats, class_names, plate_nos, face_count, frame_dets}
    align_events(events, windows)             -> [event + window 归属]
    build_clip_tags(video, window, obs, events) -> §2.5 全部标签字段（frame_dets 透传）

窗口语义（§2.3）：**右开左闭** `[t_start, t_end)`——ts 恰落在边界时归**下一窗**；
跨窗事件（窗口重叠）按命中窗口**各记一条**，`ts` 保留原值不裁剪。

用法:
    vt, src = parse_video_time("20261029_101500_cam1.mp4")       # ("2026-10-29T10:15:00+08:00", "filename")
    wins = clip_windows([0, 0.6, 1.2, 1.8, 2.4, 3.0], frames_per_clip=3)
    obs = aggregate_frame_stats([{ "ts": 0.0, "dets": [...], "plates": [...], "faces": [...] }])
    rows = align_events([{"name": "斗殴", "kind": "警情", "ts": 1.2}], wins)
    tags = build_clip_tags("x.mp4", wins[0], obs, rows)

依赖: 标准库（无 numpy / 无模型）
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# 默认时区：+08:00（国内采集；tag/文件名带时区后缀时以显式时区为准）
DEFAULT_TZ = timezone(timedelta(hours=8))
# 窗口末端兜底（单帧片段且无法推断帧间隔时用；与 VideoEmbedStage.clip_seconds 缺省一致）
DEFAULT_CLIP_SECONDS = 5.0
# 文件名年份合理域（越界视为解析失败，防 8 位数字 ID 误判）
_YEAR_MIN, _YEAR_MAX = 1990, 2100
# frame_dets 单窗 recs 上限（§2.5 原则 3：>500 截断到 500 并附 truncated 标记；
# 防异常上游把标签行撑爆——正常 ≤10 帧/窗远达不到）
FRAME_DETS_LIMIT = 500

# 日期时间：20261029_101500 / 2026-10-29_10-15-00 / 20261029101500 / 2026-10-29T10:15:00+08:00
_TIME_RE = re.compile(
    r"(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})[T _-]*"
    r"(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})"
    r"\s*(Z|[+-]\d{2}:?\d{2})?(?!\d)")
# 仅日期：20261029 / 2026-10-29（时间按 00:00:00）
_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)")


# ---------------- 契约 A-1：视频采集时间 ----------------

def _parse_tz(text: Optional[str]):
    """时区后缀 → tzinfo；无后缀用默认 +08:00。Z = UTC。"""
    if not text:
        return DEFAULT_TZ
    if text.upper() == "Z":
        return timezone.utc
    sign = -1 if text[0] == "-" else 1
    body = text[1:].replace(":", "")
    hh, mm = int(body[:2]), int(body[2:4])
    return timezone(sign * timedelta(hours=hh, minutes=mm))


def _build_dt(y, mo, d, h=0, mi=0, s=0, tz=None) -> Optional[datetime]:
    """构造 datetime（越界/非法日期 → None，不抛错）。"""
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=tz)
    except (TypeError, ValueError):
        return None
    if not (_YEAR_MIN <= dt.year <= _YEAR_MAX):
        return None
    return dt


def _parse_datetime(raw) -> Optional[datetime]:
    """任意字符串 → datetime（文件名/tag 通用）；解析不出返回 None。"""
    text = str(raw or "").strip()
    if not text:
        return None
    # 文件名只取 basename（目录里的日期不算采集时间）；tag 原样
    name = os.path.basename(text.replace("\\", "/"))
    for candidate in (name, text):
        m = _TIME_RE.search(candidate)
        if m:
            y, mo, d, h, mi, s = m.group(1, 2, 3, 4, 5, 6)
            dt = _build_dt(y, mo, d, h, mi, s, _parse_tz(m.group(7)))
            if dt is not None:
                return dt
        m = _DATE_RE.search(candidate)
        if m:
            y, mo, d = m.group(1, 2, 3)
            dt = _build_dt(y, mo, d, tz=DEFAULT_TZ)
            if dt is not None:
                return dt
    # 兜底：ISO8601 变体（如 2026-10-29T10:15 无秒）
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=DEFAULT_TZ)
        return dt if _YEAR_MIN <= dt.year <= _YEAR_MAX else None
    except (TypeError, ValueError):
        return None


def parse_video_time(filename: str, tag: str = "") -> Tuple[str, str]:
    """视频采集时间：**文件名优先，其次 tag**（§2.6；2026-10-29 定）。

    返回 `(video_time, src)`：
      - 成功：`video_time` = ISO8601 带时区（如 `2026-10-29T10:15:00+08:00`），
        `src` = `"filename"` / `"tag"`；
      - 解析不出：`("", "")`（**不阻塞索引**，时间窗过滤时该视频不参与并 why 注明）。
    覆盖形如 `20261029_101500` / `2026-10-29_10-15-00` / `20261029101500` /
    `2026-10-29T10:15:00+08:00` 的命名；仅日期（`20261029`）按当日 00:00:00。
    **任何异常都吞掉返回 `("", "")`，绝不抛错。**
    """
    try:
        for src, raw in (("filename", filename), ("tag", tag)):
            try:
                dt = _parse_datetime(raw)
            except Exception:  # 极端输入也不许抛（契约：解析失败必须返回空）
                dt = None
            if dt is not None:
                return dt.isoformat(timespec="seconds"), src
    except Exception:
        pass
    return "", ""


# ---------------- 契约 A-2：片段窗口 ----------------

def _positive_diffs(values: Sequence[float]) -> List[float]:
    """序列内正间隔列表。"""
    out = []
    for a, b in zip(values, values[1:]):
        d = float(b) - float(a)
        if d > 0:
            out.append(d)
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    vs = sorted(float(v) for v in values)
    n = len(vs)
    mid = n // 2
    return vs[mid] if n % 2 else (vs[mid - 1] + vs[mid]) / 2.0


def _group_frame_ts(frame_ts, frames_per_clip: Optional[int]) -> List[List[float]]:
    """帧 ts → 窗口内的帧 ts 分组（每组 = 一个窗口的帧）。

    支持三种入参形态：
      1) 嵌套：`[[0,0.6,1.2], [1.8,2.4,3.0]]`——每个子列表即一个片段的 frame_ts
         （直接取自 `EmbedStore` 片段 meta 的 `frame_ts`，与索引窗口逐一对齐）；
      2) 扁平 + `frames_per_clip`：按帧数切窗（与 `VideoEmbedStage` 切窗规则一致）；
      3) 扁平无帧数：按帧间隔断点分组（间隔 > 1.5×中位帧间隔 = 新窗）；
         连续均匀帧序列 → 单窗（无法反推索引切窗，调用方应传 1) 或 2)）。
    """
    seq = list(frame_ts or [])
    if not seq:
        return []
    nested = any(isinstance(x, (list, tuple)) for x in seq)
    if nested:
        groups: List[List[float]] = []
        for item in seq:
            if isinstance(item, (list, tuple)):
                g = sorted({float(v) for v in item})
            else:  # 混合形态：散点各自成组
                g = [float(item)]
            if g:
                groups.append(g)
        return groups
    values = sorted({float(v) for v in seq})
    if frames_per_clip is not None and int(frames_per_clip) > 0:
        n = int(frames_per_clip)
        return [values[i:i + n] for i in range(0, len(values), n)]
    step = _median(_positive_diffs(values))
    if not step:
        return [values]
    groups = [[values[0]]]
    for prev, cur in zip(values, values[1:]):
        if cur - prev > step * 1.5:  # 断点：帧间隔显著大于常规节奏
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def clip_windows(frame_ts: Sequence[float],
                 frames_per_clip: Optional[int] = None,
                 clip_seconds: Optional[float] = None
                 ) -> List[Tuple[float, float]]:
    """由片段帧 ts 出窗口列表 `[(t_start, t_end), ...]`（**右开左闭**，§2.3）。

    `t_end` = 下一窗首帧 ts（与索引窗口无缝铺满，事件不会落在窗缝里）；
    末窗 `t_end` = 末帧 ts + 推断帧间隔（推断不出时用 `clip_seconds`，再退
    `DEFAULT_CLIP_SECONDS`）。`frames_per_clip`/嵌套入参语义见 `_group_frame_ts`。
    """
    groups = _group_frame_ts(frame_ts, frames_per_clip)
    if not groups:
        return []
    all_diffs: List[float] = []
    for g in groups:
        all_diffs.extend(_positive_diffs(g))
    step = _median(all_diffs)
    if not step:
        step = float(clip_seconds) if clip_seconds else DEFAULT_CLIP_SECONDS
    windows: List[Tuple[float, float]] = []
    for i, g in enumerate(groups):
        if not g:
            continue
        start = float(g[0])
        if i + 1 < len(groups) and groups[i + 1]:
            end = float(groups[i + 1][0])
        else:
            end = float(g[-1]) + float(step)
        if end <= start:  # 零宽窗：用兜底步长撑开（避免窗口不可命中）
            end = start + (float(clip_seconds) if clip_seconds else DEFAULT_CLIP_SECONDS)
        windows.append((start, end))
    return windows


# ---------------- 契约 A-3：帧级统计聚合（观察层） ----------------

def _as_rec(rec) -> Mapping:
    """记录归一：dict 原样，其他对象尽力转 dict（延迟导入，保持纯逻辑轻依赖）。"""
    if isinstance(rec, Mapping):
        return rec
    try:
        from pipe.records import to_record
        return to_record(rec)
    except Exception:
        return {}


def _cls_key(rec: Mapping) -> str:
    """det_stats 的类别键：优先 `cls_id`（§2.5 原则 2），无则退 label，再无 = unknown。"""
    cid = rec.get("cls_id")
    if cid is not None:
        try:
            return str(int(cid))
        except (TypeError, ValueError):
            return str(cid)
    lbl = rec.get("label")
    return str(lbl) if lbl not in (None, "") else "unknown"


def _track_key(tid) -> Optional[Any]:
    """track_id 归一；`None`/`0`/`""` 视作**无跟踪**（YOLO 未开跟踪时的哨兵值）。

    无跟踪的检出只计入 count/peak，不计入 tracks 并集（避免把所有目标错误合并成 1 条）。
    """
    if tid is None or tid == "" or tid == 0:
        return None
    try:
        return int(tid)
    except (TypeError, ValueError):
        return str(tid)


# frame_dets 剥除键（§2.5 原则 3 第四轮，WO-0033.10）：
#   - `model_path`：模型指纹非观察事实，设计明确要求剥除（YOLO det 记录自带，见 composer）；
#   - `embedding` + 旁路图键（image/img/face_img/plate_img）：records 契约「像素与向量
#     不进 JSON 主干」——frame_dets 须可 JSON 序列化落表，ndarray 必须剥。
# 旁路图键名经 pipe.records 常量取（IMAGE_KEY/IMAGE_ALIASES，惰性解析一次并缓存，
# 取不到时退硬编码清单），与 records 契约保持同步。
_STRIP_KEYS: Optional[Tuple[str, ...]] = None


def _mainline_rec(rec: Mapping) -> dict:
    """记录 → 可落表的主干副本：剥 `model_path` 与旁路键，其余键原样保留。

    plate/face 的属性键（plate_no/color/type/kps/size…）与通用字段
    （kind/ts/frame_idx/bbox/score/cls_id/track_id/label/text）都是主干，全保留。
    """
    global _STRIP_KEYS
    if _STRIP_KEYS is None:
        keys = ("model_path", "embedding")
        try:
            from pipe.records import IMAGE_ALIASES, IMAGE_KEY
            keys += (IMAGE_KEY, *IMAGE_ALIASES)
        except Exception:
            keys += ("image", "img", "face_img", "plate_img")
        _STRIP_KEYS = keys
    return {k: v for k, v in rec.items() if k not in _STRIP_KEYS}


def _frame_recs(fr: Mapping) -> List[dict]:
    """一帧的三桶记录（dets/plates/faces）→ frame_dets 的 recs（主干化，桶序固定）。"""
    return [_mainline_rec(_as_rec(raw))
            for bucket in ("dets", "plates", "faces")
            for raw in (fr.get(bucket) or [])]


def aggregate_frame_stats(per_frame: Sequence[dict]) -> dict:
    """窗口内逐帧记录 → §2.5 **观察层**标签。

    `per_frame`: `[{ts, dets: [...], plates: [...], faces: [...]}, ...]`
    （`dets`/`plates`/`faces` 缺省按空列表；元素为 Records dict 或可转记录对象）。

    返回:
      - `det_stats`: `{cls_id(str): {"peak": 帧级 count 峰值, "avg": 帧级 count 均值
        （分母=窗口内全部帧，含 0 帧）, "tracks": track_id 并集大小}}`
        ——定位已降级为**派生缓存**（§2.5 原则 3 第四轮：可从 `frame_dets` 重算，
        口径变更走 relabel），数值口径不变；
      - `class_names`: `{cls_id(str): label}`（采集自 `dets` 记录的 `label`）
      - `plate_nos`: 车牌号集合（去重排序；取 `plate_no`，回退 `label`/`text`）
      - `face_count`: 窗口内单帧人脸数**最大值**
      - `frame_dets`: 观察层**原始记录流**（§2.5 原则 3 第四轮改定）：逐帧
        `{ts, recs: [...]}`，recs = dets/plates/faces 三桶记录**主干原样**
        （kind=det/plate/face 多态混存，桶序固定 dets→plates→faces），
        剥 `model_path` 与旁路键（像素/向量，见 `_mainline_rec`）；
        **只保留有记录的帧**（空观察源 → `[]`，不缺键）；recs 总数 >
        `FRAME_DETS_LIMIT`(500) 时**按 ts 顺序保留靠前帧**（同帧内按记录原序，
        超限记录与其后各帧全部丢弃），截断发生的帧附 `"truncated": true`。
        时间形态/数值统计与将来「检出框重放」的数据源——relabel 重算即可，
        无需重跑 YOLO、无需重嵌向量。det_stats 口径不受截断影响（全窗输入统计）。
    """
    frames = [fr if isinstance(fr, Mapping) else _as_rec(fr)
              for fr in (per_frame or [])]
    n = len(frames)
    grids: Dict[str, List[int]] = {}          # cls_id → 逐帧 count（含 0）
    tracks: Dict[str, set] = defaultdict(set)  # cls_id → track_id 并集
    class_names: Dict[str, str] = {}
    plate_set = set()
    face_count = 0

    for i, fr in enumerate(frames):
        counts: Counter = Counter()
        for raw in (fr.get("dets") or []):
            rec = _as_rec(raw)
            key = _cls_key(rec)
            counts[key] += 1
            tk = _track_key(rec.get("track_id"))
            if tk is not None:
                tracks[key].add(tk)
            lbl = rec.get("label")
            if lbl not in (None, "") and key not in class_names:
                class_names[key] = str(lbl)
        for key in counts:
            grids.setdefault(key, [0] * n)
        for key, cnt in counts.items():
            grids[key][i] = cnt
        for raw in (fr.get("plates") or []):
            rec = _as_rec(raw)
            pn = rec.get("plate_no") or rec.get("label") or rec.get("text")
            if pn not in (None, ""):
                plate_set.add(str(pn))
        face_count = max(face_count, len(fr.get("faces") or []))

    # frame_dets 原始记录流：按 ts 顺序逐帧分组（稳定排序，同 ts 保原序；
    # ts 非数值的帧排最后），只收有记录的帧，超上限截断（见 docstring）。
    order = sorted(range(n), key=lambda i: (
        frames[i].get("ts")
        if isinstance(frames[i].get("ts"), (int, float))
        and not isinstance(frames[i].get("ts"), bool) else float("inf"), i))
    frame_dets: List[dict] = []
    rec_total = 0
    truncated = False
    for i in order:
        if truncated:
            break  # 已到上限：后续帧全部丢弃（按 ts 顺序保留靠前帧）
        recs = _frame_recs(frames[i])
        if not recs:
            continue  # 空帧不进 frame_dets（无信息；空观察源 → []）
        room = FRAME_DETS_LIMIT - rec_total
        if room <= 0:
            truncated = True
            break
        take = recs if len(recs) <= room else recs[:room]
        win = {"ts": frames[i].get("ts"), "recs": take}
        if len(take) < len(recs):  # 本帧被截断：其后的帧不再收
            win["truncated"] = True
            truncated = True
        rec_total += len(take)
        frame_dets.append(win)

    det_stats: Dict[str, dict] = {}
    for key in sorted(grids, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)):
        row = grids[key]
        det_stats[key] = {
            "peak": int(max(row)) if row else 0,
            "avg": round(sum(row) / n, 3) if n else 0.0,
            "tracks": len(tracks.get(key, ())),
        }
    return {"det_stats": det_stats, "class_names": class_names,
            "plate_nos": sorted(plate_set), "face_count": int(face_count),
            "frame_dets": frame_dets}


# ---------------- 契约 A-4：事件 → 窗口归属 ----------------

def align_events(events: Sequence[dict], windows: Sequence[Sequence[float]]
                 ) -> List[dict]:
    """每条事件补窗口归属（§2.3）：右开左闭命中、**跨窗各记一条**、`ts` 保留原值。

    返回元素 = 原事件浅拷贝 + `window: [t_start, t_end]`（另附 `t_start`/`t_end`
    便于直接落表）；无 `ts` 或未命中任何窗口的事件**保留**并标 `window=None`
    （不静默丢弃，调用方按需过滤——对齐不上的事件应可解释）。
    """
    wins = []
    for w in windows or []:
        try:
            a, b = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError):
            continue
        wins.append((a, b))
    out: List[dict] = []
    for ev in events or []:
        base = dict(ev) if isinstance(ev, Mapping) else {"name": str(ev)}
        try:
            ts = float(base.get("ts"))
        except (TypeError, ValueError):
            ts = None
        hits = [w for w in wins if ts is not None and w[0] <= ts < w[1]]
        if not hits:
            row = dict(base)
            row.setdefault("window", None)
            out.append(row)
            continue
        for start, end in hits:
            row = dict(base)
            row["ts"] = base.get("ts")  # 保留原值（不裁剪、不改类型）
            row["window"] = [start, end]
            row["t_start"] = start
            row["t_end"] = end
            out.append(row)
    return out


# ---------------- 契约 A-5：片段标签组装 ----------------

def _window_of(ev: Mapping, window: Tuple[float, float]) -> bool:
    """事件是否属该窗口：已带 window 归属按归属比对，否则按 ts 右开左闭判定。"""
    w = ev.get("window")
    if isinstance(w, (list, tuple)) and len(w) >= 2:
        try:
            return float(w[0]) == window[0] and float(w[1]) == window[1]
        except (TypeError, ValueError):
            return False
    try:
        ts = float(ev.get("ts"))
    except (TypeError, ValueError):
        return False
    return window[0] <= ts < window[1]


def _plate_item(ev: Mapping) -> Optional[dict]:
    """判断层车牌事件 → `plates[]` 行（无车牌号返回 None）。

    路由（WO-0033.2 接缝②双保险）：带 `plate_no`（含 label/text 载体，现状保留）
    或 `kind=="车牌"` 的事件进 `plates[]`——web 归档原始形态
    `{name:"沪A12345", kind:"车牌"}` 无 plate_no 键，此前会错落 events[]。
    车牌号载体回退顺序：`plate_no → label → text → name`；末级 `name` 仅在
    `kind=="车牌"` 时兜底——plate 记录无 plate_no 键时载体是 label/text（已核实），
    web 归档只落 name；非车牌事件的 name 是事件名（如"斗殴"），不可兜底，
    否则警情会被误判成车牌。提取不出车牌号 → None → 落 events[]（不产空行）。
    """
    is_plate = str(ev.get("kind") or "") == "车牌"
    pn = (ev.get("plate_no") or ev.get("label") or ev.get("text")
          or (ev.get("name") if is_plate else None))
    if pn in (None, ""):
        return None
    item = {"plate_no": str(pn)}
    for k in ("color", "type", "score", "ts"):
        if ev.get(k) is not None:
            item[k] = ev.get(k)
    return item


def build_clip_tags(video: str, window: Sequence[float], obs: dict,
                    events: Sequence[dict]) -> dict:
    """组装 §2.5 片段标签（每片段一行；字段名 = `review_index` 列名）。

    - `obs` = `aggregate_frame_stats` 的结果（观察层；`frame_dets` 原始记录流
      随 obs 透传进标签行，§2.5 原则 3 第四轮）；
    - `events` = 已按窗口对齐的事件（`align_events` 结果，或带 ts 的原始事件——
      本函数按 `window` 归属优先、否则按 ts 归窗，只取落在 `window` 的）；
      带 `plate_no` 或 `kind=="车牌"` 的进 `plates[]`（判断层带属性+ts，车牌号
      载体回退顺序见 `_plate_item`），其余进 `events[]`
      （`{name, kind, ts, detail}`，车牌号不重复进 events，§2.5 原则 1）；
    - `description` 空串（语义档后置）；`thumb_path` 由调用方按片段补，
      向量版本三列（`vector_model`/`clip_dim`/`frame_dim`）同样由调用方补。

    窗口统一按右开左闭 `[t_start, t_end)` 输出，`t_start`/`t_end` 为 float。
    """
    start, end = float(window[0]), float(window[1])
    obs = dict(obs or {})
    plates: List[dict] = []
    evs: List[dict] = []
    for raw in events or []:
        ev = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        if not _window_of(ev, (start, end)):
            continue
        item = _plate_item(ev)
        if item is not None:
            plates.append(item)
            continue
        row = {"name": str(ev.get("name") or ""),
               "kind": str(ev.get("kind") or ""),
               "ts": ev.get("ts")}
        detail = ev.get("detail")
        if detail:
            row["detail"] = detail
        evs.append(row)
    # frame_dets 透传（§2.5 原则 3 第四轮：观察层原始记录流进标签行；
    # 窗口对象与 recs 列表浅拷贝隔离，观察源缺省 → [] 不缺键）
    frame_dets = [dict(w, recs=list(w.get("recs") or []))
                  for w in (obs.get("frame_dets") or [])
                  if isinstance(w, Mapping)]
    return {
        "video": str(video),
        "t_start": start,
        "t_end": end,
        "det_stats": dict(obs.get("det_stats") or {}),
        "frame_dets": frame_dets,
        "class_names": dict(obs.get("class_names") or {}),
        "plate_nos": list(obs.get("plate_nos") or []),
        "plates": plates,
        "events": evs,
        "face_count": int(obs.get("face_count") or 0),
        "description": str(obs.get("description") or ""),
        "thumb_path": "",
    }
