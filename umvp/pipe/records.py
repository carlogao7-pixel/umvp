# -*- coding: utf-8 -*-
"""
通用记录（Record）约定 + 对象适配（适配器路线）

模块间通用语：每个模块逐帧输出 = list[Record]，Record 是 dict——
  必备字段：kind（记录类型，如 det/plate/face/crop/alarm）、ts（时间戳）、
            frame_idx（帧序号）
  通用字段（跨模块同名同义）：bbox / score / cls_id / track_id / label / text
  旁路字段（允许挂在记录上、约定不序列化）：**image**（ndarray 子图/帧；**或 ndarray 列表＝图集**：
           一条记录需多图时以**集合**形式携带，**不拆记录**——下游自行处理单图/图集；基数约定见
           doc/design/assembly_ports.md §3⑥）、
             embedding（向量）——像素与向量不进 JSON
             · 记录旁路图像统一键名 = image；历史模块自造键为**兼容别名**，
               经 to_record 转换时自动补 image（旧键保留，新旧并存）：
               target_crop → img、face_detect → face_img、plate_recog → plate_img
               （见 IMAGE_KEY / IMAGE_ALIASES；对象属性 c.img / f.face_img / pl.plate_img
               不受影响）
  派生字段（条件 DSL 可直接引用，从 bbox 计算）：w/h/short/long/area
  （见 pipe.conditions）
  事件字段：events —— **列表**，元素为事件 dict {"name": 事件名, "ts": 时间戳,
            "detail": {自定义内容}}；一条记录可携带多个事件（见 add_event/get_events）。
            任何模块都可给记录追加事件；链路尾用 collect_events() 汇总为输出，
            不绑定具体模块。

适配器路线：老模块输出（list[Detection] 等对象列表）不必改造为 dict——在需要
统一消费的边界（筛选模块入口 / DB / 测试台展示 / 落盘清单）经 to_record() 转换。
dataclass（如 Detection）自动按字段名转 dict；特殊类型用 @register_adapter 注册；
新模块鼓励原生输出 Record。时间上下文（ts/frame_idx）由调用方注入而非模块自记。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as _dc_fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Type

# 类型 → 记录转换函数（按 __mro__ 就近匹配）
_ADAPTERS: Dict[Type, Callable[[Any], dict]] = {}
EVENTS_KEY = "events"  # 记录携带的事件列表字段名（通用约定）

# 旁路图像字段：统一键名 IMAGE_KEY，历史模块自造键为兼容别名（记录 dict 层新旧并存）
IMAGE_KEY = "image"
IMAGE_ALIASES = ("img", "face_img", "plate_img")  # target_crop / face_detect / plate_recog


def image_of(rec) -> Any:
    """从记录/dict 取旁路图：优先 image，回退兼容别名；都没有返回 None。"""
    if isinstance(rec, Mapping):
        for k in (IMAGE_KEY, *IMAGE_ALIASES):
            v = rec.get(k)
            if v is not None:
                return v
        return None
    for k in (IMAGE_KEY, *IMAGE_ALIASES):
        v = getattr(rec, k, None)
        if v is not None:
            return v
    return None


def _unify_image_key(rec: dict) -> dict:
    """把旧旁路图像键（img/face_img/plate_img）的值补到 image（旧键保留，原地补键）。

    统一记录旁路图像命名（records 约定 image）；缺 image 时按别名顺序补第一个非空值。
    已有 image（含显式 None）则不覆盖；无任何旁路图键（如 Detection）原样返回。
    """
    if IMAGE_KEY not in rec:
        for k in IMAGE_ALIASES:
            v = rec.get(k)
            if v is not None:
                rec[IMAGE_KEY] = v
                break
    return rec


def register_adapter(cls: Type) -> Callable[[Callable[[Any], dict]], Callable[[Any], dict]]:
    """装饰器：注册类型的自定义转记录函数（覆盖 dataclass 默认转换）。"""
    def deco(fn: Callable[[Any], dict]) -> Callable[[Any], dict]:
        _ADAPTERS[cls] = fn
        return fn
    return deco


def to_record(obj) -> dict:
    """任意对象/dict → Record（浅拷贝 dict，不改原对象）。

    Mapping 直接拷贝；注册适配器优先；dataclass 按字段名转；
    其余尽力取 __dict__，取不到给空 dict（字段缺失由消费方兜底）。
    末尾统一旁路图像键：旧键（img/face_img/plate_img）自动补 image（见 _unify_image_key）。
    """
    if isinstance(obj, Mapping):
        return _unify_image_key(dict(obj))
    for cls in type(obj).__mro__:
        fn = _ADAPTERS.get(cls)
        if fn is not None:
            return _unify_image_key(dict(fn(obj)))
    if is_dataclass(obj) and not isinstance(obj, type):
        return _unify_image_key({f.name: getattr(obj, f.name) for f in _dc_fields(obj)})
    d = getattr(obj, "__dict__", None)
    return _unify_image_key(dict(d)) if d else {}


def records(items, kind: Optional[str] = None, ts: Optional[float] = None,
            frame_idx: Optional[int] = None) -> List[dict]:
    """批量转记录 + 补齐约定字段（setdefault：已带的值不覆盖）。"""
    out: List[dict] = []
    for it in items or []:
        r = to_record(it)
        if kind is not None:
            r.setdefault("kind", kind)
        if ts is not None:
            r.setdefault("ts", ts)
        if frame_idx is not None:
            r.setdefault("frame_idx", frame_idx)
        out.append(r)
    return out


# ---------------- 通用事件（events，多事件） ----------------
def _events_list(item) -> Optional[list]:
    """取/建记录的事件列表（dict 原地；对象挂属性）。不可写对象返回 None。"""
    if isinstance(item, Mapping):
        evs = item.get(EVENTS_KEY)
        if not isinstance(evs, list):
            evs = []
            item[EVENTS_KEY] = evs
        return evs
    evs = getattr(item, EVENTS_KEY, None)
    if isinstance(evs, list):
        return evs
    try:
        setattr(item, EVENTS_KEY, [])
        return getattr(item, EVENTS_KEY)
    except Exception:
        return None


def add_event(item, name, ts: Optional[float] = None,
              detail: Optional[dict] = None, **extra) -> Optional[dict]:
    """给记录追加一个事件（支持一条记录携带多个事件）。

    事件结构：{"name": 事件名, "ts": 时间戳(可选), "detail": {自定义内容}}。
    detail 与 extra 合并进 detail；dict 输入原地 append，对象挂 events 属性。
    通用：不绑定具体模块；调用方自行决定事件名与内容（如 VLM 的 JSON 内容字段）。
    """
    ev: dict = {"name": str(name)}
    if ts is not None:
        ev["ts"] = ts
    d = dict(detail or {})
    d.update(extra)
    if d:
        ev["detail"] = d
    evs = _events_list(item)
    if evs is None:
        return None
    evs.append(ev)
    return ev


def get_events(item) -> List[dict]:
    """取记录携带的事件列表（只读副本）。"""
    if isinstance(item, Mapping):
        return list(item.get(EVENTS_KEY) or [])
    return list(getattr(item, EVENTS_KEY, None) or [])


def collect_events(items) -> List[dict]:
    """汇总一批记录携带的全部事件——链路尾的通用输出（不绑定模块）。"""
    out: List[dict] = []
    for it in items or []:
        out.extend(get_events(it))
    return out
