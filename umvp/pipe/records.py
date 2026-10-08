# -*- coding: utf-8 -*-
"""
通用记录（Record）约定 + 对象适配（适配器路线）

模块间通用语：每个模块逐帧输出 = list[Record]，Record 是 dict——
  必备字段：kind（记录类型，如 det/plate/face/crop/alarm）、ts（时间戳）、
            frame_idx（帧序号）
  通用字段（跨模块同名同义）：bbox / score / cls_id / track_id / label / text
  旁路字段（允许挂在记录上、约定不序列化）：image（ndarray 子图/帧）、
            embedding（向量）——像素与向量不进 JSON
  派生字段（条件 DSL 可直接引用，从 bbox 计算）：w/h/short/long/area
  （见 pipe.conditions）

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
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    for cls in type(obj).__mro__:
        fn = _ADAPTERS.get(cls)
        if fn is not None:
            return dict(fn(obj))
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in _dc_fields(obj)}
    d = getattr(obj, "__dict__", None)
    return dict(d) if d else {}


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
