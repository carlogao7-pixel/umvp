# -*- coding: utf-8 -*-
"""
通用条件 DSL（框筛选）：显式字段名，组内逗号 AND、组间分号 OR

条件 = `字段 操作符 值`；操作符 `== != > >= < <= in`（`=` 等价 `==`；`in` 值用 `|` 分隔）。
字段按对象属性取值（`getattr`），故不绑定具体来源（YOLO/其它），带同名属性即可用。
字段别名：class/类别/cls → cls_id；track → track_id；conf/置信度 → score。

供「目标裁剪」（按条件裁哪些框）与「标注」（按条件画哪些框）共用。

例：
    "track_id == 1"                只留 track_id=1
    "cls_id == 2"                  只留车
    "cls_id in 0|2, score >= 0.5"  人/车 且 置信度≥0.5
    "track_id == 1; cls_id == 2"   track=1 或 车
"""

from __future__ import annotations

import re
from typing import List, Tuple

# 字段别名 → 标准字段名（显式声明字段；支持中文别名）
_FIELD_ALIASES = {
    "cls_id": "cls_id", "cls": "cls_id", "class": "cls_id", "类别": "cls_id",
    "track_id": "track_id", "track": "track_id", "id": "track_id",
    "score": "score", "conf": "score", "置信度": "score",
}

# 条件：字段 操作符 值（操作符两侧空格可有可无；字段名允许中文）
_COND_RE = re.compile(
    r"^\s*([A-Za-z_\u4e00-\u9fa5][\w\u4e00-\u9fa5]*)\s*(>=|<=|==|!=|=|>|<|in)\s*(.+?)\s*$")


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_condition(text) -> List[List[Tuple[str, str, object]]]:
    """条件 DSL → [[(字段, 操作符, 值), ...](组内 AND), ...](组间 OR)。空 → []（不筛选）。

    非法表达式抛 ValueError（通俗报错，供保存/装配时快速失败）。
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return []
    if not isinstance(text, str):  # 已解析结构直接透传（内部便利）
        return list(text)
    groups: List[List[Tuple[str, str, object]]] = []
    for gi, gtext in enumerate(str(text).split(";"), 1):
        conds: List[Tuple[str, str, object]] = []
        parts = [c for c in (p.strip() for p in gtext.split(",")) if c]
        if not parts:
            raise ValueError(f"筛选条件第 {gi} 组为空（组内逗号=并且，组间分号=或者）")
        for cond in parts:
            m = _COND_RE.match(cond)
            if not m:
                raise ValueError(f"筛选条件无法解析: {cond!r}"
                                 "（应为 字段 操作符 值，如 track_id == 1）")
            field, op, vtext = m.group(1), m.group(2), m.group(3)
            field = _FIELD_ALIASES.get(field, field)
            if op == "=":
                op = "=="
            if op == "in":
                value: object = [x.strip() for x in vtext.split("|")]
            else:
                num = _to_num(vtext)
                value = num if num is not None else vtext.strip().strip("'\"")
            conds.append((field, op, value))
        groups.append(conds)
    return groups


def _field_value(obj, name: str):
    """取对象字段；缺字段返回 None。cls_id/track_id/score 走属性（不绑定具体来源）。"""
    if name == "cls_id":
        return int(getattr(obj, "cls_id", -1))
    if name == "track_id":
        return int(getattr(obj, "track_id", 0) or 0)
    if name == "score":
        return float(getattr(obj, "score", 0.0))
    return getattr(obj, name, None)


def _cond_pass(obj, cond: Tuple[str, str, object]) -> bool:
    field, op, value = cond
    a = _field_value(obj, field)
    if a is None:
        return False
    if op in (">", ">=", "<", "<="):
        x, y = _to_num(a), _to_num(value)
        if x is None or y is None:
            return False
        return x > y if op == ">" else x >= y if op == ">=" else x < y if op == "<" else x <= y
    if op == "==":
        return a == value or str(a) == str(value)
    if op == "!=":
        return not (a == value or str(a) == str(value))
    if op == "in":
        return str(a) in [str(v) for v in value]
    return False


def match_obj(obj, groups) -> bool:
    """对象是否满足条件：空条件=全通过；否则任一组内全部命中即通过（组间 OR）。"""
    if not groups:
        return True
    return any(all(_cond_pass(obj, c) for c in g) for g in groups)
