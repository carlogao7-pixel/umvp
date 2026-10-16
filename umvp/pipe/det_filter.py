# -*- coding: utf-8 -*-
"""
检测筛选模块（DetFilter）：通用下游筛选 + 事件标注（可作为链路尾）

职责：对上游输出列表做**有效性筛选**，放行元素原样返回（对象进对象出、dict 进
dict 出），链路上透明插入；并可按规格在放行记录上**追加事件**（支持多条），从而
**可直接充当链路结尾**——产物即携带 events 的记录/事件列表，不绑定上下游模块。

三层能力：
  1. 逐条筛选（filter 条件 DSL + predicates 谓词链，AND 叠加）——范围比较
     + 与（,）/或（;）/非（not）/括号分组，字段名灵活（别名 + 派生
     w/h/short/long/area；ts/frame_idx 由 run 注入）。
  2. 成组数量判定（group_limits）——按类别计数，数量越界或未登记类别整类剔除。
     None=不判定；显式空 dict=全拦。
  3. 事件追加（events，可选，多事件）——每条放行记录按规格追加事件（见下）；
     通用事件结构见 pipe.records（{"name","ts","detail"}）。

events 规格（列表，每条可含）：
    {"name": 事件名,                  # 必填
     "when": 条件 DSL(可选),          # 仅当记录满足该条件才追加（空=所有放行记录）
     "detail": [字段名...] 或 "a,b"}  # 从记录取这些字段作事件内容（如 VLM 的 JSON 内容字段）
一条记录可命中多个规格 → 追加多个事件。

用法:
    from pipe.det_filter import DetFilter
    f = DetFilter(filter="cls_id in 0|2, short >= 30", group_limits={0: (1, 300)},
                  events=[{"name": "警情", "when": "label in 治安类|交警类",
                           "detail": ["label", "content"]}])
    keep = f.run(dets, ts=1.5, frame_idx=25)   # 放行记录（可携带 events）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pipe.conditions import (
    build_count_context,
    collect_count_specs,
    match_obj,
    parse_condition,
)
from pipe.records import add_event, to_record


def _norm_events(specs) -> List[dict]:
    """事件规格归一化：str → {name}；校验 name；解析 when；detail 统一为列表。"""
    out: List[dict] = []
    for s in specs or []:
        if isinstance(s, str):
            s = {"name": s}
        if not isinstance(s, dict):
            raise ValueError(f"事件规格应为 dict 或事件名 str，得到 {type(s).__name__}")
        name = str(s.get("name") or "").strip()
        if not name:
            raise ValueError("事件规格缺少 name")
        detail = s.get("detail") or []
        if isinstance(detail, str):
            detail = [x.strip() for x in detail.replace("，", ",").split(",") if x.strip()]
        out.append({"name": name, "detail": list(detail),
                    "_ast": parse_condition(s.get("when") or "")})
    return out


class DetFilter:
    """检测筛选阶段（+可选事件标注）：run(dets, ts, frame_idx) -> 放行子集（原元素返回）。

    filter       : 条件 DSL（显式字段名，如 "cls_id in 0|2, short >= 30"）；空=全放行。
                   非法条件在构造时即抛 ValueError。支持聚合谓词 count(字段)/count(子条件)/count()。
    group_limits : 类别 -> (数量下限, 数量上限)。None=不做成组判定；dict（含空
                   dict）=未登记类别剔除、数量越界整类剔除。
    predicates   : 代码级谓词链（可选，按序 AND 叠加在 DSL 之后）。
    events       : 事件规格列表（可选，多事件）——见模块文档；放行记录上追加事件。
    """

    # 产物端口声明：吃检测记录，放行子集仍是检测记录（只取舍不改内容）。
    INPUTS = {"dets": "Records"}
    OUTPUTS = {"dets": "Records"}

    def __init__(self, filter: str = "",
                 group_limits: Optional[Dict[int, Tuple[int, int]]] = None,
                 predicates: Optional[Sequence] = None,
                 events: Optional[Sequence] = None) -> None:
        self.filter = (filter or "").strip()
        self._ast = parse_condition(self.filter)
        self.group_limits = dict(group_limits) if group_limits is not None else None
        self.predicates = list(predicates or [])
        self.events = _norm_events(events)
        # 聚合谓词：预判是否使用 count(...)，避免无谓构建批上下文
        specs = collect_count_specs(self._ast)
        self._uses_count = bool(specs["fields"] or specs["total"] or specs["conds"])

    def _view(self, item, ts: float, frame_idx: int) -> dict:
        """统一记录视图：对象/dict → dict + 注入 ts/frame_idx 时间上下文。"""
        rec = to_record(item)
        rec.setdefault("ts", ts)
        rec.setdefault("frame_idx", frame_idx)
        return rec

    def _match(self, rec: dict, item, ctx) -> bool:
        if not self._ast:
            passed = True
        else:
            passed = match_obj(rec, self._ast, ctx)
        if passed:
            passed = all(f(item) for f in self.predicates)
        return passed

    def keep(self, item, ts: float = 0.0, frame_idx: int = 0) -> bool:
        """单条判定（不追加事件）：转 Record 视图 + 注入 ts/frame_idx。

        单条调用时聚合上下文仅含本条（count()=1、count(字段)=1、count(子条件)按本条算）。
        """
        rec = self._view(item, ts, frame_idx)
        ctx = build_count_context(self._ast, [rec], frame_idx) if self._uses_count else None
        return self._match(rec, item, ctx)

    def _emit(self, item, view: dict, ts: float) -> None:
        """按规格在放行记录上追加事件（多规格 → 多事件）。"""
        for spec in self.events:
            if spec["_ast"] and not match_obj(view, spec["_ast"]):
                continue
            detail = {f: view.get(f) for f in spec["detail"] if f in view}
            add_event(item, spec["name"], ts=ts, detail=detail or None)

    def _group_keep(self, pairs: List[Tuple]) -> List[Tuple]:
        """成组数量判定（group_limits 非 None）：按类别计数，越界/未登记整类剔除。"""
        if self.group_limits is None:
            return pairs
        by_cls: Dict[int, List[Tuple]] = {}
        for it, v in pairs:
            by_cls.setdefault(int(v.get("cls_id", -1)), []).append((it, v))
        kept: List[Tuple] = []
        for cls, ps in by_cls.items():
            lo, hi = self.group_limits.get(cls, (-1, -1))
            if lo >= 0 and lo <= len(ps) <= hi:
                kept += ps
        return kept

    def run(self, dets, ts: float = 0.0, frame_idx: int = 0) -> List:
        """逐条筛选 + 成组判定 + 事件追加，返回放行子集（原元素原样返回）。

        聚合谓词 count(...) 的计数范围 = 本次入参列表（= Pipeline 的单帧）；
        记录自带 frame_idx 时按其二次分组。事件追加在放行元素上（dict 原地 / 对象挂属性）。
        """
        items = list(dets or [])
        views = [self._view(it, ts, frame_idx) for it in items]
        ctx = build_count_context(self._ast, views, frame_idx) if self._uses_count else None
        pairs = [(it, v) for it, v in zip(items, views) if self._match(v, it, ctx)]
        pairs = self._group_keep(pairs)
        if self.events:
            for it, v in pairs:
                self._emit(it, v, ts)
        return [it for it, _ in pairs]
