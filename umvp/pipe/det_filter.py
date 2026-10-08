# -*- coding: utf-8 -*-
"""
检测筛选模块（DetFilter）：通用下游筛选（一期逐条筛选 + 二期成组判定）

职责：对上游输出列表做**有效性筛选**，只取舍、不改内容，放行元素原样返回
（对象进对象出、dict 进 dict 出），链路上透明插入。两层判定：

  1. 逐条筛选（filter 条件 DSL + predicates 谓词链，AND 叠加）——范围大小比较
     + 与（,）/或（;）/非（not）/括号分组，字段名灵活（别名 + 派生
     w/h/short/long/area；ts/frame_idx 由 run 注入）。
  2. 成组数量判定（group_limits，2026-10 自 YOLODetector 迁入）——按类别计数，
     数量越界或未登记的类别**整类剔除**（视为未识别，对所有下游一致生效）。
     None=不做判定；显式传空 dict=全拦（兼容旧 class_limits 空表语义）。

明确不做（边界，见 doc/modules/检测筛选.md §8）：不改内容；不节流（向下传递
节拍归 Pipeline 编排，check_interval）；不裁图/不画框/不构造 VLM 报送请求；
时间维度确认与事件标注归姊妹模块「时序判定 SeqFilter」。

条件语句语法与字段表见 pipe.conditions；输入元素为任意对象或 dict（Record）——
经 pipe.records.to_record 统一字段视图后判定，因此可作几乎所有模块的下游。

用法:
    from pipe.det_filter import DetFilter
    f = DetFilter(filter="cls_id in 0|2, short >= 30", group_limits={0: (1, 300)})
    keep = f.run(dets, ts=1.5, frame_idx=25)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from pipe.conditions import (
    build_count_context,
    collect_count_specs,
    match_obj,
    parse_condition,
)
from pipe.records import to_record


class DetFilter:
    """检测筛选阶段：run(dets, ts, frame_idx) -> 放行子集（原元素原样返回）。

    filter       : 条件 DSL（显式字段名，如 "cls_id in 0|2, short >= 30"）；空=全放行。
                   非法条件在构造时即抛 ValueError（装配期快速失败）。支持聚合谓词
                   count(字段)/count()（同帧分组计数，见 pipe.conditions）。
    group_limits : 类别 -> (数量下限, 数量上限)。None=不做成组判定；dict（含空
                   dict）=按旧 class_limits 语义：未登记类别剔除、数量越界整类剔除。
    predicates   : 代码级谓词链（可选，按序 AND 叠加在 DSL 之后）——compose 的
                   自定义过滤链等高级用法；DSL 能表达的场景不必用它。
    """

    def __init__(self, filter: str = "",
                 group_limits: Optional[Dict[int, Tuple[int, int]]] = None,
                 predicates: Optional[Sequence] = None) -> None:
        self.filter = (filter or "").strip()
        self._ast = parse_condition(self.filter)
        self.group_limits = dict(group_limits) if group_limits is not None else None
        self.predicates = list(predicates or [])
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
        """单条判定：转 Record（统一字段视图）+ 注入 ts/frame_idx 时间上下文。

        单条调用时聚合上下文仅含本条（count()=1、count(字段)=1、count(子条件)按本条算）。
        """
        rec = self._view(item, ts, frame_idx)
        ctx = build_count_context(self._ast, [rec], frame_idx) if self._uses_count else None
        return self._match(rec, item, ctx)

    def _group_keep(self, dets: List) -> List:
        """成组数量判定（group_limits 非 None 时）：按类别计数，越界/未登记整类剔除。"""
        if self.group_limits is None:
            return dets
        by_cls: Dict[int, List] = {}
        for d in dets:
            by_cls.setdefault(int(getattr(d, "cls_id", -1)), []).append(d)
        kept: List = []
        for cls, cdets in by_cls.items():
            lo, hi = self.group_limits.get(cls, (-1, -1))
            if lo >= 0 and lo <= len(cdets) <= hi:
                kept += cdets
        return kept

    def run(self, dets, ts: float = 0.0, frame_idx: int = 0) -> List:
        """逐条筛选 + 成组判定，返回放行子集（原元素原样返回）。

        聚合谓词 count(...) 的计数范围 = 本次入参列表（= Pipeline 的单帧）；
        记录自带 frame_idx 时按其二次分组（防跨帧混入）。
        """
        items = list(dets or [])
        views = [self._view(it, ts, frame_idx) for it in items]
        ctx = build_count_context(self._ast, views, frame_idx) if self._uses_count else None
        out = [it for it, v in zip(items, views) if self._match(v, it, ctx)]
        return self._group_keep(out)
