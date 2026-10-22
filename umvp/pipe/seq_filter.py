# -*- coding: utf-8 -*-
"""
时序判定模块（SeqFilter）：有序记录流的规则确认闸 + 事件标注
（二期通用化，能力 2+3；设计与语义见 doc/modules/seq_filter.md）

两种规则、一种形式（无窗口 → 单条语义）：
  单条规则：记录命中 match 条件 → 放行并在该记录上标注事件（电平：每条命中都标）
  窗口规则：同一 key 分组内，`span` 秒时间窗 / `window_n` 计数窗内命中 ≥ 需求 → 放行
            并标注事件（N 次确认闸；未确认的中间命中扣留，边沿：达标放行一条）
  需求 = min_n；`hit_ratio`>0 时取 max(min_n, ceil(分母×比例))（计数窗需先凑满 window_n）
  边沿/抑制：`cooldown`≥0=达成后再触发抑制秒；`cooldown`<0=同 key **只触发一次**（不重报）
  动态事件名：`event` 支持 `{字段}` 占位（如 "{label}"、"警情-{cls_id}"）

事件标注（不造新对象，事件跟着证据记录走；**支持一条记录多个事件**）：
  rec["events"] = [{"name": 事件名, "ts": 达成时刻, "detail": {"key","count","span"}}, ...]
  （每条匹配并达成的规则各追加一个事件；同一记录可命中多条规则 → 多个事件）

语义细则：规则**各自独立**（一条记录可被多条规则同时命中共标注；达成才追加事件）；
窗口达标即清空重臂，cooldown 再按 key 额外抑制；时间取记录自带 ts（缺省用 run 的
ts）；key 值缺失/None/0 归 "*" 组；输出保持原类型（对象进对象出、dict 进 dict 出）。

门控（时序节流/变化，2026-10 自 Pipeline 迁入，与规则可并用）：
  throttle —— 整类节流：key（一般 cls_id）+ 间隔秒；同一 key 每隔 interval 秒才放行
              整类记录，其间扣留（等价原 Pipeline.check_interval）
  change   —— 变化门控：key（一般 cls_id）+ watch（默认 track_id）；同类 watch 值集合
              相比上次有变化才放行整类（等价原 Pipeline.track_change_only）
执行顺序：throttle → change → 规则；无规则时（纯门控）门控存活者全部放行。

用法:
    from pipe.seq_filter import SeqFilter, SeqRule
    sf = SeqFilter(rules=[
        SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=3, cooldown=30.0),
        SeqRule(match="score >= 0.9", event="高置信"),
    ])
    sf2 = SeqFilter(throttle={"key": "cls_id", "interval": 3.0})   # 纯节流门控
    out = sf.run(dets, ts=12.0, frame_idx=30)   # list[dict]：放行记录（带 event 字段）
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from pipe.conditions import match_obj, parse_condition
from pipe.records import add_event, get_events, to_record  # noqa: F401  (get_events 供调用方)


@dataclass
class SeqRule:
    """一条时序判定规则。

    窗口：`span`>0 时间窗（秒）/ `window_n`>0 计数窗（最近 N 条到达）二选一；
          两者皆 0 → 单条语义（命中即达成）。窗口内命中 ≥ 需求即达成。
    需求：`min_n`（最少命中数）；`hit_ratio`>0 时取 max(min_n, ceil(窗口内到达数×比例))。
    冷却：`cooldown`≥0=达成后再触发抑制秒数；`cooldown`<0=同 key 只触发一次（不重报）。
    事件：`event` 支持 `{字段}` 占位（动态事件名），如 "{label}"、"警情-{cls_id}"。
    """

    match: str                  # 条件 DSL（必填；非法表达式装配期抛错）
    event: str                  # 事件名（必填），支持 {字段} 占位
    key: str = "track_id"       # 分组键（窗口计数与冷却抑制均按它分组）
    span: float = 0.0           # 时间窗秒（与 window_n 二选一）
    window_n: int = 0           # 计数窗（最近 N 条到达；与 span 二选一）
    min_n: int = 1              # 窗口内命中 ≥ n 才达成
    hit_ratio: float = 0.0      # 窗口命中比例阈值（>0 时需求=max(min_n, ceil(到达数×比例))）
    cooldown: float = 0.0       # 达成后抑制秒；<0 = 同 key 只触发一次

    def __post_init__(self) -> None:
        if not (self.match or "").strip():
            raise ValueError("时序判定规则缺少条件（match）")
        if not (self.event or "").strip():
            raise ValueError("时序判定规则缺少事件名（event）")
        self.match = self.match.strip()
        self.event = self.event.strip()
        self.span = float(self.span)
        self.window_n = int(self.window_n)
        self.min_n = int(self.min_n)
        self.hit_ratio = float(self.hit_ratio)
        self.cooldown = float(self.cooldown)
        if self.span < 0 or self.window_n < 0 or self.min_n < 1:
            raise ValueError("时序判定规则参数越界（span/window_n 非负、min_n≥1）")
        if not (0.0 <= self.hit_ratio <= 1.0):
            raise ValueError("hit_ratio 需在 [0,1]")
        if self.min_n > 1 and self.span <= 0 and self.window_n <= 0:
            raise ValueError("min_n>1 需要 span>0 或 window_n>0（窗口语义）")
        self._ast = parse_condition(self.match)  # 非法条件在装配时快速失败

    @property
    def windowed(self) -> bool:
        return self.span > 0 or self.window_n > 0


class SeqFilter:
    """时序判定阶段：run(items, ts, frame_idx) -> list（放行的**原元素**，对象/记录 dict 形态保持不变）。

    rules    : 规则确认闸列表（SeqRule）；可为空（此时必须有 throttle/change）。
    throttle : {"key": 字段, "interval": 秒} 整类节流门控（等价 check_interval）。
    change   : {"key": 字段, "watch": 字段} 变化门控（等价 track_change_only）。
    执行顺序 throttle → change → rules；无规则时门控存活者全部放行。
    """

    # 产物端口声明：吃检测记录，放行子集仍是检测记录（门控/确认不改内容）。
    INPUTS = {"dets": "Records"}
    OUTPUTS = {"dets": "Records"}

    def __init__(self, rules: Optional[List[SeqRule]] = None,
                 throttle: Optional[dict] = None, change: Optional[dict] = None) -> None:
        self.rules = list(rules or [])
        self.throttle = self._norm_throttle(throttle)
        self.change = self._norm_change(change)
        if not self.rules and self.throttle is None and self.change is None:
            raise ValueError("时序判定至少需要一条规则或一个门控（throttle/change）")
        self._win: Dict[Tuple[int, str], Deque[Tuple[float, bool]]] = {}  # (规则,key) -> [(ts, 是否命中)]
        self._cool: Dict[Tuple[int, str], float] = {}  # (规则,key) -> 此时刻后可再触发
        self._throttle_last: Dict = {}   # throttle key 值 -> 上次放行 ts
        self._change_last: Dict = {}     # change key 值 -> 上次监看的集合

    # ---- 门控参数规整 ----
    @staticmethod
    def _norm_throttle(d) -> Optional[dict]:
        if not d:
            return None
        key = str(d.get("key") or "").strip()
        if not key:
            return None
        try:
            interval = float(d.get("interval", 0))
        except (TypeError, ValueError):
            interval = 0.0
        return {"key": key, "interval": interval} if interval > 0 else None

    @staticmethod
    def _norm_change(d) -> Optional[dict]:
        if not d:
            return None
        key = str(d.get("key") or "").strip()
        if not key:
            return None
        watch = str(d.get("watch") or "track_id").strip() or "track_id"
        return {"key": key, "watch": watch}

    # ---- 内部 ----
    def _key_of(self, rec: dict, key: str) -> str:
        v = rec.get(key)
        return "*" if v is None or v == "" or v == 0 else str(v)

    def _rec(self, it, ts: float, frame_idx: int) -> dict:
        """记录视图：dict 原地（事件可写回），对象转 Record 副本（只读匹配）。"""
        if isinstance(it, Mapping):
            it.setdefault("ts", ts)
            it.setdefault("frame_idx", frame_idx)
            return it
        rec = to_record(it)
        rec.setdefault("ts", ts)
        rec.setdefault("frame_idx", frame_idx)
        return rec

    def _render_event(self, tmpl: str, rec: dict) -> str:
        """事件名支持 {字段} 占位（动态事件名）；字段缺失填空串。"""
        if "{" not in tmpl:
            return tmpl

        def _sub(m):
            v = rec.get(m.group(1))
            return "" if v is None else str(v)

        return re.sub(r"\{([^}]+)\}", _sub, tmpl)

    def _register(self, rule: SeqRule, ri: int, k: str, rts: float, matched: bool) -> None:
        """登记一次到达（窗口用；比例阈值需要分母，故非命中到达也登记）。"""
        dq = self._win.setdefault((ri, k), deque())
        dq.append((rts, bool(matched)))
        if rule.window_n > 0:
            while len(dq) > rule.window_n:
                dq.popleft()
        else:
            while dq and rts - dq[0][0] > rule.span:
                dq.popleft()

    def _window_ok(self, rule: SeqRule, ri: int, k: str) -> int:
        """窗口是否达成；达成返回窗口内命中数，未达成返回 0。

        计数窗（window_n>0）需先凑满 window_n 条到达；需求 = min_n，hit_ratio>0 时取
        max(min_n, ceil(分母 × hit_ratio))（计数窗分母=window_n，时间窗分母=窗内到达数）。
        """
        dq = self._win.get((ri, k))
        if not dq:
            return 0
        if rule.window_n > 0:
            if len(dq) < rule.window_n:
                return 0
            denom = rule.window_n
        else:
            denom = len(dq)
        matched = sum(1 for _, m in dq if m)
        need = rule.min_n
        if rule.hit_ratio > 0:
            need = max(need, math.ceil(denom * rule.hit_ratio))
        return matched if matched >= need else 0

    def _mark(self, it, rec: dict, rule: SeqRule, ri: int, k: str, count: int, rts: float) -> None:
        """追加事件 + 状态更新（窗口重臂 / 冷却）：dict 原地 append、对象挂属性。"""
        add_event(it, self._render_event(rule.event, rec), ts=rts,
                  detail={"key": k, "count": count, "span": rule.span, "window_n": rule.window_n})
        if rule.windowed:
            self._win.pop((ri, k), None)   # 窗口重臂（清空重新计数）
        # cooldown<0 = 同 key 只触发一次
        self._cool[(ri, k)] = (rts + rule.cooldown) if rule.cooldown >= 0 else float("inf")

    def _throttle_gate(self, pairs: List[tuple], ts: float) -> List[tuple]:
        """整类节流：同一 key 每隔 interval 秒放行一次整类，其间扣留。"""
        key, interval = self.throttle["key"], self.throttle["interval"]
        groups: Dict = {}
        for it, rec in pairs:
            groups.setdefault(rec.get(key), []).append((it, rec))
        out: List[tuple] = []
        for k, ps in groups.items():
            last = self._throttle_last.get(k)
            if last is None or ts - last >= interval:
                self._throttle_last[k] = ts
                out.extend(ps)
        return out

    def _change_gate(self, pairs: List[tuple]) -> List[tuple]:
        """变化门控：同类 watch 值集合相比上次有变化才放行整类（空集不抑制）。"""
        key, watch = self.change["key"], self.change["watch"]
        groups: Dict = {}
        for it, rec in pairs:
            groups.setdefault(rec.get(key), []).append((it, rec))
        out: List[tuple] = []
        for k, ps in groups.items():
            cur = frozenset(v for v in (rec.get(watch) for _, rec in ps) if v not in (None, "", 0))
            last = self._change_last.get(k)
            if not (cur == last and len(cur) > 0):
                out.extend(ps)
                self._change_last[k] = cur
        return out

    # ---- 主入口 ----
    def run(self, items, ts: float = 0.0, frame_idx: int = 0) -> List:
        """门控（throttle→change）后按规则确认，返回**原元素**（对象进对象出、dict 进 dict 出）。

        无规则（纯门控）时返回门控存活者；有规则时每条记录对**所有**匹配规则独立判定
        （达成即追加事件，一条记录可命中多条 → 多个事件；任一规则达成则该记录放行，
        否则窗口未凑满/冷却的命中仅作证据、扣留）。
        """
        pairs = [(it, self._rec(it, ts, frame_idx)) for it in (items or [])]
        if self.throttle is not None:
            pairs = self._throttle_gate(pairs, ts)
        if self.change is not None:
            pairs = self._change_gate(pairs)
        if not self.rules:
            return [it for it, _ in pairs]
        out: List = []
        for it, rec in pairs:
            rts = float(rec.get("ts") or ts)
            fired = False
            for ri, rule in enumerate(self.rules):
                k = self._key_of(rec, rule.key)
                matched = bool(match_obj(rec, rule._ast))
                if rule.windowed:
                    self._register(rule, ri, k, rts, matched)  # 窗口需登记所有到达（比例分母）
                if not matched:
                    continue
                if rts < self._cool.get((ri, k), float("-inf")):
                    continue   # 该规则冷却/一次性期内：跳过（不影响其它规则）
                if not rule.windowed:
                    # 单条语义：命中即达成（电平）
                    self._mark(it, rec, rule, ri, k, 1, rts)
                    fired = True
                    continue
                cnt = self._window_ok(rule, ri, k)
                if cnt:
                    self._mark(it, rec, rule, ri, k, cnt, rts)
                    fired = True
                # 命中但窗口未凑满：仅作窗口证据（不 fired）
            if fired:
                out.append(it)
        return out
