# -*- coding: utf-8 -*-
"""
时序判定模块（SeqFilter）：有序记录流的规则确认闸 + 事件标注
（二期通用化，能力 2+3；设计与语义见 doc/modules/时序判定.md）

两种规则、一种形式（span=0 退化为单条语义）：
  单条规则：记录命中 match 条件 → 放行并在该记录上标注事件（电平：每条命中都标）
  窗口规则：同一 key 分组内，span 秒窗口命中 ≥ min_n 次 → 第 n 条命中记录放行
            并标注事件（N 次确认闸；未确认的中间命中扣留，边沿：达标放行一条）

事件标注（不造新对象，事件跟着证据记录走）：
  rec["event"]      = 事件名（str，条件 DSL 可直接筛）
  rec["event_info"] = {"key": 分组键值, "count": 达成时命中数, "span": 窗口秒, "ts": 达成时刻}

语义细则：规则有序先匹配先得（记录被其匹配的第一条规则认领：达成放行/未凑满
扣留/冷却扣留）；窗口达标即清空重臂，cooldown 再按 key 额外抑制；时间取记录
自带 ts（缺省用 run 的 ts）；key 值缺失/None/0 归 "*" 组；输出统一 Record（dict）
——输入可为对象或 dict，经 records.to_record 转换（与检测筛选对象透传的差异）。

用法:
    from pipe.seq_filter import SeqFilter, SeqRule
    sf = SeqFilter(rules=[
        SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=3, cooldown=30.0),
        SeqRule(match="score >= 0.9", event="高置信"),
    ])
    out = sf.run(dets, ts=12.0, frame_idx=30)   # list[dict]：放行记录（带 event 字段）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from pipe.conditions import match_obj, parse_condition
from pipe.records import to_record


@dataclass
class SeqRule:
    """一条时序判定规则（span=0 → 单条语义；span>0 → 窗口语义）。"""

    match: str                  # 单条条件 DSL（必填；非法表达式装配期抛错）
    event: str                  # 达成时标注的事件名（必填）
    key: str = "track_id"       # 分组键（窗口计数与冷却抑制均按它分组）
    span: float = 0.0           # 窗口秒数 m；0=单条语义（命中即达成）
    min_n: int = 1              # 窗口内命中 ≥ n 才达成
    cooldown: float = 0.0       # 达成后的再触发抑制秒数（默认 0=窗口重臂即可再报）

    def __post_init__(self) -> None:
        if not (self.match or "").strip():
            raise ValueError("时序判定规则缺少条件（match）")
        if not (self.event or "").strip():
            raise ValueError("时序判定规则缺少事件名（event）")
        self.match = self.match.strip()
        self.event = self.event.strip()
        self.span = float(self.span)
        self.min_n = int(self.min_n)
        self.cooldown = float(self.cooldown)
        if self.span < 0 or self.min_n < 1 or self.cooldown < 0:
            raise ValueError("时序判定规则参数越界（span/cooldown 非负、min_n≥1）")
        if self.min_n > 1 and self.span <= 0:
            raise ValueError("min_n>1 需要 span>0（时间窗语义）；单条语义固定一次命中即达成")
        self._ast = parse_condition(self.match)  # 非法条件在装配时快速失败


class SeqFilter:
    """时序判定阶段：run(items, ts, frame_idx) -> list[dict]（放行 Record，带 event 字段）。"""

    def __init__(self, rules: List[SeqRule]) -> None:
        if not rules:
            raise ValueError("时序判定至少需要一条规则（空规则集=全扣留，请在装配期拒绝）")
        self.rules = list(rules)
        self._hits: Dict[Tuple[int, str], Deque[float]] = {}  # (规则序号, key 值) -> 命中时刻
        self._cool: Dict[Tuple[int, str], float] = {}         # (规则序号, key 值) -> 此时刻后可再触发

    # ---- 内部 ----
    def _key_of(self, rec: dict, key: str) -> str:
        v = rec.get(key)
        return "*" if v is None or v == "" or v == 0 else str(v)

    def _fire(self, rec: dict, rule: SeqRule, ri: int, k: str, count: int, rts: float) -> None:
        rec["event"] = rule.event
        rec["event_info"] = {"key": k, "count": count, "span": rule.span, "ts": rts}
        if rule.span > 0:
            self._hits.pop((ri, k), None)   # 窗口重臂（清空重新计数）
        self._cool[(ri, k)] = rts + rule.cooldown

    # ---- 主入口 ----
    def run(self, items, ts: float = 0.0, frame_idx: int = 0) -> List[dict]:
        """逐条按规则确认：先匹配先得——达成放行标注 / 窗口未凑满扣留 / 冷却扣留。"""
        out: List[dict] = []
        for it in items or []:
            rec = to_record(it)
            rec.setdefault("ts", ts)
            rec.setdefault("frame_idx", frame_idx)
            rts = float(rec.get("ts") or ts)
            for ri, rule in enumerate(self.rules):
                if not match_obj(rec, rule._ast):
                    continue
                # 该规则认领本记录（先匹配先得）
                k = self._key_of(rec, rule.key)
                if rts < self._cool.get((ri, k), float("-inf")):
                    break   # 冷却期内：扣留
                if rule.span <= 0:
                    # 单条语义：命中即达成（电平）
                    self._fire(rec, rule, ri, k, 1, rts)
                    out.append(rec)
                    break
                dq = self._hits.setdefault((ri, k), deque())
                dq.append(rts)
                while dq and rts - dq[0] > rule.span:
                    dq.popleft()   # 淘汰滑出窗口的旧命中
                if len(dq) >= rule.min_n:
                    self._fire(rec, rule, ri, k, len(dq), rts)
                    out.append(rec)
                    break
                break   # 命中但窗口未凑满：扣留作窗口证据
        return out
