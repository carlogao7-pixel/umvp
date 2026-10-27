# -*- coding: utf-8 -*-
"""
门控阶段模块（GateFilter）：「按 key + 时间间隔放行」的独立可组合链路阶段
（设计稿 doc/design/gate_filter.md，WO-0023；缺口诊断 INB-0089）

把散落三处的同一门控语义收敛为一个阶段（IntervalGate 三值语义、唯一实现复用）：
  interval > 0 ：同 key 每 interval 秒最多放行一次（首见放行，放行即刷新）
  interval < 0 ：同 key 只放行一次（此后不再放行，对齐 PlateStage track_cooldown≤0）
  interval == 0：全放行（no-op，不记状态）

key 归一（铁律，构造期定死、无配置项）：
  key 值 None / "" / 0  →  匿名桶 "*"（统一按 interval 冷却）
  其余值                →  str(值) 各自成桶（首见放行）
  缺 key 的记录（如 track 未建立帧 track_id=0/None）落匿名桶按节奏放行——
  **修 INB-0089 缺口①**：不再是"tid=0 绕过门控每帧放行"，匿名记录同样受门控。

语义铁律（用户拍板）：主路**物理过滤**——被拦记录不传下游、**不加任何标记字段**
（返回记录与输入逐键相同）；拦截可观测全部走**统计旁路**：
  模块属性 last_stats（in/passed/blocked/blocked_detail 分键计数）
  + Pipeline 产物 gate_filter.dets / gate_filter.stats（观测专用）
  + Pipeline.last_gate = (进入, 放行)（参照 last_filter 同型，供时间线直读）。

key 字段取值经 conditions._field_value（dict/对象统一；派生字段 w/h/short/long/area
亦可用；track_id 经数值规整 None→0→匿名桶）。可选 filter（条件 DSL）限定门控范围：
未命中记录直接放行、不触碰门控状态。

track_needed 联动：key=="track_id" 时（needs_track=True）上游 YOLO 需开跟踪，
Pipeline.track_needed() 据此判定（否则内置去重关闭后全链无 track_id、门控退化）。

用法:
    from pipe.gate_filter import GateFilter
    g = GateFilter(key="track_id", interval=5.0)
    keep = g.run(dets, ts=12.0, frame_idx=30)   # 放行的原元素（物理过滤）
    g.last_stats                                # {"in":..,"passed":..,"blocked":..,"blocked_detail":{..}}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List

from pipe.conditions import _field_value, match_obj, parse_condition
from pipe.composer import IntervalGate

ANON_KEY = "*"  # 匿名桶：key 缺失/None/""/0 的记录统一落此桶受门控


class GateFilter:
    """门控阶段（Records→Records）：run(dets, ts, frame_idx) -> 放行子集（原元素返回）。

    key     : 分组键字段名（必填，空串构造期抛 ValueError）；常用 track_id / cls_id / label。
    interval: 三值语义（与 IntervalGate 完全对齐）——>0 周期冷却秒；<0 同 key 只放行一次；
              0 全放行（no-op）。
    filter  : 条件 DSL（可选，与 det_filter/target_crop/annotate 同族）——只对命中记录
              做门控，未命中记录直接放行且不触碰门控状态；空 = 全部记录受门控；
              非法表达式构造期抛 ValueError（parse_condition 快速失败）。
    """

    # 产物端口声明：吃记录列表，放行子集仍是记录（物理过滤，只取舍不改内容）。
    INPUTS = {"dets": "Records"}
    OUTPUTS = {"dets": "Records"}   # 另登记观测产物 stats（gate_filter.stats，非连线端口）

    def __init__(self, key: str = "track_id", interval: float = 5.0,
                 filter: str = "") -> None:
        k = str(key or "").strip()
        if not k:
            raise ValueError("门控缺少分组键字段（key）")
        self.key = k
        self.interval = float(interval)
        self.filter = str(filter or "").strip()
        self._ast = parse_condition(self.filter) if self.filter else None
        self._gate = IntervalGate(self.interval)   # 三值语义唯一实现（收编复用）
        self._last: Dict = self._gate._last        # bucket -> 上次放行 ts（reset 共享）
        self.last_stats: Dict[str, object] = {}

    @property
    def needs_track(self) -> bool:
        """是否需要上游 YOLO 开跟踪（key 取 track_id 时；Pipeline.track_needed 据此判定）。"""
        return self.key == "track_id"

    def reset(self) -> None:
        """清空门控状态与统计（新视频流复用实例时调用）。"""
        self._last.clear()
        self.last_stats = {}

    # ---- 内部 ----
    @staticmethod
    def _bucket(v) -> str:
        """key 值 → 桶名：None/""/0 → 匿名桶 "*"，其余 str(值)（铁律，见模块 docstring）。"""
        return ANON_KEY if v is None or v == "" or v == 0 else str(v)

    def _ts_of(self, it, ts: float) -> float:
        """时间源：记录自带 ts 优先、缺省用 run 注入的 ts（同 SeqFilter）。"""
        v = it.get("ts") if isinstance(it, Mapping) else getattr(it, "ts", None)
        try:
            return float(v) if v is not None else float(ts)
        except (TypeError, ValueError):
            return float(ts)

    def stats_record(self, ts: float = 0.0, frame_idx: int = 0) -> dict:
        """本次统计 → 一条观测记录（Pipeline 登记为 gate_filter.stats 产物，Records）。"""
        st = dict(self.last_stats or {})
        return {"kind": "gate_stats", "ts": ts, "frame_idx": frame_idx,
                "key": self.key, "interval": self.interval,
                "in": st.get("in", 0), "passed": st.get("passed", 0),
                "blocked": st.get("blocked", 0),
                "blocked_detail": dict(st.get("blocked_detail") or {})}

    # ---- 主入口 ----
    def run(self, dets, ts: float = 0.0, frame_idx: int = 0) -> List:
        """门控：返回放行的**原元素**（dict 进 dict 出、对象进对象出，不拷贝不包裹）。

        物理过滤：被拦记录不返回、不加任何标记字段。同帧同 key 多条 = 同帧同桶，
        首条放行后其余命中同一冷却窗口（拦）——防"匿名桶每帧只进一条"被同帧多条绕过。
        filter 未命中记录直接放行（不消耗该桶的放行额度）。
        """
        items = list(dets or [])
        passed: List = []
        blocked_detail: Dict[str, int] = {}
        for it in items:
            if self._ast is not None and not match_obj(it, self._ast):
                passed.append(it)   # filter 未命中：不受门控（直接放行）
                continue
            rts = self._ts_of(it, ts)
            bucket = self._bucket(_field_value(it, self.key))
            if self._gate.allow(bucket, rts):
                self._gate.note(bucket, rts)
                passed.append(it)
            else:
                blocked_detail[bucket] = blocked_detail.get(bucket, 0) + 1
        self.last_stats = {"in": len(items), "passed": len(passed),
                           "blocked": len(items) - len(passed),
                           "blocked_detail": blocked_detail}
        return passed
