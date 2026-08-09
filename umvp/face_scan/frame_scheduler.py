# -*- coding: utf-8 -*-
"""
视频帧调度器（管道节点 #0: Schedule）

参照 spaz 的抽帧与背压机制（ai/producer.py:2032-2041 + settings.py:82-83）:
    spaz 基础抽帧:  is_analysis_frame = (frame_count % min_frame_skip == 0)
    spaz 背压放宽:  队列深度 > GPU_BACKPRESSURE_THRESHOLD(32) 时
                    is_analysis_frame = (frame_count % (min_frame_skip * GPU_BACKPRESSURE_SKIP_MULTIPLIER(2)) == 0)

本模块把"抽帧间隔 + 积压放宽"抽象成可复用决策器，支持两种积压信号:

1. 队列模式（与 spaz 完全一致）:
   调用方传入待处理队列深度 pending（如 Queue.qsize()），pending > queue_threshold 时
   抽帧间隔放大 multiplier 倍；积压消化后自动恢复基础间隔。

2. 自适应模式（单线程同步推理，无外部队列）:
   推理完成后调用 note_inference(cost, budget)，用指数移动平均耗时与"帧间隔预算"
   比较：耗时超过预算则放宽（间隔放大），远低于预算则逐步恢复基础间隔。

用法:
    sch = FrameScheduler(frame_skip=3, backpressure_multiplier=2)
    if sch.decide(frame_count, pending=q.qsize()):   # 队列模式
        ...
    # 或自适应模式：
    if sch.decide(frame_count):
        t0 = time.time(); analyze(frame); sch.note_inference(time.time()-t0, sch.budget(fps))
"""

from __future__ import annotations

from typing import Optional


class FrameScheduler:
    """抽帧间隔决策器，带"模型积压自动放宽频率"能力。"""

    def __init__(
        self,
        frame_skip: int = 3,  # 基础抽帧间隔：每 N 帧分析 1 帧（对应 spaz extract_frame_rate）
        queue_threshold: int = 32,  # 队列模式：积压深度超过此值触发放宽（对应 spaz GPU_BACKPRESSURE_THRESHOLD）
        backpressure_multiplier: int = 2,  # 放宽时跳帧间隔放大倍数（对应 spaz GPU_BACKPRESSURE_SKIP_MULTIPLIER）
        max_frame_skip: Optional[int] = None,  # 放宽上限，默认 base * multiplier * 4
        adaptive_relax_ratio: float = 1.2,  # 自适应：耗时 > 预算 * 此值 -> 放宽
        adaptive_recover_ratio: float = 0.5,  # 自适应：耗时 < 预算 * 此值 -> 恢复
    ) -> None:
        self.base_skip = max(1, int(frame_skip))
        self.queue_threshold = int(queue_threshold)
        self.multiplier = max(1, int(backpressure_multiplier))
        self.max_skip = max_frame_skip or self.base_skip * self.multiplier * 4
        self.relax_ratio = adaptive_relax_ratio
        self.recover_ratio = adaptive_recover_ratio

        self._skip = self.base_skip
        self._relaxed = False
        self._relax_events = 0
        self._ema_cost = 0.0

    # ---------------- 查询状态 ----------------
    @property
    def current_skip(self) -> int:
        """当前生效的抽帧间隔（积压放宽时会大于 base_skip）。"""
        return self._skip

    @property
    def relaxed(self) -> bool:
        """是否正处于放宽状态。"""
        return self._relaxed

    @property
    def relax_events(self) -> int:
        """累计触发放宽的次数。"""
        return self._relax_events

    # ---------------- 帧决策 ----------------
    def decide(self, frame_count: int, pending: Optional[int] = None) -> bool:
        """判断第 frame_count 帧是否需要分析。

        pending: 队列模式下的积压深度（待处理帧数）。None 表示自适应模式，
                 由 note_inference() 维护的耗时统计驱动。
        """
        if pending is not None:
            # 队列模式：与 spaz 相同——积压超过阈值则放大间隔，否则用基础间隔
            if pending > self.queue_threshold:
                self._set_skip(min(self.base_skip * self.multiplier, self.max_skip), relax=True)
            else:
                self._set_skip(self.base_skip)
        return frame_count % self._skip == 0

    def budget(self, source_fps: float) -> float:
        """帧间隔预算（秒）：按当前抽帧间隔，两帧分析之间的可用时间。"""
        return self._skip / source_fps if source_fps > 0 else 0.0

    # ---------------- 自适应模式的耗时反馈 ----------------
    def note_inference(self, cost_sec: float, budget_sec: float) -> None:
        """报告一次推理耗时（秒），自适应模式下据此调整抽帧间隔。

        - cost > budget * relax_ratio   : 处理不过来，放宽（间隔放大 multiplier 倍）
        - cost < budget * recover_ratio : 很轻松，逐步恢复基础间隔
        """
        if cost_sec < 0:
            return
        self._ema_cost = 0.8 * self._ema_cost + 0.2 * cost_sec if self._ema_cost else cost_sec
        if budget_sec <= 0:
            return
        if self._ema_cost > budget_sec * self.relax_ratio:
            self._set_skip(min(self._skip * self.multiplier, self.max_skip), relax=True)
        elif self._ema_cost < budget_sec * self.recover_ratio and self._skip > self.base_skip:
            self._set_skip(max(self.base_skip, self._skip // self.multiplier))

    def reset(self) -> None:
        self._skip = self.base_skip
        self._relaxed = False
        self._ema_cost = 0.0

    # ---------------- 内部 ----------------
    def _set_skip(self, skip: int, relax: bool = False) -> None:
        skip = max(1, min(skip, self.max_skip))
        if skip != self._skip:
            if relax:
                self._relax_events += 1
            self._skip = skip
            self._relaxed = self._skip > self.base_skip
