# -*- coding: utf-8 -*-
"""
车牌识别链路阶段（PlateStage）

职责边界：**只做「从车辆子图裁车牌 + 识别车牌」**——车辆子图由上游「目标裁剪」模块
（pipe.target_crop）按 YOLO 车辆框裁好，本阶段在子图里定位/裁出车牌并识别，
再把车牌框还原回原图坐标、从原帧提取原分辨率车牌小图。不含车辆检测、不含裁车。

视频去重：同一辆车会在多帧被检出，若每帧都识别则频次过高。本阶段用上游 YOLO 的
跟踪 id（track_id，随目标裁剪产物透传）去重——同一 track 只识别一次，`track_cooldown`
秒后可重试（应对首次角度差/模糊未读出）。

由 web_lab 的 _chain_from_spec 注入 pipe.composer.Pipeline.plate：
    stage = PlateStage(plate_recognizer, tool, mode="crop")
    plates = stage.run(frame, crops, ts)   # crops 来自上游「目标裁剪」阶段
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pipe.composer import IntervalGate
from plate_recog.plate_recognizer import PlateRecognizer, PlateResult, extract_plates


class PlateStage:
    """链路中的车牌识别阶段：消费上游车辆子图，检牌→识别→坐标还原→原图出牌。

    参数:
        plate          : PlateRecognizer（只含 HyperLPR3 识别）
        tool           : Cropper（原分辨率出牌：margin 外扩 + 坐标还原，pipe.cropper）
        mode           : "crop"（默认，消费上游车辆子图）/ "direct"（整帧直读）
        dedup_iou      : 车牌框去重 IoU
        track_dedup    : 是否按 track_id 去重（同 track 只识别一次，视频强烈建议开）
        track_cooldown : 同一 track 重试间隔秒数；0=只识别一次（不再重试）
    """

    def __init__(
        self,
        plate: PlateRecognizer,
        tool=None,
        mode: str = "crop",
        dedup_iou: float = 0.6,
        track_dedup: bool = True,
        track_cooldown: float = 5.0,
    ) -> None:
        self.plate = plate
        self.tool = tool
        self.mode = mode
        self.dedup_iou = dedup_iou
        self.track_dedup = track_dedup
        self.track_cooldown = track_cooldown
        # 车辆框传递门控（与 yolo→vlm 的 check_interval 同一机制，key 换成 track_id）：
        #   track_cooldown > 0 → 每 cooldown 秒放行一次；= 0 → 只放行一次；track_dedup=False → 不设门
        self._gate = (IntervalGate(-1.0 if track_cooldown <= 0 else track_cooldown)
                      if track_dedup else None)

    @property
    def use_track(self) -> bool:
        """是否需要上游 YOLO 开启跟踪（Pipeline.track_needed 据此决定）。"""
        return bool(self.track_dedup and self.mode != "direct")

    def run(self, frame, crops: Optional[list] = None, ts: float = 0.0) -> List[PlateResult]:
        """识别一帧中的车牌。

        crops=None → 无上游目标裁剪阶段，整帧直读；
        crops=[]   → 上游本帧无可裁目标（没检出车 / 被门控拦截），不识别。
        """
        if self.mode == "direct" or crops is None:
            return self.plate.recognize(frame)

        keep = []
        for c in crops:
            tid = int(getattr(c, "track_id", 0) or 0)
            if tid > 0 and self._gate is not None:
                if not self._gate.allow(tid, ts):
                    continue  # 该 track 刚识别过，跳过
                self._gate.note(tid, ts)
            keep.append(c)

        return extract_plates(frame, keep, self.plate, self.tool, dedup_iou=self.dedup_iou)
