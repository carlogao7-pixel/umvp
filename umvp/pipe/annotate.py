# -*- coding: utf-8 -*-
"""
标注模块（Annotator）：在帧上画检测框（供 VLM 看"原图 + 画框"）

职责边界：纯绘制——接收「原帧 + 上游检测框」，在帧副本上画矩形框（可带类别标签），
返回标注后的帧。不含任何模型、不改坐标、不裁图。

框筛选（`filter`，条件 DSL）：只画满足条件的框，条件**显式声明字段名**（不绑定 YOLO）。
语法与字段见 pipe.conditions：组内逗号 AND、组间分号 OR；字段 cls_id/track_id/score。
例："track_id == 1" / "cls_id == 2" / "cls_id in 0|2, score >= 0.5"。

用法:
    from pipe.annotate import annotate_frame, Annotator
    vis = annotate_frame(frame, dets, filter="cls_id == 2", show_label=True)

依赖: numpy, opencv-python, pipe.conditions
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from pipe.conditions import match_obj, parse_condition

# COCO 常用类别中文名（标注标签用；未收录的用 cls{编号}）
_COCO_NAMES = {0: "人", 1: "自行车", 2: "车", 3: "摩托", 5: "公交", 7: "卡车"}


def _label(cls_id: int) -> str:
    return _COCO_NAMES.get(int(cls_id), f"cls{cls_id}")


def annotate_frame(
    frame: np.ndarray,
    dets=None,
    *,
    thickness: int = 2,
    show_label: bool = True,
    classes: Optional[Sequence[int]] = None,
    filter: str = "",
    color=(0, 255, 0),
    text_color=(0, 255, 0),
) -> np.ndarray:
    """在帧副本上画检测框（+可选类别标签）。

    classes: 简单类别过滤（非空时只画这些类别）。
    filter : 条件 DSL（显式字段名），非空时只画满足条件的框；与 classes 同时生效（AND）。
    """
    canvas = frame.copy()
    cls_set = set(int(c) for c in classes) if classes else None
    groups = parse_condition(filter)
    for d in (dets or []):
        cls_id = int(getattr(d, "cls_id", -1))
        if cls_set is not None and cls_id not in cls_set:
            continue
        if not match_obj(d, groups):
            continue
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, int(thickness))
        if show_label:
            tid = int(getattr(d, "track_id", 0) or 0)
            txt = _label(cls_id) + (f"#{tid}" if tid else "")
            cv2.putText(canvas, txt, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color,
                        max(1, int(thickness) - 1), cv2.LINE_AA)
    return canvas


class Annotator:
    """标注阶段：run(frame, dets, ts) -> 标注后的帧（鸭子类型对齐 Pipeline.annotate）。"""

    def __init__(self, thickness: int = 2, show_label: bool = True,
                 classes: Optional[Sequence[int]] = None, filter: str = "") -> None:
        self.thickness = int(thickness)
        self.show_label = bool(show_label)
        self.classes = list(classes) if classes else None
        self.filter = filter or ""
        parse_condition(self.filter)  # 非法条件在装配时快速失败

    def run(self, frame, dets=None, ts: float = 0.0) -> np.ndarray:
        return annotate_frame(frame, dets, thickness=self.thickness,
                              show_label=self.show_label, classes=self.classes,
                              filter=self.filter)
