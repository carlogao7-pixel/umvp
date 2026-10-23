# -*- coding: utf-8 -*-
"""
标注模块（Annotator）：在帧上画检测框（供 VLM 看"原图 + 画框"）

职责边界：纯绘制——接收「原帧 + 上游检测框」，在帧副本上画矩形框（可带类别标签）。
不含任何模型、不改坐标、不裁图。

双模式（scope，WO-0021 定案 doc/design/assembly_ports.md §9）：
  frame（默认，现状）：整帧一张标注图（画全部上游框）→ annotate.annotated（Image）。
  record（逐记录）：每条记录单独一张整帧图、只画该记录自己的框，图挂记录 image 旁路，
                    记录流透传为 Records 产物 annotate.annotated_records（主干 dets 不动）；
                    单帧带图记录数上限 max_records（超限截断 + 告警日志）。

框筛选（`filter`，条件 DSL）：只画满足条件的框，条件**显式声明字段名**（不绑定 YOLO）。
语法与字段见 pipe.conditions：组内逗号 AND、组间分号 OR；字段 cls_id/track_id/score。
例："track_id == 1" / "cls_id == 2" / "cls_id in 0|2, score >= 0.5"。

用法:
    from pipe.annotate import annotate_frame, annotate_records, Annotator
    vis = annotate_frame(frame, dets, filter="cls_id == 2", show_label=True)

依赖: numpy, opencv-python, pipe.conditions, pipe.records
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Iterator
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipe.conditions import _field_value, match_obj, parse_condition
from pipe.records import IMAGE_KEY, to_record

# COCO 常用类别中文名（标注标签用；未收录的用 cls{编号}）
_COCO_NAMES = {0: "人", 1: "自行车", 2: "车", 3: "摩托", 5: "公交", 7: "卡车"}

# 单帧带图记录数上限默认值（record 模式；依据见设计稿 §9.3：YOLO max_det=300 是
# 纯推理检出上限，record 消费的是筛选后报送级记录流；32 张 1280×720 整帧 ≈ 88MB 峰值）
DEFAULT_MAX_RECORDS = 32


def _label(cls_id: int) -> str:
    return _COCO_NAMES.get(int(cls_id), f"cls{cls_id}")


def _visible(dets, classes: Optional[Sequence[int]],
             groups) -> Iterator[Tuple[Any, Tuple[int, int, int, int], str]]:
    """按 classes + filter 筛出该画的（记录, bbox, 标签文本）序列——画框判定唯一实现
    （annotate_frame 与 annotate_records 共用，防两处逻辑漂移）。"""
    cls_set = set(int(c) for c in classes) if classes else None
    for d in (dets or []):
        cls_id = int(_field_value(d, "cls_id"))
        if cls_set is not None and cls_id not in cls_set:
            continue
        if not match_obj(d, groups):
            continue
        b = d.get("bbox") if isinstance(d, Mapping) else getattr(d, "bbox", None)
        if not b:
            continue  # bbox 缺失 → 无框可画，跳过
        tid = int(_field_value(d, "track_id") or 0)
        # 标签文本：记录的 label 通用字段优先（如人脸 "face"/车牌号，WO-0020.2），
        # 无 label 的检测记录回退 COCO 类别名（原行为零回归）。
        lbl = d.get("label") if isinstance(d, Mapping) else getattr(d, "label", None)
        txt = (str(lbl) if lbl else _label(cls_id)) + (f"#{tid}" if tid else "")
        yield d, tuple(int(v) for v in b), txt


def _draw(canvas: np.ndarray, bbox: Tuple[int, int, int, int], txt: str, *,
          thickness: int, show_label: bool, color, text_color) -> None:
    """在画布上画一个框（+可选标签）——绘制细节唯一实现。"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, int(thickness))
    if show_label:
        cv2.putText(canvas, txt, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color,
                    max(1, int(thickness) - 1), cv2.LINE_AA)


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
    """在帧副本上画检测框（+可选类别标签）——frame 模式（现状，整帧一张）。

    classes: 简单类别过滤（非空时只画这些类别）。
    filter : 条件 DSL（显式字段名），非空时只画满足条件的框；与 classes 同时生效（AND）。
    """
    canvas = frame.copy()
    for _d, bbox, txt in _visible(dets, classes, parse_condition(filter)):
        _draw(canvas, bbox, txt, thickness=thickness, show_label=show_label,
              color=color, text_color=text_color)
    return canvas


def annotate_records(
    frame: np.ndarray,
    dets=None,
    *,
    thickness: int = 2,
    show_label: bool = True,
    classes: Optional[Sequence[int]] = None,
    filter: str = "",
    color=(0, 255, 0),
    text_color=(0, 255, 0),
    max_records: int = DEFAULT_MAX_RECORDS,
) -> Tuple[List[dict], int]:
    """record 模式（WO-0021）：每条记录单独一张**整帧**标注图，只画该记录自己的框。

    返回 (records, dropped)：
      records —— 输入记录的浅拷贝列表（dict 记录，不污染上游），每条挂 `image` =
                  该记录专属的整帧标注图（下游 `image_of`/`image_set_of` 读取）；
      dropped —— 因超出 max_records 被截断的条数（0 = 未截断；截断保留前 N 条）。
    classes/filter 语义与 annotate_frame 一致（AND）；不满足条件的记录**原样透传**
    （不带图，保持"记录流透传输出"——筛选决定画不画图，不删记录）。
    """
    cap = max(1, int(max_records))
    groups = parse_condition(filter)
    out: List[dict] = []
    dropped = 0
    n_img = 0
    for d in (dets or []):
        rec = dict(d) if isinstance(d, Mapping) else to_record(d)
        vis = next(_visible([d], classes, groups), None)
        if vis is None:
            out.append(rec)  # filter/classes 外：透传不带图（删记录是 det_filter 的职责）
            continue
        if n_img >= cap:
            dropped += 1  # 超限截断：该记录不进产物（保护整帧图内存与下游图集量级）
            continue
        _d, bbox, txt = vis
        canvas = frame.copy()
        _draw(canvas, bbox, txt, thickness=thickness, show_label=show_label,
              color=color, text_color=text_color)
        rec[IMAGE_KEY] = canvas
        out.append(rec)
        n_img += 1
    return out, dropped


class Annotator:
    """标注阶段：run(frame, dets, ts) —— 鸭子类型对齐 Pipeline.annotate。

    scope="frame"（默认）→ 返回标注帧 ndarray（现状）；
    scope="record"       → 返回带图记录列表 list[dict]（每条 image=整帧标注图）。
    """

    # 产物端口声明：吃一张图 + 上游检测框；**双模式产物**——运行时按 scope 只登记其一
    # （WO-0021 定案，设计稿 §9.2）：frame → annotated（Image，帧级标注图）；
    # record → annotated_records（Records，逐记录带整帧图；产物名分立，类型静态可校验）。
    INPUTS = {"image": "Image", "dets": "Records"}
    OUTPUTS = {"annotated": "Image", "annotated_records": "Records"}

    SCOPE_FRAME = "frame"
    SCOPE_RECORD = "record"

    def __init__(self, thickness: int = 2, show_label: bool = True,
                 classes: Optional[Sequence[int]] = None, filter: str = "",
                 scope: str = SCOPE_FRAME,
                 max_records: int = DEFAULT_MAX_RECORDS) -> None:
        if scope not in (self.SCOPE_FRAME, self.SCOPE_RECORD):
            raise ValueError(
                f"annotate.scope 取值非法: {scope!r}（应为 'frame' 或 'record'）")
        self.thickness = int(thickness)
        self.show_label = bool(show_label)
        self.classes = list(classes) if classes else None
        self.filter = filter or ""
        self.scope = scope
        self.max_records = max(1, int(max_records))
        parse_condition(self.filter)  # 非法条件在装配时快速失败

    def run(self, frame, dets=None, ts: float = 0.0):
        if self.scope == self.SCOPE_RECORD:
            recs, dropped = annotate_records(
                frame, dets, thickness=self.thickness, show_label=self.show_label,
                classes=self.classes, filter=self.filter,
                max_records=self.max_records)
            if dropped:
                print(f"[annotate] record 模式单帧带图记录超上限 {self.max_records}，"
                      f"截断 {dropped} 条（可调高 max_records 或收紧上游筛选）",
                      file=sys.stderr)
            return recs
        return annotate_frame(frame, dets, thickness=self.thickness,
                              show_label=self.show_label, classes=self.classes,
                              filter=self.filter)
