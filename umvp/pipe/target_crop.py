# -*- coding: utf-8 -*-
"""
目标裁剪模块（TargetCrop）：按坐标裁图，输出分辨率可配

职责边界：只做机械裁剪与坐标换算——接收「原帧 + 坐标」（通常来自上游 YOLO 检测框），
按框裁出子图；输出分辨率可配（`out_size`，默认原图）。不含任何检测器、不做目标有效性
过滤（置信度/尺寸过滤由上游识别模块在其输出处负责）；classes/filter 仅决定"裁哪些框"。

输出分辨率（`out_size`，字符串）：
    ""        原图分辨率（默认）
    "640"     长边像素：等比缩放使输出长边=640
    "640,480" 严格尺寸：拉伸到 640×480（非等比；坐标还原按 X/Y 各自比例）

下游识别模块（人脸检测 / 车牌识别）在本模块产出的子图上做二次检测与裁剪，
并用 CropResult 的 offset/scale 把结果框还原回原图坐标。

用法:
    from pipe.target_crop import crop_targets
    crops = crop_targets(frame, dets, scale=2.0, out_size="", classes=[2])
    for c in crops:
        ...  # c.img / c.bbox / c.offset / c.scale / c.track_id

依赖: numpy, opencv-python
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipe.conditions import match_obj, parse_condition
from pipe.cropper import clamp_bbox


@dataclass
class CropResult:
    """一次目标裁剪的结果：子图 + 回到原图所需信息 + 来源检测元数据。"""

    img: np.ndarray                       # 子图（已按 scale/out_size 缩放）
    bbox: Tuple[int, int, int, int]       # 实际裁剪框（原图坐标，含 margin，clamp 后）
    offset: Tuple[int, int]               # 子图左上角在原图坐标
    scale: float                          # 缩放倍数（等比时即 X/Y；非等比时为 X 向，供显示）
    cls_id: int = -1                      # 来源检测类别（无检测时为 -1）
    score: float = 0.0                    # 来源检测置信度
    track_id: int = 0                     # 来源检测 track_id（有跟踪时）
    index: int = 0                        # 序号
    scale_x: Optional[float] = None       # 严格尺寸（非等比）时的 X 向缩放；None 表示等于 scale
    scale_y: Optional[float] = None       # 同上 Y 向

    def _sx(self) -> float:
        return float(self.scale_x if self.scale_x else self.scale)

    def _sy(self) -> float:
        return float(self.scale_y if self.scale_y else self.scale)

    def to_original(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """子图坐标系里的框 → 原图坐标（除以 scale_x/scale_y 加 offset）。"""
        x1, y1, x2, y2 = bbox
        ox, oy = self.offset
        sx, sy = self._sx() or 1.0, self._sy() or 1.0
        return (int(round(x1 / sx)) + ox, int(round(y1 / sy)) + oy,
                int(round(x2 / sx)) + ox, int(round(y2 / sy)) + oy)

    def kps_to_original(self, kps) -> list:
        """子图坐标系里的关键点/点集 → 原图坐标（人脸 5 点对齐等用）。"""
        ox, oy = self.offset
        sx, sy = self._sx() or 1.0, self._sy() or 1.0
        return [((x / sx) + ox, (y / sy) + oy) for x, y in kps]

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "offset": list(self.offset),
            "scale": round(float(self.scale), 4),
            "cls_id": int(self.cls_id),
            "score": round(float(self.score), 4),
            "track_id": int(self.track_id),
            "size": [int(self.img.shape[1]), int(self.img.shape[0])],
            "index": self.index,
        }


def _parse_out_size(text) -> Tuple[float, int, Optional[Tuple[int, int]]]:
    """解析输出分辨率：'' → (1,0,None)；'640' → (1,640,None)；
    '640,480' → (1,0,(640,480))；'×2'/'2x' → (2,0,None)（倍数）。"""
    t = str(text or "").strip()
    if not t:
        return 1.0, 0, None
    # 倍数写法：×2 / x2 / 2x / *2
    m = re.fullmatch(r"[×xX*]\s*([0-9]*\.?[0-9]+)|([0-9]*\.?[0-9]+)\s*[×xX]", t)
    if m:
        v = float(m.group(1) or m.group(2))
        return (v if v > 0 else 1.0), 0, None
    nums: List[int] = []
    for p in t.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            nums.append(int(float(p)))
        except ValueError:
            return 1.0, 0, None
    if len(nums) == 1:
        return 1.0, (nums[0] if nums[0] > 0 else 0), None
    if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
        return 1.0, 0, (nums[0], nums[1])
    return 1.0, 0, None


def _interp(scale: float) -> int:
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC


def _iter_items(dets, boxes):
    """统一成 (对象, bbox) 序列：dets 用原对象（条件 DSL 取字段）；boxes 用裸框占位对象。"""
    items = []
    for d in (dets or []):
        items.append((d, tuple(int(v) for v in d.bbox)))
    for b in (boxes or []):
        items.append((_RawBox(b), tuple(int(v) for v in b)))
    return items


class _RawBox:
    """裸框占位对象（无检测元数据），供条件 DSL 取字段。"""

    cls_id = -1
    track_id = 0
    score = 0.0

    def __init__(self, bbox):
        self.bbox = tuple(int(v) for v in bbox)


def crop_targets(
    frame: np.ndarray,
    dets: Optional[Iterable] = None,
    boxes: Optional[Iterable] = None,
    *,
    scale: float = 1.0,
    margin: float = 0.0,
    classes: Optional[Sequence[int]] = None,
    filter: str = "",
    out_size=None,
) -> List[CropResult]:
    """按坐标裁图（纯机械裁剪 + 缩放，不做有效性过滤——目标有效性由上游识别模块判定）。

    frame    : 原帧（一切坐标的基准）
    dets     : 上游检测结果（取 bbox/cls_id/score/track_id）
    boxes    : 裸框列表（无检测元数据）；与 dets 二选一
    scale    : 先按倍数缩放（1.0=不缩放；>1 放大，<1 缩小）
    margin   : 裁剪外扩比例（按框宽高外扩，clamp 到图边界）
    classes  : 只裁这些类别（None/空=全部）
    filter   : 条件 DSL（显式字段名，如 "track_id == 1"）；与 classes 同时生效（AND）
    out_size : 输出分辨率（在 scale 之后应用）：""=原图；"640"=长边像素等比；"640,480"=严格尺寸
    """
    h, w = frame.shape[:2]
    cls_set = set(int(c) for c in classes) if classes else None
    groups = parse_condition(filter)
    out_scale, long_side, exact = _parse_out_size(out_size)
    total_scale = (float(scale) if scale else 1.0) * out_scale
    out: List[CropResult] = []
    for obj, bbox in _iter_items(dets, boxes):
        cls_id = int(getattr(obj, "cls_id", -1))
        score = float(getattr(obj, "score", 0.0))
        track_id = int(getattr(obj, "track_id", 0) or 0)
        if cls_set is not None and cls_id not in cls_set:
            continue
        if not match_obj(obj, groups):
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        if margin:
            fw, fh = x2 - x1, y2 - y1
            x1 -= int(fw * margin)
            y1 -= int(fh * margin)
            x2 += int(fw * margin)
            y2 += int(fh * margin)
        x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        ow, oh = x2 - x1, y2 - y1
        tw, th = ow, oh
        if abs(total_scale - 1.0) > 1e-9:
            tw, th = max(1, round(ow * total_scale)), max(1, round(oh * total_scale))
        if exact:
            tw, th = exact
        elif long_side and long_side > 0:
            cur = max(tw, th)
            if cur != long_side:
                f = long_side / cur
                tw, th = max(1, round(tw * f)), max(1, round(th * f))
        img = frame[y1:y2, x1:x2]
        if (tw, th) != (ow, oh):
            img = cv2.resize(img, (tw, th), interpolation=_interp(tw / ow))
        sx, sy = tw / ow, th / oh
        out.append(CropResult(
            img=img, bbox=(x1, y1, x2, y2), offset=(x1, y1),
            scale=sx, scale_x=sx, scale_y=sy,
            cls_id=cls_id, score=score, track_id=track_id, index=len(out)))
    return out


class TargetCrop:
    """目标裁剪阶段：按上游检测框裁子图（供车牌识别 / 人脸检测 / VLM 消费）。

    鸭子类型对齐 pipe.composer.Pipeline.crop：实现 run(frame, dets, ts) -> list[CropResult]。
    """

    def __init__(self, scale: float = 1.0, margin: float = 0.0,
                 classes: Optional[Sequence[int]] = None, filter: str = "",
                 out_size=None) -> None:
        self.scale = float(scale)
        self.margin = float(margin)
        self.classes = list(classes) if classes else None
        self.filter = filter or ""
        parse_condition(self.filter)  # 非法条件在装配时快速失败
        self.out_size = out_size

    def run(self, frame, dets=None, ts: float = 0.0) -> List[CropResult]:
        return crop_targets(frame, dets, scale=self.scale, margin=self.margin,
                            classes=self.classes, filter=self.filter,
                            out_size=self.out_size)
