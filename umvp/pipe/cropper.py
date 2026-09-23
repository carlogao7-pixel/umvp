# -*- coding: utf-8 -*-
"""
通用裁剪模块（Cropper）：按框裁图（可放大）+ 坐标还原 + 原分辨率提取

三句话:
  1. crop(): 按任意 bbox 从原图裁出子图（边界 clamp，可选 upscale 放大——放大只为
     提升下游检测器对小目标的召回，不改变像素出处）。
  2. to_original() / kps_to_original(): 子图坐标系里的框/关键点 还原回原图坐标系
     （除以放大倍数 + 裁剪偏移，没有别的魔法）。
  3. extract(): 从原图直接按框裁出**原生分辨率**子图（不缩放，margin 外扩 + clamp
     到图边界），即"原分辨率提取"。

纯 cv2 + numpy，不依赖任何检测器；人脸链（裁人检脸出脸，crop_restore.extract_faces）
与车牌链（裁车识牌，plate_recog.extract_plates）共用同一工具。

用法:
    from pipe.cropper import Cropper
    tool = Cropper(upscale=1.0, margin=0.15)
    patch = tool.crop(img, bbox)                  # 按框裁图（可放大，"第一次裁剪"）
    bbox_o = tool.to_original(sub_bbox, patch)    # 子图坐标系框 → 原图坐标
    sub_img, fb = tool.extract(img, bbox_o)       # 原分辨率提取（从原图直接裁，不缩放）

依赖: numpy, opencv-python
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -------------------- 数据结构 --------------------
@dataclass
class CropPatch:
    """一次"按框裁图"的结果：子图 + 回到原图所需的全部信息。"""

    img: np.ndarray                       # 子图（已按 scale 放大，若 upscale>1）
    offset: Tuple[int, int]               # 子图左上角在原图中的坐标 (ox, oy)
    scale: float                          # 放大倍数（1.0 = 未放大）
    bbox_orig: Tuple[int, int, int, int]  # 目标在原图中的框（边界 clamp 后）


# -------------------- 通用小工具 --------------------
def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    """把框收紧到图边界内。"""
    x1, y1, x2, y2 = bbox
    return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """两框 IoU（重叠框去重用，人脸/车牌链共用）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area, b_area = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    return inter / float(a_area + b_area - inter)


# -------------------- 裁剪与坐标还原 --------------------
class Cropper:
    """通用裁剪工具：按框裁图（可放大）→ 下游框/关键点坐标还原 → 原图原生分辨率出图。"""

    def __init__(
        self,
        upscale: float = 1.0,   # 子图放大倍数（>1 提升小目标召回；1.0=不放大）
        margin: float = 0.15,   # 原分辨率出图 margin 比例（应对目标略超出来源框）
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self.upscale = upscale
        self.margin = margin
        self.interpolation = interpolation

    # ---- 按框裁图 ----
    def crop(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
             scale: Optional[float] = None) -> CropPatch:
        """按 bbox 裁图（边界 clamp，可选缩放），返回子图与坐标还原信息。

        scale 为 None 时用构造时的 upscale；scale=1.0 不缩放；支持 <1 缩小。
        """
        s = float(self.upscale if scale is None else scale)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"框无效或出图: {bbox} 图尺寸 {w}x{h}")
        if abs(s - 1.0) > 1e-9:
            crop = cv2.resize(crop, None, fx=s, fy=s,
                              interpolation=self.interpolation)
        return CropPatch(img=crop, offset=(x1, y1), scale=s,
                         bbox_orig=(x1, y1, x2, y2))

    # ---- 坐标还原 ----
    def to_original(self, bbox: Tuple[int, int, int, int],
                    patch: CropPatch) -> Tuple[int, int, int, int]:
        """把子图坐标系里的框还原到原图坐标系。bbox=(x1,y1,x2,y2)。"""
        x1, y1, x2, y2 = bbox
        ox, oy = patch.offset
        s = patch.scale
        return (int(round(x1 / s)) + ox, int(round(y1 / s)) + oy,
                int(round(x2 / s)) + ox, int(round(y2 / s)) + oy)

    def kps_to_original(self, kps: List[Tuple[float, float]],
                        patch: CropPatch) -> List[Tuple[float, float]]:
        """关键点/点集同样还原到原图坐标（画图/对齐用）。"""
        ox, oy = patch.offset
        s = patch.scale
        return [((x / s) + ox, (y / s) + oy) for x, y in kps]

    # ---- 原分辨率出图 ----
    def extract(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
                margin: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """从原图直接按框裁子图（原生分辨率，不缩放）。margin 比例外扩并 clamp 到图边界。
        返回 (子图, 实际裁剪框)；框无效时返回 (None, None)。"""
        h, w = img.shape[:2]
        m = self.margin if margin is None else margin
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        px1, py1 = int(x1 - fw * m), int(y1 - fh * m)
        px2, py2 = int(x2 + fw * m), int(y2 + fh * m)
        px1, py1, px2, py2 = clamp_bbox((px1, py1, px2, py2), w, h)
        if px2 <= px1 or py2 <= py1:
            return None, None
        return img[py1:py2, px1:px2], (px1, py1, px2, py2)


# -------------------- 兼容别名（历史人脸时代命名，均指向同一通用实现） --------------------
CropRestore = Cropper
PersonCrop = CropPatch
