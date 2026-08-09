# -*- coding: utf-8 -*-
"""
截图与还原分辨率工具（CropRestore）+ 原分辨率人脸提取编排（extract_faces）

链路: 图片 → YOLODetector(识别人) → CropRestore(裁人 + 可选放大 + 坐标还原)
      → FaceDetector(SCRFD 识别人脸) → 从原图直接裁出原生分辨率人脸

关键设计（三句话）:
  1. "原分辨率" = 人脸像素永远取自原图，直接裁剪，不经过任何缩放。
  2. 放大(upscale)只作用于"喂给人脸检测器的裁剪图"，用来提升 SCRFD 对小脸的召回；
     检出的人脸框坐标 除以放大倍数 + 裁剪偏移 = 还原回原图坐标。
  3. FaceDetector(SCRFD) 输出的 bbox 已在"输入图坐标系"内（它内部处理 det_size 缩放
     并还原坐标），所以"裁剪图坐标 → 原图坐标"只需 偏移 + 缩放换算，没有别的魔法。

用法:
    from pipe.crop_restore import CropRestore, extract_faces
    tool = CropRestore(upscale=1.0, margin=0.15)
    crop = tool.crop_person(img, person_bbox)        # 裁人（可放大）
    fb = tool.to_original(face_bbox, crop)           # 人脸框还原到原图坐标
    face_img, fb2 = tool.extract_face(img, fb)       # 从原图裁出原生分辨率人脸

依赖: numpy, opencv-python, face_detect.face_detector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from face_detect.face_detector import FaceDetector


# -------------------- 数据结构 --------------------
@dataclass
class PersonCrop:
    """一次"按人截图"的结果：裁剪图 + 回到原图所需的全部信息。"""

    img: np.ndarray                       # 裁剪图（已按 scale 放大，若 upscale>1）
    offset: Tuple[int, int]               # 裁剪图左上角在原图中的坐标 (ox, oy)
    scale: float                          # 放大倍数（1.0 = 未放大）
    bbox_orig: Tuple[int, int, int, int]  # 人在原图中的框（边界 clamp 后）


@dataclass
class FaceResult:
    """一张原分辨率人脸：图 + 原图坐标 + 来源元数据。"""

    person_bbox: Tuple[int, int, int, int]    # 来源人框（原图坐标）
    face_bbox: Tuple[int, int, int, int]      # 人脸框（原图坐标，含 margin）
    face_img: np.ndarray                      # 原分辨率人脸图（从原图直接裁出）
    score: float                              # SCRFD 置信度
    kps: List[Tuple[float, float]]            # 5 点关键点（已映射回原图坐标）
    upscale: float                            # 检测时使用的放大倍数
    index: int                                # 全局序号

    def to_dict(self) -> dict:
        return {
            "person_bbox": list(self.person_bbox),
            "face_bbox": list(self.face_bbox),
            "face_size": [int(self.face_img.shape[1]), int(self.face_img.shape[0])],
            "score": round(float(self.score), 4),
            "kps": [[round(float(x), 2), round(float(y), 2)] for x, y in self.kps],
            "upscale": self.upscale,
            "index": self.index,
        }


def _clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


# -------------------- 截图与还原分辨率工具 --------------------
class CropRestore:
    """截图与还原分辨率工具：裁人（可放大）→ 下游框坐标还原 → 原图原生分辨率出图。"""

    def __init__(
        self,
        upscale: float = 1.0,   # 人裁剪图放大倍数（>1 提升小脸召回；1.0=不放大）
        margin: float = 0.15,   # 人脸出图 margin 比例（应对脸超出身体框）
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self.upscale = upscale
        self.margin = margin
        self.interpolation = interpolation

    # ---- 截图 ----
    def crop_person(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> PersonCrop:
        """按人框裁图（边界 clamp，可选放大），返回裁剪图与坐标还原信息。"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"人框无效或出图: {bbox} 图尺寸 {w}x{h}")
        if self.upscale > 1.0:
            crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale,
                              interpolation=self.interpolation)
        return PersonCrop(img=crop, offset=(x1, y1), scale=self.upscale,
                          bbox_orig=(x1, y1, x2, y2))

    # ---- 坐标还原 ----
    def to_original(self, bbox: Tuple[int, int, int, int], crop: PersonCrop) -> Tuple[int, int, int, int]:
        """把裁剪图坐标系里的框还原到原图坐标系。bbox=(x1,y1,x2,y2)。"""
        x1, y1, x2, y2 = bbox
        ox, oy = crop.offset
        s = crop.scale
        return (int(round(x1 / s)) + ox, int(round(y1 / s)) + oy,
                int(round(x2 / s)) + ox, int(round(y2 / s)) + oy)

    def kps_to_original(self, kps: List[Tuple[float, float]], crop: PersonCrop) -> List[Tuple[float, float]]:
        """5 点关键点同样还原到原图坐标（画图/对齐用）。"""
        ox, oy = crop.offset
        s = crop.scale
        return [((x / s) + ox, (y / s) + oy) for x, y in kps]

    # ---- 原分辨率出图 ----
    def extract_face(self, img: np.ndarray, face_bbox_orig: Tuple[int, int, int, int],
                     margin: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """从原图直接裁人脸（原生分辨率，不缩放）。margin 比例外扩并 clamp 到图边界。
        返回 (人脸图, 实际裁剪框)；出图时返回 (None, None)。"""
        h, w = img.shape[:2]
        m = self.margin if margin is None else margin
        x1, y1, x2, y2 = face_bbox_orig
        fw, fh = x2 - x1, y2 - y1
        px1, py1 = int(x1 - fw * m), int(y1 - fh * m)
        px2, py2 = int(x2 + fw * m), int(y2 + fh * m)
        px1, py1, px2, py2 = _clamp_bbox((px1, py1, px2, py2), w, h)
        if px2 <= px1 or py2 <= py1:
            return None, None
        return img[py1:py2, px1:px2], (px1, py1, px2, py2)


# -------------------- 编排：完整链路 --------------------
def extract_faces(
    img: np.ndarray,
    yolo,
    face_det: FaceDetector,
    tool: Optional[CropRestore] = None,
    min_person_short: int = 30,  # 过滤过小的人（长边像素低于该值不裁，避免无效人脸检测）
    dedup_iou: float = 0.6,      # 人脸去重 IoU（YOLO 重叠框会让同一张脸进多个裁剪）
) -> List[FaceResult]:
    """完整链路：YOLO 识别人 → 裁人(可放大) → SCRFD 识别人脸 → 坐标还原 → 原图裁原生分辨率人脸。

    yolo: 已配置 classes=[person] 的 YOLODetector（infer() 返回 Detection 列表）
    face_det: FaceDetector 实例
    返回: 去重后的原分辨率人脸列表（每张含原图坐标与来源人框）
    """
    tool = tool or CropRestore()
    dets = yolo.infer(img, track=False)
    persons = [d for d in dets if max(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) >= min_person_short]

    results: List[FaceResult] = []
    accepted: List[Tuple[int, int, int, int]] = []  # 已接受的人脸框（原图坐标）
    for d in persons:
        try:
            crop = tool.crop_person(img, d.bbox)
        except ValueError:
            continue
        faces = face_det.detect(crop.img)
        for fd in faces:
            fb_orig = tool.to_original(fd.bbox, crop)
            if any(_iou(fb_orig, s) > dedup_iou for s in accepted):
                continue
            face_img, fb_final = tool.extract_face(img, fb_orig)
            if face_img is None:
                continue
            kps = tool.kps_to_original(fd.kps, crop) if fd.kps else []
            accepted.append(fb_orig)
            results.append(FaceResult(
                person_bbox=d.bbox, face_bbox=fb_final, face_img=face_img,
                score=fd.score, kps=kps, upscale=tool.upscale, index=len(results)))
    return results
