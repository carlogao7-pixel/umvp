# -*- coding: utf-8 -*-
"""
人脸检测链路阶段（FaceStage）

职责边界：**只做「从目标子图检脸 + 从原图裁原分辨率人脸」**——目标子图由上游「目标裁剪」
模块（pipe.target_crop）按 YOLO 人框裁好，本阶段在子图里跑 SCRFD 检脸，
再把脸框还原回原图坐标、从原帧提取原分辨率人脸图。不含 YOLO 识人、不含裁人。

与人脸检测模型（FaceDetector/SCRFD）配套：检脸是本阶段的核心，出原图人脸图是
机械裁剪（与人脸检测同属本模块，正如车牌识别模块"检牌 + 原图出牌"）。

由 web_lab 的 _chain_from_spec 注入：
    stage = FaceStage(face_detector, Cropper(margin=0.15))
    faces = stage.run(frame, crops, ts)   # crops 来自上游「目标裁剪」阶段
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from face_detect.face_detector import FaceDetector
from pipe.cropper import Cropper, iou


@dataclass
class FaceResult:
    """一张原分辨率人脸：图 + 原图坐标 + 来源元数据。"""

    person_bbox: Tuple[int, int, int, int]    # 来源人框（原图坐标）
    face_bbox: Tuple[int, int, int, int]      # 人脸框（原图坐标，含 margin）
    face_img: np.ndarray                      # 原分辨率人脸图（从原图直接裁出）
    score: float                              # SCRFD 置信度
    kps: List[Tuple[float, float]]            # 5 点关键点（已映射回原图坐标）
    upscale: float                            # 目标裁剪的缩放倍数
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


class FaceStage:
    """链路中的人脸检测阶段：消费上游目标子图，检脸 → 坐标还原 → 原图出人脸。

    参数:
        face_det : FaceDetector（SCRFD）
        tool     : Cropper（原分辨率出脸：margin 外扩 + clamp，pipe.cropper）
        dedup_iou: 人脸框去重 IoU（上游重叠目标框会让同一张脸进多个子图）
    """

    def __init__(self, face_det: FaceDetector, tool=None, dedup_iou: float = 0.6) -> None:
        self.face_det = face_det
        self.tool = tool or Cropper(upscale=1.0, margin=0.15)
        self.dedup_iou = dedup_iou

    def run(self, frame, crops: Optional[list] = None, ts: float = 0.0) -> List[FaceResult]:
        """在目标子图里检脸并出原分辨率人脸图。

        crops=None/[] → 无上游目标裁剪产物，不检测。
        """
        results: List[FaceResult] = []
        accepted: List[Tuple[int, int, int, int]] = []  # 已接受的人脸框（原图坐标）
        for c in crops or []:
            for fd in self.face_det.detect(c.img):
                fb_orig = c.to_original(fd.bbox)
                if any(iou(fb_orig, s) > self.dedup_iou for s in accepted):
                    continue
                face_img, fb_final = self.tool.extract(frame, fb_orig)
                if face_img is None:
                    continue
                kps = c.kps_to_original(fd.kps) if fd.kps else []
                accepted.append(fb_orig)
                results.append(FaceResult(
                    person_bbox=tuple(int(v) for v in c.bbox), face_bbox=fb_final,
                    face_img=face_img, score=fd.score, kps=kps, upscale=c.scale,
                    index=len(results)))
        return results
