# -*- coding: utf-8 -*-
"""
人脸提取兼容编排（extract_faces）——已拆分为「目标裁剪 + 人脸检测」两个模块

新架构（推荐直接组合，见 web_lab 链路设计器）：
    帧管理 → YOLO识人 → 目标裁剪（裁人，pipe.target_crop）
           → 人脸检测（检脸 + 原图出脸，face_detect.face_pipeline.FaceStage）

本文件的 extract_faces 仅作**兼容编排**保留（历史调用/测试用）：内部即
「YOLO 识人 → crop_targets 裁人 → FaceStage 检脸出脸」，不再承担独立模块职责。
人脸专属的数据结构 FaceResult 已移至 face_detect.face_pipeline。

依赖: numpy, face_detect
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from face_detect.face_detector import FaceDetector
from face_detect.face_pipeline import FaceResult, FaceStage
from pipe.cropper import Cropper, CropPatch, iou
from pipe.target_crop import crop_targets

# 兼容旧名（历史命名 CropRestore/PersonCrop/_iou，均指向 pipe.cropper 通用实现）
CropRestore = Cropper
PersonCrop = CropPatch
_iou = iou


def extract_faces(
    img: np.ndarray,
    yolo,
    face_det: FaceDetector,
    tool: Optional[Cropper] = None,
    min_person_short: int = 30,  # 兼容参数：过滤过小的人（目标有效性过滤归上游，这里仅兜底）
    dedup_iou: float = 0.6,      # 人脸去重 IoU（YOLO 重叠框会让同一张脸进多个裁剪）
) -> List[FaceResult]:
    """兼容编排：YOLO 识人 → 目标裁剪(裁人) → 人脸检测(检脸+原图出原分辨率人脸)。

    yolo: 已配置 classes=[person] 的 YOLODetector（infer() 返回 Detection 列表）
    face_det: FaceDetector 实例
    返回: 去重后的原分辨率人脸列表（每张含原图坐标与来源人框）
    """
    tool = tool or Cropper()
    dets = yolo.infer(img, track=False)
    if min_person_short:
        dets = [d for d in dets
                if max(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) >= min_person_short]
    crops = crop_targets(img, dets, scale=(tool.upscale or 1.0), margin=0.0)
    stage = FaceStage(face_det, Cropper(upscale=1.0, margin=tool.margin),
                      dedup_iou=dedup_iou)
    return stage.run(img, crops)
