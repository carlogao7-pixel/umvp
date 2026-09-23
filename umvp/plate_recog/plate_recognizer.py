# -*- coding: utf-8 -*-
"""
独立车牌识别模块（车牌识别链路节点）

基于 HyperLPR3（onnxruntime + opencv，中文车牌端到端）：
  多任务检测（框 + 四角点 + 颜色）→ 四点透视矫正 → CRNN 字符识别 → 颜色分类兜底。

输入 : BGR 图像（整帧，或上游裁出的车辆区域图）
输出 : list[PlateResult]（车牌号 / 类型 / 颜色 / 置信度 / 框）

模型自包含：默认加载项目内 plate_models/hyperlpr3/（本文件向上两级到项目根）。
hyperlpr3 在 import 时检查 ~/.hyperlpr3/<版本>/，缺失会联网下载；本模块在加载前
先把项目模型同步过去（若缺失），从而做到离线可跑、模型随项目走。

依赖: hyperlpr3, onnxruntime, numpy, opencv-python
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pipe.cropper import Cropper, iou

# -------------------- 默认模型路径（项目自包含） --------------------
# 本文件位于 umvp/plate_recog/，向上两级是项目根；模型根目录下含 <版本>/onnx/*.onnx
DEFAULT_MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "plate_models", "hyperlpr3"
)
_MODEL_VERSION = "20230229"
_HYPERLPR_HOME = os.path.join(os.path.expanduser("~"), ".hyperlpr3")

# -------------------- 车牌类型映射（对齐 HyperLPR3 common.typedef） --------------------
# type_id -> (中文名, 颜色)
_TYPE_NAMES = {
    -1: ("未知", "unknown"),
    0: ("蓝牌", "blue"),
    1: ("黄牌单层", "yellow"),
    2: ("白牌单层", "white"),
    3: ("绿牌新能源", "green"),
    4: ("黑牌港澳", "black"),
    5: ("香港单层", "black"),
    6: ("香港双层", "black"),
    7: ("澳门单层", "black"),
    8: ("澳门双层", "black"),
    9: ("黄牌双层", "yellow"),
}


@dataclass
class PlateResult:
    """一张车牌识别结果（bbox 为输入图坐标系）。"""

    plate_no: str  # 车牌号（如 粤ACJ0710）
    plate_type: int  # HyperLPR3 类型编号
    type_name: str  # 类型中文名（如 绿牌新能源）
    color: str  # 颜色（blue/yellow/green/white/black/unknown）
    score: float  # 识别置信度 [0,1]
    bbox: Tuple[int, int, int, int]  # 车牌框 (x1,y1,x2,y2)，输入图坐标系
    car_bbox: Optional[Tuple[int, int, int, int]] = None  # 来源车框（原图坐标，仅两级裁剪时）
    upscale: float = 1.0  # 两级裁剪时的放大倍数（1.0=未放大）
    index: int = 0  # 序号
    plate_img: Optional[np.ndarray] = None  # 原分辨率车牌图（从原图直接裁出，含 margin；仅两级裁剪）
    plate_crop_bbox: Optional[Tuple[int, int, int, int]] = None  # plate_img 对应的原图裁剪框

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        d = {
            "plate_no": self.plate_no,
            "plate_type": self.plate_type,
            "type_name": self.type_name,
            "color": self.color,
            "score": round(float(self.score), 4),
            "bbox": list(self.bbox),
            "size": [self.width, self.height],
            "index": self.index,
        }
        if self.car_bbox is not None:
            d["car_bbox"] = list(self.car_bbox)
            d["upscale"] = self.upscale
        if self.plate_img is not None:
            d["plate_crop_bbox"] = list(self.plate_crop_bbox)
            d["plate_img_size"] = [int(self.plate_img.shape[1]), int(self.plate_img.shape[0])]
        return d


def _ensure_models(model_root: str) -> None:
    """确保 ~/.hyperlpr3/<版本>/ 存在（hyperlpr3 import 时会检查并可能联网下载）。

    优先从项目内 model_root 同步；两边都没有则给出明确错误提示。
    """
    dst = os.path.join(_HYPERLPR_HOME, _MODEL_VERSION)
    if os.path.isdir(os.path.join(dst, "onnx")):
        return
    src = os.path.join(model_root, _MODEL_VERSION)
    if not os.path.isdir(os.path.join(src, "onnx")):
        raise FileNotFoundError(
            f"未找到 HyperLPR3 模型：项目内 {src} 与默认目录 {dst} 均缺失。\n"
            f"请先联网执行 `conda run -n ai python umvp/plate_recog/fetch_models.py` 拉取模型。"
        )
    os.makedirs(_HYPERLPR_HOME, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


class PlateRecognizer:
    """车牌识别模块：封装 HyperLPR3 的检测+识别+分类流水线。

    构造参数:
        model_root   : 模型根目录（默认项目内 plate_models/hyperlpr3）
        detect_level : "high"(640) / "low"(320)；远角小车牌用 high，性能优先用 low
        logger_level : onnxruntime 日志级别（3=仅错误）
    """

    def __init__(
        self,
        model_root: Optional[str] = None,
        detect_level: str = "high",
        logger_level: int = 3,
    ) -> None:
        self.model_root = os.path.abspath(model_root or DEFAULT_MODEL_ROOT)
        _ensure_models(self.model_root)

        import hyperlpr3 as lpr3  # 惰性导入：先确保模型就位，避免 import 时联网下载

        self._lpr3 = lpr3
        self.detect_level = detect_level
        level = (lpr3.DETECT_LEVEL_HIGH
                 if str(detect_level).lower() in ("high", "1", "640")
                 else lpr3.DETECT_LEVEL_LOW)
        self._catcher = lpr3.LicensePlateCatcher(
            folder=self.model_root, detect_level=level, logger_level=logger_level
        )

    def recognize(self, img: np.ndarray) -> List[PlateResult]:
        """识别一张 BGR 图（整帧或车辆区域裁剪图）中的全部车牌。

        返回 list[PlateResult]，bbox 位于「输入图坐标系」。
        """
        if img is None or getattr(img, "size", 0) == 0:
            return []
        raw = self._catcher(img)
        results: List[PlateResult] = []
        for i, r in enumerate(raw):
            code = str(r[0])
            if not code:
                continue
            ptype = int(r[2])
            name, color = _TYPE_NAMES.get(ptype, ("未知", "unknown"))
            results.append(PlateResult(
                plate_no=code, plate_type=ptype, type_name=name, color=color,
                score=float(r[1]), bbox=tuple(int(v) for v in r[3]), index=i))
        return results


def extract_plates(
    img: np.ndarray,
    crops,
    plate_rec: PlateRecognizer,
    tool=None,
    dedup_iou: float = 0.6,
) -> List[PlateResult]:
    """对「目标裁剪」产出的车辆子图做「检牌 → 坐标还原 → 原分辨率提取」。

    本函数只做机械换算与车牌识别，不裁车（裁车由上游 pipe.target_crop 目标裁剪模块完成）：
      - img    : 原帧（车牌小图的出处，坐标还原的基准）
      - crops  : list[pipe.target_crop.CropResult]（按 YOLO 车辆框裁出的车辆子图）
      - 检牌    : plate_rec.recognize(crop.img)，由 HyperLPR3 在车图里定位/裁出车牌并识别
      - 坐标还原: 子图框 / crop.scale + crop.offset = 原图坐标（crop.to_original）
      - 原分辨率提取: 按还原后的车牌框从原帧直接裁出车牌小图（tool.extract，不缩放，margin 外扩）

    裁剪/还原/原分辨率提取复用 pipe.cropper.Cropper（通用裁剪模块，与人脸链共用）。
    """
    tool = tool or Cropper(upscale=1.0, margin=0.0)

    results: List[PlateResult] = []
    accepted: List[Tuple[int, int, int, int]] = []
    for c in crops or []:
        for p in plate_rec.recognize(c.img):
            bbox_o = c.to_original(p.bbox)
            x1, y1, x2, y2 = bbox_o
            if x2 <= x1 or y2 <= y1:
                continue
            if any(iou(bbox_o, s) > dedup_iou for s in accepted):
                continue
            plate_img, pb_final = tool.extract(img, bbox_o)
            if plate_img is not None:
                plate_img = plate_img.copy()
            accepted.append(bbox_o)
            results.append(PlateResult(
                plate_no=p.plate_no, plate_type=p.plate_type, type_name=p.type_name,
                color=p.color, score=p.score, bbox=bbox_o,
                car_bbox=tuple(int(v) for v in c.bbox), upscale=c.scale,
                index=len(results), plate_img=plate_img, plate_crop_bbox=pb_final))
    return results
