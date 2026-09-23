# -*- coding: utf-8 -*-
"""车牌识别模块（PlateRecognizer + extract_plates + PlateStage）。"""

from .plate_recognizer import PlateRecognizer, PlateResult, extract_plates
from .plate_pipeline import PlateStage

__all__ = ["PlateRecognizer", "PlateResult", "extract_plates", "PlateStage"]
