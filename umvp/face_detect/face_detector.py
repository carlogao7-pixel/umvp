# -*- coding: utf-8 -*-
"""
独立人脸检测模块（管道节点 #1: Detect）

基于 InsightFace SCRFD 检测器（det_10g.onnx），只负责"定位人脸"，
不包含识别/特征提取/检索等后续环节，便于独立验证与管道化拼接。

输入 : BGR 图像 (numpy.ndarray, HxWx3)
输出 : list[FaceDetection]（bbox / 置信度 / 5点关键点 / 对齐裁剪图）

对齐裁剪（face_crop）是后续 FaceEmbed 节点的标准输入（112x112, ArcFace 模板），
在此一并产出，保证"检测→对齐"边界清晰。

依赖: insightface, onnxruntime, numpy, opencv-python
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 公共加载逻辑（provider 选择 / CUDA 回退 / is_gpu）。兼容两种导入方式：
# 包导入（face_detect.face_detector）用相对导入；以脚本方式跑 face_detect/ 下的测试时
# （该目录在 sys.path）回退为平级导入。
try:
    from .providers import DeviceAware, build_providers, load_insightface_model
except ImportError:  # pragma: no cover - 脚本运行时的兜底
    from providers import DeviceAware, build_providers, load_insightface_model

# -------------------- 默认模型路径 --------------------
# 独立项目自包含：默认加载项目内 face_models/（本文件位于 umvp/face_detect/，向上两级是项目根）。
# 传 model_root 参数仍可覆盖（如 ~/.insightface/models）。
DEFAULT_MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "face_models"
)
DEFAULT_PACK_NAME = "buffalo_l"
DEFAULT_DET_NAME = "det_10g.onnx"

# 按 pack 优先级依次尝试（优先大模型，缺哪个用哪个）
_PACK_PRIORITY = ["buffalo_l", "buffalo_s"]

# -------------------- 人脸对齐参考点（ArcFace 112x112 模板） --------------------
# 顺序与 insightface 检测关键点一致: [右眼, 左眼, 鼻尖, 右嘴角, 左嘴角]
ALIGN_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ALIGN_OUTPUT_SIZE = (112, 112)


@dataclass
class FaceDetection:
    """单张人脸检测结果（原始图像坐标系）"""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float  # 检测置信度 [0,1]
    kps: List[Tuple[float, float]]  # 5 点关键点 [(x,y)*5]
    face_crop: Optional[np.ndarray] = None  # 对齐裁剪 112x112 BGR
    index: int = 0  # 帧内序号（按置信度降序）

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "score": round(float(self.score), 4),
            "kps": [[round(float(x), 2), round(float(y), 2)] for x, y in self.kps],
            "size": [self.width, self.height],
        }


class FaceDetector(DeviceAware):
    """InsightFace SCRFD 人脸检测器封装。

    用法:
        det = FaceDetector()                      # 自动选择模型包与计算设备
        faces = det.detect(img_bgr)               # 单图
        batch = det.detect_batch([img1, img2])    # 批处理（循环，不合并批次）
    """

    def __init__(
        self,
        model_root: str = DEFAULT_MODEL_ROOT,
        pack_name: Optional[str] = None,
        det_name: str = DEFAULT_DET_NAME,
        device: str = "auto",  # auto / cuda / cpu
        det_thresh: float = 0.5,
        det_size: Tuple[int, int] = (640, 640),
        max_num: int = 0,  # 0=不限制
        use_crop: bool = True,
    ) -> None:
        self.model_root = model_root
        self.pack_name = pack_name
        self.det_name = det_name
        self.device = device
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.max_num = max_num
        self.use_crop = use_crop

        self._model_path = self._resolve_model_path()
        self._provider_names: List[str] = []
        self._ctx_id = -1
        self.det = None
        self._load()

    # ---------------- 模型文件定位 ----------------
    def _resolve_model_path(self) -> str:
        """按 pack 优先级定位检测器 ONNX 文件，找不到则给出下载提示。"""
        pack_names = [self.pack_name] if self.pack_name else _PACK_PRIORITY
        for pn in pack_names:
            cand = os.path.join(self.model_root, pn, self.det_name)
            if os.path.isfile(cand):
                return cand
            # 兼容直接传完整路径
            if os.path.isfile(pn):
                return pn
        raise FileNotFoundError(
            f"未找到人脸检测模型 {self.det_name}（已尝试 {pack_names}）。\n"
            f"请下载模型包并解压到 {self.model_root}/buffalo_l/ ：\n"
            f"  curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip\n"
            f"  解压到: {self.model_root}/buffalo_l/（本机项目内已带，检查 face_models/buffalo_l/）"
        )

    # ---------------- 加载 ----------------
    def _load(self) -> None:
        self._provider_names, self._ctx_id = build_providers(self.device)
        self.det, self._provider_names, self._ctx_id = load_insightface_model(
            self._model_path, self._provider_names, self._ctx_id, tag="FaceDetector")

        self.det.prepare(ctx_id=self._ctx_id, det_thresh=self.det_thresh, det_size=self.det_size)
        print(
            f"[FaceDetector] 模型就绪: {os.path.basename(self._model_path)} | "
            f"device={'GPU(cuda)' if self._ctx_id >= 0 else 'CPU'} | "
            f"thresh={self.det_thresh} | det_size={self.det_size}"
        )

    # ---------------- 推理 ----------------
    def detect(self, img_bgr: np.ndarray, max_num: Optional[int] = None) -> List[FaceDetection]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        limit = max_num if max_num is not None else self.max_num
        bboxes, kpss = self.det.detect(img_bgr, max_num=limit, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return []

        results: List[FaceDetection] = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            score = float(bbox[4])
            kps = (
                [[float(x), float(y)] for x, y in kpss[i]]
                if kpss is not None and i < len(kpss)
                else []
            )
            crop = self._align_crop(img_bgr, (x1, y1, x2, y2), kps) if self.use_crop else None
            results.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                score=score,
                kps=kps,
                face_crop=crop,
                index=i,
            ))
        return results

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        """多图批处理（内部逐图推理，保持调用方接口简单）。"""
        return [self.detect(img) for img in images]

    # ---------------- 对齐裁剪（供 FaceEmbed 节点复用） ----------------
    def _align_crop(
        self,
        img_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        kps: List[Tuple[float, float]],
        output_size: Tuple[int, int] = ALIGN_OUTPUT_SIZE,
    ) -> Optional[np.ndarray]:
        """5 点关键点相似变换对齐裁剪。

        - 关键点齐全时用相似变换（保形状，ArcFace 标准模板 112x112）
        - 关键点缺失时退化为 bbox 方形裁剪缩放
        """
        if len(kps) == 5:
            src = np.array(kps, dtype=np.float32)
            dst = ALIGN_REF_112.astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is not None:
                return cv2.warpAffine(
                    img_bgr, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
        # 退化路径：bbox 转方形后缩放
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * 1.2  # 留 10% margin
        half = side / 2
        sx1, sy1 = int(cx - half), int(cy - half)
        sx2, sy2 = int(cx + half), int(cy + half)
        h, w = img_bgr.shape[:2]
        sx1, sy1 = max(sx1, 0), max(sy1, 0)
        sx2, sy2 = min(sx2, w), min(sy2, h)
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        crop = img_bgr[sy1:sy2, sx1:sx2]
        return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)

    # ---------------- 可视化辅助 ----------------
    @staticmethod
    def draw(
        img_bgr: np.ndarray,
        faces: List[FaceDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        draw_kps: bool = True,
    ) -> np.ndarray:
        """在图像上绘制检测框 + 置信度 + 关键点，返回副本。"""
        canvas = img_bgr.copy()
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{f.index} {f.score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
            if draw_kps:
                for (kx, ky) in f.kps:
                    cv2.circle(canvas, (int(kx), int(ky)), 2, (0, 0, 255), -1)
        return canvas
