# -*- coding: utf-8 -*-
"""
人脸特征提取模块（管道节点 #2: Embed）

输入 : 112x112 ArcFace 对齐人脸裁剪（FaceDetector 的 face_crop）
输出 : 512 维 L2 归一化 embedding 向量

基于 InsightFace buffalo_l 的 w600k_r50.onnx（ArcFace 架构，LFW 99.83%）。
与检测节点解耦：本模块只接受裁剪图；整图流程由调用方组合 FaceDetector 完成。

用法:
    emb = FaceEmbedder(device="auto")
    feat = emb.embed_face(face_crop)          # 单张 -> (512,) 归一化向量
    feats = emb.embed_crops([crop1, crop2])   # 批量 -> (N, 512)

依赖: insightface, onnxruntime, numpy, opencv-python
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

# 独立项目自包含：默认加载项目内 face_models/（本文件位于 umvp/face_embed/，向上两级是项目根）。
# 传 model_root 参数仍可覆盖（如 ~/.insightface/models）。
DEFAULT_MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "face_models"
)
DEFAULT_EMBED_NAME = "w600k_r50.onnx"
EMBED_DIM = 512
EMBED_INPUT_SIZE = (112, 112)  # ArcFace 标准输入尺寸


class FaceEmbedder:
    """InsightFace ArcFace 特征提取器封装。"""

    def __init__(
        self,
        model_root: str = DEFAULT_MODEL_ROOT,
        model_name: str = DEFAULT_EMBED_NAME,
        device: str = "auto",  # auto / cuda / cpu
    ) -> None:
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        model_path = os.path.join(model_root, "buffalo_l", model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"未找到识别模型 {model_name}（路径: {model_path}）。\n"
                f"请确认 buffalo_l 模型包已解压到 {model_root}/buffalo_l/"
            )

        self._device_name = device
        available = set(ort.get_available_providers())
        if device == "auto":
            self._providers = [
                "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif device == "cuda":
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._providers = ["CPUExecutionProvider"]

        self._ctx_id = 0 if self._providers[0] == "CUDAExecutionProvider" else -1

        try:
            self.model = get_model(model_path, download=False, providers=self._providers)
        except Exception as e:
            if "CPUExecutionProvider" not in self._providers:
                print(f"[FaceEmbedder] CUDA 初始化失败({e})，回退 CPU")
                self._providers = ["CPUExecutionProvider"]
                self._ctx_id = -1
                self.model = get_model(model_path, download=False, providers=self._providers)
            else:
                raise

        self.model.prepare(ctx_id=self._ctx_id)
        print(
            f"[FaceEmbedder] 模型就绪: {model_name} | "
            f"device={'GPU(cuda)' if self._ctx_id >= 0 else 'CPU'} | "
            f"dim={EMBED_DIM}"
        )

    @property
    def is_gpu(self) -> bool:
        return self._ctx_id >= 0

    # ---------------- 特征提取 ----------------
    def embed_face(self, face_crop: np.ndarray) -> np.ndarray:
        """单张对齐裁剪 -> (512,) L2 归一化向量。"""
        if face_crop is None:
            raise ValueError("face_crop 为空，无法提取特征")
        feat = self.model.get_feat(face_crop)  # (1, 512)
        return self._normalize_rows(feat)[0]

    def embed_crops(self, crops: List[np.ndarray]) -> np.ndarray:
        """批量裁剪 -> (N, 512) 行归一化矩阵。"""
        if not crops:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        feats = self.model.get_feat(list(crops))  # (N, 512)
        return self._normalize_rows(feats)

    # ---------------- 工具 ----------------
    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        """按行 L2 归一化（幂等：已归一化的向量再归一化不变）。"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms
