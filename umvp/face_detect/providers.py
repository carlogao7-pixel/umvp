# -*- coding: utf-8 -*-
"""人脸模块公共：InsightFace 模型加载的 device/provider 选择与 CUDA 回退。

FaceDetector（SCRFD，face_detect/）与 FaceEmbedder（ArcFace，face_embed/）共用，
避免两处重复实现 provider 列表构造 / ctx_id 推导 / CUDA 初始化失败回退 CPU。

用法:
    from face_detect.providers import build_providers, load_insightface_model, DeviceAware

    class X(DeviceAware):
        def _load(self):
            providers, ctx_id = build_providers(self.device)
            self.model, providers, ctx_id = load_insightface_model(
                path, providers, ctx_id, tag="X")
            self.model.prepare(ctx_id=ctx_id)
            self._ctx_id = ctx_id
"""

from __future__ import annotations

from typing import List, Tuple


def build_providers(device: str) -> Tuple[List[str], int]:
    """device(auto/cuda/cpu) → (providers, ctx_id)。

    auto：onnxruntime 有 CUDA 提供方则优先 CUDA、否则 CPU；cuda：显式要求（失败由
    load_insightface_model 回退）；其它值（含 cpu）：仅 CPU。ctx_id=0 表示 GPU。
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "auto":
        names = ["CUDAExecutionProvider" if "CUDAExecutionProvider" in available
                 else "CPUExecutionProvider", "CPUExecutionProvider"]
    elif device == "cuda":
        names = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        names = ["CPUExecutionProvider"]
    return names, (0 if names[0] == "CUDAExecutionProvider" else -1)


def load_insightface_model(model_path: str, providers: List[str], ctx_id: int, tag: str):
    """加载 insightface 模型；全 CUDA 失败时回退 CPU。返回 (model, providers, ctx_id)。"""
    from insightface.model_zoo import get_model

    try:
        return get_model(model_path, download=False, providers=providers), providers, ctx_id
    except Exception as e:  # 全 CUDA 失败回退 CPU
        if "CPUExecutionProvider" not in providers:
            print(f"[{tag}] CUDA 初始化失败({e})，回退 CPU")
            providers = ["CPUExecutionProvider"]
            return get_model(model_path, download=False, providers=providers), providers, -1
        raise


class DeviceAware:
    """带 `_ctx_id`（0=GPU / -1=CPU）的类共用的 `is_gpu` 判定。"""

    _ctx_id: int = -1

    @property
    def is_gpu(self) -> bool:
        return self._ctx_id >= 0
