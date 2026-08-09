#!/usr/bin/env python
"""实测标定脚本（P1 实测层）。

遍历 models/*.pt（YOLO）+ 人脸模型（SCRFD det_10g.onnx / ArcFace w600k_r50.onnx），
分别做 CPU / GPU（若可用）实测：单次推理延迟 + 峰值显存/内存，写入指纹 JSON。

用法：
    /home/carl0/miniconda/envs/ai/bin/python web_lab/calibrate.py [--out out/fingerprints.json]
    /home/carl0/miniconda/envs/ai/bin/python web_lab/calibrate.py --gpu-only   # 只标定 GPU

产物结构（与 umvp/resources.py 的 _measured_lookup_key 对齐）：
    {
      "yolo.yolov8n.pt.640":  {"vram_mb":…, "ram_mb":…, "latency_ms_cpu":…, "latency_ms_gpu":…, "note":…},
      "scrfd.640,640":        {...},
      "arcface":              {...},
      "facestore":            {...},
      "_meta": {"date":…, "device":…, "note":…}
    }

机器变化（RTX 3060 → 别的卡）只需重跑本脚本，代码零改动。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 项目根
OUT_DEFAULT = ROOT / "out" / "fingerprints.json"
MODELS_DIR = ROOT / "models"
FACE_MODELS_DIR = ROOT / "face_models" / "buffalo_l"

# 推理参数（标定固定档位，避免与用户参数纠缠）
IMGSZ = 640
DET_SIZE = "640,640"
WARMUP = 2          # 预热次数
REPEAT = 5          # 计时次数，取中位数
MAX_FACES = 4       # 人脸模型一次批处理张数


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _bench(fn, repeat: int = REPEAT) -> float:
    """跑 warmup+repeat 次，返回中位延迟（ms）。"""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _median(times)


def _gpu_available() -> bool:
    """torch 的 CUDA 是否可用（YOLO GPU 标定依据）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _ort_cuda_available() -> bool:
    """onnxruntime 是否带 CUDA 提供方（人脸模型 GPU 标定依据）。"""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _peak_vram_mb() -> float:
    """当前进程的 GPU 峰值显存（MB）；无 torch 时返回 0。"""
    try:
        import torch
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except (ImportError, RuntimeError):
        return 0.0


def calibrate_yolo(models_dir: Path, gpu: bool) -> dict:
    """对 models/*.pt 逐个标定 YOLO 单帧延迟（CPU + 可选 GPU）。"""
    import torch
    from ultralytics import YOLO

    out: dict[str, dict] = {}
    names = sorted(p.name for p in models_dir.glob("*.pt"))
    if not names:
        print("[warn] models/ 下没有 .pt 文件")
        return out

    # 固定测试图（第一张可读的图片），避免每次重新生成
    test_img = _find_test_image()

    for name in names:
        model = YOLO(str(models_dir / name))
        key = f"yolo.{name}.{IMGSZ}"
        print(f"== YOLO {name} (imgsz={IMGSZ}) ==")

        entry: dict[str, float | str] = {"note": f"yolo imgsz={IMGSZ}"}

        # CPU
        try:
            lat = _bench(lambda: model.predict(test_img, imgsz=IMGSZ, device="cpu", verbose=False, max_det=10))
            entry["latency_ms_cpu"] = round(lat, 1)
            print(f"  cpu: {lat:.1f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"  cpu: FAIL {e}")
            entry["latency_ms_cpu"] = 0.0

        # GPU（可选）
        if gpu:
            try:
                torch.cuda.reset_peak_memory_stats()
                lat = _bench(lambda: model.predict(test_img, imgsz=IMGSZ, device="cuda", verbose=False, max_det=10))
                entry["latency_ms_gpu"] = round(lat, 1)
                entry["vram_mb"] = round(_peak_vram_mb(), 1)
                print(f"  gpu: {lat:.1f} ms  vram={entry['vram_mb']}MB")
            except Exception as e:  # noqa: BLE001
                print(f"  gpu: FAIL {e}")
                entry["latency_ms_gpu"] = 0.0

        del model
        torch.cuda.empty_cache() if gpu else None
        out[key] = entry

    return out


def calibrate_scrfd(face_models_dir: Path, gpu: bool) -> dict:
    """标定 SCRFD 人脸检测（det_10g.onnx）在 640x640 的延迟。

    直接复用项目 FaceDetector 类（insightface 内部负责预处理/对齐），
    保证标定值与真实链路一致。
    """
    sys.path.insert(0, str(ROOT / "umvp"))
    from face_detect.face_detector import FaceDetector

    test_img = _find_face_test_image()
    import cv2
    img = cv2.imread(str(test_img))

    out: dict[str, dict] = {}
    key = f"scrfd.{DET_SIZE}"
    entry: dict[str, float | str] = {"note": f"scrfd det_size={DET_SIZE}"}

    ort_gpu = _ort_cuda_available()
    for tag, device in (("cpu", "cpu"), ("gpu", "cuda")):
        if tag == "gpu" and not (gpu and ort_gpu):
            if tag == "gpu" and not ort_gpu:
                print("  gpu: SKIP（onnxruntime 无 CUDA 提供方）")
            continue
        try:
            det = FaceDetector(device=device, det_size=(640, 640))
            lat = _bench(lambda: det.detect(img))
            entry[f"latency_ms_{tag}"] = round(lat, 1)
            print(f"  {tag}: {lat:.1f} ms")
            del det
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}: FAIL {e}")
            entry[f"latency_ms_{tag}"] = 0.0

    out[key] = entry
    return out


def calibrate_arcface(face_models_dir: Path, gpu: bool) -> dict:
    """标定 ArcFace 512d 嵌入（w600k_r50.onnx）单张延迟。

    直接复用项目 FaceEmbedder 类，与真实链路一致。
    """
    sys.path.insert(0, str(ROOT / "umvp"))
    from face_embed.face_embedder import FaceEmbedder

    test_img = _find_face_test_image()
    import cv2
    img = cv2.imread(str(test_img))

    # 先检测再嵌入（与 face_embed 模块一致），但计时只算嵌入部分
    sys.path.insert(0, str(ROOT / "umvp"))
    from face_detect.face_detector import FaceDetector

    out: dict[str, dict] = {}
    key = "arcface"
    entry: dict[str, float | str] = {"note": "arcface 512d 单张嵌入"}

    ort_gpu = _ort_cuda_available()
    for tag, device in (("cpu", "cpu"), ("gpu", "cuda")):
        if tag == "gpu" and not (gpu and ort_gpu):
            if tag == "gpu" and not ort_gpu:
                print("  gpu: SKIP（onnxruntime 无 CUDA 提供方）")
            continue
        try:
            emb = FaceEmbedder(device=device)
            det = FaceDetector(device=device, det_size=(640, 640))
            faces = det.detect(img)
            if not faces:
                print(f"  {tag}: FAIL 未检测到人脸")
                entry[f"latency_ms_{tag}"] = 0.0
                continue
            crop = faces[0].face_crop
            lat = _bench(lambda: emb.embed_face(crop))
            entry[f"latency_ms_{tag}"] = round(lat, 1)
            print(f"  {tag}: {lat:.1f} ms")
            del emb, det
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}: FAIL {e}")
            entry[f"latency_ms_{tag}"] = 0.0

    out[key] = entry
    return out


def _find_test_image() -> Path:
    """找一张标定用测试图：优先 face_detect/test_imgs，其次 tests/data/pic。"""
    cands = [
        ROOT / "umvp" / "face_detect" / "test_imgs",
        ROOT / "tests" / "data" / "pic",
    ]
    for d in cands:
        if d.exists():
            for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg")):
                return p
    raise FileNotFoundError("未找到标定用测试图片")


def _find_face_test_image() -> Path:
    """找一张含人脸的标定图（人脸模型用）：t1.jpg 已知 6 张脸，优先；否则逐个试检。"""
    t1 = ROOT / "umvp" / "face_detect" / "test_imgs" / "t1.jpg"
    if t1.exists():
        return t1
    img = _find_test_image()
    sys.path.insert(0, str(ROOT / "umvp"))
    import cv2
    from face_detect.face_detector import FaceDetector
    det = FaceDetector(device="cpu", det_size=(640, 640))
    if det.detect(cv2.imread(str(img))):
        return img
    raise RuntimeError("未找到含人脸的标定图")


def main() -> None:
    ap = argparse.ArgumentParser(description="UMVP 资源指纹实测标定")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="指纹输出路径（默认 out/fingerprints.json）")
    ap.add_argument("--gpu-only", action="store_true", help="只标定 GPU")
    ap.add_argument("--cpu-only", action="store_true", help="只标定 CPU")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpu = _gpu_available() and not args.cpu_only
    if args.gpu_only:
        gpu = True

    print(f"GPU 可用: {gpu}")
    result: dict = {}

    print("\n===== YOLO 标定 =====")
    result.update(calibrate_yolo(MODELS_DIR, gpu=gpu))

    print("\n===== SCRFD 标定 =====")
    result.update(calibrate_scrfd(FACE_MODELS_DIR, gpu=gpu))

    print("\n===== ArcFace 标定 =====")
    result.update(calibrate_arcface(FACE_MODELS_DIR, gpu=gpu))

    import platform
    result["_meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "gpu": "cuda" if gpu else "cpu",
        "note": "umvp/resources.py 加载本文件；换机器重跑本脚本即可",
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] 指纹已写入 {out_path}（{len(result) - 1} 个条目）")


if __name__ == "__main__":
    sys.exit(main())
