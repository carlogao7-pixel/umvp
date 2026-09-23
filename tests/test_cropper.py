# -*- coding: utf-8 -*-
"""
通用裁剪模块（pipe.cropper.Cropper）纯逻辑测试：不加载任何模型，秒级

覆盖:
  1. crop()        : 按框裁图（clamp 出界框 / upscale 放大 / 无效框抛错 / 像素一致）
  2. to_original() : 子图框坐标还原（upscale=1 回程一致 / upscale=2 数值断言）
  3. kps_to_original(): 关键点坐标还原
  4. extract()     : 原分辨率提取（margin=0 像素一致 / margin 外扩+clamp / 出界返回 None）
  5. iou()         : 同框=1 / 相离=0 / 半重叠=1/3
  6. 兼容别名      : CropRestore/PersonCrop/_iou 指向同一通用实现
  7. 轻量性        : 子进程 import pipe.cropper 不拉起 face/onnxruntime/ultralytics/hyperlpr3

运行: conda run -n ai python tests/test_cropper.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.cropper import Cropper, CropPatch, clamp_bbox, iou

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _demo_img(w: int = 100, h: int = 80) -> np.ndarray:
    """每像素值唯一（坐标编码），便于逐像素比对。"""
    x = np.arange(w, dtype=np.int32)
    y = np.arange(h, dtype=np.int32)
    plane = (y[:, None] * 1000 + x[None, :]).astype(np.uint8)
    return np.repeat(plane[:, :, None], 3, axis=2)


def main() -> None:
    img = _demo_img()
    h, w = img.shape[:2]

    # ---- clamp_bbox ----
    check("clamp_bbox: 出界框收紧到图内", clamp_bbox((-10, -10, 50, 90), w, h) == (0, 0, 50, 80))
    check("clamp_bbox: 界内框不变", clamp_bbox((10, 10, 50, 50), w, h) == (10, 10, 50, 50))

    # ---- crop: 基础（upscale=1）----
    tool = Cropper(upscale=1.0, margin=0.15)
    patch = tool.crop(img, (10, 20, 60, 50))
    check("crop: 子图尺寸=框尺寸", patch.img.shape[:2] == (30, 50), str(patch.img.shape))
    check("crop: 像素与原图切片一致", np.array_equal(patch.img, img[20:50, 10:60]))
    check("crop: offset 记录裁剪原点", patch.offset == (10, 20) and patch.bbox_orig == (10, 20, 60, 50))
    check("crop: scale=1.0", patch.scale == 1.0)

    # ---- crop: 出界框 clamp ----
    p2 = tool.crop(img, (-10, -10, 50, 40))
    check("crop: 出界框 clamp 后有效", p2.offset == (0, 0) and p2.img.shape[:2] == (40, 50))

    # ---- crop: 无效框抛 ValueError ----
    try:
        tool.crop(img, (200, 200, 300, 300))
        check("crop: 全出界框抛 ValueError", False)
    except ValueError:
        check("crop: 全出界框抛 ValueError", True)

    # ---- crop: 放大 ----
    up = Cropper(upscale=2.0, margin=0.0)
    p3 = up.crop(img, (10, 20, 60, 50))
    check("crop: upscale=2 尺寸翻倍", p3.img.shape[:2] == (60, 100), str(p3.img.shape))
    check("crop: upscale=2 记录 scale", p3.scale == 2.0)

    # ---- to_original: round-trip（upscale=1）----
    sub_bbox = (5, 5, 25, 20)                       # 子图坐标系里的框
    o = tool.to_original(sub_bbox, patch)
    check("to_original: upscale=1 = 子框+offset", o == (15, 25, 35, 40), str(o))

    # ---- to_original: 数值断言（upscale=2）----
    o3 = up.to_original((10, 10, 30, 40), p3)       # /2 再加 (10,20)
    check("to_original: upscale=2 除以倍数加偏移", o3 == (15, 25, 25, 40), str(o3))

    # ---- kps_to_original ----
    kps = up.kps_to_original([(2.0, 4.0), (20.0, 40.0)], p3)
    check("kps_to_original: 点还原", kps == [(11.0, 22.0), (20.0, 40.0)], str(kps))

    # ---- extract: margin=0 原分辨率像素一致 ----
    sub, fb = tool.extract(img, (10, 20, 60, 50), margin=0.0)
    check("extract: margin=0 出图尺寸=框尺寸", sub.shape[:2] == (30, 50))
    check("extract: 像素与原图切片一致（无缩放）", np.array_equal(sub, img[20:50, 10:60]))
    check("extract: 返回实际裁剪框", fb == (10, 20, 60, 50))

    # ---- extract: margin 外扩 + clamp ----
    sub2, fb2 = tool.extract(img, (0, 0, 10, 10), margin=0.5)   # 外扩 5px
    check("extract: margin 外扩", fb2 == (0, 0, 15, 15) and sub2.shape[:2] == (15, 15), str(fb2))
    _, fb3 = tool.extract(img, (90, 70, 100, 80), margin=0.5)   # 右下角外扩被 clamp
    check("extract: 外扩 clamp 到图界", fb3 == (85, 65, 100, 80), str(fb3))

    # ---- extract: 出界框返回 (None, None) ----
    check("extract: 出界框返回 None", tool.extract(img, (500, 500, 600, 600)) == (None, None))

    # ---- iou ----
    check("iou: 同框=1", iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0)
    check("iou: 相离=0", iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)
    check("iou: 半重叠=1/3", abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9)

    # ---- 兼容别名（历史命名指向同一通用实现）----
    from pipe import cropper, crop_restore
    check("别名: cropper.CropRestore is Cropper", cropper.CropRestore is Cropper)
    check("别名: cropper.PersonCrop is CropPatch", cropper.PersonCrop is CropPatch)
    check("别名: crop_restore.CropRestore 可用且同源",
          crop_restore.CropRestore is Cropper and crop_restore._iou is iou)

    # ---- 轻量性: 子进程 import 不拉起重依赖 ----
    code = (f"import sys; sys.path.insert(0, r'{ROOT / 'umvp'}'); import pipe.cropper; "
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('face_detect', 'face_embed', 'insightface', 'onnxruntime', 'ultralytics', 'hyperlpr3')]; "
            "assert not bad, bad; print('light-ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    check("轻量性: import pipe.cropper 零重依赖", r.returncode == 0 and "light-ok" in r.stdout,
          r.stderr.strip()[-160:])

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
