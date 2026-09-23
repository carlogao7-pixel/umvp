# -*- coding: utf-8 -*-
"""
拉取 HyperLPR3 模型到项目内 plate_models/hyperlpr3/（供换机/新克隆重建）。

pip 包不含模型；官方在首次 import 时从 http://hyperlpr.tunm.top/raw/20230229.zip
下载（约 12MB）。本脚本把这一步前置到部署阶段，产物落到项目目录，之后即可离线运行。

用法（项目根）:
    conda run -n ai python umvp/plate_recog/fetch_models.py

产物:
    plate_models/hyperlpr3/20230229/onnx/
        y5fu_320x_sim.onnx      检测（320，detect_level=low）
        y5fu_640x_sim.onnx      检测（640，detect_level=high）
        rpv3_mdict_160_r3.onnx  识别（CRNN）
        litemodel_cls_96x_r1.onnx 颜色分类（兜底）
"""

from __future__ import annotations

import os
import sys
import zipfile

import requests

_MODEL_VERSION = "20230229"
_ONLINE_URL = f"http://hyperlpr.tunm.top/raw/{_MODEL_VERSION}.zip"

# 本文件位于 umvp/plate_recog/，向上两级是项目根
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_DEST = os.path.join(_ROOT, "plate_models", "hyperlpr3")


def main() -> int:
    os.makedirs(_DEST, exist_ok=True)
    onnx_dir = os.path.join(_DEST, _MODEL_VERSION, "onnx")
    if os.path.isdir(onnx_dir) and os.listdir(onnx_dir):
        print(f"[skip] 模型已存在: {onnx_dir}")
        return 0

    zip_path = os.path.join(_DEST, f"{_MODEL_VERSION}.zip")
    print(f"[pull] {_ONLINE_URL}")
    resp = requests.get(_ONLINE_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {done/1e6:.1f}/{total/1e6:.1f}MB ({pct}%)", end="")
    print()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(_DEST)
    os.remove(zip_path)
    files = os.listdir(onnx_dir)
    print(f"[done] 已解压到 {onnx_dir}")
    for name in sorted(files):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
