# -*- coding: utf-8 -*-
"""
重叠场景 YOLO 推理参数对比（一次性验证脚本，改用 YOLODetector 模块）

对 data/pic/ 下"人重叠"截图，用 yolov8n.pt (COCO, person=0) 跑多组推理参数，
对比检出人数/框数差异。参数按当时讨论的第一梯队（降 conf、降 iou、提 imgsz、提 max_det）。

与旧版不同：不再直接调 ultralytics，而是实例化 pipe.composer.YOLODetector 模块
（model_path 选权重 + 推理参数），只改构造参数即可换一组配置——正是
"模块化拼接、测试只调参"的用法示例。模块惰性加载模型，实例化一次即复用。

运行: ai 环境 python tests/test_yolo_params.py
"""
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.composer import YOLODetector

DATA = ROOT / "tests/data/pic"
YOLO_MODEL = ROOT / "models/yolov8n.pt"
CLS_PERSON, CLS_CAR = 0, 2  # COCO
CLS_NAME = {0: "person", 2: "car"}

# 参数组：名称 -> 模块构造参数（conf/iou/imgsz/max_det；classes 固定识别人+车）
PARAM_SETS = [
    ("基线(现配置)",    dict(conf=0.35, iou=0.7,  imgsz=640,  max_det=300)),
    ("优化A(温和)",     dict(conf=0.30, iou=0.55, imgsz=960,  max_det=500)),
    ("优化B(激进)",     dict(conf=0.25, iou=0.50, imgsz=1280, max_det=500)),
    ("优化C(降conf+iou)", dict(conf=0.25, iou=0.50, imgsz=640,  max_det=500)),
]


def make_detector(params: dict) -> YOLODetector:
    """模块实例化：model_path 决定权重，其余推理参数直接透传。"""
    return YOLODetector(model_path=str(YOLO_MODEL),
                        classes=[CLS_PERSON, CLS_CAR], **params)


def main() -> None:
    images = sorted(DATA.glob("*.png"))
    print(f"模型: {YOLO_MODEL.name} | 图片: {[p.name for p in images]}")
    for img in images:
        frame = cv2.imread(str(img))
        print("\n" + "=" * 78)
        print(f"图片: {img.name}")
        print("=" * 78)
        for name, params in PARAM_SETS:
            det = make_detector(params)
            t0 = time.time()
            dets = det.infer(frame)  # 模块内 predict，track=False
            dt = time.time() - t0
            p = params
            print(f"\n[{name}] conf={p['conf']} iou={p['iou']} imgsz={p['imgsz']} max_det={p['max_det']}  ({dt:.2f}s)")
            if not dets:
                print("  未检出 person/car")
                continue
            by_cls = {}
            for d in dets:
                by_cls.setdefault(CLS_NAME[d.cls_id], []).append(d)
            for cn, items in by_cls.items():
                print(f"  {cn}: {len(items)} 个")
                for i, d in enumerate(items):
                    x1, y1, x2, y2 = d.bbox
                    w, h = x2 - x1, y2 - y1
                    print(f"    #{i+1:<2} conf={d.score:<5} box=({x1},{y1})-({x2},{y2})  {w}x{h}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时 {time.time() - t0:.1f}s")
