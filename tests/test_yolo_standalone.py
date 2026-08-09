# -*- coding: utf-8 -*-
"""
YOLODetector 模块单独测试：处理 tests/data/pic 下图片，仅识别 person

流程: infer(推理) → filter_only(过滤链+识别对象白名单+每类数量范围) → process(按类报送)
复用模块本身，不另写推理代码；改构造参数即可换一组配置。

运行: ai 环境 python tests/test_yolo_standalone.py
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
CLS_PERSON = 0  # COCO person

# 合理参数：基线配置，仅识别人
DETECTOR = YOLODetector(
    model_path=str(YOLO_MODEL),
    conf=0.35,     # 检出阈值：中等门槛，兼顾重叠/遮挡目标
    iou=0.7,       # NMS 阈值：允许适度重叠的框并存
    imgsz=640,     # 推理分辨率：标准尺寸，算力适中
    max_det=300,   # 单帧上限足够
    classes=[CLS_PERSON],  # 只识别人
    class_limits={CLS_PERSON: (1, 300)},  # 识别对象白名单：人 1~300 个
    check_interval=3.0,
    roi=False,
)


def main() -> None:
    t_start = time.time()
    images = sorted(DATA.glob("*.png"))
    print(f"模型: {YOLO_MODEL.name} | 识别对象: person | 图片: {[p.name for p in images]}\n")
    for img_index, img in enumerate(images):
        frame = cv2.imread(str(img))
        t0 = time.time()
        dets = DETECTOR.infer(frame)  # 模块内推理，track=False
        dt = time.time() - t0

        print("=" * 78)
        print(f"图片: {img.name} | 推理耗时 {dt:.2f}s")
        print(f"检出 person: {len(dets)} 个")
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.bbox
            w, h = x2 - x1, y2 - y1
            print(f"  #{i+1:<2} conf={d.score:<5.3f} box=({x1},{y1})-({x2},{y2})  {w}x{h}")

        # 过滤链 + 白名单 + 数量范围（filter_only），再看是否满足报送条件
        valid = DETECTOR.filter_only(dets)
        print(f"过滤后有效: {len(valid)} 个 (class_limits={DETECTOR.class_limits})")
        if valid:
            # 按类报送（一类只出一条请求，同类多目标合并）；ts 用各自独立时刻，
            # 避免同类"送审节拍"(check_interval) 把后一张图当作重复报送抑制掉
            reqs = DETECTOR.process(0, float(img_index) * 10.0, dets)
            for r in reqs:
                print(f"报送请求: gran={r.gran} ref={r.ref} track_key={r.track_key} "
                      f"该类目标数={len(r.dets)}")
            if not reqs:
                print("(有效目标已检出，但距同类上次报送不足 check_interval，未再报送)")
        else:
            print("未形成报送请求")
        print()
    print(f"总耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
