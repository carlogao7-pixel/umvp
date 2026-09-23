# -*- coding: utf-8 -*-
"""
车牌识别链路测试：图片 → (可选 YOLO 识车 → 裁车放大) → HyperLPR3 车牌识别 → 坐标还原

覆盖:
  1. 直接识别：整帧送 PlateRecognizer，读出车牌号/类型/颜色
  2. 两级裁剪：YOLO 识车(cls=2) → Cropper 裁车放大 → 识别 → 车牌框还原回原图
  3. 断言：车牌号格式、类型/颜色合法、框在图内、还原坐标一致
  4. 落盘：带框标注图到 tests/data/out/plates/

素材: tests/data/pic/车牌620.jpg / 车牌868.jpg（4K 帧，含新能源绿牌）
运行: conda run -n ai python tests/test_plate_recog_pipeline.py
"""
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.composer import YOLODetector, filter_min_size
from pipe.cropper import Cropper
from pipe.target_crop import crop_targets, TargetCrop
from plate_recog.plate_recognizer import PlateRecognizer, extract_plates

DATA = ROOT / "tests/data/pic"
OUT = ROOT / "tests/data/out/plates"
YOLO_MODEL = ROOT / "models/yolov8n.pt"
CLS_CAR = 2  # COCO car

# 车牌号：省份汉字 + 字母 + 5~6 位字母数字（新能源 8 位）
_PLATE_RE = re.compile(r"^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}$")

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _draw(frame, results, tag: str):
    canvas = frame.copy()
    for p in results:
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(canvas, f"{p.plate_no} {p.score:.2f}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        if p.car_bbox:
            cx1, cy1, cx2, cy2 = p.car_bbox
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{tag}.jpg"), canvas)


def main() -> None:
    t0 = time.time()
    rec = PlateRecognizer(detect_level="high")
    print(f"PlateRecognizer 就绪 ({time.time()-t0:.1f}s, model_root={rec.model_root})")

    for stem in ("车牌620", "车牌868"):
        img_path = DATA / f"{stem}.jpg"
        check(f"{stem}: 素材存在", img_path.exists(), str(img_path))
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        # ---- 1. 直接识别（整帧） ----
        t = time.time()
        direct = rec.recognize(frame)
        dt = (time.time() - t) * 1000
        print(f"  [{stem}] 直接识别 {len(direct)} 张, {dt:.0f}ms")
        for p in direct:
            print(f"      {p.plate_no}  {p.type_name}/{p.color}  {p.score:.3f}  {p.bbox}")
        check(f"{stem}: 整帧读到车牌", len(direct) >= 1, f"{len(direct)} 张")
        check(f"{stem}: 整帧直读不带原分辨率图（两级裁剪专属）",
              all(p.plate_img is None for p in direct))
        if direct:
            p = direct[0]
            check(f"{stem}: 车牌号格式", bool(_PLATE_RE.match(p.plate_no)), p.plate_no)
            check(f"{stem}: 类型/颜色合法", p.color in
                  {"blue", "yellow", "green", "white", "black", "unknown"},
                  f"{p.type_name}/{p.color}")
            x1, y1, x2, y2 = p.bbox
            check(f"{stem}: 框在图内且非空",
                  0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, str(p.bbox))
        _draw(frame, direct, f"{stem}_direct")

        # ---- 2. 目标裁剪(裁车) → 车牌识别(裁车牌+识别) → 坐标还原 ----
        # 小目标过滤归 YOLO 输出处（长边 <40 视为未识别），裁剪模块只做机械裁剪
        yolo = YOLODetector(model_path=str(YOLO_MODEL), conf=0.25, iou=0.6,
                            imgsz=1280, max_det=300, classes=[CLS_CAR],
                            filters=[filter_min_size(0, 40)])
        dets = yolo.infer(frame, track=False)  # 上游 yolo 模块的输出（已过滤）
        crops = crop_targets(frame, dets, scale=2.0, margin=0.1,
                             classes=[CLS_CAR])  # 目标裁剪
        tool = Cropper(upscale=1.0, margin=0.1)
        t = time.time()
        two = extract_plates(frame, crops, rec, tool, dedup_iou=0.6)
        dt2 = (time.time() - t) * 1000
        print(f"  [{stem}] 目标裁剪 车辆框 {len(dets)} → 车图 {len(crops)} → 车牌 {len(two)} 张, {dt2:.0f}ms")
        for p in two:
            print(f"      {p.plate_no}  {p.type_name}/{p.color}  {p.score:.3f}  "
                  f"bbox={p.bbox} car={p.car_bbox} up={p.upscale}")
        check(f"{stem}: 两级裁剪读到车牌", len(two) >= 1, f"{len(two)} 张")
        for p in two:
            x1, y1, x2, y2 = p.bbox
            check(f"{stem}: 还原框在图内", 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, str(p.bbox))
            cx1, cy1, cx2, cy2 = p.car_bbox
            check(f"{stem}: 车框在图内", 0 <= cx1 < cx2 <= w and 0 <= cy1 < cy2 <= h, str(p.car_bbox))
            # 原分辨率提取：车牌小图 = 原图按还原框 + margin(0.1) 外扩 clamp 后直接裁出
            check(f"{stem}: {p.plate_no} 带原分辨率车牌图", p.plate_img is not None)
            fw, fh = x2 - x1, y2 - y1
            ex1, ey1 = max(int(x1 - fw * 0.1), 0), max(int(y1 - fh * 0.1), 0)
            ex2, ey2 = min(int(x2 + fw * 0.1), w), min(int(y2 + fh * 0.1), h)
            check(f"{stem}: {p.plate_no} 车牌图=原图切片（无缩放）",
                  p.plate_crop_bbox == (ex1, ey1, ex2, ey2)
                  and np.array_equal(p.plate_img, frame[ey1:ey2, ex1:ex2]),
                  f"{p.plate_crop_bbox} vs {(ex1, ey1, ex2, ey2)}")
        _draw(frame, two, f"{stem}_crop")

    # ---- 3. 链路装配：Pipeline + 目标裁剪 + PlateStage（识别 → 告警 → 跨帧去重） ----
    from pipe.composer import FrameManager, AlarmPolicy, Pipeline
    from plate_recog.plate_pipeline import PlateStage

    frame = cv2.imread(str(DATA / "车牌620.jpg"))
    yolo = YOLODetector(model_path=str(YOLO_MODEL), conf=0.25, iou=0.6, imgsz=1280,
                        max_det=300, classes=[CLS_CAR], filters=[filter_min_size(0, 40)])
    crop_stage = TargetCrop(scale=2.0, margin=0.1, classes=[CLS_CAR])
    stage = PlateStage(rec, Cropper(upscale=1.0, margin=0.1), mode="crop")
    pipe = Pipeline(frame=FrameManager(frame_skip=1),
                    alarm=AlarmPolicy(task_id=2, kind="inspection"),
                    yolo=yolo, crop=crop_stage, plate=stage)
    _, al = pipe.step(0, 0.0, frame=frame, force=True)
    check("链路: 车牌阶段产出", len(pipe.last_plates) >= 1, f"{len(pipe.last_plates)} 张")
    check("链路: 车牌阶段带原分辨率图",
          all(p.plate_img is not None for p in pipe.last_plates))
    check("链路: 首帧触发告警", len(al) >= 1, f"{len(al)} 条")
    _, al2 = pipe.step(1, 1.0, frame=frame, force=True)
    check("链路: 同 track 跨帧跳过重复识别", len(pipe.last_plates) == 0,
          f"{len(pipe.last_plates)} 张")
    check("链路: 同车牌跨帧去重（不再告警）", len(al2) == 0, f"{len(al2)} 条")
    check("阶段: 空车辆框不回退整帧识别", stage.run(frame, [], 0.0) == [])

    print(f"\n全部通过: {PASS} 项断言 | 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
