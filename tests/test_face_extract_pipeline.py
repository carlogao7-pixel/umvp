# -*- coding: utf-8 -*-
"""
原分辨率人脸提取链路测试：图片 → YOLO(识别人) → Cropper(裁人+坐标还原) → SCRFD(识别人脸) → 原图裁原生分辨率人脸

覆盖:
  1. 三张 tests/data/pic 图片，YOLO 只识别人
  2. 每张图输出: 原分辨率人脸图（out/faces/{图片}/）+ 带框标注原图（验证坐标还原准确）
  3. 断言: 人脸框在原图边界内、人脸图尺寸与框一致、去重后无重复脸

运行: ai 环境 python tests/test_face_extract_pipeline.py
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))

from face_detect.face_detector import FaceDetector
from pipe.composer import YOLODetector
from pipe.cropper import Cropper
from pipe.crop_restore import extract_faces

DATA = ROOT / "tests/data/pic"
OUT = ROOT / "tests/data/out/faces"
YOLO_MODEL = ROOT / "models/yolov8n.pt"
CLS_PERSON = 0  # COCO person

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def main() -> None:
    yolo = YOLODetector(model_path=str(YOLO_MODEL), conf=0.35, iou=0.7, imgsz=640,
                        max_det=300, classes=[CLS_PERSON])
    face_det = FaceDetector(det_thresh=0.4)  # 略降阈值召回小脸
    tool = Cropper(upscale=1.0, margin=0.15)

    images = sorted(DATA.glob("*.png"))
    for img in images:
        frame = cv2.imread(str(img))
        faces = extract_faces(frame, yolo, face_det, tool)

        img_out = OUT / img.stem
        img_out.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        h, w = frame.shape[:2]
        for f in faces:
            # 原分辨率人脸图落盘
            fname = img_out / f"face_{f.index:02d}_{f.score:.3f}.png"
            cv2.imwrite(str(fname), f.face_img)
            # 标注: 绿=人框, 红=人脸框, 蓝=关键点
            px1, py1, px2, py2 = f.person_bbox
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            fx1, fy1, fx2, fy2 = f.face_bbox
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            label = f"#{f.index} {f.score:.2f}"
            cv2.putText(annotated, label, (fx1, max(fy1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            for (kx, ky) in f.kps:
                cv2.circle(annotated, (int(kx), int(ky)), 3, (255, 0, 0), -1)

            # 断言: 人脸框在图内、尺寸与出图一致
            check(f"{img.stem}#{f.index} 人脸框在图内",
                  (0 <= fx1 < fx2 <= w) and (0 <= fy1 < fy2 <= h),
                  f"bbox={f.face_bbox}")
            check(f"{img.stem}#{f.index} 人脸图尺寸=框尺寸",
                  f.face_img.shape[1] == fx2 - fx1 and f.face_img.shape[0] == fy2 - fy1,
                  f"{f.face_img.shape[1]}x{f.face_img.shape[0]} vs 框 {fx2 - fx1}x{fy2 - fy1}")

        cv2.imwrite(str(img_out / "annotated.png"), annotated)
        print(f"\n图片: {img.name} | 检出人脸: {len(faces)} 个 | 标注图: {img_out / 'annotated.png'}")
        for f in faces:
            print(f"  #{f.index} 人脸框={f.face_bbox} 尺寸={f.face_img.shape[1]}x{f.face_img.shape[0]} "
                  f"conf={f.score:.3f} upscale={f.upscale} kps={len(f.kps)}")

        # 去重断言: 无两个人脸框重叠超过 IoU 0.5
        from pipe.cropper import iou
        dup = [1 for i in range(len(faces)) for j in range(i + 1, len(faces))
               if iou(faces[i].face_bbox, faces[j].face_bbox) > 0.5]
        check(f"{img.stem} 人脸去重无重复", not dup, f"重复对: {len(dup)}")

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
