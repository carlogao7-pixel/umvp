# -*- coding: utf-8 -*-
"""
人脸截取链路测试（PIPE0003「人脸截取链路」核心逻辑）：帧管理 → YOLO 识人 → 原分辨率人脸提取

覆盖（不含向量嵌入/底库检索——该链路明确不做身份去重）:
  1. 帧管理门控：frame_skip=3 时仅放行 0/3/6/9 帧（纯逻辑，不加载模型）
  2. 逐帧截取：每张测试图跑 extract_faces，输出原分辨率人脸（框在图内、出图=框尺寸）
  3. 同帧去重：无两个人脸框 IoU>0.5
  4. 链路装配：Pipeline(frame, yolo) 无 vlm/plate/告警依赖，人脸截取独立于 pipe.step

运行: conda run -n ai python tests/test_face_capture_pipeline.py
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))

from face_detect.face_detector import FaceDetector
from pipe.composer import FrameManager, Pipeline, AlarmPolicy, YOLODetector
from pipe.cropper import Cropper, iou
from pipe.crop_restore import extract_faces

DATA = ROOT / "tests/data/pic"
YOLO_MODEL = ROOT / "models/yolov8n.pt"
CLS_PERSON = 0

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def main() -> None:
    # ---- 1. 帧管理门控（frame_skip=3 → 放行 0/3/6/9）----
    fm = FrameManager(sampling="analysis", frame_skip=3)
    passed = [i for i in range(10) if fm.wants_frame(i, i * 0.04)]
    check("帧管理: frame_skip=3 放行 0/3/6/9", passed == [0, 3, 6, 9], str(passed))

    # ---- 2. 链路装配（无 vlm/plate；占位告警，人脸截取不走 step）----
    yolo = YOLODetector(model_path=str(YOLO_MODEL), conf=0.3, iou=0.6, imgsz=1280,
                        max_det=300, classes=[CLS_PERSON])
    fm1 = FrameManager(sampling="analysis", frame_skip=1)
    pipe = Pipeline(frame=fm1, alarm=AlarmPolicy(task_id=3, kind="inspection"), yolo=yolo)
    check("装配: 人脸截取链无 vlm/plate 依赖", pipe.vlm is None and pipe.plate is None)
    check("装配: 无车牌/驻留阶段不需跟踪", pipe.track_needed() is False)

    # ---- 3. 逐帧截取（每张图都处理，frame_skip=1）----
    face_det = FaceDetector(det_thresh=0.4)
    tool = Cropper(upscale=1.0, margin=0.15)
    total = 0
    images = sorted(DATA.glob("*.png"))
    check("素材: 测试图存在", len(images) >= 1, f"{len(images)} 张")
    for img_path in images:
        frame = cv2.imread(str(img_path))
        check(f"{img_path.stem}: 帧管理放行", fm1.wants_frame(total, total * 0.04) is True)
        faces = extract_faces(frame, yolo, face_det, tool)
        h, w = frame.shape[:2]
        for f in faces:
            fx1, fy1, fx2, fy2 = f.face_bbox
            check(f"{img_path.stem}#{f.index} 人脸框在图内",
                  0 <= fx1 < fx2 <= w and 0 <= fy1 < fy2 <= h, str(f.face_bbox))
            check(f"{img_path.stem}#{f.index} 出图=框尺寸（原分辨率）",
                  f.face_img.shape[1] == fx2 - fx1 and f.face_img.shape[0] == fy2 - fy1,
                  f"{f.face_img.shape[1]}x{f.face_img.shape[0]}")
        dup = [1 for i in range(len(faces)) for j in range(i + 1, len(faces))
               if iou(faces[i].face_bbox, faces[j].face_bbox) > 0.5]
        check(f"{img_path.stem} 同帧去重无重复", not dup, f"重复对: {len(dup)}")
        total += len(faces)
        print(f"  图片: {img_path.name} | 截取人脸 {len(faces)} 张")

    check("总检出: 全部图合计至少 1 张人脸", total >= 1, f"{total} 张")
    print(f"\n全部通过: {PASS} 项断言 | 合计截取 {total} 张原分辨率人脸")


if __name__ == "__main__":
    main()
