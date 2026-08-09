# -*- coding: utf-8 -*-
"""
人脸检测模块命令行测试工具（独立验证 FaceDetector 节点）

用法示例:
    # 单图检测，输出标注图到 ./out
    python test_face_detect.py --image test1.jpg

    # 多图 + 指定设备/阈值
    python test_face_detect.py --image a.jpg b.jpg --device auto --thresh 0.4

    # 视频逐帧检测（统计平均 FPS），可选保存标注视频
    python test_face_detect.py --video input.mp4 --save-dir out/

    # 摄像头
    python test_face_detect.py --camera 0
"""

import argparse
import time
from pathlib import Path

import cv2

from face_detector import FaceDetector


def parse_args():
    p = argparse.ArgumentParser(description="人脸检测模块独立验证工具")
    p.add_argument("--image", nargs="+", help="输入图片路径（可多个）")
    p.add_argument("--video", help="输入视频路径")
    p.add_argument("--camera", type=int, help="摄像头编号（如 0）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--thresh", type=float, default=0.5, help="检测置信度阈值")
    p.add_argument("--det-size", default="640,640", help="检测输入分辨率 宽,高")
    p.add_argument("--save-dir", default="out", help="标注输出目录")
    p.add_argument("--no-crop", action="store_true", help="关闭对齐裁剪")
    p.add_argument("--max-num", type=int, default=0, help="单图最大人脸数，0=不限")
    return p.parse_args()


def _det_size(s: str):
    w, h = s.split(",")
    return int(w.strip()), int(h.strip())


def run_image(det: FaceDetector, img_path: str, save_dir: Path) -> None:
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [跳过] 无法读取: {img_path}")
        return
    t0 = time.time()
    faces = det.detect(img)
    cost_ms = (time.time() - t0) * 1000

    print(f"[图片] {img_path} | 尺寸 {img.shape[1]}x{img.shape[0]} | "
          f"检测到 {len(faces)} 张人脸 | 耗时 {cost_ms:.0f}ms")
    for f in faces:
        print(f"    #{f.index} bbox={f.bbox} score={f.score:.3f} "
              f"kps={'有' if f.kps else '无'} crop={None if f.face_crop is None else f.face_crop.shape}")

    if faces:
        annotated = FaceDetector.draw(img, faces)
        out_path = save_dir / f"{Path(img_path).stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  -> 标注图已保存: {out_path}")
        # 单独保存第一张对齐裁剪（供人工检查对齐质量）
        if faces[0].face_crop is not None:
            crop_path = save_dir / f"{Path(img_path).stem}_crop_0.jpg"
            cv2.imwrite(str(crop_path), faces[0].face_crop)
            print(f"  -> 对齐裁剪已保存: {crop_path}")


def run_video(det: FaceDetector, video_path: str, save_dir: Path, save_out: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_out:
        out_path = save_dir / f"{Path(video_path).stem}_detected.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
        )

    frame_idx, total_faces, infer_time = 0, 0, 0.0
    t_start = time.time()
    print(f"[视频] {video_path} | {w}x{h} @{fps_src:.0f}fps | 总帧 {total}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        faces = det.detect(frame)
        infer_time += time.time() - t0
        frame_idx += 1
        total_faces += len(faces)
        if writer is not None:
            writer.write(FaceDetector.draw(frame, faces))
        if frame_idx % 50 == 0:
            print(f"  帧 {frame_idx}/{total} | 人脸 {len(faces)} | 累计 {total_faces}")
    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[完成] 处理 {frame_idx} 帧 | 检测到 {total_faces} 张人脸 | "
          f"平均 {avg_fps:.1f} fps | 推理占 {(infer_time / elapsed * 100) if elapsed else 0:.0f}%")


def run_camera(det: FaceDetector, cam_id: int) -> None:
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {cam_id}")
        return
    print(f"[摄像头] #{cam_id} 开始实时检测，Ctrl+C 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        frame = FaceDetector.draw(frame, faces)
        # WSL2 无窗口环境：仅打印，可自行用 cv2.imshow
        # cv2.imshow("face-detect", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        print(f"\r人脸数: {len(faces)}", end="")
    cap.release()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    det = FaceDetector(
        device=args.device,
        det_thresh=args.thresh,
        det_size=_det_size(args.det_size),
        max_num=args.max_num,
        use_crop=not args.no_crop,
    )

    if args.image:
        for img_path in args.image:
            run_image(det, img_path, save_dir)
    elif args.video:
        run_video(det, args.video, save_dir, save_out=True)
    elif args.camera is not None:
        run_camera(det, args.camera)
    else:
        print("请提供 --image / --video / --camera 之一")


if __name__ == "__main__":
    main()
