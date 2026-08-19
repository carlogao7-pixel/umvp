# -*- coding: utf-8 -*-
"""
视频人脸扫描验证工具（上级管道: Schedule -> Detect -> Embed -> Search）

对视频按抽帧间隔分析人脸，检测 → 特征提取 → 底库检索，输出命中身份与统计。
支持 spaz 风格的"积压自动放宽频率"机制（参照 spaz/ai/producer.py 背压保护）：
  - 自适应模式（默认，单线程）：推理耗时超过帧间隔预算时自动放宽抽帧间隔
  - 队列模式：用 --pending-sim 模拟固定积压深度，验证阈值触发放宽

用法示例:
    # 底库仍用之前注册的 out/face_db.npz（t1.jpg 6 人 + tom_hanks）
    python test_face_scan.py --video ../spaz/data/uploads/yoga.mp4 \
        --db ../face_embed/out/face_db.npz --frame-skip 3

    # 限定帧数快速验证
    python test_face_scan.py --video ... --db ... --limit 100

    # 队列模式：模拟持续积压 40 帧（> 阈值 32），观察放宽生效
    python test_face_scan.py --video ... --db ... --scheduler-mode queue --pending-sim 40

    # 摄像头
    python test_face_scan.py --camera 0 --db ...
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_detect.face_detector import FaceDetector  # noqa: E402
from face_embed.face_embedder import FaceEmbedder  # noqa: E402
from face_embed.face_store import FaceStore  # noqa: E402
from face_scan.face_monitor import FaceMonitor  # noqa: E402
from pipe.composer import FrameManager  # noqa: E402  # 帧管理（原 FrameScheduler 并入）


def parse_args():
    p = argparse.ArgumentParser(description="视频人脸扫描验证工具")
    p.add_argument("--video", help="输入视频路径")
    p.add_argument("--camera", type=int, help="摄像头编号")
    p.add_argument("--db", default="../face_embed/out/face_db.npz", help="底库文件（.npz）")
    p.add_argument("--frame-skip", type=int, default=3, help="基础抽帧间隔（每 N 帧分析 1 帧）")
    p.add_argument("--scheduler-mode", default="adaptive", choices=["adaptive", "queue"],
                   help="积压检测方式：adaptive=单线程耗时自适应；queue=队列深度")
    p.add_argument("--pending-sim", type=int, default=0,
                   help="队列模式：模拟的固定积压深度（>0 启用，对应 spaz GPU_BACKPRESSURE_THRESHOLD）")
    p.add_argument("--multiplier", type=int, default=2, help="放宽时跳帧间隔放大倍数")
    p.add_argument("--thresh", type=float, default=0.45, help="检索相似度阈值")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--det-size", default="640,640", help="检测输入分辨率 宽,高")
    p.add_argument("--save-dir", default=None, help="命中帧标注图保存目录")
    p.add_argument("--limit", type=int, default=0, help="最多处理帧数（0=全部）")
    return p.parse_args()


def _det_size(s: str):
    w, h = s.split(",")
    return int(w.strip()), int(h.strip())


def main():
    args = parse_args()
    if not args.video and args.camera is None:
        print("请提供 --video 或 --camera")
        return

    # 底库
    store = FaceStore(thresh=args.thresh)
    db_path = Path(args.db)
    if db_path.exists():
        store.load(db_path)
        print(f"[底库] 已加载: {db_path} | {store.stats()}")
    else:
        print(f"[警告] 底库不存在: {db_path}（仅检测，不检索）")

    # 调度器（FrameManager analysis 模式 = 抽帧 + 背压）
    scheduler = FrameManager(frame_skip=args.frame_skip,
                             backpressure_multiplier=args.multiplier)
    pending_signal = None
    if args.scheduler_mode == "queue":
        # 模拟固定积压深度：每帧都报告 pending 值，验证阈值触发
        pending_signal = (lambda: args.pending_sim) if args.pending_sim > 0 else None
        if pending_signal:
            print(f"[调度] 队列模式，模拟积压 {args.pending_sim}（阈值 32，放宽倍数 x{args.multiplier}）")
        else:
            print("[调度] 队列模式，无积压信号（等效基础抽帧）")

    det = FaceDetector(device=args.device, det_size=_det_size(args.det_size))
    embedder = FaceEmbedder(device=args.device)
    monitor = FaceMonitor(det, embedder, store, scheduler=scheduler,
                          save_dir=args.save_dir, pending_signal=pending_signal)

    # 帧上限包装（测试用）
    limit = args.limit
    if limit > 0 and args.video:
        cap0 = cv2.VideoCapture(args.video)
        total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) if cap0.isOpened() else 0
        cap0.release()
        if total > limit:
            print(f"[帧上限] 仅处理前 {limit} 帧（共 {total}）")

    print(f"[扫描] 视频={args.video} 抽帧间隔={args.frame_skip} 模式={args.scheduler_mode}")
    t0 = time.time()
    if args.video:
        # 截断迭代器：--limit
        src_fps = _video_fps(args.video)
        cap = cv2.VideoCapture(args.video)
        count = 0

        def frames():
            nonlocal count
            while count < limit or limit == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                count += 1
                yield fr

        stats = monitor.run_stream(frames(), src_fps)
        cap.release()
    else:
        stats = monitor.run_camera(args.camera)

    elapsed = time.time() - t0
    # 汇总
    print("\n===== 扫描统计 =====")
    print(f"解码帧: {stats.frames_read} | 分析帧: {stats.frames_analyzed} "
          f"(抽帧率 {stats.frames_analyzed / max(1, stats.frames_read):.1%})")
    print(f"检测人脸: {stats.faces_found} | 命中: {len(stats.matches)}")
    print(f"推理耗时: {stats.infer_sec:.1f}s (均值 {stats.avg_infer_ms:.0f}ms/帧) "
          f"| 总耗时 {elapsed:.1f}s")
    print(f"调度放宽: {stats.relax_events} 次")
    if stats.matches:
        uniq = {}
        for m in stats.matches:
            uniq.setdefault(m.name, []).append(m.score)
        for name, scores in uniq.items():
            print(f"  身份 {name}: 命中 {len(scores)} 次, 最高分 {max(scores):.3f}")


def _video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    cap.release()
    return fps or 25.0


if __name__ == "__main__":
    main()
