# -*- coding: utf-8 -*-
"""
人脸扫描监控（管道节点编排: Schedule -> Detect -> Embed -> Search）

将视频帧调度器与人脸检测/嵌入/检索管道组合为"上级管道"：
按帧调度器决定的分析频率处理视频流，对每帧做人脸检测 → 特征提取 → 底库检索，
输出命中身份；可选保存命中帧标注图。

用法:
    mon = FaceMonitor(det, embedder, store)
    stats = mon.run_video("input.mp4")       # 视频
    stats = mon.run_camera(0)                # 摄像头
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from pipe.composer import FrameManager  # 帧管理：抽帧 + 背压（2026-08-18 原 FrameScheduler 并入）


@dataclass
class FaceMatch:
    """视频中一次身份命中"""

    frame_idx: int  # 视频帧序号（从 0 起）
    bbox: Tuple[int, int, int, int]  # 人脸框（原图像素）
    name: str  # 命中身份
    score: float  # 余弦相似度


@dataclass
class ScanStats:
    """一次扫描统计"""

    frames_read: int = 0  # 解码的总帧数
    frames_analyzed: int = 0  # 实际送入人脸管道的帧数（<= frames_read）
    faces_found: int = 0  # 累计检测到的人脸数
    matches: List[FaceMatch] = field(default_factory=list)  # 命中记录
    infer_sec: float = 0.0  # 人脸管道（检测+嵌入+检索）总耗时
    elapsed_sec: float = 0.0  # 整个扫描耗时（含视频解码）
    relax_events: int = 0  # 调度器放宽次数

    @property
    def avg_infer_ms(self) -> float:
        return self.infer_sec / self.frames_analyzed * 1000 if self.frames_analyzed else 0.0


class FaceMonitor:
    """视频/摄像头人脸扫描：帧调度 + 检测 + 嵌入 + 检索。"""

    def __init__(
        self,
        det,  # FaceDetector 实例
        embedder,  # FaceEmbedder 实例
        store,  # FaceStore 实例（已加载底库）
        scheduler: Optional[FrameManager] = None,
        save_dir: Optional[str] = None,  # 命中帧标注图保存目录（None=不保存）
        pending_signal: Optional[Callable[[], int]] = None,  # 队列模式的积压信号，返回待处理深度
        verbose: bool = True,
    ) -> None:
        self.det = det
        self.embedder = embedder
        self.store = store
        self.scheduler = scheduler or FrameManager(frame_skip=3)
        self.save_dir = Path(save_dir) if save_dir else None
        self.pending_signal = pending_signal
        self.verbose = verbose

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 单帧分析 ----------------
    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[FaceMatch], int]:
        """单帧：检测 → 嵌入 → 检索。

        返回 (matches, n_faces)：n_faces 为本帧检测到的人脸总数（含未命中）。
        """
        matches: List[FaceMatch] = []
        faces = self.det.detect(frame)
        for f in faces:
            if f.face_crop is None:
                continue
            emb = self.embedder.embed_face(f.face_crop)
            results = self.store.search(emb, topk=1)
            if results:  # 高于底库阈值才命中
                m = results[0]
                matches.append(FaceMatch(frame_idx, f.bbox, m.name, m.score))
                if self.verbose:
                    print(f"  [命中] 帧#{frame_idx} {m.name} score={m.score:.3f} bbox={f.bbox}")
        return matches, len(faces)

    def _save_match_frame(self, frame: np.ndarray, m: FaceMatch) -> None:
        # 直接画命中框（FaceMatch 不是 FaceDetection，不能复用 FaceDetector.draw）
        canvas = frame.copy()
        x1, y1, x2, y2 = m.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{m.name} {m.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        path = self.save_dir / f"match_{m.frame_idx}_{m.name}_{m.score:.2f}.jpg"
        cv2.imwrite(str(path), canvas)
        if self.verbose:
            print(f"      已保存: {path}")

    # ---------------- 流式处理 ----------------
    def run_stream(self, frame_iter: Iterator[np.ndarray], source_fps: float) -> ScanStats:
        """通用帧流循环。frame_iter: 逐帧产生 BGR ndarray 的迭代器。"""
        stats = ScanStats()
        self.scheduler.reset()
        t_start = time.time()

        for frame_idx, frame in enumerate(frame_iter):
            stats.frames_read += 1

            # 帧调度决策（积压信号：队列模式取 pending_signal，否则自适应）
            pending = self.pending_signal() if self.pending_signal else None
            if not self.scheduler.decide(frame_idx, pending=pending):
                continue

            # 送入人脸管道
            stats.frames_analyzed += 1
            t0 = time.time()
            matches, n_faces = self.analyze_frame(frame, frame_idx)
            cost = time.time() - t0
            stats.infer_sec += cost
            stats.faces_found += n_faces

            # 自适应模式反馈（队列模式不需要耗时反馈）
            if self.pending_signal is None:
                self.scheduler.note_inference(cost, self.scheduler.budget(source_fps))

            stats.matches.extend(matches)
            for m in matches:
                if self.save_dir:
                    self._save_match_frame(frame, m)

        stats.elapsed_sec = time.time() - t_start
        stats.relax_events = self.scheduler.relax_events
        return stats

    # ---------------- 视频 / 摄像头 ----------------
    def run_video(self, video_path: str) -> ScanStats:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        def frames():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame

        stats = self.run_stream(frames(), fps)
        cap.release()
        return stats

    def run_camera(self, cam_id: int) -> ScanStats:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise IOError(f"无法打开摄像头 {cam_id}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        def frames():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame

        try:
            return self.run_stream(frames(), fps)
        finally:
            cap.release()
