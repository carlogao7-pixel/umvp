# -*- coding: utf-8 -*-
"""
视频嵌入链路阶段（VideoEmbedStage）——PIPE0004「视频向量化-嵌入查询」核心阶段

链路形态（设计稿 doc/design/video_review_search.md §3.1，入库即目的，与 PIPE0003
"截取落盘即目的"同型）：
    帧管理（复用 pipe.composer.FrameManager 取帧节奏 + 背压）
      → 视频嵌入（本阶段：片段聚帧 → WEMMClient 出 clip+frame 向量 → EmbedStore 入库）
    无 YOLO、无告警——索引完成记录即产出（链路尾事件可经 records.events 提炼）。

职责与契约：
  - **片段窗口**（§3.3，默认 5s 窗 × 6 帧）：聚合按 frames_per_clip 帧数切窗
    （帧管理节奏固定，帧数即时间）；t_start/t_end 取窗口首末帧 ts；尾段余帧出
    partial 片段（flush_partial 可关）；
  - **观察段泛化（WO-0033.2 / hybrid_retrieval.md §2.4 §12.1 契约 B）**：第二入端口
    `dets`（Records，**可缺省**）吃上游任意识别记录流（yolo.dets / plate_recog.plates /
    face_detect.faces，单观察源直连）；逐帧累计观察数据（内存），满窗时经
    `label_align.aggregate_frame_stats` 聚合成窗口级观察层，写片段记录旁路键
    `observation`（= {det_stats, class_names, plate_nos, face_count, frame_dets}；
    frame_dets = 过闸原始记录逐帧分组，剥 model_path/旁路键，WO-0033.10 第四轮）；
    **标签绝不写 EmbedStore meta**（store 只留窗口骨架元数据，标签落 review_index）；
  - **产出 = 每片段一条 dict 记录**（kind="clip"，携带片段元数据）；clip 向量走
    记录旁路字段 `embedding`（records 契约：**像素与向量不进 JSON 主干**）；
    每条记录附事件 clip_indexed（链路尾 collect_events 汇总，WO-0029 提炼块可用）；
  - 缩略图 = 窗口首帧缩放落盘（thumb_dir；缺省不落盘），路径写记录与 store 元数据；
    **路径一律项目相对**（传 root=项目根；未传时退 CWD 相对，绝不出机器绝对路径 —
    §2.5 原则 4：换机不失效）；
  - **frames_embed 开关**（默认 True 保持现状）：False 时只嵌 clip 向量，不请求也不存
    frame 向量（查询仍可用，播放定位退化到窗口头）。

装配扩展点（本单预留，web 子单接线；**不改 pipe.composer，零侵入既有模块**）：
  对齐 face_embed 先例（server.py `_chain_from_spec` 以非 dataclass 字段附挂
  `pipe.face_embedder`，驱动层逐帧消费）：
    1) server._PORT_DECL_SOURCES 登记
       {"video_embed": ("video_embed.video_embed_stage", "VideoEmbedStage")}
       ——类属性 INPUTS/OUTPUTS 自描述即进设计器连线 UI 与 validate_chain；
    2) server._build_stage_instance 增 video_embed 分支：按 preset 参数构造
       WEMMClient + EmbedStore + VideoEmbedStage（mock 档零模型加载，装配零副作用）；
    3) 附挂 `pipe.video_embed_stage = stage`；驱动层帧循环内
       `recs = stage.run(frame, ts, frame_idx, dets=<上游识别记录>)` 逐帧推入
       （dets 可缺省）、循环后 `stage.finish()`，
       返回的片段记录走 `pipe.publish(recs, ts)` 进链路尾事件流。
  PIPE0004 spec 形态建议（web 子单落地）：
    stages = [frame_manager(frame_skip≈15), video_embed(参数见 PARAMS 键清单)]
    ——video_embed 为终段（终段白名单扩充时一并登记）。
    观察段（hybrid_retrieval.md §2.4）：`frame_manager → yolo → det_filter → video_embed`
    （YOLO 观察源插 det_filter 只过有效性筛选）或 `frame_manager → yolo → target_crop
    → plate_recog / face_detect → video_embed`（识别源自带 score 质量闸，不插 det_filter）。

用法:
    stage = VideoEmbedStage(WEMMClient(), EmbedStore(), video="demo.mp4",
                            thumb_dir="out/video_thumbs", root="/path/to/project")
    for frame, ts, dets in frames:
        recs = stage.run(frame, ts=ts, dets=dets)   # 片段满窗时返回 [片段记录]
    recs += stage.finish()                          # 冲刷尾段（partial）
    store.save("out/video_vecs.npz")
    tags = [r["observation"] for r in recs]         # 观察层标签（不落 store meta）

依赖: numpy, opencv-python（缩略图）；嵌入经 WEMMClient（mock 档零网络）
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipe.records import add_event, to_record

from .embed_store import EmbedStore
from .label_align import aggregate_frame_stats
from .wemm_client import WEMMClient


def _safe_name(name: str) -> str:
    """视频名 → 文件名安全片段（路径分隔符/空白等替换为下划线）。"""
    return re.sub(r"[^\w.-]+", "_", str(name or "video")).strip("_") or "video"


def _frame_observation(ts: float, dets) -> dict:
    """一帧的上游识别记录 → 观察数据（按记录 kind 分桶：det / plate / face）。

    `dets` 端口泛化（§2.4）：任意识别记录流都收；kind=plate / face 各归其桶
    （车牌号集合、人脸计数），其余（kind=det 及未标 kind 的记录）归 dets 桶
    （类别/track 统计）。时间戳取本帧 ts。
    """
    fr = {"ts": float(ts), "dets": [], "plates": [], "faces": []}
    if dets is None:
        return fr
    items = [dets] if isinstance(dets, Mapping) else list(dets)
    for rec in items:
        r = rec if isinstance(rec, Mapping) else to_record(rec)
        kind = str(r.get("kind") or "det")
        if kind == "plate":
            fr["plates"].append(r)
        elif kind == "face":
            fr["faces"].append(r)
        else:
            fr["dets"].append(r)
    return fr


class VideoEmbedStage:
    """链路中的视频嵌入阶段：逐帧推入 → 片段窗口聚合 → 嵌入 → 入库 → 片段记录。

    参数:
        client          : WEMMClient（mock 档零网络；真档走远端 WeMM）
        store           : EmbedStore（片段向量底库；维度随首片段锁定）
        video           : 视频名（upsert 键与缩略图命名；index_video 可覆盖）
        frames_per_clip : 每片段帧数（窗口切分唯一依据，默认 6）
        clip_seconds    : 标称窗口秒数（仅写入片段元数据供展示/排查，
                          不影响切分——时间由帧 ts 承载）
        thumb_dir       : 缩略图落盘目录（None=不落盘；设计稿 §3.3 窗口首帧）；
                          相对路径按 root（未传则 CWD）解析
        thumb_long      : 缩略图长边像素（默认 320）
        flush_partial   : finish() 时余帧是否出尾片段（partial=True；False=丢弃）
        frames_embed    : 是否同时嵌 frame 向量（默认 True 保持现状；False=只嵌 clip，
                          省掉每片段 N 次请求；查询仍可用，播放定位退化到窗口头）
        root            : 项目根目录（写进记录/store 的 thumb_path 以它为基准转项目相对；
                          不传 → 退 CWD 相对路径，绝不写机器绝对路径）
    """

    # 产物端口声明（自描述；validate_chain / 设计器连线 UI 依赖）：
    # 吃帧管理产帧（Image）+ 上游识别记录流（Records，**可缺省**）；产片段记录流（Records）。
    INPUTS = {"frame": "Image", "dets": "Records"}
    OUTPUTS = {"clips": "Records"}

    def __init__(self, client: WEMMClient, store: EmbedStore, video: str = "video",
                 frames_per_clip: int = 6, clip_seconds: float = 5.0,
                 thumb_dir: Optional[str] = None, thumb_long: int = 320,
                 flush_partial: bool = True, frames_embed: bool = True,
                 root: Optional[str] = None) -> None:
        if int(frames_per_clip) <= 0:
            raise ValueError(f"frames_per_clip 须为正整数，得到 {frames_per_clip!r}")
        self.client = client
        self.store = store
        self.video = str(video)
        self.frames_per_clip = int(frames_per_clip)
        self.clip_seconds = float(clip_seconds)
        self.thumb_dir = thumb_dir
        self.thumb_long = int(thumb_long)
        self.flush_partial = bool(flush_partial)
        self.frames_embed = bool(frames_embed)
        self.root = os.path.abspath(root) if root else None
        self._buf: List[Tuple[float, Optional[int], np.ndarray]] = []  # (ts, frame_idx, frame)
        self._obs_buf: List[dict] = []  # 与 _buf 行对齐的逐帧观察数据（内存，不落盘）
        self.indexed: List[dict] = []  # 累计产出的片段记录（驱动层可直接取全量）

    # ---------------- 逐帧推入 ----------------

    def run(self, frame, ts: float = 0.0, frame_idx: Optional[int] = None,
            dets=None) -> List[dict]:
        """喂入一帧（+ 该帧上游识别记录，可缺省）。片段满窗 → 嵌入 + 入库 → 返回
        [片段记录]；未满 → []。

        `dets` = 上游 Records（yolo.dets / plate_recog.plates / face_detect.faces，
        单观察源直连）；None/[] = 无观察数据（不报错，observation 全零）。"""
        if frame is None:
            return []
        self._buf.append((float(ts), frame_idx, frame))
        self._obs_buf.append(_frame_observation(ts, dets))
        if len(self._buf) >= self.frames_per_clip:
            return self._flush_window(partial=False)
        return []

    def finish(self) -> List[dict]:
        """冲刷尾段：余帧非空且 flush_partial → 出 partial 片段；否则清空缓冲。"""
        if self._buf and not self.flush_partial:
            self._buf.clear()
            self._obs_buf.clear()
            return []
        return self._flush_window(partial=True) if self._buf else []

    # ---------------- 索引整段视频（便捷入口，驱动层/测试直用） ----------------

    def index_video(self, frames: Sequence, video: Optional[str] = None,
                    ts_list: Optional[Sequence[float]] = None,
                    frame_idx0: int = 0) -> List[dict]:
        """整段帧序列一次索引（run 循环 + finish 汇总）。ts_list 缺省按标称
        均匀时间戳（clip_seconds / frames_per_clip 帧间隔）。"""
        if video:
            self.video = str(video)
        out: List[dict] = []
        step = self.clip_seconds / self.frames_per_clip
        for i, f in enumerate(frames):
            ts = float(ts_list[i]) if ts_list is not None and i < len(ts_list) else i * step
            out.extend(self.run(f, ts=ts, frame_idx=frame_idx0 + i))
        out.extend(self.finish())
        return out

    # ---------------- 片段处理 ----------------

    def _flush_window(self, partial: bool) -> List[dict]:
        buf, self._buf = self._buf, []
        obs_frames, self._obs_buf = self._obs_buf, []
        if not buf:
            return []
        ts_list = [b[0] for b in buf]
        idx_list = [b[1] for b in buf]
        frames = [b[2] for b in buf]
        t_start, t_end = ts_list[0], ts_list[-1]
        clip_vec, frame_vecs = self.client.embed_frames(
            frames, with_frames=self.frames_embed)
        # 观察层：帧级记录聚合（统计降维 + frame_dets 原始记录流，§2.5 原则 3
        # 第四轮；空观察源 frame_dets=[]，标签由 web 侧落 review_index）
        observation = aggregate_frame_stats(obs_frames)
        thumb_path = self._save_thumb(frames[0], t_start)
        # meta 骨架（契约 B：**标签不进 store meta**；标签落 review_index / 片段记录旁路）
        meta = {"video": self.video, "t_start": t_start, "t_end": t_end,
                "frame_count": len(frames), "partial": partial,
                "clip_dim": self.client.clip_dimension,
                "frame_dim": self.client.frame_dimension,
                "clip_seconds": self.clip_seconds,
                "thumb_path": thumb_path,
                "frame_ts": [round(t, 3) for t in ts_list]}
        clip_id = self.store.add_clip(clip_vec, frame_vecs, frame_ts=ts_list, meta=meta)
        # 片段记录：向量走旁路字段 embedding（不进 JSON 主干，records 契约）；
        # 观察标签走旁路键 observation（dict，web 侧组装 review_index 行用）；
        # 事件 clip_indexed 供链路尾提炼块/展示（对齐 WO-0029 output 提炼）。
        rec = {"kind": "clip", "label": "clip", "ts": t_start,
               "frame_idx": idx_list[0] if idx_list[0] is not None else 0,
               "video": self.video, "clip_id": clip_id,
               "t_start": t_start, "t_end": t_end,
               "frame_count": len(frames), "partial": partial,
               "thumb_path": thumb_path, "embedding": clip_vec,
               "observation": observation}
        add_event(rec, "clip_indexed", ts=t_start,
                  detail={"video": self.video, "t_start": t_start, "t_end": t_end,
                          "clip_id": clip_id, "frame_count": len(frames),
                          "thumb_path": thumb_path})
        self.indexed.append(rec)
        return [rec]

    # ---------------- 缩略图（项目相对路径） ----------------

    def _thumb_dir_abs(self) -> Optional[str]:
        """缩略图落盘目录绝对路径（相对 thumb_dir 按 root 解析，未传 root 则按 CWD）。"""
        if not self.thumb_dir:
            return None
        d = str(self.thumb_dir)
        if os.path.isabs(d):
            return os.path.abspath(d)
        return os.path.abspath(os.path.join(self.root or os.getcwd(), d))

    def _reported_path(self, disk_path: str) -> str:
        """磁盘路径 → 写进记录/store 的路径：项目相对（root 基准）→ CWD 相对 → 文件名。

        三重兜底保证**绝不出机器绝对路径**（§2.5 原则 4：换机不失效）。
        分隔符统一 "/"（跨机/跨平台可移植）。
        """
        bases = ([self.root] if self.root else []) + [os.getcwd()]
        for base in bases:
            try:
                rel = os.path.relpath(disk_path, base)
            except ValueError:  # Windows 跨盘
                continue
            if rel and not rel.startswith(".."):
                return rel.replace(os.sep, "/")
        try:
            return os.path.relpath(disk_path, os.getcwd()).replace(os.sep, "/")
        except ValueError:
            return os.path.basename(disk_path)

    def _save_thumb(self, frame: np.ndarray, t_start: float) -> str:
        """窗口首帧缩略图落盘（长边 thumb_long，等比、只缩不放）。

        未配 thumb_dir 返回空串；返回值为**项目相对路径**（见 _reported_path）。
        """
        out_dir = self._thumb_dir_abs()
        if not out_dir or frame is None:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        h, w = frame.shape[:2]
        scale = self.thumb_long / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        disk_path = os.path.join(
            out_dir, f"{_safe_name(self.video)}_{t_start:.3f}.jpg")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("缩略图 JPEG 编码失败")
        with open(disk_path, "wb") as f:
            f.write(buf.tobytes())
        return self._reported_path(disk_path)
