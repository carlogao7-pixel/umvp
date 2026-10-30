# -*- coding: utf-8 -*-
"""
视频片段向量底库（EmbedStore）——PIPE0004 向量档索引的存储与检索（本机 numpy 线性）

对齐 FaceStore 先例（umvp/face_embed/face_store.py）与设计稿
doc/design/video_review_search.md §5 存储设计：
  - 形态：npz 存向量矩阵（clip 矩阵 + frame 矩阵）+ 同名 .json 存元数据；
  - 检索：numpy 线性（点积 = 余弦，向量已归一），万级片段 <10ms，无需向量库；
  - 两级索引：clip 向量粗排 + frame 向量精排（§4.1），frame 向量挂所属 clip_id + 分析帧 ts；
  - upsert 幂等：同视频重跑按 (video, t_start) 覆盖（§3.3），行数不变；
  - **库级维度固定**：建库（或首条写入）时锁定 clip_dim/frame_dim，此后维度不匹配的
    向量**拒收并指明**（§5「入库与查询必须同维度」；换维度 = 新建库，不混写）；
  - 按视频删除（重建索引用）。

用法:
    store = EmbedStore(clip_dim=1024, frame_dim=256)
    cid = store.add_clip(clip_vec, frame_vecs, frame_ts=[...],
                         meta={"video": "a.mp4", "t_start": 0.0, "t_end": 5.0})
    hits = store.search(query_vec, top_k=10, video="a.mp4")     # ① 粗排
    fine = store.search_frames(query_vec, hits[0].clip_id)      # ② 精排
    store.save("out/video_vecs.npz")

依赖: numpy（零服务/零 GPU）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClipHit:
    """片段级检索命中（① 粗排结果项）。"""

    score: float                 # 余弦相似度（向量已归一，点积即余弦）
    clip_id: int                 # 命中片段在库中的条目 id
    rank: int                    # 排名（1=最相似）
    meta: Optional[dict] = None  # 入库时附加元数据（video/t_start/t_end/thumb_path 等）


@dataclass
class FrameHit:
    """帧级检索命中（② 精排结果项，时间精定位用）。"""

    score: float
    clip_id: int
    ts: float                    # 该分析帧时间戳（命中后播放 currentTime = ts − pre_roll）
    index: int                   # 帧在片段内的序号
    rank: int


class EmbedStore:
    """内存向量底库：片段条目写入（clip + frame 双层）/ 检索 / 按视频删除 / 持久化。"""

    def __init__(self, clip_dim: Optional[int] = None,
                 frame_dim: Optional[int] = None) -> None:
        # 维度 None → 首条写入时锁定（之后不匹配报错）；显式给定 → 即刻生效
        self._clip_dim = int(clip_dim) if clip_dim else None
        self._frame_dim = int(frame_dim) if frame_dim else None
        self._clip_vecs: List[np.ndarray] = []       # 行与 _clip_meta 一一对应
        self._clip_meta: List[dict] = []
        self._frame_vecs: List[np.ndarray] = []      # 行与 _frame_meta 一一对应
        self._frame_meta: List[dict] = []            # {"clip_id": int, "ts": float}
        self._key_index: Dict[Tuple[str, float], int] = {}  # (video, t_start) -> clip_id

    # ---------------- 维度 ----------------

    @property
    def clip_dim(self) -> Optional[int]:
        """库级 clip 向量维度（未锁定时为 None）。"""
        return self._clip_dim

    @property
    def frame_dim(self) -> Optional[int]:
        """库级 frame 向量维度（未锁定时为 None）。"""
        return self._frame_dim

    def _check_vec(self, v: np.ndarray, dim: Optional[int], what: str) -> np.ndarray:
        """向量形状校验 + 库级维度锁定/比对（不匹配 → ValueError 并指明两侧维度）。"""
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if dim is None:
            if a.size <= 0:
                raise ValueError(f"{what}: 空向量（库级维度尚未锁定，首条写入须为有效向量）")
            return a
        if a.size != dim:
            raise ValueError(
                f"{what}: 向量维度 {a.size} 与库维度 {dim} 不匹配"
                f"（入库与查询必须同维度；换维度请新建库，不混写）")
        return a

    # ---------------- 写入（upsert 幂等） ----------------

    @staticmethod
    def key_of(meta: dict) -> Tuple[str, float]:
        """片段 upsert 键 (video, t_start)（§3.3：同视频重跑按此覆盖）。t_start 取 3 位小数防浮点抖动。"""
        return (str(meta.get("video", "")), round(float(meta.get("t_start", 0.0) or 0.0), 3))

    def add_clip(self, clip_vec: np.ndarray, frame_vecs: Sequence[np.ndarray] = (),
                 frame_ts: Optional[Sequence[float]] = None,
                 meta: Optional[dict] = None) -> int:
        """写入一条片段条目（clip 向量 + frame 向量组 + 元数据），返回条目 id。

        幂等 upsert：(video, t_start) 已存在 → 覆盖该条（id 不变；旧 frame 向量清除）。
        frame_ts 与 frame_vecs 行对齐（缺省补 0.0）。meta 须含 video/t_start（检索过滤与
        upsert 键依赖；t_end/thumb_path 等随 dict 自由携带）。
        """
        meta = dict(meta or {})
        cv_ = self._check_vec(clip_vec, self._clip_dim, "add_clip(clip_vec)")
        if self._clip_dim is None:
            self._clip_dim = cv_.size
        fvs = [self._check_vec(fv, self._frame_dim, "add_clip(frame_vecs)")
               for fv in frame_vecs]
        if self._frame_dim is None and fvs:
            self._frame_dim = fvs[0].size
        ts_list = [float(t) for t in (frame_ts or [])]
        if len(ts_list) < len(fvs):  # 缺省补 0.0，长度对齐
            ts_list += [0.0] * (len(fvs) - len(ts_list))

        key = self.key_of(meta)
        old_id = self._key_index.get(key)
        if old_id is not None:  # upsert：覆盖片段行 + 清旧 frame 行（id 保持）
            self._clip_vecs[old_id] = cv_
            self._clip_meta[old_id] = meta
            keep = [(fv, fm) for fv, fm in zip(self._frame_vecs, self._frame_meta)
                    if fm.get("clip_id") != old_id]
            self._frame_vecs = [fv for fv, _ in keep]
            self._frame_meta = [fm for _, fm in keep]
            cid = old_id
        else:
            self._clip_vecs.append(cv_)
            self._clip_meta.append(meta)
            cid = len(self._clip_vecs) - 1
            self._key_index[key] = cid
        for k, (fv, ts) in enumerate(zip(fvs, ts_list)):
            self._frame_vecs.append(fv)
            self._frame_meta.append({"clip_id": cid, "ts": float(ts), "index": k})
        return cid

    # ---------------- 检索（① clip 粗排 / ② frame 精排） ----------------

    def search(self, query_vec: np.ndarray, top_k: int = 10,
               video: Optional[str] = None,
               thresh: Optional[float] = None) -> List[ClipHit]:
        """查询向量 → clip 级 Top-K（点积 = 余弦，降序）。video 过滤（按视频重建索引/限定范围用）；
        thresh 可选截断（缺省按相对排序全返回 top_k——同域文本相似度普遍偏高，
        阈值调优按相对排序不按绝对值，实测 §6）。"""
        if not self._clip_vecs:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        idxs = range(len(self._clip_vecs)) if video is None else [
            i for i, m in enumerate(self._clip_meta) if m.get("video") == video]
        if not idxs:
            return []
        matrix = np.stack([self._clip_vecs[i] for i in idxs])
        sims = matrix @ q
        order = np.argsort(-sims)[:top_k]
        out: List[ClipHit] = []
        for rank, oi in enumerate(order):
            s = float(sims[oi])
            if thresh is not None and s < thresh:
                break
            i = idxs[int(oi)]
            out.append(ClipHit(score=s, clip_id=i, rank=rank + 1,
                               meta=self._clip_meta[i]))
        return out

    def search_frames(self, query_vec: np.ndarray, clip_id: int,
                      top_k: int = 3) -> List[FrameHit]:
        """查询向量 → 指定片段内的 frame 级 Top-K（两级索引第二层精排，§4.1）。
        命中后播放定位：currentTime = hit.ts − pre_roll_sec。"""
        pairs = [(i, fm) for i, fm in enumerate(self._frame_meta)
                 if fm.get("clip_id") == clip_id]
        if not pairs:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        matrix = np.stack([self._frame_vecs[i] for i, _ in pairs])
        sims = matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [FrameHit(score=float(sims[oi]), clip_id=clip_id,
                         ts=float(pairs[int(oi)][1].get("ts", 0.0)),
                         index=int(pairs[int(oi)][1].get("index", 0)), rank=r + 1)
                for r, oi in enumerate(order)]

    # ---------------- 删除（重建索引用） ----------------

    def delete_video(self, video: str) -> int:
        """删除指定视频的全部片段条目（含关联 frame 向量），返回删除的片段数。

        id 语义 = 库内索引，删除后保留条目**重映射连续 id**（frame 层引用同步重排）；
        调用方需以删除后的 search 结果为准（设计稿：按视频删除属重建索引场景）。"""
        dead = {i for i, m in enumerate(self._clip_meta) if m.get("video") == video}
        if not dead:
            return 0
        remap = {old: new for new, old in enumerate(
            i for i in range(len(self._clip_meta)) if i not in dead)}
        self._clip_vecs = [v for i, v in enumerate(self._clip_vecs) if i not in dead]
        self._clip_meta = [m for i, m in enumerate(self._clip_meta) if i not in dead]
        self._frame_vecs = [v for v, fm in zip(self._frame_vecs, self._frame_meta)
                            if fm.get("clip_id") not in dead]
        self._frame_meta = [{**fm, "clip_id": remap[fm["clip_id"]]}
                            for fm in self._frame_meta if fm.get("clip_id") not in dead]
        self._rebuild_index()
        return len(dead)

    def _rebuild_index(self) -> None:
        self._key_index = {self.key_of(m): i for i, m in enumerate(self._clip_meta)}
        # frame_meta 补 index（片段内序号，精定位展示用）
        counter: Dict[int, int] = {}
        for fm in self._frame_meta:
            cid = int(fm.get("clip_id", -1))
            fm["index"] = counter.get(cid, 0)
            counter[cid] = fm["index"] + 1

    # ---------------- 持久化（npz 向量 + json 元数据，对齐 FaceStore 形态） ----------------

    def save(self, path: str) -> str:
        """保存到磁盘，返回 .npz 主文件路径（元数据存同名 .json）。"""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            clip_vecs=np.stack(self._clip_vecs) if self._clip_vecs
            else np.empty((0, self._clip_dim or 0), dtype=np.float32),
            frame_vecs=np.stack(self._frame_vecs) if self._frame_vecs
            else np.empty((0, self._frame_dim or 0), dtype=np.float32),
        )
        with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
            json.dump({"clip_dim": self._clip_dim, "frame_dim": self._frame_dim,
                       "metas": self._clip_meta, "frame_meta": self._frame_meta},
                      f, ensure_ascii=False)
        return path

    def load(self, path: str) -> "EmbedStore":
        """从磁盘加载（配套 .npz + .json）；库级维度以文件为准。"""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        with open(path[:-4] + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        z = np.load(path)
        self._clip_dim = meta.get("clip_dim")
        self._frame_dim = meta.get("frame_dim")
        cvs, fvs = z["clip_vecs"], z["frame_vecs"]
        self._clip_vecs = [cvs[i] for i in range(cvs.shape[0])]
        self._clip_meta = [dict(m) for m in meta.get("metas", [])]
        self._frame_vecs = [fvs[i] for i in range(fvs.shape[0])]
        self._frame_meta = [dict(m) for m in meta.get("frame_meta", [])]
        self._rebuild_index()
        return self

    # ---------------- 统计 / 自检 ----------------

    def __len__(self) -> int:
        return len(self._clip_vecs)

    def stats(self) -> dict:
        """条数 / 维度自检（§5）：clips / frames / 库级维度 / 视频清单。"""
        videos = sorted({str(m.get("video", "")) for m in self._clip_meta})
        return {"clips": len(self._clip_vecs), "frames": len(self._frame_vecs),
                "clip_dim": self._clip_dim, "frame_dim": self._frame_dim,
                "videos": videos}
