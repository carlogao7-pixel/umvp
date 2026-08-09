# -*- coding: utf-8 -*-
"""
人脸向量底库与检索模块（管道节点 #3: Store/Search）

输入 : 512 维 L2 归一化 embedding
输出 : 注册条目 / 检索 Top-K 匹配

实现：零依赖 numpy 线性检索（内积 = 余弦相似度，向量已归一化）。
- 千级底库检索 < 1ms，无需外部服务/进程
- 磁盘持久化：npz（向量矩阵）+ json（身份/元数据），重启不丢
- 同一身份允许多条向量（多张登记照），检索时按身份取最高分
- 接口与 FAISS/LanceDB 对齐，底库规模增大后可平替存储层

阈值经验（ArcFace 512d 余弦相似度）:
  > 0.5    大概率同一人
  0.35~0.5 可疑，需人工确认
  < 0.35   大概率不同人
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    """单次检索命中（按身份聚合后的最高分条目）"""

    name: str  # 身份标签
    score: float  # 余弦相似度 [0,1]，越高越相似
    id: int  # 命中向量在底库中的序号
    rank: int  # 命中排序（1=最相似）
    meta: Optional[dict] = None  # 注册时附加的元数据


class FaceStore:
    """内存向量底库：注册 / 检索 / 持久化。"""

    def __init__(self, thresh: float = 0.45) -> None:
        self.thresh = thresh
        self._names: List[str] = []  # 与 _embs 行一一对应
        self._metas: List[Optional[dict]] = []
        self._embs: List[np.ndarray] = []

    # ---------------- 注册 ----------------
    def register(self, name: str, emb: np.ndarray, meta: Optional[dict] = None) -> int:
        """登记一条向量，返回条目 id。同一身份可重复注册（多张登记照）。"""
        self._names.append(name)
        self._metas.append(meta)
        self._embs.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        return len(self._names) - 1

    def register_batch(
        self, names: List[str], embs: np.ndarray, metas: Optional[List[dict]] = None
    ) -> None:
        """批量注册：embs 为 (N, dim) 矩阵，行与 names 对齐。"""
        for i, name in enumerate(names):
            self.register(name, embs[i], metas[i] if metas else None)

    # ---------------- 检索 ----------------
    def search(
        self, emb: np.ndarray, topk: int = 5, thresh: Optional[float] = None
    ) -> List[MatchResult]:
        """查询向量 -> 按身份聚合的 Top-K 匹配结果（余弦相似度降序）。"""
        thresh = self.thresh if thresh is None else thresh
        if not self._embs:
            return []
        q = np.asarray(emb, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) or 1.0)
        matrix = np.stack(self._embs)  # (N, dim) 已归一化
        sims = matrix @ q  # 内积 = 余弦

        best: Dict[str, Tuple[float, int, int]] = {}  # name -> (score, id, rank)
        order = np.argsort(-sims)
        for rank, idx in enumerate(order):
            s = float(sims[idx])
            if s < thresh:
                break
            name = self._names[idx]
            if name in best:
                continue
            best[name] = (s, int(idx), rank + 1)
            if len(best) >= topk:
                break

        return [
            MatchResult(name=name, score=score, id=eid, rank=rank, meta=self._metas[eid])
            for name, (score, eid, rank) in best.items()
        ]

    def search_matrix(self, embs: np.ndarray, topk: int = 5) -> List[List[MatchResult]]:
        """批量查询：(M, dim) -> M 组 Top-K 结果。"""
        return [self.search(e, topk=topk) for e in embs]

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> str:
        """保存到磁盘，返回 .npz 主文件路径（元数据存同名 .json）。"""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, embs=np.stack(self._embs) if self._embs else np.empty((0, 0)))
        meta_path = path[:-4] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"names": self._names, "metas": self._metas, "thresh": self.thresh},
                f, ensure_ascii=False, indent=2,
            )
        return path

    def load(self, path: str) -> "FaceStore":
        """从磁盘加载（配套 .npz + .json）。"""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        meta_path = path[:-4] + ".json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        embs = np.load(path)["embs"]
        self.thresh = meta.get("thresh", self.thresh)
        self._names = list(meta["names"])
        self._metas = list(meta["metas"])
        self._embs = [embs[i] for i in range(embs.shape[0])]
        return self

    # ---------------- 统计 ----------------
    def __len__(self) -> int:
        return len(self._names)

    def stats(self) -> dict:
        identities = sorted(set(self._names))
        return {
            "entries": len(self._names),  # 向量条数（含一人多张）
            "identities": len(identities),  # 身份数
            "per_identity": {n: self._names.count(n) for n in identities},
        }
