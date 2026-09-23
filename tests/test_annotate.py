# -*- coding: utf-8 -*-
"""
标注模块测试（纯逻辑，不加载模型）：

  1. annotate_frame：画框 / 尺寸不变 / 不改原帧
  2. classes 过滤
  3. filter 条件 DSL（显式字段名；组内 AND / 组间 OR / in / 别名）
  4. Annotator.run
  5. Pipeline.annotate 集成（YOLO → 标注 → last_annotated）

运行: conda run -n ai python tests/test_annotate.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.annotate import annotate_frame, Annotator, parse_condition
from pipe.composer import Detection, FrameManager, AlarmPolicy, Pipeline

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(bbox, cls: int = 0, tid: int = 0, score: float = 0.9) -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=bbox, score=score)


def drew(vis, bbox) -> bool:
    x1, y1, x2, y2 = bbox
    return bool(vis[y1:y2, x1:x2].any())


A = (10, 20, 60, 70)     # cls0 track1
B = (110, 20, 160, 70)   # cls2 track5


def main() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    d0 = det(A, cls=0, tid=1)
    d1 = det(B, cls=2, tid=5)

    # ---- 1. 基本 ----
    vis = annotate_frame(frame, [d0])
    check("尺寸不变", vis.shape == frame.shape)
    check("不改原帧", not frame.any())
    check("画了框", vis.any())

    # ---- 2. classes 过滤 ----
    check("classes 过滤：不匹配不画",
          not annotate_frame(frame, [d1], classes=[0]).any())

    # ---- 3. filter 条件 DSL ----
    v = annotate_frame(frame, [d0, d1])
    check("全画：两框都画", drew(v, A) and drew(v, B))
    v = annotate_frame(frame, [d0, d1], filter="track_id == 1")
    check("filter track_id==1：只画 A", drew(v, A) and not drew(v, B))
    v = annotate_frame(frame, [d0, d1], filter="cls_id == 2")
    check("filter cls_id==2：只画 B", (not drew(v, A)) and drew(v, B))
    v = annotate_frame(frame, [d0, d1], filter="类别=0")   # 中文别名 + '=' 等价 '=='
    check("别名 类别=0：只画 A", drew(v, A) and not drew(v, B))
    v = annotate_frame(frame, [d0, d1], filter="cls_id in 0|2, score >= 0.5")
    check("in + score（组内 AND）：两框都画", drew(v, A) and drew(v, B))
    v = annotate_frame(frame, [d0, d1], filter="cls_id in 0|2, score > 0.95")
    check("score>0.95：都不画", (not drew(v, A)) and (not drew(v, B)))
    v = annotate_frame(frame, [d0, d1], filter="track_id == 9; cls_id == 2")
    check("组间 OR：只画 B", (not drew(v, A)) and drew(v, B))
    v = annotate_frame(frame, [d0, d1], classes=[0], filter="track_id == 5")
    check("classes 与 filter 同时生效（AND）：都不画",
          (not drew(v, A)) and (not drew(v, B)))
    check("parse_condition 结构",
          parse_condition("cls_id in 0|2, score >= 0.5") ==
          [[("cls_id", "in", ["0", "2"]), ("score", ">=", 0.5)]])
    try:
        annotate_frame(frame, [d0], filter="track_id 1")
        check("非法条件报错", False)
    except ValueError:
        check("非法条件报错", True)

    # ---- 4. Annotator ----
    check("Annotator.run 画框",
          bool(Annotator(thickness=3, show_label=False, classes=[0]).run(frame, [d0]).any()))

    # ---- 5. Pipeline 集成 ----
    class FakeYolo:
        def infer(self, frame, track=False):
            return [d0]

        def pass_dets(self, dets, ts):
            return dets

    p = Pipeline(frame=FrameManager(frame_skip=1),
                 alarm=AlarmPolicy(task_id=1, kind="inspection"),
                 yolo=FakeYolo(), annotate=Annotator())
    p.step(0, 0.0, frame=frame, force=True)
    check("Pipeline: 产出标注图",
          p.last_annotated is not None and bool(p.last_annotated.any()))

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
