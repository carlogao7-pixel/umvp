# -*- coding: utf-8 -*-
"""
标注模块测试（纯逻辑，不加载模型）：

  1. annotate_frame：画框 / 尺寸不变 / 不改原帧
  2. classes 过滤
  3. filter 条件 DSL（显式字段名；组内 AND / 组间 OR / in / 别名）
  4. Annotator.run
  5. Pipeline.annotate 集成（YOLO → 标注 → last_annotated）
  6. dict(Record) 输入（bbox 画框 / classes 对 cls_id 生效 / 标签 track_id；
     防 getattr 静默退化回归，INB-0059）

运行: conda run -n ai python tests/test_annotate.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.annotate import annotate_frame, Annotator, parse_condition
from pipe.composer import Detection, FrameManager, Pipeline
from pipe.records import to_record

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(bbox, cls: int = 0, tid: int = 0, score: float = 0.9) -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=bbox, score=score)


def drew(vis, bbox) -> bool:
    if vis is None:
        return False
    x1, y1, x2, y2 = bbox
    return bool(vis[y1:y2, x1:x2].any())


def annotate_safe(frame, dets, **kw):
    """dict 适配未落地时 annotate_frame 抛 AttributeError → 返回 None（用例判红）。"""
    try:
        return annotate_frame(frame, dets, **kw)
    except AttributeError:
        return None


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

    p = Pipeline(frame=FrameManager(frame_skip=1),
                 yolo=FakeYolo(), annotate=Annotator())
    p.step(0, 0.0, frame=frame, force=True)
    check("Pipeline: 产出标注图",
          p.last_annotated is not None and bool(p.last_annotated.any()))

    # ---- 6. dict(Record) 输入（链路通用语；防 getattr 静默退化回归，INB-0059） ----
    # dict 统一经 to_record(Detection(...)) 构造，字段集与批次2 infer() 未来产出一致
    # （track_id/cls_id/bbox/score/model_path）。getattr 退化不报错：cls_id→-1、
    # track_id→0——只有断言字段值/画框结果才能暴露静默失效。
    r0 = to_record(det(A, cls=0, tid=1))    # 同 d0 字段
    r1 = to_record(det(B, cls=2, tid=5))    # 同 d1 字段
    check("dict 字段集与对象一致（含 model_path）",
          set(r0.keys()) == {"track_id", "cls_id", "bbox", "score", "model_path"})

    # 6a. dict bbox 驱动画框：位置正确 + 与同字段对象输入逐像素一致
    v_obj0 = annotate_frame(frame, [d0])
    v_dict0 = annotate_safe(frame, [r0])
    check("dict 输入：可标注（不抛 AttributeError）", v_dict0 is not None,
          "" if v_dict0 is not None else "AttributeError: d.bbox 直读未适配 dict")
    check("dict 输入：画框位置=A", drew(v_dict0, A))
    check("dict 输入：与对象输入逐像素一致",
          v_dict0 is not None and np.array_equal(v_obj0, v_dict0))

    # 6b. classes 过滤对 dict 的 cls_id 生效（静默失效探测器）：
    #     cls_id=0 + classes=[0] 应画——若 getattr 退化 cls_id=-1 则不画（断言红）。
    check("dict classes 正向：cls_id=0 在 classes=[0] 被画",
          drew(annotate_safe(frame, [r0], classes=[0]), A))
    #     反向对照：cls_id=2 + classes=[0] 不画（退化 -1 时恰也不画，仅作语义对照）。
    check("dict classes 反向：cls_id=2 不在 classes=[0] 不画",
          not drew(annotate_safe(frame, [r1], classes=[0]), B))

    # 6c. 标签 track_id 来自 dict：与对象输入逐像素一致（tid 退化 0 时标签无 "#5"
    #     后缀、cls_id 退化时标签变 cls-1，均导致像素差异 → 红）。
    v_obj1 = annotate_frame(frame, [d1])
    v_dict1 = annotate_safe(frame, [r1])
    check("dict 标签：track_id/cls_id 进入标签文本（与对象输入逐像素一致）",
          v_dict1 is not None and np.array_equal(v_obj1, v_dict1))

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
