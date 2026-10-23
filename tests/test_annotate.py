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
  7. record 模式（WO-0021）：每记录整帧图只画自己框（像素区域断言）/ 透传不带图 /
     上限截断 / scope 非法 fail fast / Pipeline 登记 annotated_records 且主干不动

运行: conda run -n ai python tests/test_annotate.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.annotate import annotate_frame, annotate_records, Annotator, parse_condition
from pipe.composer import Detection, FrameManager, Pipeline
from pipe.records import image_set_of, to_record

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

    # ---- 7. record 模式（WO-0021：每记录整帧图、只画自己框） ----
    # 区域断言：帧全零 → 图中仅该记录自己的 bbox 区域有非零像素。
    recs, dropped = annotate_records(frame, [r0, r1])
    check("record：两条记录各一张图、未截断", len(recs) == 2 and dropped == 0)
    im0 = image_set_of(recs[0])[0]
    im1 = image_set_of(recs[1])[0]
    check("record：图挂 image 旁路且为整帧尺寸",
          im0.shape == frame.shape and im1.shape == frame.shape)
    check("record：rec0 只画自己框 A（区域内有像素、B 区域无）",
          bool(im0[20:70, 10:60].any()) and not bool(im0[20:70, 110:160].any()))
    check("record：rec1 只画自己框 B（区域内有像素、A 区域无）",
          bool(im1[20:70, 110:160].any()) and not bool(im1[20:70, 10:60].any()))
    check("record：每张图各自成帧（互非同一对象）", im0 is not im1)
    check("record：不改原帧", not frame.any())
    check("record：不污染上游记录（浅拷贝）",
          "image" not in r0 and "image" not in r1)

    # 7a. filter/classes 只决定画不画图，记录流透传（删记录是 det_filter 职责）
    recs_f, _ = annotate_records(frame, [r0, r1], filter="cls_id == 2")
    check("record：filter 外的记录透传不带图（不删记录）",
          len(recs_f) == 2 and image_set_of(recs_f[0]) == [] and
          bool(image_set_of(recs_f[1])[0][20:70, 110:160].any()))

    # 7b. 上限保护：超限截断 + dropped 计数 + stderr 告警
    many = [to_record(det((5 + 20 * i, 20, 8 + 20 * i, 70), cls=0, tid=i + 1))
            for i in range(5)]
    recs3, dropped = annotate_records(frame, many, max_records=3)
    check("record：上限截断（保留前 N 条）", len(recs3) == 3 and dropped == 2)
    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf):
        Annotator(scope="record", max_records=3).run(frame, many)
    check("record：截断触发告警日志（stderr）", "截断 2 条" in err_buf.getvalue())

    # 7c. Annotator scope：分支返回类型 + 非法取值 fail fast（装配期）
    out = Annotator(scope="record").run(frame, [r0, r1])
    check("Annotator record.run 返回带图记录列表",
          isinstance(out, list) and len(out) == 2 and
          image_set_of(out[0])[0].shape == frame.shape)
    check("Annotator frame.run 返回 ndarray（现状）",
          isinstance(Annotator().run(frame, [r0]), np.ndarray))
    try:
        Annotator(scope="banana")
        check("scope 非法报错（装配期 fail fast）", False)
    except ValueError:
        check("scope 非法报错（装配期 fail fast）", True)

    # ---- 8. Pipeline 集成（record 模式：登记产物、主干不动、frame 零回归） ----
    class FakeYolo2:
        def infer(self, frame, track=False):
            return [to_record(det(A, cls=0, tid=1)), to_record(det(B, cls=2, tid=5))]

    p2 = Pipeline(frame=FrameManager(frame_skip=1), yolo=FakeYolo2(),
                  annotate=Annotator(scope="record"))
    p2.step(0, 0.0, frame=frame, force=True)
    check("Pipeline record：登记 annotate.annotated_records",
          "annotate.annotated_records" in p2.artifacts)
    rr = p2.artifacts["annotate.annotated_records"]
    check("Pipeline record：产物每条带整帧图", len(rr) == 2 and
          all(image_set_of(x)[0].shape == frame.shape for x in rr))
    check("Pipeline record：主干不被带图流污染（yolo.dets 无 image 键）",
          all("image" not in d for d in p2.artifacts["yolo.dets"]))
    check("Pipeline record：last_annotated 不受 record 实例影响（None）",
          p2.last_annotated is None)
    check("Pipeline record：不登记帧级 annotated",
          "annotate.annotated" not in p2.artifacts)

    p3 = Pipeline(frame=FrameManager(frame_skip=1), yolo=FakeYolo2(),
                  annotate=Annotator())
    p3.step(0, 0.0, frame=frame, force=True)
    check("Pipeline frame：不产 annotated_records（零回归）",
          "annotate.annotated_records" not in p3.artifacts)
    check("Pipeline frame：last_annotated 正常（现状）",
          p3.last_annotated is not None and bool(p3.last_annotated.any()))

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
