# -*- coding: utf-8 -*-
"""
目标裁剪模块测试（纯逻辑，不加载模型）：

  1. crop_targets 按坐标裁图 / 尺寸 / offset/scale / 来源元数据
  2. scale 放大与缩小
  3. margin 外扩 + clamp 到图界
  4. classes / filter 条件过滤
  5. to_original 坐标还原（子图框 → 原图框）
  6. TargetCrop.run / boxes 裸框
  7. Pipeline.crop 集成（帧决策 → 目标裁剪 → 下游消费）
  8. dict(Record) 输入（裁剪几何 / 产物承接元数据 / filter·classes 对 dict 生效；
     防 getattr 静默退化回归，INB-0059）

运行: conda run -n ai python tests/test_target_crop.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.target_crop import crop_targets, TargetCrop
from pipe.composer import Detection, FrameManager, Pipeline
from pipe.records import to_record

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def make_frame(h: int = 100, w: int = 200) -> np.ndarray:
    """每像素编码坐标，便于验证切片出处。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (np.arange(w) % 256).astype(np.uint8)[None, :]
    img[..., 1] = (np.arange(h) % 256).astype(np.uint8)[:, None]
    return img


def det(bbox, cls: int = 2, score: float = 0.9, tid: int = 0) -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=bbox, score=score, model_path="m")


def crop_safe(frame, dets, **kw):
    """dict 适配未落地时 _iter_items 直读 d.bbox 抛 AttributeError → 返回 None（用例判红）。"""
    try:
        return crop_targets(frame, dets=dets, **kw)
    except AttributeError:
        return None


def main() -> None:
    frame = make_frame()

    # ---- 1. 基本裁剪 ----
    crops = crop_targets(frame, [det((10, 20, 50, 60))])
    check("基本裁剪: 数量", len(crops) == 1)
    c = crops[0]
    check("基本裁剪: 尺寸", c.img.shape[:2] == (40, 40), str(c.img.shape))
    check("基本裁剪: offset", c.offset == (10, 20))
    check("基本裁剪: scale=1", c.scale == 1.0)
    check("基本裁剪: 像素与原图切片一致", np.array_equal(c.img, frame[20:60, 10:50]))
    check("基本裁剪: 来源元数据透传",
          c.cls_id == 2 and c.track_id == 0 and abs(c.score - 0.9) < 1e-6)

    # ---- 2. scale 放大 / 缩小 ----
    up = crop_targets(frame, [det((10, 20, 50, 60))], scale=2.0)[0]
    check("scale=2: 尺寸翻倍", up.img.shape[:2] == (80, 80), str(up.img.shape))
    check("scale=2: 记录 scale", up.scale == 2.0)
    check("scale=2: offset 不变", up.offset == (10, 20))
    down = crop_targets(frame, [det((0, 0, 100, 100))], scale=0.5)[0]
    check("scale=0.5: 尺寸减半", down.img.shape[:2] == (50, 50), str(down.img.shape))

    # ---- 3. margin 外扩 + clamp ----
    m = crop_targets(frame, [det((10, 20, 50, 60))], margin=0.5)[0]
    check("margin=0.5: 外扩 20px", m.bbox == (0, 0, 70, 80), str(m.bbox))
    check("margin: 子图尺寸=外扩框", m.img.shape[:2] == (80, 70), str(m.img.shape))
    mc = crop_targets(frame, [det((5, 5, 25, 25))], margin=1.0)[0]
    check("margin: clamp 到图界", mc.bbox == (0, 0, 45, 45), str(mc.bbox))

    # ---- 4. 过滤（纯机械裁剪：无尺寸过滤，只有类别/条件） ----
    check("classes: 不匹配跳过",
          crop_targets(frame, [det((0, 0, 20, 20), cls=0)], classes=[2]) == [])
    check("classes: 匹配保留",
          len(crop_targets(frame, [det((0, 0, 20, 20), cls=2)], classes=[2])) == 1)
    check("无效框跳过", crop_targets(frame, [det((50, 50, 50, 60))]) == [])

    # ---- 4b. filter 条件 DSL（默认空=全裁，等价原简单过滤）----
    check("filter 空=全裁",
          len(crop_targets(frame, [det((0, 0, 20, 20), cls=2, tid=1)])) == 1)
    check("filter track_id==1 命中",
          len(crop_targets(frame, [det((0, 0, 20, 20), cls=2, tid=1)],
                           filter="track_id == 1")) == 1)
    check("filter track_id==1 不命中",
          crop_targets(frame, [det((0, 0, 20, 20), cls=2, tid=5)],
                       filter="track_id == 1") == [])
    check("filter cls_id in 0|2",
          len(crop_targets(frame, [det((0, 0, 20, 20), cls=2),
                                   det((30, 30, 50, 50), cls=7)],
                           filter="cls_id in 0|2")) == 1)
    check("filter 与 classes 同时生效（AND）",
          crop_targets(frame, [det((0, 0, 20, 20), cls=2, tid=5)],
                       classes=[2], filter="track_id == 1") == [])

    # ---- 5. 坐标还原 ----
    c = crop_targets(frame, [det((10, 20, 50, 60))], scale=2.0)[0]
    # 子图里 (0,0,20,20) 对应原图 (10,20,20,30)
    check("to_original: scale=2 除以倍数加偏移",
          c.to_original((0, 0, 20, 20)) == (10, 20, 20, 30))
    c1 = crop_targets(frame, [det((10, 20, 50, 60))])[0]
    check("to_original: scale=1 = 子框+offset",
          c1.to_original((5, 5, 15, 15)) == (15, 25, 25, 35))

    # ---- 5b. out_size 输出分辨率 ----
    check("out_size='': 原图",
          crop_targets(frame, [det((10, 20, 50, 60))], out_size="")[0].img.shape[:2] == (40, 40))
    ls = crop_targets(frame, [det((10, 20, 50, 60))], out_size="80")[0]
    check("out_size=80: 长边=80 等比",
          ls.img.shape[:2] == (80, 80) and abs(ls.scale - 2.0) < 1e-6, str(ls.img.shape))
    ex = crop_targets(frame, [det((10, 20, 50, 70))], out_size="60,40")[0]
    check("out_size=60,40: 严格尺寸", ex.img.shape[:2] == (40, 60), str(ex.img.shape))
    check("out_size=60,40: 非等比坐标还原",
          ex.to_original((0, 0, 60, 40)) == (10, 20, 50, 70),
          str(ex.to_original((0, 0, 60, 40))))
    for form in ("×2", "2x", "x2", "*2"):
        v = crop_targets(frame, [det((10, 20, 50, 60))], out_size=form)[0]
        check(f"out_size={form}: 倍数翻倍",
              v.img.shape[:2] == (80, 80) and abs(v.scale - 2.0) < 1e-6, str(v.img.shape))

    # ---- 6. TargetCrop.run / boxes 裸框 ----
    tc = TargetCrop(scale=1.0, margin=0.0, classes=[2])
    r = tc.run(frame, [det((0, 0, 20, 20), cls=2), det((30, 30, 50, 50), cls=0)])
    check("TargetCrop: 按类过滤+裁剪", len(r) == 1 and r[0].bbox == (0, 0, 20, 20))
    rb = crop_targets(frame, boxes=[(10, 20, 50, 60)])
    check("boxes 裸框: 可裁且 cls=-1", len(rb) == 1 and rb[0].cls_id == -1)

    # ---- 7. Pipeline.crop 集成 ----
    class FakeYolo:
        def infer(self, frame, track=False):
            return [det((10, 20, 50, 60), cls=2, tid=7)]

        def pass_dets(self, dets, ts):
            return dets

    class FakePlate:
        use_track = False

        def __init__(self):
            self.got = None

        def run(self, frame, crops, ts=0.0):
            self.got = crops
            return []

    fp = FakePlate()
    p = Pipeline(frame=FrameManager(frame_skip=1),
                 yolo=FakeYolo(), crop=TargetCrop(scale=2.0, classes=[2]), plate=fp)
    p.step(0, 0.0, frame=frame, force=True)
    check("Pipeline: 产出裁剪", len(p.last_crops) == 1 and p.last_crops[0].scale == 2.0)
    check("Pipeline: 裁剪传给车牌阶段", fp.got is p.last_crops)

    # ---- 8. dict(Record) 输入（链路通用语；防 getattr 静默退化回归，INB-0059） ----
    # dict 统一经 to_record(Detection(...)) 构造（字段集含 model_path，对齐批次2 infer()）。
    # 当前 _iter_items 直读 d.bbox → dict 必 AttributeError；修复后若 getattr 退化
    # （cls_id→-1/track_id→0/score→0.0）不报错，只有断言产物字段值才能暴露。
    r7 = to_record(det((10, 20, 50, 60), cls=2, score=0.85, tid=7))
    r9 = to_record(det((120, 30, 160, 70), cls=2, score=0.85, tid=9))
    check("dict 字段集与对象一致（含 model_path）",
          set(r7.keys()) == {"track_id", "cls_id", "bbox", "score", "model_path"})

    # 8a. 裁剪几何与 dict bbox 一致（与对象输入对照）
    crops_d = crop_safe(frame, [r7])
    check("dict 输入：可裁剪（不抛 AttributeError）", crops_d is not None,
          "" if crops_d is not None else "AttributeError: d.bbox 直读未适配 dict")
    c_obj = crop_targets(frame, [det((10, 20, 50, 60), cls=2, score=0.85, tid=7)])[0]
    check("dict 输入：数量", crops_d is not None and len(crops_d) == 1)
    check("dict 输入：裁剪像素与对象输入一致",
          crops_d is not None and np.array_equal(crops_d[0].img, c_obj.img))
    check("dict 输入：offset/scale 一致",
          crops_d is not None and crops_d[0].offset == c_obj.offset
          and crops_d[0].scale == c_obj.scale)

    # 8b. 产物 CropResult 承接 dict 元数据（静默失效探测器：退化 cls_id=-1 直接红）
    check("dict 输入：产物承接 cls_id/score/track_id",
          crops_d is not None and crops_d[0].cls_id == 2
          and abs(crops_d[0].score - 0.85) < 1e-6 and crops_d[0].track_id == 7,
          str((crops_d[0].cls_id, crops_d[0].score, crops_d[0].track_id))
          if crops_d is not None else "no crop")

    # 8c. 条件 DSL 对 dict 生效：track_id==7 被裁、track_id==9 不被裁
    #     （match_obj 双形态安全；若筛选层 getattr 退化 track_id 恒 0 → 全被剔除 → 红）
    got7 = crop_safe(frame, [r7], filter="track_id == 7")
    got9 = crop_safe(frame, [r9], filter="track_id == 7")
    check("dict filter：track_id==7 命中被裁", got7 is not None and len(got7) == 1)
    check("dict filter：track_id==9 不命中不裁", got9 is not None and got9 == [])

    # 8d. classes 对 dict 的 cls_id 生效（静默失效探测器：退化 cls_id=-1 → cls_id=2
    #     的 dict 会被 classes=[2] 误剔 → 红）
    got_c2 = crop_safe(frame, [r7], classes=[2])
    got_c0 = crop_safe(frame, [to_record(det((10, 20, 50, 60), cls=0, tid=1))], classes=[2])
    check("dict classes 正向：cls_id=2 在 classes=[2] 被裁",
          got_c2 is not None and len(got_c2) == 1)
    check("dict classes 反向：cls_id=0 不在 classes=[2] 不裁",
          got_c0 is not None and got_c0 == [])

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
