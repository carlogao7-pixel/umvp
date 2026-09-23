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

运行: conda run -n ai python tests/test_target_crop.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.target_crop import crop_targets, TargetCrop
from pipe.composer import Detection, FrameManager, AlarmPolicy, Pipeline

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
                 alarm=AlarmPolicy(task_id=1, kind="inspection"),
                 yolo=FakeYolo(), crop=TargetCrop(scale=2.0, classes=[2]), plate=fp)
    p.step(0, 0.0, frame=frame, force=True)
    check("Pipeline: 产出裁剪", len(p.last_crops) == 1 and p.last_crops[0].scale == 2.0)
    check("Pipeline: 裁剪传给车牌阶段", fp.got is p.last_crops)

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
