# -*- coding: utf-8 -*-
"""
门控模块测试（GateFilter，纯逻辑，不加载模型）——设计稿 doc/design/gate_filter.md §7.1：

  1. 基本节流：interval 内拦 / >= interval 边界放
  2. 多 key 独立冷却
  3. 匿名桶：tid=0/None/缺失 同落 "*"，窗口内只放 1 条；与实名桶互不干扰（tid=0 缺口修法）
  4. interval<0：同 key 只放行一次
  5. interval=0：全放行（no-op）
  6. 物理过滤铁律：被拦不返回、原元素进出、键集合不变（无标记字段）
  7. 首见放行 + 时间源（记录自带 ts 优先）
  8. 双形态：Detection 对象进对象出 / dict 进 dict 出
  9. stats：last_stats 计数与 blocked_detail 分键
  10. reset()：清状态重新首见放行
  11. 非法构造：key 空串 / 非法 filter DSL 构造期抛错；filter 限定门控范围（未命中直放）
  12. Pipeline 集成：插槽执行顺序 / last_gate / artifacts（dets+stats）/
      被拦不达 target_crop / track_needed 联动（needs_track，设计稿 §5.4）

运行: conda run -n ai python tests/test_gate_filter.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.composer import Detection, FrameManager, Pipeline  # noqa: E402
from pipe.gate_filter import ANON_KEY, GateFilter  # noqa: E402
from pipe.target_crop import TargetCrop  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def expect_value_error(name: str, fn) -> None:
    global PASS
    try:
        fn()
        raise AssertionError(f"FAIL: {name} 未抛 ValueError")
    except ValueError:
        PASS += 1
        print(f"  [PASS] {name}")


def rec(tid=1, cls=2, score=0.9, ts=None, **extra):
    r = {"kind": "det", "cls_id": cls, "track_id": tid, "score": score,
         "bbox": (0, 0, 50, 50)}
    if ts is not None:
        r["ts"] = ts
    r.update(extra)
    return r


def main() -> None:
    print("== 1. 基本节流 ==")
    g = GateFilter(key="track_id", interval=5.0)
    out0 = g.run([rec(tid=1)], ts=0.0)
    check("首见放行", len(out0) == 1)
    check("interval 内拦", g.run([rec(tid=1)], ts=2.0) == [])
    check("interval 内拦（贴边）", g.run([rec(tid=1)], ts=4.9) == [])
    out5 = g.run([rec(tid=1)], ts=5.0)
    check(">= interval 边界放行", len(out5) == 1)

    print("== 2. 多 key 独立 ==")
    g = GateFilter(key="track_id", interval=5.0)
    check("tid1 首见放行", len(g.run([rec(tid=1)], ts=0.0)) == 1)
    check("tid2 独立放行", len(g.run([rec(tid=2)], ts=1.0)) == 1)
    check("tid1 仍在冷却", g.run([rec(tid=1)], ts=1.0) == [])
    check("tid2 仍在冷却", g.run([rec(tid=2)], ts=2.0) == [])

    print("== 3. 匿名桶（tid=0 缺口修法） ==")
    g = GateFilter(key="track_id", interval=5.0)
    check("tid=0 首条放行", len(g.run([rec(tid=0)], ts=0.0)) == 1)
    check("tid=0 窗口内拦", g.run([rec(tid=0)], ts=1.0) == [])
    check("tid=None 落同桶拦", g.run([rec(tid=None)], ts=2.0) == [])
    check("缺字段落同桶拦", g.run([{"cls_id": 2, "score": 0.9,
                                    "bbox": (0, 0, 5, 5)}], ts=3.0) == [])
    check("实名桶不受匿名桶影响", len(g.run([rec(tid=7)], ts=3.5)) == 1)
    check("匿名桶到 interval 再放", len(g.run([rec(tid=0)], ts=5.0)) == 1)
    g2 = GateFilter(key="label", interval=5.0)
    g2.run([rec(tid=1, label="car")], ts=0.0)
    check("label 缺失归匿名桶（首条放行）", len(g2.run([rec(tid=2)], ts=1.0)) == 1)
    check("匿名桶窗口内拦", g2.run([rec(tid=4)], ts=2.0) == [])
    check("匿名桶与 label 桶独立", len(g2.run([rec(tid=3, label="truck")], ts=1.0)) == 1)

    print("== 4. interval<0 同 key 一次 ==")
    g = GateFilter(key="track_id", interval=-1.0)
    check("首条放行", len(g.run([rec(tid=1)], ts=0.0)) == 1)
    check("此后永不放（100s 后也拦）", g.run([rec(tid=1)], ts=100.0) == [])
    check("他 key 各自一次", len(g.run([rec(tid=2)], ts=100.0)) == 1)

    print("== 5. interval=0 全放行 ==")
    g = GateFilter(key="track_id", interval=0.0)
    outs = g.run([rec(tid=1), rec(tid=1), rec(tid=0)], ts=0.0)
    check("同帧同 key 全放", len(outs) == 3)
    check("跨帧仍全放", len(g.run([rec(tid=1)], ts=0.1)) == 1)
    check("no-op 无拦截计数", g.last_stats["blocked"] == 0)

    print("== 6. 物理过滤铁律 ==")
    g = GateFilter(key="track_id", interval=5.0)
    r1, r2 = rec(tid=1), rec(tid=1)
    outs = g.run([r1, r2], ts=0.0)
    check("放行的是原元素（is）", outs[0] is r1)
    check("被拦记录不返回（按身份）", all(o is not r2 for o in outs) and len(outs) == 1)
    check("键集合不变（无标记字段）", set(outs[0].keys()) == set(r1.keys()))
    g.run([rec(tid=1)], ts=1.0)
    check("被拦记录键集合也不变", "blocked" not in r2 and "gated" not in r2)

    print("== 7. 首见放行 + 时间源 ==")
    g = GateFilter(key="track_id", interval=5.0)
    a, b = rec(tid=1, ts=10.0), rec(tid=1, ts=11.0)
    outs = g.run([a], ts=999.0)
    check("记录自带 ts 优先（首见）", len(outs) == 1 and outs[0] is a)
    check("自带 ts 11-10<5 拦（忽略 run ts=999）", g.run([b], ts=999.0) == [])
    c = rec(tid=2, ts=None)
    check("无自带 ts 用 run ts", len(g.run([c], ts=50.0)) == 1)

    print("== 8. 双形态（对象/记录） ==")
    g = GateFilter(key="track_id", interval=5.0)
    d1 = Detection(track_id=3, cls_id=2, bbox=(0, 0, 40, 40), score=0.8)
    d2 = Detection(track_id=3, cls_id=2, bbox=(0, 0, 40, 40), score=0.8)
    outs = g.run([d1], ts=0.0)
    check("对象进对象出", outs == [d1] and outs[0] is d1)
    check("对象第二条拦", g.run([d2], ts=1.0) == [])
    outs = g.run([d1], ts=5.0)
    check("对象到 interval 再放", outs == [d1])

    print("== 9. stats ==")
    g = GateFilter(key="track_id", interval=5.0)
    g.run([rec(tid=1), rec(tid=1), rec(tid=0), rec(tid=2)], ts=0.0)
    st = g.last_stats
    check("in/passed/blocked", (st["in"], st["passed"], st["blocked"]) == (4, 3, 1))
    check("blocked_detail 分键", st["blocked_detail"] == {"1": 1})
    g.run([rec(tid=1), rec(tid=None)], ts=1.0)
    check("分键累计（含匿名桶）", g.last_stats["blocked_detail"] == {"1": 1, ANON_KEY: 1})
    sr = g.stats_record(ts=1.0, frame_idx=9)
    check("stats_record 结构", sr["kind"] == "gate_stats" and sr["key"] == "track_id"
          and sr["interval"] == 5.0 and sr["frame_idx"] == 9
          and sr["blocked_detail"] == {"1": 1, ANON_KEY: 1})

    print("== 10. reset ==")
    g = GateFilter(key="track_id", interval=5.0)
    g.run([rec(tid=1)], ts=0.0)
    check("冷却期内拦", g.run([rec(tid=1)], ts=1.0) == [])
    g.reset()
    check("reset 后重新首见放行", len(g.run([rec(tid=1)], ts=1.0)) == 1)
    check("reset 清统计", g.last_stats == {"in": 1, "passed": 1, "blocked": 0,
                                           "blocked_detail": {}})

    print("== 11. 构造校验 + filter 限定范围 ==")
    expect_value_error("非法 key 抛 ValueError（''）", lambda: GateFilter(key=""))
    expect_value_error("非法 key 抛 ValueError（'  '）", lambda: GateFilter(key="  "))
    expect_value_error("非法 filter 构造期抛 ValueError",
                       lambda: GateFilter(key="track_id", filter="cls_id ~~ 5"))
    g = GateFilter(key="track_id", interval=5.0, filter="cls_id == 2")
    person = rec(tid=9, cls=0)
    check("filter 未命中直放（不耗额度）", g.run([person], ts=0.0) == [person])
    check("filter 未命中不触桶（再来仍放）", g.run([person], ts=0.5) == [person])
    check("filter 命中受门控", len(g.run([rec(tid=1, cls=2)], ts=0.0)) == 1)
    check("filter 命中同 key 拦", g.run([rec(tid=1, cls=2)], ts=1.0) == [])
    check("filter 下 stats blocked 只算命中者",
          g.last_stats["blocked_detail"] == {"1": 1} and g.last_stats["in"] == 1)

    print("== 12. Pipeline 集成 ==")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    pipe = Pipeline(frame=FrameManager(frame_skip=1),
                    gate_filter=GateFilter(key="track_id", interval=5.0),
                    crop=TargetCrop(margin=0.0, classes=[2]))
    _, _ = pipe.step(0, 0.0, dets=[rec(tid=1), rec(tid=1)], frame=img)
    check("last_gate=(进入,放行)", pipe.last_gate == (2, 1))
    check("被拦不达 target_crop（last_crops 1 条）", len(pipe.last_crops) == 1)
    check("产物 gate_filter.dets 登记",
          len(pipe.artifacts["gate_filter.dets"]) == 1)
    st = pipe.artifacts["gate_filter.stats"]
    check("产物 gate_filter.stats 登记",
          len(st) == 1 and st[0]["blocked_detail"] == {"1": 1}
          and st[0]["kind"] == "gate_stats")
    _, _ = pipe.step(1, 1.0, dets=[rec(tid=0)], frame=img)
    check("匿名桶首条放行（tid=0 不再绕过）", pipe.last_gate == (1, 1)
          and len(pipe.last_crops) == 1)
    _, _ = pipe.step(2, 2.0, dets=[rec(tid=0)], frame=img)
    check("匿名桶窗口内拦（不再每帧放行）", pipe.last_gate == (1, 0)
          and pipe.last_crops == [])
    _, _ = pipe.step(3, 3.0, dets=[], frame=img)
    check("空帧不运行门控（last_gate=None）", pipe.last_gate is None)
    # 与 seq_filter 的插槽顺序：seq_filter 在前、gate_filter 在后
    from pipe.seq_filter import SeqFilter
    pipe2 = Pipeline(frame=FrameManager(frame_skip=1),
                     seq_filter=SeqFilter(throttle={"key": "cls_id", "interval": 3.0}),
                     gate_filter=GateFilter(key="track_id", interval=5.0))
    _, _ = pipe2.step(0, 0.0, dets=[rec(tid=1, cls=2)])
    check("seq_filter→gate_filter 串联（首帧双放）", pipe2.last_gate == (1, 1))
    _, _ = pipe2.step(1, 1.0, dets=[rec(tid=1, cls=2)])
    check("seq 节流在前已拦（gate 见 0 条）", pipe2.last_gate is None)
    # 多实例（gate_filters 容器，按位置升序链式）
    pipe3 = Pipeline(frame=FrameManager(frame_skip=1))
    pipe3.gate_filters = [(2, GateFilter(key="track_id", interval=5.0)),
                          (5, GateFilter(key="cls_id", interval=-1.0))]
    _, _ = pipe3.step(0, 0.0, dets=[rec(tid=1, cls=2), rec(tid=2, cls=2)])
    check("多实例链式（#2 按 track 放 2 条→#5 按 cls 一次闸放 1 条）",
          pipe3.last_gate == (2, 1))
    _, _ = pipe3.step(1, 1.0, dets=[rec(tid=3, cls=2)])
    check("多实例第二道 cls 一次闸拦", pipe3.last_gate == (1, 0))
    # track_needed 联动（设计稿 §5.4 灰度必踩坑）
    check("gate(key=track_id)→需要 track",
          Pipeline(frame=FrameManager(frame_skip=1),
                   gate_filter=GateFilter(key="track_id", interval=5.0)).track_needed())
    check("gate(key=label)→不需要 track",
          not Pipeline(frame=FrameManager(frame_skip=1),
                       gate_filter=GateFilter(key="label", interval=5.0)).track_needed())
    plate_stub = type("P", (), {"use_track": False})()
    p4 = Pipeline(frame=FrameManager(frame_skip=1),
                  gate_filter=GateFilter(key="track_id"), plate=plate_stub)
    check("内置去重关 + gate(track_id) 开 → 仍要 track（灰度坑）",
          p4.track_needed() is True)

    print(f"\n全部通过：{PASS} 断言")


if __name__ == "__main__":
    main()
