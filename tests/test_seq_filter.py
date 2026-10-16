# -*- coding: utf-8 -*-
"""
时序判定模块测试（纯逻辑，不加载模型）：

  1. 单条规则：命中即放行+事件标注（event/event_info）/ 不命中扣留 / 冷却节流
  2. 窗口规则：凑满 n 次前扣留、第 n 条放行 / 窗口过期淘汰 / key 分组 /
     达标重臂 / cooldown
  3. 契约：原元素进出（对象进对象出/dict 进 dict 出；事件 dict 原地写、对象挂属性）/
     先匹配先得 / 记录自带 ts / key 缺失归 "*" / 非法构造报错
  4. 组合：DetFilter(取舍) → SeqFilter(确认) 串联
  5. 门控：throttle 整类节流 / change track 集合变化（原 Pipeline 编排迁入）

运行: conda run -n ai python tests/test_seq_filter.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.composer import Detection
from pipe.det_filter import DetFilter
from pipe.seq_filter import SeqFilter, SeqRule

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(bbox, cls: int = 0, tid: int = 0, score: float = 0.9) -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=bbox, score=score)


def rec(cls=0, tid=1, score=0.9, ts=None, label=None):
    r = {"cls_id": cls, "track_id": tid, "score": score, "bbox": (0, 0, 50, 50)}
    if ts is not None:
        r["ts"] = ts
    if label is not None:
        r["label"] = label
    return r


def main() -> None:
    # ---- 1. 单条规则 ----
    sf = SeqFilter([SeqRule(match="score >= 0.9", event="高置信")])
    out = sf.run([det((0, 0, 50, 50), tid=7, score=0.95), det((0, 0, 50, 50), tid=8, score=0.5)])
    check("单条：命中放行、不命中扣留", len(out) == 1)
    check("单条：事件字段标注（对象挂属性 events）",
          getattr(out[0], "events", [{}])[0].get("name") == "高置信")
    check("单条：事件 detail 内容",
          out[0].events[0]["detail"]["count"] == 1 and out[0].events[0]["detail"]["span"] == 0
          and out[0].events[0]["ts"] == 0.0 and out[0].events[0]["detail"]["key"] == "7")
    check("对象进对象出：原 Detection 实例", isinstance(out[0], Detection) and out[0].cls_id == 0)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="有人", cooldown=10.0)])
    a = sf.run([rec(tid=7)], ts=1.0)
    b = sf.run([rec(tid=7)], ts=5.0)
    c = sf.run([rec(tid=7)], ts=12.0)
    check("单条+cooldown：首报/期内扣留/过后再报",
          len(a) == 1 and len(b) == 0 and len(c) == 1)
    d = sf.run([rec(tid=8)], ts=5.0)
    check("cooldown 按 key 分组", len(d) == 1, "track8 不受 track7 冷却影响")

    # ---- 2. 窗口规则 ----
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=3)])
    o1 = sf.run([rec(tid=7)], ts=0.0)
    o2 = sf.run([rec(tid=7)], ts=1.0)
    check("窗口：凑满前扣留", len(o1) == 0 and len(o2) == 0)
    o3 = sf.run([rec(tid=7)], ts=4.0)
    check("窗口：第 n 条放行", len(o3) == 1 and o3[0]["events"][0]["name"] == "驻留")
    check("窗口：事件 detail.count", o3[0]["events"][0]["detail"]["count"] == 3)
    o4 = sf.run([rec(tid=7)], ts=4.5)
    check("窗口：达标即重臂（未凑满再扣留）", len(o4) == 0)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=3)])
    sf.run([rec(tid=7)], ts=0.0)
    sf.run([rec(tid=7)], ts=4.0)
    o = sf.run([rec(tid=7)], ts=10.0)
    check("窗口：旧命中滑出窗口被淘汰", len(o) == 0, "t=0/4 已出 5 秒窗，仅剩 1 次")

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=10.0, min_n=3)])
    sf.run([rec(tid=7), rec(tid=7)], ts=0.0)
    o = sf.run([rec(tid=8), rec(tid=7)], ts=1.0)
    check("key 分组：各 track 独立计数", len(o) == 1 and o[0]["track_id"] == 7)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=2, cooldown=10.0)])
    check("窗口+cooldown：达标放行",
          len(sf.run([rec(tid=7), rec(tid=7)], ts=0.0)) == 1)
    check("窗口+cooldown：期内凑满也扣留",
          len(sf.run([rec(tid=7), rec(tid=7)], ts=4.0)) == 0)
    check("窗口+cooldown：过后再报",
          len(sf.run([rec(tid=7), rec(tid=7)], ts=11.0)) == 1)

    o = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=3)]) \
        .run([rec(tid=1), rec(tid=1), rec(tid=1)], ts=0.0)
    check("同批多条按序累计：第 3 条达成", len(o) == 1 and o[0]["events"][0]["detail"]["count"] == 3)

    # ---- 3. 契约细节 ----
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="A"),
                    SeqRule(match="score >= 0.5", event="B")])
    o = sf.run([rec(cls=0, score=0.9)], ts=0.0)
    check("多规则各自命中：同记录多事件",
          len(o) == 1 and [e["name"] for e in o[0]["events"]] == ["A", "B"])
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="A", span=5.0, min_n=3),
                    SeqRule(match="score >= 0.5", event="B")])
    o = sf.run([rec(cls=0, score=0.9)], ts=0.0)
    check("窗口未凑满不拦后规则：B 单独达成",
          len(o) == 1 and [e["name"] for e in o[0]["events"]] == ["B"])

    sf = SeqFilter([SeqRule(match="ts >= 100", event="后期")])
    check("时间取记录自带 ts",
          len(sf.run([rec(ts=100.0)], ts=0.0)) == 1 and len(sf.run([rec(ts=50.0)], ts=0.0)) == 0)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="组", span=5.0, min_n=2)])
    o1 = sf.run([{"cls_id": 0, "bbox": (0, 0, 9, 9)}], ts=0.0)   # 无 track_id
    o2 = sf.run([{"cls_id": 0, "bbox": (0, 0, 9, 9)}], ts=1.0)
    check("key 缺失归 '*' 组同窗累计", len(o1) == 0 and len(o2) == 1)

    sf = SeqFilter([SeqRule(match="kind == plate", event="采信", key="label", span=3.0, min_n=2)])
    p1 = {"kind": "plate", "label": "粤A12345", "score": 0.9}
    p2 = {"kind": "plate", "label": "粤B67890", "score": 0.9}
    sf.run([p1, p2], ts=0.0)
    o = sf.run([p1], ts=1.0)
    check("key 可配（label）：按车牌号分组确认", len(o) == 1 and o[0]["label"] == "粤A12345")

    sf = SeqFilter([SeqRule(match="frame_idx >= 10", event="第十帧后")])
    check("注入 frame_idx 可作条件",
          len(sf.run([rec()], ts=0.0, frame_idx=10)) == 1
          and len(sf.run([rec()], ts=0.0, frame_idx=9)) == 0)

    for bad_kwargs, name in [
        (dict(match="", event="x"), "match 为空"),
        (dict(match="cls_id == 0", event=" "), "event 为空"),
        (dict(match="cls_id == 0", event="x", span=0, min_n=2), "min_n>1 且无窗口"),
        (dict(match="cls_id == 0", event="x", hit_ratio=1.5), "hit_ratio>1"),
        (dict(match="错误的条件 ==", event="x"), "match 非法 DSL"),
    ]:
        try:
            SeqRule(**bad_kwargs)
            check(f"非法规则报错：{name}", False)
        except ValueError:
            check(f"非法规则报错：{name}", True)
    try:
        SeqFilter([])
        check("空规则集报错", False)
    except ValueError:
        check("空规则集报错", True)

    # ---- 4. 与检测筛选组合 ----
    dets = [det((0, 0, 80, 80), tid=7, score=0.95),      # 人，够大，高置信
            det((0, 0, 10, 10), tid=8, score=0.95),      # 人，碎框
            det((0, 0, 80, 80), cls=2, tid=9, score=0.95)]  # 车
    keep = DetFilter("cls_id == 0, short >= 30").run(dets, ts=0.0, frame_idx=0)
    check("前置筛选：碎框先被取舍", len(keep) == 1 and keep[0].track_id == 7)
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="驻留", span=5.0, min_n=2)])
    o1 = sf.run(keep, ts=0.0)
    o2 = sf.run(DetFilter("cls_id == 0, short >= 30").run(
        [det((0, 0, 80, 80), tid=7, score=0.8)], ts=2.0), ts=2.0)
    check("串联：确认闸只放达标条",
          len(o1) == 0 and len(o2) == 1 and getattr(o2[0], "events", [{}])[0].get("name") == "驻留")

    # ---- 5. 门控（throttle / change；原 Pipeline check_interval/track_change_only 迁入） ----
    sf = SeqFilter(throttle={"key": "cls_id", "interval": 3.0})
    g0 = sf.run([det((0, 0, 9, 9), tid=1, cls=0), det((0, 0, 9, 9), tid=3, cls=2)], ts=0.0)
    g1 = sf.run([det((0, 0, 9, 9), tid=1, cls=0)], ts=1.0)
    g3 = sf.run([det((0, 0, 9, 9), tid=1, cls=0), det((0, 0, 9, 9), tid=3, cls=2)], ts=3.0)
    check("throttle：首帧整类放行（原元素）", len(g0) == 2 and all(isinstance(x, Detection) for x in g0))
    check("throttle：间隔内整类扣留", g1 == [])
    check("throttle：到间隔再放行", len(g3) == 2)

    sf = SeqFilter(change={"key": "cls_id", "watch": "track_id"})
    c0 = sf.run([det((0, 0, 9, 9), tid=1), det((0, 0, 9, 9), tid=2)], ts=0.0)
    c1 = sf.run([det((0, 0, 9, 9), tid=1), det((0, 0, 9, 9), tid=2)], ts=1.0)
    c2 = sf.run([det((0, 0, 9, 9), tid=1), det((0, 0, 9, 9), tid=5)], ts=2.0)
    check("change：新集合放行", len(c0) == 2)
    check("change：无变化扣留", c1 == [])
    check("change：集合变化再放行", len(c2) == 2)

    sf = SeqFilter(throttle={"key": "cls_id", "interval": 2.0})
    check("纯门控模式（无规则）合法", len(sf.run([rec(tid=1)], ts=0.0)) == 1)
    for bad in [{}, {"key": "cls_id", "interval": 0}]:
        try:
            SeqFilter(throttle=bad)
            check(f"空门控+无规则报错 {bad}", False)
        except ValueError:
            check(f"空门控+无规则报错 {bad}", True)

    # ---- 6. 三期扩展：once / 计数窗 / hit_ratio / 动态事件名 ----
    sf = SeqFilter([SeqRule(match="kind == plate", event="采信", key="label", cooldown=-1)])
    a = sf.run([{"kind": "plate", "label": "粤A1"}], ts=0.0)
    b = sf.run([{"kind": "plate", "label": "粤A1"}], ts=10.0)
    c = sf.run([{"kind": "plate", "label": "粤B2"}], ts=20.0)
    check("once（cooldown<0）：同 key 只触发一次", len(a) == 1 and len(b) == 0 and len(c) == 1)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="连拍", window_n=3, min_n=3)])
    o1 = sf.run([rec(tid=7)], ts=0.0)
    o2 = sf.run([rec(tid=7)], ts=1.0)
    o3 = sf.run([rec(tid=7)], ts=2.0)
    check("计数窗：最近 N 条内凑满达到成", len(o1) == 0 and len(o2) == 0 and len(o3) == 1)

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="比例", key="track_id",
                            window_n=4, min_n=1, hit_ratio=0.75)])
    seq = [(0, 7), (0, 7), (0, 7), (2, 7), (0, 7)]  # 3 命中 + 1 非命中 + 1 命中
    outs = [sf.run([{"cls_id": c, "track_id": t, "bbox": (0, 0, 9, 9)}], ts=float(i))
            for i, (c, t) in enumerate(seq)]
    check("hit_ratio：窗口内比例达标才触发", [len(o) for o in outs] == [0, 0, 0, 0, 1])

    sf = SeqFilter([SeqRule(match="cls_id == 0", event="警情-{label}")])
    o = sf.run([{"cls_id": 0, "label": "治安类", "bbox": (0, 0, 9, 9)}], ts=0.0)
    check("动态事件名：{字段} 占位", o[0]["events"][0]["name"] == "警情-治安类")

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
