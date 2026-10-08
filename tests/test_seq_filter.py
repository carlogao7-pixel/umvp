# -*- coding: utf-8 -*-
"""
时序判定模块测试（纯逻辑，不加载模型）：

  1. 单条规则：命中即放行+事件标注（event/event_info）/ 不命中扣留 / 冷却节流
  2. 窗口规则：凑满 n 次前扣留、第 n 条放行 / 窗口过期淘汰 / key 分组 /
     达标重臂 / cooldown
  3. 契约：输出统一 Record(dict) / 先匹配先得 / 记录自带 ts / key 缺失归 "*" /
     非法构造报错
  4. 组合：DetFilter(取舍) → SeqFilter(确认) 串联

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
    check("单条：事件字段标注", out[0]["event"] == "高置信")
    check("单条：event_info 内容",
          out[0]["event_info"]["count"] == 1 and out[0]["event_info"]["span"] == 0
          and out[0]["event_info"]["ts"] == 0.0 and out[0]["event_info"]["key"] == "7")
    check("输出为 Record(dict)，对象已转换", isinstance(out[0], dict) and out[0]["cls_id"] == 0)

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
    check("窗口：第 n 条放行", len(o3) == 1 and o3[0]["event"] == "驻留")
    check("窗口：event_info.count", o3[0]["event_info"]["count"] == 3)
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
    check("同批多条按序累计：第 3 条达成", len(o) == 1 and o[0]["event_info"]["count"] == 3)

    # ---- 3. 契约细节 ----
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="A"),
                    SeqRule(match="score >= 0.5", event="B")])
    o = sf.run([rec(cls=0, score=0.9)], ts=0.0)
    check("先匹配先得：只取第一条命中规则", len(o) == 1 and o[0]["event"] == "A")
    sf = SeqFilter([SeqRule(match="cls_id == 0", event="A", span=5.0, min_n=3),
                    SeqRule(match="score >= 0.5", event="B")])
    o = sf.run([rec(cls=0, score=0.9)], ts=0.0)
    check("窗口未凑满也认领：后规则不放行", len(o) == 0)

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
        (dict(match="cls_id == 0", event="x", span=0, min_n=2), "min_n>1 且 span=0"),
        (dict(match="cls_id == 0", event="x", cooldown=-1), "cooldown 为负"),
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
    check("串联：确认闸只放达标条", len(o1) == 0 and len(o2) == 1 and o2[0]["event"] == "驻留")

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
