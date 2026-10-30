# -*- coding: utf-8 -*-
"""
链路尾输出提炼测试（纯逻辑，零 IO；不加载模型）——doc/design/chain_output.md §3-§5

  1. scope 过滤：events=只取携带事件的记录 / records=全部（无事件链路用）
  2. 配图对位：同帧优先 / 整体顺序兜底 / 配图源耗尽 / 无图记录 → []
  3. image_pick：all=图集全出（默认）/ first=取首张；images 恒列表语义
  4. limit 截断 + truncated 标注；不做 key 去重
  5. 通用字段存在才带（ts/frame_idx/label/track_id/score）
  6. 空输入容错 / 对象输入归一 / 非法 decl 报错 / 零 IO（图原样 ndarray 返回）

运行: conda run -n ai python tests/test_chain_output.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.chain_output import DEFAULT_DECL, collect_output
from pipe.records import add_event

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def img(tag=0):
    """假图（ndarray）：不同 tag 形状不同，便于断言对位。"""
    h, w = 8 + tag, 16 + tag
    return np.zeros((h, w, 3), dtype=np.uint8)


def ev_rec(frame_idx, ts, **kw):
    r = {"kind": "det", "ts": ts, "frame_idx": frame_idx,
         "label": kw.pop("label", "person"), "track_id": kw.pop("tid", 7),
         "score": kw.pop("score", 0.9), "bbox": (0, 0, 10, 10)}
    r.update(kw)
    return r


def main() -> None:
    # ---- 1. scope 过滤 ----
    a, b, c = ev_rec(1, 0.0), ev_rec(2, 1.0), ev_rec(3, 2.0)
    add_event(a, "警情", ts=0.0, detail={"key": "7"})
    out = collect_output([a, b, c], decl={"scope": "events"})
    check("scope=events：只取携带事件的记录", out["count"] == 1
          and out["items"][0]["frame_idx"] == 1 and not out["truncated"])
    out = collect_output([a, b, c], decl={"scope": "records"})
    check("scope=records：全部记录（无事件也保留）", out["count"] == 3
          and all(it["events"] == [] for it in out["items"][1:]))
    check("缺省 decl = scope events（默认不取无事件记录）",
          collect_output([a, b, c])["count"] == 1)

    # ---- 2. 配图对位（Records 源；同帧优先 + 整体顺序兜底 + 耗尽/无图） ----
    m1, m2, m3 = ev_rec(10, 0.0), ev_rec(20, 1.0), ev_rec(30, 2.0)
    add_event(m1, "E", ts=0.0), add_event(m2, "E", ts=1.0), add_event(m3, "E", ts=2.0)
    # 配图源乱序给：帧 20 的先到、帧 10 的后到——同帧优先不受整体顺序影响
    src = [{"kind": "crop", "ts": 1.0, "frame_idx": 20, "image": img(1)},
           {"kind": "crop", "ts": 0.0, "frame_idx": 10, "image": img(2)},
           {"kind": "crop", "ts": 2.0, "frame_idx": 30, "image": [img(3), img(4)]}]  # 图集
    out = collect_output([m1, m2, m3], image_records=src, decl={"scope": "events"})
    check("配图对位：同帧优先（配图源乱序也按 frame_idx 对上）",
          out["items"][0]["images"][0].shape == (10, 18, 3)   # src[1]=img(2)，帧 10
          and out["items"][1]["images"][0].shape == (9, 17, 3))  # src[0]=img(1)，帧 20
    check("配图对位：图集记录 all 全出（默认 image_pick=all）",
          len(out["items"][2]["images"]) == 2)
    check("零 IO：图以 ndarray 原样返回（对象同一性，未拷贝未落盘）",
          out["items"][2]["images"][0] is src[2]["image"][0])

    # 整体顺序兜底：主流帧号在配图源中不存在（首个处理的主流记录取得兜底图）
    m4 = ev_rec(99, 3.0)
    add_event(m4, "E", ts=3.0)
    out = collect_output([m1, m4], image_records=src[:1], decl={"scope": "events"})
    check("配图对位：同帧无 → 整体顺序兜底（下一条未消费配图记录）",
          len(out["items"][0]["images"]) == 1
          and len(out["items"][1]["images"]) == 0)
    # 配图源耗尽 → 空
    out = collect_output([m1, m2, m3], image_records=src[:1], decl={"scope": "events"})
    check("配图对位：配图源耗尽 → 其余 images=[]",
          [len(it["images"]) for it in out["items"]] == [1, 0, 0])
    # 对位到无图记录 → 空
    out = collect_output([m1], image_records=[{"kind": "x", "ts": 0.0, "frame_idx": 10}],
                         decl={"scope": "events"})
    check("配图对位：对位到旁路无图的记录 → images=[]", out["items"][0]["images"] == [])

    # ---- 3. image_pick first/all ----
    out = collect_output([m3], image_records=[src[2]], decl={"scope": "events", "image_pick": "first"})
    check("image_pick=first：图集取首张（长度 1）", len(out["items"][0]["images"]) == 1)
    out = collect_output([m3], image_records=[src[2]], decl={"scope": "events", "image_pick": "all"})
    check("image_pick=all：图集全出（长度 2，输出侧默认）", len(out["items"][0]["images"]) == 2)
    out = collect_output([m1], decl={"scope": "events"})
    check("不配图（image_records=None）：images 恒列表（空）", out["items"][0]["images"] == [])

    # ---- 4. limit 截断 + 不去重 ----
    many = [ev_rec(i, float(i)) for i in range(5)]
    for r in many:
        add_event(r, "E", ts=r["ts"])
    out = collect_output(many, decl={"scope": "events", "limit": 3})
    check("limit 截断 + truncated 标注", out["count"] == 3 and out["truncated"]
          and len(out["items"]) == 3)
    out = collect_output(many, decl={"scope": "events", "limit": 10})
    check("limit 富余：不截断", out["count"] == 5 and not out["truncated"])
    check("缺省 limit=100（设计稿 Q3 拍板）", DEFAULT_DECL["limit"] == 100)
    twin1, twin2 = ev_rec(1, 0.0), ev_rec(1, 0.0)
    add_event(twin1, "E"), add_event(twin2, "E")
    out = collect_output([twin1, twin2], decl={"scope": "events"})
    check("不做 key 去重（同内容记录原样两条）", out["count"] == 2)

    # ---- 5. 通用字段存在才带 ----
    sparse = {"kind": "det", "ts": 1.5, "frame_idx": 8}   # 无 label/track_id/score
    add_event(sparse, "E")
    out = collect_output([sparse], decl={"scope": "events"})
    check("通用字段：缺失的键不出现（存在才带）",
          set(out["items"][0].keys()) == {"ts", "frame_idx", "events", "images"})
    rich = ev_rec(2, 1.0, tid=9, score=0.77, label="car")
    add_event(rich, "E", ts=1.0, detail={"k": 1})
    out = collect_output([rich], decl={"scope": "events"})
    check("通用字段：齐备时全携带 + 事件全明细",
          out["items"][0]["label"] == "car" and out["items"][0]["track_id"] == 9
          and out["items"][0]["score"] == 0.77
          and out["items"][0]["events"][0]["name"] == "E"
          and out["items"][0]["events"][0]["detail"] == {"k": 1})

    # ---- 6. 容错 / 归一 / 报错 ----
    empty = collect_output([], decl={"scope": "events"})
    check("空输入容错（records=[]）", empty == {"count": 0, "truncated": False, "items": []})
    check("records=None 容错", collect_output(None)["count"] == 0)
    check("image_records=[] 容错（配图源空 → 全空图）",
          all(it["images"] == [] for it in collect_output([m1], image_records=[])["items"]))

    class _Obj:   # 对象输入（__dict__ 尽力转换）
        def __init__(self):
            self.kind, self.ts, self.frame_idx, self.label = "det", 0.0, 1, "person"

    o = _Obj()
    add_event(o, "E")
    out = collect_output([o], decl={"scope": "events"})
    check("对象输入：经 to_record 归一（事件走属性）",
          out["count"] == 1 and out["items"][0]["label"] == "person"
          and out["items"][0]["events"][0]["name"] == "E")

    for bad, why in [
        ({"scope": "all"}, 'scope 非法（应 events/records）'),
        ({"image_pick": "best"}, 'image_pick 非法（应 first/all）'),
        ({"limit": 0}, 'limit 须正整数'),
        ({"limit": "x"}, 'limit 须正整数（非数字）'),
    ]:
        try:
            collect_output([m1], decl=bad)
            check(f"非法 decl 报错：{why}", False)
        except ValueError:
            check(f"非法 decl 报错：{why}", True)
    check("decl 未知键忽略 + None 值回落缺省",
          collect_output([m1], decl={"scope": None, "from": "seq_filter#6.dets"})["count"] == 1)

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
