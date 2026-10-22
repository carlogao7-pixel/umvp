# -*- coding: utf-8 -*-
"""
检测筛选模块测试（纯逻辑，不加载模型）：

  1. 条件 DSL v2：范围比较 / AND / OR / NOT / 括号嵌套 / in / 别名
  2. 派生字段：w/h/short/long/area（对象与 dict 两种来源）
  3. DetFilter：Detection 列表 / dict(Record) 列表 / ts·frame_idx 注入 /
     原元素原样返回 / 空 filter 全放行 / 非法条件装配期报错
  4. 兼容：v1 旧式条件解析结构不变（target_crop/annotate 共用引擎回归）
  8. dict/object 双形态等价（放行集合 / dict 进 dict 出 / ts·group_limits 注入）

运行: conda run -n ai python tests/test_det_filter.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.conditions import match_obj, parse_condition
from pipe.det_filter import DetFilter
from pipe.records import records, to_record
from pipe.composer import Detection

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(bbox, cls: int = 0, tid: int = 0, score: float = 0.9) -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=bbox, score=score)


def main() -> None:
    d_person = det((10, 20, 60, 70), cls=0, tid=1, score=0.95)      # 50x50
    d_car = det((110, 20, 160, 70), cls=2, tid=5, score=0.60)       # 50x50
    d_small = det((0, 0, 20, 10), cls=0, tid=2, score=0.40)         # 20x10
    dets = [d_person, d_car, d_small]

    # ---- 1. DSL v2 基础 ----
    check("空条件全通过", match_obj(d_person, parse_condition("")))
    check("单条件等值", match_obj(d_person, parse_condition("cls_id == 0"))
          and not match_obj(d_car, parse_condition("cls_id == 0")))
    check("别名 conf→score", match_obj(d_person, parse_condition("conf >= 0.9"))
          and not match_obj(d_small, parse_condition("conf >= 0.9")))
    check("别名 类别=0（中文+等号）", match_obj(d_person, parse_condition("类别=0")))
    check("in 操作符", match_obj(d_car, parse_condition("cls_id in 0|2"))
          and not match_obj(d_person, parse_condition("cls_id in 2|3")))
    check("not in", not match_obj(d_person, parse_condition("not cls_id in 0|2")))
    check("组内 AND", not match_obj(d_small, parse_condition("cls_id == 0, score >= 0.5")))
    check("组间 OR", match_obj(d_car, parse_condition("cls_id == 1; track_id == 5")))
    check("not 单条件", not match_obj(d_car, parse_condition("not cls_id == 2")))
    check("not 括号组（非·且）",
          match_obj(d_small, parse_condition("not (cls_id == 0, score >= 0.5)"))
          and not match_obj(d_person, parse_condition("not (cls_id == 0, score >= 0.5)")))
    check("not not 双重否定", match_obj(d_car, parse_condition("not not cls_id == 2")))
    ast = parse_condition("(cls_id == 0; score >= 0.99), track_id > 0")
    check("括号嵌套+组内分号：人满足、车不满足",
          match_obj(d_person, ast) and not match_obj(d_car, ast))

    # ---- 2. 派生字段（对象 / dict 双来源） ----
    check("派生 w/h/area", match_obj(d_person, parse_condition("w == 50, h == 50, area == 2500")))
    check("派生 short/long", match_obj(d_small, parse_condition("short == 10, long == 20")))
    check("中文派生别名", match_obj(d_small, parse_condition("短边 >= 10, 长边 <= 20")))
    rec_car = {"cls_id": 2, "score": 0.8, "bbox": (0, 0, 30, 60), "label": "蓝牌"}
    check("dict 记录 + 派生字段", match_obj(rec_car, parse_condition("short == 30, long == 60")))
    check("dict 记录普通字段", match_obj(rec_car, parse_condition("label == 蓝牌")))
    check("引号值可含分隔符",
          match_obj({"label": "a,b"}, parse_condition("label == 'a,b'")))

    # ---- 3. DetFilter ----
    f = DetFilter()
    check("空 filter 全放行", len(f.run(dets)) == 3)
    f = DetFilter("cls_id in 0|2, score >= 0.5")
    out = f.run(dets)
    check("置信+类别筛选", out == [d_person, d_car])
    check("对象原样返回（身份不变）", all(o is d for o, d in zip(out, [d_person, d_car])))
    check("dict 进 dict 出（身份不变）",
          DetFilter("score >= 0.7").run([rec_car]) == [rec_car])
    check("尺寸门槛：短边>=30", DetFilter("short >= 30").run(dets) == [d_person, d_car])
    check("keep 单条判定", DetFilter("not cls_id == 2").keep(d_car) is False)
    check("ts 注入：3 秒后才开始放行",
          DetFilter("ts >= 3").run([d_person], ts=2.9) == []
          and DetFilter("ts >= 3").run([d_person], ts=3.0) == [d_person])
    check("frame_idx 注入", DetFilter("frame_idx > 100").run([d_person], frame_idx=101) == [d_person])
    check("空列表输入", DetFilter("cls_id == 0").run([]) == [])
    check("排除某类：not",
          DetFilter("not cls_id == 2").run(dets) == [d_person, d_small])

    # ---- 4. records 适配 ----
    r = to_record(d_person)
    check("Detection → Record", r["cls_id"] == 0 and r["bbox"] == (10, 20, 60, 70)
          and r["track_id"] == 1)
    rs = records(dets, kind="det", ts=1.5, frame_idx=7)
    check("records 批量+约定字段补齐",
          len(rs) == 3 and all(x["kind"] == "det" and x["ts"] == 1.5 and x["frame_idx"] == 7
                               for x in rs))
    check("records 不覆盖已有字段",
          records([{"ts": 9.0}], ts=1.0)[0]["ts"] == 9.0)

    # ---- 5. 兼容性（v1 结构逐字节不变，target_crop/annotate 共用引擎） ----
    check("v1 解析结构不变",
          parse_condition("cls_id in 0|2, score >= 0.5") ==
          [[("cls_id", "in", ["0", "2"]), ("score", ">=", 0.5)]])
    legacy = parse_condition("track_id == 1")
    check("已解析结构透传", parse_condition(legacy) is legacy)
    check("match 兼容旧结构", match_obj(d_person, legacy)
          and not match_obj(d_car, legacy))
    for bad in ["track_id 1", "(cls_id == 1", "score >=",
                "()", "label == '未闭合"]:
        try:
            DetFilter(bad)
            check(f"非法条件报错: {bad!r}", False)
        except ValueError:
            check(f"非法条件报错: {bad!r}", True)

    # ---- 6. 聚合谓词 count(字段) / count()（同帧分组计数） ----
    c0a = det((0, 0, 50, 50), cls=0, tid=1)
    c0b = det((60, 0, 110, 50), cls=0, tid=2)
    c2 = det((0, 60, 50, 110), cls=2, tid=3)
    batch = [c0a, c0b, c2]
    check("count() 总数达标全放行", DetFilter("count() >= 3").run(batch) == batch)
    check("count() 总数不足全拦", DetFilter("count() >= 4").run(batch) == [])
    check("count(cls_id)>=2 只留同帧≥2的类",
          DetFilter("count(cls_id) >= 2").run(batch) == [c0a, c0b])
    check("count(cls_id)<=1 只留同帧≤1的类", DetFilter("count(cls_id) <= 1").run(batch) == [c2])
    check("count 中文别名 count(类别)", DetFilter("count(类别) >= 2").run(batch) == [c0a, c0b])
    check("count 与逐条 AND",
          DetFilter("cls_id == 0, count(cls_id) >= 2").run(batch) == [c0a, c0b])
    check("not count(...)", DetFilter("not count(cls_id) >= 2").run(batch) == [c2])
    check("count 单条 keep：count()=1",
          DetFilter("count() >= 1").keep(c0a) and not DetFilter("count() >= 2").keep(c0a))
    # 跨帧混入：按记录自带 frame_idx 二次分组（同 cls 不同帧各自计数）
    f1 = {"cls_id": 0, "frame_idx": 1, "bbox": (0, 0, 9, 9)}
    f2a = {"cls_id": 0, "frame_idx": 2, "bbox": (0, 0, 9, 9)}
    f2b = {"cls_id": 0, "frame_idx": 2, "bbox": (0, 0, 9, 9)}
    check("count 按 frame_idx 分组（跨帧不混计）",
          DetFilter("count(cls_id) >= 2").run([f1, f2a, f2b]) == [f2a, f2b])
    check("count 与 group_limits 共存",
          DetFilter("count(cls_id) >= 1", group_limits={0: (1, 300)}).run(batch) == [c0a, c0b])
    # count(子条件)：同帧满足该条件的记录数（帧级常量）
    check("count(子条件) 达标全放行",
          DetFilter("count(cls_id == 0) >= 2").run(batch) == batch)
    check("count(子条件) 不足全拦", DetFilter("count(cls_id == 2) >= 2").run(batch) == [])
    check("count(子条件 AND)",
          DetFilter("count(cls_id == 0, score >= 0.5) >= 2").run(batch) == batch)
    check("count(子条件 OR)",
          DetFilter("count(cls_id == 0; track_id == 3) >= 3").run(batch) == batch)
    check("count(子条件) 与逐条 AND",
          DetFilter("cls_id == 0, count(cls_id == 0) >= 2").run(batch) == [c0a, c0b])
    check("count(子条件) 按 frame_idx 分组",
          DetFilter("count(cls_id == 0) >= 2").run([f1, f2a, f2b]) == [f2a, f2b])
    for bad in ["count", "count(cls_id", "count(cls_id) >=", "count(cls_id == 0"]:
        try:
            DetFilter(bad)
            check(f"count 非法报错: {bad!r}", False)
        except ValueError:
            check(f"count 非法报错: {bad!r}", True)

    # ---- 7. 事件标注（可作链路尾；多事件） ----
    def _rk(cls, tid, alert):
        return {"cls_id": cls, "track_id": tid, "bbox": (0, 0, 50, 50), "content": {"alert": alert}}
    out = DetFilter("cls_id in 0|2", events=["命中"]).run([_rk(0, 1, "治安类"), _rk(2, 2, "无异常")], ts=3.0)
    check("事件：放行记录追加事件（多事件列表）",
          all(r["events"][0]["name"] == "命中" for r in out))
    check("事件：ts 落入事件", out[0]["events"][0]["ts"] == 3.0)
    out = DetFilter("", events=[{"name": "警情", "when": "cls_id == 0",
                                 "detail": ["content"]}]).run([_rk(0, 1, "治安类"), _rk(2, 2, "无异常")], ts=1.0)
    check("事件 when 过滤（未命中不加）", len(out[0]["events"]) == 1 and "events" not in out[1])
    check("事件 detail 取字段（VLM 内容字段）",
          out[0]["events"][0]["detail"]["content"] == {"alert": "治安类"})
    out = DetFilter("", events=[{"name": "A"}, {"name": "B"}]).run([_rk(0, 1, "x")], ts=0.0)
    check("多规格 → 一条记录多事件", [e["name"] for e in out[0]["events"]] == ["A", "B"])
    d_ev = det((0, 0, 50, 50), cls=0, tid=1)
    oo = DetFilter("", events=["E"]).run([d_ev], ts=2.0)
    check("对象输入：事件挂 events 属性",
          getattr(oo[0], "events", [{}])[0].get("name") == "E")
    try:
        DetFilter(events=[{"when": "cls_id == 0"}])
        check("事件规格缺 name 报错", False)
    except ValueError:
        check("事件规格缺 name 报错", True)

    # ---- 8. dict/object 双形态等价（链路通用语回归；本层双形态安全应立即绿） ----
    pair = [det((10, 20, 60, 70), cls=0, tid=1, score=0.95),
            det((110, 20, 160, 70), cls=2, tid=5, score=0.60),
            det((0, 0, 20, 10), cls=0, tid=2, score=0.40)]
    recs = [to_record(d) for d in pair]
    check("dict 字段集与对象一致（含 model_path）",
          all(set(r.keys()) == {"track_id", "cls_id", "bbox", "score", "model_path"}
              for r in recs))

    fe = DetFilter("cls_id in 0|2, score >= 0.5")
    out_obj = fe.run(pair)
    out_rec = fe.run(recs)
    check("双形态等价：放行集合一致（按 bbox/score/cls_id 比较）",
          [(d.bbox, d.score, d.cls_id) for d in out_obj] ==
          [(r["bbox"], r["score"], r["cls_id"]) for r in out_rec])
    check("对象进对象出（元素仍为 Detection）",
          all(isinstance(d, Detection) for d in out_obj))
    check("dict 进 dict 出（元素仍为 dict 且字段齐全含 model_path）",
          all(isinstance(r, dict) and "model_path" in r for r in out_rec))

    # ts/frame_idx 注入对双形态等价生效
    check("ts 注入双形态等价",
          len(DetFilter("ts >= 3").run(pair, ts=2.9)) == 0
          and len(DetFilter("ts >= 3").run(recs, ts=2.9)) == 0
          and len(DetFilter("ts >= 3").run(pair, ts=3.0)) == 3
          and len(DetFilter("ts >= 3").run(recs, ts=3.0)) == 3)
    # group_limits 成组判定双形态等价
    fg = DetFilter("", group_limits={0: (1, 300)})
    check("group_limits 双形态等价",
          [d.track_id for d in fg.run(pair)] == [r["track_id"] for r in fg.run(recs)])

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
