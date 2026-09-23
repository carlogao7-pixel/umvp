# -*- coding: utf-8 -*-
"""
四模块管道拼接验证（纯逻辑，不加载模型）

覆盖:
  1. small_only —— 驻留告警 + 按类别冷却 + 过滤链
  2. small_crop —— 按识别类型(类别)报送 + 滑动窗口确认 + 全局冷却
  3. small_full —— 按识别类型送全帧
  4. large_only —— 全局墙钟定时 + 巡检直接落盘
  5. 每类数量上下限 —— 超出上限/不足下限的类别跳过，其它类照常
  6. 过滤链自定义顺序 —— 不同顺序结果一致（交集语义）
  7. FrameManager 三种取图节奏 —— analysis / wall_clock / frame_count

运行: ai 环境 python pipe/test_composer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipe.composer import (
    Detection,
    FrameManager,
    GRAN_CLASS,
    GRAN_SCENE,
    IntervalGate,
    SAMPLING_ANALYSIS,
    SAMPLING_FRAME_COUNT,
    SAMPLING_WALL_CLOCK,
    YOLODetector,
    compose,
    filter_classes,
    filter_confidence,
    filter_min_size,
    filter_model,
)


PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(tid: int, cls: int = 0, score: float = 0.9, model: str = "m1") -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=(10, 10, 100, 200), score=score,
                     model_path=model)


# ============ 1. small_only：驻留告警 + 按类别冷却 ============
print("== 1. small_only ==")
p = compose(dict(task_id=1, algo_mode="small_only", smooth_frames=3,
                 alarm_cooldown=10, extract_frame_rate=3,
                 class_limits={0: (1, 10), 1: (1, 10)}))
_, a0 = p.step(0, 0.0, [det(1)])
_, a1 = p.step(1, 1.0, [det(1)])
_, a2 = p.step(2, 2.0, [det(1)])
check("驻留<3s 不告警", not a0 and not a1 and not a2)
_, a3 = p.step(3, 3.0, [det(1)])
check("驻留>=3s 告警", len(a3) == 1 and a3[0].action == "dwell:0")
# 注意: 帧管理抽帧(3 帧取 1)，有效分析帧为 0,3,6,9...
_, a6 = p.step(6, 6.0, [det(1)])
check("冷却期内不重复告警", not a6)
_, b6 = p.step(6, 6.0, [det(1, cls=1)])
check("按类别冷却独立", len(b6) == 1 and b6[0].action == "dwell:1")

# ============ 2. small_crop：按识别类型报送 + 滑动窗口 + 冷却 ============
print("== 2. small_crop ==")
p = compose(dict(task_id=2, algo_mode="small_crop", class_limits={0: (1, 10)},
                 check_interval=2.0, smooth_frames=2, target_actions=["fire"],
                 alarm_cooldown=10, extract_frame_rate=3))
# 注意: 帧门控(extract_frame_rate=3) 只分析帧 0,3,6,9...；ts 为墙钟秒
s0, _ = p.step(0, 0.0, [det(1), det(2)])   # t=0: 同类多个目标合并成一条"类"请求
check("同类目标合并为一条类请求", len(s0) == 1 and s0[0].track_key == 0
      and s0[0].gran == GRAN_CLASS and s0[0].ref == "full:cls0"
      and len(s0[0].dets) == 2)
s3, _ = p.step(3, 1.0, [det(1)])           # t=1: 距上次仅1s，不送
check("类节拍: 1s<2s 不送", len(s3) == 0)
s6, _ = p.step(6, 3.0, [det(1)])           # t=3: 距上次3s≥2s 再送
check("类节拍: 3s≥2s 再送", len(s6) == 1 and s6[0].ref == "full:cls0")
# 双类别：各自独立计时、各自一条请求
p2 = compose(dict(task_id=2, algo_mode="small_crop",
                  class_limits={0: (1, 10), 1: (1, 10)},
                  check_interval=2.0, smooth_frames=1, target_actions=["fire"],
                  alarm_cooldown=10))
s0, _ = p2.step(0, 0.0, [det(1, 0), det(3, 1)])
check("两类各一条请求", len(s0) == 2 and sorted(r.track_key for r in s0) == [0, 1])
_, _ = p2.step(1, 1.0, [det(1, 0), det(3, 1)])   # 1s 内都不再送
s3, _ = p2.step(3, 3.0, [det(3, 1)])             # cls1 距上次3s 再送; cls0 无目标
check("类别独立计时: 仅 cls1 送", len(s3) == 1 and s3[0].track_key == 1)
# VLM 滑动窗口（p 的 smooth_frames=2）
al = p.on_vlm_result(0, "fire", ts=4.0)
check("窗口未满不告警", not al)
al = p.on_vlm_result(0, "fire", ts=5.0)
check("窗口满+全部命中告警", len(al) == 1 and al[0].action == "fire"
      and al[0].gran == GRAN_CLASS and al[0].track_key == 0)
al = p.on_vlm_result(0, "fire", ts=6.0)
check("冷却期内不重复", not al)

# ============ 3. small_full：按识别类型送全帧 ============
print("== 3. small_full ==")
p = compose(dict(task_id=3, algo_mode="small_full", class_limits={0: (1, 10)},
                 check_interval=2.0, smooth_frames=1, target_actions=["fight"],
                 alarm_cooldown=10))
s0, _ = p.step(0, 0.0, [det(7)])
check("small_full 送该类全帧", len(s0) == 1 and s0[0].ref == "full:cls0"
      and s0[0].gran == GRAN_CLASS)
al = p.on_vlm_result(0, "fight", ts=1.0)
check("smooth_frames=1 立即告警", len(al) == 1)

# ============ 4. large_only：全局墙钟 + 巡检 ============
print("== 4. large_only ==")
p = compose(dict(task_id=4, algo_mode="large_only", check_interval=5.0,
                 smooth_frames=1, target_actions=["issue"], alarm_cooldown=10))
s0, _ = p.step(0, 0.0, None)
s1, _ = p.step(4, 4.0, None)
check("墙钟: t=0 首送", len(s0) == 1 and s0[0].track_key == -1 and s0[0].gran == GRAN_SCENE)
check("墙钟: t=4 未到5s不送", len(s1) == 0)
s2, _ = p.step(5, 5.0, None)
check("墙钟: t=5 到间隔再送", len(s2) == 1)
al = p.on_vlm_result(-1, "issue", ts=6.0)
check("常规模式命中告警", len(al) == 1 and al[0].gran == GRAN_SCENE)
p2 = compose(dict(task_id=5, algo_mode="large_only", check_interval=5.0))  # 无 target_actions
s0, _ = p2.step(0, 0.0, None)
al = p2.on_vlm_result(-1, "任意内容", ts=1.0)
check("巡检结果直接落盘", len(al) == 1 and al[0].action == "inspection")

# ============ 5. 每类数量上下限 ============
print("== 5. 每类数量上下限 ==")
# 类数量超上限 → 该类跳过，其它类照常
p = compose(dict(task_id=6, algo_mode="small_crop", class_limits={0: (1, 2), 1: (1, 10)},
                 check_interval=1.0, smooth_frames=1, target_actions=["crowd"]))
s0, _ = p.step(0, 0.0, [det(1, 0), det(2, 0), det(3, 0), det(4, 1)])
check("类0超上限(3>2)跳过, 类1照常", len(s0) == 1 and s0[0].track_key == 1)
# 类数量不足下限 → 该类跳过
p2 = compose(dict(task_id=6, algo_mode="small_crop", class_limits={0: (3, 5), 1: (1, 10)},
                  check_interval=1.0, smooth_frames=1, target_actions=["crowd"]))
s0, _ = p2.step(0, 0.0, [det(1, 0), det(2, 0), det(4, 1)])
check("类0不足下限(2<3)跳过, 类1照常", len(s0) == 1 and s0[0].track_key == 1)
# 未配置识别对象 → 不报送
p3 = compose(dict(task_id=8, algo_mode="small_crop", class_limits={},
                  check_interval=1.0, smooth_frames=1))
s0, _ = p3.step(0, 0.0, [det(1)])
check("未配置识别对象不报送", len(s0) == 0)

# ============ 6. 过滤链自定义顺序 ============
print("== 6. 过滤链顺序 ==")
filters_a = [filter_model("m1"), filter_confidence(0.8), filter_classes([0])]
filters_b = [filter_classes([0]), filter_model("m1"), filter_confidence(0.8)]
ya = YOLODetector(filters=filters_a, class_limits={0: (1, 10)}, check_interval=1.0)
yb = YOLODetector(filters=filters_b, class_limits={0: (1, 10)}, check_interval=1.0)
dets = [det(1, 0, 0.9), det(2, 0, 0.5), det(3, 1, 0.9), det(4, 0, 0.9, model="m2")]
sa = ya.process(0, 0.0, dets)
sb = yb.process(0, 0.0, dets)
check("不同顺序结果一致", len(sa) == 1 and len(sb) == 1
      and sa[0].dets[0].track_id == sb[0].dets[0].track_id == 1)

# ============ 7. FrameManager 三种取图节奏 ============
print("== 7. FrameManager 节奏 ==")
fa = FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=3)
check("analysis: 每3帧取1帧", [fa.wants_frame(i, float(i)) for i in range(6)] == [True, False, False, True, False, False])
fw = FrameManager(sampling=SAMPLING_WALL_CLOCK, interval_sec=5.0)
check("wall_clock: 每5秒取1帧", [fw.wants_frame(0, 0.0), fw.wants_frame(1, 4.0),
      fw.wants_frame(2, 5.0)] == [True, False, True])
ff = FrameManager(sampling=SAMPLING_FRAME_COUNT, interval_frames=10)
check("frame_count: 每10帧取1帧", [ff.wants_frame(0, 0.0), ff.wants_frame(9, 9.0),
      ff.wants_frame(10, 10.0)] == [True, False, True])

# ============ 8. IntervalGate：阶段间传递门控（统一节流机制） ============
print("== 8. IntervalGate 传递门控 ==")
g = IntervalGate(3.0)
check("门控: 首次放行", g.allow("a", 0.0))
g.note("a", 0.0)
check("门控: 间隔内不放行", not g.allow("a", 2.9))
check("门控: 到间隔再放行", g.allow("a", 3.0))
g.note("a", 3.0)
check("门控: key 之间独立", g.allow("b", 3.5))
g_free = IntervalGate(0.0)
check("门控: interval=0 不节流", g_free.allow("k", 0.0) and g_free.allow("k", 0.1))
g_once = IntervalGate(-1.0)
first = g_once.allow("k", 0.0)
g_once.note("k", 0.0)
check("门控: 负值=只放行一次", first and not g_once.allow("k", 100.0))
g_door = IntervalGate(2.0)
g_door.note("t1", 0.0)
seq = []
for t in (0.5, 1.5, 2.0, 3.9):
    if g_door.allow("t1", t):
        g_door.note("t1", t)
        seq.append(t)
check("门控: 序列放行符合间隔", seq == [2.0], str(seq))

# ============ 9. yolo→下游按识别类型传递门控（pass_dets，与 VLM 报送统一） ============
print("== 9. pass_dets 按识别类型传递 ==")
yg = YOLODetector(check_interval=2.0)
dl = [det(1, cls=0), det(2, cls=0), det(3, cls=2)]
check("pass_dets: 首帧整类放行", len(yg.pass_dets(dl, 0.0)) == 3)
check("pass_dets: 间隔内整类拦截", yg.pass_dets(dl, 1.0) == [])
check("pass_dets: 到间隔再放行", len(yg.pass_dets(dl, 2.0)) == 3)
yfree = YOLODetector(check_interval=0.0)
check("pass_dets: interval=0 每帧放行",
      len(yfree.pass_dets(dl, 0.0)) == 3 and len(yfree.pass_dets(dl, 0.1)) == 3)

# ============ 10. 输出漏斗（OutputPolicy，原告警策略通用化） ============
print("== 10. 输出漏斗七道闸 ==")
from pipe.composer import AlarmPolicy, Finding, OutputPolicy, parse_rules

check("别名: AlarmPolicy is OutputPolicy", AlarmPolicy is OutputPolicy)

# ---- 规则 DSL 解析 ----
rs = parse_rules("conf >= 0.8, features.color == green;features.level == 高")
check("规则: 两组三条件解析", len(rs) == 2 and len(rs[0]) == 2 and len(rs[1]) == 1)
check("规则: 值类型（数字/字符串）", rs[0][0][2] == 0.8 and rs[0][1][2] == "green")
rs2 = parse_rules("label in 交警类|治安类")
check("规则: in 多值解析", isinstance(rs2[0][0][2], list) and len(rs2[0][0][2]) == 2)
check("规则: 空串=关闸", parse_rules("") == [] and parse_rules(None) == [])
for bad in ("不合法表达式", "conf ~ 8", "features.reading >= "):
    try:
        parse_rules(bad)
        check(f"规则: 非法表达式报错 {bad!r}", False)
    except ValueError as e:
        check(f"规则: 非法表达式报错 {bad!r}", "无法解析" in str(e) or "为空" in str(e), str(e))

def fd(key, label, ts, conf=1.0, features=None):
    return Finding(key=key, label=label, ts=ts, conf=conf, features=features or {})

# ---- 求值：数值/字符串/contains/in/缺字段 ----
dash = Finding(key="g1", label="仪表", ts=10.0,
               features={"reading": 82.5, "unit": "℃", "level": "高", "description": "读数超限"})
p = OutputPolicy(task_id=9, kind="inspection", rules="features.reading >= 80, features.unit == ℃")
check("漏斗: 规则命中输出", len(p.accept(dash)) == 1)
p2 = OutputPolicy(task_id=9, kind="inspection", rules="features.reading >= 90")
check("漏斗: 规则不命中拦截", p2.accept(dash) == [])
p3 = OutputPolicy(task_id=9, kind="inspection", rules="features.level in 高|低")
check("求值: in 命中", len(p3.accept(dash)) == 1)
p4 = OutputPolicy(task_id=9, kind="inspection", rules="description contains 超限")
check("求值: contains 命中", len(p4.accept(dash)) == 1)
p5 = OutputPolicy(task_id=9, kind="inspection", rules="features.absent >= 1")
check("求值: 缺字段=不通过", p5.accept(dash) == [])
p6 = OutputPolicy(task_id=9, kind="inspection",
                  rules="features.reading < 60;features.level == 高")
check("求值: OR 任一组命中", len(p6.accept(dash)) == 1)

# ---- 置信度闸 + 事件字段透传 ----
pc = OutputPolicy(task_id=9, kind="inspection", min_conf=0.9)
ev = pc.accept(Finding(key="plate:粤A1", label="粤A1", ts=1.0, conf=0.95,
                       features={"plate_no": "粤A1", "color": "green"}))
check("漏斗: min_conf 放行+事件带特征", len(ev) == 1 and ev[0].conf == 0.95
      and ev[0].features["color"] == "green" and ev[0].action == "inspection")
check("漏斗: min_conf 拦截", pc.accept(fd("plate:粤A2", "粤A2", 2.0, conf=0.5)) == [])

# ---- 巡检同 key 只一次 / key_cooldown 语义 ----
po = OutputPolicy(task_id=9, kind="inspection")
check("巡检: 同 key 只输出一次",
      len(po.accept(fd("plate:粤B2", "粤B2", 1.0))) == 1
      and po.accept(fd("plate:粤B2", "粤B2", 2.0)) == []
      and len(po.accept(fd("plate:粤B3", "粤B3", 2.0))) == 1)
pk = OutputPolicy(task_id=9, kind="inspection", key_cooldown=3.0)
check("巡检: key_cooldown>0 冷却后可再报",
      len(pk.accept(fd("k1", "x", 0.0))) == 1 and pk.accept(fd("k1", "x", 2.0)) == []
      and len(pk.accept(fd("k1", "x", 3.5))) == 1)
pz = OutputPolicy(task_id=9, kind="inspection", key_cooldown=0)
check("巡检: key_cooldown=0 逐条落盘",
      len(pz.accept(fd("k1", "x", 1.0))) == 1 and len(pz.accept(fd("k1", "x", 2.0))) == 1)

# ---- 驻留闸（inspection + min_dwell；dwell 档自动映射见 §1 旧断言） ----
pd = OutputPolicy(task_id=9, kind="inspection", min_dwell=2.0)
check("驻留: 不足 N 秒拦截", pd.accept(fd("k9", "x", 0.0)) == []
      and pd.accept(fd("k9", "x", 1.9)) == [] and len(pd.accept(fd("k9", "x", 2.0))) == 1)

# ---- window + 规则（"命中"含规则结果，滑窗计数随之） ----
pw = OutputPolicy(task_id=9, kind="window", target_actions=["fire"],
                  smooth_frames=2, rules="features.zone == 东区")
fz = lambda ts, zone: Finding(key=0, label="fire", ts=ts, features={"zone": zone})
check("window: 规则不命中不计入滑窗",
      pw.accept(fz(1.0, "西区")) == [] and pw.accept(fz(2.0, "东区")) == []
      and len(pw.accept(fz(3.0, "东区"))) == 1)  # 仅两次规则命中满足窗口

# ============ 11. infer 输出处过滤（置信度+尺寸，对所有下游生效） ============
print("== 11. infer 输出处过滤 ==")
class _Arr:
    """极简数组替身：支持 tolist() 与逐元素 .item()（绕过 numpy/ultralytics 依赖）。"""
    def __init__(self, v): self._v = v
    def tolist(self): return [list(x) for x in self._v]
    def __getitem__(self, i):
        class _E:
            def __init__(self, x): self._x = x
            def item(self): return self._x
        return _E(self._v[i])
class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _Arr(xyxy); self.cls = _Arr(cls); self.conf = _Arr(conf); self.id = None
class _Res:
    def __init__(self, boxes): self.boxes = boxes
class _Model:
    def __init__(self, res): self._res = res
    def predict(self, frame, **kw): return [self._res]

# 三个框：小目标(10x10)、正常(100x60)、低置信(200x80 @0.3)
yd = YOLODetector(model_path="x.pt", filters=[filter_confidence(0.5), filter_min_size(0, 50)])
yd._model = _Model(_Res(_Boxes([[0, 0, 10, 10], [0, 0, 100, 60], [0, 0, 200, 80]],
                               [0, 0, 0], [0.9, 0.9, 0.3])))
d = yd.infer(None)
check("infer: 小目标+低置信视为未识别，只留正常框",
      len(d) == 1 and d[0].bbox == (0, 0, 100, 60), str([x.bbox for x in d]))
check("infer: 过滤后可直接传递（pass_dets 不再过滤）",
      len(yd.pass_dets(d, 0.0)) == 1)

print(f"\n全部通过: {PASS} 项断言")
