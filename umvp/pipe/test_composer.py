# -*- coding: utf-8 -*-
"""
管道拼接验证（纯逻辑，不加载模型）

覆盖:
  1. small_only 已弃用（compose 报错，随输出处理删除）
  2. small_crop —— 按识别类型(类别)报送 + 类节拍(seq_filter 节流)
  3. small_full —— 按识别类型送全帧
  4. large_only —— 全局墙钟定时
  5. 每类数量上下限 —— 超出上限/不足下限的类别跳过，其它类照常
  6. 有效性筛选（检测筛选阶段承接）—— 谓词链顺序无关 + DSL 等价 + 成组判定
  7. FrameManager 三种取图节奏 —— analysis / wall_clock / frame_count
  8. IntervalGate 传递门控
  9. 时序门控（seq_filter 整类节流，原 Pipeline 编排迁入）
  11. infer 纯推理 + 检测筛选 + 事件尾（last_events / publish）

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
    Pipeline,
    YOLODetector,
    compose,
)
from pipe.det_filter import DetFilter


PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(tid: int, cls: int = 0, score: float = 0.9, model: str = "m1") -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=(10, 10, 100, 200), score=score,
                     model_path=model)


# ============ 1. small_only 已弃用 ============
print("== 1. small_only 弃用 ==")
try:
    compose(dict(task_id=1, algo_mode="small_only", class_limits={0: (1, 10)}))
    check("small_only 已弃用（报错）", False)
except ValueError:
    check("small_only 已弃用（报错）", True)

# ============ 2. small_crop：按识别类型报送 + 类节拍 ============
print("== 2. small_crop ==")
p = compose(dict(task_id=2, algo_mode="small_crop", class_limits={0: (1, 10)},
                 check_interval=2.0, extract_frame_rate=3))
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
                  class_limits={0: (1, 10), 1: (1, 10)}, check_interval=2.0))
s0, _ = p2.step(0, 0.0, [det(1, 0), det(3, 1)])
check("两类各一条请求", len(s0) == 2 and sorted(r.track_key for r in s0) == [0, 1])
_, _ = p2.step(1, 1.0, [det(1, 0), det(3, 1)])   # 1s 内都不再送
s3, _ = p2.step(3, 3.0, [det(3, 1)])             # cls1 距上次3s 再送; cls0 无目标
check("类别独立计时: 仅 cls1 送", len(s3) == 1 and s3[0].track_key == 1)

# ============ 3. small_full：按识别类型送全帧 ============
print("== 3. small_full ==")
p = compose(dict(task_id=3, algo_mode="small_full", class_limits={0: (1, 10)},
                 check_interval=2.0))
s0, _ = p.step(0, 0.0, [det(7)])
check("small_full 送该类全帧", len(s0) == 1 and s0[0].ref == "full:cls0"
      and s0[0].gran == GRAN_CLASS)

# ============ 4. large_only：全局墙钟 ============
print("== 4. large_only ==")
p = compose(dict(task_id=4, algo_mode="large_only", check_interval=5.0))
s0, _ = p.step(0, 0.0, None)
s1, _ = p.step(4, 4.0, None)
check("墙钟: t=0 首送", len(s0) == 1 and s0[0].track_key == -1 and s0[0].gran == GRAN_SCENE)
check("墙钟: t=4 未到5s不送", len(s1) == 0)
s2, _ = p.step(5, 5.0, None)
check("墙钟: t=5 到间隔再送", len(s2) == 1)

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

# ============ 6. 有效性筛选（检测筛选阶段承接原 YOLO 过滤链） ============
print("== 6. 有效性筛选 ==")
pred_a = [lambda d: (not d.model_path or d.model_path == "m1"),
          lambda d: d.score >= 0.8, lambda d: d.cls_id == 0]
pred_b = list(reversed(pred_a))  # 顺序无关：交集语义
dfa = DetFilter(predicates=pred_a, group_limits={0: (1, 10)})
dfb = DetFilter(predicates=pred_b, group_limits={0: (1, 10)})
dets = [det(1, 0, 0.9), det(2, 0, 0.5), det(3, 1, 0.9), det(4, 0, 0.9, model="m2")]
ka = dfa.run(dets)
kb = dfb.run(dets)
check("谓词链不同顺序结果一致", [d.track_id for d in ka] == [d.track_id for d in kb] == [1])
dfd = DetFilter(filter="cls_id == 0, score >= 0.8", predicates=[lambda d: d.model_path in (None, "m1")],
                group_limits={0: (1, 10)})
check("DSL 与谓词等价（AND 叠加）", [d.track_id for d in dfd.run(dets)] == [1])
check("成组判定：空表=全拦（旧 class_limits 语义）", DetFilter(group_limits={}).run(dets) == [])
check("成组判定：None=不判定", [d.track_id for d in DetFilter().run(dets)] == [1, 2, 3, 4])

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

# ============ 9. 时序门控（seq_filter 整类节流，原 Pipeline 编排迁入） ============
print("== 9. 整类节流门控 ==")
from pipe.seq_filter import SeqFilter
g = SeqFilter(throttle={"key": "cls_id", "interval": 2.0})
dl = [det(1, cls=0), det(2, cls=0), det(3, cls=2)]
check("节流: 首帧整类放行", len(g.run(dl, ts=0.0)) == 3)
check("节流: 间隔内整类拦截", g.run(dl, ts=1.0) == [])
check("节流: 到间隔再放行", len(g.run(dl, ts=2.0)) == 3)
p0 = compose(dict(task_id=2, algo_mode="small_crop", class_limits={0: (1, 10)},
                  check_interval=0.0, smooth_frames=1, target_actions=["x"]))
check("check_interval=0：不建门控（每帧放行）", p0.seq_filter is None)

# ============ 11. infer 纯推理输出 + 检测筛选阶段过滤（对所有下游生效） ============
print("== 11. infer 纯推理 + det_filter ==")
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
yd = YOLODetector(model_path="x.pt")
yd._model = _Model(_Res(_Boxes([[0, 0, 10, 10], [0, 0, 100, 60], [0, 0, 200, 80]],
                               [0, 0, 0], [0.9, 0.9, 0.3])))
d = yd.infer(None)
check("infer: 纯推理输出（不过滤，3 框全出）", len(d) == 3)
dfy = DetFilter(filter="score >= 0.5, long >= 50")
check("det_filter: 小目标+低置信视为未识别，只留正常框",
      [x.bbox for x in dfy.run(d)] == [(0, 0, 100, 60)], str([x.bbox for x in dfy.run(d)]))
# Pipeline 集成：det_filter 对裁剪/标注/报送一致生效
class _Crop:
    def __init__(self): self.n = []
    def run(self, frame, dets, ts): self.n.append(len(dets)); return list(dets)
cp = _Crop()
pi = Pipeline(frame=FrameManager(frame_skip=1),
              yolo=yd, crop=cp, det_filter=DetFilter(filter="score >= 0.5, long >= 50"))
subs, _ = pi.step(0, 0.0, frame=None, force=True)  # frame=None → 不自推理，dets=None
yd2 = YOLODetector(model_path="x.pt"); yd2._model = yd._model
pi2 = Pipeline(frame=FrameManager(frame_skip=1),
                yolo=yd2, crop=cp,
               det_filter=DetFilter(filter="score >= 0.5, long >= 50", events=["筛出"]))
subs2, _ = pi2.step(0, 0.0, frame=object(), force=True)
check("Pipeline: 裁剪拿到有效性筛选后的检测", cp.n[-1] == 1)
check("Pipeline: last_events 汇总末段记录事件（链路尾通用输出）",
      len(pi2.last_events) == 1 and pi2.last_events[0]["name"] == "筛出")
# Pipeline.publish：外部记录（如 VLM 结果）经事件尾出事件
pp = Pipeline(frame=FrameManager(frame_skip=1),
              event_filter=DetFilter(filter="cls_id == 0",
                                     events=[{"name": "vlm", "detail": ["content"]}]))
evs = pp.publish([{"cls_id": 0, "content": {"r": 1}}, {"cls_id": 2, "content": {}}], ts=1.0)
check("publish：外部记录经尾部出事件",
      len(evs) == 1 and evs[0]["name"] == "vlm" and evs[0]["detail"]["content"] == {"r": 1})

print(f"\n全部通过: {PASS} 项断言")
