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
    SAMPLING_ANALYSIS,
    SAMPLING_FRAME_COUNT,
    SAMPLING_WALL_CLOCK,
    YOLODetector,
    compose,
    filter_classes,
    filter_confidence,
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
      and s0[0].gran == GRAN_CLASS and s0[0].ref == "crop:cls0"
      and len(s0[0].dets) == 2)
s3, _ = p.step(3, 1.0, [det(1)])           # t=1: 距上次仅1s，不送
check("类节拍: 1s<2s 不送", len(s3) == 0)
s6, _ = p.step(6, 3.0, [det(1)])           # t=3: 距上次3s≥2s 再送
check("类节拍: 3s≥2s 再送", len(s6) == 1 and s6[0].ref == "crop:cls0")
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
# roi_output 覆盖: small_full 也送裁剪
p4 = compose(dict(task_id=9, algo_mode="small_full", class_limits={0: (1, 10)},
                  check_interval=1.0, smooth_frames=1, target_actions=["x"],
                  roi_output=True))
s0, _ = p4.step(0, 0.0, [det(1)])
check("roi_output=True → crop:cls", len(s0) == 1 and s0[0].ref == "crop:cls0")

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

print(f"\n全部通过: {PASS} 项断言")
