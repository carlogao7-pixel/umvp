# -*- coding: utf-8 -*-
"""
链路运行器（web_lab runner=chain）纯逻辑验证

web_lab 把"已完成链路"按模块链路（stages）存进 presets(kind=pipeline)，
运行前由 _chain_from_spec 按 stages 组装 Pipeline 实例；spec 若带 algo_mode
顶层键则按 _ALGO_EXPECT 校验结构与模式一致。本测试（自拼 spec，不依赖播种）：

  1. 校验断言 —— 未知 algo_mode / stages 与 algo_mode 不一致 / 缺必需阶段
     都抛 ValueError；
  2. 驱动断言 —— 直接喂合成 Detection 驱动 Pipeline.step：
     small_only 按 track 驻留告警、small_yolo_vlm 按类报送+VLM 回填窗口告警、
     large_only 按墙钟秒送整帧，含冷却抑制与 action 命中判定。

2026-08-17：原"播种断言/装配断言"部分随 3 条四模式种子链路（small_only /
small_yolo_vlm / large_only）从 _SEED_PIPES 与 SQL 移除而删除；链路改由
链路拼接方式重建后，本文件仅验证保留的 _chain_from_spec 装配与驱动逻辑。

运行: ai 环境 python tests/test_chain_runner.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from pipe.composer import (
    Detection, Pipeline,
    SAMPLING_ANALYSIS, SAMPLING_WALL_CLOCK,
    GRAN_CLASS, GRAN_SCENE, ALARM_DWELL, ALARM_WINDOW,
)
import server

N_ASSERT = 0


def ok(cond: bool, msg: str) -> None:
    """断言计数（一次运行一条信息）。"""
    global N_ASSERT
    assert cond, msg
    N_ASSERT += 1


def det(track: int = 0, cls: int = 0, score: float = 0.9,
        box=(0, 0, 20, 30)) -> Detection:
    return Detection(track_id=track, cls_id=cls, bbox=tuple(box), score=score)


def chain_spec(algo_mode: str, stages: list) -> dict:
    """构造最小链路 spec（供驱动测试自拼）。stages 为 (module, params) 对。"""
    return {
        "algo_mode": algo_mode,
        "stages": [{"module": mid, "params": params} for mid, params in stages],
    }


# ================= 1. 校验断言 =================
def expect_raise(spec: dict, frag: str) -> None:
    try:
        server._chain_from_spec(spec)
        raise AssertionError(f"[3] 应抛 ValueError（{frag}）")
    except ValueError as e:
        ok(frag in str(e), f"[3] {frag}")

stages_ok = [("frame_manager", {}), ("yolo", {}), ("vlm", {}), ("alarm", {"kind": "window"})]
expect_raise(chain_spec("nope", stages_ok), "未知 algo_mode")
# small_only 要求无 vlm，给了 vlm 应报不一致
expect_raise(chain_spec("small_only", stages_ok), "不一致")
# 缺 frame_manager 必需阶段
expect_raise(chain_spec("small_yolo_vlm", [("yolo", {}), ("vlm", {}), ("alarm", {})]),
             "缺少必需阶段")

# ================= 2. 驱动断言（合成检测，无模型） =================
# --- small_only: 驻留告警 + 冷却 ---
spec = chain_spec("small_only", [
    ("frame_manager", {"sampling": "analysis", "frame_skip": 1}),
    ("yolo", {"class_limits": "0:1,300"}),
    ("alarm", {"kind": "dwell", "smooth_frames": 2, "alarm_cooldown": 60.0}),
])
pipe = server._chain_from_spec(spec)
sub, alarms = pipe.step(0, 0.0, [det(track=7)])
ok(alarms == [], "[4] small_only: 首帧驻留 <2s 不告警")
ok(pipe.step(1, 1.0, [det(track=7)]) == ([], []), "[4] small_only: 1s 驻留 <2s 不告警")
sub, al = pipe.step(2, 2.0, [det(track=7)])
ok(len(al) == 1 and al[0].action == "dwell:0", "[4] small_only: 2s 驻留触发 dwell:0")
ok(sub == [], "[4] small_only: 无 vlm 不产生报送")
ok(pipe.step(3, 3.0, [det(track=7)]) == ([], []), "[4] small_only: 冷却期内不重复告警")

# --- small_yolo_vlm: 按类报送（节拍）+ VLM 回填窗口告警 ---
spec = chain_spec("small_yolo_vlm", [
    ("frame_manager", {"sampling": "analysis", "frame_skip": 1}),
    ("yolo", {"class_limits": "0:1,300", "check_interval": 3.0}),
    ("vlm", {}),
    ("alarm", {"kind": "window", "target_actions": "fire,fight", "smooth_frames": 1}),
])
pipe = server._chain_from_spec(spec)
sub, al = pipe.step(0, 0.0, [det(track=1)])
ok(len(sub) == 1 and sub[0].gran == GRAN_CLASS and sub[0].track_key == 0,
   "[4] small_yolo_vlm: 按类报送 full:cls0")
ok(sub[0].ref == "full:cls0", "[4] small_yolo_vlm: ref=full:cls0")
ok(al == [], "[4] small_yolo_vlm: 报送本身不告警")
ok(pipe.step(1, 1.0, [det(track=1)]) == ([], []), "[4] small_yolo_vlm: 节拍内不重复报送")
ok(pipe.step(3, 3.0, [det(track=1)])[0] != [], "[4] small_yolo_vlm: 3s 节拍到再次报送")
al = pipe.on_vlm_result(0, "fire", 0.0)
ok(len(al) == 1 and al[0].action == "fire", "[4] small_yolo_vlm: VLM 命中 fire 告警")
ok(pipe.on_vlm_result(0, "fire", 0.0) == [], "[4] small_yolo_vlm: 全局冷却抑制重复告警")
ok(pipe.on_vlm_result(0, "walk", 1.0) == [], "[4] small_yolo_vlm: 非目标 action 不告警")
ok(len(pipe.on_vlm_result(0, "fight", 61.0)) == 1, "[4] small_yolo_vlm: 冷却结束 fight 告警")

# --- large_only: 墙钟秒整帧报送 + scene 告警 ---
spec = chain_spec("large_only", [
    ("frame_manager", {"sampling": "wall_clock", "interval_sec": 3.0}),
    ("vlm", {}),
    ("alarm", {"kind": "window", "target_actions": "fire", "smooth_frames": 1}),
])
pipe = server._chain_from_spec(spec)
sub, al = pipe.step(0, 0.0)
ok(len(sub) == 1 and sub[0].gran == GRAN_SCENE and sub[0].ref == "full:global",
   "[4] large_only: 整帧 scene 报送")
ok(pipe.step(1, 1.0) == ([], []), "[4] large_only: 墙钟间隔内不送")
ok(pipe.step(3, 3.0)[0] != [], "[4] large_only: 3s 到再次送整帧")
al = pipe.on_vlm_result(-1, "fire", 0.0)
ok(len(al) == 1 and al[0].gran == GRAN_SCENE, "[4] large_only: scene 命中告警")

print(f"\n全部 {N_ASSERT} 断言通过 ✓")
