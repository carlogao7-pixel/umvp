# -*- coding: utf-8 -*-
"""
警用无人机链路测试（验证扩展功能）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import cv2
import numpy as np
from pipe.composer import (
    FrameManager, YOLODetector, VLMAnalyzer, AlarmPolicy, Pipeline,
    Detection, SAMPLING_ANALYSIS,
    filter_confidence, filter_min_size,
)

ALARM_PROMPT = """你是一个警情分析专家。请分析画面中的情况，并以 JSON 格式返回分析结果：

{
  "alert_type": "治安类/交警类/群体性事件类/救援救助类/无异常",
  "description": "简要描述观察到的异常情况（无异常时描述正常场景）"
}

警情类型说明：
- 治安类：打架斗殴等治安事件
- 交警类：车辆明显形变、车尾有三角警示牌等车辆抛锚情况
- 群体性事件类：多人聚集进行不正常活动（持械或有条幅拦路）
- 救援救助类：有人溺水、倒地并挥手呼救等情况
- 无异常：画面中未发现上述异常情况

请根据实际情况准确判断，输出标准的 JSON 格式（不要添加额外说明）。"""

def _class_limits(s: str):
    result = {}
    if not s:
        return result
    for part in s.split(";"):
        if ":" not in part:
            continue
        cls_part, limit_part = part.split(":", 1)
        try:
            cls_id = int(cls_part.strip())
        except (TypeError, ValueError):
            continue
        if "," in limit_part:
            lo_str, hi_str = limit_part.split(",", 1)
            try:
                lo = int(lo_str.strip())
                hi = int(hi_str.strip())
                result[cls_id] = (lo, hi)
            except (TypeError, ValueError):
                continue
    return result

def test_police_uav_pipeline():
    """测试警用无人机链路：帧管理→YOLO识人车→VLM研判警情→告警"""
    print("=" * 60)
    print("警用无人机链路测试")
    print("=" * 60)
    
    # 构建链路
    fm = FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=5)
    
    yolo = YOLODetector(
        filters=[filter_confidence(0.35)],
        class_limits=_class_limits("0:2,300;2:2,300"),
        check_interval=5.0,
        roi=True,
        track_change_only=True,
        model_path=str(ROOT / "models/yolov8n.pt"),
        conf=0.35,
        iou=0.7,
        imgsz=640,
        max_det=300,
        classes=[0, 2],
        device=None,
    )
    
    vlm = VLMAnalyzer(
        crop_padding=0.15,
        max_resolution=(1280, 720),
        prompt=ALARM_PROMPT,
    )
    
    alarm = AlarmPolicy(
        task_id=1,
        kind="window",
        target_actions=["治安类", "交警类", "群体性事件类", "救援救助类"],
        smooth_frames=1,
        hit_ratio=1.0,
        alarm_cooldown=30.0,
        cooldown_by_action=True,
    )
    
    pipe = Pipeline(frame=fm, yolo=yolo, vlm=vlm, alarm=alarm)
    
    print("✓ 链路构建成功")
    print(f"  - 帧管理: {fm.sampling}, frame_skip={fm.frame_skip}")
    print(f"  - YOLO: class_limits={yolo.class_limits}, track_change_only={yolo.track_change_only}")
    print(f"  - VLM: crop_padding={vlm.crop_padding}")
    print(f"  - 告警: cooldown_by_action={alarm.cooldown_by_action}, alarm_cooldown={alarm.alarm_cooldown}")
    
    # 测试帧决策
    print("\n" + "-" * 40)
    print("测试帧决策")
    print("-" * 40)
    
    wants = []
    for i in range(20):
        ts = i * 0.1
        want = fm.wants_frame(i, ts)
        wants.append(want)
        status = "●" if want else "○"
        print(f"帧 {i:2d} ts={ts:.1f}s {status} {'取图' if want else '跳过'}")
    
    assert sum(wants) == 4, f"应该取 4 帧，实际取 {sum(wants)} 帧"
    print(f"\n✓ 帧决策测试通过（取图 {sum(wants)} 帧）")
    
    # 测试 YOLO track_id 变化检测
    print("\n" + "-" * 40)
    print("测试 YOLO track_id 变化检测")
    print("-" * 40)
    
    # 模拟检测数据（人=0, 车=2），每个类别至少2个目标
    dets_no_change = [
        Detection(track_id=1, cls_id=0, bbox=(100, 100, 200, 200), score=0.8),
        Detection(track_id=2, cls_id=0, bbox=(300, 300, 400, 400), score=0.7),
        Detection(track_id=3, cls_id=2, bbox=(500, 500, 600, 600), score=0.6),
        Detection(track_id=4, cls_id=2, bbox=(700, 700, 800, 800), score=0.5),
    ]
    
    dets_with_change = [
        Detection(track_id=5, cls_id=0, bbox=(100, 100, 200, 200), score=0.8),
        Detection(track_id=6, cls_id=0, bbox=(300, 300, 400, 400), score=0.7),
        Detection(track_id=7, cls_id=2, bbox=(500, 500, 600, 600), score=0.6),
        Detection(track_id=8, cls_id=2, bbox=(700, 700, 800, 800), score=0.5),
    ]
    
    # 第一次应该报送
    submits1 = yolo.process(0, 0.0, dets_no_change)
    print(f"第一次处理: {len(submits1)} 个报送请求")
    # 检查检测结果分布
    by_cls = {}
    for d in dets_no_change:
        by_cls.setdefault(d.cls_id, []).append(d.track_id)
    print(f"  检测分布: cls0(人):{by_cls.get(0, [])} | cls2(车):{by_cls.get(2, [])}")
    print(f"  YOLO class_limits: {yolo.class_limits}")
    
    # 过滤后检查
    filtered = yolo.filter_only(dets_no_change)
    print(f"  过滤后: {len(filtered)} 个检测")
    filtered_by_cls = {}
    for d in filtered:
        filtered_by_cls.setdefault(d.cls_id, []).append(d.track_id)
    print(f"  过滤后分布: cls0:{filtered_by_cls.get(0, [])} | cls2:{filtered_by_cls.get(2, [])}")
    
    # 由于 check_interval=5.0，只有 ts >= 0 的才会立即报送（首次为 _NEG_INF）
    # 但实际上，由于条件是 ts - last_check >= check_interval，首次应该通过
    assert len(submits1) == 2, f"应该有 2 个类别的报送（人、车），实际 {len(submits1)}"
    
    # track_id 相同，不应报送
    submits2 = yolo.process(1, 5.0, dets_no_change)
    print(f"第二次处理（track_id 相同）: {len(submits2)} 个报送请求")
    assert len(submits2) == 0, f"track_id 相同不应报送，实际 {len(submits2)}"
    
    # track_id 变化，应该报送
    submits3 = yolo.process(2, 10.0, dets_with_change)
    print(f"第三次处理（track_id 变化）: {len(submits3)} 个报送请求")
    assert len(submits3) == 2, f"track_id 变化应该有 2 个类别的报送，实际 {len(submits3)}"
    
    print("\n✓ YOLO track_id 变化检测测试通过")
    
    # 测试告警按警情类型独立冷却
    print("\n" + "-" * 40)
    print("测试告警按警情类型独立冷却")
    print("-" * 40)
    
    # 模拟连续的警情检测结果
    test_cases = [
        (0, 0.0, "治安类", True),   # 第一个治安类，应该告警
        (0, 5.0, "治安类", False),  # 5秒内重复治安类，不应告警
        (0, 10.0, "交警类", True),  # 不同警情类型，应该告警
        (0, 15.0, "交警类", False), # 5秒内重复交警类，不应告警
        (0, 35.0, "治安类", True),  # 30秒后治安类冷却结束，应该告警
    ]
    
    for track_key, ts, action, should_alarm in test_cases:
        alarms = alarm.on_vlm_result(track_key, action, ts)
        expected = 1 if should_alarm else 0
        actual = len(alarms)
        status = "✓" if actual == expected else "✗"
        print(f"{status} ts={ts:5.1f}s action={action:8s} 告警={actual} (期望={expected})")
        assert actual == expected, f"期望 {expected} 个告警，实际 {actual}"
    
    print("\n✓ 告警按警情类型独立冷却测试通过")
    
    # 测试链路集成
    print("\n" + "-" * 40)
    print("测试链路集成")
    print("-" * 40)
    
    # 模拟一帧处理
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 128  # 灰色背景
    submits, alarms = pipe.step(0, 0.0, dets=dets_no_change, frame=test_frame)
    print(f"链路 step(0): {len(submits)} 个报送请求, {len(alarms)} 个告警")
    
    # 模拟 VLM 回填
    for submit in submits:
        # 模拟 VLM 返回警情类型
        vlm_result = "治安类"
        new_alarms = pipe.on_vlm_result(submit.track_key, vlm_result, 0.0)
        print(f"  VLM 回填 track_key={submit.track_key} action={vlm_result} -> {len(new_alarms)} 个告警")
    
    print("\n✓ 链路集成测试通过")
    
    print("\n" + "=" * 60)
    print("全部测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_police_uav_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)