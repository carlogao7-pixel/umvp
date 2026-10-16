# -*- coding: utf-8 -*-
"""
警用无人机链路测试（验证扩展功能）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import numpy as np
from pipe.composer import (
    FrameManager, YOLODetector, VLMAnalyzer, Pipeline,
    Detection, SAMPLING_ANALYSIS,
)
from pipe.det_filter import DetFilter
from pipe.seq_filter import SeqFilter, SeqRule

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
    
    # 构建链路（YOLO 纯推理；有效性筛选/节流/报送编排在 Pipeline 与检测筛选阶段）
    fm = FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=5)

    yolo = YOLODetector(
        model_path=str(ROOT / "models/yolov8n.pt"),
        conf=0.35,
        iou=0.7,
        imgsz=640,
        max_det=300,
        classes=[0, 2],
        device=None,
    )
    det_filter = DetFilter(filter="score >= 0.35",
                           group_limits=_class_limits("0:2,300;2:2,300"))
    
    vlm = VLMAnalyzer(
        max_resolution=(1280, 720),
        prompt=ALARM_PROMPT,
    )

    # 事件尾（原「告警策略」window 的等价）：词表命中 + 按警情类型独立冷却 30s，动态事件名
    tail = SeqFilter(rules=[SeqRule(
        match="label in 治安类|交警类|群体性事件类|救援救助类",
        event="{label}", key="label", cooldown=30.0)])

    pipe = Pipeline(frame=fm, yolo=yolo, vlm=vlm,
                    det_filter=det_filter,
                    seq_filter=SeqFilter(throttle={"key": "cls_id", "interval": 5.0},
                                         change={"key": "cls_id", "watch": "track_id"}),
                    event_filter=tail)

    print("✓ 链路构建成功")
    print(f"  - 帧管理: {fm.sampling}, frame_skip={fm.frame_skip}")
    print(f"  - YOLO: 纯推理（conf={yolo.conf}）| 检测筛选: {det_filter.filter}, "
          f"group_limits={det_filter.group_limits}")
    print(f"  - 时序门控: throttle={pipe.seq_filter.throttle}, change={pipe.seq_filter.change}")
    print(f"  - 事件尾: rules={pipe.event_filter.rules}")
    print(f"  - VLM: max_resolution={vlm.max_resolution}")
    
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
    
    # 第一次应该报送（track_change_only 首见必送）
    submits1, _ = pipe.step(0, 0.0, dets=dets_no_change, force=True)
    print(f"第一次处理: {len(submits1)} 个报送请求")
    # 有效性筛选后检查（检测筛选阶段：逐条条件 + 成组数量）
    filtered = det_filter.run(dets_no_change)
    print(f"  过滤后: {len(filtered)} 个检测")
    filtered_by_cls = {}
    for d in filtered:
        filtered_by_cls.setdefault(d.cls_id, []).append(d.track_id)
    print(f"  过滤后分布: cls0:{filtered_by_cls.get(0, [])} | cls2:{filtered_by_cls.get(2, [])}")

    assert len(submits1) == 2, f"应该有 2 个类别的报送（人、车），实际 {len(submits1)}"

    # track_id 相同，不应报送（间隔虽到 5s，track 集合无变化）
    submits2, _ = pipe.step(1, 5.0, dets=dets_no_change, force=True)
    print(f"第二次处理（track_id 相同）: {len(submits2)} 个报送请求")
    assert len(submits2) == 0, f"track_id 相同不应报送，实际 {len(submits2)}"

    # track_id 变化，应该报送
    submits3, _ = pipe.step(2, 10.0, dets=dets_with_change, force=True)
    print(f"第三次处理（track_id 变化）: {len(submits3)} 个报送请求")
    assert len(submits3) == 2, f"track_id 变化应该有 2 个类别的报送，实际 {len(submits3)}"

    print("\n✓ 报送 track_id 变化检测测试通过（Pipeline 编排 + 检测筛选）")
    
    # 事件尾：按警情类型独立冷却（原告警 window 的等价）
    print("\n" + "-" * 40)
    print("测试事件尾：按警情类型独立冷却")
    print("-" * 40)

    test_cases = [
        (0.0, "治安类", True),    # 第一个治安类，应出事件
        (5.0, "治安类", False),   # 5秒内重复治安类，不出
        (10.0, "交警类", True),   # 不同警情类型，出
        (15.0, "交警类", False),  # 5秒内重复交警类，不出
        (35.0, "治安类", True),   # 30秒后治安类冷却结束，出
    ]
    for ts, action, should in test_cases:
        evs = tail.run([{"label": action}], ts=ts)
        actual = len(evs)
        expected = 1 if should else 0
        status = "✓" if actual == expected else "✗"
        print(f"{status} ts={ts:5.1f}s action={action:8s} 事件={actual} (期望={expected})")
        assert actual == expected, f"期望 {expected} 个事件，实际 {actual}"

    print("\n✓ 事件尾按警情类型独立冷却测试通过")

    # 测试链路集成
    print("\n" + "-" * 40)
    print("测试链路集成")
    print("-" * 40)

    # 模拟一帧处理
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 128  # 灰色背景
    submits, events = pipe.step(0, 0.0, dets=dets_no_change, frame=test_frame)
    print(f"链路 step(0): {len(submits)} 个报送请求, {len(events)} 个事件")

    # 模拟 VLM 回填 → 事件尾出事件
    for submit in submits:
        vlm_result = "治安类"
        new_events = pipe.publish([{"kind": "vlm", "label": vlm_result, "text": ""}], ts=0.0)
        print(f"  VLM 回填 track_key={submit.track_key} action={vlm_result} -> {len(new_events)} 个事件")

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