# -*- coding: utf-8 -*-
"""
警用无人机链路视频测试（真实视频）
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import cv2
from pipe.composer import (
    FrameManager, YOLODetector, VLMAnalyzer, AlarmPolicy, Pipeline,
    SAMPLING_ANALYSIS,
    filter_confidence,
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

def _parse_vlm_action(content: str) -> tuple:
    """解析 VLM 返回的 JSON 内容，返回 (alert_type, description)"""
    try:
        # 尝试提取 JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)
            alert_type = data.get("alert_type", "无异常")
            description = data.get("description", "")
            return alert_type, description
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    return "无异常", content

def _mock_vlm_infer(payload: dict) -> str:
    """模拟 VLM 推理（真实场景需要调用外部 API）"""
    # 这里根据测试视频返回不同的警情类型
    # 实际使用时需要替换为真实的 VLM 调用
    
    # 模拟返回 JSON 格式的警情类型和描述
    # 根据时间戳和track_key返回不同的警情类型
    track_key = payload.get("track_key", 0)
    ts = getattr(payload, "ts", 0.0)
    
    # 模拟一些警情变化
    if track_key == 0:  # 人
        if ts < 5.0:
            return json.dumps({
                "alert_type": "治安类",
                "description": "检测到多人聚集，有人持械"
            })
        elif ts < 10.0:
            return json.dumps({
                "alert_type": "群体性事件类",
                "description": "多人聚集进行不正常活动，有条幅拦路"
            })
        else:
            return json.dumps({
                "alert_type": "无异常",
                "description": "正常行人活动"
            })
    elif track_key == 2:  # 车
        if ts < 5.0:
            return json.dumps({
                "alert_type": "交警类",
                "description": "车辆明显形变，疑似交通事故"
            })
        elif ts < 10.0:
            return json.dumps({
                "alert_type": "交警类",
                "description": "车尾有三角警示牌，车辆抛锚"
            })
        else:
            return json.dumps({
                "alert_type": "无异常",
                "description": "正常车辆行驶"
            })
    else:
        return json.dumps({
            "alert_type": "无异常",
            "description": "画面正常"
        })

def run_police_uav_video_test(video_path: str, max_frames: int = 300):
    """运行警用无人机视频测试"""
    print("=" * 80)
    print(f"警用无人机视频测试: {Path(video_path).name}")
    print("=" * 80)
    
    # 构建链路
    fm = FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=15)
    
    yolo = YOLODetector(
        filters=[filter_confidence(0.35)],
        class_limits=_class_limits("0:2,300;2:1,300"),
        check_interval=5.0,
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
    
    print(f"✓ 链路构建成功")
    print(f"  - 抽帧间隔: 每 15 帧")
    print(f"  - YOLO 报送间隔: 5.0 秒")
    print(f"  - YOLO track_id 变化检测: 开启")
    print(f"  - VLM 警情类型: 治安类/交警类/群体性事件类/救援救助类")
    print(f"  - 告警冷却: 30 秒（按警情类型独立）")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频信息: {frame_count} 帧, {fps:.1f} FPS, 约 {frame_count/fps:.1f} 秒")
    
    # 创建输出目录
    out_dir = ROOT / "tests/data/out/police_uav"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 告警输出
    alarm_outputs = []
    
    frame_idx = 0
    analyzed_count = 0
    submit_count = 0
    vlm_call_count = 0
    alarm_count = 0
    
    start_time = time.time()
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            ts = frame_idx / fps
            
            # 链路处理
            submits, alarms = pipe.step(frame_idx, ts, frame=frame)
            
            if frame_idx % 50 == 0 or submits or alarms:
                status = "● 分析" if submits or alarms else "○ 跳过"
                print(f"帧 {frame_idx:4d} ts={ts:6.1f}s {status} "
                      f"→ 报送:{len(submits)} 告警:{len(alarms)}")
            
            if submits:
                analyzed_count += 1
                submit_count += len(submits)
                
                # 模拟 VLM 调用
                for submit in submits:
                    payload = vlm.prepare(submit, frame)
                    vlm_result = _mock_vlm_infer(payload)
                    alert_type, description = _parse_vlm_action(vlm_result)
                    vlm_call_count += 1
                    
                    # VLM 结果回填告警
                    new_alarms = pipe.on_vlm_result(submit.track_key, alert_type, ts, description)
                    
                    if new_alarms:
                        alarm_count += len(new_alarms)
                        for alarm in new_alarms:
                            print(f"  🚨 告警: ts={alarm.ts:.1f}s "
                                  f"警情={alarm.action} "
                                  f"描述={alarm.description}")
                            
                            # 记录告警输出
                            alarm_outputs.append({
                                "timestamp": round(alarm.ts, 2),
                                "alert_type": alarm.action,
                                "description": alarm.description,
                                "frame_idx": frame_idx,
                                "track_key": alarm.track_key
                            })
                            
                            # 保存告警帧
                            alarm_frame_path = out_dir / f"alarm_{alarm_count:03d}_ts{alarm.ts:.1f}s.jpg"
                            cv2.imwrite(str(alarm_frame_path), frame)
            
            if alarms:
                # 直接告警（如驻留告警）
                for alarm in alarms:
                    print(f"  🚨 直接告警: ts={alarm.ts:.1f}s action={alarm.action}")
                    
                    alarm_outputs.append({
                        "timestamp": round(alarm.ts, 2),
                        "alert_type": alarm.action,
                        "description": alarm.description,
                        "frame_idx": frame_idx,
                        "track_key": alarm.track_key
                    })
                    
                    # 保存告警帧
                    alarm_frame_path = out_dir / f"alarm_{alarm_count:03d}_ts{alarm.ts:.1f}s.jpg"
                    cv2.imwrite(str(alarm_frame_path), frame)
                    alarm_count += 1
            
            frame_idx += 1
            
            # 进度显示
            if frame_idx % 100 == 0:
                progress = (frame_idx / min(max_frames, frame_count)) * 100
                print(f"进度: {frame_idx}/{min(max_frames, frame_count)} ({progress:.1f}%)")
    
    finally:
        cap.release()
    
    elapsed = time.time() - start_time
    
    # 保存告警JSON输出
    alarm_json_path = out_dir / f"alarms_{Path(video_path).stem}.json"
    with open(alarm_json_path, 'w', encoding='utf-8') as f:
        json.dump(alarm_outputs, f, ensure_ascii=False, indent=2)
    
    # 统计结果
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print(f"视频: {Path(video_path).name}")
    print(f"处理帧数: {frame_idx}/{frame_count}")
    print(f"分析帧数: {analyzed_count}")
    print(f"YOLO 报送: {submit_count} 次")
    print(f"VLM 调用: {vlm_call_count} 次")
    print(f"告警次数: {alarm_count}")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"告警帧保存: {out_dir}")
    print(f"告警JSON: {alarm_json_path}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="警用无人机视频测试")
    ap.add_argument("video", help="测试视频路径（tests/data/vid下）")
    ap.add_argument("--max-frames", type=int, default=300, help="最大处理帧数")
    
    args = ap.parse_args()
    
    video_path = args.video
    if not Path(video_path).exists():
        video_path = str(ROOT / "tests/data/vid" / video_path)
        if not Path(video_path).exists():
            print(f"✗ 视频不存在: {video_path}")
            print(f"可用的视频文件:")
            vid_dir = ROOT / "tests/data/vid"
            for f in vid_dir.glob("*.*"):
                print(f"  - {f.name}")
            sys.exit(1)
    
    try:
        success = run_police_uav_video_test(video_path, args.max_frames)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)