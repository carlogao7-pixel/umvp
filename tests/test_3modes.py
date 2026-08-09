# -*- coding: utf-8 -*-
"""
三种分析模式实机测试（真实视频 + 真实 YOLO + 规则桩 VLM）

配置来源：tests/分析模式测试.md
  模式一 small_only  face1.MP4    识别对象=人 驻留 5s + 按类别冷却 60s
  模式二 small_full  rescue1.MP4  按画面送全帧(full:scene) + 窗口命中(security/gathering/traffic/rescue)
  模式三 large_only  traffic.MP4  按秒取图(full:global) + 窗口命中(同上)

VLM 说明：拼接层把素材包+画面发给真实大模型（qwen3-vl-32b，OpenAI 兼容端点），
  拿到 JSON 后取 result 字段作 action，喂回告警策略。这验证的是
  "管道行为"（取图节奏/报送粒度/命中判定/冷却）+ 真实大模型端到端。

YOLO 说明：推理走 YOLODetector 模块（p.yolo.infer），不再有独立 yolo_dets() 桩。
  推理参数（conf/iou/imgsz/max_det 等）直接写进各模式 compose 的 spec（yolo_* 前缀），
  调参只需改 spec，正好验证"模块化拼接、测试只调参"的设计初衷。

运行: ai 环境 python tests/test_3modes.py

注意：本文件为独立部署包内的路径适配版（原版 tinymodel_test/test_3modes.py）。
  改动：模块根 tinymodel → umvp；模型 spaz/custom_model → models；
  测试数据 tinymodel_test/data → tests/data。VLM 端点需在目标机可访问。
"""
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import requests

# ---- 大模型 API（OpenAI 兼容，无 key）----
VLM_URL = "http://117.42.21.253:8000/v1/chat/completions"
VLM_MODEL = "qwen3-vl-32b"

ROOT = Path(__file__).resolve().parents[1]  # umvp_standalone/
sys.path.insert(0, str(ROOT / "umvp"))

from pipe.composer import (
    GRAN_CLASS,
    GRAN_SCENE,
    compose,
    filter_confidence,
)

DATA = ROOT / "tests/data"
YOLO_MODEL = ROOT / "models/yolov8n.pt"  # COCO: person=0, car=2
CLS_PERSON, CLS_CAR = 0, 2
DET_CONF = 0.35          # YOLO 检出阈值（放宽，交给模块过滤链把关）
FILTER_CONF = 0.5        # 模块过滤链：置信度 >= 0.5
TARGET_ACTIONS = ["security", "gathering", "traffic", "rescue"]  # 与提示词 result 键一致

PROMPT_RES = (
    "你是一个严谨的视觉分类器，负责对视频帧画面进行分析，判断是否存在目标动作。"
    "请按以下规则依次判断：\n"
    "【治安类 - security】画面出现打架斗殴、胁迫行为、抢劫、破坏公物行为\n"
    "【群体事件类 - gathering】画面中出现明显的人群聚集区域\n"
    "【交通类 - traffic】画面中的车辆有明显的变形，或是在其所在道路的延申不远处有三角警示牌\n"
    "【救援类 - rescue】失能倒地：人员呈非自然卧姿（俯卧/仰卧/侧蜷），或旁人有弯腰查看/搀扶动作\n"
    "请根据上述规则，结合多帧画面进行综合判断，最终以 JSON 格式输出结果，"
    '格式为：{"result": "xxx", "description": "对检测到的警情的简要中文描述"}，'
    '其中 description 为一句简短的简体中文描述，说明画面中具体发生了什么。'
    '无警情时，输出result="none"，无需额外描述'
)

PASS = 0
FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))
    else:
        FAILS.append(name)
        print(f"  [FAIL] {name}" + (f" | {detail}" if detail else ""))


# YOLO 推理走 YOLODetector 模块（p.yolo.infer），参数经 compose spec 的 yolo_* 配置；
# 测试里只调 compose 的参数即可，不再有独立的 yolo_dets() 桩


def _encode_frame(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("帧转 JPEG 失败")
    return base64.b64encode(buf.tobytes()).decode()


def vlm_call(payload: dict, frame) -> str:
    """拼接层：素材包 + 画面 → 大模型 → 取 result 作 action（无 key 的 OpenAI 兼容端点）。"""
    # 按素材包的缩放上限先缩图，省带宽和省大模型算力
    mr = tuple(payload.get("max_resolution") or (1280, 720))
    h, w = frame.shape[:2]
    scale = min(mr[0] / w, mr[1] / h)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    b64 = _encode_frame(frame)
    prompt = payload.get("prompt") or PROMPT_RES
    body = {
        "model": VLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    for attempt in (1, 2):  # 重试一次后报错
        try:
            resp = requests.post(VLM_URL, json=body, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_vlm_result(content)
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == 1:
                print(f"    [VLM重试] {e}")
                continue
            raise RuntimeError(f"大模型调用失败: {e}") from e
    return "none"


def parse_vlm_result(content: str) -> str:
    """大模型可能回纯 JSON 或 ```json 包裹，取 result 字段；失败兜底正则。"""
    txt = content.strip()
    if txt.startswith("```"):
        parts = txt.split("\n", 1)
        txt = parts[1] if len(parts) > 1 else ""
        txt = txt.rsplit("```", 1)[0].strip()
    try:
        return str(json.loads(txt).get("result", "none"))
    except Exception:
        m = re.search(r'"result"\s*:\s*"([^"]+)"', txt)
        return m.group(1) if m else "none"


# ---------------- 模式一：small_only（驻留告警） ----------------
def run_small_only() -> None:
    print("\n== 模式一 small_only | face1.MP4 | 识别对象=人(0) | 驻留5s 冷却60s ==")
    p = compose(dict(
        task_id=1, algo_mode="small_only", extract_frame_rate=3,
        filters=[filter_confidence(FILTER_CONF)],
        class_limits={CLS_PERSON: (1, 10)},  # 识别对象=人，数量范围 1~10
        smooth_frames=5, alarm_cooldown=60.0,
        # YOLO 推理参数（dwell 需稳定 track_id → 模块内自动 track=True）
        yolo_model=str(YOLO_MODEL), yolo_conf=DET_CONF, yolo_imgsz=640,
    ))
    cap = cv2.VideoCapture(str(DATA / "face1.MP4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alarms = []
    for idx in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        # step 内部: wants_frame 为真 → yolo.infer(track=True) → 驻留判定
        _, al = p.step(idx, ts, None, frame)
        alarms += al
    cap.release()
    print(f"  分析帧数: {n//3+1} | 告警 {len(alarms)} 条")
    for a in alarms:
        print(f"    ts={a.ts:6.1f}s  action={a.action}  track={a.track_key}")
    check("驻留触发告警(人待够5s)", len(alarms) >= 1, f"实际 {len(alarms)} 条")
    check("告警动作是 dwell:{类别}", all(a.action == "dwell:0" for a in alarms))
    gaps = [b.ts - a.ts for a, b in zip(alarms, alarms[1:])]
    check("同类别冷却≥60s", all(g >= 59.9 for g in gaps), f"间隔 {gaps}")


# ---------------- 模式二：small_full（按画面送全帧 + 大模型窗口命中） ----------------
def run_small_full() -> None:
    print("\n== 模式二 small_full | rescue1.MP4 | 按画面送全帧 命中则报 ==")
    p = compose(dict(
        task_id=2, algo_mode="small_full",
        extract_frame_rate=3, check_interval=3.0,
        filters=[filter_confidence(FILTER_CONF)],
        class_limits={CLS_PERSON: (1, 10)},  # 识别对象=人，数量范围 1~10
        crop_padding=0.15, vlm_max_resolution=(1280, 720),
        vlm_prompt=PROMPT_RES, target_actions=TARGET_ACTIONS,
        smooth_frames=1, vlm_hit_ratio=1.0, alarm_cooldown=60.0,
        # YOLO 推理参数（窗口模式 → 模块内 track=False）
        yolo_model=str(YOLO_MODEL), yolo_conf=DET_CONF, yolo_imgsz=640,
    ))
    cap = cv2.VideoCapture(str(DATA / "rescue1.MP4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    submits, alarms, cls_ts = [], [], []
    for idx in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        ss, _ = p.step(idx, ts, None, frame)  # 模块自推理（非 dwell → track=False）
        for req in ss:
            submits.append(req)
            payload = p.vlm.prepare(req, frame)
            action = vlm_call(payload, frame)  # 拼接层：送审 → 大模型 → 取 result
            print(f"    [送审] ts={ts:6.1f}s {req.ref} -> 大模型: {action}")
            alarms += p.on_vlm_result(req.track_key, action, ts)
            if req.gran == GRAN_CLASS:
                cls_ts.append(ts)
    cap.release()
    print(f"  报送 {len(submits)} 条 | 告警 {len(alarms)} 条 | 报送时刻: {[round(t,1) for t in cls_ts[:10]]}...")
    check("报送为按识别类型全帧 full:cls*", all(r.ref.startswith("full:cls") and r.gran == GRAN_CLASS for r in submits),
          f"实际 ref 集合: {sorted({r.ref for r in submits})}")
    # 每类独立节拍: 同类隔够 check_interval(3s) 才再送；YOLO 漏检的帧不产生报送，
    # 因此允许出现 >3s 的空档（场景粒度正确行为），只验证"不短于节拍"
    gaps = [round(b - a, 2) for a, b in zip(cls_ts, cls_ts[1:])]
    check("按识别类型报送节拍≥3s(同类)", all(b - a >= 3.0 - 0.2 for a, b in zip(cls_ts, cls_ts[1:])),
          f"间隔 {gaps[:10]}")
    check("命中 rescue 告警", len(alarms) >= 1 and all(a.action == "rescue" for a in alarms),
          f"实际 {len(alarms)} 条: {[(round(a.ts,1), a.action) for a in alarms]}")
    gaps = [b.ts - a.ts for a, b in zip(alarms, alarms[1:])]
    check("全局冷却≥60s", all(g >= 59.9 for g in gaps), f"间隔 {gaps}")


# ---------------- 模式三：large_only（按秒整帧 + 大模型窗口命中） ----------------
def run_large_only() -> None:
    print("\n== 模式三 large_only | traffic.MP4 | 每3s整帧 命中则报 ==")
    p = compose(dict(
        task_id=3, algo_mode="large_only", check_interval=3.0,
        crop_padding=0.15, vlm_max_resolution=(1280, 720),
        vlm_prompt=PROMPT_RES, target_actions=TARGET_ACTIONS,
        smooth_frames=1, vlm_hit_ratio=1.0, alarm_cooldown=60.0,
    ))
    cap = cv2.VideoCapture(str(DATA / "traffic.MP4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    submits, alarms, sub_ts = [], [], []
    for idx in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        ss, _ = p.step(idx, ts, None, frame)  # large_only 无 yolo，frame 无实际效果
        for req in ss:
            submits.append(req)
            payload = p.vlm.prepare(req, frame)
            action = vlm_call(payload, frame)  # 拼接层：送审 → 大模型 → 取 result
            print(f"    [送审] ts={ts:6.1f}s {req.ref} -> 大模型: {action}")
            alarms += p.on_vlm_result(req.track_key, action, ts)
            sub_ts.append(ts)
    cap.release()
    print(f"  报送 {len(submits)} 条 | 告警 {len(alarms)} 条 | 报送时刻: {[round(t,1) for t in sub_ts[:12]]}...")
    check("报送为整帧 full:global", all(r.ref == "full:global" and r.gran == GRAN_SCENE for r in submits))
    check("按秒报送节拍≈3s", all(abs(b - a - 3.0) < 0.05 for a, b in zip(sub_ts, sub_ts[1:])))
    check("命中 traffic 告警", len(alarms) >= 1 and all(a.action == "traffic" for a in alarms),
          f"实际 {len(alarms)} 条: {[(round(a.ts,1), a.action) for a in alarms]}")
    gaps = [b.ts - a.ts for a, b in zip(alarms, alarms[1:])]
    check("全局冷却≥60s", all(g >= 59.9 for g in gaps), f"间隔 {gaps}")


if __name__ == "__main__":
    t0 = __import__("time").time()
    run_small_only()
    run_small_full()
    run_large_only()
    print(f"\n通过 {PASS} 项", "全部通过" if not FAILS else f"失败 {len(FAILS)} 项: {FAILS}")
    print(f"总耗时 {__import__('time').time()-t0:.1f}s")
