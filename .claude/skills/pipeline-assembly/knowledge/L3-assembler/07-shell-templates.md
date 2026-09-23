# 壳代码骨架（组装工专属）

> 层：L3-assembler ｜ 读者：组装工 ｜ 参照：umvp/pipe/test_composer.py（脚本化 feed 测试壳）、umvp/pipe/composer.py（Pipeline API）

核心 API 速记：
`compose(spec) → Pipeline`；`p.step(frame_idx, ts, dets)` → `(submits, alarms)`；
`p.vlm.prepare(req, frame)` → payload；`p.vlm.analyze(payload, infer)` → action；
`p.on_vlm_result(track_key, action, ts)` → alarms。

> 2026-09-20 现状：产品链路走「模块自定义拼接」——链路由 pipelines + pipeline_steps
> 存"模块 preset 引用 + 顺序"，web_lab 运行入口 `_chain_from_spec(stages)` 按 stages
> 组装模块。下文的 `compose(spec)` 四模式封装保留作独立脚本/回归测试兼容（`test_composer.py`
> 仍覆盖），新链路优先按 stages 组装。

## 3.1 驱动壳（帧循环）

```python
# -*- coding: utf-8 -*-
"""链路驱动壳 —— 组装工按 SPEC 填充，接真实数据源/VLM 端点后即可运行。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 相对定位项目根

from pipe.composer import compose

SPEC = {
    "task_id": 1,
    "algo_mode": "small_full",             # small_only / small_crop / small_full / large_only（兼容封装；small_crop/full 现均送全帧 full:cls，ROI 裁剪改由目标裁剪阶段）
    "extract_frame_rate": 3,               # analysis 抽帧间隔
    "class_limits": {0: (1, 300)},         # 类别:(数量下限,上限)，超界跳过该类
    "check_interval": 3.0,                 # 送审节拍(秒)
    "target_actions": ["fire", "fight"],   # VLM 输出词表 —— 必须与 vlm_prompt 对齐
    "smooth_frames": 2,                    # window 滑动窗口长度（dwell 模式=驻留秒数）
    "vlm_hit_ratio": 1.0,                  # window 命中比例
    "alarm_cooldown": 60.0,                # 全局冷却(秒)
    "vlm_prompt": "画面中是否存在火或斗殴？只回答: fire / fight / none",
    "yolo_model": "yolov8n.pt",            # yolo_* 透传推理参数（可选，缺省用模块默认）
}

p = compose(SPEC)

def infer(payload: dict) -> str:
    """真实场景：payload 送 vlm_endpoint（qwen3-vl），返回动作词表内字符串；
    测试/离线：返回模拟结果。payload 由 VLMAnalyzer.prepare() 生成。"""
    raise NotImplementedError("由拼接层注入：远端调用或模拟")

for frame_idx, ts, frame_bgr in frame_source():          # 视频/摄像头/图集流
    if not p.frame.wants_frame(frame_idx, ts):
        continue
    dets = p.yolo.infer(frame_bgr, track=(p.alarm.kind == "dwell")) if p.yolo else None
    submits, alarms = p.step(frame_idx, ts, dets)        # large_only 传 None 即可
    for req in submits:                                  # 每"识别类型(类别)"一条请求
        action = p.vlm.analyze(p.vlm.prepare(req, frame_bgr), infer)
        p.on_vlm_result(req.track_key, action, ts)
    for ev in alarms:                                    # AlarmEvent: task_id/gran/track_key/action/ts
        handle_alarm(ev)                                 # 落盘 / 上报
```

## 3.2 接线壳（VLM 真/模拟 + 告警消费）

```python
# 真 VLM：payload → 远端端点（spec: vlm_endpoint / vlm_model）
def infer_real(payload: dict) -> str:
    # 把 payload 发给 VLM 端点，解析返回的动作字符串
    # 注意：输出必须落在 target_actions 词表内，否则告警命中率归零
    return post_vlm(payload)

# 模拟 VLM（离线/回归）：按脚本或规则返回，用于测试管道逻辑
def infer_mock(payload: dict) -> str:
    return MOCK_ACTIONS.get(payload.get("ts"), "none")

# 告警消费：落盘（JSON 行）或上报
def handle_alarm(ev):
    with open(OUT_DIR / "alarms.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev.__dict__, ensure_ascii=False) + "\n")
```

## 3.3 测试壳（断言式，脚本化 feed，不加载模型）

```python
# -*- coding: utf-8 -*-
"""链路测试壳 —— 不加载模型，秒级。真实推理烟测用 tests/test_yolo_params.py 风格。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipe.composer import compose, Detection

PASS = 0
def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))

def det(tid, cls=0, score=0.9, model="m1"):
    return Detection(track_id=tid, cls_id=cls, bbox=(10, 10, 100, 200),
                     score=score, model_path=model)

p = compose(dict(task_id=1, algo_mode="small_crop", class_limits={0: (1, 10)},
                 check_interval=2.0, smooth_frames=2, target_actions=["fire"],
                 alarm_cooldown=10, extract_frame_rate=3))
s0, _ = p.step(0, 0.0, [det(1), det(2)])                  # 帧门控: 只分析帧 0,3,6...
check("同类目标合并为一条类请求", len(s0) == 1 and s0[0].track_key == 0
      and len(s0[0].dets) == 2)
s3, _ = p.step(3, 1.0, [det(1)])
check("类节拍: 1s<2s 不送", len(s3) == 0)
al = p.on_vlm_result(0, "fire", ts=4.0)
check("窗口未满不告警", not al)
al = p.on_vlm_result(0, "fire", ts=5.0)
check("窗口满+命中告警", len(al) == 1 and al[0].action == "fire")
print(f"\n全部通过: {PASS} 项断言")
```

## 3.4 估算接入（ResourceEstimator）

```python
from resources import ResourceEstimator

est = ResourceEstimator(project_root=".")          # 自动读 out/fingerprints.json
est_spec = {"stages": [
    {"module": "frame_manager", "params": {"sampling": "wall_clock", "interval_sec": 1.0}},
    {"module": "yolo", "params": {"model_path": "yolov8n.pt", "classes": "0"}},
    {"module": "vlm", "params": {"ref": "full:cls0", "check_interval": 3.0, "fps": 25}},
    {"module": "alarm", "params": {"kind": "window", "target_actions": "fire"}},
]}
pe = est.estimate_pipeline(est_spec, streams=16, vram_budget_mb=8192)
print(pe.to_dict())   # 单路 + 16 路显存/内存/帧率 + 超预算建议
reco = est.back_calculate(est_spec, streams=16, vram_budget_mb=8192)
print(reco)           # {frame_skip, fps_gpu, expected_vram_mb, suggestions}
```

> 注意：VLM 模块在 `estimate_pipeline` 里按 `check_interval × fps` 折算"每帧平均成本"，
> vlm stage 必须带 `fps` 参数，否则按每帧全成本高估。
