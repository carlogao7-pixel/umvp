# -*- coding: utf-8 -*-
"""validate_chain 语义校验断言测试（用例取自 2026-10 免费模型链路组装实验的真实病例）。

运行: conda run -n ai python tests/test_validate_chain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from validate_chain import validate  # noqa: E402


def _stages(*pairs):
    return {"name": "t", "stages": [{"module_id": m, "params": p} for m, p in pairs]}


BASE_PLATE = [
    ("frame_manager", {"sampling": "analysis", "frame_skip": 5, "interval_sec": 3.0,
                       "interval_frames": 30, "backpressure": "standard"}),
    ("yolo", {"model_path": "models/yolov8n.pt", "conf": 0.25, "iou": 0.6,
              "imgsz": 1280, "max_det": 300, "classes": "2", "device": ""}),
    ("det_filter", {"filter": "long >= 40", "group_limits": "2:1,300"}),
    ("target_crop", {"out_size": "×2", "margin": 0.1, "classes": "2", "filter": ""}),
    ("plate_recog", {"detect_level": "high", "mode": "crop", "margin": 0.1,
                     "dedup_iou": 0.6, "track_dedup": True, "track_cooldown": 5.0}),
    ("seq_filter", {"throttle": "", "change": "",
                    "rules": '[{"match": "kind == plate", "event": "{label}", '
                             '"key": "label", "cooldown": -1.0}]'}),
]


def test_baseline_ok():
    assert validate(_stages(*BASE_PLATE)) == [], "PIPE0002 形态基线应全绿"


def test_seed_forms_ok():
    # 另两条保留链路核心形态（警用含 annotate/vlm；人脸截取终段 face_detect）
    uav = _stages(
        ("frame_manager", {"sampling": "analysis", "frame_skip": 15}),
        ("yolo", {"model_path": "yolov8n.pt", "classes": "0,2"}),
        ("det_filter", {"group_limits": "0:2,300;2:1,300"}),
        ("annotate", {"thickness": 2, "show_label": True}),
        ("vlm", {"use_real_vlm": False}),
        ("seq_filter", {"throttle": "cls_id:3.0",
                        "rules": '[{"match": "label in 治安类|交警类", "event": "{label}",'
                                 '"key": "label", "cooldown": 30.0}]'}))
    assert validate(uav) == [], f"警用形态应通过: {validate(uav)}"
    face = _stages(
        ("frame_manager", {"frame_skip": 5}),
        ("yolo", {"model_path": "models/yolo11n.pt", "classes": "0"}),
        ("det_filter", {"filter": "long >= 30"}),
        ("target_crop", {"classes": "0"}),
        ("face_detect", {"det_thresh": 0.4, "det_size": "640,640"}))
    assert validate(face) == [], f"人脸截取形态应通过: {validate(face)}"


def test_experiment_round3_bad_dsl():
    # 实验③同型：det_filter.filter 硬语法错（缺值）必须被拦
    errs = validate(_stages(*_with(2, "det_filter", {"filter": "long >="})))
    assert any("det_filter" in e for e in errs), errs


def test_experiment_round3_fake_model():
    # 实验③：yolo.model_path="yolov5s.pt"（编造模型路径）曾被放过
    errs = validate(_stages(*_with(1, "yolo", {"model_path": "yolov5s.pt"})))
    assert any("不存在" in e for e in errs), errs


def test_experiment_round2_bad_events():
    # 实验②：det_filter.events="unique(plate_id)"（非 JSON 列表）曾被放过
    errs = validate(_stages(*_with(2, "det_filter", {"events": "unique(plate_id)"})))
    assert any("装配" in e for e in errs), errs


def test_experiment_round2_rule_without_window():
    # 实验②类：seq_filter 规则 min_n>1 却无窗口（SeqRule 装配期拒绝）
    errs = validate(_stages(*_with(5, "seq_filter", {
        "rules": '[{"match": "kind == plate", "event": "x", "min_n": 3}]'})))
    assert any("装配" in e for e in errs), errs


def test_crop_filter_dsl():
    # 真·非法语法（缺值/缺括号）；注：字段名编造与方向写反属语义幻觉，静态校验不拦
    for bad in [">= 40", "long >=", "(long >= 40"]:
        errs = validate(_stages(*_with(3, "target_crop", {"filter": bad})))
        assert any("无法解析" in e or "值为空" in e or "右括号" in e for e in errs), (bad, errs)
    assert validate(_stages(*_with(3, "target_crop", {"filter": "long >= 40"}))) == []


def test_numeric_bounds():
    errs = validate(_stages(*_with(0, "frame_manager", {"frame_skip": 0.5})))
    assert any("frame_skip" in e for e in errs), errs
    errs = validate(_stages(*_with(1, "yolo", {"conf": 1.5})))
    assert any("conf" in e for e in errs), errs


def test_seq_filter_all_empty_ok():
    # 门控/规则全空：与运行时一致（静默跳过），不算 error
    assert validate(_stages(*_with(5, "seq_filter", {"throttle": "", "change": "", "rules": ""}))) == []


def test_experiment_round3_dsl_not_static_checkable():
    # 实验③病例 "w<h40"：DSL 语法合法（h40=字符串值），语义错误静态校验不拦——记录局限
    assert validate(_stages(*_with(2, "det_filter", {"filter": "w<h40"}))) == []


def test_test_only_key_rejected():
    # 实验①病例：bbox 是 target_crop 测试页专用参数，不得进链路 spec
    errs = validate(_stages(*_with(3, "target_crop", {"bbox": "10,10,200,200"})))
    assert any("test_only" in e for e in errs), errs


def _with(idx, mid, patch):
    stages = list(BASE_PLATE)
    params = dict(stages[idx][1])
    params.update(patch)
    stages[idx] = (mid, params)
    return stages


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)
