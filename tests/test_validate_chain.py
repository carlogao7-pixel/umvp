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


def test_seed_face_forms_assembled_ok():
    # 人脸链新形态（face_detect）/ 嵌入形态（face_embed）均在可装配集合 → 不触发「声明即执行」
    face = _stages(
        ("frame_manager", {"frame_skip": 5}),
        ("yolo", {"model_path": "models/yolov8n.pt", "classes": "0"}),
        ("det_filter", {"filter": "long >= 30"}),
        ("target_crop", {"classes": "0"}),
        ("face_detect", {"det_thresh": 0.4, "det_size": "640,640"}))
    assert validate(face) == [], f"face_detect 形态应通过: {validate(face)}"
    embed = _stages(
        ("frame_manager", {"frame_skip": 5}),
        ("yolo", {"model_path": "models/yolov8n.pt", "classes": "0"}),
        ("det_filter", {"filter": ""}),
        ("target_crop", {"classes": "0"}),
        ("face_detect", {"det_thresh": 0.4}),
        ("face_embed", {}))
    assert validate(embed) == [], f"face_embed 形态应通过: {validate(embed)}"


def test_declared_but_not_assembled_rejected():
    # WO-0001 事故回归：face_store 已注册产品模块，但 _chain_from_spec 无对应装配——
    # 「声明了也静默不执行」，校验器必须拦（旧版曾放过，正是人脸链 det_filter 失效同类）。
    stages = list(BASE_PLATE[:-1]) + [("face_store", {})] + [BASE_PLATE[-1]]
    errs = validate(_stages(*stages))
    assert any("声明即执行" in e and "face_store" in e for e in errs), errs
    # 未装配阶段被拦后，其余内容合法（其它校验项不误报同类错）
    assert not any("未知/非产品模块" in e for e in errs), errs


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


# ---------------- image_source（vlm 送审图像源，WO-0002.2） ----------------

def _uav_vlm(vlm_params: dict, annotate: bool = True, target_crop: bool = False):
    """警用形态（含 vlm），可开关 annotate / target_crop 阶段；seq_filter 收尾。
    vlm_params 直接透传（如 {"image_source": "crops"}）。"""
    stages = [
        ("frame_manager", {"sampling": "analysis", "frame_skip": 15}),
        ("yolo", {"model_path": "yolov8n.pt", "classes": "0,2"}),
        ("det_filter", {"group_limits": "0:2,300;2:1,300"}),
    ]
    if target_crop:
        stages.append(("target_crop", {"classes": "0,2"}))
    if annotate:
        stages.append(("annotate", {"thickness": 2, "show_label": True}))
    stages.append(("vlm", vlm_params))
    stages.append(("seq_filter", {"throttle": "", "change": "", "rules": ""}))
    return _stages(*stages)


def _warns(spec):
    w: list = []
    errs = validate(spec, warnings=w)
    return errs, w


def test_image_source_valid_no_warn():
    # annotated 上游有 annotate / crops 上游有 target_crop → 无 error 无 warning
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "annotated"}))
    assert errs == [] and w == [], (errs, w)
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "crops"},
                              target_crop=True))
    assert errs == [] and w == [], (errs, w)
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "frame"}))
    assert errs == [] and w == [], (errs, w)


def test_image_source_invalid_value_rejected():
    # 非法取值（如 lowres，尚未实现）必须报错，不能静默回退
    errs, _ = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "lowres"}))
    assert any("不在允许值" in e and "image_source" in e for e in errs), errs


def test_image_source_annotated_without_annotate_warns_not_fatal():
    # 显式 annotated 但上游无 annotate → 非致命：error 空、warnings 有提示
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "annotated"},
                              annotate=False))
    assert errs == [], errs
    assert any("annotate" in x for x in w), w


def test_image_source_crops_without_target_crop_warns_not_fatal():
    # 显式 crops 但上游无 target_crop → 非致命警告（运行时会回退）
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": "crops"},
                              target_crop=False))
    assert errs == [], errs
    assert any("target_crop" in x for x in w), w


def test_image_source_absent_old_spec_no_warn():
    # 旧 spec（未声明 image_source）：缺省=现状，不报错不警告（零回归）
    errs, w = _warns(_uav_vlm({"use_real_vlm": False}, annotate=False))
    assert errs == [] and w == [], (errs, w)


def test_image_source_none_value_no_error():
    # DB 空列还原/未填 = None → 回落模块默认，不算非法取值，也不触发一致性警告
    errs, w = _warns(_uav_vlm({"use_real_vlm": False, "image_source": None}, annotate=False))
    assert errs == [] and w == [], (errs, w)


def test_validate_legacy_signature_unchanged():
    # 旧调用方（validate(spec) 不传 warnings）签名兼容：仍有 error、不因 warning 抛错
    spec = _uav_vlm({"use_real_vlm": False, "image_source": "crops"}, target_crop=False)
    assert validate(spec) == [], validate(spec)


# ---------------- 端口连线 inputs（P5 第 2 步 DAG 校验） ----------------
# 模块 INPUTS/OUTPUTS 声明 + 产物类型表见 pipe.composer.ARTIFACT_TYPES / web_lab.server。
# 规则逐条：①未知产物 ②非法端口 ③非上游/自引用 ④类型不符 ⑤值非字符串（列表）
# ⑥同一产物多下游→允许 ⑦vlm.inputs 仅 dets（image→提示 image_source）
# ⑧旧 spec（无 inputs）→ 无错无警告。

_PORT_PARAMS = {
    "frame_manager": {"sampling": "analysis", "frame_skip": 1},
    "yolo": {"model_path": "models/yolov8n.pt", "classes": "0"},
    "det_filter": {"filter": ""},
    "target_crop": {"classes": "0"},
    "annotate": {"thickness": 2},
    "vlm": {"use_real_vlm": False},
    "plate_recog": {"mode": "crop"},
    "face_detect": {"det_thresh": 0.4},
    "seq_filter": {"throttle": "", "change": "", "rules": ""},
}


def _port_chain(inputs_by_mid: dict, mods: list):
    """按给定模块顺序构造链路；inputs_by_mid={module_id: {"端口": "上游产物"}} 注入 inputs。"""
    stages = []
    for m in mods:
        st = {"module_id": m, "params": dict(_PORT_PARAMS.get(m, {}))}
        if m in inputs_by_mid:
            st["inputs"] = inputs_by_mid[m]
        stages.append(st)
    return {"name": "t", "stages": stages}


_PLAIN_MODS = ["frame_manager", "yolo", "det_filter", "annotate", "seq_filter"]


def test_inputs_valid_no_warn():
    # 合法连线：annotate.dets ← yolo.dets（端口声明 Records、产物 Records、yolo 在上游）
    errs, w = _warns(_port_chain({"annotate": {"dets": "yolo.dets"}}, _PLAIN_MODS))
    assert errs == [] and w == [], (errs, w)


def test_inputs_unknown_artifact_rejected():
    # ① 未知产物名 → error，文案含产物名与可用产物清单
    errs, w = _warns(_port_chain({"annotate": {"dets": "nope.dets"}}, _PLAIN_MODS))
    assert any("未知产物" in e and "nope.dets" in e for e in errs), errs
    assert any("yolo.dets" in e for e in errs), errs  # 可用产物清单可读
    assert w == [], w


def test_inputs_unknown_port_rejected():
    # ② 端口未在模块 INPUTS 声明 → error，文案含端口名与该模块已声明端口
    errs, w = _warns(_port_chain({"annotate": {"foo": "yolo.dets"}}, _PLAIN_MODS))
    assert any("端口 'foo'" in e and "INPUTS" in e for e in errs), errs
    assert any("'dets'" in e and "'image'" in e for e in errs), errs  # 已声明端口清单
    assert w == [], w


def test_inputs_non_upstream_rejected():
    # ③a 引用下游产物：det_filter 引用 seq_filter.dets（类型同名 Records，仅触发"非上游"）
    errs, w = _warns(_port_chain({"det_filter": {"dets": "seq_filter.dets"}}, _PLAIN_MODS))
    assert any("不是上游产物" in e and "seq_filter" in e for e in errs), errs
    assert not any("类型不符" in e for e in errs), errs  # 不误报同类


def test_inputs_self_reference_rejected():
    # ③b 自引用：det_filter.dets ← det_filter.dets（producer 不在自身之前）
    errs, w = _warns(_port_chain({"det_filter": {"dets": "det_filter.dets"}}, _PLAIN_MODS))
    assert any("不是上游产物" in e for e in errs), errs


def test_inputs_type_mismatch_rejected():
    # ④ 类型不符：annotate.image（Image）喂 det_filter.dets（Records）
    errs, w = _warns(_port_chain({"annotate": {"image": "det_filter.dets"}}, _PLAIN_MODS))
    assert any("类型不符" in e and "Records" in e and "Image" in e for e in errs), errs


def test_inputs_multivalue_rejected():
    # ⑤ 值非字符串（列表）→ 每端口单一上游，报错并回显类型
    errs, w = _warns(_port_chain({"annotate": {"dets": ["yolo.dets"]}}, _PLAIN_MODS))
    assert any("值须为字符串" in e and "list" in e for e in errs), errs


def test_inputs_same_artifact_multi_downstream_ok():
    # ⑥ 同一产物被多个下游引用（分支留路）→ 允许，零 error 零 warning
    mods = ["frame_manager", "yolo", "det_filter", "target_crop", "annotate", "seq_filter"]
    errs, w = _warns(_port_chain(
        {"target_crop": {"dets": "yolo.dets"}, "annotate": {"dets": "yolo.dets"}}, mods))
    assert errs == [] and w == [], (errs, w)


def test_inputs_newly_resolved_modules_ok_and_checked():
    # plate_recog/face_detect 本期已纳入解析：合法连线零 error 零 warning；
    # 非法连线走既有 DAG 规则报错（不再只给警告）。
    mods_plate = ["frame_manager", "yolo", "det_filter", "target_crop", "plate_recog"]
    errs, w = _warns(_port_chain({"plate_recog": {"crops": "target_crop.crops"}}, mods_plate))
    assert errs == [] and w == [], (errs, w)
    errs, w = _warns(_port_chain({"plate_recog": {"frame": "frame_manager.frame"}}, mods_plate))
    assert errs == [] and w == [], (errs, w)
    # 类型不符（crops:Records 喂 frame:Image）
    errs, _ = _warns(_port_chain({"plate_recog": {"frame": "target_crop.crops"}}, mods_plate))
    assert any("类型不符" in e for e in errs), errs
    # 未知产物
    errs, _ = _warns(_port_chain({"plate_recog": {"crops": "nope.crops"}}, mods_plate))
    assert any("未知产物" in e for e in errs), errs

    mods_face = ["frame_manager", "yolo", "det_filter", "target_crop", "face_detect"]
    errs, w = _warns(_port_chain({"face_detect": {"crops": "target_crop.crops"}}, mods_face))
    assert errs == [] and w == [], (errs, w)
    errs, _ = _warns(_port_chain({"face_detect": {"frame": "target_crop.crops"}}, mods_face))
    assert any("类型不符" in e for e in errs), errs


def test_inputs_vlm_dets_ok():
    # vlm.dets 已开放（驱动层据 artifacts 构造送审）：det_filter.dets / yolo.dets 均合法。
    mods = ["frame_manager", "yolo", "det_filter", "vlm", "seq_filter"]
    errs, w = _warns(_port_chain({"vlm": {"dets": "det_filter.dets"}}, mods))
    assert errs == [] and w == [], (errs, w)
    errs, w = _warns(_port_chain({"vlm": {"dets": "yolo.dets"}}, mods))
    assert errs == [] and w == [], (errs, w)
    # 非上游（det_filter 在 vlm 之前，yolo 也在前；改用下游 seq_filter.dets 应报"非上游"）
    errs, _ = _warns(_port_chain({"vlm": {"dets": "seq_filter.dets"}}, mods))
    assert any("不是上游产物" in e for e in errs), errs


def test_inputs_vlm_image_rejected_hint_image_source():
    # vlm.image 不开放（图片源统一由 image_source 管）→ error 并提示改用 image_source
    mods = ["frame_manager", "yolo", "det_filter", "annotate", "vlm", "seq_filter"]
    errs, w = _warns(_port_chain({"vlm": {"image": "annotate.annotated"}}, mods))
    assert any("只允许端口 'dets'" in e and "image_source" in e for e in errs), errs
    assert w == [], w


def test_inputs_vlm_other_port_rejected():
    # vlm 其余端口（如 frame）同样报错（无 image_source 提示）
    mods = ["frame_manager", "yolo", "det_filter", "vlm", "seq_filter"]
    errs, w = _warns(_port_chain({"vlm": {"frame": "frame_manager.frame"}}, mods))
    assert any("只允许端口 'dets'" in e and "'frame'" in e for e in errs), errs
    assert w == [], w


def test_inputs_unresolved_list_empty():
    # 未纳入解析名单本期为空（plate_recog/face_detect/vlm.dets 均已解析）
    import server
    assert server.CHAIN_UNRESOLVED_INPUT_MODULES == (), server.CHAIN_UNRESOLVED_INPUT_MODULES


def test_inputs_old_spec_no_error_no_warn():
    # ⑧ 旧 spec（未声明 inputs）→ 零 error 零 warning（缺省=现状，零回归）
    for mods in (_PLAIN_MODS,
                 ["frame_manager", "yolo", "det_filter", "target_crop", "plate_recog"],
                 ["frame_manager", "yolo", "det_filter", "target_crop", "face_detect"]):
        errs, w = _warns(_port_chain({}, mods))
        assert errs == [] and w == [], (mods, errs, w)


# ---------------- annotate 双模式（WO-0021：scope + record 连线合法性） ----------------

def _rec_chain(scope_at_3="record", inputs_at_4=None, extra_params=None):
    """fm→yolo→det_filter→annotate#3→annotate#4→seq_filter；可调 #3 scope 与 #4 连线。"""
    mods = ["frame_manager", "yolo", "det_filter", "annotate", "annotate", "seq_filter"]
    spec = _port_chain({}, mods)
    p3 = {"thickness": 2, "scope": scope_at_3}
    if extra_params:
        p3.update(extra_params)
    spec["stages"][3]["params"] = p3
    if inputs_at_4 is not None:
        spec["stages"][4]["inputs"] = inputs_at_4
    return spec


def test_annotate_scope_record_valid():
    # record 实例的带图流被下游实例连线（全名）→ 合法（零 error 零 warning）
    spec = _rec_chain("record", {"dets": "annotate#3.annotated_records"})
    errs, w = _warns(spec)
    assert errs == [] and w == [], (errs, w)


def test_annotate_scope_short_name_ambiguous_still_rejected():
    # 多实例短名歧义照常被 P2-0 规则⑥拦（"不猜"优先于 scope 匹配）
    spec = _rec_chain("record", {"dets": "annotate.annotated_records"})
    errs, w = _warns(spec)
    assert any("歧义" in e and "全名" in e for e in errs), errs
    assert w == [], w


def test_annotate_scope_record_frame_artifact_rejected():
    # record 实例不产帧级标注图：连 annotate#3.annotated → 错（提示用 annotated_records）
    spec = _rec_chain("record", {"dets": "annotate#3.annotated"})
    errs, w = _warns(spec)
    assert any("record" in e and "annotated_records" in e for e in errs), errs
    assert w == [], w


def test_annotate_scope_default_records_artifact_rejected():
    # 缺省 frame 实例不产带图流：连 annotate.annotated_records → 错（提示开 record 或连 annotated）
    spec = _rec_chain("frame", {"dets": "annotate#3.annotated_records"})
    errs, w = _warns(spec)
    assert any("annotated_records" in e and "record" in e for e in errs), errs
    # 显式 scope=frame 同样拦
    spec = _rec_chain("frame", {"dets": "annotate.annotated_records"})
    errs, _ = _warns(spec)
    assert any("annotated_records" in e for e in errs), errs


def test_annotate_scope_invalid_value_rejected():
    # scope 非法取值（经参数表 options 校验）
    errs, _ = _warns(_rec_chain("banana"))
    assert any("scope" in e and "不在允许值" in e for e in errs), errs


def test_annotate_max_records_nonpositive_rejected():
    # max_records 必须 ≥ 1
    errs, _ = _warns(_rec_chain("record", extra_params={"max_records": 0}))
    assert any("max_records" in e for e in errs), errs
    errs, _ = _warns(_rec_chain("record", extra_params={"max_records": 8}))
    assert errs == [], errs


def test_annotate_scope_omitted_old_spec_ok():
    # 旧 spec（无 scope / 无 inputs）：零 error 零 warning（零回归）
    errs, w = _warns(_rec_chain(None, None))
    assert errs == [] and w == [], (errs, w)


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
