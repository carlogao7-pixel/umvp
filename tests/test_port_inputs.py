# -*- coding: utf-8 -*-
"""
端口连线（P5 第 2 步）：装配透传 + 端到端生效 + CLI 退出码 断言测试

背景（WO-0002.3，2026-10）：pipe-chain 段落地「产物端口协议」——Pipeline.artifacts
（每帧产物表）+ stage_inputs（{module_id:{端口:上游产物名}}）+ 9 模块 INPUTS/OUTPUTS
声明 + step() 按声明解析 `_port(...)`；web 段 `_chain_from_spec` 把 spec 各 stage 的
`inputs` 透传成 stage_inputs，`validate_chain` 对 inputs 做 DAG 校验。

本测试覆盖（校验规则逐条见 tests/test_validate_chain.py；此处聚焦装配/运行/CLI）：
  1. module_port_decls：9 模块自描述 INPUTS/OUTPUTS 声明正确（设计器/校验单一来源）。
  2. _chain_from_spec 透传：spec 含 inputs → pipe.stage_inputs 形状正确；无 inputs → {}；
     非字符串值（列表）防御性跳过。
  3. 端到端（关键）：进程内 _chain_from_spec + step()，用间谍捕获 annotate.run() 实参，
     验证声明 inputs（annotate.dets = yolo.dets / = det_filter.dets）时模块**确实收到
     指定上游产物**（过滤前 3 条 vs 过滤后 2 条，结果不同）；不声明时与默认一致。
  4. fail fast：经 spec 声明的非法连线在 step() 抛 PortResolutionError（可读文案）。
  5. CLI 退出码：web_lab/validate_chain.py 子进程——非法 spec exit=1 且 stdout 含 [ERR]；
     合法 spec exit=0 且 stdout 含 [OK]。

运行: conda run -n ai python tests/test_port_inputs.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402
from pipe.composer import (  # noqa: E402
    ARTIFACT_ANNOTATED,
    ARTIFACT_CROPS,
    ARTIFACT_DET_FILTER_DETS,
    ARTIFACT_DETS,
    ARTIFACT_FACES,
    ARTIFACT_FRAME,
    ARTIFACT_PLATES,
    ARTIFACT_SEQ_FILTER_DETS,
    ARTIFACT_TYPES,
    Detection,
    PortResolutionError,
)

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


# ==================== 1. module_port_decls：模块自描述端口声明 ====================

def decl_checks() -> None:
    decl = server.module_port_decls()

    check("decl: 9 个产品模块均有端口声明",
          set(decl) == {"frame_manager", "yolo", "vlm", "det_filter", "seq_filter",
                        "target_crop", "annotate", "face_detect", "plate_recog"},
          str(sorted(decl)))
    check("decl: frame_manager 是源头（无输入，出 frame:Image）",
          decl["frame_manager"]["inputs"] == {}
          and decl["frame_manager"]["outputs"] == {"frame": "Image"})
    check("decl: yolo 吃 image:Image 出 dets:Records",
          decl["yolo"]["inputs"] == {"image": "Image"}
          and decl["yolo"]["outputs"] == {"dets": "Records"})
    check("decl: det_filter 吃/出 dets:Records",
          decl["det_filter"]["inputs"] == {"dets": "Records"}
          and decl["det_filter"]["outputs"] == {"dets": "Records"})
    check("decl: seq_filter 吃/出 dets:Records",
          decl["seq_filter"]["inputs"] == {"dets": "Records"}
          and decl["seq_filter"]["outputs"] == {"dets": "Records"})
    check("decl: target_crop 吃 frame:Image+dets:Records 出 crops:Records",
          decl["target_crop"]["inputs"] == {"frame": "Image", "dets": "Records"}
          and decl["target_crop"]["outputs"] == {"crops": "Records"})
    check("decl: annotate 吃 image:Image+dets:Records 出 annotated:Image",
          decl["annotate"]["inputs"] == {"image": "Image", "dets": "Records"}
          and decl["annotate"]["outputs"] == {"annotated": "Image"})
    check("decl: face_detect 吃 frame:Image+crops:Records 出 faces:Records",
          decl["face_detect"]["inputs"] == {"frame": "Image", "crops": "Records"}
          and decl["face_detect"]["outputs"] == {"faces": "Records"})
    check("decl: plate_recog 吃 frame:Image+crops:Records 出 plates:Records",
          decl["plate_recog"]["inputs"] == {"frame": "Image", "crops": "Records"}
          and decl["plate_recog"]["outputs"] == {"plates": "Records"})
    # 本期已在 step() 纳入解析的端口集合（设计器开放连线 / 校验硬校验的依据）
    check("decl: CHAIN_RESOLVED_PORTS 对齐已解析端口",
          server.CHAIN_RESOLVED_PORTS == {"yolo": ("image",), "det_filter": ("dets",),
                                          "seq_filter": ("dets",),
                                          "target_crop": ("frame", "dets"),
                                          "annotate": ("image", "dets")},
          str(server.CHAIN_RESOLVED_PORTS))
    check("decl: 未解析模块集合含 vlm/plate_recog/face_detect/extract_faces",
          set(server.CHAIN_UNRESOLVED_INPUT_MODULES) == {"vlm", "plate_recog",
                                                         "face_detect", "extract_faces"})
    # 产物类型表覆盖 Pipeline 登记的全部产物常量
    check("decl: ARTIFACT_TYPES 覆盖全部登记产物常量",
          {ARTIFACT_FRAME, ARTIFACT_DETS, ARTIFACT_DET_FILTER_DETS, ARTIFACT_SEQ_FILTER_DETS,
           ARTIFACT_CROPS, ARTIFACT_ANNOTATED, ARTIFACT_FACES, ARTIFACT_PLATES}
          <= set(ARTIFACT_TYPES), str(sorted(ARTIFACT_TYPES)))
    check("decl: 产物类型标注正确（Image/Records）",
          ARTIFACT_TYPES[ARTIFACT_FRAME] == "Image"
          and ARTIFACT_TYPES[ARTIFACT_ANNOTATED] == "Image"
          and ARTIFACT_TYPES[ARTIFACT_DETS] == "Records"
          and ARTIFACT_TYPES[ARTIFACT_CROPS] == "Records")


# ==================== 2. _chain_from_spec：stage_inputs 透传 ====================

def _spec(annotate_inputs=None, det_filter_inputs=None) -> dict:
    """fm→yolo→det_filter→annotate→seq_filter；按需给阶段注入 inputs。

    不含 target_crop（其 run 需真实图像），保证纯逻辑测试可传 object() 帧。
    """
    stages = [
        {"module_id": "frame_manager",
         "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "0"}},
        {"module_id": "det_filter", "params": {"filter": "score >= 0.5"}},
        {"module_id": "annotate", "params": {"thickness": 2}},
        {"module_id": "seq_filter", "params": {"throttle": "", "change": "", "rules": ""}},
    ]
    if det_filter_inputs is not None:
        stages[2]["inputs"] = det_filter_inputs
    if annotate_inputs is not None:
        stages[3]["inputs"] = annotate_inputs
    return {"name": "qa-ports", "stages": stages}


def passthrough_checks() -> None:
    # 无 inputs → stage_inputs == {}（缺省=现状，零回归）
    check("装配: 无 inputs → stage_inputs == {}", server._chain_from_spec(_spec()).stage_inputs == {})

    # 单模块单端口
    p = server._chain_from_spec(_spec(annotate_inputs={"dets": "yolo.dets"}))
    check("装配: 单端口 inputs 透传形状正确",
          p.stage_inputs == {"annotate": {"dets": "yolo.dets"}}, str(p.stage_inputs))

    # 多模块多端口
    p = server._chain_from_spec(_spec(
        annotate_inputs={"dets": "yolo.dets", "image": "frame_manager.frame"},
        det_filter_inputs={"dets": "yolo.dets"}))
    check("装配: 多模块多端口均透传",
          p.stage_inputs == {"annotate": {"dets": "yolo.dets", "image": "frame_manager.frame"},
                             "det_filter": {"dets": "yolo.dets"}},
          str(p.stage_inputs))

    # 非字符串值（列表）防御性跳过：不抛错，且不进入 stage_inputs
    p = server._chain_from_spec(_spec(annotate_inputs={"dets": ["yolo.dets"]}))
    check("装配: 非字符串值（列表）被防御性跳过 → 不改 stage_inputs",
          p.stage_inputs == {}, str(p.stage_inputs))

    # 空白字符串值同样跳过
    p = server._chain_from_spec(_spec(annotate_inputs={"dets": "   "}))
    check("装配: 空白产物名被跳过", p.stage_inputs == {}, str(p.stage_inputs))

    # inputs 非对象（字符串）→ 不误装配
    spec = _spec()
    spec["stages"][3]["inputs"] = "yolo.dets"
    check("装配: inputs 非对象 → 视为未声明", server._chain_from_spec(spec).stage_inputs == {})


# ==================== 3. 端到端：间谍捕获模块 run() 实参 ====================

class _AnnoSpy:
    """annotate 间谍：记录 run(image, dets, ts) 实参，INPUTS 声明与真模块一致。"""

    INPUTS = {"image": "Image", "dets": "Records"}
    OUTPUTS = {"annotated": "Image"}

    def __init__(self) -> None:
        self.calls: list = []

    def run(self, image, dets, ts=0.0):
        self.calls.append({"image": image, "dets": list(dets or []), "ts": ts})
        return ("ANNOTATED", image)


_RAW = [
    Detection(track_id=1, cls_id=0, bbox=(0, 0, 10, 10), score=0.9),
    Detection(track_id=2, cls_id=0, bbox=(0, 0, 10, 10), score=0.2),  # 低于 0.5 会被筛掉
    Detection(track_id=3, cls_id=0, bbox=(0, 0, 10, 10), score=0.7),
]


def _run_spy(ann_inputs=None):
    pipe = server._chain_from_spec(_spec(annotate_inputs=ann_inputs))
    spy = _AnnoSpy()
    pipe.annotate = spy  # 替换真 Annotator，捕获实参
    frame = object()
    dets = [Detection(track_id=d.track_id, cls_id=d.cls_id, bbox=d.bbox, score=d.score)
            for d in _RAW]
    pipe.step(0, 0.0, dets=dets, frame=frame, force=True)
    assert len(spy.calls) == 1, f"annotate 应被调用一次，实为 {len(spy.calls)}"
    call = spy.calls[0]
    return pipe, call, frame


def e2e_checks() -> None:
    # 默认（无 inputs）：annotate 收到 det_filter 过滤后的 dets（0.9/0.7 两条）
    p_def, c_def, frame = _run_spy(None)
    scores_def = sorted(d.score for d in c_def["dets"])
    check("e2e 默认: stage_inputs 为空（零回归）", p_def.stage_inputs == {})
    check("e2e 默认: annotate 收到过滤后 dets（2 条，含 0.9/0.7）",
          scores_def == [0.7, 0.9], str(scores_def))
    check("e2e 默认: image 端口仍取本帧原图", c_def["image"] is frame)

    # 声明 annotate.dets = yolo.dets（过滤前原始 3 条，绕过检测筛选）
    p_raw, c_raw, _ = _run_spy({"dets": "yolo.dets"})
    scores_raw = sorted(d.score for d in c_raw["dets"])
    check("e2e yolo.dets: 声明透传为 stage_inputs",
          p_raw.stage_inputs == {"annotate": {"dets": "yolo.dets"}})
    check("e2e yolo.dets: annotate 收到 yolo.dets 原始 3 条（含被筛掉的 0.2）",
          scores_raw == [0.2, 0.7, 0.9], str(scores_raw))
    check("e2e: yolo.dets 与默认（det_filter.dets）结果确实不同",
          scores_raw != scores_def, f"{scores_raw} vs {scores_def}")

    # 声明 annotate.dets = det_filter.dets（显式取过滤后）→ 与默认一致
    p_fil, c_fil, _ = _run_spy({"dets": "det_filter.dets"})
    scores_fil = sorted(d.score for d in c_fil["dets"])
    check("e2e det_filter.dets: annotate 收到过滤后 2 条",
          scores_fil == [0.7, 0.9], str(scores_fil))
    check("e2e: 显式 det_filter.dets 与默认结果一致",
          scores_fil == scores_def, f"{scores_fil} vs {scores_def}")
    check("e2e: 显式 det_filter.dets 仍与 yolo.dets 不同",
          scores_fil != scores_raw)

    # 同时声明 image 端口指向上游产物（Image 类型）→ 间谍收到该产物
    p_img, c_img, frame2 = _run_spy({"dets": "yolo.dets", "image": "frame_manager.frame"})
    check("e2e: image 端口声明指向 frame_manager.frame 时间谍确实收到本帧原图",
          c_img["image"] is frame2)


# ==================== 4. fail fast：step() 抛 PortResolutionError ====================

def _step_error(pipe) -> str:
    try:
        pipe.step(0, 0.0, dets=list(_RAW), frame=object(), force=True)
        return ""
    except PortResolutionError as e:
        return str(e)


def failfast_checks() -> None:
    # 未知产物（spec 声明经装配透传后在运行期 fail fast）
    e = _step_error(server._chain_from_spec(_spec(annotate_inputs={"dets": "nope.dets"})))
    check("fail-fast: 未知产物 → 报错含阶段/端口/产物名/可用清单",
          "annotate" in e and "'dets'" in e and "nope.dets" in e and "yolo.dets" in e, e)
    # 类型不符：annotate.image（Image）喂 det_filter.dets（Records）
    e = _step_error(server._chain_from_spec(
        _spec(annotate_inputs={"image": "det_filter.dets"})))
    check("fail-fast: 类型不符 → 报错含类型与产物名",
          "类型不符" in e and "Records" in e and "Image" in e, e)
    # 未在本帧产物表的产物：seq_filter.dets（链路声明了 seq_filter，但将其当 annotate 输入时未登记）
    e = _step_error(server._chain_from_spec(
        _spec(annotate_inputs={"dets": "seq_filter.dets"})))
    check("fail-fast: 引用下游/未登记产物 → 报错",
          "不在本帧产物表" in e and "seq_filter.dets" in e, e)


# ==================== 5. CLI 退出码（validate_chain.py 子进程） ====================

def _cli(spec: dict):
    """子进程调用 web_lab/validate_chain.py 校验 spec，返回 (returncode, stdout+stderr)。"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8",
                                     delete=False) as f:
        json.dump(spec, f, ensure_ascii=False)
        path = f.name
    try:
        r = subprocess.run([sys.executable, str(ROOT / "web_lab" / "validate_chain.py"), path],
                           capture_output=True, text=True, timeout=180)
        return r.returncode, (r.stdout + r.stderr)
    finally:
        Path(path).unlink(missing_ok=True)


def cli_checks() -> None:
    ok_spec = _spec(annotate_inputs={"dets": "yolo.dets"})
    code, out = _cli(ok_spec)
    check("CLI: 合法 spec（含 inputs）→ exit=0", code == 0, f"exit={code}")
    check("CLI: 合法 spec stdout 含 [OK]", "[OK]" in out, out.strip()[-80:])
    check("CLI: 合法 spec 无 [ERR]", "[ERR]" not in out)

    bad_spec = _spec(annotate_inputs={"dets": "nope.dets"})
    code, out = _cli(bad_spec)
    check("CLI: 非法 spec（未知产物）→ exit=1", code == 1, f"exit={code}")
    check("CLI: 非法 spec stdout 含 [ERR]", "[ERR]" in out, out.strip()[-80:])
    check("CLI: 非法 spec [ERR] 文案含产物名", "nope.dets" in out)
    check("CLI: 非法 spec 不出 [OK]", "[OK]" not in out)


def main() -> None:
    print("== 1. module_port_decls：模块自描述端口声明 ==")
    decl_checks()
    print("== 2. _chain_from_spec：stage_inputs 透传 ==")
    passthrough_checks()
    print("== 3. 端到端：间谍捕获 annotate.run() 实参 ==")
    e2e_checks()
    print("== 4. fail fast：PortResolutionError ==")
    failfast_checks()
    print("== 5. CLI 退出码 ==")
    cli_checks()
    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
