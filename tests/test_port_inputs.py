# -*- coding: utf-8 -*-
"""
端口连线（P5 第 2 步 + P2-0 位置化）：装配透传 + 端到端生效 + CLI 退出码 断言测试

背景（WO-0002.3，2026-10）：pipe-chain 段落地「产物端口协议」——Pipeline.artifacts
（每帧产物表）+ stage_inputs + 9 模块 INPUTS/OUTPUTS 声明 + step() 按声明解析
`_port(...)`；web 段 `_chain_from_spec` 把 spec 各 stage 的 `inputs` 组装成
stage_inputs。**P2-0 节点实例身份（WO-0008，doc/design/graph_chain.md §3.1/§3.4）**：
stage_inputs 键 = 位置串 `"#<stages 数组下标>"`，值经 composer.resolve_artifact 归一为
产物全名 `{module}#{position}.{artifact}`（短名唯一实例兼容、多实例歧义报错）；
validate_chain 对 inputs 做按位置的 DAG 校验。

本测试覆盖（校验规则逐条见 tests/test_validate_chain.py；此处聚焦装配/运行/CLI）：
  1. module_port_decls：9 模块自描述 INPUTS/OUTPUTS 声明正确（设计器/校验单一来源）。
  2. _chain_from_spec 组装：spec 含 inputs → pipe.stage_inputs 为位置键 + 产物全名；
     无 inputs → {}；非字符串值（列表）防御性跳过；全名原样保留；多实例短名装配期抛错。
  3. 端到端（关键）：进程内 _chain_from_spec + step()，用间谍捕获 annotate.run() 实参，
     验证声明 inputs（annotate.dets = yolo.dets / = det_filter.dets）时模块**确实收到
     指定上游产物**（过滤前 3 条 vs 过滤后 2 条，结果不同）；不声明时与默认一致。
  4. fail fast：未知模块产物在**装配期**抛 PortResolutionError；类型不符 / 引用下游
     在 step() 抛（可读文案）。
  5. CLI 退出码：web_lab/validate_chain.py 子进程——非法 spec exit=1 且 stdout 含 [ERR]；
     合法 spec exit=0 且 stdout 含 [OK]。
  6. vlm.inputs.dets：报送检测集由产物决定（驱动层 _apply_vlm_inputs 按位置读连线）。
  7. 同模块多实例（P2-0 验收，graph_chain §7.1/§7.2）：两个 det_filter 各自 filter
     独立生效（变更前第二实例被静默丢弃）、各带 inputs.dets 时 stage_inputs 互不覆盖。

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
    # 本期已纳入解析的端口集合（设计器开放连线 / 校验硬校验的依据）——**单一来源**：
    # 推导自 pipe.composer.RESOLVED_PORTS（step 解析）+ 驱动层 vlm.dets，不再硬编码。
    from pipe.composer import RESOLVED_PORTS as _COMPOSER_PORTS
    check("decl: CHAIN_RESOLVED_PORTS 推导自 composer.RESOLVED_PORTS + 驱动层 vlm.dets",
          server.CHAIN_RESOLVED_PORTS == {**{m: tuple(p) for m, p in _COMPOSER_PORTS.items()},
                                          "vlm": ("dets",)},
          str(server.CHAIN_RESOLVED_PORTS))
    check("decl: plate_recog/face_detect 已纳入解析端口",
          server.CHAIN_RESOLVED_PORTS.get("plate_recog") == ("frame", "crops")
          and server.CHAIN_RESOLVED_PORTS.get("face_detect") == ("frame", "crops"),
          str(server.CHAIN_RESOLVED_PORTS))
    check("decl: vlm 仅开放 dets（image 走 image_source）",
          server.CHAIN_RESOLVED_PORTS.get("vlm") == ("dets",))
    check("decl: 未纳入解析名单已清空（全部已解析）",
          server.CHAIN_UNRESOLVED_INPUT_MODULES == (),
          str(server.CHAIN_UNRESOLVED_INPUT_MODULES))
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

    # 单模块单端口（P2-0：键 = 位置串 "#<position>"，值经 resolve_artifact 归一为全名）
    p = server._chain_from_spec(_spec(annotate_inputs={"dets": "yolo.dets"}))
    check("装配: 单端口 inputs → 位置键 + 产物全名",
          p.stage_inputs == {"#3": {"dets": "yolo#1.dets"}}, str(p.stage_inputs))

    # 全名输入原样保留（装配归一不改动已是全名的产物名）
    p = server._chain_from_spec(_spec(annotate_inputs={"dets": "det_filter#2.dets"}))
    check("装配: 全名产物原样保留",
          p.stage_inputs == {"#3": {"dets": "det_filter#2.dets"}}, str(p.stage_inputs))

    # 多模块多端口（各阶段各自位置键；fm=0 yolo=1 det_filter=2 annotate=3）
    p = server._chain_from_spec(_spec(
        annotate_inputs={"dets": "yolo.dets", "image": "frame_manager.frame"},
        det_filter_inputs={"dets": "yolo.dets"}))
    check("装配: 多模块多端口均透传（各自位置键 + 全名）",
          p.stage_inputs == {"#3": {"dets": "yolo#1.dets", "image": "frame_manager#0.frame"},
                             "#2": {"dets": "yolo#1.dets"}},
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

    # 多实例下用短名（P2-0：不猜）：被引用模块多实例时装配归一即抛 PortResolutionError
    amb = _spec(annotate_inputs={"dets": "det_filter.dets"})
    amb["stages"].append({"module_id": "det_filter", "params": {"filter": "score >= 0.9"}})
    try:
        server._chain_from_spec(amb)
        check("装配: 多实例短名 → 装配期抛 PortResolutionError", False, "未抛错")
    except PortResolutionError as e:
        check("装配: 多实例短名 → 装配期抛 PortResolutionError",
              "det_filter" in str(e) and ("歧义" in str(e) or "全名" in str(e)), str(e))


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
    scores_def = sorted(d["score"] for d in c_def["dets"])
    check("e2e 默认: stage_inputs 为空（零回归）", p_def.stage_inputs == {})
    check("e2e 默认: annotate 收到过滤后 dets（2 条，含 0.9/0.7）",
          scores_def == [0.7, 0.9], str(scores_def))
    check("e2e 默认: image 端口仍取本帧原图", c_def["image"] is frame)

    # 声明 annotate.dets = yolo.dets（过滤前原始 3 条，绕过检测筛选）
    p_raw, c_raw, _ = _run_spy({"dets": "yolo.dets"})
    scores_raw = sorted(d["score"] for d in c_raw["dets"])
    check("e2e yolo.dets: 声明组装为位置键 + 全名",
          p_raw.stage_inputs == {"#3": {"dets": "yolo#1.dets"}})
    check("e2e yolo.dets: annotate 收到 yolo.dets 原始 3 条（含被筛掉的 0.2）",
          scores_raw == [0.2, 0.7, 0.9], str(scores_raw))
    check("e2e: yolo.dets 与默认（det_filter.dets）结果确实不同",
          scores_raw != scores_def, f"{scores_raw} vs {scores_def}")

    # 声明 annotate.dets = det_filter.dets（显式取过滤后）→ 与默认一致
    p_fil, c_fil, _ = _run_spy({"dets": "det_filter.dets"})
    scores_fil = sorted(d["score"] for d in c_fil["dets"])
    check("e2e det_filter.dets: annotate 收到过滤后 2 条",
          scores_fil == [0.7, 0.9], str(scores_fil))
    check("e2e: 显式 det_filter.dets 与默认结果一致",
          scores_fil == scores_def, f"{scores_fil} vs {scores_def}")
    check("e2e: 显式 det_filter.dets 仍与 yolo.dets 不同",
          scores_fil != scores_raw)

    # 形态断言（WO-0013 定案清单第 5 条）：声明 inputs 连线后，spy 收到的 dets 元素
    # 仍为 dict 记录（kind=det，且带 step 注入的时间上下文）——产物形态不因
    # 中间环节（直传入口归一 / 产物解析 / 筛选传递）回退成对象。
    _DET_KEYS = {"kind", "track_id", "cls_id", "bbox", "score", "ts", "frame_idx"}
    check("e2e 形态: yolo.dets 连线后 spy 收到的元素仍为 dict 记录（kind=det + 全契约字段）",
          all(isinstance(d, dict) and d.get("kind") == "det" and _DET_KEYS <= set(d)
              for d in c_raw["dets"]),
          f"{type(c_raw['dets'][0]).__name__ if c_raw['dets'] else '空'}")
    check("e2e 形态: det_filter.dets 连线后元素仍为 dict 记录（kind=det）",
          all(isinstance(d, dict) and d.get("kind") == "det" for d in c_fil["dets"]))
    check("e2e 形态: 默认（检测流末端）元素同样为 dict 记录（零回归）",
          all(isinstance(d, dict) and d.get("kind") == "det" for d in c_def["dets"]))

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
    # 未知产物（P2-0：装配归一即 fail fast——resolve_artifact 对未知模块报错，不进 step）
    try:
        server._chain_from_spec(_spec(annotate_inputs={"dets": "nope.dets"}))
        check("fail-fast: 未知产物 → 装配期报错含未知模块名", False, "未抛错")
    except PortResolutionError as e:
        check("fail-fast: 未知产物 → 装配期报错含未知模块名",
              "nope" in str(e) and "未知模块" in str(e), str(e))
    # 类型不符：annotate.image（Image）喂 det_filter.dets（Records）——归一通过，step 期抛
    e = _step_error(server._chain_from_spec(
        _spec(annotate_inputs={"image": "det_filter.dets"})))
    check("fail-fast: 类型不符 → 报错含类型与产物名",
          "类型不符" in e and "Records" in e and "Image" in e, e)
    # 未在本帧产物表的产物：seq_filter.dets（链路声明了 seq_filter，但 annotate 在其前执行）
    # P2-0：报错中的产物名为归一后的全名 seq_filter#4.dets（seq_filter 在 stages[4]）
    e = _step_error(server._chain_from_spec(
        _spec(annotate_inputs={"dets": "seq_filter.dets"})))
    check("fail-fast: 引用下游/未登记产物 → 报错",
          "不在本帧产物表" in e and "seq_filter#4.dets" in e, e)


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

    # vlm.inputs.image 不开放（图片源走 image_source）→ 校验器拒绝（CLI exit=1）
    vlm_spec = {"name": "qa-vlm", "stages": [
        {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "0"}},
        {"module_id": "det_filter", "params": {"filter": ""}},
        {"module_id": "annotate", "params": {"thickness": 2}},
        {"module_id": "vlm", "params": {"use_real_vlm": False},
         "inputs": {"image": "annotate.annotated"}},
        {"module_id": "seq_filter", "params": {"throttle": "", "change": "", "rules": ""}},
    ]}
    code, out = _cli(vlm_spec)
    check("CLI: vlm.inputs.image → exit=1 且提示 image_source",
          code == 1 and "image_source" in out, f"exit={code} out={out.strip()[-100:]}")


# ==================== 6. vlm.inputs.dets：报送检测集由产物决定（驱动层） ====================

def _vlm_dets_spec(art: str | None) -> dict:
    """fm→yolo→det_filter→vlm→seq_filter；按需给 vlm 注入 inputs.dets。"""
    vlm_stage = {"module_id": "vlm", "params": {"use_real_vlm": False}}
    if art is not None:
        vlm_stage["inputs"] = {"dets": art}
    return {"name": "qa-vlm-dets", "stages": [
        {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "0"}},
        {"module_id": "det_filter", "params": {"filter": "score >= 0.5"}},
        vlm_stage,
        {"module_id": "seq_filter", "params": {"throttle": "", "change": "", "rules": ""}},
    ]}


_RAW_VLM = [
    Detection(track_id=1, cls_id=0, bbox=(0, 0, 10, 10), score=0.9),
    Detection(track_id=2, cls_id=0, bbox=(0, 0, 10, 10), score=0.2),  # 低于 0.5，筛选前在/后不在
    Detection(track_id=3, cls_id=0, bbox=(0, 0, 10, 10), score=0.7),
]


def vlm_dets_checks() -> None:
    def run(art):
        pipe = server._chain_from_spec(_vlm_dets_spec(art))
        subs, _ = pipe.step(0, 0.0, dets=list(_RAW_VLM), frame=object(), force=True)
        subs = server._apply_vlm_inputs(pipe, subs)
        return pipe, subs

    # 默认（不声明 inputs）：step 内 submits（det_filter 过滤后 2 条）
    p_def, s_def = run(None)
    check("vlm-dets 默认: stage_inputs 为空（零回归）", p_def.stage_inputs == {})
    check("vlm-dets 默认: 报送 = 过滤后 2 条",
          sum(len(s.dets) for s in s_def) == 2, str([len(s.dets) for s in s_def]))

    # vlm.inputs.dets = yolo.dets（筛选前 3 条）→ 送审检测数确实不同
    p_raw, s_raw = run("yolo.dets")
    check("vlm-dets: 声明组装为位置键 + 全名",
          p_raw.stage_inputs == {"#3": {"dets": "yolo#1.dets"}}, str(p_raw.stage_inputs))
    check("vlm-dets: yolo.dets 报送 3 条（含被筛掉的 0.2）",
          sum(len(s.dets) for s in s_raw) == 3, str([len(s.dets) for s in s_raw]))
    check("vlm-dets: yolo.dets 与默认送审检测数不同",
          sum(len(s.dets) for s in s_raw) != sum(len(s.dets) for s in s_def))

    # vlm.inputs.dets = det_filter.dets（筛选后 2 条）→ 与默认一致、与 yolo.dets 不同
    p_fil, s_fil = run("det_filter.dets")
    check("vlm-dets: det_filter.dets 报送 2 条",
          sum(len(s.dets) for s in s_fil) == 2, str([len(s.dets) for s in s_fil]))
    check("vlm-dets: det_filter.dets 与默认一致、与 yolo.dets 不同",
          sum(len(s.dets) for s in s_fil) == sum(len(s.dets) for s in s_def)
          and sum(len(s.dets) for s in s_fil) != sum(len(s.dets) for s in s_raw))


# ==================== 7. 同模块多实例（P2-0 验收 §7.1/§7.2） ====================

def _mi_spec(df1_inputs=None, df2_inputs=None, anno_inputs=None) -> dict:
    """fm→yolo→det_filter(紧 #2)→det_filter(松 #3)→annotate(#4)→seq_filter。

    变更前（P2-0 之前）第二个 det_filter 会被装配器静默丢弃；现在两实例各自构造、
    按位置升序链式执行（见 doc/design/graph_chain.md §7）。
    """
    stages = [
        {"module_id": "frame_manager",
         "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "0"}},
        {"module_id": "det_filter", "params": {"filter": "score >= 0.8"}},  # 紧（#2）
        {"module_id": "det_filter", "params": {"filter": "score >= 0.5"}},  # 松（#3）
        {"module_id": "annotate", "params": {"thickness": 2}},
        {"module_id": "seq_filter", "params": {"throttle": "", "change": "", "rules": ""}},
    ]
    if df1_inputs is not None:
        stages[2]["inputs"] = df1_inputs
    if df2_inputs is not None:
        stages[3]["inputs"] = df2_inputs
    if anno_inputs is not None:
        stages[4]["inputs"] = anno_inputs
    return {"name": "qa-multi-instance", "stages": stages}


def multi_instance_checks() -> None:
    def run_spy(spec):
        pipe = server._chain_from_spec(spec)
        spy = _AnnoSpy()
        pipe.annotate = spy  # 替换 #4 位置的 annotate 实例（单插槽显式赋值覆盖同位置实例）
        pipe.step(0, 0.0, dets=[Detection(track_id=d.track_id, cls_id=d.cls_id,
                                          bbox=d.bbox, score=d.score) for d in _RAW],
                  frame=object(), force=True)
        assert len(spy.calls) == 1, f"annotate 应被调用一次，实为 {len(spy.calls)}"
        return pipe, spy

    # 装配：两实例各自构造进 det_filters（位置升序），不再静默取第一个
    pipe, _ = run_spy(_mi_spec())
    check("多实例: det_filters 两实例按位置装配", [p for p, _ in pipe.det_filters] == [2, 3],
          str([p for p, _ in pipe.det_filters]))

    # §7.1 各自 filter 独立生效（链式默认）：紧(#2) 先筛掉 0.7/0.2 → 松(#3) 再筛一遍 →
    # annotate 默认收检测流末端（两道筛选后仅 0.9）。若第二实例被静默丢弃，只过紧筛
    # 仍是 0.9——但松筛接 yolo.dets（绕过 #2）后结果不同，见下一组对照。
    _, spy = run_spy(_mi_spec())
    scores_chain = sorted(d["score"] for d in spy.calls[0]["dets"])
    check("多实例: 链式默认 → 两道筛选后仅 0.9", scores_chain == [0.9], str(scores_chain))

    # §7.1 行为对照：松筛(#3) 声明 inputs.dets=yolo#1.dets（绕过紧筛，原始 3 条）→
    # annotate 默认收检测流末端 = 松筛输出 0.9/0.7 两条 → 证明 #3 实例确实在跑且参数独立
    p2, spy2 = run_spy(_mi_spec(df2_inputs={"dets": "yolo#1.dets"}))
    scores_bypass = sorted(d["score"] for d in spy2.calls[0]["dets"])
    check("多实例: #3 绕过 #2 接 yolo 原始检出 → 0.7/0.9 两条（各自 filter 独立生效）",
          scores_bypass == [0.7, 0.9], str(scores_bypass))
    check("多实例: 绕过紧筛与链式结果确实不同", scores_bypass != scores_chain,
          f"{scores_bypass} vs {scores_chain}")
    check("多实例: #3 的连线组装为各自位置键",
          p2.stage_inputs == {"#3": {"dets": "yolo#1.dets"}}, str(p2.stage_inputs))

    # §7.2 两个 det_filter 各带 inputs.dets 接不同上游 → stage_inputs 互不覆盖；
    # 再让 annotate.dets 显式接 #2 输出（而非检测流末端 #3）→ 收到 1 条 0.9。
    p3, spy3 = run_spy(_mi_spec(df1_inputs={"dets": "yolo#1.dets"},
                                df2_inputs={"dets": "det_filter#2.dets"},
                                anno_inputs={"dets": "det_filter#2.dets"}))
    check("多实例: 两实例各带 inputs → 位置键互不覆盖",
          p3.stage_inputs == {"#2": {"dets": "yolo#1.dets"},
                              "#3": {"dets": "det_filter#2.dets"},
                              "#4": {"dets": "det_filter#2.dets"}},
          str(p3.stage_inputs))
    scores_a = sorted(d["score"] for d in spy3.calls[0]["dets"])
    check("多实例: annotate 显式接 #2 输出 → 1 条 0.9（未被 #3 连线覆盖）",
          scores_a == [0.9], str(scores_a))
    # 同一产物 det_filter#2.dets 被 #3 与 #4 两个下游引用（fan-out 允许，分支留路）
    check("多实例: fan-out（#2.dets 被 #3/#4 两下游引用）不报错",
          True, "det_filter#2.dets ×2 下游")


def main() -> None:
    print("== 1. module_port_decls：模块自描述端口声明 ==")
    decl_checks()
    print("== 2. _chain_from_spec：stage_inputs 组装（位置键 + 全名） ==")
    passthrough_checks()
    print("== 3. 端到端：间谍捕获 annotate.run() 实参 ==")
    e2e_checks()
    print("== 4. fail fast：PortResolutionError ==")
    failfast_checks()
    print("== 5. CLI 退出码 ==")
    cli_checks()
    print("== 6. vlm.inputs.dets：报送检测集由产物决定 ==")
    vlm_dets_checks()
    print("== 7. 同模块多实例（P2-0 验收） ==")
    multi_instance_checks()
    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
