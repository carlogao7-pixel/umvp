# -*- coding: utf-8 -*-
"""
链路 spec 校验器（低参 LLM / 组装工的自检工具）

用法:
    conda run -n ai python web_lab/validate_chain.py <spec.json>
    conda run -n ai python web_lab/validate_chain.py -      # 从 stdin 读 JSON

校验 web_lab 模块链路 spec（{"stages": [{"module_id","params"}, ...]}）是否满足硬性规则：
  1) stages 非空；frame_manager 存在且恰一段（惯例首位）
  2) 终段 ∈ {det_filter, seq_filter, plate_recog, face_detect, face_embed}
     （**同模块多实例已放行**——节点身份 = stages 数组下标，见 P2-0/设计稿 §3.1）
  3) 每个 module_id 是已注册产品模块；params 键在其参数白名单内；select 取值合法
     （含 vlm.image_source：annotated/frame/crops，非法取值即报错）
  4) 声明即执行：spec 声明的每个已注册阶段必须被装配器装配进执行路径
     （server.CHAIN_ASSEMBLED_MODULES）；声明了但装配器无处理（如 face_store）会报错。
     这条正是「人脸链声明 det_filter 却静默不执行」事故（WO-0001）的回归测试。
  5) 端口连线（P5 第 2 步 + P2-0 位置化）：各 stage 可选字段 inputs（{端口: 上游产物名}）
     DAG 校验——产物名（短名/全名）须在 ARTIFACT_TYPES、端口须在该模块 INPUTS 声明、
     引用须指向**上游阶段**（全名按位置判定）、类型匹配、每端口单一上游（fan_in=0 时
     值须字符串；多实例短名歧义报错要求写全名）。**可连线端口清单与"未纳入解析"名单均
      从 server.CHAIN_RESOLVED_PORTS 推导**，不再硬编码。`vlm.inputs` 只允许 `dets`；
      `vlm.inputs.image` 不接受（图片源统一由 vlm 参数 image_source 管），提示改用 image_source。
      annotate 双模式（WO-0021）：scope ∈ {frame, record}（经参数表 options 校验）、
      max_records ≥ 1；连线产物与该 annotate 实例 scope 匹配（record 实例不产
      annotated、frame 实例不产 annotated_records，连线期拦截）。
      未声明 inputs 的旧 spec 不报错不警告。
  6) 参数值语义干跑（复用权威解析器，防 LLM 编造值"侥幸过闸"）：
     - det_filter / seq_filter 全参数 → server._det_filter_of/_seq_filter_of 构造干跑
       （条件 DSL 语法、group_limits/events/rules JSON 规则、SeqRule 参数越界）
     - target_crop/annotate 的 filter → pipe.conditions.parse_condition 干跑
     - yolo 的 model_path → models/ 权重池文件存在性
     - yolo 的 conf/iou ∈ (0,1]、frame_manager.frame_skip 为正整数
  7) vlm.image_source 一致性（非致命警告，经 warnings 列表返回）：显式声明 annotated 但
     上游无 annotate / 声明 crops 但上游无 target_crop → 运行时静默回退，提示但不算错。
  退出码：0=通过（含仅有警告）；1=有 error（逐条打印）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from server import (  # noqa: E402
    CHAIN_ASSEMBLED_MODULES,
    CHAIN_RESOLVED_PORTS,
    CHAIN_UNRESOLVED_INPUT_MODULES,
    MODULES,
    MODELS_DIR,
    _det_filter_of,
    _find_file,
    _positions_by_module,
    _seq_filter_of,
    module_port_decls,
    port_fan_in,
    resolve_artifact,
)
from pipe.composer import ARTIFACT_TYPES, PortResolutionError, artifact_short  # noqa: E402
from pipe.conditions import parse_condition  # noqa: E402

TERMINAL_OK = {"det_filter", "seq_filter", "plate_recog", "face_detect", "face_embed"}

# 轻量数值范围检查（键级；条件见 _num_check）
_NUM_RULES = {
    ("yolo", "conf"): (0.0, 1.0), ("yolo", "iou"): (0.0, 1.0),
    ("frame_manager", "frame_skip"): (1, None),
    ("annotate", "max_records"): (1, None),  # record 模式单帧带图记录上限（WO-0021）
}


def _module_params(mid: str) -> dict:
    return (MODULES.get(mid) or {}).get("params") or {}


def _semantic_dry_run(i: int, mid: str, params: dict) -> list:
    """参数值语义校验：复用 server/pipe 权威解析器构造干跑，抛错即不合法。"""
    errs = []

    def _s(k) -> str:
        return str(params.get(k, "") or "").strip()

    if mid == "det_filter":
        try:
            _det_filter_of(params)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            errs.append(f"stages[{i}] det_filter 参数无法装配: {e}")
    elif mid == "seq_filter":
        if any(_s(k) for k in ("throttle", "change", "rules")):
            try:
                _seq_filter_of(params, what="all")
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                errs.append(f"stages[{i}] seq_filter 参数无法装配: {e}")
    elif mid in ("target_crop", "annotate"):
        cond = _s("filter")
        if cond:
            try:
                parse_condition(cond)
            except (ValueError, TypeError) as e:
                errs.append(f"stages[{i}] {mid}.filter 条件无法解析: {cond!r}（{e}）")
    elif mid == "yolo":
        mp = _s("model_path")
        if mp and _find_file(mp, [MODELS_DIR]) is None:
            errs.append(f"stages[{i}] yolo.model_path={mp!r} 在模型池中不存在"
                        f"（models/ 下可用: {sorted(p.name for p in MODELS_DIR.glob('*.pt'))}）")

    for (m, k), (lo, hi) in _NUM_RULES.items():
        if m != mid or k not in params:
            continue
        try:
            v = float(params[k])
        except (TypeError, ValueError):
            errs.append(f"stages[{i}] {mid}.{k}={params[k]!r} 不是数值")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            errs.append(f"stages[{i}] {mid}.{k}={v} 越界（应在 [{lo}, {hi}]）")
        if mid == "frame_manager" and k == "frame_skip" and v != int(v):
            errs.append(f"stages[{i}] frame_manager.frame_skip 应为正整数")
    return errs


def _validate_stage_inputs(i: int, mid: str, st: dict, upstream_ids: list,
                           positions_by_module: dict, decl: dict, warns: list,
                           anno_scope_by_idx: dict | None = None) -> list:
    """校验阶段 inputs 连线（P5 第 2 步 DAG 规则，P2-0 位置化）。返回 error 列表。

    规则（设计稿 §3④/§3.4，逐条可读报错）：
      ① 单上游端口的值须为字符串产物名（列表/多值 → 错）；fan_in=1 端口（本期无）才允许多条；
      ② 产物名（短名/全名）须在 ARTIFACT_TYPES 中（未知产物 → 错）；
      ③ 端口须在该模块 INPUTS 声明中（非法端口 → 错）；
      ④ 引用须指向**上游阶段**（全名按位置：位置须 < 本阶段下标；短名须为上游模块 → 错）；
      ⑤ 类型匹配：产物类型 vs 模块 INPUTS[端口]（不匹配 → 错）；
      ⑥ 短名在多实例下歧义 → 报错要求写全名（resolve_artifact）；
      ⑦ annotate 双模式产物与该实例 scope 匹配（WO-0021）：连 `annotate.annotated`
        （帧级标注图）要求该实例 scope≠record；连 `annotate.annotated_records`（逐记录
        带图流）要求该实例 scope=record——产物名静态在表，产不产由 scope 定，连线期拦截。
    同一产物被多个下游引用**允许**（分支留路，不报警）。
    对本期未纳入解析的阶段（见 server.CHAIN_UNRESOLVED_INPUT_MODULES）：写了 inputs
    只给非致命警告；本期该名单已为空（plate_recog/face_detect/vlm.dets 均已解析）。
    未声明 inputs（None/空）→ 不报错不警告（零回归）。
    """
    ins = st.get("inputs")
    if not ins:
        return []
    if not isinstance(ins, dict):
        return [f"stages[{i}] {mid}.inputs 应为对象（{{端口: 上游产物名}}）"]
    errs: list = []
    declared = (decl.get(mid) or {}).get("inputs") or {}
    for port, art in ins.items():
        if isinstance(art, list):
            if not port_fan_in(mid, port):
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 值须为字符串产物名"
                            f"（每端口单一上游，不支持列表/多值）；得到 list")
                continue
            arts = art  # fan_in=1 端口（本期无）才接受多入边
        else:
            arts = [art]
        for a in arts:
            if not isinstance(a, str):
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 值须为字符串产物名"
                            f"（每端口单一上游，不支持列表/多值）；得到 {type(a).__name__}")
                continue
            base = artifact_short(a)
            if base not in ARTIFACT_TYPES:
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 引用未知产物 {a!r}"
                            f"（可用产物: {sorted(ARTIFACT_TYPES)}）")
                continue
            # vlm 仅开放 dets（决定"报送哪些检测"）；图片源统一由 image_source 管，避免双机制。
            if mid == "vlm" and port != "dets":
                hint = ("；图片源请改用 vlm 参数 image_source（annotated/frame/crops）"
                        if port == "image" else "")
                errs.append(f"stages[{i}] vlm.inputs 只允许端口 'dets'（报送检测集），"
                            f"不接受 {port!r}{hint}")
                continue
            if port not in declared:
                errs.append(f"stages[{i}] {mid}.inputs 端口 {port!r} 未在该模块 INPUTS 声明中"
                            f"（已声明端口: {sorted(declared)}）")
            want = declared.get(port)
            if want and ARTIFACT_TYPES[base] != want:
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 类型不符：产物 {a!r} 为 "
                            f"{ARTIFACT_TYPES[base]}，端口声明为 {want}")
            # 上游判定（P2-0 位置化）：短名先归一为全名（多实例歧义在此报错，不猜）
            try:
                full = resolve_artifact(a, positions_by_module)
            except PortResolutionError as e:
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 产物 {a!r} 无法解析: {e}")
                continue
            head, _dot, _tail = full.partition(".")
            producer, _hash, pos_str = head.partition("#")
            # ⑦ annotate 双模式：产物须与该实例 scope 匹配（WO-0021，设计稿 §9）。
            #    产物名静态在 ARTIFACT_TYPES，但"产不产"由该 annotate 实例的 scope 决定。
            if producer == "annotate" and pos_str.isdigit() and anno_scope_by_idx:
                scope = str(anno_scope_by_idx.get(int(pos_str)) or "frame").strip().lower()
                short = artifact_short(a)
                if short == "annotate.annotated" and scope == "record":
                    errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 引用 {a!r}：该 annotate "
                                f"实例为 record 模式，不产帧级标注图 annotate.annotated"
                                f"（逐记录带图流请连线 annotate.annotated_records）")
                elif short == "annotate.annotated_records" and scope != "record":
                    errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 引用 {a!r}：该 annotate "
                                f"实例未开 record 模式（scope={scope or 'frame'}），不产 "
                                f"annotate.annotated_records（帧级标注图请连线 annotate.annotated）")
            if pos_str.isdigit():
                if int(pos_str) >= i:
                    errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 引用 {a!r} 不是上游产物"
                                f"（由阶段 {producer!r} 产生，须位于本阶段之前）")
            elif producer not in upstream_ids:
                errs.append(f"stages[{i}] {mid}.inputs[{port!r}] 引用 {a!r} 不是上游产物"
                            f"（由阶段 {producer!r} 产生，须位于本阶段之前）")
    if mid in CHAIN_UNRESOLVED_INPUT_MODULES:
        warns.append(f"stages[{i}] {mid} 本期不解析 inputs 连线，将按默认取数"
                     f"（可连线端口见设计器；已开放连线的模块: {sorted(CHAIN_RESOLVED_PORTS)}）")
    return errs


def validate(spec: dict, warnings: list | None = None) -> list[str]:
    """校验链路 spec。返回 error 列表（非空=不通过）。

    warnings（可选）：传入列表时，追加非致命提示（如 vlm 声明的 image_source 其上游
    无对应产物阶段，运行时会静默回退）。不传则忽略——保持旧调用方（agent/tests）签名兼容。
    """
    errs: list[str] = []
    warns: list = warnings if warnings is not None else []
    stages = (spec or {}).get("stages")
    if not isinstance(stages, list) or not stages:
        return ["stages 缺失或为空（应为非空数组，首段 frame_manager）"]

    ids = [s.get("module_id") for s in stages]
    # 1) frame_manager（每链路恰一段；同模块其它实例可多，见 P2-0）
    if "frame_manager" not in ids:
        errs.append("缺少必备阶段 frame_manager")
    elif ids.count("frame_manager") > 1:
        errs.append("frame_manager 出现多次（帧源每链路恰一段）")
    elif ids[0] != "frame_manager":
        errs.append("frame_manager 惯例放首位（当前不在首位）")
    # 2) 终段
    if ids and ids[-1] not in TERMINAL_OK:
        errs.append(f"终段 {ids[-1]!r} 非法；须为 {sorted(TERMINAL_OK)} 之一")
    # 4) 模块 + 参数
    for i, st in enumerate(stages):
        mid = st.get("module_id")
        if not mid:
            errs.append(f"stages[{i}] 缺 module_id")
            continue
        mod = MODULES.get(mid)
        if mod is None or mod.get("hidden"):
            errs.append(f"stages[{i}] 未知/非产品模块: {mid!r}")
            continue
        params = st.get("params") or {}
        if not isinstance(params, dict):
            errs.append(f"stages[{i}] {mid}.params 应为对象")
            continue
        schema = _module_params(mid)
        for k, v in params.items():
            spec_k = schema.get(k)
            if spec_k is None:
                errs.append(f"stages[{i}] {mid}: 未知参数 {k!r}（不在模块参数表）")
                continue
            if spec_k.get("test_only"):
                errs.append(f"stages[{i}] {mid}: 参数 {k!r} 为模块测试页专用（test_only），"
                            f"不能进链路 spec")
            opts = spec_k.get("options")
            # 选项支持纯字符串或 {value,label} 对象（vlm.vlm_provider，WO-0019）；按 value 比较。
            # v 为 None=该列未设置（DB 空列还原/未填），回落模块默认，不算非法取值；
            # 显式填了非允许字符串才报错（如 image_source='lowres'）。
            opt_vals = [str(o["value"] if isinstance(o, dict) else o) for o in (opts or [])]
            if opt_vals and v is not None and str(v) not in opt_vals:
                errs.append(f"stages[{i}] {mid}.{k}={v!r} 不在允许值 {opts}")
        errs.extend(_semantic_dry_run(i, mid, params))
    # 5) 声明即执行：已注册产品模块若未被装配器装配，声明了也不会执行（静默失效）。
    # 未知/隐藏模块已在上一步报错，这里只查能通过注册检查的阶段。
    for m in ids:
        if not m or m not in MODULES or (MODULES.get(m) or {}).get("hidden"):
            continue
        if m not in CHAIN_ASSEMBLED_MODULES:
            errs.append(f"声明即执行: 阶段 {m!r} 未被 _chain_from_spec 装配进执行路径"
                        f"（可装配阶段: {sorted(CHAIN_ASSEMBLED_MODULES)}）")
    # 6) 端口连线 DAG 校验（P5 第 2 步 + P2-0 位置化）：inputs={端口: 上游产物名}。
    #    未声明 inputs 的旧 spec 不报错不警告（零回归）；端口声明与类型取模块 INPUTS 声明。
    #    仅对已注册产品模块校验（未知/隐藏模块已在第 4 步报错，避免重复噪声）。
    #    annotate 的 scope 按实例下标收集（产物↔scope 匹配检查用，WO-0021）。
    decl = module_port_decls()
    positions_by_module = _positions_by_module(ids)
    anno_scope_by_idx = {i: (st.get("params") or {}).get("scope")
                         for i, st in enumerate(stages)
                         if st.get("module_id") == "annotate"}
    for i, st in enumerate(stages):
        mid = st.get("module_id")
        if not mid or mid not in MODULES or (MODULES.get(mid) or {}).get("hidden"):
            continue
        errs.extend(_validate_stage_inputs(i, mid, st, ids[:i], positions_by_module,
                                           decl, warns, anno_scope_by_idx))
    # 7) vlm.image_source 与上游产物一致性（非致命警告）：
    #    显式声明 annotated 但链路上游无 annotate / 声明 crops 但上游无 target_crop 时，
    #    驱动层会静默回退到默认源（annotated→原帧），行为正确但非用户本意，故提示。
    #    未声明 image_source 的旧 spec 不提示（缺省=现状，零回归）。
    for i, st in enumerate(stages):
        if st.get("module_id") != "vlm":
            continue
        params = st.get("params") or {}
        if not isinstance(params, dict) or "image_source" not in params:
            continue
        src = str(params.get("image_source") or "").strip().lower()
        upstream = ids[:i]
        if src == "annotated" and "annotate" not in upstream:
            warns.append(f"stages[{i}] vlm.image_source='annotated' 但其上游无 annotate "
                         f"阶段，运行时将回退本帧原图（如需带框图请加 annotate 阶段）")
        elif src == "crops" and "target_crop" not in upstream:
            warns.append(f"stages[{i}] vlm.image_source='crops' 但其上游无 target_crop "
                         f"阶段，运行时将回退（annotated→原帧）")
    return errs


def _load(arg: str) -> dict:
    text = sys.stdin.read() if arg == "-" else Path(arg).read_text(encoding="utf-8")
    return json.loads(text)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        spec = _load(sys.argv[1])
    except Exception as e:
        print(f"[ERR] 无法读取/解析 spec: {e}")
        return 1
    warns: list = []
    errs = validate(spec, warnings=warns)
    for w in warns:
        print(f"  [WARN] {w}")
    if errs:
        print("链路 spec 校验失败：")
        for e in errs:
            print(f"  [ERR] {e}")
        return 1
    n = len(spec.get("stages") or [])
    print(f"[OK] 链路 spec 通过（{n} 个阶段，终段 {spec['stages'][-1].get('module_id')}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
