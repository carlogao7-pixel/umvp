# -*- coding: utf-8 -*-
"""
链路 spec 校验器（低参 LLM / 组装工的自检工具）

用法:
    conda run -n ai python web_lab/validate_chain.py <spec.json>
    conda run -n ai python web_lab/validate_chain.py -      # 从 stdin 读 JSON

校验 web_lab 模块链路 spec（{"stages": [{"module_id","params"}, ...]}）是否满足硬性规则：
  1) stages 非空；frame_manager 存在且恰一段（惯例首位）
  2) 无重复 module_id
  3) 终段 ∈ {det_filter, seq_filter, plate_recog, face_detect, face_embed}
  4) 每个 module_id 是已注册产品模块；params 键在其参数白名单内；select 取值合法
  5) 参数值语义干跑（复用权威解析器，防 LLM 编造值"侥幸过闸"）：
     - det_filter / seq_filter 全参数 → server._det_filter_of/_seq_filter_of 构造干跑
       （条件 DSL 语法、group_limits/events/rules JSON 规则、SeqRule 参数越界）
     - target_crop/annotate 的 filter → pipe.conditions.parse_condition 干跑
     - yolo 的 model_path → models/ 权重池文件存在性
     - yolo 的 conf/iou ∈ (0,1]、frame_manager.frame_skip 为正整数
退出码：0=通过；1=有 error（逐条打印）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from server import MODULES, MODELS_DIR, _det_filter_of, _find_file, _seq_filter_of  # noqa: E402
from pipe.conditions import parse_condition  # noqa: E402

TERMINAL_OK = {"det_filter", "seq_filter", "plate_recog", "face_detect", "face_embed"}

# 轻量数值范围检查（键级；条件见 _num_check）
_NUM_RULES = {
    ("yolo", "conf"): (0.0, 1.0), ("yolo", "iou"): (0.0, 1.0),
    ("frame_manager", "frame_skip"): (1, None),
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


def validate(spec: dict) -> list[str]:
    errs: list[str] = []
    stages = (spec or {}).get("stages")
    if not isinstance(stages, list) or not stages:
        return ["stages 缺失或为空（应为非空数组，首段 frame_manager）"]

    ids = [s.get("module_id") for s in stages]
    # 1) frame_manager
    if "frame_manager" not in ids:
        errs.append("缺少必备阶段 frame_manager")
    elif ids.count("frame_manager") > 1:
        errs.append("frame_manager 出现多次（每模块最多一次）")
    elif ids[0] != "frame_manager":
        errs.append("frame_manager 惯例放首位（当前不在首位）")
    # 2) 重复模块
    dup = sorted({m for m in ids if m and ids.count(m) > 1})
    if dup:
        errs.append(f"重复模块阶段（每模块最多一次）: {dup}")
    # 3) 终段
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
            if opts and str(v) not in [str(o) for o in opts]:
                errs.append(f"stages[{i}] {mid}.{k}={v!r} 不在允许值 {opts}")
        errs.extend(_semantic_dry_run(i, mid, params))
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
    errs = validate(spec)
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
