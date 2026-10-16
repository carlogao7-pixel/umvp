# -*- coding: utf-8 -*-
"""
链路组装 agent 评测运行器（两段式：总装 → 组装）。

读取 `tests/agent_eval_questions.json`，对每条场景：
  - 先跑 **总装工**（architect）；
  - 若场景 kind ∈ {build, modify}，再把总装的产出作为"链路需求"交给 **组装工**（assembler）。
组装工当前处于 **plan 模式**（仅表述工作流程，见 .claude/agents/pipeline-assembler.md）。

需要 LLM：环境变量 LLM_API_KEY（或 SILICONFLOW_API_KEY）/ LLM_BASE_URL / LLM_MODEL，
或用 --api-key / --base-url / --model 覆盖。

用法:
    conda run -n ai python tests/agent_eval.py --dry-run          # 只列场景，不联网
    conda run -n ai python tests/agent_eval.py --limit 3 --verbose
    conda run -n ai python tests/agent_eval.py --only build-plate
输出: out/agent_eval/<id>.md（逐场景）+ out/agent_eval/_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web_lab"))
sys.path.insert(0, str(ROOT / "umvp"))

import agent_llm  # noqa: E402

QUESTIONS = ROOT / "tests/agent_eval_questions.json"
OUTDIR = ROOT / "out/agent_eval"
NEEDS_ASSEMBLER = {"build", "modify"}


def _load_scenarios() -> list[dict]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))["scenarios"]


def _handoff(arch_text: str, scenario: dict) -> str:
    """把总装产出包装成交给组装工的任务包。"""
    return ("总装工已把用户需求整理为下方需求/语义链条。请按你的工作流程处理，"
            "（当前为 plan 模式）只用文字表述你会执行的具体工作流程。\n\n"
            f"【场景】{scenario['message']}\n\n【总装产出】\n{arch_text}")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def run_scenario(sc: dict, *, model, base_url, api_key, verbose: bool) -> dict:
    def on_event(ev):
        if verbose and ev.get("type") in ("tool_call", "tool_result"):
            print(f"    [{ev['type']}] {ev.get('name')}", flush=True)

    print(f"  → 总装工 architect …", flush=True)
    arch = agent_llm.run("architect", sc["message"], model=model, base_url=base_url,
                         api_key=api_key, on_event=on_event)
    out = {"architect": arch}
    asm = None
    if sc["kind"] in NEEDS_ASSEMBLER and arch.get("ok"):
        print(f"  → 组装工 assembler（plan 模式）…", flush=True)
        asm = agent_llm.run("assembler", _handoff(arch["text"], sc),
                            model=model, base_url=base_url, api_key=api_key, on_event=on_event)
        out["assembler"] = asm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="链路组装 agent 评测")
    ap.add_argument("--dry-run", action="store_true", help="只列场景，不联网")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--only", default="", help="只跑指定 id")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="", help="输出目录（默认 out/agent_eval）")
    args = ap.parse_args()

    scenarios = _load_scenarios()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s["id"] in wanted]
    if args.limit:
        scenarios = scenarios[:args.limit]

    print(f"场景数: {len(scenarios)}")
    if args.dry_run:
        for s in scenarios:
            roles = "architect+assembler" if s["kind"] in NEEDS_ASSEMBLER else "architect"
            print(f"  - {s['id']:20s} kind={s['kind']:9s} [{roles}] {s['focus']}")
        return 0

    if not args.api_key and not agent_llm._api_key():
        print("未配置 LLM：请导出 LLM_API_KEY（或 SILICONFLOW_API_KEY），"
              "或用 --api-key/--base-url/--model。", file=sys.stderr)
        return 3

    outdir = Path(args.out) if args.out else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = []
    for i, sc in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {sc['id']} ({sc['kind']}) — {sc['focus']}")
        try:
            res = run_scenario(sc, model=args.model, base_url=args.base_url,
                               api_key=args.api_key, verbose=args.verbose)
        except Exception as e:  # 单条失败不影响其余
            res = {"error": f"{type(e).__name__}: {e}"}
        results.append((sc, res))

        md = [f"# {sc['id']}  ({sc['kind']})", "", f"> 场景：{sc['message']}",
              f"> 关注点：{sc['focus']}", f"> 采集：{stamp}", ""]
        a = res.get("architect")
        md += ["## 总装工（architect）",
               f"_ok={a.get('ok')} steps={a.get('steps')} "
               f"tools={[c['name'] for c in a.get('tool_calls', [])]}_" if a else "_（未跑）_", "",
               (a or {}).get("text") or (a or {}).get("error", ""), ""]
        if "assembler" in res:
            s = res["assembler"]
            md += ["## 组装工（assembler · plan 模式）",
                   f"_ok={s.get('ok')} steps={s.get('steps')} "
                   f"tools={[c['name'] for c in s.get('tool_calls', [])]}_", "",
                   s.get("text") or s.get("error", ""), ""]
        if res.get("error"):
            md += ["## 错误", res["error"], ""]
        (outdir / f"{_safe(sc['id'])}.md").write_text("\n".join(md), encoding="utf-8")

    summary = [f"# 链路组装 agent 评测汇总", "", f"> 时间：{stamp}｜模型："
               f"{args.model or os.environ.get('LLM_MODEL') or agent_llm.DEFAULT_MODEL}", "",
               "| id | kind | 关注点 | 总装ok | 组装ok |", "|---|---|---|---|---|"]
    for sc, res in results:
        a = res.get("architect") or {}
        s = res.get("assembler") or {}
        summary.append(f"| {sc['id']} | {sc['kind']} | {sc['focus']} | "
                       f"{a.get('ok')} | {s.get('ok', '—')} |")
    (outdir / "_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\n完成，输出目录: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
