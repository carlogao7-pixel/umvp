# -*- coding: utf-8 -*-
"""
挂载 agent 全流程记录器：真跑一遍「需求 → 总装工 → 组装工 → 链路 spec」，
把**完整对话、工具调用记录、最终产出**落成一份 Markdown 文档。

    architect（总装工）: 用户需求 → 语义模块链条
    assembler（组装工）: 语义链条 → 校验/估算 → 完整 stages spec

用法:
    conda run -n ai python web_lab/agent_run.py --ask "小区门口识别车辆车牌…" \
        --out doc/agent/full_run_record.md
环境变量/参数同 agent_llm：LLM_API_KEY（或 SILICONFLOW_API_KEY）/ LLM_BASE_URL / LLM_MODEL，
也可用 --api-key / --base-url / --model 覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web_lab"))
sys.path.insert(0, str(ROOT / "umvp"))

import agent  # noqa: E402
import agent_llm  # noqa: E402

RESULT_LIMIT = 3000  # 文档里单个工具结果的展示上限（超出截断加注）


def _short(text: str, limit: int = RESULT_LIMIT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"\n…（截断，原长 {len(text)} 字符）"


def _convo(md: list, messages: list) -> None:
    """把 run() 返回的 messages（跳过 system）渲染成对话 + 工具调用。"""
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            md += ["**用户**", "", m.get("content", ""), ""]
        elif role == "assistant":
            c = m.get("content")
            if c:
                md += ["**助手**", "", str(c), ""]
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                md += [f"**→ 工具调用** `{fn.get('name')}`", "",
                       "```json", str(fn.get("arguments") or "{}"), "```", ""]
        elif role == "tool":
            md += ["**← 工具结果**", "", "```", _short(m.get("content", "")), "```", ""]


def _phase(md: list, title: str, role: str, message: str, res: dict) -> None:
    md += [f"## {title}", "",
           f"> 角色 `{role}` ｜ ok={res.get('ok')} ｜ 轮数 steps={res.get('steps')} ｜ "
           f"工具={[c['name'] for c in res.get('tool_calls', [])]}", ""]
    _convo(md, res.get("messages") or [])
    md += ["### 本阶段产出", "", res.get("text") or (res.get("error") or "（无）"), "", "---", ""]


def main() -> int:
    ap = argparse.ArgumentParser(description="挂载 agent 全流程记录器")
    ap.add_argument("--ask", required=True, help="用户需求（自然语言）")
    ap.add_argument("--out", default="doc/agent/full_run_record.md")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    if not args.api_key and not agent_llm._api_key():
        print("未配置 LLM：导出 LLM_API_KEY 或用 --api-key/--base-url/--model。", file=sys.stderr)
        return 3

    kk = dict(model=args.model, base_url=args.base_url, api_key=args.api_key)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    model = args.model or __import__("os").environ.get("LLM_MODEL") or agent_llm.DEFAULT_MODEL

    print("[1/2] 总装工 architect …", flush=True)
    arch = agent_llm.run("architect", args.ask, **kk)
    print("[2/2] 组装工 assembler …", flush=True)
    handoff = ("总装工已把用户需求整理为下方语义链条 / 需求。请按你的工作流程落成**可校验的 stages spec**"
               "并交付（写 spec → 校验 → 估算 → 汇报）。\n\n"
               f"【用户原话】{args.ask}\n\n【总装产出】\n{arch.get('text') or arch.get('error') or ''}")
    asm = agent_llm.run("assembler", handoff, **kk)

    md = [
        "# 挂载 Agent 全流程记录（需求 → 总装工 → 组装工 → 链路 spec）",
        "",
        f"> 生成时间：{stamp} ｜ 模型：{model}",
        ">",
        "> ⚠️ **本文件由脚本自动生成，勿手改**；更新请重跑 `web_lab/agent_run.py`。",
        ">",
        "> 本记录由 `web_lab/agent_run.py` 真跑生成：包含**完整对话、工具调用记录、最终产出**。",
        "> 两角色 system 提示词（静态）见 `doc/agent/static_prompts.md`，本文不重复。",
        "> 运行顺序由壳固定，stages 数组顺序仅供声明 / 存储 / 展示（见组装工 §3）。",
        "",
        "## 需求（用户输入）",
        "",
        args.ask,
        "",
        "---",
        "",
    ]
    _phase(md, "第一阶段：总装工（architect）", "architect", args.ask, arch)
    _phase(md, "第二阶段：组装工（assembler）", "assembler", handoff, asm)

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwritten: {out}")
    print(f"  architect ok={arch.get('ok')} tools={[c['name'] for c in arch.get('tool_calls', [])]}")
    print(f"  assembler ok={asm.get('ok')} tools={[c['name'] for c in asm.get('tool_calls', [])]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
