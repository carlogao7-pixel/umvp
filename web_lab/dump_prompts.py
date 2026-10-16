# -*- coding: utf-8 -*-
"""
导出「挂载 agent」两角色的**静态 system 提示词快照**到文档（可重跑）。

system 提示词按块组装（`agent.context_blocks(role)`）：角色提示词 → 系统简介 → 模块介绍 → 可用工具。
本文在开头列出「块 → 来源文档」对照，再逐角色、逐块展示正文 + 工具 JSON Schema。
运行时由工具调用获得的动态内容（模块实参、知识库正文）不入本文。

用法:
    conda run -n ai python web_lab/dump_prompts.py
    conda run -n ai python web_lab/dump_prompts.py --out doc/agent/static_prompts.md
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

ROLES = [("architect", "总装工"), ("assembler", "组装工")]


def build_md() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 挂载 Agent 静态提示词（总装工 / 组装工）",
        "",
        "> 本文是 web 端挂载 agent 两角色**运行时的静态 system 提示词**快照，按块展示。"
        "system 由 `agent.build_context(role)` 给出：**自包含的角色提示词**（§1-§8，"
        "内联了系统介绍/链路机制/模块地图/工作流/产出格式/文件索引/工具）+ 工具 JSON Schema。",
        ">",
        "> ⚠️ **本文件由脚本自动生成，勿手改**；更新请重跑：`conda run -n ai python web_lab/dump_prompts.py`。",
        ">",
        f"> - 生成方式：`conda run -n ai python web_lab/dump_prompts.py`（可重跑）｜生成时间：{stamp}",
        "> - **动态部分**（工具返回的模块实参等）在运行时经 function calling 获得，不在本文。",
        "> - 口径以 `.claude/agents/` 与 `web_lab/agent.py` 为准；本文是静态快照。",
        "",
        "## 提示词分块与来源",
        "",
        "| 块 | 来源 |",
        "|---|---|",
    ]
    for i, b in enumerate(agent.context_blocks("architect"), 1):
        src = b["source"]
        if i == 1:  # 块1 两角色文件名不同
            src = ("`.claude/agents/pipeline-architect.md`（总装）/ `pipeline-assembler.md`（组装），"
                   "去 frontmatter")
        lines.append(f"| {i}. {b['title']} | {src} |")
    lines.append("| 附. 工具清单（JSON Schema） | `web_lab/agent.py` `tool_schemas()`（function-calling 参数） |")
    lines += ["", "---", ""]

    for role, name in ROLES:
        lines += [f"## {name}（`{role}`）", ""]
        for i, b in enumerate(agent.context_blocks(role), 1):
            lines += [f"### 块 {i} · {b['title']}", "",
                      f"> 来源：{b['source']}", "",
                      "```text", b["content"].rstrip(), "```", ""]
        lines += ["### 附 · 工具清单（JSON Schema）", "",
                  "> 来源：`web_lab/agent.py` `tool_schemas()`", "",
                  "```json",
                  json.dumps(agent.tool_schemas(), ensure_ascii=False, indent=2),
                  "```", "", "---", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出挂载 agent 两角色静态提示词快照（分块）")
    ap.add_argument("--out", default="doc/agent/static_prompts.md", help="输出文档路径（相对项目根）")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    out.write_text(build_md(), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
