# -*- coding: utf-8 -*-
"""
轻量 Agent 运行时：上下文披露 + 工具调用（不依赖 Claude Code / CC 工具）

web 端"挂载 agent"的运行时内核：把「角色提示词 + 工具清单」披露给任意 LLM（含低参），
并执行它发起的工具调用。

两角色（初始提示词不同，均自包含，见 .claude/agents/*.md）：

    architect（总装工）：整理复述需求 + 业务层链路设计——只认模块的"业务功能 + 可调参数
        的业务语义"，输出"语义模块链条"，不落具体参数、不碰代码。
    assembler（组装工）：把语义链落成合规 stages spec——侧重各模块的代码参数逻辑与职责
        红线，负责拼接、参数、校验、估算、测试。

角色提示词 = .claude/agents/*.md（纯 Markdown 直读）；技术文档 = 项目内 Markdown
（doc/ 与 .claude/agents/assembler/），经 read_doc 工具按需读取。**不依赖任何外部技能库**。
工具用 JSON Schema 描述（OpenAI 通用 function-calling 形态），宿主 LLM 直接照用。

CLI（自测）:
    conda run -n ai python web_lab/agent.py context [architect|assembler]
    conda run -n ai python web_lab/agent.py tool <name> '<json>'
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / ".claude/agents"
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

_ROLE_FILES = {"architect": "pipeline-architect.md", "assembler": "pipeline-assembler.md"}

# 未知角色的兜底（给裸 LLM 的最小角色声明）
_DEFAULT_ROLE = (
    "你是 UMVP 链路组装助手。产出符合「链路壳标准」的 stages spec；"
    "不改核心模块，不新增依赖；用工具自检后再交付。"
)


# ---------------- 文档 / 提示词读取（纯 Markdown） ----------------
def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def read_agent_prompt(role: str) -> str:
    f = _ROLE_FILES.get(role)
    if f and (AGENTS / f).is_file():
        return _strip_frontmatter((AGENTS / f).read_text(encoding="utf-8"))
    return _DEFAULT_ROLE


def read_doc_file(rel: str) -> str:
    """读项目内 Markdown 文档（相对项目根，如 doc/modules/frame_manager.md、
    .claude/agents/assembler/01-shell-standard.md）。仅允许 .md 且须在项目根内。"""
    p = (ROOT / rel).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise ValueError("路径越界")
    if p.suffix.lower() != ".md" or not p.is_file():
        raise FileNotFoundError(rel)
    return p.read_text(encoding="utf-8")


# 文档索引来源目录（不再关联任何技能库）
_DOC_DIRS = ("doc", ".claude/agents")


def docs_index() -> dict:
    """可读文档索引：项目 doc/ 与 .claude/agents/ 下的全部 .md；index = 组装工文档 README。"""
    files: List[str] = []
    for base in _DOC_DIRS:
        d = ROOT / base
        if d.is_dir():
            files += [str(p.relative_to(ROOT)) for p in d.rglob("*.md")]
    readme = AGENTS / "assembler" / "README.md"
    index = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    return {"index": index, "files": sorted(set(files))}


# ---------------- 平台权威信息（惰性导入 server） ----------------
def module_catalog() -> dict:
    from server import MODULES  # 惰性：避免与 server 循环导入
    out = {}
    for mid, m in MODULES.items():
        if m.get("hidden"):
            continue
        params = {}
        for k, spec in (m.get("params") or {}).items():
            if spec.get("test_only"):
                continue
            params[k] = {"label": spec.get("label", k), "type": spec.get("type", "str"),
                         "default": spec.get("default", "")}
            if spec.get("options"):
                params[k]["options"] = spec["options"]
        out[mid] = {"name": m.get("name", mid), "group": m.get("group", ""),
                    "desc": m.get("desc", ""), "params": params}
    return out


def _tools_text() -> str:
    """工具清单（名字 + 一句话），供未内联工具表的角色提示词在运行时补全。"""
    return "\n".join(f"- {n}: {t['desc']}" for n, t in TOOLS.items())


# ---------------- 工具定义（JSON Schema + handler） ----------------
def _t_read_doc(path: str = "") -> dict:
    if not path:
        return docs_index()
    return {"path": path, "content": read_doc_file(path)}


def _t_list_modules() -> dict:
    return {"modules": module_catalog()}


def _t_validate_spec(spec: Optional[dict] = None, **_) -> dict:
    from validate_chain import validate
    errs = validate(spec or {})
    return {"ok": not errs, "errors": errs}


def _t_estimate(spec: Optional[dict] = None, streams: int = 1,
                vram_budget_mb: Optional[float] = None, **_) -> dict:
    from resources import ResourceEstimator
    # spec stages 用 module_id；估算器用 module
    est_stages = [{"module": s.get("module_id"), "params": s.get("params") or {}}
                  for s in (spec or {}).get("stages", [])]
    est = ResourceEstimator(project_root=str(ROOT))
    pe = est.estimate_pipeline({"stages": est_stages}, streams=int(streams or 1),
                               vram_budget_mb=vram_budget_mb)
    result = pe.to_dict() if hasattr(pe, "to_dict") else {"estimate": str(pe)}
    if vram_budget_mb:
        result["back_calculate"] = est.back_calculate(
            {"stages": est_stages}, streams=int(streams or 1), vram_budget_mb=vram_budget_mb)
    return result


def _t_list_pipelines() -> dict:
    """列出已存链路（pipelines + 每步引用的模块）。DB 不可用时如实返回。"""
    try:
        import db
        return {"pipelines": db.list_pipelines()}
    except Exception as e:  # MySQL 未启用 / 连接失败
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "pipelines": []}


def _t_get_pipeline(pipeline_id: str = "", **_) -> dict:
    """取回一条已存链路的完整 stages（含各步 params）。"""
    if not pipeline_id:
        return {"ok": False, "error": "缺少 pipeline_id（如 PIPE0001）"}
    try:
        import db
        return {"pipeline": db.load_pipeline(pipeline_id)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


TOOLS: Dict[str, dict] = {
    "read_doc": {
        "desc": "读项目文档（相对项目根，如 doc/modules/frame_manager.md、.claude/agents/assembler/01-shell-standard.md；空=返回文档索引）",
        "when": "需要系统介绍 / 业务决策 / 壳规范 / 坑与清单等知识时",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "相对项目根的文件路径，空=返回索引"},
            },
        },
        "handler": _t_read_doc,
    },
    "list_modules": {
        "desc": "列出所有产品模块及其精确参数（键/类型/默认/取值）——构建 spec 的依据",
        "when": "要在链条里选模块 / 落参数时（组装工主用）",
        "schema": {"type": "object", "properties": {}},
        "handler": _t_list_modules,
    },
    "validate_spec": {
        "desc": "校验链路 spec 是否满足硬性规则与参数白名单；返回 {ok, errors}",
        "when": "写出/改动 spec 后自检",
        "schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object",
                         "description": '{"stages":[{"module_id","params"}]}'},
            },
            "required": ["spec"],
        },
        "handler": _t_validate_spec,
    },
    "estimate": {
        "desc": "对链路 spec 做资源估算（单链路 + 多路 + 预算反推）",
        "when": "方案定稿前后评估显存/内存/帧率",
        "schema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object"},
                "streams": {"type": "integer", "default": 1},
                "vram_budget_mb": {"type": "number",
                                   "description": "给显存预算则附反推建议"},
            },
            "required": ["spec"],
        },
        "handler": _t_estimate,
    },
    "list_pipelines": {
        "desc": "列出已保存的链路（id/名称/步骤模块）——修改旧链路时先看现状",
        "when": "用户要改/优化已存链路时",
        "schema": {"type": "object", "properties": {}},
        "handler": _t_list_pipelines,
    },
    "get_pipeline": {
        "desc": "按 id 取回已存链路的完整 stages（含各步参数）",
        "when": "定位到某条链路后看它的每阶段参数",
        "schema": {
            "type": "object",
            "properties": {
                "pipeline_id": {"type": "string", "description": "如 PIPE0001"},
            },
            "required": ["pipeline_id"],
        },
        "handler": _t_get_pipeline,
    },
}


def tool_schemas() -> List[dict]:
    """OpenAI/通用 function-calling 形态的工具清单。"""
    return [{"name": n, "description": t["desc"], "parameters": t["schema"]}
            for n, t in TOOLS.items()]


def call_tool(name: str, arguments: Optional[dict] = None) -> dict:
    t = TOOLS.get(name)
    if not t:
        return {"ok": False, "error": f"未知工具: {name}（可用: {list(TOOLS)}）"}
    try:
        res = t["handler"](**(arguments or {}))
        return res if isinstance(res, dict) else {"ok": True, "result": res}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------- 上下文披露 ----------------
def context_blocks(role: str = "assembler") -> list:
    """返回 system 提示词的分块列表（供 build_context 与快照文档复用）。

    角色提示词若自包含（§1-§8，内联工具表）则为单块；否则运行时补一块「可用工具」。
    """
    prompt = read_agent_prompt(role)
    blocks = [{"title": "系统提示词",
               "source": f".claude/agents/{_ROLE_FILES.get(role, '（兜底角色声明）')}（去 frontmatter）",
               "content": prompt}]
    if "## 8. 可用工具" not in prompt:
        blocks.append({"title": "可用工具",
                       "source": "web_lab/agent.py `TOOLS`（角色提示词未内联工具表时运行时补）",
                       "content": "## 可用工具（一次只调一个）\n" + _tools_text()})
    return blocks


def build_context(role: str = "assembler") -> dict:
    """披露给 LLM 的完整上下文：角色提示词（自包含）+ 工具清单（缺则补）+ 工具 JSON Schema。

    提示词自包含（内联系统介绍/机制/模块地图/工具，§1-§8）；未内联工具表的角色运行时补一段，
    保证其仍知道有哪些工具。
    """
    system = "\n\n".join(b["content"] for b in context_blocks(role))
    return {"role": role, "system": system, "tools": tool_schemas()}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "context":
        role = sys.argv[2] if len(sys.argv) > 2 else "assembler"
        print(json.dumps(build_context(role), ensure_ascii=False, indent=2))
    elif cmd == "tool":
        name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        print(json.dumps(call_tool(name, args), ensure_ascii=False, indent=2))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
