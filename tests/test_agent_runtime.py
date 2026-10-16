# -*- coding: utf-8 -*-
"""
挂载 agent 运行时测试（纯逻辑，不联网、不加载模型）：

  1. 两角色提示词：可读、去 frontmatter、两角色确有差异
  2. 上下文披露 build_context：system/tools/knowledge 齐备，角色差异注入
  3. 知识库索引：四层文件齐、路径越界被拒
  4. 工具：read_doc / list_modules / validate_spec / estimate / list·get_pipeline
  5. LLM 接入层：工具 schema 对齐 + function-calling 循环（monkeypatch _chat，离线）

运行: conda run -n ai python tests/test_agent_runtime.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web_lab"))
sys.path.insert(0, str(ROOT / "umvp"))

import agent  # noqa: E402
import agent_llm  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


VALID_SPEC = {"name": "t", "stages": [
    {"module_id": "frame_manager", "params": {}},
    {"module_id": "det_filter", "params": {}},
]}


def main() -> None:
    # ---- 1. 两角色提示词 ----
    pa = agent.read_agent_prompt("architect")
    ps = agent.read_agent_prompt("assembler")
    check("architect 提示词可读且非空", bool(pa.strip()) and "---" not in pa.splitlines()[0])
    check("assembler 提示词可读且非空", bool(ps.strip()))
    check("两角色提示词确有差异", pa != ps)
    check("architect 定位为业务设计", "业务" in pa and "语义模块链条" in pa)
    check("architect 自包含（§1-§8）",
          "## 1. 项目介绍" in pa and "## 8. 可用工具" in pa and "## 3. 概念介绍" in pa)
    check("architect 内联模块地图", "图像旁路" in pa and "可改什么" in pa)
    check("architect 不再关联技能库", "knowledge/" not in pa and ".claude/skills" not in pa)
    check("architect 列可用工具", "list_modules" in pa and "get_pipeline" in pa)
    check("assembler 侧重参数逻辑与红线", "参数" in ps and "红线" in ps)
    check("assembler 自包含（§1-§8）",
          "## 1. 项目介绍" in ps and "## 8. 可用工具" in ps and "## 3. 链路壳" in ps)
    check("assembler 指向自有技术文档",
          ".claude/agents/assembler/01-shell-standard.md" in ps)
    check("assembler 不再关联技能库", "knowledge/" not in ps and ".claude/skills" not in ps)

    # ---- 2. 上下文披露 ----
    ca = agent.build_context("architect")
    cs = agent.build_context("assembler")
    for name, ctx in (("architect", ca), ("assembler", cs)):
        check(f"{name} 上下文三要素齐备", {"role", "system", "tools"} <= set(ctx))
        check(f"{name} 不再含知识索引字段", "knowledge" not in ctx)
    check("两角色 system 不同", ca["system"] != cs["system"])
    check("architect system = 自包含提示词", ca["system"] == agent.read_agent_prompt("architect"))
    check("assembler system = 自包含提示词", cs["system"] == agent.read_agent_prompt("assembler"))

    # ---- 3. 文档索引（项目内，无技能库） ----
    di = agent.docs_index()
    files = di["files"]
    check("文档索引含项目模块文档", any(f.startswith("doc/modules/") for f in files))
    check("文档索引含组装工技术文档",
          any(f.startswith(".claude/agents/assembler/") for f in files))
    check("文档索引无技能库路径", not any("skills/pipeline-assembly" in f for f in files))
    try:
        agent.read_doc_file("../../etc/passwd")
        escaped = False
    except (ValueError, FileNotFoundError):
        escaped = True
    check("路径越界被拒", escaped)
    try:
        agent.read_doc_file("web_lab/agent.py")   # 非 .md 拒绝
        non_md = False
    except (ValueError, FileNotFoundError):
        non_md = True
    check("非 Markdown 被拒", non_md)

    # ---- 4. 工具 ----
    rd = agent.call_tool("read_doc", {})
    check("read_doc 空参返回索引", "index" in rd and "files" in rd)
    rd2 = agent.call_tool("read_doc", {"path": ".claude/agents/assembler/01-shell-standard.md"})
    check("read_doc 取单篇非空", bool(rd2.get("content")) and "链路壳标准" in rd2.get("content", ""))

    mods = agent.call_tool("list_modules", {})["modules"]
    check("list_modules 覆盖保留链路关键模块",
          {"frame_manager", "yolo", "det_filter", "seq_filter", "target_crop"} <= set(mods))
    check("list_modules 参数带类型与默认", all(
        "type" in p and "default" in p
        for m in mods.values() for p in m.get("params", {}).values()))

    ok = agent.call_tool("validate_spec", {"spec": VALID_SPEC})
    check("validate_spec 合法 spec 通过", ok.get("ok") is True, str(ok.get("errors")))
    bad = agent.call_tool("validate_spec", {"spec": {"stages": [{"module_id": "yolo"}]}})
    check("validate_spec 缺 frame_manager 报错", bad.get("ok") is False and bad.get("errors"))

    est = agent.call_tool("estimate", {"spec": VALID_SPEC, "streams": 2})
    check("estimate 返回结构", isinstance(est, dict) and "latency_per_frame_ms" in est)

    lp = agent.call_tool("list_pipelines", {})
    check("list_pipelines 容错返回结构", isinstance(lp, dict) and "pipelines" in lp)
    gp = agent.call_tool("get_pipeline", {})
    check("get_pipeline 缺参报错", gp.get("ok") is False)
    unknown = agent.call_tool("nope", {})
    check("未知工具报错", unknown.get("ok") is False)

    # ---- 5. LLM 接入层 ----
    check("工具 schema 与工具表对齐",
          [t["name"] for t in agent.tool_schemas()] ==
          [t["function"]["name"] for t in agent_llm._openai_tools()])

    # 无 key 时 _chat 明确报错
    env_backup = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "SILICONFLOW_API_KEY")}
    try:
        try:
            agent_llm._chat([{"role": "user", "content": "hi"}], None)
            raised = False
        except RuntimeError:
            raised = True
        check("无 key 时 _chat 拒绝", raised)
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v

    # function-calling 循环（离线：脚本化 _chat）
    scripts = [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "list_modules", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "已拿到模块清单，完成。"}}]},
    ]
    real_chat = agent_llm._chat
    agent_llm._chat = lambda *a, **k: scripts.pop(0)
    events = []
    try:
        res = agent_llm.run("assembler", "有哪些模块？", on_event=events.append)
    finally:
        agent_llm._chat = real_chat
    check("循环成功收敛", res["ok"] is True and "完成" in res["text"])
    check("循环执行了工具调用", [c["name"] for c in res["tool_calls"]] == ["list_modules"])
    types = [e["type"] for e in events]
    check("事件流含 tool_call/tool_result/text",
          {"tool_call", "tool_result", "text"} <= set(types))

    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
