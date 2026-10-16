# -*- coding: utf-8 -*-
"""
挂载 agent 的 LLM 接入层：OpenAI 兼容 function-calling 循环（零新增依赖）。

把 web_lab/agent.py 披露的「角色提示词 + 知识 + 工具」交给任意 OpenAI 兼容端点，
在循环里执行模型发起的工具调用，直到模型给出最终答复。

    build_context(role) → system + tools + knowledge   （agent.py）
    run(role, message)  → 多轮：LLM 调用 ↔ 工具调用 ↔ 追加结果 → 最终文本

环境变量（缺省值按 SiliconFlow）:
    LLM_BASE_URL   默认 https://api.siliconflow.cn/v1
    LLM_API_KEY    优先；兼容 SILICONFLOW_API_KEY
    LLM_MODEL      默认 Qwen/Qwen2.5-32B-Instruct（需支持 function calling）

CLI（自测）:
    conda run -n ai python web_lab/agent_llm.py --demo build
    conda run -n ai python web_lab/agent_llm.py --ask "系统能识别什么？" --role architect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web_lab"))
sys.path.insert(0, str(ROOT / "umvp"))

import agent  # noqa: E402  （工具层 + 上下文披露）

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
TOOL_RESULT_LIMIT = 12000     # 单个工具结果回灌给模型的字符上限（防爆上下文）
MAX_STEPS = 8                 # 单次提问的工具调用轮数上限

# 演示用例（--demo）
DEMOS = {
    "ask": ("architect", "这套系统能识别什么？一句话说清。"),
    "build": ("architect", "我要在商场门口检测有没有人打架，单路 1080p，8G 显存，"
                           "打架了要报警。帮我设计一条链路。"),
    "optimize": ("architect", "已有链路里人像图分辨率太低了，帮我看看哪里要改。"),
}


# ---------------- HTTP（stdlib） ----------------
def _api_key() -> str:
    return os.environ.get("LLM_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or ""


def _chat(messages: list, tools: Optional[list] = None, *,
          base_url: Optional[str] = None, api_key: Optional[str] = None,
          model: Optional[str] = None, timeout: int = 120) -> dict:
    """调用 OpenAI 兼容 /chat/completions，返回完整响应 dict。"""
    base = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or _api_key()
    if not key:
        raise RuntimeError("未配置 LLM_API_KEY / SILICONFLOW_API_KEY（导出环境变量后再试）")
    body: dict = {
        "model": model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(f"{base}/chat/completions", data=data, method="POST",
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {key}"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:500]
        raise RuntimeError(f"LLM 端点 HTTP {e.code}: {detail}") from e


def _msg_of(resp: dict) -> dict:
    return (resp.get("choices") or [{}])[0].get("message") or {}


def _openai_tools() -> list:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in agent.tool_schemas()]


def _clip(text: str) -> str:
    if len(text) <= TOOL_RESULT_LIMIT:
        return text
    return text[:TOOL_RESULT_LIMIT] + f"\n…（截断，原长 {len(text)} 字符）"


# ---------------- function-calling 循环 ----------------
def run(role: str = "assembler", message: str = "", *, history: Optional[list] = None,
        base_url: Optional[str] = None, api_key: Optional[str] = None,
        model: Optional[str] = None, max_steps: int = MAX_STEPS,
        on_event: Optional[Callable[[dict], None]] = None, **_ignored) -> dict:
    """执行一次多轮问答，返回 {ok, role, text, steps, tool_calls, messages}。

    on_event(event) 可选：宿主（如 web SSE）据此流式推送
    {"type":"text"|"tool_call"|"tool_result"|"error", ...}。
    """
    def emit(ev: dict) -> None:
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    ctx = agent.build_context(role)
    messages: list = [{"role": "system", "content": ctx["system"]}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    tools = _openai_tools()
    calls: list = []
    for step in range(max(1, max_steps)):
        try:
            resp = _chat(messages, tools, base_url=base_url, api_key=api_key, model=model)
        except Exception as e:
            emit({"type": "error", "error": f"{type(e).__name__}: {e}"})
            return {"ok": False, "role": role, "text": "",
                    "steps": step, "tool_calls": calls, "messages": messages,
                    "error": f"{type(e).__name__}: {e}"}
        msg = _msg_of(resp)
        tool_calls = msg.get("tool_calls") or []
        # 记录 assistant 轮（含 tool_calls 时也要原样回填，供下一轮关联）
        messages.append(msg if tool_calls else
                        {"role": "assistant", "content": msg.get("content") or ""})
        if not tool_calls:
            text = msg.get("content") or ""
            emit({"type": "text", "text": text})
            return {"ok": True, "role": role, "text": text,
                    "steps": step + 1, "tool_calls": calls, "messages": messages}

        # 顺序执行工具调用（一次一个，符合"一次仅使用一个工具"约定）
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name = fn.get("name") or ""
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                args = {}
            emit({"type": "tool_call", "name": name, "arguments": args})
            result = agent.call_tool(name, args)
            calls.append({"name": name, "arguments": args})
            emit({"type": "tool_result", "name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or name,
                             "content": _clip(json.dumps(result, ensure_ascii=False))})

    emit({"type": "error", "error": f"超过工具调用轮数上限 {max_steps}"})
    return {"ok": False, "role": role, "text": "",
            "steps": max_steps, "tool_calls": calls, "messages": messages,
            "error": f"超过工具调用轮数上限 {max_steps}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="挂载 agent：LLM function-calling 循环")
    ap.add_argument("--ask", help="用户提问（自然语言）")
    ap.add_argument("--demo", choices=sorted(DEMOS), help="跑内置演示用例")
    ap.add_argument("--role", default=None, choices=["architect", "assembler"],
                    help="角色（默认 architect；--demo 用其自带角色）")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--verbose", action="store_true", help="打印工具调用过程")
    args = ap.parse_args()

    if args.demo:
        role, message = DEMOS[args.demo]
        role = args.role or role
    elif args.ask:
        role, message = (args.role or "architect"), args.ask
    else:
        ap.print_help()
        return 2

    def on_event(ev: dict) -> None:
        if not args.verbose:
            return
        t = ev.get("type")
        if t == "tool_call":
            print(f"[tool_call] {ev['name']} {json.dumps(ev['arguments'], ensure_ascii=False)}",
                  file=sys.stderr)
        elif t == "tool_result":
            ok = ev["result"].get("ok", True)
            print(f"[tool_result] {ev['name']} ok={ok}", file=sys.stderr)

    if not _api_key():
        print("（演示需要 API key）请先导出 LLM_API_KEY 或 SILICONFLOW_API_KEY 后重试。",
              file=sys.stderr)
        print(f"提示：LLM_BASE_URL 默认 {DEFAULT_BASE_URL}，LLM_MODEL 默认 {DEFAULT_MODEL}。",
              file=sys.stderr)
        return 3

    res = run(role, message, base_url=args.base_url, model=args.model,
              max_steps=args.max_steps, on_event=on_event)
    print(f"\n=== role={role} ok={res['ok']} steps={res['steps']} "
          f"tool_calls={[c['name'] for c in res['tool_calls']]} ===")
    print(res.get("text") or res.get("error") or "")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
