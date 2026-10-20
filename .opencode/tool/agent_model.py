#!/usr/bin/env python3
"""代理模型切换工具（stdlib 零依赖）

背景（WO-0010）：五代理 frontmatter 已不写 model，默认模型 = 项目
.opencode/opencode.json 的 model；会话内 /model 切换即时生效且子代理跟随
父会话模型。本工具解决「差异化/持久」切换：免手改文件。

两种生效语义（show/每次写操作都会提示）：
  · 会话内 /model      → 即时生效，子代理自动跟随（临时切换用这个）
  · 本工具的写操作     → 写入 frontmatter / opencode.json，新会话生效
                        （对已开会话无影响，重启 opencode 后生效）

用法（项目根执行）：
  python3 .opencode/tool/agent_model.py show
  python3 .opencode/tool/agent_model.py set <agent> <provider/model>
      幂等：frontmatter 增/换 model 行（无 → 插在闭合 --- 之前；有 → 整行替换）；
      其余内容（description/mode/正文）逐字节不动
  python3 .opencode/tool/agent_model.py clear <agent>     # 删 model 行回项目默认
  python3 .opencode/tool/agent_model.py default <provider/model>
      改项目 .opencode/opencode.json 的 model 键，其余键与格式不动（保持合法 JSON）
  python3 .opencode/tool/agent_model.py --self-test       # 自检（临时副本，不碰真实文件）

代理名单固定为 UMVP 五代理；agent 名不在名单 / 模型串不合 provider/model
形状 → fail-fast 中文报错（exit 2），绝不写文件。

本模块是模型管理的**唯一真相源**：ticket_web.py「模型」页签直接 import 本模块
的 apply_set / apply_clear / apply_default / status，不做第二套逻辑。
所有路径参数（agent_dir= / config_path=）可被测试与页签后端覆盖（临时副本）。
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parent.parent
DEFAULT_AGENT_DIR = _REPO_ROOT / ".opencode" / "agent"
DEFAULT_CONFIG = _REPO_ROOT / ".opencode" / "opencode.json"

AGENTS = ["orchestrator", "pipe-chain", "web", "agent-builder", "qa"]

# provider/model 形状：两段均非空、无空白无引号无竖线（防破坏 YAML/JSON）。
# 例：zai-coding-plan/glm-5.2、anthropic/claude-sonnet-4-5、openai/gpt-4o。
_MODEL_RE_STR = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:/+-]*"
MODEL_RE = re.compile(rf"^{_MODEL_RE_STR}$")

# frontmatter 内的 model 行（行首无缩进；models: 等其它键不会误匹配）
_MODEL_LINE_RE = re.compile(r"^model:\s*(.*)$")

# opencode.json 内的 "model" 键行（用于字节级替换，只动这一行）
_JSON_MODEL_RE = re.compile(r'^(\s*"model"\s*:\s*")([^"\n]*)("[ \t]*(?:,.*)?$)')

HINT = "改动新会话生效（重启 opencode 后的新会话）；会话内临时切换请用 /model"


class AgentModelError(Exception):
    """输入/文件非法（中文消息）；CLI 捕获后 ✗ 前缀打 stderr 并 exit 2。"""


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate_agent(agent):
    if agent not in AGENTS:
        raise AgentModelError(f"未知代理: {agent!r}（应为 {'/'.join(AGENTS)}）")
    return agent


def validate_model(model):
    if not isinstance(model, str) or not MODEL_RE.match(model):
        raise AgentModelError(
            f"模型串非法: {model!r}（应为 provider/model 形状，"
            f"如 zai-coding-plan/glm-5.2；不含空白/引号/竖线）")
    return model


def agent_path(agent, agent_dir=DEFAULT_AGENT_DIR):
    return Path(agent_dir) / f"{agent}.md"


# --------------------------------------------------------------------------- #
# frontmatter 解析与写盘（行级，逐字节保留其余内容）
# --------------------------------------------------------------------------- #
def _load_lines(path):
    p = Path(path)
    if not p.exists():
        raise AgentModelError(f"代理文件不存在: {p}")
    return p.read_text(encoding="utf-8").splitlines(keepends=True)


def _fm_bounds(lines, path):
    """→ (frontmatter 内容行范围 start, end)；end 为闭合 --- 的行号。

    首行必须恰为 ---，随后第一个恰为 --- 的行闭合；都按去行尾+去空白比较，
    缩进的 --- 不算闭合（YAML 语义）。
    """
    if not lines or lines[0].rstrip("\r\n").strip() != "---":
        raise AgentModelError(f"{Path(path).name} 缺少 frontmatter（首行应为 ---）")
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n").strip() == "---":
            return 1, i
    raise AgentModelError(f"{Path(path).name} frontmatter 未闭合（缺第二个 ---）")


def _model_line_idxs(lines, path):
    s, e = _fm_bounds(lines, path)
    idxs = [i for i in range(s, e) if _MODEL_LINE_RE.match(lines[i].rstrip("\r\n"))]
    if len(idxs) > 1:
        raise AgentModelError(f"{Path(path).name} frontmatter 有 {len(idxs)} 个 model 行，请先手工修整")
    return idxs


def _strip_quotes(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v


def _newline_of(lines):
    for l in lines:
        if l.endswith("\r\n"):
            return "\r\n"
    return "\n"


def get_model(agent, agent_dir=DEFAULT_AGENT_DIR):
    """读代理 frontmatter 的 model 值 → str 或 None（未设置/值为空）。"""
    validate_agent(agent)
    path = agent_path(agent, agent_dir)
    lines = _load_lines(path)
    idxs = _model_line_idxs(lines, path)
    if not idxs:
        return None
    return _strip_quotes(_MODEL_LINE_RE.match(lines[idxs[0]].rstrip("\r\n")).group(1)) or None


def _write_atomic(path, text):
    """临时文件 + os.replace 原子替换（保留原文件权限；新建 0644）。"""
    p = Path(path)
    d = str(p.parent) or "."
    fd, tmp = tempfile.mkstemp(prefix="." + p.name + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            mode = os.stat(p).st_mode & 0o7777
        except OSError:
            mode = 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_set(agent, model, agent_dir=DEFAULT_AGENT_DIR):
    """设置代理模型（幂等）。

    frontmatter 无 model 行 → 插在闭合 --- 之前；有 → 整行替换为 `model: <值>`；
    已同值 → 不写盘（changed=False）。其余行逐字节不动。
    → {"agent", "model", "changed", "hint"}
    """
    validate_agent(agent)
    validate_model(model)
    path = agent_path(agent, agent_dir)
    lines = _load_lines(path)
    idxs = _model_line_idxs(lines, path)
    if idxs:
        cur = _strip_quotes(_MODEL_LINE_RE.match(lines[idxs[0]].rstrip("\r\n")).group(1))
        if cur == model:
            return {"agent": agent, "model": model, "changed": False, "hint": HINT}
        lines[idxs[0]] = f"model: {model}{_newline_of(lines)}"
    else:
        _, close = _fm_bounds(lines, path)
        lines.insert(close, f"model: {model}{_newline_of(lines)}")
    _write_atomic(path, "".join(lines))
    return {"agent": agent, "model": model, "changed": True, "hint": HINT}


def apply_clear(agent, agent_dir=DEFAULT_AGENT_DIR):
    """清除代理模型（删 frontmatter 的 model 行，回项目默认；幂等）。"""
    validate_agent(agent)
    path = agent_path(agent, agent_dir)
    lines = _load_lines(path)
    idxs = _model_line_idxs(lines, path)
    if not idxs:
        return {"agent": agent, "model": None, "changed": False, "hint": HINT}
    del lines[idxs[0]]
    _write_atomic(path, "".join(lines))
    return {"agent": agent, "model": None, "changed": True, "hint": HINT}


# --------------------------------------------------------------------------- #
# 项目默认模型（.opencode/opencode.json，只动 model 键且保持合法 JSON）
# --------------------------------------------------------------------------- #
def get_default(config_path=DEFAULT_CONFIG):
    """→ 项目默认模型 str；文件缺失 → None。"""
    p = Path(config_path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise AgentModelError(f"opencode.json 不是合法 JSON: {p}（{e}）")
    if not isinstance(obj, dict):
        raise AgentModelError(f"opencode.json 顶层应为 JSON 对象: {p}")
    return obj.get("model")


def _replace_json_model_line(text, new):
    """字节级只换 "model" 键行的值 → 新文本；找不到唯一目标行 → None（走兜底）。"""
    lines = text.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if _JSON_MODEL_RE.match(l.rstrip("\r\n"))]
    if len(hits) != 1:
        return None
    i = hits[0]
    raw = lines[i]
    eol = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
    m = _JSON_MODEL_RE.match(raw.rstrip("\r\n"))
    lines[i] = m.group(1) + new + m.group(3) + eol
    return "".join(lines)


def apply_default(model, config_path=DEFAULT_CONFIG):
    """设置项目默认模型（幂等）。

    文件存在：优先字节级替换 "model" 键行（其余逐字节不动）；无法唯一定位该行
    （如紧凑单行 JSON）则 json 往返重排（语义上仍只改 model 键）。写前双重校验
    结果是合法 JSON 且 model == 目标值，不满足绝不落盘。文件缺失 → 按项目标准
    布局（$schema + model）新建。
    → {"model", "changed", "hint"}
    """
    validate_model(model)
    p = Path(config_path)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        try:
            obj = json.loads(text)
        except ValueError as e:
            raise AgentModelError(f"opencode.json 不是合法 JSON: {p}（{e}）")
        if not isinstance(obj, dict):
            raise AgentModelError(f"opencode.json 顶层应为 JSON 对象: {p}")
        if obj.get("model") == model:
            return {"model": model, "changed": False, "hint": HINT}
        new_text = _replace_json_model_line(text, model)
        if new_text is None or not _json_model_ok(new_text, model):
            obj["model"] = model
            new_text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    else:
        new_text = json.dumps({"$schema": "https://opencode.ai/config.json",
                               "model": model}, ensure_ascii=False, indent=2) + "\n"
    if not _json_model_ok(new_text, model):
        raise AgentModelError("内部错误：改写后的 opencode.json 校验不过，已放弃写入")
    _write_atomic(p, new_text)
    return {"model": model, "changed": True, "hint": HINT}


def _json_model_ok(text, model):
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("model") == model


# --------------------------------------------------------------------------- #
# 现状汇总（show / ticketd 模型页签共用）
# --------------------------------------------------------------------------- #
def status(agent_dir=DEFAULT_AGENT_DIR, config_path=DEFAULT_CONFIG):
    """→ {"default": {...}, "agents": [...], "agents_dir", "hint"}（只读）。"""
    cfgp = Path(config_path)
    try:
        default_model = get_default(cfgp)
        cfg_err = None
    except AgentModelError as e:
        default_model = None  # 解析失败按未设置展示，错误文案带回前端
        cfg_err = str(e)
    agents = []
    for a in AGENTS:
        p = agent_path(a, agent_dir)
        item = {"name": a, "file": str(p), "exists": p.exists(),
                "model": None, "error": None}
        if p.exists():
            try:
                item["model"] = get_model(a, agent_dir)
            except AgentModelError as e:
                item["error"] = str(e)
        agents.append(item)
    return {"default": {"model": default_model, "file": str(cfgp),
                        "exists": cfgp.exists(), "error": cfg_err},
            "agents": agents, "agents_dir": str(Path(agent_dir)),
            "hint": HINT}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_show(agent_dir, config_path):
    st = status(agent_dir, config_path)
    print("代理模型现状（frontmatter model ?? 项目默认）")
    d = st["default"]
    for a in st["agents"]:
        if not a["exists"]:
            print(f"  {a['name']:<14} ✗ 文件缺失: {a['file']}")
        elif a["error"]:
            print(f"  {a['name']:<14} ✗ {a['error']}")
        elif a["model"]:
            print(f"  {a['name']:<14} {a['model']}（独立设置）")
        else:
            print(f"  {a['name']:<14} {d['model'] or '（未设置）'}（项目默认）")
    if d["error"]:
        print(f"项目默认: ✗ {d['error']}")
    else:
        print(f"项目默认: {d['model'] or '（未设置）'}（{d['file']}）")
    print("生效语义：会话内 /model 即时切换且子代理跟随；本工具改动 → 新会话生效")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="代理模型切换（frontmatter / 项目默认；改动新会话生效）")
    ap.add_argument("--agent-dir", type=Path, default=None,
                    help="代理目录（默认 <仓库>/.opencode/agent；测试指向临时副本用）")
    ap.add_argument("--config", type=Path, default=None,
                    help="opencode.json 路径（默认 <仓库>/.opencode/opencode.json；测试用）")
    ap.add_argument("--self-test", action="store_true",
                    help="自检（临时副本上跑断言，真实文件只读）")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("show", help="查看五代理与项目默认模型现状")
    p = sub.add_parser("set", help="设置代理模型（幂等改 frontmatter）")
    p.add_argument("agent"); p.add_argument("model")
    p = sub.add_parser("clear", help="清除代理模型（回项目默认）")
    p.add_argument("agent")
    p = sub.add_parser("default", help="设置项目默认模型（opencode.json）")
    p.add_argument("model")
    a = ap.parse_args(argv)

    if a.self_test:
        sys.exit(run_self_test())

    agent_dir = a.agent_dir or DEFAULT_AGENT_DIR
    config_path = a.config or DEFAULT_CONFIG
    try:
        if a.cmd == "show":
            cmd_show(agent_dir, config_path)
        elif a.cmd == "set":
            r = apply_set(a.agent, a.model, agent_dir=agent_dir)
            print(f"✓ {r['agent']} → {r['model']}"
                  f"（{'已写入' if r['changed'] else '已同值，未改动'}）")
        elif a.cmd == "clear":
            r = apply_clear(a.agent, agent_dir=agent_dir)
            print(f"✓ {r['agent']} 模型已{'清除（回项目默认）' if r['changed'] else '是默认，未改动'}")
        elif a.cmd == "default":
            r = apply_default(a.model, config_path=config_path)
            print(f"✓ 项目默认 → {r['model']}"
                  f"（{'已写入' if r['changed'] else '已同值，未改动'}）")
        else:
            ap.print_usage()
            print("✗ 缺少子命令（show/set/clear/default）", file=sys.stderr)
            sys.exit(2)
        if a.cmd in ("set", "clear", "default"):
            print(f"提示：{HINT}")
    except AgentModelError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------- #
# 自检（临时副本；真实 agent 文件与 opencode.json 只读，前后 md5 断言）
# --------------------------------------------------------------------------- #
def run_self_test():
    import hashlib
    import shutil
    import subprocess

    passed = [0]

    def check(cond, msg):
        if cond:
            passed[0] += 1
            print(f"  ✓ {msg}")
        else:
            raise AssertionError(f"自检失败: {msg}")

    def expect_error(fn, substr, msg):
        try:
            fn()
        except AgentModelError as e:
            check(substr in str(e), msg)
            return
        raise AssertionError(f"自检失败: {msg}（未抛出 AgentModelError）")

    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    # 真实文件基线（自检全程只读）
    real_agents = {n: agent_path(n) for n in AGENTS}
    real_cfg = Path(DEFAULT_CONFIG)
    base_real = {n: md5(p) for n, p in real_agents.items()}
    base_real_cfg = md5(real_cfg)
    check(all(p.exists() for p in real_agents.values()),
          "五份真实代理文件存在（基线只读）")

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="umvp-agent-model-selftest-"))
    try:
        adir = tmpdir / "agents"
        adir.mkdir()
        M1 = "zai-coding-plan/glm-5.2"
        M2 = "exa/test-model-2"
        MD1 = "zai-coding-plan/test-default-1"
        MD2 = "exa/test-default-2"

        # 造五份合成代理文件（覆盖：无 model / 已有 model / 正文含 model 陷阱 /
        # description 值含 model 字样 / 文件末行无换行）
        def w(name, text):
            (adir / f"{name}.md").write_text(text, encoding="utf-8", newline="")

        w("orchestrator", "---\ndescription: 主代理。\nmode: primary\n---\n\n正文 A 第一行\n正文 A 第二行\n")
        w("pipe-chain", f"---\ndescription: 模块代理。\nmode: all\nmodel: old/provider-m1\n---\n\n正文 B\n")
        w("web", "---\ndescription: web 代理。\nmode: all\n---\n\n正文 C\nmodel: body-trap/x\n正文 C 尾\n")
        w("agent-builder", "---\ndescription: 构建 agent 代理（正题含 model: 字样的描述行）。\nmode: all\n---\n\n正文 D\n")
        w("qa", "---\ndescription: 测试代理。\nmode: all\n---\n\n正文 E（末行无换行）")

        # 1. 读取现状
        check(get_model("pipe-chain", adir) == "old/provider-m1", "get_model 读出已有 model 行")
        check(get_model("orchestrator", adir) is None, "get_model 无 model 行 → None")
        check(get_model("web", adir) is None, "正文里的 model: 陷阱行不参与解析")

        # 2. set：新增行（插在闭合 --- 之前），其余逐字节不动
        qa_orig = (adir / "qa.md").read_bytes()
        r = apply_set("qa", M1, agent_dir=adir)
        check(r["changed"] and r["model"] == M1 and "新会话生效" in r["hint"],
              "apply_set 返回 changed=True 且提示新会话生效")
        qa_new = (adir / "qa.md").read_bytes()
        lines = qa_orig.splitlines(keepends=True)
        close = next(i for i in range(1, len(lines))
                     if lines[i].rstrip(b"\r\n").strip() == b"---")
        expect_insert = b"".join(lines[:close] + [f"model: {M1}\n".encode()] + lines[close:])
        check(qa_new == expect_insert, "set(新增) 只在闭合 --- 前插入 model 行，其余逐字节一致")
        check(qa_new.endswith("正文 E（末行无换行）".encode()),
              "末行无换行的文件不被补换行")

        # 3. set 幂等（同值不写盘）
        mtime_before = (adir / "qa.md").stat().st_mtime_ns
        r = apply_set("qa", M1, agent_dir=adir)
        check(r["changed"] is False and (adir / "qa.md").read_bytes() == qa_new
              and (adir / "qa.md").stat().st_mtime_ns == mtime_before,
              "set 同值幂等（changed=False，文件未动）")

        # 4. set 换值：仅该行被替换
        r = apply_set("qa", M2, agent_dir=adir)
        check(r["changed"] and (adir / "qa.md").read_bytes()
              == qa_new.replace(f"model: {M1}\n".encode(), f"model: {M2}\n".encode()),
              "set(换值) 仅替换 model 行")
        check(get_model("qa", adir) == M2, "get_model 读回新值")

        # 5. clear：字节级还原（往返）
        r = apply_clear("qa", agent_dir=adir)
        check(r["changed"] and (adir / "qa.md").read_bytes() == qa_orig,
              "clear 后文件与原始逐字节一致（往返无损）")
        r = apply_clear("qa", agent_dir=adir)
        check(r["changed"] is False and (adir / "qa.md").read_bytes() == qa_orig,
              "clear 幂等（未设置时 changed=False）")

        # 6. set/clear 于已有 model 行的文件（pipe-chain）
        pc_orig = (adir / "pipe-chain.md").read_bytes()
        apply_set("pipe-chain", M2, agent_dir=adir)
        check(get_model("pipe-chain", adir) == M2
              and (adir / "pipe-chain.md").read_bytes()
              == pc_orig.replace(b"model: old/provider-m1\n", f"model: {M2}\n".encode()),
              "set(替换已有行) 仅换该行")
        apply_clear("pipe-chain", agent_dir=adir)
        check((adir / "pipe-chain.md").read_bytes()
              == pc_orig.replace(b"model: old/provider-m1\n", b""),
              "clear(已有行) 删行后其余逐字节一致")

        # 7. 正文陷阱：set/clear 不碰正文里的 model: 行
        web_orig = (adir / "web.md").read_bytes()
        apply_set("web", M1, agent_dir=adir)
        check(get_model("web", adir) == M1 and b"model: body-trap/x\n" in (adir / "web.md").read_bytes(),
              "正文中的 model 陷阱行不受 frontmatter 写操作影响")
        apply_clear("web", agent_dir=adir)
        check((adir / "web.md").read_bytes() == web_orig, "clear 后正文陷阱原样保留")

        # 8. 坏输入拒绝（fail-fast，文件不变）
        for bad_agent in ("boss", "", "qa.md"):
            expect_error(lambda b=bad_agent: apply_set(b, M1, agent_dir=adir),
                         "未知代理", f"坏代理名 {bad_agent!r} 报错")
        for bad_model in ("glm-5.2", "/x", "a/", "a/b c", 'a/"b', "a/b|c", "", "zai-coding-plan/glm 5"):
            expect_error(lambda b=bad_model: apply_set("qa", b, agent_dir=adir),
                         "模型串非法", f"坏模型串 {bad_model!r} 报错")
        expect_error(lambda: apply_default("no-slash", config_path=tmpdir / "c.json"),
                     "模型串非法", "default 坏模型串报错")
        snapshot = {n: (adir / f"{n}.md").read_bytes() for n in AGENTS}
        check(all((adir / f"{n}.md").read_bytes() == b for n, b in snapshot.items()),
              "所有被拒操作均未写文件")

        # 9. frontmatter 异常文件
        orch = (adir / "orchestrator.md").read_bytes()
        (adir / "orchestrator.md").write_text("没有 frontmatter 的文件\n", encoding="utf-8")
        expect_error(lambda: apply_set("orchestrator", M1, agent_dir=adir),
                     "frontmatter", "无 frontmatter 报错")
        (adir / "orchestrator.md").write_bytes(
            b"---\ndescription: x\nmodel: a/b\nmodel: c/d\n---\n")
        expect_error(lambda: apply_set("orchestrator", M1, agent_dir=adir),
                     "model 行", "多个 model 行报错")
        (adir / "orchestrator.md").write_bytes(orch)

        # 10. 代理文件缺失
        adir2 = tmpdir / "agents-missing"
        adir2.mkdir()
        for n in AGENTS[:-1]:
            shutil.copyfile(adir / f"{n}.md", adir2 / f"{n}.md")
        expect_error(lambda: apply_set(AGENTS[-1], M1, agent_dir=adir2),
                     "不存在", "代理文件缺失报错")
        st2 = status(agent_dir=adir2, config_path=real_cfg)
        check(next(a for a in st2["agents"] if a["name"] == AGENTS[-1])["exists"] is False,
              "status 标记缺失文件 exists=False")

        # 11. default：标准 JSON（字节级只动 model 行）
        cfg = tmpdir / "opencode.json"
        shutil.copyfile(real_cfg, cfg)
        cfg_orig = cfg.read_bytes()
        cur_default = json.loads(cfg_orig)["model"]
        r = apply_default(MD1, config_path=cfg)
        check(r["changed"] and json.loads(cfg.read_text())["model"] == MD1,
              "default 写入后 model 为新值")
        expect_d = cfg_orig.replace(f'"{cur_default}"'.encode(), f'"{MD1}"'.encode())
        check(cfg.read_bytes() == expect_d, "default 只动 model 值，其余逐字节一致")
        obj = json.loads(cfg.read_text())
        check(obj.get("$schema") == json.loads(cfg_orig).get("$schema"),
              "default 不动其它键（$schema 保留）")
        mtime_cfg = cfg.stat().st_mtime_ns
        r = apply_default(MD1, config_path=cfg)
        check(r["changed"] is False and cfg.stat().st_mtime_ns == mtime_cfg,
              "default 同值幂等（不写盘）")

        # 12. default：紧凑单行 JSON（兜底重排路径，语义只改 model）
        compact = tmpdir / "compact.json"
        compact.write_text('{"$schema": "https://opencode.ai/config.json", '
                           '"model": "a/b", "theme": "dark"}', encoding="utf-8")
        apply_default(MD2, config_path=compact)
        obj = json.loads(compact.read_text())
        check(obj["model"] == MD2 and obj["theme"] == "dark" and "$schema" in obj,
              "default 紧凑 JSON 走重排兜底，其它键保留")

        # 13. default：坏 JSON 拒绝 / 缺失新建
        broken = tmpdir / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        expect_error(lambda: apply_default(MD1, config_path=broken),
                     "合法 JSON", "default 目标文件坏 JSON 报错")
        fresh = tmpdir / "fresh.json"
        apply_default(MD1, config_path=fresh)
        obj = json.loads(fresh.read_text())
        check(obj["model"] == MD1 and "$schema" in obj,
              "default 文件缺失时按标准布局新建")

        # 14. status 汇总
        st = status(agent_dir=adir, config_path=cfg)
        check([a["name"] for a in st["agents"]] == AGENTS, "status 代理名单顺序固定")
        check(st["default"]["model"] == MD1 and "新会话生效" in st["hint"],
              "status 默认模型与提示语")

        # 15. CLI（临时副本上跑；--agent-dir/--config 显式指定）
        def cli(*args):
            return subprocess.run(
                [sys.executable, str(_TOOL_DIR / "agent_model.py"),
                 "--agent-dir", str(adir), "--config", str(cfg), *args],
                capture_output=True, text=True)

        cp = cli("show")
        check(cp.returncode == 0 and all(n in cp.stdout for n in AGENTS)
              and "项目默认" in cp.stdout, "CLI show 列五代理与默认")
        cp = cli("set", "qa", M1)
        check(cp.returncode == 0 and "新会话生效" in cp.stdout and "qa" in cp.stdout,
              "CLI set 成功且输出含「新会话生效」")
        check(get_model("qa", adir) == M1, "CLI set 落盘生效")
        cp = cli("clear", "qa")
        check(cp.returncode == 0 and "新会话生效" in cp.stdout
              and get_model("qa", adir) is None, "CLI clear 成功还原默认")
        cp = cli("default", MD2)
        check(cp.returncode == 0 and "新会话生效" in cp.stdout
              and json.loads(cfg.read_text())["model"] == MD2, "CLI default 写入成功")
        cp = cli("set", "boss", M1)
        check(cp.returncode == 2 and "✗" in cp.stderr and "未知代理" in cp.stderr,
              "CLI 坏代理名 rc=2 中文报错")
        cp = cli("set", "qa", "no-slash")
        check(cp.returncode == 2 and "模型串非法" in cp.stderr, "CLI 坏模型串 rc=2")
        cp = cli()
        check(cp.returncode == 2, "CLI 无子命令 rc=2")

        # 16. 真实文件全程未被修改
        check(all(md5(p) == base_real[n] for n, p in real_agents.items()),
              "五份真实代理文件未被修改（md5 一致）")
        check(md5(real_cfg) == base_real_cfg, "真实 opencode.json 未被修改")

        print(f"\n自检通过：{passed[0]} 项断言；真实文件只读（opencode.json md5={base_real_cfg}）")
        return 0
    except AssertionError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        print(f"（已通过 {passed[0]} 项）", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
