#!/usr/bin/env python3
"""工单看板 · 本机可视化可写页面（stdlib 零依赖）

方案 A：独立小工具，不并入 web_lab（不依赖 MySQL / conda ai / 视频目录）。
仅监听本机（默认 127.0.0.1:8010），不做鉴权。

启动：
  python3 .opencode/tool/ticket_web.py                # → http://127.0.0.1:8010/
  python3 .opencode/tool/ticket_web.py --port 8010 --file .opencode/memory/tickets.md
  python3 .opencode/tool/ticket_web.py --config ~/.config/ticketd/umvp.json
  python3 .opencode/tool/ticket_web.py --name umvp    # 查 ~/.config/ticketd/umvp.json
  python3 .opencode/tool/ticket_web.py --self-test    # 端到端自检（临时副本，秒级）

多项目复用 · 配置来源优先级（从高到低）：
  CLI 参数（--file/--host/--port/--title/--config/--name/--inbox-file/--deployment-file）
  > 环境变量（TICKET_CONFIG，兼容 TICKETD_CONFIG）> 配置文件
  （~/.config/ticketd/<name>.json，尊重 XDG_CONFIG_HOME）> 内置默认（UMVP）。
  配置字段：name/title/file/host/port/statuses/agents/readonly/inbox_file/deployment_file
  （后两者 null 或缺省 = 隐藏收件箱/部署台账页签与 API，向后兼容旧配置）；
  非法配置 fail-fast（中文错误 + 非 0 退出，绝不写工单文件）。

唯一真相源：所有写操作都调用 `ticket.py` 的 apply_new/apply_set（flock 排他锁 +
临时文件原子替换），页面与 CLI 共用同一实现，不做任何 Markdown 表格拼接。
每次写操作写前校验结构（不合法不落盘），响应回显 check 结果。
收件箱 / 部署台账同理复用 `ledger.py` 的解析与写操作（同一锁与原子写模式）。
"""
import argparse
import hashlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ticket  # noqa: E402  （同目录，唯一真相源）
import ledger  # noqa: E402  （同目录，收件箱/部署台账解析与写操作）
import agent_model  # noqa: E402  （同目录，「模型」页签的唯一实现）

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_HTML = TOOL_DIR / "tickets.html"
DEFAULT_FILE = ticket.DEFAULT_FILE



class ApiError(Exception):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.message = message
        self.code = code


# --------------------------------------------------------------------------- #
# 数据组装
# --------------------------------------------------------------------------- #
def ticket_json(t):
    tid = t["id"]
    sub = "." in tid
    return {"id": tid, "date": t["date"], "by": t["by"], "to": t["to"],
            "desc": t["desc"], "verify": t["verify"], "note": t["note"],
            "section": t["section"], "sub": sub,
            "parent": tid.split(".")[0] if sub else None}


def check_payload(path):
    try:
        errs, count = ticket.validate_file(path)
    except ticket.TicketError as e:
        return {"ok": False, "errors": [str(e)], "count": 0, "summary": str(e)}
    return {"ok": not errs, "errors": errs, "count": count,
            "summary": f"{count} 条工单，结构合法" if not errs else f"{len(errs)} 处问题"}


def tickets_payload(path, meta):
    tickets, _ = ticket.read_tickets(path)
    return {"ok": True, "file": str(path), "name": meta["name"], "title": meta["title"],
            "readonly": meta["readonly"],
            "statuses": list(meta["statuses"]), "agents": list(meta["agents"]),
            "tickets": [ticket_json(t) for t in tickets],
            "check": check_payload(path)}


def build_meta(name, title, statuses, agents, file, readonly,
               inbox_file=None, deployment_file=None, models_tab=True,
               agent_dir=None, opencode_config=None):
    """models_tab 开启时 agent_dir/opencode_config 缺省取 agent_model 的真实路径；
    自检/测试用临时副本显式覆盖。"""
    return {"name": name, "title": title, "statuses": list(statuses),
            "agents": list(agents), "file": str(file), "readonly": bool(readonly),
            "inbox_file": str(inbox_file) if inbox_file else None,
            "deployment_file": str(deployment_file) if deployment_file else None,
            "models_tab": bool(models_tab),
            "agent_dir": str(agent_dir if agent_dir is not None
                             else agent_model.DEFAULT_AGENT_DIR),
            "opencode_config": str(opencode_config if opencode_config is not None
                                   else agent_model.DEFAULT_CONFIG)}


# --------------------------------------------------------------------------- #
# 收件箱 / 部署台账 / 统一编号查询（解析复用 ledger 模块）
# --------------------------------------------------------------------------- #
def inbox_json(e):
    return {"id": e["id"], "date": e["date"], "source": e["source"], "type": e["type"],
            "text": e["text"], "followups": list(e["followups"]), "section": e["section"]}


def inbox_payload(path, query):
    entries = ledger.parse_inbox(path)
    q = {k: v[0] for k, v in (query or {}).items() if v}
    shown = ledger.filter_inbox(entries, status=q.get("status"), type_=q.get("type"),
                                source=q.get("source"), since=q.get("since"), q=q.get("q"))
    counts = {"待同步": sum(1 for e in entries if e["section"] == "待同步"),
              "已同步": sum(1 for e in entries if e["section"] == "已同步"),
              "total": len(entries)}
    types = sorted({e["type"] for e in entries})
    sources = sorted({e["source"] for e in entries})
    return {"ok": True, "file": str(path), "counts": counts, "types": types,
            "sources": sources, "entries": [inbox_json(e) for e in shown]}


def deployment_json(e):
    return {"id": e["id"], "date": e["date"], "weekday": e.get("weekday") or "",
            "title": e["title"], "wo": e.get("wo") or "", "impact": list(e["impact"]),
            "values": dict(e["values"])}


def deployment_payload(path, query):
    entries = ledger.parse_deployment(path)
    q = {k: v[0] for k, v in (query or {}).items() if v}
    impact = q.get("impact")
    if impact and impact not in ledger.IMPACT_CATEGORIES:
        raise ApiError(f"impact 应为 {'/'.join(ledger.IMPACT_CATEGORIES)}")
    shown = ledger.filter_deployment(entries, impact=impact, since=q.get("since"),
                                     q=q.get("q"))
    impacts = sorted({c for e in entries for c in e["impact"]},
                     key=ledger.IMPACT_CATEGORIES.index)
    return {"ok": True, "file": str(path), "total": len(entries),
            "categories": list(ledger.IMPACT_CATEGORIES), "impacts": impacts,
            "entries": [deployment_json(e) for e in shown]}


def entry_payload(ident, ticket_file, inbox_file, deployment_file):
    r = ledger.get_entry(ident, inbox_file=inbox_file, tickets_file=ticket_file,
                         deployment_file=deployment_file)
    if r["kind"] == "inbox":
        return {"ok": True, "kind": "inbox", "entry": inbox_json(r["entry"])}
    if r["kind"] == "ticket":
        t = r["ticket"]
        return {"ok": True, "kind": "ticket", "ticket": ticket_json(t),
                "children": [ticket_json(c) for c in r["children"]]}
    return {"ok": True, "kind": "deployment", "entry": deployment_json(r["entry"]),
            "matches": [deployment_json(e) for e in r.get("matches", [])]}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_handler(ticket_file, html_file, meta, quiet=False):
    ticket_file = Path(ticket_file)
    html_file = Path(html_file)

    class Handler(BaseHTTPRequestHandler):
        server_version = "TicketWeb/1.1"

        def log_message(self, fmt, *args):
            if not quiet:
                super().log_message(fmt, *args)

        # ---- 响应工具 ----
        def _send(self, body, content_type, code=200):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8", code)

        def _error(self, message, code=400):
            self._json({"ok": False, "error": message}, code)

        # ---- 路由 ----
        def do_GET(self):
            u = urlparse(self.path)
            path, query = u.path, parse_qs(u.query)
            try:
                if path in ("/", "/index.html", "/tickets.html"):
                    self._serve_html()
                elif path == "/api/meta":
                    self._json({"ok": True, **meta})
                elif path == "/api/tickets":
                    self._json(tickets_payload(ticket_file, meta))
                elif path == "/api/ticket":
                    self._json(self._ticket_detail(query))
                elif path == "/api/inbox":
                    if not meta.get("inbox_file"):
                        raise ApiError("实例未配置 inbox_file（收件箱页签隐藏）", 404)
                    self._json(inbox_payload(Path(meta["inbox_file"]), query))
                elif path == "/api/deployment":
                    if not meta.get("deployment_file"):
                        raise ApiError("实例未配置 deployment_file（部署台账页签隐藏）", 404)
                    self._json(deployment_payload(Path(meta["deployment_file"]), query))
                elif path == "/api/entry":
                    ident = (query.get("id") or [None])[0]
                    if not ident:
                        raise ApiError("缺少参数 id")
                    self._json(entry_payload(ident, ticket_file,
                                             Path(meta["inbox_file"]) if meta.get("inbox_file") else None,
                                             Path(meta["deployment_file"]) if meta.get("deployment_file") else None))
                elif path == "/api/models":
                    if not meta.get("models_tab"):
                        raise ApiError("实例未启用模型页签（models_tab=false）", 404)
                    self._json(self._models_status())
                elif path == "/api/check":
                    self._json({"ok": True, **check_payload(ticket_file)})
                else:
                    self._error(f"未知路径: {path}", 404)
            except ApiError as e:
                self._error(e.message, e.code)
            except agent_model.AgentModelError as e:
                self._error(str(e), 400)
            except ticket.TicketError as e:
                self._error(str(e), 400)

        def do_POST(self):
            u = urlparse(self.path)
            path = u.path
            try:
                if meta["readonly"]:
                    raise ApiError("只读实例：已禁用写操作", 403)
                params = self._read_json()
                if path == "/api/new":
                    self._json(self._new(params))
                elif path == "/api/status":
                    self._json(self._set_field(params, "status", required=True))
                elif path == "/api/note":
                    self._json(self._set_field(params, "note", required=True, append=True))
                elif path == "/api/verify":
                    self._json(self._set_field(params, "verify", required=True))
                elif path == "/api/to":
                    self._json(self._set_field(params, "to", required=True))
                elif path == "/api/inbox/append":
                    if not meta.get("inbox_file"):
                        raise ApiError("实例未配置 inbox_file（收件箱页签隐藏）", 404)
                    self._json(self._inbox_append(params))
                elif path == "/api/inbox/move":
                    if not meta.get("inbox_file"):
                        raise ApiError("实例未配置 inbox_file（收件箱页签隐藏）", 404)
                    self._json(self._inbox_move(params))
                elif path == "/api/deployment/new":
                    if not meta.get("deployment_file"):
                        raise ApiError("实例未配置 deployment_file（部署台账页签隐藏）", 404)
                    self._json(self._deployment_new(params))
                elif path == "/api/models/set":
                    self._json(self._models_write(params, "set"))
                elif path == "/api/models/clear":
                    self._json(self._models_write(params, "clear"))
                elif path == "/api/models/default":
                    self._json(self._models_write(params, "default"))
                else:
                    self._error(f"未知路径: {path}", 404)
            except ApiError as e:
                self._error(e.message, e.code)
            except agent_model.AgentModelError as e:
                self._error(str(e), 400)
            except ticket.TicketError as e:
                self._error(str(e), 400)

        # ---- 业务 ----
        def _serve_html(self):
            if not html_file.exists():
                self._error(f"页面不存在: {html_file}", 500)
                return
            self._send(html_file.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise ApiError(f"请求体非合法 JSON: {e}")
            if not isinstance(obj, dict):
                raise ApiError("请求体应为 JSON 对象")
            return obj

        def _ticket_detail(self, query):
            tid = (query.get("id") or [None])[0]
            if not tid:
                raise ApiError("缺少参数 id")
            tid = ticket.parse_id(tid)
            tickets, _ = ticket.read_tickets(ticket_file)
            main = next((t for t in tickets if t["id"] == tid), None)
            if main is None:
                raise ApiError(f"工单不存在: {tid}", 404)
            children = [ticket_json(x) for x in tickets
                        if x["id"].startswith(tid + ".")]
            return {"ok": True, "ticket": ticket_json(main), "children": children,
                    "check": check_payload(ticket_file)}

        def _new(self, params):
            desc = params.get("desc")
            by = params.get("by")
            if not desc:
                raise ApiError("缺少 desc（需求/改动描述）")
            if not by:
                raise ApiError("缺少 by（发起人）")
            t = ticket.apply_new(
                ticket_file, desc=desc, by=by,
                to=params.get("to"), parent=params.get("parent"),
                verify=params.get("verify"), note=params.get("note"),
                date=params.get("date"), validate=True,
                business=params.get("business"), approach=params.get("approach"))
            return {"ok": True, "id": t["id"], "ticket": ticket_json(t),
                    "check": check_payload(ticket_file)}

        def _set_field(self, params, field, required=False, append=False):
            tid = params.get("id") or params.get("ticket")
            if not tid:
                raise ApiError("缺少 id")
            if required and field not in params:
                raise ApiError(f"缺少字段 {field}")
            value = params.get(field)
            if isinstance(value, str) and "|" in value:
                raise ApiError("字段内禁用竖线 |")
            kwargs = {"append_note": value} if append else {field: value}
            r = ticket.apply_set(ticket_file, tid, validate=True, **kwargs)
            return {"ok": True, "id": r["id"], "status": r["target"],
                    "warning": r["warning"], "ticket": ticket_json(r["ticket"]),
                    "check": check_payload(ticket_file)}

        def _inbox_append(self, params):
            path = Path(meta["inbox_file"])
            e = ledger.apply_inbox_append(
                path, source=params.get("source") or "", type_=params.get("type") or "",
                text=params.get("text") or "", followup=params.get("followup"),
                date=params.get("date"))
            return {"ok": True, "id": e["id"], "entry": inbox_json(e),
                    "counts": inbox_payload(path, {}).get("counts")}

        def _inbox_move(self, params):
            ident = params.get("id") or ""
            if not ident:
                raise ApiError("缺少 id（INB-xxxx）")
            e = ledger.apply_inbox_move(Path(meta["inbox_file"]), ident,
                                        destination=params.get("destination"))
            return {"ok": True, "id": e["id"], "entry": inbox_json(e)}

        def _deployment_new(self, params):
            impact = params.get("impact")
            if isinstance(impact, str):
                impact = [x.strip() for x in impact.split(",") if x.strip()]
            if not isinstance(impact, list):
                raise ApiError("impact 应为字符串数组（影响面 8 选 N）")
            e = ledger.apply_deployment_append(
                Path(meta["deployment_file"]), title=params.get("title") or "",
                changes=params.get("changes") or "", impact=impact,
                notes=params.get("notes"), verify=params.get("verify"),
                date=params.get("date"))
            return {"ok": True, "id": e["id"], "entry": deployment_json(e)}

        # ---- 模型（「模型」页签；逻辑全在 agent_model，勿在此复制） ----
        def _models_status(self):
            st = agent_model.status(agent_dir=Path(meta["agent_dir"]),
                                    config_path=Path(meta["opencode_config"]))
            st.update({"ok": True, "models_tab": True,
                       "readonly": bool(meta["readonly"])})
            return st

        def _models_write(self, params, op):
            if not meta.get("models_tab"):
                raise ApiError("实例未启用模型页签（models_tab=false）", 404)
            agent_dir = Path(meta["agent_dir"])
            cfgp = Path(meta["opencode_config"])
            if op == "default":
                r = agent_model.apply_default(params.get("model") or "", config_path=cfgp)
            else:
                agent = params.get("agent") or ""
                if not agent:
                    raise ApiError("缺少 agent")
                if op == "set":
                    r = agent_model.apply_set(agent, params.get("model") or "",
                                              agent_dir=agent_dir)
                else:
                    r = agent_model.apply_clear(agent, agent_dir=agent_dir)
            out = dict(r)
            out.update({"ok": True, "status": self._models_status()})
            return out

    return Handler


def serve(host="127.0.0.1", port=8010, ticket_file=DEFAULT_FILE, html_file=DEFAULT_HTML,
          meta=None, quiet=False):
    if meta is None:
        meta = build_meta(ticket.DEFAULT_NAME, ticket.DEFAULT_TITLE,
                          ticket.STATUSES, ticket.AGENTS, ticket_file, False)
    handler = make_handler(ticket_file, html_file, meta, quiet=quiet)
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd


# --------------------------------------------------------------------------- #
# 自检（临时副本端到端）
# --------------------------------------------------------------------------- #
def run_self_test():
    import json as _json
    import os as _os
    import shutil
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    passed = [0]
    def check(cond, msg):
        if cond:
            passed[0] += 1
            print(f"  ✓ {msg}")
        else:
            raise AssertionError(f"自检失败: {msg}")

    def md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    def http(method, url, data=None):
        body = None
        req = urllib.request.Request(url, method=method)
        if data is not None:
            body = _json.dumps(data, ensure_ascii=False).encode("utf-8")
            req.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urllib.request.urlopen(req, body, timeout=10) as r:
                return r.status, _json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, _json.loads(e.read().decode("utf-8"))

    def cli(file, *args):
        return subprocess.run(
            [sys.executable, str(TOOL_DIR / "ticket.py"), "--file", str(file), *args],
            capture_output=True, text=True)

    def expect_ticket_error(fn, substr, msg):
        try:
            fn()
        except ticket.TicketError as e:
            check(substr in str(e), msg)
            return
        raise AssertionError(f"自检失败: {msg}（未抛出 TicketError）")

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def wait_port(port, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    return True
            except OSError:
                time.sleep(0.05)
        raise AssertionError(f"自检失败: 临时实例端口 {port} 未就绪")

    real = Path(DEFAULT_FILE)
    if not real.exists():
        real = Path(ticket.DEFAULT_FILE)
    check(Path(ticket.DEFAULT_FILE).is_absolute(), "内置默认工单文件为绝对路径（cwd 无关）")
    real_md5_before = md5(real)
    real_lines_before = len(real.read_text(encoding="utf-8").splitlines())
    print(f"真实工单文件基线: {real} md5={real_md5_before} 行数={real_lines_before}")

    tmpdir = Path(tempfile.mkdtemp(prefix="umvp-ticket-selftest-"))
    try:
        tmp_file = tmpdir / "tickets.md"
        shutil.copyfile(real, tmp_file)

        # 模型页签自检用临时 agent 目录与临时 opencode.json（真实文件只读）
        tmp_agents = tmpdir / "agents"
        tmp_agents.mkdir()
        for src in sorted(agent_model.DEFAULT_AGENT_DIR.glob("*.md")):
            shutil.copyfile(src, tmp_agents / src.name)
        tmp_opencode = tmpdir / "opencode.json"
        shutil.copyfile(agent_model.DEFAULT_CONFIG, tmp_opencode)
        real_agents_before = {p.name: md5(p)
                              for p in agent_model.DEFAULT_AGENT_DIR.glob("*.md")}
        real_cfg_before = md5(agent_model.DEFAULT_CONFIG)

        httpd = serve("127.0.0.1", 0, ticket_file=tmp_file, html_file=DEFAULT_HTML,
                      meta=build_meta("umvp", ticket.DEFAULT_TITLE, ticket.DEFAULT_STATUSES,
                                      ticket.DEFAULT_AGENTS, tmp_file, False,
                                      models_tab=True, agent_dir=tmp_agents,
                                      opencode_config=tmp_opencode),
                      quiet=True)

        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        print(f"临时服务: {base}（文件 {tmp_file}）")

        try:
            # 1. 列表
            st, body = http("GET", f"{base}/api/tickets")
            check(st == 200 and body.get("ok"), "GET /api/tickets 返回 ok")
            check(body["statuses"] == ticket.STATUSES, "状态清单与 ticket.STATUSES 一致")
            check(body["agents"] == ticket.AGENTS, "代理清单与 ticket.AGENTS 一致")
            local_tickets, _ = ticket.read_tickets(tmp_file)
            check(len(body["tickets"]) == len(local_tickets),
                  f"列表条数一致（{len(body['tickets'])}）")
            check(body["check"]["ok"], "列表内嵌 check 合法")

            # 1b. /api/meta（默认 = UMVP）
            st, body = http("GET", f"{base}/api/meta")
            check(st == 200 and body.get("ok") and body["name"] == "umvp"
                  and body["title"] == "UMVP 工单看板" and body["readonly"] is False,
                  "默认 /api/meta = UMVP 标题/实例名/非只读")

            # 2. 首页
            with urllib.request.urlopen(f"{base}/", timeout=10) as r:
                html = r.read().decode("utf-8")
                code = r.status
            check(code == 200 and "工单看板" in html, "GET / 返回单页（含标题）")

            # 3. 结构校验端点
            st, body = http("GET", f"{base}/api/check")
            check(st == 200 and body["ok"], "GET /api/check 全绿")

            base_count = len(local_tickets)

            # 4. 新建主单（.2 起主单必填 business/approach，描述自动拼两段前缀）
            st, body = http("POST", f"{base}/api/new", {"desc": "自检主单", "by": "web",
                                                        "to": "qa", "verify": "cmd",
                                                        "note": "n",
                                                        "business": "自检业务",
                                                        "approach": "自检做法"})
            check(st == 200 and body.get("ok"), "POST /api/new 主单成功")
            new_id = body["id"]
            check(new_id.startswith("WO-") and "." not in new_id, f"新主单编号 {new_id}")
            check(body["ticket"]["desc"] == "业务：自检业务。实现方式：自检做法。补充：自检主单",
                  "主单描述拼装为「业务：…。实现方式：…。补充：…」固定格式")
            check(body["check"]["ok"], "新建后 check 合法")
            ids_cli = cli(tmp_file, "list").stdout + cli(tmp_file, "list").stderr
            check(new_id in ids_cli, f"CLI list 可见 {new_id}")

            # 5. 改状态
            st, body = http("POST", f"{base}/api/status", {"id": new_id, "status": "待测试"})
            check(st == 200 and body.get("ok") and body["status"] == "待测试",
                  "改状态 → 待测试")
            check(new_id in cli(tmp_file, "list", "--status", "待测试").stdout,
                  "CLI 待测试节能见该单")

            # 6. 追加备注
            st, body = http("POST", f"{base}/api/note", {"id": new_id, "note": "自检追加"})
            check(st == 200 and "自检追加" in body["ticket"]["note"], "追加备注成功")

            # 7. 编辑验证
            st, body = http("POST", f"{base}/api/verify", {"id": new_id, "verify": "python3 -V"})
            check(st == 200 and body["ticket"]["verify"] == "python3 -V", "编辑验证成功")

            # 8. 编辑承接
            st, body = http("POST", f"{base}/api/to", {"id": new_id, "to": "pipe-chain,qa"})
            check(st == 200 and body["ticket"]["to"] == "pipe-chain,qa", "编辑承接成功")

            # 9. 新建子单
            st, body = http("POST", f"{base}/api/new", {"desc": "自检子单", "by": "web",
                                                        "parent": new_id})
            check(st == 200 and body["id"] == f"{new_id}.1", f"新建子单 {new_id}.1")

            # 10. 写后 check 全绿 + 条数
            cp = subprocess.run([sys.executable, str(TOOL_DIR / "ticket.py"),
                                 "--file", str(tmp_file), "check"],
                                capture_output=True, text=True)
            check(cp.returncode == 0 and "check OK" in cp.stdout, "CLI check 全绿（exit 0）")

            # 11. 页面列表与 CLI list 一致
            st, body = http("GET", f"{base}/api/tickets")
            api_ids = sorted(t["id"] for t in body["tickets"])
            cli_id_set = sorted({line.split()[1] for line in
                                 cli(tmp_file, "list").stdout.splitlines() if line.strip()})
            check(api_ids == cli_id_set, "页面列表与 CLI list 编号一致")

            # 12. 非法输入被拒且文件不变
            before = md5(tmp_file)
            st, body = http("POST", f"{base}/api/new",
                            {"desc": "带|竖线", "by": "web", "business": "b", "approach": "a"})
            check(st == 400 and body.get("ok") is False and "竖线" in body["error"],
                  "含竖线 desc 被拒（400）")
            check(md5(tmp_file) == before, "被拒后文件未变（新）")

            st, body = http("POST", f"{base}/api/status", {"id": new_id, "status": "不存在的状态"})
            check(st == 400 and body.get("ok") is False, "非法状态被拒（400）")
            check(md5(tmp_file) == before, "被拒后文件未变（状态）")

            st, body = http("POST", f"{base}/api/new", {"desc": "x", "by": "web", "parent": "WO-9999"})
            check(st == 400 and "父单不存在" in body["error"], "不存在父单被拒（400）")
            check(md5(tmp_file) == before, "被拒后文件未变（父单）")

            st, body = http("POST", f"{base}/api/note", {"id": new_id, "note": "坏|备注"})
            check(st == 400 and body.get("ok") is False, "备注含竖线被拒（400）")
            check(md5(tmp_file) == before, "被拒后文件未变（备注）")

            # 13. 详情端点
            st, body = http("GET", f"{base}/api/ticket?id={new_id}")
            check(st == 200 and body["ok"] and len(body["children"]) == 1,
                  "GET /api/ticket 详情含 1 个子单")

            # 14. 缺字段
            st, body = http("POST", f"{base}/api/new", {"desc": "无发起人"})
            check(st == 400 and body.get("ok") is False, "缺 by 被拒")

            # 14b. 主单两段式必填（WO-0009.2）：缺一即 400 且文件不变；子单不要求
            before = md5(tmp_file)
            st, body = http("POST", f"{base}/api/new",
                            {"desc": "缺业务", "by": "web", "approach": "做法"})
            check(st == 400 and "business" in body.get("error", ""), "主单缺 business 被拒（400）")
            st, body = http("POST", f"{base}/api/new",
                            {"desc": "缺做法", "by": "web", "business": "业务"})
            check(st == 400 and "approach" in body.get("error", ""), "主单缺 approach 被拒（400）")
            check(md5(tmp_file) == before, "被拒后文件未变（主单两段式）")
            st, body = http("POST", f"{base}/api/new",
                            {"desc": "子单不带前缀", "by": "web", "parent": new_id})
            check(st == 200 and body["ticket"]["desc"] == "子单不带前缀",
                  "子单不要求 business/approach 且描述不拼前缀")

            # 15. 并发写入（flock 排他锁）
            import concurrent.futures
            n = 8
            cnt_before = len(ticket.read_tickets(tmp_file)[0])
            def mk(i):
                return http("POST", f"{base}/api/new",
                            {"desc": f"并发-{i}", "by": "web",
                             "business": "并发业务", "approach": "并发做法"})
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                results = list(ex.map(mk, range(n)))
            check(all(st == 200 and b.get("ok") for st, b in results),
                  f"并发 {n} 次新建全部成功")
            all_ids = [t["id"] for t in ticket.read_tickets(tmp_file)[0]]
            check(len(all_ids) == cnt_before + n,
                  f"并发后条数 +{n}（{cnt_before}→{len(all_ids)}）")
            check(len(all_ids) == len(set(all_ids)), "并发后无重复编号")
            cp = subprocess.run([sys.executable, str(TOOL_DIR / "ticket.py"),
                                 "--file", str(tmp_file), "check"],
                                capture_output=True, text=True)
            check(cp.returncode == 0 and "check OK" in cp.stdout, "并发后 CLI check 全绿")

        finally:
            httpd.shutdown()
            httpd.server_close()
            th.join(timeout=5)

        # 16. 配置化（多项目复用）：合法/非法配置 + 第二实例隔离
        cfgdir = tmpdir / "cfg"
        cfgdir.mkdir()
        custom_md = cfgdir / "custom_tickets.md"
        custom_md.write_text(
            "# 工单\n\n"
            "## 待办\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n"
            "| WO-0001 | 2026-10-19 | alice | bob | 示例 | - | - |\n\n"
            "## 处理中\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n\n"
            "## 归档\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8")

        def write_cfg(name, obj):
            p = cfgdir / name
            p.write_text(_json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            return p

        cfg_path = write_cfg("proj2.json", {
            "name": "proj2", "title": "项目二看板", "file": "custom_tickets.md",
            "host": "127.0.0.1", "port": 8012,
            "statuses": ["待办", "处理中", "归档"], "agents": ["alice", "bob"],
            "models_tab": False})
        cfg2 = ticket.load_config(cfg_path)
        check(cfg2.file == custom_md.resolve(), "配置相对 file 解析到配置文件所在目录")
        check(cfg2.port == 8012 and cfg2.title == "项目二看板" and cfg2.name == "proj2",
              "配置 title/port/name 生效")
        check(cfg2.statuses == ["待办", "处理中", "归档"], "配置 statuses 覆盖内置")
        check(cfg2.agents == ["alice", "bob"], "配置 agents 覆盖内置")

        home_cfg = write_cfg("home.json", {"file": "~/x_tickets.md"})
        check(ticket.load_config(home_cfg).file == (Path.home() / "x_tickets.md").resolve(),
              "配置 file 支持 ~ 展开")
        rel_cfg = write_cfg("rel.json", {"file": "custom_tickets.md"})
        check(ticket.load_config(rel_cfg).statuses == ticket.DEFAULT_STATUSES,
              "未给 statuses 时用内置默认")

        check(ticket.load_config_from_args(env={}) is None,
              "无配置来源 → None（内置默认）")
        check(ticket.load_config_from_args(env={"TICKET_CONFIG": str(cfg_path)}).name == "proj2",
              "环境变量 TICKET_CONFIG 生效")
        check(ticket.load_config_from_args(env={"TICKETD_CONFIG": str(cfg_path)}).name == "proj2",
              "环境变量 TICKETD_CONFIG 兼容生效")
        check(ticket.load_config_from_args(
            config_arg=str(cfg_path), env={"TICKET_CONFIG": "/nonexistent.json"}).name == "proj2",
            "CLI --config 优先于环境变量")
        # --name → XDG 默认路径
        xdg = tmpdir / "xdg"
        (xdg / "ticketd").mkdir(parents=True)
        (xdg / "ticketd" / "proj2.json").write_text(
            _json.dumps({"name": "proj2", "file": str(custom_md)}, ensure_ascii=False),
            encoding="utf-8")
        by_name = ticket.load_config_from_args(name_arg="proj2",
                                               env={"XDG_CONFIG_HOME": str(xdg)})
        check(by_name is not None and by_name.name == "proj2",
              "--name 经 XDG_CONFIG_HOME 定位 ~/.config/ticketd/<name>.json")

        bad = cfgdir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        expect_ticket_error(lambda: ticket.load_config(bad), "JSON", "坏 JSON 报中文错误")
        nf = write_cfg("nofile.json", {"title": "x"})
        expect_ticket_error(lambda: ticket.load_config(nf), "file", "缺 file 报错")
        for badport in (0, 70000):
            bp = write_cfg(f"port{badport}.json", {"file": "custom_tickets.md", "port": badport})
            expect_ticket_error(lambda p=bp: ticket.load_config(p), "port", f"非法端口 {badport} 报错")
        empty = write_cfg("empty.json", {"file": "custom_tickets.md", "statuses": []})
        expect_ticket_error(lambda: ticket.load_config(empty), "statuses", "空 statuses 报错")
        pipe = write_cfg("pipe.json", {"file": "custom_tickets.md", "statuses": ["a|b"]})
        expect_ticket_error(lambda: ticket.load_config(pipe), "竖线", "statuses 含竖线报错")
        dup = write_cfg("dup.json", {"file": "custom_tickets.md", "agents": ["a", "a"]})
        expect_ticket_error(lambda: ticket.load_config(dup), "重复", "agents 重复报错")
        ro = write_cfg("ro.json", {"file": "custom_tickets.md", "readonly": "yes"})
        expect_ticket_error(lambda: ticket.load_config(ro), "readonly", "readonly 非布尔报错")
        expect_ticket_error(lambda: ticket.load_config(cfgdir / "nope.json"), "不存在",
                            "配置文件不存在报错")

        # 16b. 非法配置 CLI：非 0 退出 + 中文错误 + 绝不写文件
        target = cfgdir / "should_not_exist.md"
        cp = subprocess.run(
            [sys.executable, str(TOOL_DIR / "ticket.py"), "--config", str(bad),
             "--file", str(target), "new", "x", "--by", "web"],
            capture_output=True, text=True)
        check(cp.returncode != 0, "非法配置 CLI 非 0 退出")
        check("✗" in cp.stderr and "JSON" in cp.stderr, "非法配置中文错误走 stderr")
        check(not target.exists(), "非法配置不写工单文件")

        # 16c. CLI 经环境变量配置读到自定义工单（自定义 agents）
        env_cfg = dict(_os.environ)
        env_cfg["TICKET_CONFIG"] = str(cfg_path)
        cp = subprocess.run([sys.executable, str(TOOL_DIR / "ticket.py"), "list"],
                            capture_output=True, text=True, env=env_cfg)
        check(cp.returncode == 0 and "alice" in cp.stdout and "WO-0001" in cp.stdout,
              "CLI 经 TICKET_CONFIG 读到自定义工单/代理")

        # 16d. 第二实例（子进程）与 UMVP 并存互不干扰
        port2 = free_port()
        proc = subprocess.Popen(
            [sys.executable, str(TOOL_DIR / "ticket_web.py"), "--port", str(port2)],
            env=env_cfg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base2 = f"http://127.0.0.1:{port2}"
            wait_port(port2)
            st, body = http("GET", f"{base2}/api/meta")
            check(st == 200 and body["name"] == "proj2" and body["title"] == "项目二看板",
                  "第二实例 /api/meta 标题/实例名=配置值")
            check(body["statuses"] == ["待办", "处理中", "归档"], "第二实例状态=配置值")
            check(body["agents"] == ["alice", "bob"], "第二实例代理=配置值")
            st, body = http("GET", f"{base2}/api/tickets")
            check(st == 200 and body["statuses"] == ["待办", "处理中", "归档"],
                  "第二实例 /api/tickets 状态=配置值")
            check(md5(real) == real_md5_before, "第二实例启动未触碰 UMVP 真实文件")
            custom_before = md5(custom_md)
            st, body = http("POST", f"{base2}/api/new",
                            {"desc": "第二项目新单", "by": "alice",
                             "business": "业务二", "approach": "做法二"})
            check(st == 200 and body.get("ok"), "第二实例可用自定义代理写自己的文件")
            check(md5(custom_md) != custom_before, "第二实例写入落到自己的文件")
            check(md5(real) == real_md5_before, "第二实例写入不影响 UMVP 真实文件")
            st, body = http("POST", f"{base2}/api/new", {"desc": "x", "by": "web"})
            check(st == 400, "第二实例拒绝非配置代理 web（400）")
            # 16d.bis 未配置实例隐藏 ledger 页签与 API（proj2 配置无 inbox_file/deployment_file）
            st, body = http("GET", f"{base2}/api/meta")
            check(st == 200 and body.get("inbox_file") is None
                  and body.get("deployment_file") is None, "未配置实例 meta 两字段为 null")
            st, body = http("GET", f"{base2}/api/inbox")
            check(st == 404 and "inbox_file" in body.get("error", ""), "未配置实例 /api/inbox 404")
            st, body = http("GET", f"{base2}/api/deployment")
            check(st == 404 and "deployment_file" in body.get("error", ""),
                  "未配置实例 /api/deployment 404")
            st, body = http("GET", f"{base2}/api/models")
            check(st == 404 and "模型页签" in body.get("error", ""),
                  "models_tab=false 实例 /api/models 404")
            st, body = http("POST", f"{base2}/api/inbox/append",
                            {"source": "a", "type": "b", "text": "c"})
            check(st == 404, "未配置实例收件箱写操作 404")
            st, body = http("GET", f"{base2}/api/entry?id=INB-0001")
            check(st == 400 and "收件箱" in body.get("error", ""),
                  "未配置实例 /api/entry INB 报可读错误")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # 16e. 只读实例：meta.readonly=true 且写操作 403
        port3 = free_port()
        proc3 = subprocess.Popen(
            [sys.executable, str(TOOL_DIR / "ticket_web.py"), "--port", str(port3), "--readonly"],
            env=env_cfg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base3 = f"http://127.0.0.1:{port3}"
            wait_port(port3)
            st, body = http("GET", f"{base3}/api/meta")
            check(st == 200 and body["readonly"] is True, "只读实例 meta.readonly=true")
            before3 = md5(custom_md)
            st, body = http("POST", f"{base3}/api/new", {"desc": "x", "by": "alice"})
            check(st == 403 and body.get("ok") is False, "只读实例写操作被拒（403）")
            check(md5(custom_md) == before3, "只读实例被拒后文件未变")
        finally:
            proc3.terminate()
            try:
                proc3.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc3.kill()

        # 16f. 只读 + 收件箱配置实例：ledger 读可用、写被 403
        ro_inbox = cfgdir / "ro_inbox.md"
        shutil.copyfile(ledger.DEFAULT_INBOX, ro_inbox)
        ro2 = write_cfg("ro2.json", {"file": str(custom_md), "readonly": True,
                                     "inbox_file": str(ro_inbox),
                                     "deployment_file": None})
        port4 = free_port()
        proc4 = subprocess.Popen(
            [sys.executable, str(TOOL_DIR / "ticket_web.py"), "--port", str(port4),
             "--config", str(ro2)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base4 = f"http://127.0.0.1:{port4}"
            wait_port(port4)
            st, body = http("GET", f"{base4}/api/meta")
            check(st == 200 and body["readonly"] is True and body["inbox_file"],
                  "只读实例 meta 带 inbox_file")
            st, body = http("GET", f"{base4}/api/inbox")
            check(st == 200 and body.get("ok"), "只读实例可读收件箱")
            ro_before = md5(ro_inbox)
            st, body = http("POST", f"{base4}/api/inbox/append",
                            {"source": "web", "type": "成果", "text": "x"})
            check(st == 403 and body.get("ok") is False, "只读实例收件箱写操作被拒（403）")
            st, body = http("POST", f"{base4}/api/deployment/new",
                            {"title": "t", "changes": "c", "impact": ["DB"]})
            check(st == 403, "只读实例台账登记被拒（403）")
            check(md5(ro_inbox) == ro_before, "只读实例收件箱文件未变")
        finally:
            proc4.terminate()
            try:
                proc4.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc4.kill()

        # 18. ledger：真实文件解析往返（只读）+ 全文件重构无丢字
        real_inbox = Path(ledger.DEFAULT_INBOX)
        real_dep = Path(ledger.DEFAULT_DEPLOYMENT)
        real_inbox_md5 = md5(real_inbox)
        real_dep_md5 = md5(real_dep)
        es = ledger.parse_inbox(real_inbox)
        ds = ledger.parse_deployment(real_dep)
        check(len(es) >= 40, f"真实收件箱解析 ≥40 条（实为 {len(es)}）")
        n_pend = sum(1 for e in es if e["section"] == "待同步")
        n_sync = sum(1 for e in es if e["section"] == "已同步")
        check(n_pend + n_sync == len(es), f"两节条数闭合（待同步 {n_pend} + 已同步 {n_sync}）")
        check(not ledger.inbox_roundtrip_errors(real_inbox),
              f"收件箱 {len(es)} 条逐条结构渲染与原文一致（无丢字）")
        check(not ledger.deployment_roundtrip_errors(real_dep),
              f"台账 {len(ds)} 条逐条结构渲染与原文一致（无丢字）")

        def full_recon(path, entries):
            lines = path.read_text(encoding="utf-8").splitlines()
            pos, out = 0, []
            for e in entries:
                start = lines.index(e["lines"][0], pos)
                out += lines[pos:start] + e["lines"]
                pos = start + len(e["lines"])
            out += lines[pos:]
            return out == lines

        check(full_recon(real_inbox, es), "收件箱全文件重构逐行一致（prelude+条目+尾）")
        check(full_recon(real_dep, ds), "台账全文件重构逐行一致")
        check([e["id"] for e in ds] == [f"DEP-{i:04d}" for i in range(1, len(ds) + 1)],
              "台账 DEP 编号按序分配（最新在前）")
        wo_entries = [e for e in ds if e.get("wo")]
        check(bool(wo_entries), "台账存在含 WO 号的条目（可按 WO 索引）")
        if wo_entries:
            r = ledger.get_entry(wo_entries[0]["wo"], deployment_file=real_dep,
                                 force_deployment=True)
            check(r["kind"] == "deployment" and r["entry"]["wo"] == wo_entries[0]["wo"],
                  f"get {wo_entries[0]['wo']} --from-deployment 命中台账条目 {r['entry']['id']}")
        r = ledger.get_entry("2026-09-23 抽出", deployment_file=real_dep)
        check(r["kind"] == "deployment" and r["matches"], "台账模糊查询（日期+标题）命中")
        max_inb = max(int(e["id"][4:]) for e in es)
        check(max_inb >= 49, f"当前 INB 最大编号 {max_inb:04d}（基线 0049）")

        # 19. ledger 写操作（独立副本 + HTTP 实例）
        inbox_copy = tmpdir / "ledger_inbox.md"
        dep_copy = tmpdir / "ledger_dep.md"
        shutil.copyfile(real_inbox, inbox_copy)
        shutil.copyfile(real_dep, dep_copy)
        tmp_tickets2 = tmpdir / "ledger_tickets.md"
        shutil.copyfile(real, tmp_tickets2)

        port5 = free_port()
        httpd2 = serve("127.0.0.1", port5, ticket_file=tmp_tickets2,
                       html_file=DEFAULT_HTML,
                       meta=build_meta("umvp", ticket.DEFAULT_TITLE, ticket.DEFAULT_STATUSES,
                                       ticket.DEFAULT_AGENTS, tmp_tickets2, False,
                                       inbox_file=inbox_copy, deployment_file=dep_copy),
                       quiet=True)
        base5 = f"http://127.0.0.1:{port5}"
        th2 = threading.Thread(target=httpd2.serve_forever, daemon=True)
        th2.start()
        try:
            st, body = http("GET", f"{base5}/api/meta")
            check(st == 200 and body["inbox_file"] == str(inbox_copy)
                  and body["deployment_file"] == str(dep_copy),
                  "ledger 实例 meta 携带 inbox/deployment 文件路径")

            st, body = http("GET", f"{base5}/api/inbox")
            check(st == 200 and body["ok"] and body["counts"]["total"] == len(es),
                  f"GET /api/inbox 条数一致（{body['counts']['total']}）")
            from urllib.parse import quote
            st, body = http("GET", f"{base5}/api/inbox?status=" + quote("待同步"))
            check(st == 200 and all(x["section"] == "待同步" for x in body["entries"])
                  and len(body["entries"]) == n_pend, "按状态筛选（待同步）")
            st, body = http("GET", f"{base5}/api/inbox?source=web&type=" + quote("成果"))
            check(st == 200 and body["entries"]
                  and all(x["source"] == "web" and "成果" in x["type"] for x in body["entries"]),
                  "按来源+类型筛选")
            st, body = http("GET", f"{base5}/api/inbox?q=WO-0008")
            check(st == 200 and body["entries"], "全文搜索命中")

            # 追加（编号自动递增：两节合取 max+1）
            before_ids = {e["id"] for e in ledger.parse_inbox(inbox_copy)}
            st, body = http("POST", f"{base5}/api/inbox/append",
                            {"source": "web", "type": "成果/部署", "text": "自检追加条目",
                             "followup": "→ 去向：临时副本", "date": "2026-10-20"})
            new_id = body.get("id", "")
            check(st == 200 and body.get("ok") and new_id not in before_ids,
                  f"POST /api/inbox/append 新编号 {new_id}（max+1）")
            es2 = ledger.parse_inbox(inbox_copy)
            check(not ledger.inbox_roundtrip_errors(inbox_copy), "追加后收件箱往返仍无丢字")
            pend = [e for e in es2 if e["section"] == "待同步"]
            check(pend[0]["id"] == new_id and pend[0]["text"] == "自检追加条目",
                  "新条目位于待同步顶部（最新在上）")

            # 流转（待同步 → 已同步 + 去向更新）
            st, body = http("POST", f"{base5}/api/inbox/move",
                            {"id": new_id, "destination": "自检验证"})
            check(st == 200 and body["ok"]
                  and body["entry"]["section"] == "已同步", "POST /api/inbox/move 流转成功")
            es3 = ledger.parse_inbox(inbox_copy)
            moved = next(e for e in es3 if e["id"] == new_id)
            synced = [e for e in es3 if e["section"] == "已同步"]
            check(moved is synced[0], "流转后条目位于已同步顶部")
            check(any("自检验证" in f for f in moved["followups"]), "去向行已写入")
            check(not ledger.inbox_roundtrip_errors(inbox_copy), "流转后往返仍无丢字")

            # 非法输入
            ib = md5(inbox_copy)
            st, body = http("POST", f"{base5}/api/inbox/append",
                            {"source": "web", "type": "成果"})
            check(st == 400 and not body.get("ok"), "缺 text 被拒（400）")
            st, body = http("POST", f"{base5}/api/inbox/move", {"id": "INB-9999"})
            check(st == 400 and "不存在" in body.get("error", ""), "流转不存在条目被拒")
            st, body = http("POST", f"{base5}/api/inbox/move", {"id": new_id})
            check(st == 400 and "已同步" in body.get("error", ""), "重复流转被拒")
            check(md5(inbox_copy) == ib, "被拒后收件箱文件未变")

            # 台账
            st, body = http("GET", f"{base5}/api/deployment")
            check(st == 200 and body["ok"] and body["total"] == len(ds),
                  f"GET /api/deployment 条数一致（{body['total']}）")
            st, body = http("GET", f"{base5}/api/deployment?impact=DB")
            check(st == 200 and body["entries"]
                  and all("DB" in x["impact"] for x in body["entries"]), "按影响面筛选（DB）")
            st, body = http("GET", f"{base5}/api/deployment?impact=" + quote("不存在"))
            check(st == 400, "非法影响面被拒（400）")
            st, body = http("POST", f"{base5}/api/deployment/new",
                            {"title": "WO-0009.1 自检台账登记", "changes": "自检变更\n第二行",
                             "impact": ["入口", "配置"], "notes": "无", "verify": "自检",
                             "date": "2026-10-20"})
            check(st == 200 and body.get("ok") and body["id"] == "DEP-0001",
                  "POST /api/deployment/new 登记为 DEP-0001（最新在前）")
            ds2 = ledger.parse_deployment(dep_copy)
            check(ds2[0]["title"] == "WO-0009.1 自检台账登记"
                  and ds2[0]["weekday"] == "二" and ds2[0]["impact"] == ["入口", "配置"],
                  "新条目插 ## 记录 正下方且四字段/星期/影响面正确")
            check(not ledger.deployment_roundtrip_errors(dep_copy), "登记后台账往返仍无丢字")
            db_before = md5(dep_copy)
            st, body = http("POST", f"{base5}/api/deployment/new",
                            {"title": "x", "changes": "y", "impact": ["不是分类"]})
            check(st == 400 and "影响面" in body.get("error", ""), "非法影响面分类被拒")
            st, body = http("POST", f"{base5}/api/deployment/new", {"changes": "y", "impact": ["DB"]})
            check(st == 400, "缺标题被拒")
            check(md5(dep_copy) == db_before, "被拒后台账文件未变")

            # 统一编号查询
            first_wo = next(t["id"] for t in ticket.read_tickets(tmp_tickets2)[0]
                            if "." not in t["id"])
            st, body = http("GET", f"{base5}/api/entry?id={new_id}")
            check(st == 200 and body["kind"] == "inbox" and body["entry"]["id"] == new_id,
                  f"/api/entry?id={new_id}（INB）")
            st, body = http("GET", f"{base5}/api/entry?id={first_wo}")
            check(st == 200 and body["kind"] == "ticket", f"/api/entry?id={first_wo}（WO）")
            st, body = http("GET", f"{base5}/api/entry?id=DEP-0001")
            check(st == 200 and body["kind"] == "deployment"
                  and body["entry"]["title"] == "WO-0009.1 自检台账登记",
                  "/api/entry?id=DEP-0001（台账）")
            st, body = http("GET", f"{base5}/api/entry?id=BOGUS-1")
            check(st == 400 and not body.get("ok"), "未知编号被拒（400）")
            st, body = http("GET", f"{base5}/api/entry?id=" + wo_entries[0]["wo"])
            check(st == 200 and body["kind"] == "ticket", "/api/entry WO 默认转发工单")

            # 并发追加（flock + 编号递增）
            import concurrent.futures
            n = 8
            cnt_before = len(ledger.parse_inbox(inbox_copy))
            def app(i):
                return http("POST", f"{base5}/api/inbox/append",
                            {"source": "web", "type": "成果", "text": f"并发-{i}"})
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
                results = list(ex.map(app, range(n)))
            check(all(st == 200 and b.get("ok") for st, b in results),
                  f"并发 {n} 次收件箱追加全部成功")
            ids_after = [e["id"] for e in ledger.parse_inbox(inbox_copy)]
            check(len(ids_after) == cnt_before + n, f"并发后条数 +{n}")
            check(len(set(ids_after)) == len(ids_after), "并发后无重复 INB 编号")
            check(not ledger.inbox_roundtrip_errors(inbox_copy), "并发后往返仍无丢字")
        finally:
            httpd2.shutdown()
            httpd2.server_close()
            th2.join(timeout=5)

        # 19c. ledger 配置字段校验 + 配置驱动的 CLI
        bad_inbox_cfg = write_cfg("bad_inbox.json", {"file": "custom_tickets.md", "inbox_file": 123})
        expect_ticket_error(lambda: ticket.load_config(bad_inbox_cfg), "inbox_file",
                            "inbox_file 非字符串/空串报错")
        empty_dep_cfg = write_cfg("empty_dep.json", {"file": "custom_tickets.md", "deployment_file": ""})
        expect_ticket_error(lambda: ticket.load_config(empty_dep_cfg), "deployment_file",
                            "deployment_file 空串报错")
        ok_cfg = write_cfg("ok_ledger.json", {"file": "custom_tickets.md",
                                              "inbox_file": str(inbox_copy),
                                              "deployment_file": str(dep_copy)})
        cfg_l = ticket.load_config(ok_cfg)
        check(cfg_l.inbox_file == inbox_copy.resolve()
              and cfg_l.deployment_file == dep_copy.resolve(), "配置 ledger 两字段路径生效")
        tilde_cfg = write_cfg("tilde_inbox.json", {"file": "custom_tickets.md", "inbox_file": "~/x.md"})
        check(ticket.load_config(tilde_cfg).inbox_file == (Path.home() / "x.md").resolve(),
              "配置 inbox_file 支持 ~ 展开")

        def ledger_cli(*args):
            return subprocess.run([sys.executable, str(TOOL_DIR / "ledger.py"), *args],
                                  capture_output=True, text=True)

        # 19d. ledger CLI（独立副本，显式 --file/--deployment-file/--tickets）
        inbox_c2 = tmpdir / "ledger_inbox2.md"
        dep_c2 = tmpdir / "ledger_dep2.md"
        shutil.copyfile(real_inbox, inbox_c2)
        shutil.copyfile(real_dep, dep_c2)
        cp = ledger_cli("get", "INB-0049", "--file", str(inbox_c2))
        check(cp.returncode == 0 and "INB-0049" in cp.stdout and "WO-0008" in cp.stdout,
              "CLI get INB-0049 输出单条完整内容")
        target_inb = max(int(e["id"][4:]) for e in ledger.parse_inbox(inbox_c2))
        tid = f"INB-{target_inb:04d}"
        cp = ledger_cli("get", tid, "--file", str(inbox_c2))
        check(cp.returncode == 0 and f"[待同步] {tid}" in cp.stdout, f"CLI get {tid}")
        cp = ledger_cli("get", tid, "--file", str(tmpdir / "nonexistent.md"))
        check(cp.returncode == 2 and "不存在" in cp.stderr, "CLI get 文件不存在 rc=2 中文错误")
        cp = ledger_cli("get", first_wo, "--tickets", str(tmp_tickets2))
        check(cp.returncode == 0 and first_wo in cp.stdout, f"CLI get {first_wo}（转发工单解析）")
        cp = ledger_cli("get", "DEP-0001", "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and "[DEP-0001]" in cp.stdout, "CLI get DEP-0001")
        cp = ledger_cli("get", "2026-09-23 抽出", "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and "2026-09-23" in cp.stdout, "CLI 台账模糊查询")
        cp = ledger_cli("get", wo_entries[0]["wo"], "--from-deployment",
                        "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and wo_entries[0]["wo"] in cp.stdout,
              f"CLI get {wo_entries[0]['wo']} --from-deployment 查台账")
        cp = ledger_cli("list", "--file", str(inbox_c2), "--status", "已同步")
        check(cp.returncode == 0 and "INB-0014" in cp.stdout, "CLI list --status 已同步")
        cp = ledger_cli("list", "--deployment", "--impact", "DB", "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and "｜影响面: DB" in cp.stdout, "CLI list --deployment --impact DB")
        cp = ledger_cli("list", "--deployment", "--q", ds[0]["id"], "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and f"[{ds[0]['id']}]" in cp.stdout,
              f"CLI list --deployment --q {ds[0]['id']} 命中（sum→join 回归路径）")
        cp = ledger_cli("list", "--deployment", "--q", "自检绝不命中串XYZ", "--deployment-file", str(dep_c2))
        check(cp.returncode == 0 and "[DEP-" not in cp.stdout, "CLI list --deployment --q 不命中输出为空")
        cp = ledger_cli("append", "--file", str(inbox_c2), "--source", "web", "--type", "成果",
                        "--text", "CLI 自检追加")
        new_cli_id = f"INB-{target_inb + 1:04d}"
        check(cp.returncode == 0 and new_cli_id in cp.stdout, f"CLI append 编号递增 {new_cli_id}")
        cp = ledger_cli("move", new_cli_id, "--destination", "CLI 验证", "--file", str(inbox_c2))
        check(cp.returncode == 0 and "已同步" in cp.stdout, "CLI move 流转")
        cp = ledger_cli("append", "--file", str(inbox_c2), "--source", "web", "--type", "成果")
        check(cp.returncode == 2 and "text" in cp.stderr, "CLI append 缺 text rc=2")
        cp = ledger_cli("append", "--deployment-file", str(dep_c2), "--deployment",
                        "--title", "CLI 台账自检", "--changes", "x", "--impact", "DB,入口",
                        "--date", "2026-10-20")
        check(cp.returncode == 0 and "DEP-0001" in cp.stdout, "CLI 台账登记（--deployment）")
        cp = ledger_cli("append", "--deployment-file", str(dep_c2), "--deployment",
                        "--title", "x", "--changes", "y", "--impact", "瞎选")
        check(cp.returncode == 2 and "影响面" in cp.stderr, "CLI 台账非法影响面 rc=2")
        check(not ledger.inbox_roundtrip_errors(inbox_c2)
              and not ledger.deployment_roundtrip_errors(dep_c2),
              "CLI 写操作后两文件往返仍无丢字")
        cp = ledger_cli("--config", str(ok_cfg), "list")
        check(cp.returncode == 0 and "INB-" in cp.stdout, "CLI 经配置文件读收件箱")
        cp = ledger_cli("get", "INB-9999", "--file", str(inbox_c2))
        check(cp.returncode == 2 and "不存在" in cp.stderr, "CLI get 不存在条目 rc=2")

        # 19e. ticket.py 主单两段式强制 + check 警告（WO-0009.2）
        tk_c = tmpdir / "tk_c.md"
        shutil.copyfile(real, tk_c)
        tk_before = md5(tk_c)
        cp = cli(tk_c, "new", "主单描述", "--by", "web")
        check(cp.returncode == 2 and "--business" in cp.stderr, "CLI 主单缺 --business rc=2 中文报错")
        cp = cli(tk_c, "new", "主单描述", "--by", "web", "--business", "业务")
        check(cp.returncode == 2 and "--approach" in cp.stderr, "CLI 主单缺 --approach rc=2 中文报错")
        check(md5(tk_c) == tk_before, "CLI 拒绝后文件未变（不落盘）")
        cp = cli(tk_c, "new", "主单描述", "--by", "web", "--business", "业务描述",
                 "--approach", "实现方式", "--to", "qa")
        check(cp.returncode == 0 and "已创建" in cp.stdout, "CLI 主单带两参数创建成功")
        created = [t for t in ticket.read_tickets(tk_c)[0]
                   if t["desc"] == "业务：业务描述。实现方式：实现方式。补充：主单描述"]
        check(len(created) == 1, "落盘描述为固定拼装格式（业务：…。实现方式：…。补充：…）")
        cp = cli(tk_c, "new", "带|竖线业务", "--by", "web", "--business", "业|务",
                 "--approach", "做法")
        check(cp.returncode == 2 and "竖线" in cp.stderr, "business 含竖线被拒（拼装不引入竖线）")
        # 子单不带两参数不受影响（既有行为）
        cp = cli(tk_c, "new", "子单描述", "--by", "web", "--parent", created[0]["id"])
        check(cp.returncode == 0 and "已创建" in cp.stdout, "CLI 子单不要求两参数仍可创建")

        # check 警告：进行中主单缺前缀 → stderr ⚠、退出码不变；已完成旧单不回溯
        warn_md = cfgdir / "warn.md"
        warn_md.write_text(
            "# 工单\n\n## 进行中\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n"
            "| WO-0001 | 2026-10-20 | web | - | 旧格式主单无前缀 | - | - |\n\n"
            "## 待测试\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n"
            "| WO-0002 | 2026-10-20 | web | - | 待测试也无前缀 | - | - |\n\n"
            "## 已完成\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n"
            "| WO-0003 | 2026-10-20 | web | - | 已完成旧单无前缀不回溯 | - | - |\n\n"
            "## 已取消\n\n"
            "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8")
        cp = cli(warn_md, "check")
        check(cp.returncode == 0 and "check OK" in cp.stdout, "仅缺前缀时 check 退出码仍 0")
        check("⚠ 主单 WO-0001" in cp.stderr and "⚠ 主单 WO-0002" in cp.stderr,
              "进行中/待测试主单缺前缀 → stderr ⚠ 警告")
        check("WO-0003" not in cp.stderr, "已完成旧单不回溯（无警告）")
        cp = cli(real, "check")
        check(cp.returncode == 0 and "⚠" not in cp.stderr,
              "真实工单当前在办主单均带前缀（check 无 ⚠）")

        # 16e. 模型页签 API（临时 agent 目录 + 临时 opencode.json；主实例此时已关闭，单独起一个）
        httpd_m = serve("127.0.0.1", 0, ticket_file=tmp_file, html_file=DEFAULT_HTML,
                        meta=build_meta("umvp", ticket.DEFAULT_TITLE,
                                        ticket.DEFAULT_STATUSES, ticket.DEFAULT_AGENTS,
                                        tmp_file, False, models_tab=True,
                                        agent_dir=tmp_agents, opencode_config=tmp_opencode),
                        quiet=True)
        port_m = httpd_m.server_address[1]
        base_m = f"http://127.0.0.1:{port_m}"
        threading.Thread(target=httpd_m.serve_forever, daemon=True).start()
        st, body = http("GET", f"{base_m}/api/models")
        check(st == 200 and body.get("ok")
              and [a["name"] for a in body["agents"]] == agent_model.AGENTS,
              "GET /api/models 返回五代理清单")
        check(body["default"]["model"] == agent_model.get_default(tmp_opencode),
              "GET /api/models 默认模型与临时 opencode.json 一致")
        qa_before = (tmp_agents / "qa.md").read_text(encoding="utf-8")
        st, body = http("POST", f"{base_m}/api/models/set",
                        {"agent": "qa", "model": "x/y-test"})
        check(st == 200 and body["ok"] and body["changed"] is True,
              "POST /api/models/set 成功")
        qa_after = (tmp_agents / "qa.md").read_text(encoding="utf-8")
        check(agent_model.get_model("qa", tmp_agents) == "x/y-test",
              "set 后 frontmatter 生效")
        check(qa_after.replace("model: x/y-test\n", "", 1) == qa_before,
              "set 仅新增 model 行，其余逐字不动")
        st, body = http("POST", f"{base_m}/api/models/set",
                        {"agent": "qa", "model": "x/y-test"})
        check(st == 200 and body["changed"] is False, "set 同值幂等")
        st, body = http("POST", f"{base_m}/api/models/set",
                        {"agent": "nobody", "model": "x/y"})
        check(st == 400 and "未知代理" in body["error"], "set 坏代理 400 中文")
        st, body = http("POST", f"{base_m}/api/models/set",
                        {"agent": "qa", "model": "no-slash"})
        check(st == 400 and "模型串非法" in body["error"], "set 坏模型串 400")
        st, body = http("POST", f"{base_m}/api/models/clear", {"agent": "qa"})
        check(st == 200 and body["changed"] is True
              and (tmp_agents / "qa.md").read_text(encoding="utf-8") == qa_before,
              "clear 还原 frontmatter 逐字一致")
        st, body = http("POST", f"{base_m}/api/models/clear", {"agent": "qa"})
        check(st == 200 and body["changed"] is False, "clear 幂等")
        cfg_txt_before = tmp_opencode.read_text(encoding="utf-8")
        st, body = http("POST", f"{base_m}/api/models/default", {"model": "z/z-test"})
        check(st == 200 and body["changed"] is True,
              "POST /api/models/default 成功")
        check(agent_model.get_default(tmp_opencode) == "z/z-test",
              "default 写入临时 opencode.json")
        check(_json.loads(tmp_opencode.read_text(encoding="utf-8")).get("$schema")
              == _json.loads(cfg_txt_before).get("$schema"),
              "default 保留其它键（$schema）")
        httpd_m.shutdown()
        httpd_m.server_close()

        # 17. 真实文件未被污染（工单 + 收件箱 + 部署台账 + 代理/配置）
        real_md5_after = md5(real)
        check(real_md5_after == real_md5_before,
              f"真实 tickets.md 未被修改（md5 {real_md5_before}）")
        check(md5(real_inbox) == real_inbox_md5, "真实 inbox.md 未被修改")
        check(md5(real_dep) == real_dep_md5, "真实 DEPLOYMENT.md 未被修改")
        check({p.name: md5(p) for p in agent_model.DEFAULT_AGENT_DIR.glob("*.md")}
              == real_agents_before, "真实五份代理文件未被模型 API 修改")
        check(md5(agent_model.DEFAULT_CONFIG) == real_cfg_before,
              "真实 opencode.json 未被模型 API 修改")

        print(f"\n自检通过：{passed[0]} 项断言；真实文件 md5（tickets={real_md5_after}）")
        return 0
    except AssertionError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        print(f"（已通过 {passed[0]} 项）", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="工单看板（本机可写页面；配置优先级 CLI > 环境变量 > 配置文件 > 内置默认）")
    ap.add_argument("--config", help="配置文件路径（默认查 $TICKET_CONFIG / ~/.config/ticketd/<name>.json）")
    ap.add_argument("--name", help="实例名（用于查找 ~/.config/ticketd/<name>.json）")
    ap.add_argument("--host", default=None, help="监听地址（默认取配置，内置 127.0.0.1）")
    ap.add_argument("--port", type=int, default=None, help="监听端口（默认取配置，内置 8010）")
    ap.add_argument("--file", type=Path, default=None,
                    help="工单文件路径（默认取配置，内置 .opencode/memory/tickets.md）")
    ap.add_argument("--inbox-file", dest="inbox_file", type=Path, default=None,
                    help="收件箱文件路径（默认取配置 inbox_file；未配置=隐藏收件箱页签）")
    ap.add_argument("--deployment-file", dest="deployment_file", type=Path, default=None,
                    help="部署台账文件路径（默认取配置 deployment_file；未配置=隐藏台账页签）")
    ap.add_argument("--title", default=None, help="看板标题（默认取配置，内置 UMVP 工单看板）")
    ap.add_argument("--readonly", action="store_true", help="只读实例（禁用所有写操作）")
    ap.add_argument("--self-test", action="store_true", help="运行端到端自检后退出")
    a = ap.parse_args(argv)

    try:
        cfg = ticket.load_config_from_args(a.config, a.name)
        if cfg is not None:
            ticket.apply_config(cfg)
    except ticket.TicketError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(2)

    if a.self_test:
        sys.exit(run_self_test())

    name = cfg.name if cfg else ticket.DEFAULT_NAME
    title = a.title if a.title is not None else (cfg.title if cfg else ticket.DEFAULT_TITLE)
    host = a.host if a.host is not None else (cfg.host if cfg else ticket.DEFAULT_HOST)
    port = a.port if a.port is not None else (cfg.port if cfg else ticket.DEFAULT_PORT)
    file = a.file if a.file is not None else (cfg.file if cfg else ticket.DEFAULT_FILE)
    file = Path(file)
    readonly = bool(a.readonly or (cfg.readonly if cfg else False))
    inbox_file = a.inbox_file if a.inbox_file is not None else (
        cfg.inbox_file if cfg else None)
    deployment_file = a.deployment_file if a.deployment_file is not None else (
        cfg.deployment_file if cfg else None)

    if not file.exists():
        print(f"✗ 工单文件不存在: {file}", file=sys.stderr)
        sys.exit(2)
    meta = build_meta(name, title, ticket.STATUSES, ticket.AGENTS, file, readonly,
                      inbox_file=inbox_file, deployment_file=deployment_file,
                      models_tab=(cfg.models_tab if cfg else True))
    httpd = serve(host, port, ticket_file=file, meta=meta)
    url = f"http://{host}:{port}/"
    print(f"{title} → {url}")
    print(f"工单文件: {file}（实例 {name}；Ctrl+C 停止）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
