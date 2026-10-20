#!/usr/bin/env python3
"""UMVP 开发代理·工单工具（stdlib 零依赖）

工单文件默认 .opencode/memory/tickets.md，规约见该文件头与 CLAUDE.md §四。

用法（项目根执行）：
  python3 .opencode/tool/ticket.py new "描述" --by orchestrator --to web \
      [--parent WO-0001] [--verify "命令"] [--note "..."] [--date YYYY-MM-DD]
      主单（无 --parent）另需 --business "业务描述" 与 --approach "实现方式"（缺失即报错，
      描述自动拼装为「业务：<business>。实现方式：<approach>。补充：<desc>」）
  python3 .opencode/tool/ticket.py set WO-0001.2 [--status 待测试] \
      [--append-note "..."]/--note "..."（整字段覆盖） [--verify "..."] [--to "..."]
  python3 .opencode/tool/ticket.py list [--status 进行中] [--parent WO-0001]
  python3 .opencode/tool/ticket.py show WO-0001
  python3 .opencode/tool/ticket.py check   （对进行中/待测试主单缺两段前缀另发 ⚠ 警告）

约定：主单 WO-0001 全局递增；子单 WO-0001.n 挂在主单下；状态 ∈
{进行中, 待测试, 已完成, 已取消}；字段内禁用竖线；新行/移动插入对应状态节表格顶部。

多项目复用（配置来源优先级，从高到低）：
  1. CLI 参数        --file PATH / --config PATH / --name NAME
  2. 环境变量        TICKET_CONFIG（兼容 TICKETD_CONFIG）= 配置文件路径
  3. 配置文件        ~/.config/ticketd/<name>.json（尊重 XDG_CONFIG_HOME）
  4. 内置默认        UMVP（本仓库 .opencode/memory/tickets.md + 默认状态/代理）
配置文件字段：name / title / file（必填，支持 ~，相对路径按配置文件所在目录解析）
  / host / port / statuses / agents / readonly / inbox_file / deployment_file
  / models_tab（布尔，缺省 true = 启用 ticketd「模型」页签，逻辑复用 agent_model.py，
  管理代码所在仓库的五代理 frontmatter 与 opencode.json；false = 隐藏该页签与 API）
  （inbox_file/deployment_file 为 null 或缺省 = 隐藏收件箱/部署台账页签与 API，向后兼容）；
  非法配置 fail-fast
  （中文错误 + 非 0 退出，绝不写工单文件）。不传 --config / 不设环境变量时**功能子命令**
  （new/set/list/show/check）
  的 stdout/stderr/退出码/生成文件与历史版本逐字一致；`--help` 因新增 --config/--name
  两选项而多出数行（有意为之，便于其它项目发现配置入口）。

本模块同时是**唯一真相源**：CLI 与网页（`ticket_web.py`）共用同一实现——
写操作一律走 `apply_new()` / `apply_set()`（含 `_file_lock` 排他锁 + 原子替换），
`cmd_*` 只是薄封装，保证输出与退出码不变。导入方用于只读的入口：
`load_lines` / `scan` / `row_from_line` / `make_row` / `insert_row` / `next_numbers` /
`read_tickets` / `validate_tickets` / `validate_file`。
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

try:  # Unix 文件锁；非 Unix 平台退化为无锁（本工具面向本机 Linux）
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

# 基于脚本自身 __file__ 定位默认工单文件：任意 cwd、经 systemd/包装脚本启动均正确。
_TOOL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parent.parent
DEFAULT_FILE = _REPO_ROOT / ".opencode" / "memory" / "tickets.md"

DEFAULT_STATUSES = ["进行中", "待测试", "已完成", "已取消"]
DEFAULT_AGENTS = ["orchestrator", "pipe-chain", "web", "agent-builder", "qa"]
STATUSES = list(DEFAULT_STATUSES)
AGENTS = list(DEFAULT_AGENTS)
DEFAULT_NAME = "umvp"
DEFAULT_TITLE = "UMVP 工单看板"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
CONFIG_ENV_VARS = ("TICKET_CONFIG", "TICKETD_CONFIG")
ID_RE = re.compile(r"^WO-(\d{4})(?:\.(\d+))?$")
HEADER_ROW = "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |"
SEP_ROW = "|---|---|---|---|---|---|---|"
FIELDS = ["id", "date", "by", "to", "desc", "verify", "note"]


class TicketError(Exception):
    pass


# --------------------------------------------------------------------------- #
# 配置（多项目复用）：CLI 参数 > 环境变量 > 配置文件 > 内置默认
# --------------------------------------------------------------------------- #
class Config:
    """解析后的运行配置（file 为绝对 Path；statuses/agents 已校验）。

    inbox_file / deployment_file：收件箱与部署台账文件（绝对 Path 或 None）；
    None（缺省或显式 null）= 隐藏对应页签与 API（向后兼容旧配置）。

    models_tab：是否启用「模型」页签（布尔，缺省 True）。逻辑在 agent_model.py，
    此处只做开关（复用本工具的其它项目可设 false 关掉 UMVP 专属页签）。
    """

    def __init__(self, name, title, file, host, port, statuses, agents,
                 readonly=False, path=None, inbox_file=None, deployment_file=None,
                 models_tab=True):
        self.name = name
        self.title = title
        self.file = file
        self.host = host
        self.port = port
        self.statuses = statuses
        self.agents = agents
        self.readonly = readonly
        self.path = path
        self.inbox_file = inbox_file
        self.deployment_file = deployment_file
        self.models_tab = models_tab


def config_home(env=None):
    """配置根目录：尊重 XDG_CONFIG_HOME，默认 ~/.config。"""
    env = os.environ if env is None else env
    return Path(env.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))


def default_config_path(name, env=None):
    return config_home(env) / "ticketd" / f"{name}.json"


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_names(value, field):
    if not isinstance(value, list) or not value:
        raise TicketError(f"配置 {field} 应为非空字符串数组")
    out = []
    for i, x in enumerate(value):
        if not isinstance(x, str) or not x.strip():
            raise TicketError(f"配置 {field}[{i}] 应为非空字符串")
        if "|" in x:
            raise TicketError(f"配置 {field}[{i}] 内含竖线 | : {x!r}")
        out.append(x)
    if len(set(out)) != len(out):
        raise TicketError(f"配置 {field} 存在重复项: {value}")
    return out


def _optional_path(raw, key, base_dir, path):
    """可选路径字段：null/缺省 → None；须为非空字符串（~ 展开，相对按 base_dir）。"""
    v = raw.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        raise TicketError(f"配置 {key} 应为非空字符串或 null: {v!r}（{path or '<inline>'}）")
    p = Path(v).expanduser()
    if not p.is_absolute():
        base = Path(base_dir) if base_dir else Path.cwd()
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def config_from_dict(raw, base_dir=None, path=None):
    """校验并归一化配置 dict → Config。非法则抛 TicketError（fail-fast）。"""
    if not isinstance(raw, dict):
        raise TicketError(f"配置应为 JSON 对象: {path or '<inline>'}")
    f = raw.get("file")
    if not isinstance(f, str) or not f.strip():
        raise TicketError(f"配置缺少 file（工单文件路径）: {path or '<inline>'}")
    fp = Path(f).expanduser()
    if not fp.is_absolute():
        base = Path(base_dir) if base_dir else Path.cwd()
        fp = (base / fp).resolve()
    else:
        fp = fp.resolve()
    port = raw.get("port", DEFAULT_PORT)
    if not _is_int(port) or not (1 <= port <= 65535):
        raise TicketError(f"配置 port 非法（应为 1-65535 的整数）: {port!r}")
    statuses = (_validate_names(raw["statuses"], "statuses")
                if "statuses" in raw else list(DEFAULT_STATUSES))
    agents = (_validate_names(raw["agents"], "agents")
              if "agents" in raw else list(DEFAULT_AGENTS))
    name = raw.get("name", DEFAULT_NAME)
    if not isinstance(name, str) or not name.strip():
        raise TicketError(f"配置 name 应为非空字符串: {name!r}")
    title = raw.get("title", DEFAULT_TITLE)
    if not isinstance(title, str):
        raise TicketError(f"配置 title 应为字符串: {title!r}")
    host = raw.get("host", DEFAULT_HOST)
    if not isinstance(host, str) or not host.strip():
        raise TicketError(f"配置 host 应为非空字符串: {host!r}")
    readonly = raw.get("readonly", False)
    if not isinstance(readonly, bool):
        raise TicketError(f"配置 readonly 应为布尔值: {readonly!r}")
    models_tab = raw.get("models_tab", True)
    if not isinstance(models_tab, bool):
        raise TicketError(f"配置 models_tab 应为布尔值: {models_tab!r}")
    return Config(name=name, title=title, file=fp, host=host, port=port,
                  statuses=statuses, agents=agents, readonly=readonly, path=path,
                  inbox_file=_optional_path(raw, "inbox_file", base_dir, path),
                  deployment_file=_optional_path(raw, "deployment_file", base_dir, path),
                  models_tab=models_tab)


def load_config(path):
    """读取并校验配置文件 → Config。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise TicketError(f"配置文件不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise TicketError(f"配置文件 JSON 非法: {p}（{e}）")
    return config_from_dict(raw, base_dir=p.resolve().parent, path=p)


def load_config_from_args(config_arg=None, name_arg=None, env=None):
    """→ Config 或 None（无任何配置来源）。

    优先级：CLI --config > env TICKET_CONFIG/TICKETD_CONFIG > --name 对应默认路径。
    """
    env = os.environ if env is None else env
    path = config_arg
    if not path:
        for var in CONFIG_ENV_VARS:
            if env.get(var):
                path = env[var]
                break
    if not path and name_arg:
        path = str(default_config_path(name_arg, env))
    if not path:
        return None
    return load_config(path)


def apply_config(cfg):
    """把配置里的状态/代理名单设为当前模块运行值（进程级，启动时调用一次）。"""
    global STATUSES, AGENTS
    STATUSES = list(cfg.statuses)
    AGENTS = list(cfg.agents)


def resolve_file(cli_file, cfg):
    """文件路径优先级：CLI --file > 配置文件 file > 内置默认。"""
    if cli_file is not None:
        return Path(cli_file)
    if cfg is not None:
        return Path(cfg.file)
    return DEFAULT_FILE


def parse_id(s):
    m = ID_RE.match(s.strip())
    if not m:
        raise TicketError(f"非法编号: {s!r}（应为 WO-0001 或 WO-0001.2）")
    return s.strip()


def load_lines(path):
    if not path.exists():
        raise TicketError(f"工单文件不存在: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def row_from_line(line):
    """| a | b | ... | → dict；不合法返回 None。"""
    t = line.strip()
    if not (t.startswith("|") and t.endswith("|")):
        return None
    parts = [p.strip() for p in t[1:-1].split("|")]
    if len(parts) != 7:
        return None
    if not ID_RE.match(parts[0]):
        return None
    return dict(zip(FIELDS, parts))


def scan(lines):
    """→ (tickets, insert_at)  tickets=[{...,'section','line'}]  insert_at={状态: 行号}"""
    tickets, insert_at, section = [], {}, None
    for i, line in enumerate(lines):
        m = re.match(r"^##\s*(.+?)\s*$", line)
        if m:
            name = m.group(1).strip()
            section = name if name in STATUSES else None
            continue
        if section and line.strip() == SEP_ROW:
            insert_at[section] = i + 1  # 表头顶部分隔行之后
            continue
        if section:
            row = row_from_line(line)
            if row:
                row["section"] = section
                row["line"] = line
                tickets.append(row)
    return tickets, insert_at


def make_row(t):
    cells = [t[f] for f in FIELDS]
    for c in cells:
        if "|" in c:
            raise TicketError(f"字段内禁用竖线: {c!r}")
    return "| " + " | ".join(cells) + " |"


def next_numbers(tickets):
    """→ (下一主单号, {(主单号): 下一子单序号})"""
    next_main, subs = 1, {}
    for t in tickets:
        m = ID_RE.match(t["id"])
        n, s = int(m.group(1)), m.group(2)
        if s is None:
            next_main = max(next_main, n + 1)
        else:
            subs[n] = max(subs.get(n, 0), int(s))
    subs = {k: v + 1 for k, v in subs.items()}
    return f"WO-{next_main:04d}", subs


def insert_row(lines, insert_at, section, row):
    if section not in insert_at:
        raise TicketError(f"工单文件缺少「## {section}」表格节")
    lines.insert(insert_at[section], row)
    return lines


def new_status():
    """新建工单落入的状态节：列表首个状态（内置默认 = 进行中）。"""
    return STATUSES[0]


def open_statuses():
    """视为「未关闭」的状态（用于主单收尾提醒）：末两个之外（内置默认 = 进行中/待测试）。"""
    return STATUSES[:-2]


def closed_statuses():
    """视为「已关闭」的终态状态（内置默认 = 已完成/已取消）。"""
    return STATUSES[-2:]


# --------------------------------------------------------------------------- #
# 只读入口 / 校验
# --------------------------------------------------------------------------- #
def read_tickets(path):
    """读取并解析工单文件 → (tickets, insert_at)。只读，不加锁。"""
    lines = load_lines(path)
    return scan(lines)


def validate_tickets(tickets, insert_at):
    """结构校验（纯函数）→ 错误信息列表（空=合法）。"""
    errs = []
    if len(insert_at) != len(STATUSES):
        missing = [s for s in STATUSES if s not in insert_at]
        errs.append(f"缺少状态节表格: {missing}")
    seen = set()
    for t in tickets:
        if t["id"] in seen:
            errs.append(f"编号重复: {t['id']}")
        seen.add(t["id"])
        if t["by"] not in AGENTS:
            errs.append(f"{t['id']} 发起人非法: {t['by']}")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", t["date"]):
            errs.append(f"{t['id']} 日期非法: {t['date']}")
        if t["section"] not in STATUSES:
            errs.append(f"{t['id']} 所在节非法: {t['section']}")
        if "." in t["id"] and t["id"].split(".")[0] not in seen - {t["id"]} \
                and not any(x["id"] == t["id"].split(".")[0] for x in tickets):
            errs.append(f"{t['id']} 的父单不存在")
    return errs


def validate_file(path):
    """读取文件并校验 → (errors, count)。"""
    tickets, insert_at = read_tickets(path)
    return validate_tickets(tickets, insert_at), len(tickets)


# --------------------------------------------------------------------------- #
# 写盘（排他锁 + 临时文件原子替换）
# --------------------------------------------------------------------------- #
def _lock_path(path):
    """锁文件放在系统临时目录（按目标文件绝对路径哈希），不污染仓库。"""
    key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"umvp-ticket-{key}.lock"


@contextmanager
def _file_lock(path, exclusive=True):
    """对目标文件的锁文件加 flock（读-改-写全程持锁）。非 Unix 退化为无锁。"""
    if fcntl is None:
        yield
        return
    lock_path = _lock_path(path)
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as e:
        raise TicketError(f"无法创建锁文件 {lock_path}: {e}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_atomic(path, lines):
    """临时文件 + os.replace 原子替换（内容与 Path.write_text 一致，且保留原文件权限）。"""
    text = "\n".join(lines) + "\n"
    d = str(path.parent) or "."
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:  # mkstemp 固定 0600：沿用原文件权限；新建文件用 0644（旧 write_text 行为）
            mode = os.stat(path).st_mode & 0o7777
        except OSError:
            mode = 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _guard_write(lines):
    """写前结构校验（validate=True 时用）；不合法则抛 TicketError，绝不落盘。"""
    tickets, insert_at = scan(lines)
    errs = validate_tickets(tickets, insert_at)
    if errs:
        raise TicketError("写前校验失败：" + "；".join(errs))


# --------------------------------------------------------------------------- #
# 核心写操作（CLI 与网页共用）
# --------------------------------------------------------------------------- #
def apply_new(path, desc, by, to=None, parent=None, verify=None, note=None,
              date=None, validate=False, business=None, approach=None):
    """新建工单（进行中）→ 返回新工单 dict（键同 FIELDS + section）。

    参数与 CLI `new` 一一对应；`validate=True` 时写前做结构校验（不合法不落盘）。
    **主单**（无 parent）必须给 business/approach，描述拼装为
    「业务：<business>。实现方式：<approach>。补充：<desc>」固定前缀格式；
    子单不要求、不拼前缀。
    """
    if by not in AGENTS:
        raise TicketError(f"--by 应为 {'/'.join(AGENTS)}，收到 {by!r}")
    if parent is None:
        business = (business or "").strip()
        approach = (approach or "").strip()
        if not business:
            raise TicketError("主单（无 --parent）必须提供 --business（业务描述：要解决什么问题）")
        if not approach:
            raise TicketError("主单（无 --parent）必须提供 --approach（实现方式：打算怎么做）")
        desc = f"业务：{business}。实现方式：{approach}。补充：{desc}"
    if "|" in desc or (note and "|" in note) or (verify and "|" in verify):
        raise TicketError("字段内禁用竖线 |")
    with _file_lock(path):
        lines = load_lines(path)
        tickets, insert_at = scan(lines)
        main, subs = next_numbers(tickets)
        if parent:
            p = parse_id(parent)
            if "." in p:
                raise TicketError("--parent 必须是主单（WO-0001 形式）")
            if not any(t["id"] == p for t in tickets):
                raise TicketError(f"父单不存在: {p}")
            tid = f"{p}.{subs.get(int(p[3:]), 1)}"
        else:
            tid = main
        t = {"id": tid, "date": date or datetime.date.today().isoformat(),
             "by": by, "to": to or "", "desc": desc,
             "verify": verify or "", "note": note or ""}
        lines = insert_row(lines, insert_at, new_status(), make_row(t))
        if validate:
            _guard_write(lines)
        _write_atomic(path, lines)
    t["section"] = new_status()
    return t


def apply_set(path, ticket, status=None, to=None, verify=None, note=None,
              append_note=None, validate=False):
    """更新工单（状态/承接/验证/备注覆盖/备注追加）。

    → 返回 {"id", "target", "warning", "ticket"}；`warning` 为主单收尾提醒文案，
    与 CLI 一样**已打印到 stderr**（保持 CLI 逐字行为）。`validate=True` 写前校验。
    """
    tid = parse_id(ticket)
    with _file_lock(path):
        lines = load_lines(path)
        tickets, insert_at = scan(lines)
        cur = next((t for t in tickets if t["id"] == tid), None)
        if cur is None:
            raise TicketError(f"工单不存在: {tid}")
        if status and status not in STATUSES:
            raise TicketError(f"--status 应为 {'/'.join(STATUSES)}")
        t = dict(cur)
        for f, v in (("to", to), ("verify", verify), ("note", note)):  # 仅这三字段可更新
            if v is not None:
                t[f] = v
        if append_note:
            t["note"] = f"{cur['note']}；{append_note}" if cur["note"] else append_note
        if "|" in t["note"] or "|" in t["verify"]:
            raise TicketError("字段内禁用竖线 |")
        # 主单收尾检查：仍有未关闭子单时警告（不阻断）
        warning = None
        if status in closed_statuses() and "." not in tid:
            open_subs = [x["id"] for x in tickets
                         if x["id"].startswith(tid + ".") and x["section"] in open_statuses()]
            if open_subs:
                warning = f"⚠ 主单 {tid} 仍有未关闭子单: {', '.join(open_subs)}"
                print(warning, file=sys.stderr)
        target = status or cur["section"]
        lines = [l for l in lines if l != cur["line"]]
        tickets2, insert_at2 = scan(lines)
        lines = insert_row(lines, insert_at2, target, make_row(t))
        if validate:
            _guard_write(lines)
        _write_atomic(path, lines)
    t["section"] = target
    return {"id": tid, "target": target, "warning": warning, "ticket": t}


# --------------------------------------------------------------------------- #
# CLI（薄封装，输出/退出码与历史版本逐字一致）
# --------------------------------------------------------------------------- #
def cmd_new(a):
    t = apply_new(a.file, a.desc, a.by, to=a.to, parent=a.parent,
                  verify=a.verify, note=a.note, date=a.date,
                  business=a.business, approach=a.approach)
    print(f"已创建 {t['id']}（{new_status()}）→ 承接: {a.to or '-'}")


def cmd_set(a):
    r = apply_set(a.file, a.ticket, status=a.status, to=a.to, verify=a.verify,
                  note=a.note, append_note=a.append_note)
    print(f"{r['id']} → {r['target']}" + (f"（备注: {a.note}）" if a.note else ""))


def cmd_list(a):
    if a.status and a.status not in STATUSES:
        raise TicketError(f"--status 应为 {'/'.join(STATUSES)}")
    tickets, _ = scan(load_lines(a.file))
    shown = 0
    for t in tickets:
        if a.status and t["section"] != a.status:
            continue
        if a.parent and not t["id"].startswith(parse_id(a.parent) + ".") and t["id"] != parse_id(a.parent):
            continue
        print(f"[{t['section']}] {t['id']} {t['date']} {t['by']}→{t['to'] or '-'}: {t['desc']}"
              + (f" ｜备注: {t['note']}" if t["note"] else ""))
        shown += 1
    if not shown:
        print("（无匹配工单）")


def cmd_show(a):
    tid = parse_id(a.ticket)
    tickets, _ = scan(load_lines(a.file))
    main = next((t for t in tickets if t["id"] == tid), None)
    if main is None:
        raise TicketError(f"工单不存在: {tid}")
    for t in [main] + [x for x in tickets if x["id"].startswith(tid + ".")]:
        print(f"[{t['section']}] {t['id']} {t['date']} 发起:{t['by']} 承接:{t['to'] or '-'}")
        print(f"  描述: {t['desc']}")
        if t["verify"]:
            print(f"  验证: {t['verify']}")
        if t["note"]:
            print(f"  备注: {t['note']}")


def cmd_check(a):
    errs, count = validate_file(a.file)
    for e in errs:
        print("✗", e, file=sys.stderr)
    # 主单两段式前缀提醒（WO-0009.2）：进行中/待测试的主单缺前缀 → stderr 警告，
    # 不改变退出码；已完成/已取消的旧单不回溯。
    for t in read_tickets(a.file)[0]:
        if "." in t["id"] or t["section"] not in open_statuses():
            continue
        if not (t["desc"].startswith("业务：") and "。实现方式：" in t["desc"]):
            print(f"⚠ 主单 {t['id']} 描述缺「业务：/实现方式：」两段前缀"
                  f"（新建主单请用 new --business/--approach；旧单补齐请人工编辑）",
                  file=sys.stderr)
    print(f"check OK：{count} 条工单，结构合法" if not errs else f"check 失败：{len(errs)} 处问题",
          file=sys.stderr if errs else sys.stdout)
    sys.exit(1 if errs else 0)


def main():
    ap = argparse.ArgumentParser(description="UMVP 工单工具")
    ap.add_argument("--file", type=Path, default=None,
                    help="工单文件路径（默认 .opencode/memory/tickets.md）")
    # 多项目复用：显式列出（可发现性优先；功能子命令输出仍与历史版本逐字一致）。
    ap.add_argument("--config", default=None,
                    help="配置文件路径（JSON；优先级 高于 TICKET_CONFIG/TICKETD_CONFIG 与内置默认）")
    ap.add_argument("--name", default=None,
                    help="实例名，读 ~/.config/ticketd/<name>.json（尊重 XDG_CONFIG_HOME）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="新建工单（默认进行中）")
    p.add_argument("desc", help="需求/改动描述（主单会拼入「补充：」段）")
    p.add_argument("--by", required=True, help="发起人（代理名）")
    p.add_argument("--to", help="承接方（逗号分隔代理名）")
    p.add_argument("--parent", help="父主单编号（生成子单 WO-xxxx.n）")
    p.add_argument("--business", help="业务描述（主单必填：要解决什么业务问题）")
    p.add_argument("--approach", help="实现方式（主单必填：打算怎么实现）")
    p.add_argument("--verify", help="验证命令/方式")
    p.add_argument("--note", help="备注")
    p.add_argument("--date", help="日期覆盖（YYYY-MM-DD）")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("set", help="更新工单（状态/备注/验证/承接）")
    p.add_argument("ticket", help="工单编号")
    p.add_argument("--status", help=f"{'/'.join(STATUSES)}")
    p.add_argument("--note", help="备注（整字段覆盖，慎用）")
    p.add_argument("--append-note", dest="append_note", help="备注（追加到现有备注，推荐）")
    p.add_argument("--verify", help="验证（整字段覆盖）")
    p.add_argument("--to", help="承接（整字段覆盖）")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("list", help="列出工单")
    p.add_argument("--status", help="按状态过滤")
    p.add_argument("--parent", help="只看某主单及其子单")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="查看工单及其子单")
    p.add_argument("ticket")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="校验工单文件（断言式，退出码 0/1）")
    p.set_defaults(func=cmd_check)

    a = ap.parse_args()
    try:
        cfg = load_config_from_args(a.config, a.name)
        if cfg is not None:
            apply_config(cfg)
        a.file = resolve_file(a.file, cfg)
        a.func(a)
    except TicketError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
