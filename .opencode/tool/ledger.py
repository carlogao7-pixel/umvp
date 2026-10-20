#!/usr/bin/env python3
"""UMVP 统一记录 CLI（ledger.py）—— 收件箱 / 工单 / 部署台账 按编号取单条（stdlib 零依赖）

背景：inbox 条目、工单备注、DEPLOYMENT 台账三处曾全文互抄；本工具让各代理
**按编号取单条**（get INB-xxxx / WO-xxxx / DEP-xxxx），文件仍为唯一真相源，
不迁移、不改写历史条目。

数据文件与格式契约：
  inbox.md（默认 .opencode/memory/inbox.md）
    - 文件头为说明（勿动）；数据在「## 待同步」「## 已同步」两节；
    - 条目 = `- [YYYY-MM-DD][INB-xxxx][来源代理][类型] 结论文本`，其后可跟缩进续行
      （如 `  → 去向：...`）；条目之间常以空行分隔（已同步节亦有紧邻排列）；
    - 新条目追加在「待同步」节注释行 `<!-- 新条目追加在此行下方，最新在上 -->` 下方；
    - INB 编号全局递增（两节合起来取 max+1）；「待同步→已同步」= 整条剪切到已同步节
      顶部（其注释行下方），可在去向行注明去向。
  DEPLOYMENT.md（默认仓库根）
    - 文件头为维护规则+判断清单+格式模板（勿动）；数据在「## 记录」节；
    - 每条 = `### YYYY-MM-DD(weekday) · 标题`（老条目可无 weekday）+ 四字段块
      `- 变更内容：/- 影响面：/- 部署注意：/- 验证：`（值可多行缩进续行；历史尾部
      存在同字段重复的孤儿块，按出现顺序合并展示）；
    - 新条目插在「## 记录」标题正下方（最新在前）；标题内含 WO 号的按 WO 号索引，
      无 WO 号的老条目按「日期+标题」模糊匹配；解析顺序分配 DEP-0001…（最新在前）。
  tickets.md：复用 ticket.py 解析（get WO-xxxx 转发同一实现）。

用法（项目根执行；**写测试务必显式 --file/--deployment-file 指向临时副本**，INB-0039 教训）：
  python3 .opencode/tool/ledger.py get INB-0049 [--file <inbox副本>]
  python3 .opencode/tool/ledger.py get WO-0009  [--tickets <tickets副本>]
  python3 .opencode/tool/ledger.py get DEP-0001 [--deployment-file <台账副本>]
  python3 .opencode/tool/ledger.py get WO-0008 --from-deployment   # 按标题内 WO 号查台账条目
  python3 .opencode/tool/ledger.py get "2026-09-23 抽出目标裁剪"    # 台账模糊（日期+标题）
  python3 .opencode/tool/ledger.py list [--status 待同步] [--type 成果] [--source web]
                                        [--since 2026-10-16] [--q 关键字]
  python3 .opencode/tool/ledger.py list --deployment [--impact DB] [--since 2026-10-01] [--q 关键字]
  python3 .opencode/tool/ledger.py append --source web --type 成果 --text "一句话结论"
                                        [--followup "→ 去向：..."] [--date YYYY-MM-DD]
  python3 .opencode/tool/ledger.py append --deployment --title "标题" --changes "改了什么"
                                        --impact DB,入口 [--notes "部署注意"] [--verify "命令"]
                                        [--date YYYY-MM-DD]
  python3 .opencode/tool/ledger.py move INB-0049 [--destination "去向说明"]

文件路径解析优先级（与 ticket.py 生态一致）：CLI 参数 > 配置文件（--config/--name 或
$TICKET_CONFIG，取其中 inbox_file/deployment_file）> 内置默认（按脚本自身位置定位仓库）。

导入复用（ticket_web.py 同源）：parse_inbox / parse_deployment / filter_inbox /
filter_deployment / get_entry / apply_inbox_append / apply_inbox_move /
apply_deployment_append / inbox_roundtrip_errors / deployment_roundtrip_errors。
写操作复用 ticket._file_lock + _write_atomic（flock 排他锁 + 临时文件原子替换）。
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ticket  # noqa: E402  （同目录：锁/原子写/工单解析复用）

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
DEFAULT_INBOX = REPO_ROOT / ".opencode" / "memory" / "inbox.md"
DEFAULT_DEPLOYMENT = REPO_ROOT / "DEPLOYMENT.md"

INBOX_SECTIONS = ("待同步", "已同步")
APPEND_ANCHOR = "<!-- 新条目追加在此行下方，最新在上 -->"
MOVED_ANCHOR_PREFIX = "<!-- /curate 处理后移到这里"
ENTRY_RE = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\]\[(INB-\d{4})\]\[([^\]]+)\]\[([^\]]+)\]\s?(.*)$")
DEP_TITLE_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})(?:\(([一二三四五六日])\))?\s*·\s*(.*)$")
DEP_FIELD_RE = re.compile(r"^- (变更内容|影响面|部署注意|验证)：(.*)$")
WO_IN_TITLE_RE = re.compile(r"WO-\d{4}(?:\.\d+)?")
WEEKDAYS = "一二三四五六日"
IMPACT_CATEGORIES = ["依赖", "模型", "环境变量", "DB", "标定", "入口", "结构", "配置"]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LedgerError(ticket.TicketError):
    """ledger 专用错误（继承 TicketError：ticket_web 的统一异常处理可直接捕获）。"""


# --------------------------------------------------------------------------- #
# 载入 / 收件箱解析
# --------------------------------------------------------------------------- #
def _load_lines(path, what):
    p = Path(path)
    if not p.exists():
        raise LedgerError(f"{what}文件不存在: {p}")
    try:
        return p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise LedgerError(f"{what}文件读取失败: {p}（{e}）")


def parse_inbox_lines(lines):
    """解析 inbox 行列表 → 条目 dict 列表（保持文件顺序）。

    条目键：id/date/source/type/text/followups/section/lines（原始行，往返用）。
    """
    entries, section, cur = [], None, None

    def close(end):
        nonlocal cur
        if cur is not None:
            block = lines[cur["_start"]:end]
            cur["lines"] = block
            cur["followups"] = [l for l in block[1:] if l.strip()]
            del cur["_start"]
            entries.append(cur)
            cur = None

    for idx, line in enumerate(lines):
        m = re.match(r"^##(?!#)\s*(.+?)\s*$", line)
        if m:
            close(idx)
            name = m.group(1).strip()
            section = name if name in INBOX_SECTIONS else None
            continue
        em = ENTRY_RE.match(line)
        if em and section:
            close(idx)
            cur = {"date": em.group(1), "id": em.group(2), "source": em.group(3),
                   "type": em.group(4), "text": em.group(5).rstrip(), "section": section,
                   "_start": idx}
    close(len(lines))
    return entries


def parse_inbox(path):
    return parse_inbox_lines(_load_lines(path, "收件箱"))


def render_inbox_entry(e):
    """结构 → 原始行（往返断言用；followups 逐字保留缩进）。"""
    out = [f"- [{e['date']}][{e['id']}][{e['source']}][{e['type']}] {e['text']}"]
    out.extend(e["followups"])
    return out


def _rstrip_blanks(block):
    while block and not block[-1].strip():
        block = block[:-1]
    return block


def inbox_roundtrip_errors(path):
    """逐条结构渲染 vs 原始行（去尾部空行）比对 → 不一致描述列表（空=无丢字）。"""
    errs = []
    for e in parse_inbox(path):
        raw = "\n".join(_rstrip_blanks(e["lines"]))
        ren = "\n".join(render_inbox_entry(e))
        if raw != ren:
            errs.append(f"{e['id']} 结构渲染与原文不一致")
    return errs


# --------------------------------------------------------------------------- #
# 部署台账解析
# --------------------------------------------------------------------------- #
def parse_deployment_lines(lines):
    """解析 DEPLOYMENT 行列表 → 台账条目列表（文件顺序 = 最新在前）。

    条目键：id(DEP-xxxx)/date/weekday/title/wo/impact/values{四字段}/fields/lines。
    fields = [{key, first, cont, extra}] 按出现顺序（含历史同字段重复块）；extra 为
    字段块之后不匹配任何规则的原始行（逐字保留，往返用）。
    """
    entries, in_records, cur = [], False, None

    def close(end):
        nonlocal cur
        if cur is not None:
            cur["lines"] = lines[cur["_start"]:end]
            del cur["_start"]
            entries.append(cur)
            cur = None

    for idx, line in enumerate(lines):
        m = re.match(r"^##(?!#)\s*(.+?)\s*$", line)
        if m:
            close(idx)
            in_records = m.group(1).strip() == "记录"
            continue
        if not in_records:
            continue
        tm = DEP_TITLE_RE.match(line)
        if tm:
            close(idx)
            cur = {"date": tm.group(1), "weekday": tm.group(2) or "",
                   "title": tm.group(3).strip(), "fields": [], "_start": idx}
            continue
        if cur is None:
            continue
        fm = DEP_FIELD_RE.match(line)
        if fm:
            cur["fields"].append({"key": fm.group(1), "first": fm.group(2),
                                  "cont": [], "extra": []})
            continue
        if cur["fields"] and (line[:1] in (" ", "\t")):
            cur["fields"][-1]["cont"].append(line)
            continue
        if cur["fields"]:  # 空行/未知行：逐字挂到最后一个字段块（往返保真）
            cur["fields"][-1]["extra"].append(line)
    close(len(lines))

    for i, e in enumerate(entries, 1):
        e["id"] = f"DEP-{i:04d}"
        wm = WO_IN_TITLE_RE.search(e["title"])
        e["wo"] = wm.group(0) if wm else ""
        values = {}
        for f in e["fields"]:
            v = "\n".join([f["first"]] + [l[2:] if l.startswith("  ") else l.lstrip()
                                          for l in f["cont"] if l.strip()] +
                          [l for l in f["extra"] if l.strip()])
            values[f["key"]] = v if f["key"] not in values else f"{values[f['key']]}\n{v}"
        e["values"] = values
        scope = values.get("影响面", "")
        e["impact"] = [c for c in IMPACT_CATEGORIES if c in scope]
    return entries


def parse_deployment(path):
    return parse_deployment_lines(_load_lines(path, "部署台账"))


def render_deployment_entry(e):
    out = [f"### {e['date']}" + (f"({e['weekday']})" if e["weekday"] else "") + f" · {e['title']}"]
    for f in e["fields"]:
        out.append(f"- {f['key']}：{f['first']}")
        out.extend(f["cont"])
        out.extend(f["extra"])
    while out and not out[-1].strip():  # 尾部空行不参与结构比对
        out.pop()
    return out


def deployment_roundtrip_errors(path):
    errs = []
    for e in parse_deployment(path):
        raw = "\n".join(_rstrip_blanks(e["lines"]))
        ren = "\n".join(render_deployment_entry(e))
        if raw != ren:
            errs.append(f"{e['id']}({e['title'][:24]}…) 结构渲染与原文不一致")
    return errs


# --------------------------------------------------------------------------- #
# 筛选（CLI 与 web 共用）
# --------------------------------------------------------------------------- #
def filter_inbox(entries, status=None, type_=None, source=None, since=None, q=None):
    out = []
    for e in entries:
        if status and e["section"] != status:
            continue
        if type_ and type_ not in e["type"]:
            continue
        if source and e["source"] != source:
            continue
        if since and e["date"] < since:
            continue
        if q:
            hay = " ".join([e["id"], e["date"], e["source"], e["type"], e["text"],
                            *e["followups"]]).lower()
            if q.lower() not in hay:
                continue
        out.append(e)
    return out


def filter_deployment(entries, impact=None, since=None, q=None):
    out = []
    for e in entries:
        if impact and impact not in e["impact"]:
            continue
        if since and e["date"] < since:
            continue
        if q:
            hay = " ".join([e["id"], e["date"], e["title"], e.get("wo") or "",
                            *"".join(e["values"].values())]).lower()
            if q.lower() not in hay:
                continue
        out.append(e)
    return out


# --------------------------------------------------------------------------- #
# 统一查询（INB / WO / DEP / 台账模糊）
# --------------------------------------------------------------------------- #
def _find_inbox(entries, ident):
    for e in entries:
        if e["id"] == ident:
            return e
    raise LedgerError(f"收件箱条目不存在: {ident}")


def _ticket_detail(tickets_file, ident):
    """WO-xxxx → (主单, 子单列表)，复用 ticket.py 解析。"""
    tid = ticket.parse_id(ident)
    tickets, _ = ticket.read_tickets(tickets_file)
    main = next((t for t in tickets if t["id"] == tid), None)
    if main is None:
        raise LedgerError(f"工单不存在: {tid}")
    return main, [x for x in tickets if x["id"].startswith(tid + ".")]


def _find_deployment_fuzzy(entries, query):
    q = query.strip()
    dm = re.match(r"^(\d{4}-\d{2}-\d{2})\s*(.*)$", q)
    date, rest = (dm.group(1), dm.group(2).strip()) if dm else ("", q)
    hits = []
    for e in entries:
        if date and e["date"] != date:
            continue
        title_hit = (rest in e["title"]) if rest else True
        wo_hit = bool(rest) and rest == e.get("wo")
        if title_hit or wo_hit:
            hits.append(e)
    if not hits:
        raise LedgerError(f"台账无匹配条目: {query!r}（可用 `list --deployment` 查看全部）")
    return hits


def get_entry(ident, inbox_file=None, tickets_file=None, deployment_file=None,
              force_deployment=False):
    """统一编号查询 → {"kind": inbox|ticket|deployment, ...}；台账模糊查询可多条。

    编号歧义约定：WO-xxxx 默认转发**工单**（与 ticket.py 同源）；要按标题内 WO 号
    查**台账条目**用 force_deployment=True（或 CLI `get --from-deployment`）。
    """
    ident = ident.strip()
    if not force_deployment:
        if re.match(r"^INB-\d{4}$", ident, re.I):
            if not inbox_file:
                raise LedgerError("实例未配置收件箱文件（inbox），无法查询 INB 编号")
            return {"kind": "inbox", "entry": _find_inbox(parse_inbox(inbox_file), ident.upper())}
        if ticket.ID_RE.match(ident):
            if not tickets_file:
                raise LedgerError("实例未配置工单文件（tickets），无法查询 WO 编号")
            main, children = _ticket_detail(tickets_file, ident)
            return {"kind": "ticket", "ticket": main, "children": children}
    if not force_deployment and re.match(r"^DEP-\d{4}$", ident, re.I):
        if not deployment_file:
            raise LedgerError("实例未配置部署台账文件（deployment），无法查询 DEP 编号")
        entries = parse_deployment(deployment_file)
        for e in entries:
            if e["id"] == ident.upper():
                return {"kind": "deployment", "entry": e, "matches": [e]}
        raise LedgerError(f"台账条目不存在: {ident.upper()}（共 {len(entries)} 条）")
    # 其余 → 台账查询（DEP 精确 / 标题内 WO 号 / 日期+标题模糊）
    if not deployment_file:
        raise LedgerError("实例未配置部署台账文件（deployment），无法查询台账")
    entries = parse_deployment(deployment_file)
    dm = re.match(r"^(DEP-\d{4})$", ident, re.I)
    if dm:
        for e in entries:
            if e["id"] == dm.group(1).upper():
                return {"kind": "deployment", "entry": e, "matches": [e]}
        raise LedgerError(f"台账条目不存在: {dm.group(1).upper()}（共 {len(entries)} 条）")
    hits = _find_deployment_fuzzy(entries, ident)
    return {"kind": "deployment", "entry": hits[0], "matches": hits}


# --------------------------------------------------------------------------- #
# 写操作（flock 排他锁 + 原子替换；与 ticket.py 同一模式）
# --------------------------------------------------------------------------- #
def _section_anchor(lines, section, anchor_prefix=None, exact=None):
    """→ (节内锚注释行号, 首个非空行号)。锚不存在时报错（契约要求锚注释在）。"""
    heading_re = re.compile(r"^##(?!#)\s*(.+?)\s*$")
    sec_start, sec_end, in_sec = None, None, False
    anchor_idx = None
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            if in_sec:
                sec_end = i
                break
            in_sec = m.group(1).strip() == section
            if in_sec:
                sec_start = i
            continue
        if in_sec:
            if exact is not None and line.strip() == exact:
                anchor_idx = i
            elif anchor_prefix and line.strip().startswith(anchor_prefix):
                anchor_idx = i
    if sec_end is None:
        sec_end = len(lines)
    if sec_start is None:
        raise LedgerError(f"文件缺少「## {section}」节")
    if anchor_idx is None or not (sec_start < anchor_idx < sec_end):
        tip = f"（需含锚注释 {'`' + exact + '`' if exact else '以 `' + anchor_prefix + '` 开头'}）"
        raise LedgerError(f"「## {section}」节缺少追加锚注释{tip}")
    insert_at = anchor_idx + 1
    while insert_at < sec_end and not lines[insert_at].strip():
        insert_at += 1
    return anchor_idx, insert_at


def _next_inb_number(entries):
    mx = 0
    for e in entries:
        m = re.match(r"^INB-(\d+)$", e["id"])
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def _validate_inbox_mutation(lines, what):
    """写前防丢字断言：结构渲染往返全绿才允许落盘。"""
    entries = parse_inbox_lines(lines)
    for e in entries:
        if "\n".join(render_inbox_entry(e)) != "\n".join(_rstrip_blanks(e["lines"])):
            raise LedgerError(f"写前校验失败：{what} 将破坏条目 {e['id']} 的原文，已放弃落盘")


def apply_inbox_append(path, source, type_, text, followup=None, date=None):
    """追加收件箱新条目（待同步顶部，最新在上）→ 新条目 dict。"""
    text = (text or "").strip()
    source = (source or "").strip()
    type_ = (type_ or "").strip()
    if not text:
        raise LedgerError("缺少条目文本（--text）")
    if "\n" in text or "\n" in source or "\n" in type_:
        raise LedgerError("首行字段不允许换行（多行内容请用续行 --followup）")
    if not source or not type_:
        raise LedgerError("缺少 --source 或 --type")
    date = date or datetime.date.today().isoformat()
    if not DATE_RE.match(date):
        raise LedgerError(f"--date 应为 YYYY-MM-DD: {date!r}")
    if followup is not None:
        followup = followup.strip("\n")
        if not followup.strip():
            followup = None
    with ticket._file_lock(path):
        lines = _load_lines(path, "收件箱")
        entries = parse_inbox_lines(lines)
        nid = f"INB-{_next_inb_number(entries):04d}"
        _, insert_at = _section_anchor(lines, "待同步", exact=APPEND_ANCHOR)
        block = [f"- [{date}][{nid}][{source}][{type_}] {text}"]
        if followup:
            block.append(f"  {followup}")
        block.append("")
        new_lines = lines[:insert_at] + block + lines[insert_at:]
        _validate_inbox_mutation(new_lines, "追加收件箱")
        ticket._write_atomic(path, new_lines)
    e = {"id": nid, "date": date, "source": source, "type": type_, "text": text,
         "followups": ([f"  {followup}"] if followup else []), "section": "待同步",
         "lines": block}
    return e


def _inbox_entry_span(lines, target_idx):
    """条目行区间 [start, end)：到下一条目 / 节标题 / EOF（尾部空行归本条块）。"""
    n = len(lines)
    end = n
    for j in range(target_idx + 1, n):
        if ENTRY_RE.match(lines[j]) or re.match(r"^##\s", lines[j]):
            end = j
            break
    return end


def apply_inbox_move(path, ident, destination=None):
    """待同步 → 已同步：整条剪切到已同步节顶部；destination 写入/更新去向行。"""
    ident = ident.strip().upper()
    if not re.match(r"^INB-\d{4}$", ident):
        raise LedgerError(f"非法收件箱编号: {ident!r}（应为 INB-0049 形式）")
    if destination is not None:
        destination = destination.strip()
    with ticket._file_lock(path):
        lines = _load_lines(path, "收件箱")
        entries = parse_inbox_lines(lines)
        e = _find_inbox(entries, ident)
        if e["section"] != "待同步":
            raise LedgerError(f"{ident} 已在「已同步」节，无需流转")
        # 精确定位（避免同文本多义）：按解析时的行块内容整体匹配
        for i, line in enumerate(lines):
            if line == e["lines"][0] and "\n".join(lines[i:i + len(e["lines"])]) == \
                    "\n".join(e["lines"]):
                start = i
                break
        if start is None:
            raise LedgerError(f"{ident} 原文定位失败（文件在解析后被外部修改？）")
        end = _inbox_entry_span(lines, start)
        block = [l for l in lines[start:end]]
        # 去掉块尾空行（插入时统一补一个分隔空行）
        block = _rstrip_blanks(block)
        # 去向行：有则改写、无则追加
        if destination:
            replaced = False
            for k, l in enumerate(block[1:], 1):
                if l.strip().startswith("→ 去向：") or l.strip().startswith("→ 处置："):
                    block[k] = f"  → 去向：{destination}"
                    replaced = True
                    break
            if not replaced:
                block.append(f"  → 去向：{destination}")
        del lines[start:end]
        # 已同步节顶部插入（重扫锚——删除后行号已变）
        _, insert_at = _section_anchor(lines, "已同步", anchor_prefix=MOVED_ANCHOR_PREFIX)
        lines[insert_at:insert_at] = block + [""]
        _validate_inbox_mutation(lines, f"流转 {ident}")
        ticket._write_atomic(path, lines)
    new_e = dict(e)
    new_e["section"] = "已同步"
    if destination:
        new_e["followups"] = [f"  → 去向：{destination}"] + [
            f for f in new_e["followups"] if not f.strip().startswith("→ 去向：")]
    return new_e


def apply_deployment_append(path, title, changes, impact, notes=None, verify=None,
                            date=None):
    """台账登记新条目（插「## 记录」正下方，最新在前）→ 新条目 dict。"""
    title = (title or "").strip()
    changes = (changes or "").strip()
    if not title:
        raise LedgerError("缺少 --title（一句话标题）")
    if not changes:
        raise LedgerError("缺少 --changes（变更内容）")
    if "|" in title:
        raise LedgerError("标题内禁用竖线 |（标题进工单备注引用时会同字段）")
    if isinstance(impact, str):
        impact = [x.strip() for x in impact.split(",") if x.strip()]
    impact = impact or []
    for c in impact:
        if c not in IMPACT_CATEGORIES:
            raise LedgerError(f"影响面取值非法: {c!r}（应为 {'/'.join(IMPACT_CATEGORIES)}，逗号分隔）")
    if not impact:
        raise LedgerError("缺少 --impact（影响面，至少一项：" + "/".join(IMPACT_CATEGORIES) + "）")
    date = date or datetime.date.today().isoformat()
    if not DATE_RE.match(date):
        raise LedgerError(f"--date 应为 YYYY-MM-DD: {date!r}")
    try:
        weekday = WEEKDAYS[datetime.date.fromisoformat(date).weekday()]
    except ValueError:
        raise LedgerError(f"--date 非法日期: {date!r}")

    def field_block(key, value):
        vlines = [l for l in (value or "").strip().splitlines()] or ["—"]
        return [f"- {key}：{vlines[0]}"] + [f"  {l}" for l in vlines[1:]]

    block = ([f"### {date}({weekday}) · {title}"]
             + field_block("变更内容", changes)
             + field_block("影响面", "、".join(f"**{c}**" for c in impact))
             + field_block("部署注意", notes)
             + field_block("验证", verify)
             + [""])
    with ticket._file_lock(path):
        lines = _load_lines(path, "部署台账")
        heading_re = re.compile(r"^##(?!#)\s*记录\s*$")
        rec_idx = next((i for i, l in enumerate(lines) if heading_re.match(l)), None)
        if rec_idx is None:
            raise LedgerError("台账缺少「## 记录」节")
        insert_at = rec_idx + 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        new_lines = lines[:insert_at] + block + lines[insert_at:]
        entries = parse_deployment_lines(new_lines)
        for e in entries:
            if "\n".join(render_deployment_entry(e)) != "\n".join(_rstrip_blanks(e["lines"])):
                raise LedgerError(f"写前校验失败：台账登记将破坏条目 {e['id']} 的原文，已放弃落盘")
        ticket._write_atomic(path, new_lines)
    return {"id": "DEP-0001", "date": date, "weekday": weekday, "title": title,
            "wo": "", "impact": list(impact),
            "values": {"变更内容": changes, "影响面": "、".join(impact),
                       "部署注意": (notes or "—").strip() or "—",
                       "验证": (verify or "—").strip() or "—"},
            "fields": [], "lines": block}


# --------------------------------------------------------------------------- #
# 展示（CLI 输出）
# --------------------------------------------------------------------------- #
def format_inbox_entry(e, matches=None):
    out = [f"[{e['section']}] {e['id']} · {e['date']} · {e['source']} · {e['type']}"]
    out.append(e["text"])
    out.extend(e["followups"])
    return "\n".join(out)


def format_ticket(main, children):
    out = []
    for t in [main] + children:
        out.append(f"[{t['section']}] {t['id']} {t['date']} 发起:{t['by']} 承接:{t['to'] or '-'}")
        out.append(f"  描述: {t['desc']}")
        if t["verify"]:
            out.append(f"  验证: {t['verify']}")
        if t["note"]:
            out.append(f"  备注: {t['note']}")
    return "\n".join(out)


def format_deployment_entry(e):
    wd = f"({e['weekday']})" if e["weekday"] else ""
    head = f"[{e['id']}] {e['date']}{wd} · {e['title']}"
    if e.get("wo"):
        head += f"（{e['wo']}）"
    out = [head]
    for key in ("变更内容", "影响面", "部署注意", "验证"):
        v = e["values"].get(key, "")
        if v:
            out.append(f"- {key}：{v}")
    return "\n".join(out)


def _one_line_inbox(e):
    t = e["text"]
    return (f"[{e['section']}] {e['id']} {e['date']} {e['source']} {e['type']} "
            + (t[:97] + "…" if len(t) > 100 else t))


def _one_line_deployment(e):
    wd = f"({e['weekday']})" if e["weekday"] else ""
    t = e["title"]
    title = t[:60] + "…" if len(t) > 60 else t
    imp = "/".join(e["impact"]) or "-"
    return f"[{e['id']}] {e['date']}{wd} {title} ｜影响面: {imp}" + (
        f" ｜{e['wo']}" if e.get("wo") else "")


# --------------------------------------------------------------------------- #
# 路径解析（CLI > 配置 > 内置默认）
# --------------------------------------------------------------------------- #
def resolve_files(args):
    cfg = None
    try:
        cfg = ticket.load_config_from_args(args.config, args.name)
    except ticket.TicketError:
        raise  # 配置非法 fail-fast（与 ticket.py 一致）
    inbox = args.file
    if inbox is None:
        inbox = cfg.inbox_file if (cfg and cfg.inbox_file) else DEFAULT_INBOX
    tickets_f = args.tickets
    if tickets_f is None and cfg is not None:
        tickets_f = cfg.file
    if tickets_f is None:
        tickets_f = ticket.DEFAULT_FILE
    deployment = args.deployment
    if deployment is None:
        deployment = cfg.deployment_file if (cfg and cfg.deployment_file) else DEFAULT_DEPLOYMENT
    return Path(inbox), Path(tickets_f), Path(deployment)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_get(a):
    inbox_f, tickets_f, dep_f = resolve_files(a)
    r = get_entry(a.ident, inbox_file=inbox_f, tickets_file=tickets_f,
                  deployment_file=dep_f, force_deployment=a.from_deployment)
    if r["kind"] == "inbox":
        print(format_inbox_entry(r["entry"]))
    elif r["kind"] == "ticket":
        print(format_ticket(r["ticket"], r["children"]))
    else:
        matches = r.get("matches") or [r["entry"]]
        if len(matches) > 1:
            print(f"（模糊查询命中 {len(matches)} 条，按新→旧列出；可用 DEP 编号精确取单条）")
        for e in matches:
            print(format_deployment_entry(e))
            print()


def cmd_list(a):
    inbox_f, _, dep_f = resolve_files(a)
    if a.status and a.status not in INBOX_SECTIONS:
        raise LedgerError(f"--status 应为 {'/'.join(INBOX_SECTIONS)}")
    if a.dep_list:
        entries = filter_deployment(parse_deployment(dep_f), impact=a.impact,
                                    since=a.since, q=a.q)
        for e in entries:
            print(_one_line_deployment(e))
        if not entries:
            print("（无匹配台账条目）")
    else:
        entries = filter_inbox(parse_inbox(inbox_f), status=a.status, type_=a.type,
                               source=a.source, since=a.since, q=a.q)
        for e in entries:
            print(_one_line_inbox(e))
        if not entries:
            print("（无匹配条目）")


def cmd_append(a):
    inbox_f, _, dep_f = resolve_files(a)
    if a.dep_list:
        e = apply_deployment_append(dep_f, title=a.title, changes=a.changes,
                                    impact=a.impact, notes=a.notes, verify=a.verify,
                                    date=a.date)
        print(f"已登记 {e['id']}（{e['date']} · {e['title']}）")
    else:
        e = apply_inbox_append(inbox_f, source=a.source, type_=a.type, text=a.text,
                               followup=a.followup, date=a.date)
        extra = f"（含续行）" if e["followups"] else ""
        print(f"已追加 {e['id']} 到「待同步」顶部{extra}")


def cmd_move(a):
    inbox_f, _, _ = resolve_files(a)
    e = apply_inbox_move(inbox_f, a.ident, destination=a.destination)
    note = f"，去向：{a.destination}" if a.destination else ""
    print(f"{e['id']} 已流转到「已同步」{note}")


def _add_file_opts(parser, suppress=False):
    """全局路径选项；子解析器也注册（SUPPRESS 默认，不覆盖主解析器已设值），
    使 `ledger.py get INB-0049 --file <副本>`（选项在子命令后）同样合法。"""
    d = argparse.SUPPRESS if suppress else None
    parser.add_argument("--file", type=Path, default=d,
                        help="收件箱文件路径（默认取配置或 .opencode/memory/inbox.md）")
    parser.add_argument("--tickets", type=Path, default=d,
                        help="工单文件路径（默认取配置或 .opencode/memory/tickets.md）")
    parser.add_argument("--deployment-file", dest="deployment", type=Path, default=d,
                        help="部署台账文件路径（默认取配置或仓库根 DEPLOYMENT.md）")
    parser.add_argument("--config", default=d,
                        help="ticketd 配置文件路径（读其中 inbox_file/deployment_file）")
    parser.add_argument("--name", default=d,
                        help="实例名，读 ~/.config/ticketd/<name>.json（尊重 XDG_CONFIG_HOME）")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="统一记录 CLI：收件箱（INB）/ 工单（WO）/ 部署台账（DEP）按编号取单条")
    _add_file_opts(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("get", help="按编号取单条（INB-xxxx / WO-xxxx / DEP-xxxx / 台账模糊）")
    p.add_argument("ident", help="编号或「日期+标题」模糊串")
    p.add_argument("--from-deployment", dest="from_deployment", action="store_true",
                   help="强制查台账（WO-xxxx 按标题内 WO 号索引台账条目，而非工单）")
    _add_file_opts(p, suppress=True)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("list", help="列条目（默认收件箱；--deployment 列台账）")
    p.add_argument("--status", help="按状态过滤（待同步/已同步）")
    p.add_argument("--type", dest="type", help="按类型过滤（子串匹配，如 成果）")
    p.add_argument("--source", help="按来源代理过滤")
    p.add_argument("--since", help="起始日期（YYYY-MM-DD，含当日）")
    p.add_argument("--q", help="全文关键字")
    p.add_argument("--deployment", dest="dep_list", action="store_true",
                   help="列部署台账（配合 --impact）")
    p.add_argument("--impact", help="按影响面过滤（依赖/模型/环境变量/DB/标定/入口/结构/配置）")
    _add_file_opts(p, suppress=True)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("append", help="追加（默认收件箱新条目；--deployment 登记台账）")
    p.add_argument("--source", help="来源代理（如 web）")
    p.add_argument("--type", dest="type", help="类型（成果/坑/待归档/部署，可组合如 成果/部署）")
    p.add_argument("--text", help="一句话结论（单行）")
    p.add_argument("--followup", help="续行（如 → 去向：...）")
    p.add_argument("--date", help="日期覆盖（YYYY-MM-DD，默认今天）")
    p.add_argument("--deployment", dest="dep_list", action="store_true", help="登记部署台账新条目")
    p.add_argument("--title", help="（台账）一句话标题")
    p.add_argument("--changes", help="（台账）变更内容（可多行）")
    p.add_argument("--impact", help="（台账）影响面，逗号分隔（8 选 N）")
    p.add_argument("--notes", help="（台账）部署注意")
    p.add_argument("--verify", help="（台账）验证")
    _add_file_opts(p, suppress=True)
    p.set_defaults(func=cmd_append)

    p = sub.add_parser("move", help="收件箱条目流转：待同步 → 已同步（可注明去向）")
    p.add_argument("ident", help="条目编号（INB-xxxx）")
    p.add_argument("--destination", help="去向说明（写入/更新 → 去向： 行）")
    _add_file_opts(p, suppress=True)
    p.set_defaults(func=cmd_move)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except ticket.TicketError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
