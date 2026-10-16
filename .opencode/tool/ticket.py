#!/usr/bin/env python3
"""UMVP 开发代理·工单工具（stdlib 零依赖）

工单文件默认 .opencode/memory/tickets.md，规约见该文件头与 CLAUDE.md §四。

用法（项目根执行）：
  python3 .opencode/tool/ticket.py new "描述" --by orchestrator --to web \
      [--parent WO-0001] [--verify "命令"] [--note "..."] [--date YYYY-MM-DD]
  python3 .opencode/tool/ticket.py set WO-0001.2 [--status 待测试] \
      [--append-note "..."]/--note "..."（整字段覆盖） [--verify "..."] [--to "..."]
  python3 .opencode/tool/ticket.py list [--status 进行中] [--parent WO-0001]
  python3 .opencode/tool/ticket.py show WO-0001
  python3 .opencode/tool/ticket.py check

约定：主单 WO-0001 全局递增；子单 WO-0001.n 挂在主单下；状态 ∈
{进行中, 待测试, 已完成, 已取消}；字段内禁用竖线；新行/移动插入对应状态节表格顶部。
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

DEFAULT_FILE = Path(".opencode/memory/tickets.md")
STATUSES = ["进行中", "待测试", "已完成", "已取消"]
AGENTS = ["orchestrator", "pipe-chain", "web", "agent-builder", "qa"]
ID_RE = re.compile(r"^WO-(\d{4})(?:\.(\d+))?$")
HEADER_ROW = "| 编号 | 日期 | 发起人 | 承接 | 需求/改动描述 | 验证 | 备注 |"
SEP_ROW = "|---|---|---|---|---|---|---|"
FIELDS = ["id", "date", "by", "to", "desc", "verify", "note"]


class TicketError(Exception):
    pass


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


def cmd_new(a):
    if a.by not in AGENTS:
        raise TicketError(f"--by 应为 {'/'.join(AGENTS)}，收到 {a.by!r}")
    if "|" in a.desc or (a.note and "|" in a.note) or (a.verify and "|" in a.verify):
        raise TicketError("字段内禁用竖线 |")
    lines = load_lines(a.file)
    tickets, insert_at = scan(lines)
    main, subs = next_numbers(tickets)
    if a.parent:
        parent = parse_id(a.parent)
        if "." in parent:
            raise TicketError("--parent 必须是主单（WO-0001 形式）")
        if not any(t["id"] == parent for t in tickets):
            raise TicketError(f"父单不存在: {parent}")
        tid = f"{parent}.{subs.get(int(parent[3:]), 1)}"
    else:
        tid = main
    t = {"id": tid, "date": a.date or datetime.date.today().isoformat(),
         "by": a.by, "to": a.to or "", "desc": a.desc,
         "verify": a.verify or "", "note": a.note or ""}
    lines = insert_row(lines, insert_at, "进行中", make_row(t))
    a.file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已创建 {tid}（进行中）→ 承接: {a.to or '-'}")


def cmd_set(a):
    tid = parse_id(a.ticket)
    lines = load_lines(a.file)
    tickets, insert_at = scan(lines)
    cur = next((t for t in tickets if t["id"] == tid), None)
    if cur is None:
        raise TicketError(f"工单不存在: {tid}")
    if a.status and a.status not in STATUSES:
        raise TicketError(f"--status 应为 {'/'.join(STATUSES)}")
    t = dict(cur)
    for f in ("to", "verify", "note"):  # 仅这三字段可更新，其余以现值为准
        v = getattr(a, f, None)
        if v is not None:
            t[f] = v
    if getattr(a, "append_note", None):
        t["note"] = f"{cur['note']}；{a.append_note}" if cur["note"] else a.append_note
    if "|" in t["note"] or "|" in t["verify"]:
        raise TicketError("字段内禁用竖线 |")
    # 主单收尾检查：仍有未关闭子单时警告（不阻断）
    if a.status in ("已完成", "已取消") and "." not in tid:
        open_subs = [x["id"] for x in tickets
                     if x["id"].startswith(tid + ".") and x["section"] in ("进行中", "待测试")]
        if open_subs:
            print(f"⚠ 主单 {tid} 仍有未关闭子单: {', '.join(open_subs)}", file=sys.stderr)
    target = a.status or cur["section"]
    lines = [l for l in lines if l != cur["line"]]
    tickets2, insert_at2 = scan(lines)
    lines = insert_row(lines, insert_at2, target, make_row(t))
    a.file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{tid} → {target}" + (f"（备注: {a.note}）" if a.note else ""))


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
    errs = []
    lines = load_lines(a.file)
    tickets, insert_at = scan(lines)
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
    for e in errs:
        print("✗", e, file=sys.stderr)
    print(f"check OK：{len(tickets)} 条工单，结构合法" if not errs else f"check 失败：{len(errs)} 处问题",
          file=sys.stderr if errs else sys.stdout)
    sys.exit(1 if errs else 0)


def main():
    ap = argparse.ArgumentParser(description="UMVP 工单工具")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE, help="工单文件路径（默认 .opencode/memory/tickets.md）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="新建工单（默认进行中）")
    p.add_argument("desc", help="需求/改动描述")
    p.add_argument("--by", required=True, help="发起人（代理名）")
    p.add_argument("--to", help="承接方（逗号分隔代理名）")
    p.add_argument("--parent", help="父主单编号（生成子单 WO-xxxx.n）")
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
        a.func(a)
    except TicketError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
