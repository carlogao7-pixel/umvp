# -*- coding: utf-8 -*-
"""
通用条件 DSL（框/记录筛选）：显式字段名，与(,) 或(;) 非(not) + 括号分组

条件 = `字段 操作符 值`；操作符 `== != > >= < <= in`（`=` 等价 `==`；`in` 值用 `|` 分隔）。
组合：组内逗号 `,` = 并且；组间分号 `;` = 或者；`not` = 非（可连用 `not not`）；
括号 `(` `)` 分组，括号内可再写 `,` / `;`，可任意嵌套。值本身含 `, ; )` 时用引号包住。

字段按 dict 键或对象属性取值（不绑定具体来源：YOLO Detection / Record dict /
其它带同名属性的对象皆可用）。字段别名：class/类别/cls → cls_id；track/id → track_id；
conf/置信度 → score；width/宽 → w；height/高 → h；帧号/frame → frame_idx。
派生字段（从 bbox 计算，无需自带）：w/h（宽/高）、short/long（短边/长边）、area（面积）。

聚合谓词（需批上下文，仅 DetFilter 的单帧筛选提供）：
  `count(字段)`=当前记录在该字段值上的**同帧分组大小**（按该字段值分组计数）；
  `count(子条件)`=同帧满足该子条件的记录数（帧级常量，对所有记录一致）；
  `count()`=同帧总数。对同组记录返回相同值 → 整组同进同出。
例："count(cls_id) >= 2"（同帧该类至少 2 个）、"count(cls_id == 0, score >= 0.5) >= 2"
（同帧满足该条件的≥2）、"count() <= 100"（同帧总量上限）。
无上下文（如目标裁剪/标注的逐框筛选）时 count 取不到值 → 该条件不通过；
count 的子条件里不再支持嵌套 count。

供「检测筛选」（DetFilter）、「目标裁剪」（按条件裁哪些框）与「标注」（画哪些框）共用。

例：
    "track_id == 1"                        只留 track_id=1
    "cls_id == 2"                          只留车
    "cls_id in 0|2, score >= 0.5"          人/车 且 置信度≥0.5
    "track_id == 1; cls_id == 2"           track=1 或 车
    "not cls_id == 2"                      非车
    "not (cls_id == 0, score < 0.5)"       非（人且置信不足）
    "(cls_id == 0; short >= 60), area > 100"   (人 或 够大) 且 面积>100
    "count(cls_id) >= 2, score >= 0.5"     同帧该类≥2 且 置信度≥0.5
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import List, Tuple

# 字段别名 → 标准字段名（显式声明字段；支持中文别名）
_FIELD_ALIASES = {
    "cls_id": "cls_id", "cls": "cls_id", "class": "cls_id", "类别": "cls_id",
    "track_id": "track_id", "track": "track_id", "id": "track_id",
    "score": "score", "conf": "score", "置信度": "score",
    "w": "w", "width": "w", "宽": "w",
    "h": "h", "height": "h", "高": "h",
    "short": "short", "短边": "short",
    "long": "long", "长边": "long",
    "area": "area", "面积": "area",
    "frame_idx": "frame_idx", "frame": "frame_idx", "帧号": "frame_idx",
    "ts": "ts", "时间戳": "ts",
}

# 派生字段：从 bbox 算宽高后套用（w, h → 数值）
_DERIVED = {
    "w": lambda w, h: w,
    "h": lambda w, h: h,
    "short": lambda w, h: min(w, h),
    "long": lambda w, h: max(w, h),
    "area": lambda w, h: w * h,
}

_WORD_STOP = ",;()=<>!"  # 字段名/关键字词元的终止字符


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class _Parser:
    """递归下降解析。

    语法：expr := and (";" and)* ；and := not ("," not)* ；
          not := "not" not | primary ；primary := "(" expr ")" | 条件。
    解析树（与旧结构兼容）：[[item, ...](组内 AND), ...](组间 OR)；
    item = (字段, 操作符, 值) | ("not", item) | ("expr", 嵌套树)。
    纯 `,`/`;` 的旧式条件解析结果与 v1 逐字节一致。
    """

    _OPS = ("!=", ">=", "<=", "==", "=", ">", "<")  # 长操作符在前，避免 "=<" 误切

    def __init__(self, text: str) -> None:
        self.s, self.i, self.n = text, 0, len(text)

    # ---- 基元 ----
    def _ws(self) -> None:
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def _peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def _eat_ch(self, ch: str) -> bool:
        self._ws()
        if self._peek() == ch:
            self.i += 1
            return True
        return False

    def _eat_kw(self, kw: str) -> bool:
        """吃掉整词关键字（要求词边界，避免 note/into 误判）。"""
        self._ws()
        if self.s.startswith(kw, self.i):
            j = self.i + len(kw)
            if j >= self.n or not self.s[j].isalnum():
                self.i = j
                return True
        return False

    # ---- 文法 ----
    def parse(self) -> List[List[tuple]]:
        groups = self._or()
        self._ws()
        if self.i < self.n:
            raise ValueError(f"筛选条件无法解析: {self.s[self.i:]!r}（多余内容）")
        return groups

    def _or(self) -> List[List[tuple]]:
        groups = [self._and()]
        while self._eat_ch(";"):
            groups.append(self._and())
        return groups

    def _and(self) -> List[tuple]:
        items = [self._not()]
        while self._eat_ch(","):
            items.append(self._not())
        return items

    def _not(self) -> tuple:
        if self._eat_kw("not"):
            return ("not", self._not())
        if self._eat_ch("("):
            e = self._or()
            if not self._eat_ch(")"):
                raise ValueError("筛选条件缺少右括号 ')'（括号须成对）")
            return ("expr", e)
        return self._cond()

    def _cond(self) -> Tuple[str, str, object]:
        field = self._word()
        if not field:
            raise ValueError("筛选条件无法解析（应为 字段 操作符 值，如 track_id == 1）")
        if field == "count":
            field = ("count", self._count_arg())
        else:
            field = _FIELD_ALIASES.get(field, field)
        op = self._op()
        vtext = self._value()
        if op == "in":
            value: object = [x.strip() for x in vtext.split("|")]
        else:
            num = _to_num(vtext)
            value = num if num is not None else vtext
        return (field, op, value)

    def _count_arg(self):
        """解析 count(...) 括号内容，返回参数：
        None=总数；("field", 名)=按字段分组计数；("cond", 表达式树)=子条件计数。"""
        self._ws()
        if self._peek() != "(":
            raise ValueError("count 须写作 count(字段)、count(子条件) 或 count()")
        self.i += 1  # 吃掉 '('
        self._ws()
        if self._peek() == ")":
            self.i += 1
            return None  # count() 总数
        save = self.i
        word = self._word()  # 先试"单个字段"（分组计数）
        self._ws()
        if word and self._peek() == ")":
            self.i += 1
            return ("field", _FIELD_ALIASES.get(word, word))
        # 否则回退按子条件表达式解析（到匹配的 ')'）
        self.i = save
        expr = self._or()
        self._ws()
        if self._peek() != ")":
            raise ValueError("count(...) 缺少右括号 ')'")
        self.i += 1
        return ("cond", expr)

    def _word(self) -> str:
        self._ws()
        j = self.i
        while j < self.n:
            c = self.s[j]
            if c.isspace() or c in _WORD_STOP:
                break
            j += 1
        w, self.i = self.s[self.i:j], j
        return w

    def _op(self) -> str:
        self._ws()
        if self._eat_kw("in"):
            return "in"
        for op in self._OPS:
            if self.s.startswith(op, self.i):
                self.i += len(op)
                return "==" if op == "=" else op
        raise ValueError("筛选条件缺少操作符（应为 字段 操作符 值，如 track_id == 1）")

    def _value(self) -> str:
        self._ws()
        if self._peek() in ("'", '"'):  # 引号值：可含 , ; ) 等分隔符
            q = self.s[self.i]
            j = self.s.find(q, self.i + 1)
            if j < 0:
                raise ValueError("筛选条件的引号未闭合")
            raw, self.i = self.s[self.i + 1:j], j + 1
            return raw
        j = self.i
        while j < self.n and self.s[j] not in ",;)":
            j += 1
        raw = self.s[self.i:j].strip()
        if not raw:
            raise ValueError("筛选条件的值为空（应为 字段 操作符 值，如 score >= 0.5）")
        self.i = j
        return raw


def parse_condition(text) -> List[List[tuple]]:
    """条件 DSL → 解析树：[[item, ...](组内 AND), ...](组间 OR)；item 为
    (字段, 操作符, 值) / ("not", item) / ("expr", 嵌套树)。空 → []（不筛选）。

    非法表达式抛 ValueError（通俗报错，供保存/装配时快速失败）；
    传入已解析结构直接透传（内部便利）。
    """
    if text is None:
        return []
    if not isinstance(text, str):
        return text
    if not text.strip():
        return []
    return _Parser(text).parse()


def _bbox_wh(obj):
    """取 bbox 算 (宽, 高)；dict 取键、对象取属性，无效返回 None。"""
    b = obj.get("bbox") if isinstance(obj, Mapping) else getattr(obj, "bbox", None)
    try:
        x1, y1, x2, y2 = b
        return float(x2) - float(x1), float(y2) - float(y1)
    except (TypeError, ValueError):
        return None


def _frame_key(obj, default=None):
    """取记录的帧键（frame_idx 字段/属性），缺省回落到 default。"""
    if isinstance(obj, Mapping):
        fk = obj.get("frame_idx")
    else:
        fk = getattr(obj, "frame_idx", None)
    return default if fk is None else fk


def _hashable(v):
    """把可枚举值（list）转成可哈希键，便于计数分组。"""
    return tuple(v) if isinstance(v, list) else v


def _count_value(obj, arg, ctx):
    """聚合谓词取值：arg=None 总数 / ("field",名) 同帧分组计数 / ("cond",树) 同帧子条件计数。
    无 ctx → None（条件不通过）。"""
    if ctx is None:
        return None
    fk = _frame_key(obj, ctx.get("default_frame_idx"))
    if arg is None:
        return ctx.get("totals", {}).get(fk, 0)
    if isinstance(arg, tuple) and arg and arg[0] == "field":
        val = _hashable(_field_value(obj, arg[1], ctx))
        return ctx.get("counts", {}).get((fk, arg[1]), {}).get(val, 0)
    if isinstance(arg, tuple) and arg and arg[0] == "cond":
        return ctx.get("conds", {}).get(id(arg[1]), {}).get(fk, 0)
    return None


def _field_value(obj, name, ctx=None):
    """取对象/dict 字段；派生字段（w/h/short/long/area）从 bbox 计算；
    聚合字段 ("count", inner) 从批上下文取同帧计数；缺字段返回 None。
    cls_id/track_id/score 走数值规整（不绑定具体来源）。"""
    if isinstance(name, tuple) and name and name[0] == "count":
        return _count_value(obj, name[1], ctx)
    fn = _DERIVED.get(name)
    if fn is not None:
        wh = _bbox_wh(obj)
        return fn(*wh) if wh else None
    raw = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    if name == "cls_id":
        return int(raw) if raw is not None else -1
    if name == "track_id":
        return int(raw or 0)
    if name == "score":
        return float(raw) if raw is not None else 0.0
    return raw


def _cond_pass(obj, cond: Tuple[str, str, object], ctx=None) -> bool:
    field, op, value = cond
    a = _field_value(obj, field, ctx)
    if a is None:
        return False
    if op in (">", ">=", "<", "<="):
        x, y = _to_num(a), _to_num(value)
        if x is None or y is None:
            return False
        return x > y if op == ">" else x >= y if op == ">=" else x < y if op == "<" else x <= y
    if op == "==":
        return a == value or str(a) == str(value)
    if op == "!=":
        return not (a == value or str(a) == str(value))
    if op == "in":
        return str(a) in [str(v) for v in value]
    return False


def _item_pass(obj, item, ctx=None) -> bool:
    """求值单个组内项：3 元组=条件；("not", x)=取反；("expr", 树)=嵌套表达式。"""
    if isinstance(item, tuple) and len(item) == 3:
        return _cond_pass(obj, item, ctx)
    kind, sub = item
    if kind == "not":
        return not _item_pass(obj, sub, ctx)
    if kind == "expr":
        return match_obj(obj, sub, ctx)
    raise ValueError(f"筛选条件节点无法求值: {item!r}")


def match_obj(obj, groups, ctx=None) -> bool:
    """对象/记录是否满足条件：空=全通过；否则组间任一组内全部命中即通过
    （OR of AND；组内项可含 not / 括号嵌套 expr）。兼容 v1 解析结构。

    ctx 为聚合上下文（含 count(...) 时由调用方提供，见 build_count_context）。
    """
    if not groups:
        return True
    return any(all(_item_pass(obj, it, ctx) for it in g) for g in groups)


# ---------------- 聚合谓词（count）上下文 ----------------
def collect_count_specs(groups):
    """扫描解析树，返回 {"fields": 分组字段集合, "total": 是否用 count(),
    "conds": [(节点id, 子条件树), ...]}。无 count → 全空。"""
    fields: set = set()
    conds: list = []
    state = {"total": False}

    def visit_item(it):
        if isinstance(it, tuple) and len(it) == 3:
            f = it[0]
            if isinstance(f, tuple) and f and f[0] == "count":
                arg = f[1]
                if arg is None:
                    state["total"] = True
                elif isinstance(arg, tuple) and arg and arg[0] == "field":
                    fields.add(arg[1])
                elif isinstance(arg, tuple) and arg and arg[0] == "cond":
                    conds.append((id(arg[1]), arg[1]))
        elif isinstance(it, tuple) and len(it) == 2:
            kind, sub = it
            if kind == "not":
                visit_item(sub)
            elif kind == "expr":
                for g in sub:
                    for x in g:
                        visit_item(x)

    for g in groups or []:
        for it in g:
            visit_item(it)
    return {"fields": fields, "total": state["total"], "conds": conds}


def build_count_context(groups, items, frame_idx=0):
    """为含 count(...) 的条件预计算**单帧**计数上下文。

    items 应已统一为记录视图（dict，含 frame_idx）。分组键 = (frame_idx, 字段值)；
    frame_idx 取记录自带值，缺省用传入的 frame_idx。返回
    {"counts": {(帧键,字段):{值:次数}}, "totals": {帧键:次数},
     "conds": {子条件节点id: {帧键:次数}}, "default_frame_idx": frame_idx}。
    无 count 条件时返回空结构。子条件里不再支持嵌套 count（按无上下文求值）。
    """
    specs = collect_count_specs(groups)
    fields, total, conds = specs["fields"], specs["total"], specs["conds"]
    ctx: dict = {"counts": {}, "totals": {}, "conds": {}, "default_frame_idx": frame_idx}
    if not fields and not total and not conds:
        return ctx
    items = list(items or [])
    for it in items:
        fk = _frame_key(it, frame_idx)
        if total:
            ctx["totals"][fk] = ctx["totals"].get(fk, 0) + 1
        for f in fields:
            val = _hashable(_field_value(it, f, ctx))
            by_val = ctx["counts"].setdefault((fk, f), {})
            by_val[val] = by_val.get(val, 0) + 1
    for node_id, tree in conds:
        per_frame: dict = {}
        for it in items:
            if match_obj(it, tree):
                fk = _frame_key(it, frame_idx)
                per_frame[fk] = per_frame.get(fk, 0) + 1
        ctx["conds"][node_id] = per_frame
    return ctx
