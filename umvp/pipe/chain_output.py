# -*- coding: utf-8 -*-
"""
链路尾输出提炼（collect_output）：spec 顶层 output 段的执行纯逻辑（零 IO）

设计见 doc/design/chain_output.md（v2，P4 职责边界：**收敛归筛选模块、配图出场归
output**）——本层不做去重、不做字段投影（事件 detail 已由事件尾 SeqRule.detail_fields /
DetFilter detail 投影好），只做三件事：

  1. scope 过滤  —— "events"=只取携带事件的记录（收敛已由事件尾完成，默认）；
                     "records"=全部放行记录（无事件链路如 PIPE0003 用它提炼图清单）。
  2. 配图对位    —— 配图源记录流（Records 产物）按 frame_idx + 顺序对位到主流记录，
                     取旁路图集（image_set_of）；image_pick: "all"=图集全出（默认，
                     存档宁多勿缺）/ "first"=取首张；images 统一列表语义。
  3. limit 截断  —— 超限截断并标 truncated；不做任何 key 去重（P4）。

零 IO：图以 ndarray 列表放在 items 里返回，**不落盘、不 import cv2**；
落盘命名 / max_saved_images 计数 / saved_files 清单归驱动层（web WO-0029.2）。

Image 产物（帧级整图单张）约定：本函数只处理 Records→Records 对位；帧级图请由
驱动层直接处理（同帧提炼记录共用该图、各自落盘引用），或包成与主流等长同帧的
记录流（如 [{"frame_idx": f, "image": 帧}] 逐条包）再传入——经本函数时按
Records 语义**消费式**对位（一条配图记录只配一条主流记录）。

用法:
    from pipe.chain_output import collect_output
    out = collect_output(last_records, image_records=annotate_records,
                         decl={"scope": "events", "image_pick": "all", "limit": 100})
    # {"count": 2, "truncated": False, "items": [{"ts","frame_idx","label",
    #  "track_id","score"(存在才带), "events":[...], "images":[ndarray,...]}, ...]}
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from pipe.records import get_events, image_set_of, to_record

# output 段缺省值（doc/design/chain_output.md §3；输出侧 image_pick 默认 all）
DEFAULT_DECL: Dict[str, Any] = {"scope": "events", "image_pick": "all", "limit": 100}
_GENERIC_FIELDS = ("ts", "frame_idx", "label", "track_id", "score")  # 通用字段（存在才带）


def _norm_decl(decl) -> Dict[str, Any]:
    """decl 规整 + 校验：未给的键用缺省；scope/image_pick/limit 非法抛 ValueError
    （装配/驱动层快速失败；spec 层另有 validate_chain 把关）。未知键忽略。"""
    d = dict(DEFAULT_DECL)
    for k in DEFAULT_DECL:
        v = (decl or {}).get(k)
        if v is not None:
            d[k] = v
    scope = str(d["scope"]).strip().lower()
    pick = str(d["image_pick"]).strip().lower()
    if scope not in ("events", "records"):
        raise ValueError(f'output.scope 非法: {d["scope"]!r}（应为 "events" 或 "records"）')
    if pick not in ("first", "all"):
        raise ValueError(f'output.image_pick 非法: {d["image_pick"]!r}（应为 "first" 或 "all"）')
    try:
        limit = int(d["limit"])
    except (TypeError, ValueError):
        raise ValueError(f"output.limit 需为正整数，得到 {d['limit']!r}")
    if limit <= 0:
        raise ValueError(f"output.limit 需为正整数，得到 {limit}")
    return {"scope": scope, "image_pick": pick, "limit": limit}


def _pair_images(mains: List[dict], image_records) -> List[List[Any]]:
    """配图对位：配图源记录流 → 每条主流记录的图列表（顺序与 mains 一一对应）。

    策略（从简，consume-on-use：一条配图记录只配一条主流记录）：
      1. 配图源按 frame_idx 分桶（None 帧号自成一桶，桶内保持出现顺序）；
      2. 主流逐条对位：优先取**同 frame_idx** 桶中最早未消费的一条；
      3. 同帧无可用 → 整体顺序兜底（取下一条未消费的配图记录）；
      4. 配图源耗尽 → images=[]；对位到的记录旁路无图（image_set_of 空）→ images=[]。
    """
    pools: Dict[Any, deque] = {}
    order: deque = deque()          # 整体顺序兜底用（未消费的配图记录队列）
    for ir in image_records or []:
        r = to_record(ir)
        pools.setdefault(r.get("frame_idx"), deque()).append(r)
        order.append(r)
    used: set = set()               # 已消费配图记录（id 级，防同对象重复配）
    out: List[List[Any]] = []
    for rec in mains:
        img_rec = None
        q = pools.get(rec.get("frame_idx"))
        while q and id(q[0]) in used:
            q.popleft()
        if q:                       # 同帧优先
            img_rec = q.popleft()
        else:                       # 整体顺序兜底
            while order and id(order[0]) in used:
                order.popleft()
            img_rec = order.popleft() if order else None
        if img_rec is None:
            out.append([])
            continue
        used.add(id(img_rec))
        out.append(image_set_of(img_rec))
    return out


def collect_output(records, image_records=None,
                   decl: Optional[dict] = None) -> dict:
    """链路尾输出提炼（纯逻辑，零 IO）。

    records      : 末段记录流（list；dict 或对象，内部经 to_record 归一）。
    image_records: 配图源（None=不配图；Records 产物记录流；Image 产物由调用方包一层，
                   见模块 docstring 约定）。
    decl         : {"scope": "events"|"records", "image_pick": "first"|"all",
                    "limit": 正整数}；缺省 {"events", "all", 100}；未知键忽略。

    返回 {"count": n, "truncated": bool, "items": [...]}：
      item = {"ts","frame_idx","label","track_id","score"（通用字段，存在且非 None 才带）,
              "events": [该记录事件全明细（get_events）],
              "images": [旁路图列表（未落盘！ndarray 由驱动层落盘）]}
    images 恒为列表（不配图/无图为 []；image_pick="first" 时长度 ≤1）。

    顺序（doc/design/chain_output.md §4）：scope 过滤 → 配图对位 → limit 截断；
    不做 key 去重（P4：收敛是事件尾职责）。
    """
    nd = _norm_decl(decl)
    mains = [to_record(r) for r in (records or [])]
    if nd["scope"] == "events":
        mains = [r for r in mains if get_events(r)]
    img_sets = _pair_images(mains, image_records) if image_records is not None else None
    items: List[dict] = []
    for i, rec in enumerate(mains):
        item: Dict[str, Any] = {"events": get_events(rec)}
        for f in _GENERIC_FIELDS:
            v = rec.get(f)
            if v is not None:
                item[f] = v
        if img_sets is None:
            item["images"] = []
        else:
            imgs = img_sets[i]
            item["images"] = list(imgs[:1]) if nd["image_pick"] == "first" else list(imgs)
        items.append(item)
    truncated = len(items) > nd["limit"]
    if truncated:
        items = items[:nd["limit"]]
    return {"count": len(items), "truncated": truncated, "items": items}
