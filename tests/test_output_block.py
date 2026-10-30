# -*- coding: utf-8 -*-
"""链路尾输出提炼块 web 侧断言测试（WO-0029.2，设计 doc/design/chain_output.md §3-§7）。

覆盖（全部纯逻辑/进程内，不起 HTTP、不连 MySQL、不加载模型权重）：
   1. _resolve_output_plan：缺省 from（运行时解析 deferred 计划）/显式 from/image 解析、
      类型与取值校验、非法声明 fail-soft 报错、meta.output 回落。
   2. _make_output_accumulator：scope 粗筛、同帧 consume-on-use 配图预对位（与
      整流 collect_output 直提结果逐条一致）、占位对齐、硬上限截断。
   3. _finish_output：collect_output 提炼 → 落盘命名 output_*（共用计数/清单）→
      items.images 相对路径 + thumbs data URL；Image 帧级产物 saved_files 引用；
      流式（output_dir=None）不落盘 images 恒空 + note。
   4. _output_frame_image_index：帧号映射 + start_frame 偏移 + 同帧最后一张优先。
   5. _build_report_html：快照含 output 块 →「链路尾输出」区块优先呈现（事件表前）、
      内嵌图 data URL、上限；无声明时报告与现状一致（无该区块）。
   6. 缺省 from 运行时解析（WO-0029.5）：_output_pick_from 三级裁决（末段实际登记
      → pipeline.tail → None 不误报）、publish 通路 note、空块 error「末段无可用
      记录产物」不静默、显式 from 仍 fail-soft。

运行: conda run -n ai python tests/test_output_block.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402  （顶层无 HTTP/DB 副作用，db 连接惰性）
from pipe.chain_output import collect_output  # noqa: E402
from pipe.records import add_event  # noqa: E402

PASS = 0


def ok(cond, msg: str) -> None:
    global PASS
    assert cond, f"断言失败: {msg}"
    PASS += 1
    print(f"  [ok] {msg}")


def make_img(w: int = 320, h: int = 240) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 200, 0), -1)
    return img


# 警用无人机形态（fm → yolo → det_filter → annotate → vlm → seq_filter；参数取自播种）
_UAV_STAGES = [
    {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 15}},
    {"module_id": "yolo", "params": {"model_path": "yolov8n.pt", "classes": "0,2"}},
    {"module_id": "det_filter", "params": {"group_limits": "0:2,300;2:1,300"}},
    {"module_id": "annotate", "params": {"thickness": 2, "show_label": True}},
    {"module_id": "vlm", "params": {"use_real_vlm": False}},
    {"module_id": "seq_filter", "params": {"throttle": "cls_id:3.0", "rules": ""}},
]

# 车牌链路形态（PIPE0002 播种：… → plate_recog → seq_filter rules-only 事件尾）
_PLATE_STAGES = [
    {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 5}},
    {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "classes": "2"}},
    {"module_id": "det_filter", "params": {"filter": "long >= 40", "group_limits": "2:1,300"}},
    {"module_id": "gate_filter", "params": {"key": "track_id", "interval": 5.0}},
    {"module_id": "target_crop", "params": {"out_size": "×2", "classes": "2"}},
    {"module_id": "plate_recog", "params": {"mode": "crop", "track_dedup": False}},
    {"module_id": "seq_filter",
     "params": {"throttle": "", "change": "",
                "rules": '[{"match": "kind == plate", "event": "{label}",'
                         ' "key": "track_id,label", "cooldown": -1.0}]'}},
]


def _spec(output=None, stages=None):
    s = {"name": "t", "stages": stages or _UAV_STAGES}
    if output is not None:
        s["output"] = output
    return s


# ---------------- 1. _resolve_output_plan / _output_decl_of ----------------

def test_plan_resolve():
    print("[1] _resolve_output_plan")
    # 无声明 → (None, None)：完全现状
    plan, err = server._resolve_output_plan(_spec())
    ok(plan is None and err is None, "无 output 声明 → 不产生计划（现状）")
    ok(server._output_decl_of({"name": "t"}) is None, "_output_decl_of 无段 → None")

    # 缺省 from = 运行时解析 deferred 计划（WO-0029.5）：UAV 末段 seq_filter 带 gates
    # （静态候选 seq_filter.dets），运行时是否登记由 _output_pick_from 裁决
    plan, err = server._resolve_output_plan(_spec({"scope": "events"}))
    ok(err is None and plan.get("from_deferred") and plan["from_full"] is None,
       "缺省 from → deferred 计划（计划期不报错，运行时裁决）")
    ok(plan["tail_froms"] == ["seq_filter.dets", "seq_filter#5.dets"],
       f"静态候选全名展开（短名+位置）: {plan['tail_froms']}")
    ok(plan["scope"] == "events" and plan["image_pick"] == "all" and plan["limit"] == 100,
       "缺省 decl: events/all/100（输出侧 image_pick 默认 all）")

    # 显式 from 短名归一全名；image Records 产物解析
    plan, err = server._resolve_output_plan(_spec(
        {"from": "det_filter.dets", "image": "annotate.annotated_records",
         "image_pick": "first", "limit": 7}))
    ok(err is None and plan["from_full"] == "det_filter#2.dets", "显式短名 from 归一全名")
    ok(plan["image_full"] == "annotate#3.annotated_records" and plan["image_type"] == "Records",
       "image Records 产物解析")
    ok(plan["image_pick"] == "first" and plan["limit"] == 7, "image_pick/limit 规整")

    # Image 帧级产物
    plan, err = server._resolve_output_plan(_spec({"image": "annotate.annotated"}))
    ok(err is None and plan["image_type"] == "Image", "image Image 帧级产物类型识别")

    # meta.output 回落（DB 往返持久化在 meta）
    ok(server._output_decl_of({"meta": {"output": {"scope": "records"}}}) == {"scope": "records"},
       "meta.output 回落生效")
    plan, err = server._resolve_output_plan({"stages": _UAV_STAGES,
                                             "meta": {"output": {"scope": "records"}}})
    ok(err is None and plan["scope"] == "records", "meta.output → 计划 scope=records")

    # 非法声明 fail-soft：(None, 原因)
    cases = [
        ({"from": "nope.dets"}, "无法解析"),
        ({"scope": "banana"}, "scope"),
        ({"image_pick": "many"}, "image_pick"),
        ({"limit": 0}, "limit"),
        ({"limit": "abc"}, "limit"),
    ]
    for decl, keyword in cases:
        plan, err = server._resolve_output_plan(_spec(decl))
        ok(plan is None and isinstance(err, str) and
           (keyword is None or keyword in err), f"非法声明报错: {decl} → {err}")
    # 未知模块产物 / 已知模块未知产物
    plan, err = server._resolve_output_plan(_spec({"from": "yolo.nonexist"}))
    ok(plan is None and err and "未知产物" in err, f"已知模块未知产物报错: {err[:60]}…")
    # face_embed 终段（无 Records 产物声明）：缺省 from 仍转运行时裁决（不计划期报错），
    # 候选只剩 ② pipeline.tail；取不到时收尾块报「末段无可用记录产物」（test_default_from_runtime）
    # （output={} 视为未声明合法；给非空键触发缺省 from 解析）
    face_spec = {"stages": [
        {"module_id": "frame_manager", "params": {}},
        {"module_id": "yolo", "params": {}},
        {"module_id": "face_embed", "params": {}}], "output": {"scope": "records"}}
    plan, err = server._resolve_output_plan(face_spec)
    ok(err is None and plan.get("from_deferred") and plan["tail_froms"] == [],
       f"face_embed 终段缺省 from 转运行时裁决（无静态候选）: {plan and plan['tail_froms']}")


# ---------------- 2. 累积器（scope 粗筛 + 同帧配图预对位 + 硬上限） ----------------

def _rec(f, tid, with_event=False, img=None):
    r = {"kind": "det", "label": "person", "track_id": tid, "score": 0.5 + tid * 0.01,
         "ts": f * 0.04, "frame_idx": f}
    if with_event:
        add_event(r, "警情", ts=f * 0.04, detail={"key": str(tid)})
    if img is not None:
        r["image"] = img
    return r


def test_accumulator():
    print("[2] _make_output_accumulator")
    plan = {"scope": "events", "image_pick": "all", "limit": 100,
            "from_full": "seq_filter#5.dets", "from_type": "Records",
            "image_full": "annotate#3.annotated_records", "image_type": "Records"}
    acc = server._make_output_accumulator(plan)
    # 帧 3：r1(事件,配 i1)、r2(无事件,粗筛掉)、r3(事件,配 i2)；帧 5：r4(事件,配 i4)；帧 7：r5(无事件)
    i1, i2, i4 = {"frame_idx": 3, "image": "A"}, {"frame_idx": 3, "image": "B"}, \
        {"frame_idx": 5, "image": "C"}
    acc["add"]([_rec(3, 1, True), _rec(3, 2), _rec(3, 3, True)], [i1, i2])
    acc["add"]([_rec(5, 4, True)], [i4])
    acc["add"]([_rec(7, 5)])                      # 无事件帧：连配图流一并丢弃
    acc["add"]([], [i1])                          # 空记录帧：不炸
    ok(len(acc["recs"]) == 3 and not acc["truncated"], "scope=events 粗筛后累积 3 条")
    # 收尾提炼与「整流直提」逐条一致（配图预对位 == collect_output 同策略）
    got = collect_output(acc["recs"], acc["imgs"],
                         {"scope": "events", "image_pick": "all", "limit": 100})
    ref = collect_output(
        [_rec(3, 1, True), _rec(3, 2), _rec(3, 3, True), _rec(5, 4, True), _rec(7, 5)],
        [i1, i2, i4], {"scope": "events", "image_pick": "all", "limit": 100})
    ok([sorted(it.keys()) for it in got["items"]] == [sorted(it.keys()) for it in ref["items"]]
       and [it.get("track_id") for it in got["items"]] == [it.get("track_id") for it in ref["items"]]
       and [len(it["images"]) for it in got["items"]] == [1, 1, 1],
       "累积收尾 == 整流直提（配图 1:1 对位）")
    # 占位对齐：帧 3 只剩 1 张配图 → r3 占位（空图）；帧 5 无配图 → r4 占位
    acc2 = server._make_output_accumulator(plan)
    acc2["add"]([_rec(3, 1, True), _rec(3, 3, True)], [i1])
    acc2["add"]([_rec(5, 4, True)], [])
    got2 = collect_output(acc2["recs"], acc2["imgs"],
                          {"scope": "events", "image_pick": "all", "limit": 100})
    ok([len(it["images"]) for it in got2["items"]] == [1, 0, 0],
       "配图不足 → 占位条目 images=[]（与整流语义一致）")
    # 硬上限 max(10×limit, 200)：limit=1 → 200 条截断
    plan_lo = dict(plan, limit=1)
    acc3 = server._make_output_accumulator(plan_lo)
    acc3["add"]([_rec(f, f, True) for f in range(260)], None)
    ok(len(acc3["recs"]) == 200 and acc3["truncated"],
       f"硬上限 200 条截断（实际累积 {len(acc3['recs'])}）")
    acc3["add"]([_rec(999, 999, True)], None)
    ok(len(acc3["recs"]) == 200, "截断后停止累积")


# ---------------- 2.5 缺省 from 运行时解析（WO-0029.5，PIPE0002 形态） ----------------

def test_default_from_runtime():
    print("[2.5] 缺省 from 运行时解析（rules-only 事件尾 / publish 通路）")
    spec = {"name": "t", "stages": _PLATE_STAGES,
            "output": {"scope": "events", "image": "plate_recog.plates"}}
    plan, err = server._resolve_output_plan(spec)
    ok(err is None and plan.get("from_deferred"), "缺省 from → deferred（PIPE0002 形态）")
    ok(plan["tail_froms"] == ["seq_filter.dets", "seq_filter#6.dets"],
       f"静态候选（末段位置 6）: {plan['tail_froms']}")
    ok(plan["image_full"] == "plate_recog#5.plates", "image 配图产物解析不受缺省 from 影响")

    # 显式 from=pipeline.tail：运行时产物原样通过（无 #position 形态）
    plan_ex, err = server._resolve_output_plan(
        {"name": "t", "stages": _PLATE_STAGES,
         "output": {"from": "pipeline.tail", "scope": "events"}})
    ok(err is None and plan_ex["from_full"] == "pipeline.tail"
       and not plan_ex["from_deferred"], "显式 from=pipeline.tail 原样解析（运行时产物）")

    # 逐帧裁决：仅 pipeline.tail（publish 通路）→ 命中 + 打 publish 标记
    tail_rec = {"kind": "plate", "label": "京A12345", "track_id": 7, "frame_idx": 3}
    add_event(tail_rec, "京A12345", ts=0.12, detail={"plate_no": "京A12345"})
    plan_pub = dict(plan)
    picked = server._output_pick_from(plan_pub, {"pipeline.tail": [tail_rec]})
    ok(picked == "pipeline.tail" and plan_pub.get("_used_publish") is True
       and plan_pub["_used_from"] == "pipeline.tail",
       "仅 pipeline.tail → 命中 publish 通路（② 级）+ 记录实际来源")
    # ① 级优先：末段实际登记（全名）压过 pipeline.tail
    plan_p1 = dict(plan)
    p1 = server._output_pick_from(plan_p1, {"pipeline.tail": [tail_rec],
                                            "seq_filter#6.dets": [_rec(3, 7)]})
    ok(p1 == "seq_filter#6.dets" and "_used_publish" not in plan_p1,
       "末段实际登记产物优先（① > ②）")
    # 短名登记（单实例别名）同样命中；空列表也算登记（键在 = 本帧登记过）
    ok(server._output_pick_from(dict(plan), {"seq_filter.dets": []}) == "seq_filter.dets",
       "末段产物短名登记命中（空列表也算登记）")
    # 无事件帧：键不存在 → None（本帧跳过，不误报 error）
    ok(server._output_pick_from(dict(plan), {}) is None,
       "无事件帧（键不存在）→ None 不误报")
    # 显式 from：原样透传（缺产物即本帧取不到，现状语义，不裁决）
    ok(server._output_pick_from(plan_ex, {"seq_filter#6.dets": []}) == "pipeline.tail",
       "显式 from 透传不裁决")

    # 驱动闭环：逐帧 pick → 累积 → 收尾（publish 通路 + 同帧车牌图配图）
    pimg = {"kind": "plate", "label": "京A12345", "frame_idx": 3, "image": make_img()}
    a3 = {"pipeline.tail": [tail_rec], "plate_recog#5.plates": [pimg]}
    frm = server._output_pick_from(plan, a3)
    acc = server._make_output_accumulator(plan)
    acc["add"](a3[frm], a3.get(plan["image_full"]))
    a4 = {"plate_recog#5.plates": [pimg]}  # 无事件帧：无 pipeline.tail → 跳过
    if server._output_pick_from(plan, a4) is not None:
        acc["add"](a4.get("pipeline.tail") or [], None)
    ok(plan.get("_used_publish") is True and len(acc["recs"]) == 1,
       "publish 通路累积 1 条承载记录（无事件帧未混入）")
    with tempfile.TemporaryDirectory(prefix="umvp_tail_") as td:
        out_dir = Path(td)
        saved: list = []

        def save_image(path, im):
            cv2.imwrite(str(path), im)
            saved.append(path.name)
            return True

        blk = server._finish_output(plan, acc, out_dir, "jpg", save_image)
        ok(blk.get("error") is None and blk["count"] == 1,
           f"publish 通路提炼块 count=1 无 error（{blk.get('error')}）")
        ok(blk["from"] == "pipeline.tail", f"块头 from=运行时实际来源: {blk['from']}")
        ok("publish" in (blk.get("note") or ""), f"publish 通路 note: {blk.get('note')}")
        it0 = blk["items"][0]
        ok(bool(it0["events"]) and bool(it0["images"]),
           f"events 明细 + 车牌图同帧配图落盘（{it0['images']}）")
        ok(all((out_dir / f).is_file() for f in it0["images"]), "配图文件真实存在")
        blk_s = server._finish_output(plan, acc, None, note="流式运行不落盘，提炼项不产出配图")
        ok("流式" in blk_s["note"] and "publish" in blk_s["note"],
           "流式 note 与来源 note 并存")

    # 真实空事件：通路在位（publish_wired）但本次无事件 → count=0 + note，不报 error
    plan_wired, _ = server._resolve_output_plan(spec)
    plan_wired["publish_wired"] = True
    blk0 = server._finish_output(plan_wired,
                                 server._make_output_accumulator(plan_wired), None)
    ok(blk0.get("count") == 0 and "error" not in blk0,
       "通路在位无事件 → count=0 不报 error（「没有」≠「取不到」）")
    ok("publish" in (blk0.get("note") or "") and "无事件" in blk0["note"],
       f"空事件 note 说明来源与「没有」: {blk0.get('note')}")

    # ③ 全程取不到：静态候选未命中、链路无 publish 事件尾 → error 不再静默空块
    plan_dead, _ = server._resolve_output_plan(spec)
    blk_dead = server._finish_output(plan_dead,
                                     server._make_output_accumulator(plan_dead), None)
    ok("末段无可用记录产物" in (blk_dead.get("error") or ""),
       f"取不到 → error 提示（不静默）: {blk_dead.get('error')}")
    ok(blk_dead["count"] == 0 and blk_dead["items"] == [] and blk_dead["from"] is None,
       "error 块仍带完整结构（count=0 / items=[] / from=None）")


# ---------------- 3. _finish_output（落盘/命名/thumbs/流式/Image 引用） ----------------

def test_finish_output():
    print("[3] _finish_output")
    plan = {"scope": "events", "image_pick": "all", "limit": 2,
            "from_full": "seq_filter#5.dets", "from_type": "Records",
            "image_full": "annotate#3.annotated_records", "image_type": "Records"}
    acc = server._make_output_accumulator(plan)
    img = make_img()
    acc["add"]([_rec(3, 7, True), _rec(3, 8, True), _rec(5, 9, True)],
               [{"frame_idx": 3, "image": img}, {"frame_idx": 3, "image": img},
                {"frame_idx": 5, "image": img}])
    with tempfile.TemporaryDirectory(prefix="umvp_outblk_") as td:
        out_dir = Path(td)
        saved: list = []
        counters = [0]

        def save_image(path, im):
            if counters[0] >= 200:
                return False
            cv2.imwrite(str(path), im)
            counters[0] += 1
            saved.append(path.name)
            return True

        block = server._finish_output(plan, acc, out_dir, "jpg", save_image,
                                      start_frame=0, saved_files=saved)
        ok(block["scope"] == "events" and block["from"] == "seq_filter#5.dets",
           "块头 scope/from")
        ok(block["count"] == 2 and block["truncated"] is True,
           f"limit=2 截断（count={block['count']}, truncated={block['truncated']}）")
        it0 = block["items"][0]
        ok(it0["track_id"] == 7 and it0["events"] and it0["events"][0]["name"] == "警情",
           "item 通用字段 + events 明细")
        ok(len(it0["images"]) == 1 and it0["images"][0].startswith("output_001_tid7_")
           and it0["images"][0].endswith("_1.jpg"), f"落盘命名: {it0['images'][0]}")
        ok((out_dir / it0["images"][0]).is_file(), "落盘文件存在；images 为相对 output_dir 路径")
        ok(it0.get("thumbs") and it0["thumbs"][0].startswith("data:image/jpeg;base64,"),
           "thumbs 缩略 data URL 随块返回")
        ok(block["items"][1]["track_id"] == 8 and block["items"][1]["images"],
           "第二条 item 配图落盘")
        ok(saved and all(n.startswith("output_") for n in saved) and saved == [
            block["items"][0]["images"][0], block["items"][1]["images"][0]],
           "共用 saved_files 清单（报告只嵌本次落盘）")
        # 落盘上限共用：计数已满 → 不落盘、images 不引用
        counters[0] = 200
        block2 = server._finish_output(plan, acc, out_dir, "jpg", save_image,
                                       start_frame=0, saved_files=saved)
        ok(all(not it.get("images") for it in block2["items"]) and not block2["items"][0].get("thumbs"),
           "落盘达上限 → 跳过写盘、images 不引用（共用 max_saved_images 语义）")

    # Image 帧级产物：saved_files 按帧号引用（不新落盘）
    plan_img = dict(plan, image_type="Image", image_full="annotate#3.annotated")
    acc_i = server._make_output_accumulator(plan_img)
    acc_i["add"]([_rec(93, 1, True), _rec(95, 2, True)], None)
    saved_i = ["frame_0003_x_orig.jpg", "frame_0003_x_det.jpg", "frame_0005_y_det.jpg"]
    block_i = server._finish_output(plan_img, acc_i, None, save_image=None,
                                    start_frame=90, saved_files=saved_i)
    ok(block_i["items"][0]["images"] == ["frame_0003_x_det.jpg"],
       "Image 产物引用 saved_files（start_frame 偏移 + 同帧最后一张优先）")
    ok(block_i["items"][1]["images"] == ["frame_0005_y_det.jpg"], "第二帧引用")

    # 流式：不落盘，images 恒空 + note
    block_s = server._finish_output(plan, acc, None,
                                    note="流式运行不落盘，提炼项不产出配图（需配图请用批量运行）")
    ok(all(not it.get("images") for it in block_s["items"]) and "note" in block_s,
       "流式块 images 恒空 + note")


def test_frame_index():
    print("[4] _output_frame_image_index")
    files = ["plate_001_京A_3.1s.jpg", "frame_0000_a_orig.jpg",
             "frame_0000_a_annotated.jpg", "frame_0012_b_det.jpg", "face_001_1.0s.jpg"]
    idx = server._output_frame_image_index(files, start_frame=90)
    ok(idx.get(90) == "frame_0000_a_annotated.jpg", "start_frame 偏移 + 同帧最后一张")
    ok(idx.get(102) == "frame_0012_b_det.jpg", "帧号映射")
    ok(90 + 0 in idx and len(idx) == 2, "非 frame_ 前缀不入映射")


# ---------------- 5. 报告消费（_build_report_html） ----------------

def test_report_output_block():
    print("[5] _build_report_html 链路尾输出区块")
    with tempfile.TemporaryDirectory(prefix="umvp_outrep_") as td:
        out_dir = Path(td)
        f1, f2 = "output_001_tid7_3.1s_1.jpg", "output_002_tid8_5.2s_1.jpg"
        cv2.imwrite(str(out_dir / f1), make_img())
        cv2.imwrite(str(out_dir / f2), make_img(160, 60))
        snap = {
            "name": "输出链路", "pipeline_id": "PIPE0001",
            "stages": ["frame_manager", "yolo", "det_filter", "annotate", "vlm", "seq_filter"],
            "media": {"type": "video", "name": "rescue1.MP4", "start_time": 0.0,
                      "end_time": 10.0, "start_frame": 0, "end_frame": 250},
            "stats": {"processed": 9, "detections": 5, "plates": 0, "events": 2,
                      "vlm_state": "模拟回填(fire)", "vlm_fail": 0,
                      "saved_images": 2, "dropped_saves": 0, "elapsed_sec": 1.2},
            "timeline": [{"kind": "submit", "stage": "seq_filter", "text": "事件 警情",
                          "ts": 3.1, "frame": 77,
                          "data": {"name": "警情", "detail": {"key": "7"}}}],
            "output_dir": str(out_dir),
            "output": {"scope": "events", "from": "seq_filter#5.dets", "count": 2,
                       "truncated": False,
                       "items": [
                           {"ts": 3.12, "frame_idx": 77, "label": "person", "track_id": 7,
                            "score": 0.91, "images": [f1],
                            "events": [{"name": "警情", "ts": 3.1,
                                        "detail": {"key": "7", "result": "斗殴"}}]},
                           {"ts": 5.2, "frame_idx": 130, "label": "person", "track_id": 8,
                            "score": 0.88, "images": [f2], "events": []},
                       ]},
            "generated_at": "2026-10-27 12:00:00", "mode": "sync",
        }
        html = server._build_report_html(snap)
        ok("链路尾输出（2 条）" in html, "区块标题与条数")
        i_out = html.index("链路尾输出")
        i_ev = html.index("事件列表")
        ok(i_out < i_ev, "区块优先呈现（事件表之前）")
        ok("scope=events" in html and "from=seq_filter#5.dets" in html, "scope/from 元信息")
        ok("斗殴" in html and "0.91" in html, "events 明细（detail_fields 投影键在 detail 内）与通用字段")
        ok(html.count("data:image/jpeg;base64,") >= 2 and f1 in html and f2 in html,
           "items.images 落盘图内嵌为 data URL")
        ok("事件列表（1 条）" in html, "既有事件表不受影响")

        # 上限生效
        html_lim = server._build_report_html(snap, max_output=1)
        ok(f1 in html_lim and f2 not in html_lim, "max_output=1 只嵌第一张")
        ok("超出上限未嵌入" in html_lim, "超限注明")

        # 声明解析失败 / 运行时取不到 → error 呈现
        snap_err = dict(snap)
        snap_err["output"] = {"error": "output.from='x.y' 无法解析: 未知产物"}
        html_err = server._build_report_html(snap_err)
        ok("输出块错误" in html_err and "无法解析" in html_err, "error 块呈现")

        # 空事件块（count=0 + note，无 error）：报告同样可见（「没有」可解释不静默）
        snap_empty = dict(snap)
        snap_empty["output"] = {"scope": "events", "from": "pipeline.tail", "count": 0,
                                "truncated": False, "items": [],
                                "note": "事件由链路尾 publish 产出，来源产物 pipeline.tail"
                                        "（本次运行无事件）"}
        html_empty = server._build_report_html(snap_empty)
        ok("链路尾输出（0 条）" in html_empty and "publish" in html_empty,
           "空事件块报告可见（note 说明来源）")

    # 无声明：现状一致（无该区块，其余内容不受影响）
    snap_wo = {k: v for k, v in snap.items() if k != "output"}
    snap_wo["output_dir"] = None
    html_wo = server._build_report_html(snap_wo)
    ok("链路尾输出" not in html_wo, "无 output 声明 → 无区块（与现状一致）")
    ok("事件列表（1 条）" in html_wo, "无声明时事件表照常")


def main() -> None:
    test_plan_resolve()
    test_accumulator()
    test_default_from_runtime()
    test_finish_output()
    test_frame_index()
    test_report_output_block()
    print(f"\n全部通过: {PASS} 断言")


if __name__ == "__main__":
    main()
