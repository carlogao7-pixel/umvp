# -*- coding: utf-8 -*-
"""
VLM 送审图像源（image_source）与旁路图像命名统一（records.image）测试

背景（WO-0002.2，2026-10；WO-0003.2 扩展图集+多图送审）：
  - pipe-chain A 段：records 统一旁路图像键 = image，并区分 image_of（单图语义）/
    image_set_of（集合语义：单图→[图]、图集→原样、无→[]）。
  - web 段：vlm 阶段参数 image_source（annotated=默认 / frame / crops）决定取图来源，
    参数 image_pick（first=默认=现状 / all）决定单图或图集全部；驱动层按源取图
    （crops 用 req.det.bbox / req.dets 与 last_crops 精确或 IoU≥0.5 匹配，收集**全部**
    命中子图=图集），任一路径缺源一律回退不报错，timeline 备注素材来源（多图带张数）。
  - pipe-chain 补漏：composer.step() 由「存在 target_crop 即不产生 submits」改为按
    「是否有 vlm」判定——target_crop + vlm 组合现正常送审（此前静默失效）。

覆盖：
  1. records：to_record 对 CropResult/FaceResult/PlateResult 均补 image 且旧键仍在、
     同对象；image_of 两类键都能读到；无旁路图记录（Detection）不受影响；不改原对象。
  2. _pick_vlm_images：annotated 默认缺省语义（长度1）/ frame 与 annotated 确不同 /
     crops 命中取**全部**匹配子图（多对象=图集）、未命中回退有提示；
     _match_crops_for_req 全命中 / _match_crop_for_req 首图（兼容）。
  3. crop+vlm 端到端（真实 TargetCrop）：装配 step() 产生 submits 且 last_crops 有值，
     image_source=crops 能取到匹配子图；无 vlm 时仍不产生 submits（既有行为）。
  4. 驱动层（h_pipeline_run 图片分支，monkeypatch db.load_pipeline + _vlm_real +
     _endpoint_reachable，进程内、不起 HTTP/MySQL）：缺省（不填 image_source）与显式
     annotated 送出的图逐像素一致；image_pick 缺省与显式 first 逐像素一致（零回归）；
     frame / crops 切图确实不同；timeline 带素材来源备注。
  5. 多图 payload：image_pick=all 时 _vlm_real 构造 OpenAI 兼容多图消息（N 个 image_url
     content 项 + 末尾 text），first/单图仍为 [image_url, text]（现状语义）。

运行: conda run -n ai python tests/test_image_source.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

from pipe.composer import Detection, FrameManager, Pipeline, SubmitRequest, VLMAnalyzer  # noqa: E402
from pipe.records import IMAGE_ALIASES, IMAGE_KEY, image_of, to_record  # noqa: E402
from pipe.target_crop import CropResult, TargetCrop  # noqa: E402
from face_detect.face_pipeline import FaceResult  # noqa: E402
from plate_recog.plate_recognizer import PlateResult  # noqa: E402

DATA = ROOT / "tests/data/pic"
PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _img(v: int, shape=(20, 30, 3)) -> np.ndarray:
    return np.full(shape, v, dtype=np.uint8)


# ==================== 1. records：旁路图像命名统一 ====================

def records_checks() -> None:
    check("records: IMAGE_KEY=image", IMAGE_KEY == "image")
    check("records: 兼容别名含 img/face_img/plate_img",
          set(IMAGE_ALIASES) == {"img", "face_img", "plate_img"}, str(IMAGE_ALIASES))

    # --- CropResult：img → image（旧键仍在，同对象）---
    c_img = _img(11)
    c = CropResult(img=c_img, bbox=(10, 10, 40, 30), offset=(10, 10), scale=1.0,
                   cls_id=0, score=0.9, track_id=0, index=0)
    rc = to_record(c)
    check("records: CropResult 补 image", "image" in rc, str(sorted(rc)))
    check("records: CropResult 旧键 img 仍在", "img" in rc)
    check("records: CropResult image 与 img 同对象", rc["image"] is rc["img"])
    check("records: to_record 不改原对象（无 image 属性）", not hasattr(c, "image"))

    # --- FaceResult：face_img → image ---
    f_img = _img(22, (16, 16, 3))
    f = FaceResult(person_bbox=(0, 0, 50, 80), face_bbox=(5, 5, 21, 21), face_img=f_img,
                   score=0.8, kps=[(0.0, 0.0)] * 5, upscale=1.0, index=0)
    rf = to_record(f)
    check("records: FaceResult 补 image", rf.get("image") is f_img)
    check("records: FaceResult 旧键 face_img 仍在且同对象",
          rf.get("face_img") is f_img and rf["image"] is rf["face_img"])

    # --- PlateResult（有 plate_img）：plate_img → image ---
    p_img = _img(33, (12, 40, 3))
    p = PlateResult(plate_no="粤ACJ0710", plate_type=3, type_name="绿牌新能源", color="green",
                    score=0.95, bbox=(0, 0, 40, 12), plate_img=p_img)
    rp = to_record(p)
    check("records: PlateResult 补 image", rp.get("image") is p_img)
    check("records: PlateResult 旧键 plate_img 仍在且同对象",
          rp.get("plate_img") is p_img and rp["image"] is rp["plate_img"])

    # --- PlateResult（plate_img=None）：不应凭空造 image ---
    p0 = PlateResult(plate_no="京A00000", plate_type=0, type_name="蓝牌", color="blue",
                     score=0.5, bbox=(0, 0, 40, 12))
    r0 = to_record(p0)
    check("records: PlateResult 无旁路图时不加 image", "image" not in r0, str(sorted(r0)))
    check("records: PlateResult 无旁路图时 image_of=None", image_of(r0) is None)

    # --- Detection：无任何旁路图键，不受影响 ---
    rd = to_record(Detection(track_id=1, cls_id=0, bbox=(1, 2, 3, 4), score=0.7))
    check("records: Detection 不加 image/别名",
          "image" not in rd and not any(k in rd for k in IMAGE_ALIASES), str(sorted(rd)))
    check("records: Detection image_of=None", image_of(rd) is None)

    # --- image_of：两类键都能读到；priority=image ---
    check("image_of: dict 仅 img", image_of({"img": c_img}) is c_img)
    check("image_of: dict 仅 image", image_of({"image": f_img}) is f_img)
    check("image_of: 两键并存优先 image", image_of({"image": f_img, "img": c_img}) is f_img)
    check("image_of: 对象回退别名属性", image_of(c) is c_img)
    check("image_of: 空 dict → None", image_of({}) is None)

    # --- dict 输入：原地补 image 于副本，源 dict 不被改 ---
    src = {"kind": "crop", "img": c_img}
    out = to_record(src)
    check("records: dict 输入补 image 到副本", out.get("image") is c_img)
    check("records: dict 输入源对象不被改", "image" not in src)

    # --- 已带 image（含显式 None）不被别名覆盖 ---
    keep = to_record({"image": f_img, "img": c_img})
    check("records: 已有 image 不被别名覆盖", keep["image"] is f_img)


# ==================== 2. _pick_vlm_images / _match_crops_for_req ====================

def _fake_pipe(annotated=None, crops=None):
    class _P:
        pass
    p = _P()
    p.last_annotated = annotated
    p.last_crops = list(crops or [])
    return p


def _req(bbox=(20, 10, 80, 50), use_det=False):
    det = Detection(track_id=1, cls_id=0, bbox=bbox, score=0.9) if use_det else None
    return SubmitRequest(track_key=0, gran="class", ref="full:cls0", det=det,
                         dets=[] if use_det else [Detection(track_id=1, cls_id=0,
                                                            bbox=bbox, score=0.9)])


def pick_checks() -> None:
    import server

    frame = _img(10, (100, 120, 3))
    annotated = _img(20, (100, 120, 3))
    sub = frame[10:50, 20:80].copy()
    crop = CropResult(img=sub, bbox=(20, 10, 80, 50), offset=(20, 10), scale=1.0,
                      cls_id=0, score=0.9, track_id=0, index=0)
    req = _req()

    # ① annotated 默认语义：有标注图（现状）→ 列表长度 1
    imgs, note, fb = server._pick_vlm_images("annotated", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: annotated 取标注图（长度1）", imgs == [annotated] and note == "标注图" and fb is False)
    # annotated 缺上游（无标注图）→ 回退原帧
    imgs, note, fb = server._pick_vlm_images("annotated", req, frame, _fake_pipe(None, []))
    check("pick: annotated 无标注图回退原帧", imgs == [frame] and fb is True and "回退" in note, note)

    # ② frame 与 annotated 确不同（无标注图时 frame 仍取原帧）
    imgs, note, fb = server._pick_vlm_images("frame", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: frame 取原帧（长度1）", imgs == [frame] and note == "原帧" and fb is False)
    check("pick: frame 与 annotated 内容不同",
          not np.array_equal(frame, annotated), f"max diff={int(np.abs(frame.astype(int)-annotated.astype(int)).max())}")
    imgs, note, fb = server._pick_vlm_images("frame", req, None, _fake_pipe(annotated, []))
    check("pick: frame 无本帧图回退标注图", imgs == [annotated] and fb is True and "回退" in note, note)

    # ③ crops 命中：取匹配子图（列表）
    imgs, note, fb = server._pick_vlm_images("crops", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: crops 命中取子图（长度1）", imgs == [sub] and note == "裁剪子图" and fb is False)
    check("pick: crops 子图与标注图不同", not np.array_equal(imgs[0], annotated))
    check("pick: crops 子图与 last_crops[0].img 一致", np.array_equal(imgs[0], crop.img))

    # crops 命中（用 req.det.bbox，单目标素材）
    imgs, note, fb = server._pick_vlm_images("crops", _req(use_det=True), frame,
                                             _fake_pipe(annotated, [crop]))
    check("pick: crops 用 req.det.bbox 命中", imgs == [sub] and fb is False)

    # crops 多对象（图集语义，基数约定 §3⑥）：req.dets 两个 bbox → 收集全部命中子图
    req2 = SubmitRequest(track_key=0, gran="class", ref="full:cls0", det=None, dets=[
        Detection(track_id=1, cls_id=0, bbox=(20, 10, 80, 50), score=0.9),
        Detection(track_id=2, cls_id=0, bbox=(40, 10, 90, 50), score=0.8)])
    sub2 = frame[10:50, 40:90].copy()
    crop2 = CropResult(img=sub2, bbox=(40, 10, 90, 50), offset=(40, 10), scale=1.0,
                       cls_id=0, score=0.8, track_id=0, index=1)
    crop3 = CropResult(img=frame[0:20, 0:20].copy(), bbox=(0, 0, 20, 20), offset=(0, 0),
                       scale=1.0, cls_id=0, score=0.5, track_id=0, index=2)  # 不相关，不应入选
    imgs, note, fb = server._pick_vlm_images("crops", req2, frame,
                                             _fake_pipe(annotated, [crop, crop2, crop3]))
    check("pick: crops 图集收集全部命中（长度2，未命中的不入选）",
          len(imgs) == 2 and fb is False, str(len(imgs)))
    check("pick: 图集按 crops 顺序且与各自子图一致",
          np.array_equal(imgs[0], crop.img) and np.array_equal(imgs[1], crop2.img))

    # crops 未命中：回退标注图 + 提示
    miss_req = _req(bbox=(500, 400, 560, 460))
    imgs, note, fb = server._pick_vlm_images("crops", miss_req, frame, _fake_pipe(annotated, [crop]))
    check("pick: crops 未命中回退标注图", imgs == [annotated] and fb is True and "回退" in note, note)
    # crops 未命中且无标注图：回退原帧
    imgs, note, fb = server._pick_vlm_images("crops", miss_req, frame, _fake_pipe(None, [crop]))
    check("pick: crops 未命中且无标注图回退原帧", imgs == [frame] and fb is True and "回退" in note, note)
    # crops 无 last_crops：回退
    imgs, note, fb = server._pick_vlm_images("crops", req, frame, _fake_pipe(annotated, []))
    check("pick: crops 无 last_crops 回退", imgs == [annotated] and fb is True, note)

    # --- _match_crops_for_req（全部命中）/ _match_crop_for_req（首个，兼容） ---
    hit = server._match_crop_for_req(_req(bbox=(20, 10, 80, 50)), [crop])
    check("match: bbox 精确命中", hit is crop)
    # 略偏但 IoU≥0.5：crop (20,10,80,50) 面积 60*40=2400，偏移后交集占比>0.5
    hit2 = server._match_crop_for_req(_req(bbox=(24, 14, 84, 54)), [crop])
    check("match: IoU≥0.5 命中", hit2 is crop)
    miss = server._match_crop_for_req(_req(bbox=(200, 200, 260, 240)), [crop])
    check("match: 不命中返回 None", miss is None)
    check("match: 空 crops → None", server._match_crop_for_req(_req(), []) is None)
    check("match: 多命中返回全部（列表，按 crops 顺序）",
          server._match_crops_for_req(req2, [crop, crop2, crop3]) == [crop, crop2])


# ==================== 3. crop+vlm 端到端（真实 TargetCrop） ====================

def _crop_vlm_spec(with_vlm: bool) -> dict:
    stages = [
        {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "conf": 0.25,
                                         "classes": "0,2"}},
        {"module_id": "target_crop", "params": {"classes": "0,2"}},
    ]
    if with_vlm:
        stages.append({"module_id": "vlm", "params": {"use_real_vlm": False}})
    # 链尾：seq_filter（空规则=不判定），满足「终段合法」约定
    stages.append({"module_id": "seq_filter",
                   "params": {"throttle": "", "change": "", "rules": ""}})
    return {"name": "qa-crop-vlm", "stages": stages}


def crop_vlm_checks() -> None:
    import cv2
    import server

    frame = cv2.imread(str(DATA / "人遮挡1.png"))
    check("crop+vlm 素材: 测试图存在", frame is not None, str(DATA / "人遮挡1.png"))
    h, w = frame.shape[:2]
    det = Detection(track_id=1, cls_id=0, bbox=(int(w * 0.1), int(h * 0.1),
                                                int(w * 0.5), int(h * 0.6)), score=0.9)

    # 有 vlm：target_crop 与 vlm 共存 → submits 产生（补漏回归）
    pipe = server._chain_from_spec(_crop_vlm_spec(with_vlm=True))
    check("crop+vlm 装配: crop 是真实 TargetCrop", isinstance(pipe.crop, TargetCrop))
    check("crop+vlm 装配: vlm 插槽已注入", pipe.vlm is not None)
    subs, _ = pipe.step(0, 0.0, dets=[det], frame=frame, force=True)
    check("crop+vlm step: 产生 submits（此前静默失效）", len(subs) == 1, f"{len(subs)} 条")
    check("crop+vlm step: last_crops 有值", len(pipe.last_crops) == 1, f"{len(pipe.last_crops)} 张")
    check("crop+vlm step: last_crops[0] 为 CropResult",
          isinstance(pipe.last_crops[0], CropResult))

    # image_source=crops 能取到匹配子图（列表）
    imgs, note, fb = server._pick_vlm_images("crops", subs[0], frame, pipe)
    check("crop+vlm: image_source=crops 命中子图", fb is False and note == "裁剪子图", note)
    check("crop+vlm: 子图与 last_crops[0].img 一致",
          bool(imgs) and np.array_equal(imgs[0], pipe.last_crops[0].img))
    check("crop+vlm: 子图尺寸=裁剪框尺寸",
          bool(imgs) and imgs[0].shape[:2] == pipe.last_crops[0].img.shape[:2],
          f"{imgs[0].shape[1]}x{imgs[0].shape[0]}" if imgs else "无")

    # 无 vlm：仅有 crop → 不产生 submits（车牌/人脸链既有行为不变）
    pipe2 = server._chain_from_spec(_crop_vlm_spec(with_vlm=False))
    subs2, _ = pipe2.step(0, 0.0, dets=[det], frame=frame, force=True)
    check("仅 crop 无 vlm: 不产生 submits（既有行为）", subs2 == [], f"{len(subs2)} 条")
    check("仅 crop 无 vlm: last_crops 仍有值（供下游识别）", len(pipe2.last_crops) == 1)


# ==================== 4. 驱动层：缺省零回归 + 切图差异（进程内） ====================

def driver_checks() -> None:
    import tempfile
    import server

    orig_load = server.db.load_pipeline
    orig_real = server._vlm_real
    orig_reach = server._endpoint_reachable
    captured: dict = {}

    def fake_real(p, payload, img):
        captured["img"] = img
        captured.setdefault("all", []).append(img)
        return ("none", "", "")

    server._vlm_real = fake_real
    server._endpoint_reachable = lambda url, timeout=2.0: True

    def spec(image_source, image_pick=None):
        vp = {"use_real_vlm": True}
        if image_source is not None:
            vp["image_source"] = image_source
        if image_pick is not None:
            vp["image_pick"] = image_pick
        return {"name": "qa-src", "stages": [
            {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
            {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt",
                                             "conf": 0.25, "classes": "0,2"}},
            {"module_id": "target_crop", "params": {"classes": "0,2"}},
            {"module_id": "annotate", "params": {"thickness": 2, "show_label": True}},
            {"module_id": "vlm", "params": vp},
            {"module_id": "det_filter", "params": {"filter": ""}},  # 链尾（终段合法）
        ]}

    def run(image_source, image_pick=None):
        captured.clear()
        with tempfile.TemporaryDirectory(prefix="umvp_src_") as td:
            server.db.load_pipeline = lambda pid: spec(image_source, image_pick)
            r = server.h_pipeline_run({"pipeline_id": "PIPE0001", "image": "人遮挡1.png",
                                       "output_dir": td, "image_format": "jpg",
                                       "vlm_feedback": True, "vlm_action": "fire"})
        return r, captured.get("img"), list(captured.get("all") or [])

    try:
        r_abs, img_abs, _ = run(None)                # 不填 → 默认 annotated
        r_ann, img_ann, _ = run("annotated")         # 显式 annotated
        r_frm, img_frm, _ = run("frame")             # 原帧
        r_crp, img_crp, _ = run("crops")             # 裁剪子图（image_pick 不填=默认 first）
        r_crp1, img_crp1, _ = run("crops", "first")  # 显式 first
        r_crpa, _, imgs_crpa = run("crops", "all")   # 图集全部

        check("驱动: 缺省与 annotated 均运行成功",
              all(x.get("ok") is True for x in (r_abs, r_ann, r_frm, r_crp, r_crp1, r_crpa)),
              str([x.get("error") for x in (r_abs, r_ann, r_frm, r_crp, r_crp1, r_crpa)]))
        check("驱动: 缺省真实调用拿到送审图", img_abs is not None and img_ann is not None)
        check("驱动: 缺省送出图 == 显式 annotated（零回归）",
              np.array_equal(img_abs, img_ann))
        check("驱动: frame 与 annotated 送出图确实不同",
              not np.array_equal(img_frm, img_ann),
              f"frame{img_frm.shape} vs anno{img_ann.shape}")
        check("驱动: crops 与 annotated 送出图确实不同",
              not np.array_equal(img_crp, img_ann),
              f"crops{img_crp.shape} vs anno{img_ann.shape}")
        check("驱动: crops 送出图为子图（尺寸小于整帧）",
              img_crp.shape[0] < img_ann.shape[0] or img_crp.shape[1] < img_ann.shape[1],
              f"crops{img_crp.shape[:2]} anno{img_ann.shape[:2]}")
        # image_pick：不填（默认 first）与显式 first 逐像素一致（零回归）
        check("驱动: image_pick 缺省 == 显式 first（逐像素零回归）",
              np.array_equal(img_crp, img_crp1))
        # all：若该类命中多目标则送多图列表（否则仍单图，行为不劣化）
        multi = [x for x in imgs_crpa if isinstance(x, list) and len(x) > 1]
        check("驱动: crops+all 运行成功且（该类多目标时）送出多图列表",
              r_crpa.get("ok") is True
              and (not multi or all(len(x) >= 2 for x in multi)),
              f"多图提交数={len(multi)}")

        def tl_text(r):
            return " ".join((t.get("text") or "") for t in (r.get("timeline") or []))

        check("驱动: annotated 无素材备注（现状语义）", "素材:" not in tl_text(r_ann))
        check("驱动: 缺省（=annotated）无素材备注", "素材:" not in tl_text(r_abs))
        check("驱动: frame 研判日志带素材来源备注",
              "素材:原帧" in tl_text(r_frm), tl_text(r_frm)[-60:])
        check("驱动: crops 研判日志带素材来源备注（默认 first 不加张数）",
              "素材:裁剪子图" in tl_text(r_crp) and "素材:裁剪子图×" not in tl_text(r_crp),
              tl_text(r_crp)[-60:])
        # all 且该类命中多目标时，备注带张数（·素材:裁剪子图×N）
        if multi:
            check("驱动: crops+all 多图时备注带张数",
                  "素材:裁剪子图×" in tl_text(r_crpa), tl_text(r_crpa)[-80:])
    finally:
        server.db.load_pipeline = orig_load
        server._vlm_real = orig_real
        server._endpoint_reachable = orig_reach


# ==================== 5. 多图 payload（image_pick=all）+ 张数备注（进程内） ====================

def _spy_requests(cap: dict):
    """替换 sys.modules['requests'] 为捕获 post 的桩，返回恢复函数。"""
    import types

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"result": "none", "description": ""}'}}]}

    def _post(url, json=None, headers=None, timeout=None):
        cap["body"] = json
        return _Resp()

    stub = types.ModuleType("requests")
    stub.post = _post
    orig = sys.modules.get("requests")
    sys.modules["requests"] = stub

    def restore():
        if orig is not None:
            sys.modules["requests"] = orig
        else:
            sys.modules.pop("requests", None)
    return restore


def multi_image_checks() -> None:
    import server

    frame = _img(10, (100, 120, 3))
    annotated = _img(20, (100, 120, 3))
    n = 3
    crops, dets = [], []
    for i in range(n):
        x1 = 10 + i * 20
        crops.append(CropResult(img=_img(30 + i, (20, 20, 3)), bbox=(x1, 10, x1 + 20, 30),
                                offset=(x1, 10), scale=1.0, cls_id=0, score=0.9,
                                track_id=i + 1, index=i))
        dets.append(Detection(track_id=i + 1, cls_id=0, bbox=(x1, 10, x1 + 20, 30), score=0.9))
    req = SubmitRequest(track_key=0, gran="class", ref="full:cls0", det=None, dets=dets)
    pipe = _fake_pipe(annotated, crops)

    imgs, note, fb = server._pick_vlm_images("crops", req, frame, pipe)
    check("multi: crops 图集长度=该类命中数(3)", len(imgs) == n and fb is False, str(len(imgs)))
    check("multi: first 只取首张（默认单图）", imgs[:1] == [crops[0].img])
    check("multi: all 取图集全部（顺序一致）",
          [id(x) for x in imgs] == [id(c.img) for c in crops])
    check("multi: 多图备注带张数", (note + "×3") == "裁剪子图×3", note)

    # 捕获送给 _vlm_real 的多图 payload（OpenAI 兼容 content 项）
    cap: dict = {}
    restore = _spy_requests(cap)
    try:
        server._vlm_real({}, {"max_resolution": (1280, 720)}, imgs)   # all：列表 → 多图
        content_multi = cap["body"]["messages"][0]["content"]
        server._vlm_real({}, {"max_resolution": (1280, 720)}, imgs[0])  # first：单图
        content_single = cap["body"]["messages"][0]["content"]
    finally:
        restore()

    img_items = [c for c in content_multi if c.get("type") == "image_url"]
    txt_items = [c for c in content_multi if c.get("type") == "text"]
    check("multi: 多图 payload 含 3 个 image_url content 项", len(img_items) == n, str(len(img_items)))
    check("multi: 多图 payload 末尾单个 text 项", len(txt_items) == 1
          and content_multi[-1].get("type") == "text")
    check("multi: 单图 payload 为 [image_url, text]（现状语义）",
          len(content_single) == 2 and content_single[0].get("type") == "image_url"
          and content_single[-1].get("type") == "text")
    check("multi: 单图/多图 image_url 均为 data:image/jpeg;base64",
          all(c["image_url"]["url"].startswith("data:image/jpeg;base64,")
              for c in img_items))


def main() -> None:
    print("== 1. records：旁路图像命名统一 ==")
    records_checks()
    print("== 2. _pick_vlm_images / _match_crops_for_req ==")
    pick_checks()
    print("== 3. crop+vlm 端到端（真实 TargetCrop） ==")
    crop_vlm_checks()
    print("== 4. 驱动层：缺省零回归 + 切图差异 + image_pick ==")
    driver_checks()
    print("== 5. 多图 payload（image_pick=all）+ 张数备注 ==")
    multi_image_checks()
    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
