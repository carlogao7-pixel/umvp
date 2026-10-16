# -*- coding: utf-8 -*-
"""
VLM 送审图像源（image_source）与旁路图像命名统一（records.image）测试

背景（WO-0002.2，2026-10）：
  - pipe-chain A 段：records 统一旁路图像键 = image（to_record 自动补，旧键 img/face_img/
    plate_img 保留为兼容别名），新增 records.image_of(rec)（优先 image，回退旧别名）。
  - web 段：vlm 阶段新增参数 image_source（annotated=默认=现状 / frame / crops），
    驱动层按值取图（crops 用 req.det.bbox / req.dets 与 last_crops 精确或 IoU≥0.5 匹配），
    任一路径缺源一律回退不报错，timeline 备注素材来源。
  - pipe-chain 补漏：composer.step() 由「存在 target_crop 即不产生 submits」改为按
    「是否有 vlm」判定——target_crop + vlm 组合现正常送审（此前静默失效）。

覆盖：
  1. records：to_record 对 CropResult/FaceResult/PlateResult 均补 image 且旧键仍在、
     同对象；image_of 两类键都能读到；无旁路图记录（Detection）不受影响；不改原对象。
  2. _pick_vlm_image：annotated 默认缺省语义 / frame 与 annotated 确不同 /
     crops 命中取子图、未命中回退有提示。
  3. crop+vlm 端到端（真实 TargetCrop）：装配 step() 产生 submits 且 last_crops 有值，
     image_source=crops 能取到匹配子图；无 vlm 时仍不产生 submits（既有行为）。
  4. 驱动层（h_pipeline_run 图片分支，monkeypatch db.load_pipeline + _vlm_real +
     _endpoint_reachable，进程内、不起 HTTP/MySQL）：缺省（不填 image_source）与显式
     annotated 送出的图逐像素一致；frame / crops 切图确实不同；timeline 带素材来源备注。

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


# ==================== 2. _pick_vlm_image / _match_crop_for_req ====================

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

    # ① annotated 默认语义：有标注图（现状）
    img, note, fb = server._pick_vlm_image("annotated", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: annotated 取标注图", img is annotated and note == "标注图" and fb is False)
    # annotated 缺上游（无标注图）→ 回退原帧
    img, note, fb = server._pick_vlm_image("annotated", req, frame, _fake_pipe(None, []))
    check("pick: annotated 无标注图回退原帧", img is frame and fb is True and "回退" in note, note)

    # ② frame 与 annotated 确不同（无标注图时 frame 仍取原帧）
    img, note, fb = server._pick_vlm_image("frame", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: frame 取原帧", img is frame and note == "原帧" and fb is False)
    check("pick: frame 与 annotated 内容不同",
          not np.array_equal(img, annotated), f"max diff={int(np.abs(img.astype(int)-annotated.astype(int)).max())}")
    img, note, fb = server._pick_vlm_image("frame", req, None, _fake_pipe(annotated, []))
    check("pick: frame 无本帧图回退标注图", img is annotated and fb is True and "回退" in note, note)

    # ③ crops 命中：取匹配子图（req.dets bbox 精确匹配）
    img, note, fb = server._pick_vlm_image("crops", req, frame, _fake_pipe(annotated, [crop]))
    check("pick: crops 命中取子图", img is sub and note == "裁剪子图" and fb is False)
    check("pick: crops 子图与标注图不同", not np.array_equal(img, annotated))
    check("pick: crops 子图与 last_crops[0].img 一致", np.array_equal(img, crop.img))

    # crops 命中（用 req.det.bbox，单目标素材）
    img, note, fb = server._pick_vlm_image("crops", _req(use_det=True), frame,
                                           _fake_pipe(annotated, [crop]))
    check("pick: crops 用 req.det.bbox 命中", img is sub and fb is False)

    # crops 未命中：回退标注图 + 提示
    miss_req = _req(bbox=(500, 400, 560, 460))
    img, note, fb = server._pick_vlm_image("crops", miss_req, frame, _fake_pipe(annotated, [crop]))
    check("pick: crops 未命中回退标注图", img is annotated and fb is True and "回退" in note, note)
    # crops 未命中且无标注图：回退原帧
    img, note, fb = server._pick_vlm_image("crops", miss_req, frame, _fake_pipe(None, [crop]))
    check("pick: crops 未命中且无标注图回退原帧", img is frame and fb is True and "回退" in note, note)
    # crops 无 last_crops：回退
    img, note, fb = server._pick_vlm_image("crops", req, frame, _fake_pipe(annotated, []))
    check("pick: crops 无 last_crops 回退", img is annotated and fb is True, note)

    # --- _match_crop_for_req：精确 / IoU≥0.5 命中 / 不命中 ---
    hit = server._match_crop_for_req(_req(bbox=(20, 10, 80, 50)), [crop])
    check("match: bbox 精确命中", hit is crop)
    # 略偏但 IoU≥0.5：crop (20,10,80,50) 面积 60*40=2400，偏移后交集占比>0.5
    hit2 = server._match_crop_for_req(_req(bbox=(24, 14, 84, 54)), [crop])
    check("match: IoU≥0.5 命中", hit2 is crop)
    miss = server._match_crop_for_req(_req(bbox=(200, 200, 260, 240)), [crop])
    check("match: 不命中返回 None", miss is None)
    check("match: 空 crops → None", server._match_crop_for_req(_req(), []) is None)


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

    # image_source=crops 能取到匹配子图
    img, note, fb = server._pick_vlm_image("crops", subs[0], frame, pipe)
    check("crop+vlm: image_source=crops 命中子图", fb is False and note == "裁剪子图", note)
    check("crop+vlm: 子图与 last_crops[0].img 一致",
          img is not None and np.array_equal(img, pipe.last_crops[0].img))
    check("crop+vlm: 子图尺寸=裁剪框尺寸",
          img.shape[:2] == pipe.last_crops[0].img.shape[:2],
          f"{img.shape[1]}x{img.shape[0]}")

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
        return ("none", "", "")

    server._vlm_real = fake_real
    server._endpoint_reachable = lambda url, timeout=2.0: True

    def spec(image_source):
        vp = {"use_real_vlm": True}
        if image_source is not None:
            vp["image_source"] = image_source
        return {"name": "qa-src", "stages": [
            {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
            {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt",
                                             "conf": 0.25, "classes": "0,2"}},
            {"module_id": "target_crop", "params": {"classes": "0,2"}},
            {"module_id": "annotate", "params": {"thickness": 2, "show_label": True}},
            {"module_id": "vlm", "params": vp},
            {"module_id": "det_filter", "params": {"filter": ""}},  # 链尾（终段合法）
        ]}

    def run(image_source):
        captured.clear()
        with tempfile.TemporaryDirectory(prefix="umvp_src_") as td:
            server.db.load_pipeline = lambda pid: spec(image_source)
            r = server.h_pipeline_run({"pipeline_id": "PIPE0001", "image": "人遮挡1.png",
                                       "output_dir": td, "image_format": "jpg",
                                       "vlm_feedback": True, "vlm_action": "fire"})
        return r, captured.get("img")

    try:
        r_abs, img_abs = run(None)              # 不填 → 默认 annotated
        r_ann, img_ann = run("annotated")       # 显式 annotated
        r_frm, img_frm = run("frame")           # 原帧
        r_crp, img_crp = run("crops")           # 裁剪子图

        check("驱动: 缺省与 annotated 均运行成功",
              all(x.get("ok") is True for x in (r_abs, r_ann, r_frm, r_crp)),
              str([x.get("error") for x in (r_abs, r_ann, r_frm, r_crp)]))
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

        def tl_text(r):
            return " ".join((t.get("text") or "") for t in (r.get("timeline") or []))

        check("驱动: annotated 无素材备注（现状语义）", "素材:" not in tl_text(r_ann))
        check("驱动: 缺省（=annotated）无素材备注", "素材:" not in tl_text(r_abs))
        check("驱动: frame 研判日志带素材来源备注",
              "素材:原帧" in tl_text(r_frm), tl_text(r_frm)[-60:])
        check("驱动: crops 研判日志带素材来源备注",
              "素材:裁剪子图" in tl_text(r_crp), tl_text(r_crp)[-60:])
    finally:
        server.db.load_pipeline = orig_load
        server._vlm_real = orig_real
        server._endpoint_reachable = orig_reach


def main() -> None:
    print("== 1. records：旁路图像命名统一 ==")
    records_checks()
    print("== 2. _pick_vlm_image / _match_crop_for_req ==")
    pick_checks()
    print("== 3. crop+vlm 端到端（真实 TargetCrop） ==")
    crop_vlm_checks()
    print("== 4. 驱动层：缺省零回归 + 切图差异 ==")
    driver_checks()
    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
