# -*- coding: utf-8 -*-
"""
人脸截取链路测试（PIPE0003「人脸截取链路」核心逻辑）：帧管理 → YOLO 识人 → 原分辨率人脸提取

覆盖（不含向量嵌入/底库检索——该链路明确不做身份去重）:
  1. 帧管理门控：frame_skip=3 时仅放行 0/3/6/9 帧（纯逻辑，不加载模型）
  2. 逐帧截取：每张测试图跑 extract_faces，输出原分辨率人脸（框在图内、出图=框尺寸）
  3. 同帧去重：无两个人脸框 IoU>0.5
  4. 链路装配：Pipeline(frame, yolo) 无 vlm/plate/告警依赖
  5. 服务级装配（进程内，不起 HTTP/MySQL）：web_lab.server._chain_from_spec 把人脸阶段
     注入 Pipeline.face 插槽（face_detect 新形态 + extract_faces 兼容形态），人脸链与
     普通链一样走 pipe.step（2026-10 WO-0002.1 起，原 _face_step 专用单帧路径已删）
  6. det_filter 生效：链路上游 det_filter 严格过滤（放行 0）时下游无裁剪/无人脸；
     放行时 last_face_input/last_faces 与 last_filter 统计一致（WO-0001 事故回归）
  7. 驱动层（h_pipeline_run 图片分支，monkeypatch db.load_pipeline 避开 MySQL/HTTP）：
     人脸落盘命名 face_{n:03d}_{ts}s、落盘上限跳盘语义、时间线含各阶段条目

运行: conda run -n ai python tests/test_face_capture_pipeline.py
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]  # umvp 项目根
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))  # 服务级装配（server._chain_from_spec）

from face_detect.face_detector import FaceDetector
from face_detect.face_pipeline import FaceStage
from pipe.composer import Detection, FrameManager, Pipeline, YOLODetector
from pipe.cropper import Cropper, iou
from pipe.crop_restore import extract_faces

DATA = ROOT / "tests/data/pic"
YOLO_MODEL = ROOT / "models/yolov8n.pt"
CLS_PERSON = 0

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _spec(*pairs):
    return {"name": "qa-face", "stages": [{"module_id": m, "params": p} for m, p in pairs]}


def _face_detect_stages(filter_expr: str):
    """人脸截取链路 stages（新形态：face_detect 阶段），仅 det_filter 条件可调。"""
    return [
        {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt", "conf": 0.3,
                                         "iou": 0.6, "imgsz": 1280, "max_det": 300, "classes": "0"}},
        {"module_id": "det_filter", "params": {"filter": filter_expr}},
        {"module_id": "target_crop", "params": {"classes": "0"}},
        {"module_id": "face_detect", "params": {"det_thresh": 0.4, "det_size": "640,640"}},
    ]


def _face_detect_chain(filter_expr: str):
    return {"name": "qa-face", "stages": _face_detect_stages(filter_expr)}


def service_checks() -> None:
    """服务级装配 + det_filter 生效（进程内调装配/驱动，不起 HTTP / MySQL）。"""
    import server  # 惰性导入：server 顶层无 HTTP/DB 副作用（db 连接惰性）
    from face_embed.face_embedder import FaceEmbedder

    # 4.0 「声明即执行」依据的目标集合覆盖人脸三形态与裁剪/筛选阶段
    check("服务级: CHAIN_ASSEMBLED_MODULES 登记人脸阶段",
          {"face_detect", "face_embed", "extract_faces"} <= server.CHAIN_ASSEMBLED_MODULES)
    check("服务级: CHAIN_ASSEMBLED_MODULES 登记裁剪/筛选阶段",
          {"det_filter", "target_crop"} <= server.CHAIN_ASSEMBLED_MODULES)

    # 4.1 新形态：face_detect → 装配层注入 Pipeline.face（FaceStage）
    p_loose = server._chain_from_spec(_face_detect_chain("long >= 1"))
    check("服务级装配: face_detect 注入 Pipeline.face（FaceStage）",
          isinstance(p_loose.face, FaceStage))
    check("服务级装配: face 插槽初始 last_faces/last_face_input 为空",
          p_loose.last_faces == [] and p_loose.last_face_input == [])
    check("服务级装配: 无 face_embed 时 face_embedder 为 None", p_loose.face_embedder is None)

    # 4.2 兼容形态：extract_faces（无 yolo/target_crop）自动补齐 face/crop/yolo
    legacy = server._chain_from_spec(_spec(
        ("frame_manager", {"frame_skip": 1}),
        ("det_filter", {"filter": ""}),
        ("extract_faces", {"face_thresh": 0.4, "margin": 0.15, "upscale": 1.0,
                           "yolo_classes": "0"}),
    ))
    check("服务级装配(兼容): extract_faces 注入 Pipeline.face（FaceStage）",
          isinstance(legacy.face, FaceStage))
    check("服务级装配(兼容): 缺 target_crop/yolo 时按 extract_faces 补齐",
          legacy.crop is not None and legacy.yolo is not None)

    # 4.3 face_embed 形态：额外注入 FaceEmbedder（身份去重在驱动层读 last_faces 后处理）
    p_emb = server._chain_from_spec(_spec(
        ("frame_manager", {"frame_skip": 1}),
        ("yolo", {"model_path": "models/yolov8n.pt", "classes": "0"}),
        ("det_filter", {"filter": ""}),
        ("target_crop", {"classes": "0"}),
        ("face_detect", {"det_thresh": 0.4}),
        ("face_embed", {}),
    ))
    check("服务级装配(嵌入): face_embed 注入 FaceEmbedder + Pipeline.face",
          isinstance(p_emb.face, FaceStage) and isinstance(p_emb.face_embedder, FaceEmbedder))

    # 4.4 det_filter 生效：同帧 + 同整帧检测，仅 det_filter 条件不同 → 下游裁剪/检脸随放行数
    frame = cv2.imread(str(DATA / "人遮挡1.png"))
    h, w = frame.shape[:2]
    det = Detection(track_id=0, cls_id=0, bbox=(0, 0, w, h), score=0.9)

    p_loose.step(0, 0.0, dets=[det], frame=frame, force=True)
    check("det_filter 放行: last_filter=(1,1)", p_loose.last_filter == (1, 1),
          str(p_loose.last_filter))
    check("det_filter 放行: 下游裁出 1 张人子图", len(p_loose.last_crops) == 1,
          f"{len(p_loose.last_crops)} 张")
    check("det_filter 放行: last_face_input 与 last_crops 一致",
          len(p_loose.last_face_input) == len(p_loose.last_crops) == 1)
    check("det_filter 放行: 检到人脸（>0）", len(p_loose.last_faces) >= 1,
          f"{len(p_loose.last_faces)} 张")

    p_strict = server._chain_from_spec(_face_detect_chain("long >= 100000"))
    p_strict.step(0, 0.0, dets=[det], frame=frame, force=True)
    check("det_filter 严格: last_filter=(1,0) 放行 0", p_strict.last_filter == (1, 0),
          str(p_strict.last_filter))
    check("det_filter 严格: 无裁剪", p_strict.last_crops == [])
    check("det_filter 严格: 无下游人脸输入", p_strict.last_face_input == [])
    check("det_filter 严格: 无人脸产出", p_strict.last_faces == [])


def driver_checks() -> None:
    """驱动层直调 web_lab.h_pipeline_run 图片分支：装配→pipe.step→_step_out 落盘/时间线。

    只 monkeypatch db.load_pipeline 返回内存 spec（不连 MySQL、不起 HTTP）；YOLO 真实推理。
    验证 WO-0001 事故在真实驱动路径上已修复（人脸链走 pipe.step，det_filter 生效），
    并覆盖落盘命名 face_{n:03d}_{ts}s 与落盘上限语义。
    """
    import re
    import tempfile
    import server

    orig_load = server.db.load_pipeline
    try:
        # 放行：严格阈值内的检测通过 → 裁人 → 检脸 → 落盘（上限=2：原图 1 + 人脸 1）
        with tempfile.TemporaryDirectory(prefix="umvp_face_drv_") as td:
            server.db.load_pipeline = lambda pid: {"name": "PIPE0003-qa",
                                                   "stages": _face_detect_stages("long >= 1")}
            r = server.h_pipeline_run({"pipeline_id": "PIPE0003", "image": "人遮挡1.png",
                                       "output_dir": td, "image_format": "jpg",
                                       "max_saved_images": 2})
            text = r.get("text") or ""
            head = text.splitlines()[0] if text else ""
            check("驱动: 人脸链图片分支运行成功", r.get("ok") is True, str(r.get("error")))
            names = sorted(p.name for p in Path(td).iterdir())
            check("驱动: 人脸落盘命名 face_{n:03d}_{ts}s",
                  [n for n in names if n.startswith("face_")] == ["face_001_0.0s.jpg"], str(names))
            m = re.search(r"检出人脸 (\d+) 张", text)
            n_det = int(m.group(1)) if m else 0
            check("驱动: 上游放行时检出人脸 >= 2", n_det >= 2, f"{n_det} 张")
            check("驱动: 落盘上限=2（原图 1 + 人脸 1，其余跳过）", "落盘 2 张" in text, head)
            stages = {t.get("stage") for t in (r.get("timeline") or []) if t.get("stage")}
            check("驱动: 时间线含 yolo/target_crop/face_detect 阶段",
                  {"yolo", "target_crop", "face_detect"} <= stages, str(sorted(stages)))

        # 严格：det_filter 放行 0 → 无裁剪/无人脸/不落人脸图（WO-0001 回归核心）
        with tempfile.TemporaryDirectory(prefix="umvp_face_drv_") as td:
            server.db.load_pipeline = lambda pid: {"name": "PIPE0003-qa",
                                                   "stages": _face_detect_stages("long >= 100000")}
            r = server.h_pipeline_run({"pipeline_id": "PIPE0003", "image": "人遮挡1.png",
                                       "output_dir": td, "image_format": "jpg"})
            names = sorted(p.name for p in Path(td).iterdir())
            check("驱动: det_filter 严格放行 0 → 不落人脸图",
                  not any(n.startswith("face_") for n in names), str(names))
            tl = " ".join((t.get("text") or "") for t in (r.get("timeline") or []))
            check("驱动: 时间线出现 det_filter 筛选条目（→0）",
                  "有效性筛选" in tl and "→ 0" in tl, tl[-80:])
    finally:
        server.db.load_pipeline = orig_load


def main() -> None:
    # ---- 1. 帧管理门控（frame_skip=3 → 放行 0/3/6/9）----
    fm = FrameManager(sampling="analysis", frame_skip=3)
    passed = [i for i in range(10) if fm.wants_frame(i, i * 0.04)]
    check("帧管理: frame_skip=3 放行 0/3/6/9", passed == [0, 3, 6, 9], str(passed))

    # ---- 2. 链路装配（无 vlm/plate；人脸阶段由服务级装配层注入 face 插槽）----
    yolo = YOLODetector(model_path=str(YOLO_MODEL), conf=0.3, iou=0.6, imgsz=1280,
                        max_det=300, classes=[CLS_PERSON])
    fm1 = FrameManager(sampling="analysis", frame_skip=1)
    pipe = Pipeline(frame=fm1, yolo=yolo)
    check("装配: 人脸截取链无 vlm/plate 依赖", pipe.vlm is None and pipe.plate is None)
    check("装配: 无车牌/驻留阶段不需跟踪", pipe.track_needed() is False)
    check("装配: 未注入 face 时 last_faces/last_face_input 恒为空",
          pipe.last_faces == [] and pipe.last_face_input == [])

    # ---- 3. 逐帧截取（每张图都处理，frame_skip=1）----
    face_det = FaceDetector(det_thresh=0.4)
    tool = Cropper(upscale=1.0, margin=0.15)
    total = 0
    images = sorted(DATA.glob("*.png"))
    check("素材: 测试图存在", len(images) >= 1, f"{len(images)} 张")
    for img_path in images:
        frame = cv2.imread(str(img_path))
        check(f"{img_path.stem}: 帧管理放行", fm1.wants_frame(total, total * 0.04) is True)
        faces = extract_faces(frame, yolo, face_det, tool)
        h, w = frame.shape[:2]
        for f in faces:
            fx1, fy1, fx2, fy2 = f.face_bbox
            check(f"{img_path.stem}#{f.index} 人脸框在图内",
                  0 <= fx1 < fx2 <= w and 0 <= fy1 < fy2 <= h, str(f.face_bbox))
            check(f"{img_path.stem}#{f.index} 出图=框尺寸（原分辨率）",
                  f.face_img.shape[1] == fx2 - fx1 and f.face_img.shape[0] == fy2 - fy1,
                  f"{f.face_img.shape[1]}x{f.face_img.shape[0]}")
        dup = [1 for i in range(len(faces)) for j in range(i + 1, len(faces))
               if iou(faces[i].face_bbox, faces[j].face_bbox) > 0.5]
        check(f"{img_path.stem} 同帧去重无重复", not dup, f"重复对: {len(dup)}")
        total += len(faces)
        print(f"  图片: {img_path.name} | 截取人脸 {len(faces)} 张")

    check("总检出: 全部图合计至少 1 张人脸", total >= 1, f"{total} 张")

    # ---- 4. 服务级装配 + det_filter 生效（进程内，不起 HTTP/MySQL）----
    service_checks()

    # ---- 5. 驱动层（h_pipeline_run 图片分支）落盘/时间线/上限（进程内）----
    driver_checks()

    print(f"\n全部通过: {PASS} 项断言 | 合计截取 {total} 张原分辨率人脸")


if __name__ == "__main__":
    main()
