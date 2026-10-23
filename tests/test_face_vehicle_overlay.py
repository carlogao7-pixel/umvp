# -*- coding: utf-8 -*-
"""
人脸+车辆同图画框 演示链路测试（WO-0020，端口连线 inputs 实战样例）

spec: tests/data/specs/face_vehicle_overlay.json
链路: 帧管理 → YOLO(classes=0,2 人+车) → 目标裁剪(人) → 人脸检测
      → 标注A(画人脸框, dets←face_detect#3.faces)
      → 标注B(叠画车辆框, image←annotate#4.annotated 接力 + dets←yolo#1.dets)
      → det_filter(空=事件尾, 终段白名单)

覆盖:
  1. spec 过 validate_chain（进程内 validate() 无 error + CLI 子进程 exit=0）。
  2. 装配: stage_inputs 位置键 + 产物全名（#4/#5）；annotate 两实例按位置装配。
  3. 连线解析（纯逻辑 stub + 间谍）: 标注A 实参收到人脸记录（faces 产物）；
     标注B 实参收到 yolo.dets 原始检出 + 底图**是**标注A 产物（接力生效）。
  4. 端到端（真实 YOLO + 真实 SCRFD, 素材 人遮挡1.png: 人=10 车=4 脸=3）:
     faces 连线可解析（不抛 PortResolutionError——composer step 插槽顺序
     annotate 移至 face_detect 之后的回归）、dets/faces 两类记录非空、
     标注A/B 产物图尺寸=原帧、末段产物=标注B、标注图确有画框（像素差异）。

已知缺口（记录于回报，不在本测试修）: face_detect.faces 产物为 FaceResult
对象且无 bbox 字段（仅 face_bbox/person_bbox），annotate 只读 bbox → 人脸框
当前画不出（静默跳过），标注A 产物与原帧像素相同属预期。

运行: conda run -n ai python tests/test_face_vehicle_overlay.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402
from pipe.annotate import Annotator  # noqa: E402
from pipe.composer import PortResolutionError  # noqa: E402
from validate_chain import validate  # noqa: E402  (web_lab 在 sys.path)

SPEC_PATH = ROOT / "tests/data/specs/face_vehicle_overlay.json"
IMG_PATH = ROOT / "tests/data/pic/人遮挡1.png"
YOLO_MODEL = ROOT / "models/yolov8n.pt"

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


class _SpyAnno:
    """标注间谍：记录 run 实参并转发真实例（INPUTS/OUTPUTS 与 Annotator 一致）。"""

    INPUTS = {"image": "Image", "dets": "Records"}
    OUTPUTS = {"annotated": "Image"}

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def run(self, image, dets, ts=0.0):
        out = self.inner.run(image, dets, ts)
        self.calls.append({"image": image, "dets": list(dets or []), "out": out})
        return out


# ==================== 1. spec 校验 ====================

def validate_checks() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    errs: list = []
    n_err = validate(spec, warnings=errs)
    check("校验: 进程内 validate 无 error", n_err == [], str(n_err))
    r = subprocess.run([sys.executable, str(ROOT / "web_lab/validate_chain.py"),
                        str(SPEC_PATH)], capture_output=True, text=True, timeout=180)
    check("校验: CLI exit=0 且含 [OK]",
          r.returncode == 0 and "[OK]" in r.stdout, (r.stdout + r.stderr).strip()[-80:])
    ids = [s["module_id"] for s in spec["stages"]]
    check("校验: 链路形态 fm→yolo→crop→face→标注A→标注B→det_filter 终段",
          ids == ["frame_manager", "yolo", "target_crop", "face_detect",
                  "annotate", "annotate", "det_filter"], str(ids))
    a, b = spec["stages"][4], spec["stages"][5]
    check("校验: 标注A 连线 dets←face_detect#3.faces（全名）",
          a.get("inputs") == {"dets": "face_detect#3.faces"}, str(a.get("inputs")))
    check("校验: 标注B 连线 image←annotate#4.annotated 接力 + dets←yolo#1.dets",
          b.get("inputs") == {"image": "annotate#4.annotated", "dets": "yolo#1.dets"},
          str(b.get("inputs")))
    return spec


# ==================== 2. 装配断言 ====================

def assemble_checks(spec: dict):
    pipe = server._chain_from_spec(spec)
    check("装配: stage_inputs 位置键 + 产物全名",
          pipe.stage_inputs == {"#4": {"dets": "face_detect#3.faces"},
                                "#5": {"image": "annotate#4.annotated",
                                       "dets": "yolo#1.dets"}},
          str(pipe.stage_inputs))
    check("装配: annotate 两实例按位置装配 [4, 5]",
          [p for p, _ in pipe.annotators] == [4, 5],
          str([p for p, _ in pipe.annotators]))
    check("装配: face_detect 阶段已注入", len(pipe.face_stages) == 1)
    return pipe


# ==================== 3. 连线解析（纯逻辑 stub + 间谍） ====================

class _YoloStub:
    INPUTS = {"image": "Image"}
    OUTPUTS = {"dets": "Records"}

    def run_stub_dets(self):
        return [{"kind": "det", "track_id": 1, "cls_id": 0, "bbox": (10, 10, 60, 160),
                 "score": 0.9, "ts": 0.0, "frame_idx": 0},
                {"kind": "det", "track_id": 2, "cls_id": 2, "bbox": (200, 100, 400, 260),
                 "score": 0.8, "ts": 0.0, "frame_idx": 0}]


class _CropStub:
    INPUTS = {"frame": "Image", "dets": "Records"}
    OUTPUTS = {"crops": "Records"}

    def run(self, frame, dets, ts=0.0):
        return [{"kind": "crop", "bbox": d["bbox"], "image": f"IMG{d['track_id']}",
                 "track_id": d["track_id"]} for d in (dets or [])]


class _FaceStub:
    """人脸桩：产出带 face_bbox 的人脸记录（模拟 FaceResult 产物形态）。"""
    INPUTS = {"frame": "Image", "crops": "Records"}
    OUTPUTS = {"faces": "Records"}

    def run(self, frame, crops, ts=0.0):
        return [{"kind": "face", "face_bbox": (20, 20, 50, 50),
                 "person_bbox": (10, 10, 60, 160), "score": 0.95}]


def wire_checks(spec: dict) -> None:
    import numpy as np
    pipe = server._chain_from_spec(spec)
    frame = np.zeros((64, 80, 3), dtype=np.uint8)  # 合成小图（spy 转发真实 Annotator 需 copy()）
    yolo = _YoloStub()

    class _YoloHost:  # yolo 单插槽：直接以对象承载 infer 所需接口
        pass

    # stub yolo：Pipeline.step 里 yolo 只在 dets=None 时自推理；此处直接传 dets，
    # 故 yolo 实例本身可保持真实（不触发推理）。spy 替换 annotate 两个实例：
    spy_a, spy_b = _SpyAnno(Annotator(thickness=2)), _SpyAnno(Annotator(thickness=2))
    pipe.annotate = None
    pipe.annotators = [(4, spy_a), (5, spy_b)]
    pipe.crop = _CropStub()
    pipe.face_stages = [(3, _FaceStub())]
    pipe.face = pipe.face_stages[0][1]
    # 手动喂 yolo.dets：绕过推理（不加载权重），直接走 step 的产物登记路径
    dets = yolo.run_stub_dets()
    pipe.step(0, 0.0, dets=dets, frame=frame, force=True)

    check("连线: 标注A 被调用一次", len(spy_a.calls) == 1, str(len(spy_a.calls)))
    check("连线: 标注B 被调用一次", len(spy_b.calls) == 1, str(len(spy_b.calls)))
    ca, cb = spy_a.calls[0], spy_b.calls[0]
    check("连线: 标注A dets 实参 = face_detect.faces 人脸记录（含 face_bbox）",
          len(ca["dets"]) == 1 and ca["dets"][0].get("kind") == "face"
          and "face_bbox" in ca["dets"][0], str(ca["dets"])[:80])
    check("连线: 标注A image 实参 = 本帧原图（默认）", ca["image"] is frame)
    check("连线: 标注B dets 实参 = yolo.dets 原始检出（人+车 2 条，绕过筛选）",
          [d["cls_id"] for d in cb["dets"]] == [0, 2], str(cb["dets"])[:80])
    check("连线: 标注B image 实参 = 标注A 产物（接力底图，对象身份一致）",
          cb["image"] is ca["out"])
    check("连线: 末段产物 = 标注B 输出（last_annotated）",
          pipe.last_annotated is cb["out"])
    check("连线: 产物表登记 annotate#4/#5 两份 annotated",
          "annotate#4.annotated" in pipe.artifacts
          and "annotate#5.annotated" in pipe.artifacts, str(sorted(pipe.artifacts)))


# ==================== 4. 端到端（真实 YOLO + 真实 SCRFD） ====================

def e2e_checks(spec: dict) -> None:
    frame = cv2.imread(str(IMG_PATH))
    check("端到端: 素材可读", frame is not None, str(IMG_PATH.name))
    h, w = frame.shape[:2]
    pipe = server._chain_from_spec(spec)

    # faces 连线在执行顺序上可解析（annotate 已移至 face_detect 之后）：
    # 旧顺序下此 step 抛 PortResolutionError（faces 未登记）。
    try:
        pipe.step(0, 0.0, frame=frame, force=True)
    except PortResolutionError as e:
        check("端到端: faces 连线可解析（不抛 PortResolutionError）", False, str(e))
        return
    check("端到端: faces 连线可解析（不抛 PortResolutionError）", True)

    dets = pipe.artifacts.get("yolo#1.dets", [])
    faces = pipe.artifacts.get("face_detect#3.faces", [])
    persons = [d for d in dets if d.get("cls_id") == 0]
    cars = [d for d in dets if d.get("cls_id") == 2]
    check("端到端: YOLO 检出 人>0 且 车>0",
          len(persons) > 0 and len(cars) > 0,
          f"人={len(persons)} 车={len(cars)}")
    check("端到端: 人脸记录非空（SCRFD 检出）", len(faces) > 0,
          f"脸={len(faces)}")

    img_a = pipe.artifacts.get("annotate#4.annotated")
    img_b = pipe.artifacts.get("annotate#5.annotated")
    check("端到端: 标注A/B 产物图存在", img_a is not None and img_b is not None)
    check("端到端: 标注A 产物尺寸=原帧",
          img_a.shape[:2] == (h, w), f"{img_a.shape[:2]} vs {(h, w)}")
    check("端到端: 标注B 产物尺寸=原帧",
          img_b.shape[:2] == (h, w), f"{img_b.shape[:2]} vs {(h, w)}")
    check("端到端: 末段产物=标注B（last_annotated is annotate#5）",
          pipe.last_annotated is img_b)
    check("端到端: 标注B 确有画框（与原帧存在像素差异；车框来自 yolo.dets）",
          abs(int(cv2.sumElems(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))[0]
                  - cv2.sumElems(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0])) > 0)
    # 标注A 只收人脸记录且人脸框当前画不出（无 bbox 字段，见文件头缺口说明）→
    # 其产物与原帧像素相同属预期；此处断言该行为已被理解而非意外。
    diff_a = abs(int(cv2.sumElems(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY))[0]
                     - cv2.sumElems(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]))
    check("端到端: 标注A 与原帧像素相同（缺口预期：faces 无 bbox 画不出，待模块修复）",
          diff_a == 0, f"diff={diff_a}")

    # 证据落盘（供人工查验框效果；临时目录，测试结束保留打印路径）
    out_dir = Path(tempfile.gettempdir()) / "umvp_overlay_evidence"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "annotateA_faces.png"), img_a)
    cv2.imwrite(str(out_dir / "annotateB_vehicle.png"), img_b)
    cv2.imwrite(str(out_dir / "origin.png"), frame)
    print(f"  [证据] 产物图已存 {out_dir}/（origin/annotateA_faces/annotateB_vehicle.png）")


def main() -> None:
    print("== 1. spec 校验（validate + CLI） ==")
    spec = validate_checks()
    print("== 2. 装配断言（stage_inputs / 多实例位置） ==")
    assemble_checks(spec)
    print("== 3. 连线解析（纯逻辑 stub + 间谍） ==")
    wire_checks(spec)
    print("== 4. 端到端（真实 YOLO + SCRFD, 素材=人遮挡1.png） ==")
    e2e_checks(spec)
    print(f"\n全部通过: {PASS} 项断言")


if __name__ == "__main__":
    main()
