# -*- coding: utf-8 -*-
"""
管道拼接验证（纯逻辑，不加载模型）

覆盖:
  1. small_only 已弃用（compose 报错，随输出处理删除）
  2. small_crop —— 按识别类型(类别)报送 + 类节拍(seq_filter 节流)
  3. small_full —— 按识别类型送全帧
  4. large_only —— 全局墙钟定时
  5. 每类数量上下限 —— 超出上限/不足下限的类别跳过，其它类照常
  6. 有效性筛选（检测筛选阶段承接）—— 谓词链顺序无关 + DSL 等价 + 成组判定
  7. FrameManager 三种取图节奏 —— analysis / wall_clock / frame_count
  8. IntervalGate 传递门控
  9. 时序门控（seq_filter 整类节流，原 Pipeline 编排迁入）
  10. 端口协议 —— 产物表 / 模块 INPUTS-OUTPUTS 声明 / stage_inputs 连线解析
  11. infer 纯推理 + 检测筛选 + 事件尾（last_events / publish）
  12. 产物全名 / 短名解析（P2-0：resolve_artifact / artifact_at / artifact_short / artifact_type）
  13. 同模块多实例（P2-0 解除单插槽：det_filter 独立生效 + 位置键 inputs + 全名产物不覆盖）
  14. infer 出口 dict 化（WO-0014.3：kind=det 契约形态 + 直传盲区——对象/dict 入口归一、缺 cls_id 跳过）

运行: ai 环境 python pipe/test_composer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipe.composer import (
    ARTIFACT_ANNOTATED,
    ARTIFACT_CROPS,
    ARTIFACT_DET_FILTER_DETS,
    ARTIFACT_DETS,
    ARTIFACT_FACES,
    ARTIFACT_FRAME,
    ARTIFACT_PLATES,
    ARTIFACT_SEQ_FILTER_DETS,
    ARTIFACT_TYPES,
    Detection,
    FrameManager,
    GRAN_CLASS,
    GRAN_SCENE,
    IntervalGate,
    Pipeline,
    PortResolutionError,
    RESOLVED_PORTS,
    SAMPLING_ANALYSIS,
    SAMPLING_FRAME_COUNT,
    SAMPLING_WALL_CLOCK,
    YOLODetector,
    artifact_at,
    artifact_short,
    artifact_type,
    compose,
    resolve_artifact,
)
from pipe.det_filter import DetFilter
from pipe.records import image_of, image_set_of, to_record
from pipe.target_crop import as_crop


PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def det(tid: int, cls: int = 0, score: float = 0.9, model: str = "m1") -> Detection:
    return Detection(track_id=tid, cls_id=cls, bbox=(10, 10, 100, 200), score=score,
                     model_path=model)


# ============ 1. small_only 已弃用 ============
print("== 1. small_only 弃用 ==")
try:
    compose(dict(task_id=1, algo_mode="small_only", class_limits={0: (1, 10)}))
    check("small_only 已弃用（报错）", False)
except ValueError:
    check("small_only 已弃用（报错）", True)

# ============ 2. small_crop：按识别类型报送 + 类节拍 ============
print("== 2. small_crop ==")
p = compose(dict(task_id=2, algo_mode="small_crop", class_limits={0: (1, 10)},
                 check_interval=2.0, extract_frame_rate=3))
# 注意: 帧门控(extract_frame_rate=3) 只分析帧 0,3,6,9...；ts 为墙钟秒
s0, _ = p.step(0, 0.0, [det(1), det(2)])   # t=0: 同类多个目标合并成一条"类"请求
check("同类目标合并为一条类请求", len(s0) == 1 and s0[0].track_key == 0
      and s0[0].gran == GRAN_CLASS and s0[0].ref == "full:cls0"
      and len(s0[0].dets) == 2)
s3, _ = p.step(3, 1.0, [det(1)])           # t=1: 距上次仅1s，不送
check("类节拍: 1s<2s 不送", len(s3) == 0)
s6, _ = p.step(6, 3.0, [det(1)])           # t=3: 距上次3s≥2s 再送
check("类节拍: 3s≥2s 再送", len(s6) == 1 and s6[0].ref == "full:cls0")
# 双类别：各自独立计时、各自一条请求
p2 = compose(dict(task_id=2, algo_mode="small_crop",
                  class_limits={0: (1, 10), 1: (1, 10)}, check_interval=2.0))
s0, _ = p2.step(0, 0.0, [det(1, 0), det(3, 1)])
check("两类各一条请求", len(s0) == 2 and sorted(r.track_key for r in s0) == [0, 1])
_, _ = p2.step(1, 1.0, [det(1, 0), det(3, 1)])   # 1s 内都不再送
s3, _ = p2.step(3, 3.0, [det(3, 1)])             # cls1 距上次3s 再送; cls0 无目标
check("类别独立计时: 仅 cls1 送", len(s3) == 1 and s3[0].track_key == 1)

# ============ 3. small_full：按识别类型送全帧 ============
print("== 3. small_full ==")
p = compose(dict(task_id=3, algo_mode="small_full", class_limits={0: (1, 10)},
                 check_interval=2.0))
s0, _ = p.step(0, 0.0, [det(7)])
check("small_full 送该类全帧", len(s0) == 1 and s0[0].ref == "full:cls0"
      and s0[0].gran == GRAN_CLASS)

# ============ 4. large_only：全局墙钟 ============
print("== 4. large_only ==")
p = compose(dict(task_id=4, algo_mode="large_only", check_interval=5.0))
s0, _ = p.step(0, 0.0, None)
s1, _ = p.step(4, 4.0, None)
check("墙钟: t=0 首送", len(s0) == 1 and s0[0].track_key == -1 and s0[0].gran == GRAN_SCENE)
check("墙钟: t=4 未到5s不送", len(s1) == 0)
s2, _ = p.step(5, 5.0, None)
check("墙钟: t=5 到间隔再送", len(s2) == 1)

# ============ 5. 每类数量上下限 ============
print("== 5. 每类数量上下限 ==")
# 类数量超上限 → 该类跳过，其它类照常
p = compose(dict(task_id=6, algo_mode="small_crop", class_limits={0: (1, 2), 1: (1, 10)},
                 check_interval=1.0, smooth_frames=1, target_actions=["crowd"]))
s0, _ = p.step(0, 0.0, [det(1, 0), det(2, 0), det(3, 0), det(4, 1)])
check("类0超上限(3>2)跳过, 类1照常", len(s0) == 1 and s0[0].track_key == 1)
# 类数量不足下限 → 该类跳过
p2 = compose(dict(task_id=6, algo_mode="small_crop", class_limits={0: (3, 5), 1: (1, 10)},
                  check_interval=1.0, smooth_frames=1, target_actions=["crowd"]))
s0, _ = p2.step(0, 0.0, [det(1, 0), det(2, 0), det(4, 1)])
check("类0不足下限(2<3)跳过, 类1照常", len(s0) == 1 and s0[0].track_key == 1)
# 未配置识别对象 → 不报送
p3 = compose(dict(task_id=8, algo_mode="small_crop", class_limits={},
                  check_interval=1.0, smooth_frames=1))
s0, _ = p3.step(0, 0.0, [det(1)])
check("未配置识别对象不报送", len(s0) == 0)

# ============ 6. 有效性筛选（检测筛选阶段承接原 YOLO 过滤链） ============
print("== 6. 有效性筛选 ==")
# 注：本节走**对象直入 DetFilter** 路径——DetFilter 原元素透传，谓词/断言收到的仍是
# Detection 原对象（属性读成立）；链路直传（Pipeline.step）入口才归一为 dict（见第 21 节）。
pred_a = [lambda d: (not d.model_path or d.model_path == "m1"),
          lambda d: d.score >= 0.8, lambda d: d.cls_id == 0]
pred_b = list(reversed(pred_a))  # 顺序无关：交集语义
dfa = DetFilter(predicates=pred_a, group_limits={0: (1, 10)})
dfb = DetFilter(predicates=pred_b, group_limits={0: (1, 10)})
dets = [det(1, 0, 0.9), det(2, 0, 0.5), det(3, 1, 0.9), det(4, 0, 0.9, model="m2")]
ka = dfa.run(dets)
kb = dfb.run(dets)
check("谓词链不同顺序结果一致", [d.track_id for d in ka] == [d.track_id for d in kb] == [1])
dfd = DetFilter(filter="cls_id == 0, score >= 0.8", predicates=[lambda d: d.model_path in (None, "m1")],
                group_limits={0: (1, 10)})
check("DSL 与谓词等价（AND 叠加）", [d.track_id for d in dfd.run(dets)] == [1])
check("成组判定：空表=全拦（旧 class_limits 语义）", DetFilter(group_limits={}).run(dets) == [])
check("成组判定：None=不判定", [d.track_id for d in DetFilter().run(dets)] == [1, 2, 3, 4])

# ============ 7. FrameManager 三种取图节奏 ============
print("== 7. FrameManager 节奏 ==")
fa = FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=3)
check("analysis: 每3帧取1帧", [fa.wants_frame(i, float(i)) for i in range(6)] == [True, False, False, True, False, False])
fw = FrameManager(sampling=SAMPLING_WALL_CLOCK, interval_sec=5.0)
check("wall_clock: 每5秒取1帧", [fw.wants_frame(0, 0.0), fw.wants_frame(1, 4.0),
      fw.wants_frame(2, 5.0)] == [True, False, True])
ff = FrameManager(sampling=SAMPLING_FRAME_COUNT, interval_frames=10)
check("frame_count: 每10帧取1帧", [ff.wants_frame(0, 0.0), ff.wants_frame(9, 9.0),
      ff.wants_frame(10, 10.0)] == [True, False, True])

# ============ 8. IntervalGate：阶段间传递门控（统一节流机制） ============
print("== 8. IntervalGate 传递门控 ==")
g = IntervalGate(3.0)
check("门控: 首次放行", g.allow("a", 0.0))
g.note("a", 0.0)
check("门控: 间隔内不放行", not g.allow("a", 2.9))
check("门控: 到间隔再放行", g.allow("a", 3.0))
g.note("a", 3.0)
check("门控: key 之间独立", g.allow("b", 3.5))
g_free = IntervalGate(0.0)
check("门控: interval=0 不节流", g_free.allow("k", 0.0) and g_free.allow("k", 0.1))
g_once = IntervalGate(-1.0)
first = g_once.allow("k", 0.0)
g_once.note("k", 0.0)
check("门控: 负值=只放行一次", first and not g_once.allow("k", 100.0))
g_door = IntervalGate(2.0)
g_door.note("t1", 0.0)
seq = []
for t in (0.5, 1.5, 2.0, 3.9):
    if g_door.allow("t1", t):
        g_door.note("t1", t)
        seq.append(t)
check("门控: 序列放行符合间隔", seq == [2.0], str(seq))

# ============ 9. 时序门控（seq_filter 整类节流，原 Pipeline 编排迁入） ============
print("== 9. 整类节流门控 ==")
from pipe.seq_filter import SeqFilter
g = SeqFilter(throttle={"key": "cls_id", "interval": 2.0})
dl = [det(1, cls=0), det(2, cls=0), det(3, cls=2)]
check("节流: 首帧整类放行", len(g.run(dl, ts=0.0)) == 3)
check("节流: 间隔内整类拦截", g.run(dl, ts=1.0) == [])
check("节流: 到间隔再放行", len(g.run(dl, ts=2.0)) == 3)
p0 = compose(dict(task_id=2, algo_mode="small_crop", class_limits={0: (1, 10)},
                  check_interval=0.0, smooth_frames=1, target_actions=["x"]))
check("check_interval=0：不建门控（每帧放行）", p0.seq_filter is None)

# ============ 11. infer 纯推理输出 + 检测筛选阶段过滤（对所有下游生效） ============
print("== 11. infer 纯推理 + det_filter ==")
class _Arr:
    """极简数组替身：支持 tolist() 与逐元素 .item()（绕过 numpy/ultralytics 依赖）。"""
    def __init__(self, v): self._v = v
    def tolist(self): return [list(x) for x in self._v]
    def __getitem__(self, i):
        class _E:
            def __init__(self, x): self._x = x
            def item(self): return self._x
        return _E(self._v[i])
class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _Arr(xyxy); self.cls = _Arr(cls); self.conf = _Arr(conf); self.id = None
class _Res:
    def __init__(self, boxes): self.boxes = boxes
class _Model:
    def __init__(self, res): self._res = res
    def predict(self, frame, **kw): return [self._res]

# 三个框：小目标(10x10)、正常(100x60)、低置信(200x80 @0.3)
yd = YOLODetector(model_path="x.pt")
yd._model = _Model(_Res(_Boxes([[0, 0, 10, 10], [0, 0, 100, 60], [0, 0, 200, 80]],
                               [0, 0, 0], [0.9, 0.9, 0.3])))
d = yd.infer(None)
check("infer: 纯推理输出（不过滤，3 框全出）", len(d) == 3)
_infer_keys = {"kind", "track_id", "cls_id", "bbox", "score", "model_path"}
_dkeys = _infer_keys | {"ts", "frame_idx"}  # 产物记录全契约字段（infer 集 + 时间上下文）
check("infer: 元素是 dict 记录（kind=det 契约形态，键集齐全）",
      all(isinstance(x, dict) and _infer_keys <= set(x) and x["kind"] == "det" for x in d))
check("infer: 不注入时间上下文（ts/frame_idx 由调用方注入，infer 内无帧上下文）",
      all("ts" not in x and "frame_idx" not in x for x in d))
check("infer: bbox 为 (x1,y1,x2,y2) int 元组",
      all(isinstance(x["bbox"], tuple) and len(x["bbox"]) == 4
          and all(isinstance(v, int) for v in x["bbox"]) for x in d))
dfy = DetFilter(filter="score >= 0.5, long >= 50")
check("det_filter: 小目标+低置信视为未识别，只留正常框",
      [x["bbox"] for x in dfy.run(d)] == [(0, 0, 100, 60)], str([x["bbox"] for x in dfy.run(d)]))
# Pipeline 集成：det_filter 对裁剪/标注/报送一致生效
class _Crop:
    def __init__(self): self.n = []
    def run(self, frame, dets, ts): self.n.append(len(dets)); return list(dets)
cp = _Crop()
pi = Pipeline(frame=FrameManager(frame_skip=1),
              yolo=yd, crop=cp, det_filter=DetFilter(filter="score >= 0.5, long >= 50"))
subs, _ = pi.step(0, 0.0, frame=None, force=True)  # frame=None → 不自推理，dets=None
yd2 = YOLODetector(model_path="x.pt"); yd2._model = yd._model
pi2 = Pipeline(frame=FrameManager(frame_skip=1),
                yolo=yd2, crop=cp,
               det_filter=DetFilter(filter="score >= 0.5, long >= 50", events=["筛出"]))
subs2, _ = pi2.step(0, 0.0, frame=object(), force=True)
check("Pipeline: 裁剪拿到有效性筛选后的检测", cp.n[-1] == 1)
check("Pipeline: yolo.dets 产物元素是 dict 记录（step 注入时间上下文后键集齐全）",
      all(isinstance(x, dict) and _dkeys <= set(x) for x in pi2.artifacts.get(ARTIFACT_DETS, [])))
_fa = pi2.artifacts.get(ARTIFACT_DET_FILTER_DETS, [])
check("Pipeline: det_filter.dets 产物元素仍是 dict（筛选只取舍不改形态）",
      len(_fa) == 1 and isinstance(_fa[0], dict) and _dkeys <= set(_fa[0]))
check("Pipeline: last_events 汇总末段记录事件（链路尾通用输出）",
      len(pi2.last_events) == 1 and pi2.last_events[0]["name"] == "筛出")
# Pipeline.publish：外部记录（如 VLM 结果）经事件尾出事件
pp = Pipeline(frame=FrameManager(frame_skip=1),
              event_filter=DetFilter(filter="cls_id == 0",
                                     events=[{"name": "vlm", "detail": ["content"]}]))
evs = pp.publish([{"cls_id": 0, "content": {"r": 1}}, {"cls_id": 2, "content": {}}], ts=1.0)
check("publish：外部记录经尾部出事件",
      len(evs) == 1 and evs[0]["name"] == "vlm" and evs[0]["detail"]["content"] == {"r": 1})

# ============ 12. 产物表 artifacts（P5 第 2 步端口协议） ============
print("== 12. 产物表 artifacts ==")
def _mk_yolo() -> YOLODetector:
    """带桩模型的 YOLODetector（复用 11 节的 _Model/_Res/_Boxes，纯逻辑不加载权重）。"""
    y = YOLODetector(model_path="x.pt")
    y._model = yd._model
    return y

class _Out:
    """最小裁剪记录（非 dataclass → to_record 走 __dict__ + 补 image 别名）。"""
    def __init__(self, img): self.img, self.bbox, self.track_id = img, (0, 0, 4, 4), 0

class _CropStub:
    INPUTS = {"frame": "Image", "dets": "Records"}
    OUTPUTS = {"crops": "Records"}
    def __init__(self): self.outs = []
    def run(self, frame, dets, ts=0.0):
        self.outs = [_Out("IMG%d" % d["track_id"]) for d in (dets or [])]
        return self.outs

class _AnnoStub:
    INPUTS = {"image": "Image", "dets": "Records"}
    OUTPUTS = {"annotated": "Image"}
    def __init__(self): self.ret = object()
    def run(self, image, dets, ts=0.0): return self.ret

class _FaceStub:
    INPUTS = {"frame": "Image", "crops": "Records"}
    OUTPUTS = {"faces": "Records"}
    def __init__(self): self.ret = [{"kind": "face"}]
    def run(self, frame, crops, ts=0.0): return list(self.ret)

class _PlateStub:
    INPUTS = {"frame": "Image", "crops": "Records"}
    OUTPUTS = {"plates": "Records"}
    def __init__(self): self.ret = []
    def run(self, frame, crops, ts=0.0): return list(self.ret)

crop_st, anno_st, face_st, plate_st = _CropStub(), _AnnoStub(), _FaceStub(), _PlateStub()
frame_obj = object()
p3 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=crop_st, annotate=anno_st,
              face=face_st, plate=plate_st,
              det_filter=DetFilter(filter="score >= 0.5, long >= 50"),
              seq_filter=SeqFilter(throttle={"key": "cls_id", "interval": 2.0}))
p3.step(0, 0.0, frame=frame_obj, force=True)
A = p3.artifacts
check("frame_manager.frame 登记本帧原图", A.get(ARTIFACT_FRAME) is frame_obj)
check("yolo.dets 登记纯推理原始检出（过滤前 3 框，dict 键读）",
      [d["bbox"] for d in A.get(ARTIFACT_DETS, [])] == [(0, 0, 10, 10), (0, 0, 100, 60), (0, 0, 200, 80)])
check("产物形态: yolo.dets / det_filter.dets / seq_filter.dets 元素是 dict 记录且键集齐全"
      "（step 注入 ts/frame_idx 后带全契约字段）",
      all(isinstance(x, dict) and _dkeys <= set(x)
          for lst in (A.get(ARTIFACT_DETS, []), A.get(ARTIFACT_DET_FILTER_DETS, []),
                      A.get(ARTIFACT_SEQ_FILTER_DETS, []))
          for x in lst))
check("det_filter.dets 登记放行后的 dets（1 框）", len(A.get(ARTIFACT_DET_FILTER_DETS, [])) == 1)
check("seq_filter.dets 登记门控后的 dets（1 框）", len(A.get(ARTIFACT_SEQ_FILTER_DETS, [])) == 1)
check("target_crop.crops 登记裁剪记录列表", len(A.get(ARTIFACT_CROPS, [])) == 1)
check("target_crop.crops 元素含 image 键（旁路图统一名；img 兼容别名保留）",
      A[ARTIFACT_CROPS][0].get("image") == "IMG0" and A[ARTIFACT_CROPS][0].get("img") == "IMG0")
check("annotate.annotated 登记带框图", A.get(ARTIFACT_ANNOTATED) is anno_st.ret)
check("face_detect.faces 登记人脸记录", A.get(ARTIFACT_FACES) == [{"kind": "face"}])
check("plate_recog.plates 登记车牌记录", A.get(ARTIFACT_PLATES) == [])
check("零回归：last_crops 仍是对象列表（驱动层在用）",
      p3.last_crops and p3.last_crops[0] is crop_st.outs[0])
check("零回归：last_annotated 未变", p3.last_annotated is anno_st.ret)
check("ARTIFACT_TYPES 覆盖全部登记产物",
      set(A) <= set(ARTIFACT_TYPES) and ARTIFACT_TYPES[ARTIFACT_FRAME] == "Image"
      and ARTIFACT_TYPES[ARTIFACT_CROPS] == "Records")
# 阶段缺失 → 产物不登记；未取帧 → 产物表清空
p_na = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo())
p_na.step(0, 0.0, frame=object(), force=True)
check("缺 annotate 阶段 → 无 annotate.annotated 产物", ARTIFACT_ANNOTATED not in p_na.artifacts)
p_skip = Pipeline(frame=FrameManager(frame_skip=3), yolo=_mk_yolo())
p_skip.step(1, 1.0, frame=object())   # 帧 1 非取帧点
check("未取帧时产物表清空（每帧重建）", p_skip.artifacts == {})

# ============ 13. inputs 连线解析 ============
print("== 13. inputs 连线解析 ==")
class _RecvCrop:
    """桩目标裁剪：记录收到的实参，断言连线确实按产物名取上游值。"""
    INPUTS = {"frame": "Image", "dets": "Records"}
    OUTPUTS = {"crops": "Records"}
    def __init__(self): self.got_frame = None; self.got_dets = None
    def run(self, frame, dets, ts=0.0):
        self.got_frame, self.got_dets = frame, dets
        return []

fdef = DetFilter(filter="score >= 0.5, long >= 50")
# 默认（无 stage_inputs）→ 目标裁剪收到 det_filter 过滤后的 1 条
recv_c = _RecvCrop()
pc = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=recv_c, det_filter=fdef)
pc.step(0, 0.0, frame=object(), force=True)
check("默认（无 inputs）：目标裁剪收到过滤后 dets（1 条）", len(recv_c.got_dets) == 1)
check("默认 stage_inputs 为空（零回归）", pc.stage_inputs == {})

# 声明 inputs：target_crop.dets ← yolo.dets（过滤前原始 3 条）→ 模块收到指定产物
recv_i = _RecvCrop()
frame_i = object()
pi = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=recv_i, det_filter=fdef,
              stage_inputs={"target_crop": {"dets": ARTIFACT_DETS}})
pi.step(0, 0.0, frame=frame_i, force=True)
check("声明 inputs：目标裁剪收到 yolo.dets（原始 3 条，绕过过滤）", len(recv_i.got_dets) == 3)
check("声明 inputs：frame 端口仍按现状默认（本帧原图）", recv_i.got_frame is frame_i)
# 同一声明可同时指定多个端口
recv_i2 = _RecvCrop()
pi2 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=recv_i2, det_filter=fdef,
               stage_inputs={"target_crop": {"frame": ARTIFACT_FRAME, "dets": ARTIFACT_DET_FILTER_DETS}})
pi2.step(0, 0.0, frame=frame_i, force=True)
check("声明 inputs：dets 端口指向 det_filter.dets（过滤后 1 条）", len(recv_i2.got_dets) == 1)

# ============ 14. 非法连线 fail fast（可读异常） ============
print("== 14. 非法连线 fail fast ==")
def _run_err(pipe: Pipeline):
    try:
        pipe.step(0, 0.0, frame=object(), force=True)
        return None
    except PortResolutionError as e:
        return str(e)

# ① 未知产物名
e1 = _run_err(Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_RecvCrop(),
                       stage_inputs={"target_crop": {"dets": "nope.dets"}}))
check("未知产物名 → 报错含阶段/端口/产物名/可用清单",
      bool(e1) and "target_crop" in e1 and "'dets'" in e1 and "nope.dets" in e1
      and ARTIFACT_FRAME in e1)
# ② 类型不符：Image 端口喂 Records
e2 = _run_err(Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), annotate=_AnnoStub(),
                       det_filter=fdef, stage_inputs={"annotate": {"image": ARTIFACT_DET_FILTER_DETS}}))
check("类型不符 → 报错含类型与产物名（Image 端口喂 Records）",
      bool(e2) and "类型不符" in e2 and ARTIFACT_DET_FILTER_DETS in e2 and "Image" in e2)
# ③ 产物本帧未登记：seq_filter.dets 存在类型（Records）但链路无 seq_filter 阶段
e3 = _run_err(Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_RecvCrop(),
                       stage_inputs={"target_crop": {"dets": ARTIFACT_SEQ_FILTER_DETS}}))
check("产物本帧未登记 → 报错含产物名与可用清单",
      bool(e3) and "不在本帧产物表" in e3 and ARTIFACT_SEQ_FILTER_DETS in e3)
# ④ 端口未在模块 INPUTS 声明：crop 只声明 dets，却连 frame
class _RecvDetsOnly:
    INPUTS = {"dets": "Records"}
    OUTPUTS = {}
    def run(self, frame, dets, ts=0.0): return []
e4 = _run_err(Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_RecvDetsOnly(),
                       stage_inputs={"target_crop": {"frame": ARTIFACT_FRAME}}))
check("端口未声明 → 报错含模块已声明端口", bool(e4) and "INPUTS" in e4 and "frame" in e4)
# 零回归：非法声明只在被消费时触发；未声明 inputs 的旧链路完全不受影响（见 1-11 节全绿）

# ============ 15. 图集契约：image_set_of / image_of 两读取器语义 ============
print("== 15. 图集契约（records 读取器） ==")
_a, _b = object(), object()
check("image_set_of: 单图 → [图]（长度 1）", image_set_of({"image": _a}) == [_a])
check("image_set_of: 图集 → 原样列表", image_set_of({"image": [_a, _b]}) == [_a, _b])
check("image_set_of: 无旁路图 → []", image_set_of({}) == [])
check("image_set_of: 兼容别名 img 也识别", image_set_of({"img": _a}) == [_a])
check("image_of: 单图语义 → 该图", image_of({"image": _a}) is _a)
check("image_of: 图集 → 首张（单图语义，兼容既有调用）", image_of({"image": [_a, _b]}) is _a)
check("image_of: 无 → None", image_of({}) is None)

# ============ 16. crops 产物记录化（记录 dict 列表，键齐全） ============
print("== 16. crops 产物记录化 ==")
class _CropObj:
    """最小裁剪对象：与 CropResult 同形的几何字段 + to_original/kps_to_original。"""
    def __init__(self, img, bbox, offset, scale, track_id=0, index=0):
        self.img, self.bbox, self.offset = img, tuple(bbox), tuple(offset)
        self.scale, self.scale_x, self.scale_y = float(scale), None, None
        self.cls_id, self.score = 0, 0.9
        self.track_id, self.index = track_id, index
    def _s(self): return self.scale or 1.0
    def to_original(self, bbox):
        x1, y1, x2, y2 = bbox; ox, oy = self.offset; s = self._s()
        return (int(round(x1 / s)) + ox, int(round(y1 / s)) + oy,
                int(round(x2 / s)) + ox, int(round(y2 / s)) + oy)
    def kps_to_original(self, kps):
        ox, oy = self.offset; s = self._s()
        return [((x / s) + ox, (y / s) + oy) for x, y in kps]

class _CropGeom:
    """桩目标裁剪：产带完整几何字段的对象（供 to_record → 记录 dict 列表）。"""
    INPUTS = {"frame": "Image", "dets": "Records"}
    OUTPUTS = {"crops": "Records"}
    def __init__(self): self.outs = []
    def run(self, frame, dets, ts=0.0):
        self.outs = [_CropObj("IMG%d" % d["track_id"], (0, 0, 8, 8), (2, 2), 2.0,
                              track_id=d["track_id"], index=i)
                     for i, d in enumerate(dets or [])]
        return self.outs

_cg = _CropGeom()
_pr = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_cg, det_filter=fdef)
_pr.step(0, 0.0, frame=object(), force=True)
_cr = _pr.artifacts[ARTIFACT_CROPS][0]
_need = {"image", "bbox", "offset", "scale", "scale_x", "scale_y",
         "cls_id", "score", "track_id", "index"}
check("crops 产物元素是记录 dict（非对象）", isinstance(_cr, dict) and len(_pr.artifacts[ARTIFACT_CROPS]) == 1)
check("crops 记录键齐全（image + 几何字段），供下游归一消费", _need <= set(_cr), str(sorted(_cr)))
check("crops 记录 image＝该对象单张子图（旁路键统一）", _cr["image"] == "IMG0" and _cr["img"] == "IMG0")

# as_crop：对象原样 / 记录重建，行为逐字等价
_rec = to_record(_CropObj("IMG", (0, 0, 8, 8), (2, 2), 2.0, track_id=7, index=3))
_orig = _CropObj("IMG", (0, 0, 8, 8), (2, 2), 2.0, track_id=7, index=3)
_norm = as_crop(_rec)
check("as_crop: 裁剪对象原样返回（零拷贝，现路径不变）", as_crop(_orig) is _orig)
check("as_crop: 记录重建后 to_original 等价（同框同参）",
      _norm.to_original((0, 0, 4, 4)) == _orig.to_original((0, 0, 4, 4)))
check("as_crop: 记录重建后 kps_to_original 等价",
      _norm.kps_to_original([(2, 2), (4, 4)]) == _orig.kps_to_original([(2, 2), (4, 4)]))
check("as_crop: 重建保留 track_id/bbox/offset/scale",
      _norm.track_id == 7 and tuple(_norm.bbox) == (0, 0, 8, 8)
      and tuple(_norm.offset) == (2, 2) and _norm.scale == 2.0)

# ============ 17. face/plate 的 crops 端口纳入 inputs 解析（默认零回归） ============
print("== 17. face/plate crops 端口 inputs ==")
class _SpyStage:
    """桩识别阶段：记录收到的 crops，结果仅由归一后的几何决定（对象/记录应一致）。"""
    INPUTS = {"frame": "Image", "crops": "Records"}
    OUTPUTS = {"faces": "Records"}
    def __init__(self): self.got = None; self.ret = None; self.frame_in = None
    def run(self, frame, crops, ts=0.0):
        self.got = list(crops or []); self.frame_in = frame
        self.ret = [list(as_crop(c).to_original((0, 0, 4, 4))) for c in self.got]
        return self.ret

class _PlateRes:
    """最小车牌结果（供 composer 车牌事件尾读字段）。"""
    def __init__(self, geo):
        self.plate_no = "P%s" % (geo,); self.score = 0.9
        self.type_name = "t"; self.color = "c"; self.bbox = (0, 0, 1, 1)

class _SpyPlate(_SpyStage):
    """桩车牌阶段：composer 会读结果的 plate_no 等字段，故返回对象（几何仍归一）。"""
    OUTPUTS = {"plates": "Records"}
    def run(self, frame, crops, ts=0.0):
        self.got = list(crops or []); self.frame_in = frame
        self.ret = [list(as_crop(c).to_original((0, 0, 4, 4))) for c in self.got]
        return [_PlateRes(g) for g in self.ret]

_face_def, _face_in = _SpyStage(), _SpyStage()
_pd = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_CropGeom(),
               face=_face_def, det_filter=fdef)
_frame_d = object()
_pd.step(0, 0.0, frame=_frame_d, force=True)
check("默认：face 收到对象列表（last_crops，零回归）",
      _face_def.got and not isinstance(_face_def.got[0], dict))
check("默认：face frame 端口＝本帧原图", _face_def.frame_in is _frame_d)
check("默认：last_face_input 与 last_crops 一致（现状不变）",
      _face_def.got == _pd.last_face_input and len(_pd.last_face_input) == len(_pd.last_crops))

_pd2 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_CropGeom(),
                face=_face_in, det_filter=fdef,
                stage_inputs={"face_detect": {"crops": ARTIFACT_CROPS}})
_pd2.step(0, 0.0, frame=object(), force=True)
check("声明 inputs：face 收到记录 dict 列表（来自 artifacts）",
      _face_in.got and isinstance(_face_in.got[0], dict))
check("声明 inputs：face 结果与传对象时一致（同帧同参）", _face_in.ret == _face_def.ret)
check("声明 inputs：last_face_input 记录实际收到的 crops（记录列表）",
      _face_in.got == _pd2.last_face_input)

_plate_def, _plate_in = _SpyPlate(), _SpyPlate()
_pd3 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_CropGeom(),
                plate=_plate_def, det_filter=fdef)
_pd3.step(0, 0.0, frame=object(), force=True)
check("默认：plate 收到对象列表（last_crops，零回归）",
      _plate_def.got and not isinstance(_plate_def.got[0], dict))
_pd4 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_CropGeom(),
                plate=_plate_in, det_filter=fdef,
                stage_inputs={"plate_recog": {"crops": ARTIFACT_CROPS}})
_pd4.step(0, 0.0, frame=object(), force=True)
check("声明 inputs：plate 收到记录 dict 列表", _plate_in.got and isinstance(_plate_in.got[0], dict))
check("声明 inputs：plate 结果与传对象时一致（同帧同参）", _plate_in.ret == _plate_def.ret)

# 未声明 crops 端口时，plate/face 的 frame 端口仍按现状默认
_pd5 = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(), crop=_CropGeom(),
                plate=_SpyPlate(), det_filter=fdef,
                stage_inputs={"plate_recog": {"crops": ARTIFACT_CROPS}})
_f5 = object()
_pd5.step(0, 0.0, frame=_f5, force=True)
check("声明 crops 不动 frame 端口（仍默认本帧原图）", _pd5.plate.frame_in is _f5)

# ============ 18. RESOLVED_PORTS 清单（web 段开放连线 / 校验的唯一来源） ============
print("== 18. RESOLVED_PORTS ==")
check("RESOLVED_PORTS：plate_recog/face_detect 的 frame+crops 已纳入解析",
      RESOLVED_PORTS.get("plate_recog") == ("frame", "crops")
      and RESOLVED_PORTS.get("face_detect") == ("frame", "crops"))
check("RESOLVED_PORTS：既有端口保持（yolo/target_crop/annotate）",
      RESOLVED_PORTS.get("yolo") == ("image",) and RESOLVED_PORTS.get("target_crop") == ("frame", "dets")
      and RESOLVED_PORTS.get("annotate") == ("image", "dets"))

# ============ 19. 产物全名 / 短名解析（P2-0 节点实例身份，设计稿 §3.1/§3.4） ============
print("== 19. 产物全名 / 短名解析 ==")
_pos = {"yolo": [1], "det_filter": [2, 5], "target_crop": [3]}
check("resolve_artifact: 短名 + 单实例 → 全名",
      resolve_artifact("yolo.dets", _pos) == "yolo#1.dets")
check("resolve_artifact: 已是全名 → 原样返回",
      resolve_artifact("target_crop#3.crops", _pos) == "target_crop#3.crops")
check("resolve_artifact: 多实例写全名可解析",
      resolve_artifact("det_filter#5.dets", _pos) == "det_filter#5.dets")


def _ra_err(name: str):
    try:
        resolve_artifact(name, _pos)
        return None
    except PortResolutionError as e:
        return str(e)


_e_amb = _ra_err("det_filter.dets")
check("resolve_artifact: 多实例短名歧义 → 报错（要求写全名，不猜）",
      bool(_e_amb) and "歧义" in _e_amb and "det_filter" in _e_amb and "全名" in _e_amb, str(_e_amb))
_e_unk = _ra_err("nope.dets")
check("resolve_artifact: 未知模块 → 报错", bool(_e_unk) and "nope" in _e_unk, str(_e_unk))
_e_emp = _ra_err("   ")
check("resolve_artifact: 空名 → 报错", bool(_e_emp), str(_e_emp))
_e_pos = _ra_err("yolo#9.dets")
check("resolve_artifact: 全名位置不在本链路 → 报错",
      bool(_e_pos) and "yolo#9.dets" in _e_pos, str(_e_pos))
# ARTIFACT_* 常量体系的实例化 + 短名/全名互转 + 产物名 → 类型（唯一解析实现）
check("artifact_at: 常量 + position → 全名", artifact_at(ARTIFACT_CROPS, 3) == "target_crop#3.crops")
check("artifact_at: 全名幂等（替换位置）", artifact_at("target_crop#3.crops", 7) == "target_crop#7.crops")
check("artifact_short: 全名 → 短名", artifact_short("det_filter#5.dets") == "det_filter.dets")
check("artifact_short: 短名幂等", artifact_short("det_filter.dets") == "det_filter.dets")
check("artifact_type: 短名 → 类型",
      artifact_type(ARTIFACT_DETS) == "Records" and artifact_type(ARTIFACT_FRAME) == "Image")
check("artifact_type: 全名 → 类型（剥位置）",
      artifact_type("target_crop#3.crops") == "Records"
      and artifact_type("annotate#4.annotated") == "Image")
try:
    artifact_type("nope.dets")
    _e_at = None
except PortResolutionError as e:
    _e_at = str(e)
check("artifact_type: 未知产物 → 报错", bool(_e_at) and "nope.dets" in _e_at, str(_e_at))

# ============ 20. 同模块多实例（P2-0 解除单插槽） ============
print("== 20. 同模块多实例 det_filter ==")


class _DFSpy:
    """包装 DetFilter，记录收到的 dets（断言位置键 inputs 各自生效）。"""
    INPUTS = {"dets": "Records"}
    OUTPUTS = {"dets": "Records"}

    def __init__(self, inner):
        self.inner = inner
        self.got = None

    def run(self, dets, ts=0.0, frame_idx=0):
        self.got = list(dets)
        return self.inner.run(self.got, ts=ts, frame_idx=frame_idx)


# 两个 det_filter（位置 2 / 5）各自 filter 独立生效：都取 yolo#1.dets（原始 3 条）
_a_spy = _DFSpy(DetFilter(filter="score >= 0.5"))
_b_spy = _DFSpy(DetFilter(filter="long >= 50"))
_pma = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(),
                det_filters=[(2, _a_spy), (5, _b_spy)], stage_positions={"yolo": 1},
                stage_inputs={"#2": {"dets": "yolo#1.dets"},
                              "#5": {"dets": "yolo#1.dets"}})
_pma.step(0, 0.0, frame=object(), force=True)
check("多实例: 实例 a 收到 yolo#1.dets 原始 3 条（位置键）",
      _a_spy.got is not None and len(_a_spy.got) == 3)
check("多实例: 实例 b 收到 yolo#1.dets 原始 3 条（各自独立）",
      _b_spy.got is not None and len(_b_spy.got) == 3)
check("多实例: #2 filter(score>=0.5) → 2 条（b1,b2）",
      [d["bbox"] for d in _pma.artifacts.get("det_filter#2.dets", [])]
      == [(0, 0, 10, 10), (0, 0, 100, 60)])
check("多实例: #5 filter(long>=50) → 2 条（b2,b3，与 #2 不同）",
      [d["bbox"] for d in _pma.artifacts.get("det_filter#5.dets", [])]
      == [(0, 0, 100, 60), (0, 0, 200, 80)])
check("多实例: det_filter#N.dets 产物元素是 dict（键集含全契约字段）",
      all(isinstance(d, dict) and _dkeys <= set(d)
          for lst in (_pma.artifacts.get("det_filter#2.dets", []),
                      _pma.artifacts.get("det_filter#5.dets", []))
          for d in lst))
check("多实例: 短名 det_filter.dets 不登记（避免歧义覆盖）",
      ARTIFACT_DET_FILTER_DETS not in _pma.artifacts)
check("多实例: yolo#1.dets 全名登记（过滤器上游）", len(_pma.artifacts.get("yolo#1.dets", [])) == 3)

# 位置键 inputs 接不同上游互不覆盖：b 取 a 的输出 det_filter#2.dets（升序执行保证已登记）
_a2 = _DFSpy(DetFilter(filter="score >= 0.5"))
_b2 = _DFSpy(DetFilter(filter="long >= 50"))
_pmb = Pipeline(frame=FrameManager(frame_skip=1), yolo=_mk_yolo(),
                det_filters=[(2, _a2), (5, _b2)], stage_positions={"yolo": 1},
                stage_inputs={"#2": {"dets": "yolo#1.dets"},
                              "#5": {"dets": "det_filter#2.dets"}})
_pmb.step(0, 0.0, frame=object(), force=True)
check("多实例: 实例 b 的 inputs.dets 指向实例 a 的输出（2 条，位置键互不覆盖）",
      _b2.got is not None and len(_b2.got) == 2)
check("多实例: #2 放行 2 条 / #5 在 #2 基础上再筛 → 1 条（b2）",
      [d["bbox"] for d in _pmb.artifacts.get("det_filter#2.dets", [])] == [(0, 0, 10, 10), (0, 0, 100, 60)]
      and [d["bbox"] for d in _pmb.artifacts.get("det_filter#5.dets", [])] == [(0, 0, 100, 60)])
check("多实例: 按位置升序执行，last_filter=首次进入 3 / 最终放行 1",
      _pmb.last_filter == (3, 1), str(_pmb.last_filter))

# ============ 21. infer 出口 dict 化 + 直传盲区（WO-0014.3 批次2） ============
print("== 21. infer 出口 dict 化 + 直传盲区 ==")
# 盲区1：step(dets=[dict,...]) 直传 dict——入口归一为副本 + 补时间上下文，submits 照常分组
pd = compose(dict(task_id=21, algo_mode="small_crop", class_limits={0: (1, 10)},
                  check_interval=0.0))
dd = [{"kind": "det", "track_id": 1, "cls_id": 0, "bbox": (10, 10, 100, 200),
       "score": 0.9, "model_path": "m1"},
      {"kind": "det", "track_id": 2, "cls_id": 0, "bbox": (20, 20, 110, 210),
       "score": 0.8, "model_path": "m1"}]
sd, _ = pd.step(0, 0.0, dets=dd, force=True)
check("盲区: step 直传 dict → submits 照常产生（按类分组）",
      len(sd) == 1 and sd[0].track_key == 0 and sd[0].gran == GRAN_CLASS)
check("盲区: 直传 dict 合并进该类请求（len(s.dets)=2）", len(sd[0].dets) == 2)
check("盲区: 直传 dict 归一为副本并补 ts/frame_idx（不改调用方原 dict）",
      "ts" not in dd[0] and "frame_idx" not in dd[0]
      and sd[0].dets[0] is not dd[0]
      and sd[0].dets[0]["ts"] == 0.0 and sd[0].dets[0]["frame_idx"] == 0)
# 盲区2：直传对象（Detection）→ 入口 to_record 归一为 dict（字段齐全 + 补 kind/时间上下文）
po = compose(dict(task_id=21, algo_mode="small_crop", class_limits={0: (1, 10)},
                  check_interval=0.0))
so, _ = po.step(0, 0.0, dets=[det(9)], force=True)
check("盲区: 直传 Detection 对象 → 归一为 dict 且全字段 + kind + 时间上下文",
      len(so) == 1 and len(so[0].dets) == 1
      and isinstance(so[0].dets[0], dict)
      and so[0].dets[0]["track_id"] == 9 and so[0].dets[0]["cls_id"] == 0
      and so[0].dets[0]["bbox"] == (10, 10, 100, 200) and so[0].dets[0]["score"] == 0.9
      and so[0].dets[0]["model_path"] == "m1" and so[0].dets[0]["kind"] == "det"
      and so[0].dets[0]["ts"] == 0.0 and so[0].dets[0]["frame_idx"] == 0)
# 盲区3：_submits 对缺 cls_id / cls_id=-1 的 dict 项跳过（与 web _as_detections 语义对齐）
p3x = compose(dict(task_id=22, algo_mode="small_crop", class_limits={0: (1, 10)},
                   check_interval=0.0))
subs_x = p3x._submits([
    {"kind": "det", "cls_id": 0, "bbox": (0, 0, 1, 1), "score": 0.9},
    {"kind": "det", "bbox": (0, 0, 1, 1), "score": 0.8},                # 缺 cls_id → 跳过
    {"kind": "det", "cls_id": -1, "bbox": (0, 0, 1, 1), "score": 0.7},  # -1 → 跳过
    {"kind": "det", "cls_id": 1, "bbox": (0, 0, 1, 1), "score": 0.6},
])
check("盲区: _submits 跳过缺 cls_id/-1 的 dict 项（其余照常分组）",
      len(subs_x) == 2 and [s.track_key for s in subs_x] == [0, 1]
      and all(len(s.dets) == 1 for s in subs_x))

print(f"\n全部通过: {PASS} 项断言")
