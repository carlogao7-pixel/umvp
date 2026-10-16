# -*- coding: utf-8 -*-
"""
模块化管道拼接器（重构版）：模块 + 装配器

按"参数归属模块"重构:
  1. FrameManager  帧管理 —— 取图节奏(抽帧 / 墙钟秒 / 帧数) + 背压
  2. YOLODetector  小模型识别 —— **纯推理**：ultralytics 参数可调(conf/iou/imgsz/
                     max_det/classes/device)，infer() 输出 Detection。有效性过滤/成组判定
                     归「检测筛选 DetFilter」（Pipeline.det_filter）；时序门控归「时序判定
                     SeqFilter」（Pipeline.seq_filter / event_filter）
  3. VLMAnalyzer   大模型研判 —— 素材准备(缩放/prompt) + 推理
  （原「输出处理 OutputPolicy」已弃用删除：链路输出 = 事件尾产出的事件（记录携带 events，
    见 pipe.records）；有效性/时序/冷却/同对象一次由 DetFilter/SeqFilter 承担。）

装配器 compose(): 把分析模式（small_crop/small_full/large_only）映射为模块组合与参数。

YOLO 输出粒度（2026-10 编排上移到 Pipeline）:
  class —— 按识别类型（类别）单独报送：画面里每出现一类识别对象（且数量在
           "该类上下限"范围内——由 DetFilter.group_limits 判定）就发一条请求，
           一类多个目标合并成一条（ref=full:cls{类别} 送全帧、拼接层画该类框标注）。
  scene —— 仅 large_only 的周期性整帧（ref=full:global），不经过小模型。

依赖: 纯 Python（抽帧+背压已在 FrameManager 内部，不再依赖 face_scan）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from pipe.det_filter import DetFilter
from pipe.records import collect_events
from pipe.seq_filter import SeqFilter

_NEG_INF = -1e18  # 首见必送：模拟 spaz 以 unix 时间戳减 0 的语义

# ---------------- 通用数据 ----------------
@dataclass
class Detection:
    """一次 YOLO 检测（含跟踪 id）——与 spaz producer 对齐。"""

    track_id: int
    cls_id: int
    bbox: Tuple[int, int, int, int]
    score: float
    model_path: Optional[str] = None


@dataclass
class SubmitRequest:
    """一次报送请求（检测 → VLM）。"""

    track_key: int  # 按识别类型=类别编号; large_only 整帧=-1
    gran: str  # "class"（按识别类型） | "scene"（周期性整帧）
    ref: str  # "full:cls{类别}" / "full:global"
    det: Optional[Detection] = None  # 保留字段（单目标素材用）；按类报送时为空
    dets: List[Detection] = field(default_factory=list)  # 该类全部有效检测（large_only 为空）


# ---------------- 模块 1: 帧管理 ----------------
SAMPLING_ANALYSIS = "analysis"  # 随抽帧节奏（extract_frame_rate，含背压）
SAMPLING_WALL_CLOCK = "wall_clock"  # 按墙钟秒（large_only: check_interval）
SAMPLING_FRAME_COUNT = "frame_count"  # 按帧数（temporal: check_interval × fps）

# 背压档位 → FrameManager 细粒度参数（产品表单只暴露档位；细节转内部常量）
BACKPRESSURE_LEVELS = {
    "off": {"queue_threshold": 10 ** 9, "backpressure_multiplier": 1,
            "adaptive_relax_ratio": 10 ** 9, "adaptive_recover_ratio": 0.5},
    "standard": {"queue_threshold": 32, "backpressure_multiplier": 2,
                 "adaptive_relax_ratio": 1.2, "adaptive_recover_ratio": 0.5},
    "aggressive": {"queue_threshold": 16, "backpressure_multiplier": 4,
                   "adaptive_relax_ratio": 1.5, "adaptive_recover_ratio": 0.3},
}


def backpressure_params(level: str) -> dict:
    """档位 → FrameManager 背压参数（未知档位回退 standard）。"""
    return dict(BACKPRESSURE_LEVELS.get(str(level or "standard"), BACKPRESSURE_LEVELS["standard"]))


@dataclass
class FrameManager:
    """取图节奏决策（2026-08-18 并入原 face_scan.frame_scheduler 的抽帧+背压）。

    sampling 决定"按什么取图"，与"为什么取图"解耦：
      analysis    —— 随抽帧节奏：每 frame_skip 帧取一张；积压(队列深度)或推理耗时
                      超过预算时自动放宽抽帧间隔（背压），消化后恢复基础间隔
      wall_clock  —— 按墙钟秒：每隔 interval_sec 秒取一张
      frame_count —— 按帧数：每隔 interval_frames 帧取一张
    """

    sampling: str = SAMPLING_ANALYSIS
    frame_skip: int = 3  # analysis: 基础抽帧间隔（= extract_frame_rate）
    interval_sec: float = 3.0  # wall_clock: 秒
    interval_frames: int = 30  # frame_count: 帧数
    # ---- 背压（原 FrameScheduler 参数，仅 analysis 模式生效） ----
    queue_threshold: int = 32  # 队列模式：积压深度超过此值触发放宽
    backpressure_multiplier: int = 2  # 放宽时跳帧间隔放大倍数
    max_frame_skip: Optional[int] = None  # 放宽上限，默认 base * multiplier * 4
    adaptive_relax_ratio: float = 1.2  # 自适应：耗时 > 预算 * 此值 -> 放宽
    adaptive_recover_ratio: float = 0.5  # 自适应：耗时 < 预算 * 此值 -> 恢复

    def __post_init__(self) -> None:
        self.base_skip = max(1, int(self.frame_skip))
        self.max_skip = self.max_frame_skip or self.base_skip * self.backpressure_multiplier * 4
        self._skip = self.base_skip
        self._relaxed = False
        self._relax_events = 0
        self._ema_cost = 0.0
        self._last_wall = _NEG_INF
        self._last_frame = _NEG_INF

    # ---------------- 查询状态（analysis 背压） ----------------
    @property
    def current_skip(self) -> int:
        """当前生效的抽帧间隔（积压放宽时会大于 base_skip）。"""
        return self._skip

    @property
    def relaxed(self) -> bool:
        """是否正处于放宽状态。"""
        return self._relaxed

    @property
    def relax_events(self) -> int:
        """累计触发放宽的次数。"""
        return self._relax_events

    # ---------------- 帧决策 ----------------
    def wants_frame(self, frame_idx: int, ts: float, pending: Optional[int] = None) -> bool:
        """第 frame_idx 帧（时刻 ts）是否需要取图分析。

        pending 仅在 analysis 模式下生效（队列积压深度），None 走自适应/纯抽帧。
        """
        if self.sampling == SAMPLING_ANALYSIS:
            return self.decide(frame_idx, pending=pending)
        if self.sampling == SAMPLING_WALL_CLOCK:
            if ts - self._last_wall >= self.interval_sec:
                self._last_wall = ts
                return True
            return False
        if self.sampling == SAMPLING_FRAME_COUNT:
            if frame_idx - self._last_frame >= self.interval_frames:
                self._last_frame = frame_idx
                return True
            return False
        return False

    def decide(self, frame_count: int, pending: Optional[int] = None) -> bool:
        """analysis 模式决策（原 FrameScheduler.decide）。

        pending 非 None 时为队列模式：积压超过 queue_threshold 则放大间隔，
        否则用基础间隔；pending=None 时为自适应/纯抽帧（由 note_inference
        维护的耗时统计驱动放宽）。
        """
        if pending is not None:
            if pending > self.queue_threshold:
                self._set_skip(min(self.base_skip * self.backpressure_multiplier, self.max_skip), relax=True)
            else:
                self._set_skip(self.base_skip)
        return frame_count % self._skip == 0

    def budget(self, source_fps: float) -> float:
        """帧间隔预算（秒）：按当前抽帧间隔，两帧分析之间的可用时间。"""
        return self._skip / source_fps if source_fps > 0 else 0.0

    # ---------------- 自适应模式的耗时反馈 ----------------
    def note_inference(self, cost_sec: float, budget_sec: float) -> None:
        """报告一次推理耗时（秒），自适应模式下据此调整抽帧间隔。

        - cost > budget * adaptive_relax_ratio   : 处理不过来，放宽（间隔放大 multiplier 倍）
        - cost < budget * adaptive_recover_ratio : 很轻松，逐步恢复基础间隔
        """
        if cost_sec < 0:
            return
        self._ema_cost = 0.8 * self._ema_cost + 0.2 * cost_sec if self._ema_cost else cost_sec
        if budget_sec <= 0:
            return
        if self._ema_cost > budget_sec * self.adaptive_relax_ratio:
            self._set_skip(min(self._skip * self.backpressure_multiplier, self.max_skip), relax=True)
        elif self._ema_cost < budget_sec * self.adaptive_recover_ratio and self._skip > self.base_skip:
            self._set_skip(max(self.base_skip, self._skip // self.backpressure_multiplier))

    def reset(self) -> None:
        """回到初始状态（新视频流开始前调用）。"""
        self._skip = self.base_skip
        self._relaxed = False
        self._ema_cost = 0.0
        self._last_wall = _NEG_INF
        self._last_frame = _NEG_INF

    # ---------------- 内部 ----------------
    def _set_skip(self, skip: int, relax: bool = False) -> None:
        skip = max(1, min(skip, self.max_skip))
        if skip != self._skip:
            if relax:
                self._relax_events += 1
            self._skip = skip
            self._relaxed = self._skip > self.base_skip


# ---------------- 通用：阶段间传递门控 ----------------
class IntervalGate:
    """按 key 的间隔门控——"向下一模块传递数据"的统一节流机制。

    同一 key 距上次放行 >= interval 才再次放行：
      interval > 0 ：每 interval 秒最多放行一次（周期刷新，如 check_interval=3）
      interval == 0：不节流，每次放行
      interval < 0 ：只放行一次（此后不再放行，如"同一车牌只告警一次"）

    用法: if gate.allow(key, ts): 传递数据; gate.note(key, ts)
    现有两处使用（同一机制，不同 key/interval）：
      yolo→vlm 报送（key=类别, interval=check_interval）、
      yolo→plate 车辆框（key=track_id, interval=track_cooldown）。
    （原 plate→alarm 新车牌门控已由「时序判定 SeqFilter」的同对象一次（cooldown<0）承担。）
    """

    def __init__(self, interval: float = 0.0) -> None:
        self.interval = interval
        self._last: Dict = {}  # key -> 上次放行时刻

    def allow(self, key, ts: float) -> bool:
        """该 key 本次是否放行（不记录；放行后需调用 note 记录时刻）。"""
        if self.interval == 0:
            return True
        last = self._last.get(key)
        if last is None:
            return True
        if self.interval < 0:
            return False
        return ts - last >= self.interval

    def note(self, key, ts: float) -> None:
        """记录该 key 的放行时刻。"""
        self._last[key] = ts


# ---------------- 模块 2: YOLO 识别（纯推理） ----------------
GRAN_SCENE = "scene"  # 周期性整帧（large_only）
GRAN_CLASS = "class"  # 按识别类型（类别）报送


class YOLODetector:
    """小模型识别（**纯推理**，2026-10 起过滤/报送外移）。

    - 推理参数（conf/iou/imgsz/max_det/classes/device/tracker）收进模块，
      测试/拼接层直接实例化传参即可，无需另写推理代码；model_path 为空时
      保留旧"纯后处理"兼容（不推理、供外部检测列表直传 Pipeline）。
    - 有效性过滤链与成组数量判定 → 「检测筛选 DetFilter」（pipe.det_filter，
      Pipeline.det_filter 阶段承接，对所有下游一致生效）。
    - 向下传递节流（check_interval）与按类报送编排（含 track_change_only、
      SubmitRequest 构造）→ Pipeline（本文件的装配器）。
    - 装配期翻译见 web_lab.server._chain_from_spec / compose()：preset 上的
      filter_conf/min_short/min_long/class_limits/check_interval/track_change_only
      仍可在 YOLO 参数面配置，由装配层译成 DetFilter/Pipeline 参数。
    """

    def __init__(
        self,
        # ---- 推理参数 ----
        model_path: Optional[str] = None,  # YOLO 权重路径 (.pt)；空=不推理（外部检测直传）
        conf: float = 0.35,  # 检出阈值（调低召回重叠/被遮挡目标）
        iou: float = 0.7,  # NMS 阈值（调低允许高度重叠的框并存）
        imgsz: int = 640,  # 推理分辨率（调高召回小目标/重叠目标，算力约平方增长）
        max_det: int = 300,  # 单帧最多检出的目标数
        classes: Optional[List[int]] = None,  # 只检这些类别（None=不限制）
        device: Optional[str] = None,  # "0"/"cpu"/None=ultralytics 自动
        tracker: str = "bytetrack.yaml",  # 跟踪器配置（track=True 时用；必须显式传，否则本版 ultralytics 不分配 id）
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes
        self.device = device
        self.tracker = tracker
        self._model = None  # 惰性加载

    # ---- 推理 ----
    def infer(self, frame, track: bool = False) -> List[Detection]:
        """跑 YOLO 推理，返回 Detection 列表（track=True 用跟踪器保持 track_id 稳定）。

        输出为**纯推理结果**（不做有效性过滤）；有效性（置信/尺寸/成组数量）
        由链上的检测筛选阶段（Pipeline.det_filter）判定，对全部下游一致生效。
        """
        if not self.model_path:
            raise RuntimeError("YOLODetector 未配置 model_path，不能推理（外部检测直传模式）")
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            if self.device:
                self._model.to(self.device)
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                  max_det=self.max_det, classes=self.classes, verbose=False)
        if self.device:
            kw["device"] = self.device
        r = (self._model.track(frame, persist=True, tracker=self.tracker, **kw) if track
             else self._model.predict(frame, **kw))[0]
        dets: List[Detection] = []
        if r.boxes is None:
            return dets
        for i, b in enumerate(r.boxes.xyxy.tolist()):
            tid = r.boxes.id[i].item() if (track and r.boxes.id is not None) else 0
            dets.append(Detection(
                track_id=int(tid), cls_id=int(r.boxes.cls[i].item()),
                bbox=tuple(int(v) for v in b), score=float(r.boxes.conf[i].item()),
                model_path=self.model_path.rsplit("/", 1)[-1]))
        return dets


# ---------------- 模块 3: VLM 研判 ----------------
class VLMAnalyzer:
    """素材准备 + 推理。prepare 产出 payload；推理由外部回调注入（真实场景为 spaz consumer）。"""

    def __init__(
        self,
        max_resolution: Tuple[int, int] = (1280, 720),  # 送审前缩放上限
        prompt: str = "",  # VLM 提示词
    ) -> None:
        self.max_resolution = max_resolution
        self.prompt = prompt

    def prepare(self, req: SubmitRequest, frame) -> dict:
        """把报送请求转成 VLM payload（素材准备，由拼接层提供 frame/ROI 取图）。"""
        return {
            "ref": req.ref,
            "gran": req.gran,
            "track_key": req.track_key,
            "prompt": self.prompt,
            "max_resolution": self.max_resolution,
            "bbox": req.det.bbox if req.det else None,
            "det_count": len(req.dets),
        }

    def analyze(self, payload: dict, infer: Callable[[dict], str]) -> str:
        """执行推理（infer 为外部 VLM 调用，返回 action 字符串）。"""
        return infer(payload)




# ---------------- 装配器 ----------------
@dataclass
class Pipeline:
    """组装好的分析管道。step() 驱动：帧决策 → YOLO → 目标裁剪 → 报送/事件（无输出处理模块）。

    crop 为可选的"目标裁剪阶段"（鸭子类型：需实现 run(frame, dets, ts) -> list），
    plate 为可选的"车牌识别阶段"（需实现 run(frame, crops, ts) -> list），
    均由拼接层注入（pipe.target_crop.TargetCrop / plate_recog.PlateStage），
    本模块不引入 cv2/onnx 依赖。链路输出 = 事件尾（event_filter）产出的事件（last_events）。
    """

    frame: FrameManager
    yolo: Optional[YOLODetector] = None
    vlm: Optional[VLMAnalyzer] = None
    infer: Optional[Callable[[dict], str]] = None  # 外部 VLM 推理回调
    crop: Optional[object] = None  # 目标裁剪阶段（TargetCrop.run(frame, dets, ts)）
    annotate: Optional[object] = None  # 标注阶段（Annotator.run(frame, dets, ts) -> 带框图）
    plate: Optional[object] = None  # 车牌识别阶段（PlateStage.run(frame, crops, ts)）
    det_filter: Optional[DetFilter] = None  # 检测筛选阶段（有效性：逐条+成组，2026-10 自 YOLO 迁入）
    seq_filter: Optional[object] = None  # 时序判定阶段（节流/变化门控 + 事件，2026-10 自 Pipeline 迁入）
    event_filter: Optional[object] = None  # 事件尾（VLM/车牌等外部记录经 publish 走它出事件，可作链路尾）
    last_plates: List = field(default_factory=list)  # 最近一帧识别到的车牌（供驱动层读取）
    last_crops: List = field(default_factory=list)  # 最近一帧目标裁剪产物（供驱动层读取）
    last_annotated: object = None  # 最近一帧标注图（供驱动层/VLM 素材读取）
    last_plate_input: List = field(default_factory=list)  # 最近一帧传给车牌阶段的车辆子图（门控后）
    last_filter: Optional[Tuple[int, int]] = None  # 最近一帧检测筛选统计 (进入, 放行)（供时间线展示；未运行=None）
    last_events: List = field(default_factory=list)  # 最近一帧末段记录携带的事件（链路尾通用输出，见 records.events）

    def _submits(self, dets: List[Detection]) -> List[SubmitRequest]:
        """按识别类型（类别）构造报送请求（有效性/门控已由 det_filter/seq_filter 完成）。

        只分组 + 造语料：一类目标再多也只算一条请求（ref=full:cls{cls}）。何时报、
        报不报由链上 det_filter（有效性）与 seq_filter（节流/变化门控）决定。
        """
        by_cls: Dict[int, List[Detection]] = {}
        for d in dets:
            by_cls.setdefault(d.cls_id, []).append(d)
        return [SubmitRequest(cls, GRAN_CLASS, f"full:cls{cls}", dets=cdets)
                for cls, cdets in sorted(by_cls.items())]

    def publish(self, records, ts: float = 0.0, frame_idx: int = 0) -> List[dict]:
        """把外部产生的记录（如 VLM 结果/车牌结果）经**事件尾**（event_filter）处理后
        汇总事件返回（collect_events）。

        通用：不绑定记录来源与内容；供以筛选模块收尾（无 alarm 阶段）的链路把
        VLM 的 JSON 内容等作为链路尾输出。记录用 dict 携带字段（如 content/label）。
        """
        items = list(records or [])
        if items and self.event_filter is not None:
            items = self.event_filter.run(items, ts=ts, frame_idx=frame_idx)
        return collect_events(items)

    def step(self, frame_idx: int, ts: float, dets: Optional[List[Detection]] = None,
             frame=None, force: bool = False):
        """处理一帧。返回 (submits, events)。
        dets 由拼接层在 wants_frame 为真时做 YOLO 推理提供（无 YOLO 的模式传 None）；
        不传 dets 但配置了 yolo 时，传 frame 由模块自推理（车牌按 track 去重时 track=True）。
        force=True 时跳过帧决策检查——供调用方已先行 wants_frame() 门控避免重复采样。"""
        submits: List[SubmitRequest] = []
        if not force and not self.frame.wants_frame(frame_idx, ts):
            return submits, []
        if dets is None and self.yolo is not None and frame is not None:
            dets = self.yolo.infer(frame, track=self.track_needed())

        # 检测筛选阶段：有效性判定（逐条条件 + 成组数量；原 YOLO 输出过滤链）——
        # 不达标的检测视为"未识别"，对**所有**下游（裁剪/标注/报送/事件）一致生效。
        self.last_filter = None
        if dets and self.det_filter is not None:
            n_in = len(dets)
            dets = self.det_filter.run(dets, ts=ts, frame_idx=frame_idx)
            self.last_filter = (n_in, len(dets))

        # 时序判定阶段：节流/变化门控（原 Pipeline check_interval/track_change_only）——
        # 门控下游可见的检测集。
        if dets and self.seq_filter is not None:
            dets = self.seq_filter.run(dets, ts=ts, frame_idx=frame_idx)

        # 目标裁剪阶段：按上游检测框裁子图（供车牌识别/人脸检测/VLM 下游消费）。
        self.last_crops = []
        if self.crop is not None and frame is not None and dets is not None:
            self.last_crops = list(self.crop.run(frame, dets, ts))

        # 标注阶段：在帧副本上画上游检测框（供 VLM 看"原图 + 画框"）
        self.last_annotated = None
        if self.annotate is not None and frame is not None and dets is not None:
            self.last_annotated = self.annotate.run(frame, dets, ts)

        if self.yolo is not None:
            if self.crop is not None:
                pass  # 目标裁剪已产出，供下游阶段消费
            elif dets is not None and self.vlm is not None:
                # 报送请求的唯一消费者是 VLM 阶段（门控已在上游 det_filter/seq_filter 生效）
                submits = self._submits(dets)
        elif self.vlm is not None:
            # large_only: 帧决策即送整帧
            submits = [SubmitRequest(-1, GRAN_SCENE, "full:global")]

        # 车牌识别阶段：识别 → 事件尾（event_filter 经 publish 出事件，幸存者=告警载体）
        self.last_plates = []
        self.last_plate_input = []
        plate_events: List[dict] = []
        if self.plate is not None and frame is not None:
            self.last_plate_input = list(self.last_crops)
            self.last_plates = list(self.plate.run(frame, self.last_crops, ts))
            for p in self.last_plates:
                if not p.plate_no:
                    continue
                rec = {"kind": "plate", "label": p.plate_no, "score": p.score,
                       "content": {"plate_no": p.plate_no, "type_name": p.type_name,
                                   "color": p.color, "bbox": list(p.bbox)}}
                plate_events += self.publish([rec], ts, frame_idx)
        # 链路尾通用输出：末段检测记录携带的事件（det_filter/seq_filter 等模块追加）+
        # 车牌事件尾产出。通用、不绑定模块——供驱动层读取。
        self.last_events = (collect_events(dets) if dets else []) + plate_events
        return submits, self.last_events

    def track_needed(self) -> bool:
        """是否需要 YOLO 跟踪（车牌阶段按 track 去重时需要）。"""
        return bool(getattr(self.plate, "use_track", False))


def _yolo_kw(spec: dict) -> dict:
    """从 spec 提取 YOLO 推理参数（yolo_* 前缀），仅透传显式配置的字段。"""
    return {k: v for k, v in {
        "model_path": spec.get("yolo_model"),
        "conf": spec.get("yolo_conf"),
        "iou": spec.get("yolo_iou"),
        "imgsz": spec.get("yolo_imgsz"),
        "max_det": spec.get("yolo_max_det"),
        "classes": spec.get("yolo_classes"),
        "device": spec.get("yolo_device"),
    }.items() if v is not None}


def _seq_filter_of_spec(spec: dict):
    """spec 的 check_interval/track_change_only → SeqFilter 门控（无则 None）。

    兼容旧代码级 spec：check_interval → throttle(key=cls_id)；track_change_only →
    change(key=cls_id, watch=track_id)。2026-10 起这两个语义由 seq_filter 承担。
    """
    throttle = change = None
    try:
        ci = float(spec.get("check_interval", 0.0) or 0.0)
    except (TypeError, ValueError):
        ci = 0.0
    if ci > 0:
        throttle = {"key": "cls_id", "interval": ci}
    if spec.get("track_change_only"):
        change = {"key": "cls_id", "watch": "track_id"}
    if throttle is None and change is None:
        return None
    return SeqFilter(throttle=throttle, change=change)


def compose(spec: dict) -> Pipeline:
    """把任务配置映射为模块组合。spec 各字段来自数据库任务配置。

    注：`small_only`（小模型驻留告警）已随「输出处理」弃用删除；链路输出改用
    事件尾（det_filter/seq_filter 的 events）。
    """
    mode = spec["algo_mode"]
    extract_rate = spec.get("extract_frame_rate", 3)
    if mode == "small_only":
        raise ValueError("small_only（驻留告警）已弃用：改用检测筛选/时序判定的链路事件尾")
    # 有效性筛选（原 YOLO 过滤链+识别对象数量范围）：逐条谓词 + 成组数量判定
    det_filter = DetFilter(predicates=spec.get("filters"),       # 自定义顺序过滤链（可选）
                           group_limits=spec.get("class_limits", {}))  # 空 dict=全拦（旧语义）
    seq_filter = _seq_filter_of_spec(spec)  # 节流/变化门控（原 check_interval/track_change_only）

    if mode in ("small_crop", "small_full"):
        # 报送粒度固定为"按识别类型"；均送该类全帧(full:cls)、由拼接层画该类框标注。
        # 需送该类 ROI 裁剪时，链路在设计器里 YOLO 之后插入「目标裁剪」阶段。
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=extract_rate),
            yolo=YOLODetector(**_yolo_kw(spec)),
            det_filter=det_filter,
            seq_filter=seq_filter,
            vlm=VLMAnalyzer(max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
        )

    if mode == "large_only":
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_WALL_CLOCK, interval_sec=spec.get("check_interval", 3.0)),
            vlm=VLMAnalyzer(max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
        )

    raise ValueError(f"未知模式: {mode}")
