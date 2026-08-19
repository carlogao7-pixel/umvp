# -*- coding: utf-8 -*-
"""
模块化管道拼接器（重构版）：四模块 + 装配器

按"参数归属模块"重构（用户确认的分工）:
  1. FrameManager  帧管理 —— 取图节奏(抽帧 / 墙钟秒 / 帧数) + 背压（原 FrameScheduler 并入）
  2. YOLODetector  小模型识别 —— 推理参数可调(conf/iou/imgsz/max_det/classes/device) +
                     可插拔过滤链(顺序自定义) + 按识别类型(类别)报送 + 每类数量上下限
  3. VLMAnalyzer   大模型研判 —— 素材准备(crop_padding/缩放/prompt) + 推理
  4. AlarmPolicy   告警策略 —— 判定(window 滑动窗口 / dwell 驻留 / inspection 巡检) + 冷却(scope 显式)

装配器 compose(): 把四种分析模式（small_only/small_crop/small_full/large_only）
映射为模块组合与参数，模式差异收敛为配置，不产生分支代码。

YOLO 输出粒度（用户定义，2026-08-04 改）:
  class —— 按识别类型（类别）单独报送：画面里每出现一类识别对象（且数量在
           "该类上下限"范围内）就发一条请求，一类多个目标合并成一条（ref=full:cls{类别}
           送全帧、拼接层画该类框标注；或 ref=crop:cls{类别} 送该类 ROI 裁剪）。
  scene —— 仅 large_only 的周期性整帧（ref=full:global），不经过小模型。
  说明: per_track（按目标独立报送）已删除；驻留告警(dwell)仍按 track 记首次出现，
        但那是告警策略内部逻辑，不产生报送请求。

依赖: 纯 Python（抽帧+背压已在 FrameManager 内部，不再依赖 face_scan）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
    """一次报送请求（YOLO → VLM 或 直接告警）。"""

    track_key: int  # 按识别类型=类别编号; large_only 整帧=-1
    gran: str  # "class"（按识别类型） | "scene"（周期性整帧）
    ref: str  # "full:cls{类别}" / "crop:cls{类别}" / "full:global"
    det: Optional[Detection] = None  # 保留字段（单目标素材用）；按类报送时为空
    dets: List[Detection] = field(default_factory=list)  # 该类全部有效检测（large_only 为空）


@dataclass
class AlarmEvent:
    """一次告警触发。"""

    task_id: int
    gran: str
    track_key: int
    action: str  # dwell:{cls} / 命中 action / inspection
    ts: float


# ---------------- 模块 1: 帧管理 ----------------
SAMPLING_ANALYSIS = "analysis"  # 随抽帧节奏（extract_frame_rate，含背压）
SAMPLING_WALL_CLOCK = "wall_clock"  # 按墙钟秒（large_only: check_interval）
SAMPLING_FRAME_COUNT = "frame_count"  # 按帧数（temporal: check_interval × fps）


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


# ---------------- 模块 2: YOLO 识别 ----------------
GRAN_TRACK = "per_track"  # 仅驻留告警(dwell)事件标注用；报送不再按 track
GRAN_SCENE = "scene"  # 周期性整帧（large_only）
GRAN_CLASS = "class"  # 按识别类型（类别）报送

# per-detection 过滤器（顺序可自定义，都是布尔谓词）
Filter = Callable[[Detection], bool]


def filter_model(path: str) -> Filter:
    return lambda d: not d.model_path or d.model_path == path


def filter_confidence(t: float) -> Filter:
    return lambda d: d.score >= t


def filter_classes(cs: List[int]) -> Filter:
    return lambda d: d.cls_id in cs


def filter_min_size(min_short: int, min_long: int) -> Filter:
    def _f(d: Detection) -> bool:
        w, h = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
        return min(w, h) >= min_short and max(w, h) >= min_long
    return _f


class YOLODetector:
    """推理(可选) + 过滤链 + 按识别类型（类别）报送。
    filters 为可自定义顺序的 per-detection 谓词序列。

    model_path 非空时支持 infer()：把推理参数(conf/iou/imgsz/max_det/classes/device)
    收进模块，测试/拼接层直接实例化本模块传参即可，无需另写推理代码。
    model_path 为空则退化为纯后处理（只过滤+报送，供外部已跑好的检测列表用）。
    """

    def __init__(
        self,
        filters: Optional[List[Filter]] = None,
        class_limits: Optional[Dict[int, Tuple[int, int]]] = None,
        # 识别对象: 类别编号 -> (数量下限, 数量上限)，未登记的类别直接忽略；
        # 数量按"过滤链后的该类目标数"算，不在范围内 → 该类不报送
        check_interval: float = 3.0,  # 每类独立计时的送审节拍
        roi: bool = False,  # False=送全帧(full:cls{类别}); True=送该类 ROI 裁剪(crop:cls{类别})
        # ---- 推理参数（model_path 为空则不推理，纯后处理） ----
        model_path: Optional[str] = None,  # YOLO 权重路径 (.pt)
        conf: float = 0.35,  # 检出阈值（调低召回重叠/被遮挡目标）
        iou: float = 0.7,  # NMS 阈值（调低允许高度重叠的框并存）
        imgsz: int = 640,  # 推理分辨率（调高召回小目标/重叠目标，算力约平方增长）
        max_det: int = 300,  # 单帧最多检出的目标数
        classes: Optional[List[int]] = None,  # 只检这些类别（None=不限制）
        device: Optional[str] = None,  # "0"/"cpu"/None=ultralytics 自动
    ) -> None:
        self.filters = list(filters or [])
        self.class_limits = dict(class_limits or {})
        self.check_interval = check_interval
        self.roi = roi
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes
        self.device = device
        self._model = None  # 惰性加载
        self._last_check: Dict[int, float] = {}  # 类别 -> 上次报送时刻

    # ---- 推理（model_path 非空时可用） ----
    def infer(self, frame, track: bool = False) -> List[Detection]:
        """跑 YOLO 推理，返回 Detection 列表（track=True 用内置 BOTSORT 保持 track_id 稳定）。"""
        if not self.model_path:
            raise RuntimeError("YOLODetector 未配置 model_path，不能推理（纯后处理模式）")
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            if self.device:
                self._model.to(self.device)
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                  max_det=self.max_det, classes=self.classes, verbose=False)
        r = (self._model.track(frame, persist=True, **kw) if track
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

    # ---- 过滤链（可自定义顺序，供纯过滤/dwell 复用） ----
    def filter_only(self, dets: List[Detection]) -> List[Detection]:
        """全局过滤链 + 识别对象白名单 + 每类数量范围。数量越界的那一类整体忽略。"""
        valid = list(dets)
        for f in self.filters:
            valid = [d for d in valid if f(d)]
        if not self.class_limits:
            return []
        by_cls: Dict[int, List[Detection]] = {}
        for d in valid:
            by_cls.setdefault(d.cls_id, []).append(d)
        kept: List[Detection] = []
        for cls, cdets in by_cls.items():
            lo, hi = self.class_limits.get(cls, (-1, -1))
            if lo >= 0 and lo <= len(cdets) <= hi:
                kept += cdets
        return kept

    # ---- 主入口：dets -> 按识别类型报送 ----
    def process(self, frame_idx: int, ts: float, dets: List[Detection]) -> List[SubmitRequest]:
        valid = self.filter_only(dets)
        if not valid:
            return []
        by_cls: Dict[int, List[Detection]] = {}
        for d in valid:
            by_cls.setdefault(d.cls_id, []).append(d)
        reqs: List[SubmitRequest] = []
        for cls, cdets in sorted(by_cls.items()):
            # 每类独立计时：一类目标再多也只算一条请求
            if ts - self._last_check.get(cls, _NEG_INF) >= self.check_interval:
                self._last_check[cls] = ts
                ref = f"crop:cls{cls}" if self.roi else f"full:cls{cls}"
                reqs.append(SubmitRequest(cls, GRAN_CLASS, ref, dets=cdets))
        return reqs


# ---------------- 模块 3: VLM 研判 ----------------
class VLMAnalyzer:
    """素材准备 + 推理。prepare 产出 payload；推理由外部回调注入（真实场景为 spaz consumer）。"""

    def __init__(
        self,
        crop_padding: float = 0.15,  # 裁剪外扩比例（crop:* 素材外扩）
        max_resolution: Tuple[int, int] = (1280, 720),  # 送审前缩放上限
        prompt: str = "",  # VLM 提示词
    ) -> None:
        self.crop_padding = crop_padding
        self.max_resolution = max_resolution
        self.prompt = prompt

    def prepare(self, req: SubmitRequest, frame) -> dict:
        """把报送请求转成 VLM payload（素材准备，由拼接层提供 frame/ROI 取图）。"""
        return {
            "ref": req.ref,
            "gran": req.gran,
            "track_key": req.track_key,
            "prompt": self.prompt,
            # 仅裁剪送审(crop:*)带外扩；全帧(full:*)不带
            "crop_padding": self.crop_padding if req.ref.startswith("crop:") else None,
            "max_resolution": self.max_resolution,
            "bbox": req.det.bbox if req.det else None,
            "det_count": len(req.dets),
        }

    def analyze(self, payload: dict, infer: Callable[[dict], str]) -> str:
        """执行推理（infer 为外部 VLM 调用，返回 action 字符串）。"""
        return infer(payload)


# ---------------- 模块 4: 告警策略 ----------------
ALARM_WINDOW = "window"  # VLM 滑动窗口确认
ALARM_DWELL = "dwell"  # 驻留秒数（small_only）
ALARM_INSPECTION = "inspection"  # 巡检：结果直接落盘


class AlarmPolicy:
    """告警判定 + 冷却。scope 规则随 kind 显式指定：
       dwell → (task_id, cls_id) 按类别；window/inspection → (task_id, task_id) 全局。
    """

    def __init__(
        self,
        task_id: int,
        kind: str = ALARM_WINDOW,
        target_actions: Optional[List[str]] = None,  # 空 = 巡检（kind 自动 inspection）
        smooth_frames: int = 1,  # 双语义: dwell=驻留秒数阈值; window=滑动窗口长度
        hit_ratio: float = 1.0,  # window: 命中比例
        alarm_cooldown: float = 60.0,
    ) -> None:
        self.task_id = task_id
        self.target_actions = target_actions or []
        # 巡检模式仅在 VLM 窗口模式下由"无 target_actions"自动派生；
        # dwell（驻留告警）模式天然无 target_actions，不能误判为巡检
        self.kind = ALARM_INSPECTION if (not self.target_actions and kind == ALARM_WINDOW) else kind
        self.smooth_frames = max(1, int(smooth_frames))
        self.hit_ratio = hit_ratio
        self.alarm_cooldown = alarm_cooldown
        self._history: Dict[int, List[str]] = {}
        self._first_seen: Dict[int, float] = {}
        self._cooldown_until: Dict[Tuple[int, int], float] = {}

    def _cooling(self, scope: Tuple[int, int], ts: float) -> bool:
        return self._cooldown_until.get(scope, _NEG_INF) > ts

    # ---- VLM 结果回收（window / inspection） ----
    def on_vlm_result(self, track_key: int, action: str, ts: float) -> List[AlarmEvent]:
        # 报送粒度: 周期性整帧=-1 → scene; 其余(按识别类型) track_key=类别编号 → class
        gran = GRAN_SCENE if track_key == -1 else GRAN_CLASS
        if self.kind == ALARM_INSPECTION:  # 巡检：不匹配 action，直接落盘
            return [AlarmEvent(self.task_id, gran, track_key, "inspection", ts)]
        h = self._history.setdefault(track_key, [])
        h.append(action)
        if len(h) > self.smooth_frames:
            h.pop(0)
        required = max(1, math.ceil(self.smooth_frames * self.hit_ratio))
        hits = sum(1 for a in h if a in self.target_actions)
        if action in self.target_actions and len(h) >= self.smooth_frames and hits >= required:
            scope = (self.task_id, self.task_id)  # VLM 模式全局冷却
            if not self._cooling(scope, ts):
                self._cooldown_until[scope] = ts + self.alarm_cooldown
                return [AlarmEvent(self.task_id, gran, track_key, action, ts)]
        return []

    # ---- 驻留告警（small_only） ----
    def on_dwell(self, track_id: int, cls_id: int, ts: float) -> List[AlarmEvent]:
        self._first_seen.setdefault(track_id, ts)  # 按 track 记首次出现
        dwell = ts - self._first_seen[track_id]
        scope = (self.task_id, cls_id)  # 按类别冷却
        if dwell >= self.smooth_frames and not self._cooling(scope, ts):
            self._cooldown_until[scope] = ts + self.alarm_cooldown
            return [AlarmEvent(self.task_id, GRAN_TRACK, track_id, f"dwell:{cls_id}", ts)]
        return []


# ---------------- 装配器 ----------------
@dataclass
class Pipeline:
    """组装好的分析管道。step() 驱动：帧决策 → YOLO → 报送/告警。"""

    frame: FrameManager
    alarm: AlarmPolicy
    yolo: Optional[YOLODetector] = None
    vlm: Optional[VLMAnalyzer] = None
    infer: Optional[Callable[[dict], str]] = None  # 外部 VLM 推理回调

    def step(self, frame_idx: int, ts: float, dets: Optional[List[Detection]] = None,
             frame=None, force: bool = False):
        """处理一帧。返回 (submits, alarms)。
        dets 由拼接层在 wants_frame 为真时做 YOLO 推理提供（无 YOLO 的模式传 None）；
        不传 dets 但配置了 yolo 时，传 frame 由模块自推理（dwell 用 track=True 保持
        track_id 稳定，其它模式 track=False）。force=True 时跳过帧决策检查——
        供调用方已先行 wants_frame() 门控（如 h_chain 视频驱动）避免重复采样。"""
        submits: List[SubmitRequest] = []
        alarms: List[AlarmEvent] = []
        if not force and not self.frame.wants_frame(frame_idx, ts):
            return submits, alarms
        if dets is None and self.yolo is not None and frame is not None:
            dets = self.yolo.infer(frame, track=(self.alarm.kind == ALARM_DWELL))

        if self.yolo is not None:
            if self.alarm.kind == ALARM_DWELL and dets is not None:
                # small_only: 无 VLM，过滤后逐 track 驻留告警
                for d in self.yolo.filter_only(dets):
                    alarms += self.alarm.on_dwell(d.track_id, d.cls_id, ts)
            elif dets is not None:
                submits = self.yolo.process(frame_idx, ts, dets)
        elif self.vlm is not None and self.alarm.kind != ALARM_DWELL:
            # large_only: 帧决策即送整帧
            submits = [SubmitRequest(-1, GRAN_SCENE, "full:global")]
        return submits, alarms

    def on_vlm_result(self, track_key: int, action: str, ts: float) -> List[AlarmEvent]:
        return self.alarm.on_vlm_result(track_key, action, ts)


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


def compose(spec: dict) -> Pipeline:
    """把任务配置映射为模块组合。spec 各字段来自数据库任务配置。"""
    mode = spec["algo_mode"]
    tid = spec["task_id"]
    extract_rate = spec.get("extract_frame_rate", 3)
    target_actions = spec.get("target_actions", [])
    smooth = spec.get("smooth_frames", 1)
    filters = spec.get("filters")  # 自定义顺序的过滤链（可选）

    if mode == "small_only":
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=extract_rate),
            yolo=YOLODetector(filters=filters, class_limits=spec.get("class_limits", {}),
                              **_yolo_kw(spec)),
            alarm=AlarmPolicy(task_id=tid, kind=ALARM_DWELL, smooth_frames=smooth,
                              alarm_cooldown=spec.get("alarm_cooldown", 60.0)),
        )

    if mode in ("small_crop", "small_full"):
        # 报送粒度固定为"按识别类型"；small_crop 送该类裁剪图(crop:cls)，
        # small_full 送全帧(full:cls)、由拼接层画该类框标注；roi_output 可覆盖
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=extract_rate),
            yolo=YOLODetector(filters=filters,
                              class_limits=spec.get("class_limits", {}),
                              check_interval=spec.get("check_interval", 3.0),
                              roi=spec.get("roi_output", mode == "small_crop"),
                              **_yolo_kw(spec)),
            vlm=VLMAnalyzer(crop_padding=spec.get("crop_padding", 0.15),
                            max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
            alarm=AlarmPolicy(task_id=tid, kind=ALARM_WINDOW, target_actions=target_actions,
                              smooth_frames=smooth,
                              hit_ratio=spec.get("vlm_hit_ratio", 1.0),
                              alarm_cooldown=spec.get("alarm_cooldown", 60.0)),
        )

    if mode == "large_only":
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_WALL_CLOCK, interval_sec=spec.get("check_interval", 3.0)),
            vlm=VLMAnalyzer(crop_padding=spec.get("crop_padding", 0.15),
                            max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
            alarm=AlarmPolicy(task_id=tid, kind=ALARM_WINDOW, target_actions=target_actions,
                              smooth_frames=smooth,
                              hit_ratio=spec.get("vlm_hit_ratio", 1.0),
                              alarm_cooldown=spec.get("alarm_cooldown", 60.0)),
        )

    raise ValueError(f"未知模式: {mode}")
