# -*- coding: utf-8 -*-
"""
模块化管道拼接器（重构版）：四模块 + 装配器

按"参数归属模块"重构（用户确认的分工）:
  1. FrameManager  帧管理 —— 取图节奏(抽帧 / 墙钟秒 / 帧数) + 背压（原 FrameScheduler 并入）
  2. YOLODetector  小模型识别 —— 推理参数可调(conf/iou/imgsz/max_det/classes/device) +
                     可插拔过滤链(顺序自定义) + 按识别类型(类别)报送 + 每类数量上下限
  3. VLMAnalyzer   大模型研判 —— 素材准备(crop_padding/缩放/prompt) + 推理
  4. OutputPolicy  输出处理（原告警策略）—— 统一漏斗：Finding → 驻留/词表/置信/规则/
                     滑窗/去重冷却 七道闸 → OutputEvent（AlarmPolicy 为兼容别名）

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
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    """一次输出/告警事件。"""

    task_id: int
    gran: str
    track_key: Any  # 对象标识：int（track/类别）或 str（如 "plate:粤ACJ0710"）
    action: str  # dwell:{cls} / 命中 action / inspection
    ts: float
    description: str = ""  # VLM 返回的警情描述 / 车牌摘要
    conf: float = 1.0  # 来源置信度（车牌 score / 人脸相似度…；无则 1.0）
    features: Optional[dict] = None  # 特征包（车牌号/颜色/读数等，原样透传）


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
    （原 plate→alarm 新车牌门控已并入输出漏斗 OutputPolicy 的 ⑦去重冷却闸。）
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
        track_change_only: bool = False,  # 只在 track_id 变化时才报送（需要跟踪）
        # ---- 推理参数（model_path 为空则不推理，纯后处理） ----
        model_path: Optional[str] = None,  # YOLO 权重路径 (.pt)
        conf: float = 0.35,  # 检出阈值（调低召回重叠/被遮挡目标）
        iou: float = 0.7,  # NMS 阈值（调低允许高度重叠的框并存）
        imgsz: int = 640,  # 推理分辨率（调高召回小目标/重叠目标，算力约平方增长）
        max_det: int = 300,  # 单帧最多检出的目标数
        classes: Optional[List[int]] = None,  # 只检这些类别（None=不限制）
        device: Optional[str] = None,  # "0"/"cpu"/None=ultralytics 自动
        tracker: str = "bytetrack.yaml",  # 跟踪器配置（track=True 时用；必须显式传，否则本版 ultralytics 不分配 id）
    ) -> None:
        self.filters = list(filters or [])
        self.class_limits = dict(class_limits or {})
        self.check_interval = check_interval
        self.track_change_only = track_change_only
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes
        self.device = device
        self.tracker = tracker
        self._model = None  # 惰性加载
        self._gate = IntervalGate(check_interval)  # 向下游(VLM)报送的间隔门控（key=类别）
        self._last_track_ids: Dict[int, set] = {}  # 类别 -> 上次 track_id 集合（track_change_only 模式）

    # ---- 推理（model_path 非空时可用） ----
    def infer(self, frame, track: bool = False) -> List[Detection]:
        """跑 YOLO 推理，返回 Detection 列表（track=True 用跟踪器保持 track_id 稳定）。"""
        if not self.model_path:
            raise RuntimeError("YOLODetector 未配置 model_path，不能推理（纯后处理模式）")
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
        # 过滤链在输出处生效：置信度/尺寸（filter_conf/min_short/min_long 等）不达标的
        # 目标视为"未识别"，不进入任何下游（VLM 报送 / 车牌·人脸传递 / 标注 / 计数）。
        # 顺序保证：逐框过滤在前，class_limits 的按类计数只数过滤后的目标。
        return [d for d in dets if all(f(d) for f in self.filters)]

    # ---- 向非 VLM 下游传递（统一按识别类型门控） ----
    def pass_dets(self, dets: List[Detection], ts: float) -> List[Detection]:
        """按 check_interval（key=识别类型）放行检测结果——向非 VLM 下游（如车牌阶段）
        传递数据的统一间隔门控，与 process()（VLM 报送）共用同一 gate 状态。

        入参 dets 通常来自 infer()（已在输出处过完滤链，不达标目标不出现）；
        本方法只做按类间隔门控，不再过滤。
        同一帧内同一识别类型要么整类放行、要么整类拦截；0=每分析帧都放行。
        新目标的入场延迟由帧管理取帧间隔与下游自身去重（如车牌按 track）兜底。
        """
        out: List[Detection] = []
        passed: set = set()
        for d in dets:
            cls = d.cls_id
            if cls not in passed and self._gate.allow(cls, ts):
                self._gate.note(cls, ts)
                passed.add(cls)
            if cls in passed:
                out.append(d)
        return out

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
            if self._gate.allow(cls, ts):
                # track_change_only 模式：检查 track_id 是否变化
                if self.track_change_only:
                    current_track_ids = {d.track_id for d in cdets if d.track_id > 0}
                    last_track_ids = self._last_track_ids.get(cls, set())
                    if current_track_ids == last_track_ids and len(current_track_ids) > 0:
                        # track_id 无变化，跳过报送
                        continue
                    self._last_track_ids[cls] = current_track_ids
                
                self._gate.note(cls, ts)
                ref = f"full:cls{cls}"
                reqs.append(SubmitRequest(cls, GRAN_CLASS, ref, dets=cdets))
        return reqs


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


# ---------------- 模块 4: 输出处理（原告警策略） ----------------
ALARM_WINDOW = "window"  # VLM 滑动窗口确认
ALARM_DWELL = "dwell"  # 驻留秒数（small_only）
ALARM_INSPECTION = "inspection"  # 巡检：结果直接落盘（同 key 只一次，见 key_cooldown）


@dataclass
class Finding:
    """一次发现（输出漏斗的统一输入）：多帧视觉分析的公共形态 =
    对象(key) + 主分类(label) + 置信度(conf) + 特征包(features) + 时间(ts)。"""

    key: Any  # 对象标识：track_id / 类别编号 / "plate:粤ACJ0710" / "scene"
    label: str  # 主分类标签：VLM action 词 / 车牌号 / "dwell:{cls}"
    ts: float
    gran: str = "class"  # 报送粒度（class/scene/track，进事件透传）
    conf: float = 1.0  # 置信度
    features: Optional[dict] = None  # 特征包：车牌号/颜色/读数/VLM 描述等，原样透传到事件


# ---- 规则 DSL：组内逗号 AND、组间分号 OR；条件 = 字段 操作符 值 ----
# 字段: conf | dwell | label | key | features.<名>（缺字段 → 该条不通过）
# 操作符: >= > <= < == != contains in（in 值用 | 分隔多选）；值: 数字或字符串（可带引号）
# 语法约定：操作符两侧必须留空格（如 conf >= 0.8），否则按无法解析报错——
# 这同时阻止正则回退把字段内的 "in"、孤立的 ">=" 误当操作符。
_RULE_RE = re.compile(
    r"^\s*([A-Za-z_][\w.]*)\s+(>=|<=|==|!=|>(?!=)|<(?!=)|contains|in)\s+(.+?)\s*$")


def _to_num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_rule_value(text: str):
    """值解析：数字 → float；引号包裹 → 去引号；其余原样字符串。"""
    t = text.strip()
    if len(t) >= 2 and t[0] in ("'", '"') and t[-1] == t[0]:
        return t[1:-1]
    n = _to_num(t)
    return n if n is not None else t


def parse_rules(rules) -> List[List[Tuple[str, str, Any]]]:
    """规则 DSL → [[(字段, 操作符, 值), ...](组内 AND), ...](组间 OR)。空 → []（闸关）。

    非法表达式抛 ValueError（通俗报错：指明第几组第几条、错在哪）——供保存/装配时快速失败。
    """
    if rules is None or (isinstance(rules, str) and not rules.strip()):
        return []
    if not isinstance(rules, str):  # 已解析的结构直接透传（内部便利）
        return list(rules)
    groups: List[List[Tuple[str, str, Any]]] = []
    for gi, gtext in enumerate(str(rules).split(";"), 1):
        conds: List[Tuple[str, str, Any]] = []
        parts = [c for c in (p.strip() for p in gtext.split(",")) if c]
        if not parts:
            raise ValueError(f"输出规则第 {gi} 组为空（组内逗号=并且，组间分号=或者）")
        for ci, cond in enumerate(parts, 1):
            m = _RULE_RE.match(cond)
            if not m:
                raise ValueError(
                    f"输出规则第 {gi} 组第 {ci} 条无法解析: {cond!r}"
                    "（应为 字段 操作符 值，如 features.reading >= 80）")
            f, op, vtext = m.group(1), m.group(2), m.group(3)
            value = ([_parse_rule_value(x) for x in vtext.split("|")]
                     if op == "in" else _parse_rule_value(vtext))
            conds.append((f, op, value))
        groups.append(conds)
    return groups


def _rule_field(f: Finding, dwell: float, name: str):
    """取字段值；缺字段 → None（该条件视为不通过，适配上游能力差异）。"""
    if name == "conf":
        return f.conf
    if name == "dwell":
        return dwell
    if name == "label":
        return f.label
    if name == "key":
        return f.key
    return (f.features or {}).get(name[9:] if name.startswith("features.") else name)


def _cond_pass(cond: Tuple[str, str, Any], f: Finding, dwell: float) -> bool:
    field, op, value = cond
    a = _rule_field(f, dwell, field)
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
    if op == "contains":
        return str(value) in str(a)
    if op == "in":
        return str(a) in [str(v) for v in value]
    return False


class OutputPolicy:
    """通用输出漏斗：Finding → 七道闸 → AlarmEvent。闸门全可关，默认组合=旧三 kind 行为。

    闸序：①登记(驻留记账) → ②min_dwell 驻留 → ③词表(仅 window) → ④min_conf 置信度
         → ⑥rules 规则 DSL → ⑤滑窗(仅 window，"命中"=通过③④⑥) → ⑦去重冷却。
    kind = 预设档位：window(词表+滑窗+按 action 冷却) / dwell(驻留+按类冷却，
    smooth_frames 旧语义=驻留秒数自动映射 min_dwell) / inspection(全收+同 key 只一次，
    key_cooldown 可改：0=不去重逐条落盘、>0=同 key 冷却后可再报)。
    """

    def __init__(
        self,
        task_id: int,
        kind: str = ALARM_WINDOW,
        target_actions: Optional[Sequence[str]] = None,  # 空 = 巡检（kind 自动 inspection）
        smooth_frames: int = 1,  # 双语义: dwell=驻留秒数阈值; window=滑动窗口长度
        hit_ratio: float = 1.0,  # window: 命中比例
        alarm_cooldown: float = 60.0,
        cooldown_by_action: bool = False,  # window: 是否按警情类型独立冷却
        min_conf: float = 0.0,  # ④ 置信度闸（0=关）
        min_dwell: float = 0.0,  # ② 驻留闸秒数（0=关；dwell 预设未显式给时取 smooth_frames）
        key_cooldown: float = -1.0,  # inspection: -1=同 key 只一次(默认) / 0=不去重 / >0=冷却间隔
        rules: str = "",  # ⑥ 规则 DSL（空=关；语法见 parse_rules）
    ) -> None:
        self.task_id = task_id
        self.target_actions = list(target_actions or [])
        # 巡检模式仅在 VLM 窗口模式下由"无 target_actions"自动派生；
        # dwell（驻留）模式天然无 target_actions，不能误判为巡检
        self.kind = ALARM_INSPECTION if (not self.target_actions and kind == ALARM_WINDOW) else kind
        self.smooth_frames = max(1, int(smooth_frames))
        self.hit_ratio = hit_ratio
        self.alarm_cooldown = alarm_cooldown
        self.cooldown_by_action = cooldown_by_action
        self.min_conf = float(min_conf or 0.0)
        if self.kind == ALARM_DWELL and not float(min_dwell or 0):
            self.min_dwell = float(self.smooth_frames)  # 兼容：dwell 旧语义 smooth_frames=驻留秒数
        else:
            self.min_dwell = float(min_dwell or 0.0)
        self.key_cooldown = -1.0 if key_cooldown is None else float(key_cooldown)
        self._rules = parse_rules(rules)  # 非法表达式在此快速失败
        self._history: Dict[Any, List[str]] = {}
        self._first_seen: Dict[Any, float] = {}
        self._cooldown_until: Dict[Tuple, float] = {}

    def _cooling(self, scope: Tuple, ts: float) -> bool:
        return self._cooldown_until.get(scope, _NEG_INF) > ts

    # ---- 统一入口：漏斗 ----
    def accept(self, f: Finding) -> List[AlarmEvent]:
        # ① 登记（驻留记账：按 key 记首次出现）
        first = self._first_seen.setdefault(f.key, f.ts)
        dwell = max(0.0, f.ts - first)
        # ② 驻留闸
        if self.min_dwell > 0 and dwell < self.min_dwell:
            return []
        # ③ 词表闸（window 专属；inspection/dwell 全收）
        if self.kind == ALARM_WINDOW and f.label not in self.target_actions:
            return []
        # ④ 置信度闸
        if self.min_conf > 0 and (f.conf is None or f.conf < self.min_conf):
            return []
        # ⑥ 规则闸（"命中" = 通过 ③④⑥，供 ⑤ 滑窗计数）
        if self._rules and not any(all(_cond_pass(c, f, dwell) for c in g)
                                   for g in self._rules):
            return []
        # ⑤ 滑窗闸（window 专属）
        if self.kind == ALARM_WINDOW:
            h = self._history.setdefault(f.key, [])
            h.append(f.label)
            if len(h) > self.smooth_frames:
                h.pop(0)
            required = max(1, math.ceil(self.smooth_frames * self.hit_ratio))
            hits = sum(1 for a in h if a in self.target_actions)
            if not (f.label in self.target_actions
                    and len(h) >= self.smooth_frames and hits >= required):
                return []
        # ⑦ 去重冷却闸（scope 规则随 kind 显式指定）
        if self.kind == ALARM_WINDOW:
            scope = ((self.task_id, self.task_id, f.label) if self.cooldown_by_action
                     else (self.task_id, self.task_id))
            cooldown = self.alarm_cooldown
        elif self.kind == ALARM_DWELL:
            scope = (self.task_id, f.label)  # label="dwell:{cls}" ≡ 旧按类别冷却
            cooldown = self.alarm_cooldown
        else:  # inspection：按对象 key（车牌号/巡检对象）
            scope = (self.task_id, "key", f.key)
            cooldown = self.key_cooldown
        if cooldown < 0:  # 同 key 只输出一次
            if scope in self._cooldown_until:
                return []
            self._cooldown_until[scope] = math.inf
        elif cooldown == 0:  # 不去重（巡检逐条落盘）
            pass
        elif self._cooling(scope, f.ts):
            return []
        else:
            self._cooldown_until[scope] = f.ts + cooldown
        action = "inspection" if self.kind == ALARM_INSPECTION else f.label
        desc = (f.features or {}).get("description", "")
        return [AlarmEvent(self.task_id, f.gran, f.key, action, f.ts, desc,
                           conf=f.conf, features=f.features)]

    # ---- 兼容入口（旧调用形态，内部走统一漏斗） ----
    def on_vlm_result(self, track_key: int, action: str, ts: float, description: str = "") -> List[AlarmEvent]:
        gran = GRAN_SCENE if track_key == -1 else GRAN_CLASS
        return self.accept(Finding(key=track_key, label=action, ts=ts, gran=gran,
                                   features=({"description": description} if description else None)))

    def on_dwell(self, track_id: int, cls_id: int, ts: float) -> List[AlarmEvent]:
        return self.accept(Finding(key=track_id, label=f"dwell:{cls_id}", ts=ts,
                                   gran=GRAN_TRACK))


# 兼容别名：历史命名（表/文档/外部调用零改动）
AlarmPolicy = OutputPolicy


# ---------------- 装配器 ----------------
@dataclass
class Pipeline:
    """组装好的分析管道。step() 驱动：帧决策 → YOLO → 目标裁剪 → 报送/告警。

    crop 为可选的"目标裁剪阶段"（鸭子类型：需实现 run(frame, dets, ts) -> list），
    plate 为可选的"车牌识别阶段"（需实现 run(frame, crops, ts) -> list），
    均由拼接层注入（pipe.target_crop.TargetCrop / plate_recog.PlateStage），
    本模块不引入 cv2/onnx 依赖。
    """

    frame: FrameManager
    alarm: AlarmPolicy
    yolo: Optional[YOLODetector] = None
    vlm: Optional[VLMAnalyzer] = None
    infer: Optional[Callable[[dict], str]] = None  # 外部 VLM 推理回调
    crop: Optional[object] = None  # 目标裁剪阶段（TargetCrop.run(frame, dets, ts)）
    annotate: Optional[object] = None  # 标注阶段（Annotator.run(frame, dets, ts) -> 带框图）
    plate: Optional[object] = None  # 车牌识别阶段（PlateStage.run(frame, crops, ts)）
    last_plates: List = field(default_factory=list)  # 最近一帧识别到的车牌（供驱动层读取）
    last_crops: List = field(default_factory=list)  # 最近一帧目标裁剪产物（供驱动层读取）
    last_annotated: object = None  # 最近一帧标注图（供驱动层/VLM 素材读取）
    last_plate_input: List = field(default_factory=list)  # 最近一帧传给车牌阶段的车辆子图（门控后）

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
            dets = self.yolo.infer(frame, track=self.track_needed())

        # 目标裁剪阶段：按上游 YOLO 框裁子图（供车牌识别/人脸检测/VLM 下游消费）。
        # 传递门控沿用 yolo.pass_dets（按识别类型 check_interval，key=类别）。
        self.last_crops = []
        if self.crop is not None and frame is not None and dets is not None:
            gated = self.yolo.pass_dets(dets, ts) if self.yolo is not None else dets
            self.last_crops = list(self.crop.run(frame, gated, ts))

        # 标注阶段：在帧副本上画上游检测框（供 VLM 看"原图 + 画框"）
        self.last_annotated = None
        if self.annotate is not None and frame is not None and dets is not None:
            self.last_annotated = self.annotate.run(frame, dets, ts)

        if self.yolo is not None:
            if self.alarm.kind == ALARM_DWELL and dets is not None:
                # small_only: 无 VLM，过滤后逐 track 驻留告警
                for d in self.yolo.filter_only(dets):
                    alarms += self.alarm.on_dwell(d.track_id, d.cls_id, ts)
            elif self.crop is not None:
                pass  # 目标裁剪已产出，供下游阶段消费
            elif self.plate is not None:
                # yolo→车牌（无目标裁剪的旧形态）：按识别类型间隔统一门控
                if dets is not None:
                    dets = self.yolo.pass_dets(dets, ts)
            elif dets is not None and self.vlm is not None:
                # 报送请求的唯一消费者是 VLM 阶段（门控在 process() 内生效）
                submits = self.yolo.process(frame_idx, ts, dets)
        elif self.vlm is not None and self.alarm.kind != ALARM_DWELL:
            # large_only: 帧决策即送整帧
            submits = [SubmitRequest(-1, GRAN_SCENE, "full:global")]

        # 车牌识别阶段：识别 → 输出漏斗（inspection 默认同车牌号只输出一次，key_cooldown 可调）
        self.last_plates = []
        self.last_plate_input = []
        if self.plate is not None and frame is not None:
            self.last_plate_input = list(self.last_crops)
            self.last_plates = list(self.plate.run(frame, self.last_crops, ts))
            for p in self.last_plates:
                if p.plate_no:
                    alarms += self.alarm.accept(Finding(
                        key=f"plate:{p.plate_no}", label=p.plate_no, ts=ts, conf=p.score,
                        features={"plate_no": p.plate_no, "type_name": p.type_name,
                                  "color": p.color, "bbox": list(p.bbox),
                                  "description": f"{p.plate_no} {p.type_name} {p.score:.2f}"}))
        return submits, alarms

    def track_needed(self) -> bool:
        """是否需要 YOLO 跟踪（dwell 按 track 驻留；车牌阶段按 track 去重）。"""
        return (self.alarm.kind == ALARM_DWELL
                or bool(getattr(self.plate, "use_track", False)))

    def on_vlm_result(self, track_key: int, action: str, ts: float, description: str = "") -> List[AlarmEvent]:
        return self.alarm.on_vlm_result(track_key, action, ts, description)


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
        # 报送粒度固定为"按识别类型"；均送该类全帧(full:cls)、由拼接层画该类框标注。
        # 需送该类 ROI 裁剪时，链路在设计器里 YOLO 之后插入「目标裁剪」阶段。
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_ANALYSIS, frame_skip=extract_rate),
            yolo=YOLODetector(filters=filters,
                              class_limits=spec.get("class_limits", {}),
                              check_interval=spec.get("check_interval", 3.0),
                              **_yolo_kw(spec)),
            vlm=VLMAnalyzer(max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
            alarm=AlarmPolicy(task_id=tid, kind=ALARM_WINDOW, target_actions=target_actions,
                              smooth_frames=smooth,
                              hit_ratio=spec.get("vlm_hit_ratio", 1.0),
                              alarm_cooldown=spec.get("alarm_cooldown", 60.0)),
        )

    if mode == "large_only":
        return Pipeline(
            frame=FrameManager(sampling=SAMPLING_WALL_CLOCK, interval_sec=spec.get("check_interval", 3.0)),
            vlm=VLMAnalyzer(max_resolution=spec.get("vlm_max_resolution", (1280, 720)),
                            prompt=spec.get("vlm_prompt", "")),
            alarm=AlarmPolicy(task_id=tid, kind=ALARM_WINDOW, target_actions=target_actions,
                              smooth_frames=smooth,
                              hit_ratio=spec.get("vlm_hit_ratio", 1.0),
                              alarm_cooldown=spec.get("alarm_cooldown", 60.0)),
        )

    raise ValueError(f"未知模式: {mode}")
