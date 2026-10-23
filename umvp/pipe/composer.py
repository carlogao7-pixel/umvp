# -*- coding: utf-8 -*-
"""
模块化管道拼接器（重构版）：模块 + 装配器

按"参数归属模块"重构:
  1. FrameManager  帧管理 —— 取图节奏(抽帧 / 墙钟秒 / 帧数) + 背压
  2. YOLODetector  小模型识别 —— **纯推理**：ultralytics 参数可调(conf/iou/imgsz/
                     max_det/classes/device)，infer() 输出 dict 记录（kind=det，
                     时间上下文 ts/frame_idx 由 Pipeline.step 注入）。有效性过滤/成组判定
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pipe.det_filter import DetFilter
from pipe.records import collect_events, to_record
from pipe.seq_filter import SeqFilter

_NEG_INF = -1e18  # 首见必送：模拟 spaz 以 unix 时间戳减 0 的语义


# ---------------- 产物端口协议（P5 第 2 步，设计稿 doc/design/assembly_ports.md §3） ----------------
# 产物命名 = {阶段标识}.{产物名}，逐帧登记在 Pipeline.artifacts；值类型三选一：
#   Image（ndarray 原帧/带框图）/ Records（dict 记录列表，链路通用语）/ Vector（向量）。
#   Records 类产物统一为 dict 记录（对象在登记边界经 to_record 归一，WO-0020.2）；
#   驱动层的 last_crops/last_faces/last_plates 等仍为对象形态（读属性，零回归）。
# 模块以类属性 INPUTS / OUTPUTS 自描述端口（类型字符串）；装配器按 stage_inputs 连线解析。
# 这些常量对外导出，供 web 段（spec 解析/校验/设计器）引用，避免字符串散落。
ARTIFACT_FRAME = "frame_manager.frame"            # 本帧原图（Image）
ARTIFACT_DETS = "yolo.dets"                        # YOLO 检出记录列表（Records）
ARTIFACT_DET_FILTER_DETS = "det_filter.dets"       # 检测筛选放行后的 dets（Records）
ARTIFACT_SEQ_FILTER_DETS = "seq_filter.dets"       # 时序判定门控后的 dets（Records）
ARTIFACT_CROPS = "target_crop.crops"               # 裁剪记录列表（元素含 image 键，Records）
ARTIFACT_ANNOTATED = "annotate.annotated"          # 带框图（Image；annotate scope=frame，默认）
ARTIFACT_ANNOTATED_RECORDS = "annotate.annotated_records"  # 逐记录带整帧标注图（Records；annotate scope=record，WO-0021）
ARTIFACT_FACES = "face_detect.faces"               # 人脸检测记录（dict，含 bbox/label/image，Records）
ARTIFACT_PLATES = "plate_recog.plates"             # 车牌识别记录（dict，含 label/text/image，Records）

# 产物名 → 类型（校验连线类型匹配；也是"可用产物清单"的唯一来源）
ARTIFACT_TYPES: Dict[str, str] = {
    ARTIFACT_FRAME: "Image",
    ARTIFACT_DETS: "Records",
    ARTIFACT_DET_FILTER_DETS: "Records",
    ARTIFACT_SEQ_FILTER_DETS: "Records",
    ARTIFACT_CROPS: "Records",
    ARTIFACT_ANNOTATED: "Image",
    ARTIFACT_ANNOTATED_RECORDS: "Records",
    ARTIFACT_FACES: "Records",
    ARTIFACT_PLATES: "Records",
}

# Pipeline.step 已纳入 inputs 解析的端口 —— **唯一来源**，与 step() 中 _port(...) 调用点一一对应。
# {module_id: (端口名, ...)}：web 段（validate_chain 的 DAG 硬校验 / 设计器可连线下拉）据此开放；
# 其余端口（vlm 等）写了 inputs 只给非致命警告、运行按默认取数。改核心时同步此表。
RESOLVED_PORTS: Dict[str, Tuple[str, ...]] = {
    "yolo": ("image",),
    "det_filter": ("dets",),
    "seq_filter": ("dets",),
    "target_crop": ("frame", "dets"),
    "annotate": ("image", "dets"),
    "plate_recog": ("frame", "crops"),
    "face_detect": ("frame", "crops"),
}


class PortResolutionError(ValueError):
    """端口连线解析失败（产物不存在 / 类型不符 / 端口未声明 / 产物名歧义）——fail fast，含可读上下文。"""


# ---------------- 产物全名 / 短名解析（P2-0 节点实例身份，设计稿 doc/design/graph_chain.md §3.1/§3.4） ----------------
# 产物全名 = {module_id}#{position}.{artifact}（例 target_crop#3.crops、yolo#1.dets）；
# 上列 ARTIFACT_* 常量即"短名"（省略 #position），与旧约定同族、迁移直觉自然。
# 短名在本链路该模块只有单实例时等效；多实例必须写全名（**不猜**）。
# 解析逻辑集中在本节（唯一实现）——web 段 validate_chain / server 都复用，不各写一份。
_POSITION_SEP = "#"


def artifact_short(name: str) -> str:
    """全名 {module}#{position}.{artifact} → 短名 {module}.{artifact}（短名幂等返回）。"""
    module_id, sep, tail = name.partition(_POSITION_SEP)
    if not sep:
        return name
    _, _, artifact = tail.partition(".")
    return f"{module_id}.{artifact}"


def artifact_at(artifact: str, position: int) -> str:
    """短名（或全名）+ position → 全名：ARTIFACT_* 常量体系的实例化。

    例：artifact_at(ARTIFACT_CROPS, 3) == "target_crop#3.crops"。
    """
    base = artifact_short(artifact)
    module_id, sep, art = base.partition(".")
    if not module_id or not sep or not art:
        raise PortResolutionError(
            f"非法产物名 {artifact!r}（形如 {{module}}.{{artifact}}）")
    return f"{module_id}{_POSITION_SEP}{int(position)}.{art}"


def artifact_type(name: str) -> str:
    """产物名（短名或全名）→ 类型（Image / Records / …）；未知 → PortResolutionError。

    这是"产物名 → 类型"的唯一解析实现：全名先剥 #position 再查 ARTIFACT_TYPES。
    """
    base = artifact_short(name)
    if base not in ARTIFACT_TYPES:
        raise PortResolutionError(
            f"未知产物 {name!r}（可用产物: {sorted(ARTIFACT_TYPES)}）")
    return ARTIFACT_TYPES[base]


def resolve_artifact(name: str, positions_by_module: Mapping[str, Sequence[int]]) -> str:
    """短名 {module}.{artifact} → 全名 {module}#{position}.{artifact}。

    - 已是全名 → 原样返回（校验其存在：模块与位置须在 positions_by_module 中）
    - 该模块在本链路只有一个实例 → 返回其全名
    - 多实例（歧义）→ 抛 PortResolutionError（要求写全名，不猜）
    - 未知模块 / 空名 / 非法名 → 抛 PortResolutionError

    positions_by_module 形如 {"yolo": [1], "det_filter": [2, 5]}（spec.stages 下标）。
    """
    if not isinstance(name, str) or not name.strip():
        raise PortResolutionError("产物名为空，无法解析")
    name = name.strip()
    if _POSITION_SEP in name:
        module_id, _, tail = name.partition(_POSITION_SEP)
        pos_str, dot, artifact = tail.partition(".")
        if not module_id or not pos_str.isdigit() or not dot or not artifact:
            raise PortResolutionError(
                f"非法全名产物 {name!r}（形如 {{module}}#{{position}}.{{artifact}}）")
        position = int(pos_str)
        positions = positions_by_module.get(module_id) or ()
        if position not in positions:
            raise PortResolutionError(
                f"全名产物 {name!r} 的模块/位置不在本链路"
                f"（已知: { {m: list(p) for m, p in positions_by_module.items()} }）")
        return name
    module_id, dot, artifact = name.partition(".")
    if not module_id or not dot or not artifact:
        raise PortResolutionError(
            f"非法产物名 {name!r}（短名形如 {{module}}.{{artifact}}）")
    positions = positions_by_module.get(module_id)
    if not positions:
        raise PortResolutionError(
            f"未知模块 {module_id!r}（本链路模块: {sorted(positions_by_module)}）")
    if len(positions) > 1:
        raise PortResolutionError(
            f"模块 {module_id!r} 在本链路有多个实例 {list(positions)}，"
            f"短名 {name!r} 有歧义，请写全名 {{module}}#{{position}}.{{artifact}}")
    return f"{module_id}{_POSITION_SEP}{positions[0]}.{artifact}"


# 阶段实例 → 模块 id（供按位置装配的 `stages` 容器归类实例；仅真实产品模块类）。
_CLASS_TO_MODULE: Dict[str, str] = {
    "FrameManager": "frame_manager",
    "YOLODetector": "yolo",
    "VLMAnalyzer": "vlm",
    "DetFilter": "det_filter",
    "SeqFilter": "seq_filter",
    "TargetCrop": "target_crop",
    "Annotator": "annotate",
    "FaceStage": "face_detect",
    "PlateStage": "plate_recog",
}


def _module_of(obj) -> Optional[str]:
    """阶段实例 → 模块 id（按类名；非产品模块实例如 FaceEmbedder 返回 None）。"""
    return _CLASS_TO_MODULE.get(type(obj).__name__)

# ---------------- 通用数据 ----------------
@dataclass
class Detection:
    """一次 YOLO 检测（含跟踪 id）——与 spaz producer 对齐。

    **兼容形态**：保留供外部构造 / 直传（老调用、test_image_source 等导入依赖）；
    链路通用产物已是 dict 记录（Record，kind=det）——`infer()` 直接产 dict，
    直传的 Detection 在 `Pipeline.step` 入口经 `to_record` 归一为 dict。
    """

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
    det: Optional[dict] = None  # 保留字段（单目标素材用）；按类报送时为空；双形态读（过渡期兼容 Detection）
    dets: List[dict] = field(default_factory=list)  # 该类全部有效检测（dict 记录；large_only 为空）


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

    # 产物端口声明（自描述）：帧管理是链路源头，不消费上游产物，只产出本帧原图。
    # 注：FrameManager 是 dataclass，端口声明用**无注解**类属性（不会成为 dataclass 字段）。
    INPUTS = {}
    OUTPUTS = {"frame": "Image"}

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

    # 产物端口声明：吃一张图（原帧），产检测记录列表。
    INPUTS = {"image": "Image"}
    OUTPUTS = {"dets": "Records"}

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
    def infer(self, frame, track: bool = False) -> List[dict]:
        """跑 YOLO 推理，返回 dict 记录列表（kind=det；track=True 用跟踪器保持 track_id 稳定）。

        输出为**纯推理结果**（不做有效性过滤）；有效性（置信/尺寸/成组数量）
        由链上的检测筛选阶段（Pipeline.det_filter）判定，对全部下游一致生效。
        记录形态：{"kind": "det", "track_id", "cls_id", "bbox"(x1,y1,x2,y2)tuple,
        "score", "model_path"}；不含 ts/frame_idx——时间上下文由调用方注入
        （Pipeline.step 统一补，契约见 pipe.records）。
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
        dets: List[dict] = []
        if r.boxes is None:
            return dets
        for i, b in enumerate(r.boxes.xyxy.tolist()):
            tid = r.boxes.id[i].item() if (track and r.boxes.id is not None) else 0
            dets.append({"kind": "det",
                         "track_id": int(tid),
                         "cls_id": int(r.boxes.cls[i].item()),
                         "bbox": tuple(int(v) for v in b),
                         "score": float(r.boxes.conf[i].item()),
                         "model_path": self.model_path.rsplit("/", 1)[-1]})
        return dets


# ---------------- 模块 3: VLM 研判 ----------------
class VLMAnalyzer:
    """素材准备 + 推理。prepare 产出 payload；推理由外部回调注入（真实场景为 spaz consumer）。"""

    # 产物端口声明：吃一张送审图（默认带框图）+ 检测记录；研判内容经 publish 走事件尾，
    # 不在 Pipeline.step 内登记为产物（故 OUTPUTS 为空——见回报「本期未纳入」说明）。
    INPUTS = {"image": "Image", "dets": "Records"}
    OUTPUTS = {}

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
            "bbox": (req.det.get("bbox") if isinstance(req.det, Mapping)
                     else getattr(req.det, "bbox", None)) if req.det else None,
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
    face 为可选的"人脸识别阶段"（鸭子类型：需实现 run(frame, crops, ts) -> list），
    均由拼接层注入（pipe.target_crop.TargetCrop / plate_recog.PlateStage / face_detect.FaceStage），
    本模块不引入 cv2/onnx 依赖。链路输出 = 事件尾（event_filter）产出的事件（last_events）。
    """

    frame: FrameManager
    yolo: Optional[YOLODetector] = None
    vlm: Optional[VLMAnalyzer] = None
    infer: Optional[Callable[[dict], str]] = None  # 外部 VLM 推理回调
    crop: Optional[object] = None  # 目标裁剪阶段（TargetCrop.run(frame, dets, ts)）
    annotate: Optional[object] = None  # 标注阶段（Annotator.run(frame, dets, ts) -> 带框图）
    plate: Optional[object] = None  # 车牌识别阶段（PlateStage.run(frame, crops, ts)）
    face: Optional[object] = None  # 人脸识别阶段（FaceStage.run(frame, crops, ts)）
    det_filter: Optional[DetFilter] = None  # 检测筛选阶段（有效性：逐条+成组，2026-10 自 YOLO 迁入）
    seq_filter: Optional[object] = None  # 时序判定阶段（节流/变化门控 + 事件，2026-10 自 Pipeline 迁入）
    event_filter: Optional[object] = None  # 事件尾（VLM/车牌等外部记录经 publish 走它出事件，可作链路尾）
    # ---- 同模块多实例（P2-0 节点实例身份，设计稿 §3.1）：list[(position, obj)]，按 position 升序执行 ----
    # 每类同模块阶段可多实例并存、各自独立生效（产物全名 {module}#{position}.{artifact} 互不覆盖）。
    # 旧单插槽字段（yolo/crop/...）保留读取兼容：单插槽实例 position 取 stage_positions（未声明=None）。
    det_filters: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, DetFilter), ...]
    seq_filters: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, SeqFilter), ...]
    target_crops: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, TargetCrop), ...]
    annotators: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, Annotator), ...]
    plate_stages: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, PlateStage), ...]
    face_stages: List[Tuple[int, Any]] = field(default_factory=list)  # [(位置, FaceStage), ...]
    # stage_positions: {module_id: position（或位置列表）}——实例位置声明（供 stage_inputs 的
    #           "#<position>" 键与短名解析；缺省空 = 单插槽仅按旧 module_id 键解析，零回归）。
    stage_positions: Dict[str, Any] = field(default_factory=dict)
    # stages: 按位置装配的实例容器 {position: 阶段实例}（web 段 `_chain_from_spec` 用；
    #           实例优先级：多实例列表 > stages > 单插槽字段，按位置升序驱动执行）。
    stages: Dict[int, Any] = field(default_factory=dict)
    last_plates: List = field(default_factory=list)  # 最近一帧识别到的车牌（供驱动层读取）
    last_faces: List = field(default_factory=list)  # 最近一帧检测到的人脸（供驱动层读取）
    last_crops: List = field(default_factory=list)  # 最近一帧目标裁剪产物（供驱动层读取）
    last_annotated: object = None  # 最近一帧标注图（供驱动层/VLM 素材读取）
    last_plate_input: List = field(default_factory=list)  # 最近一帧传给车牌阶段的车辆子图（门控后）
    last_face_input: List = field(default_factory=list)  # 最近一帧传给人脸阶段的子图（门控后）
    last_filter: Optional[Tuple[int, int]] = None  # 最近一帧检测筛选统计 (进入, 放行)（供时间线展示；未运行=None）
    last_events: List = field(default_factory=list)  # 最近一帧末段记录携带的事件（链路尾通用输出，见 records.events）
    # ---- 产物端口协议（P5 第 2 步；P2-0 扩展节点实例身份） ----
    # artifacts: 逐帧产物表 {产物名: 值}，step() 每帧开始时重建、各阶段执行后登记；
    #           有位置实例 → 全名 {module_id}#{position}.{artifact}；单实例另登记短名别名（兼容旧读法）。
    # stage_inputs: 拼接层注入的连线映射 {位置串 "#<position>" 或 module_id(兼容): {端口名: 上游产物名}}；
    #           缺省空 = 完全按现状规则（零回归）；声明则按产物名解析该端口的值。
    artifacts: Dict[str, Any] = field(default_factory=dict)
    stage_inputs: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # ---------------- 多实例辅助 ----------------
    def _pos_of(self, module_id: str) -> Optional[int]:
        """单插槽实例的声明位置：stage_positions 值兼容 int 与位置序列（多实例取首个）。"""
        v = (self.stage_positions or {}).get(module_id)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return int(v[0]) if v else None
        return int(v)

    def _instances(self, module_id: str, single, plural) -> List[Tuple[Optional[int], Any]]:
        """产出某模块的 (position, obj) 序列（位置升序）。

        基础序：显式多实例列表 `plural` → 位置容器 `stages`（按类名归类）→ 旧单插槽 `single`。
        **显式单插槽覆盖**：若基础序非空且 `single` 已设（含装配后被替换的实例，如测试间谍
        `pipe.annotate = spy`），用 `single` 替换其声明位置的实例，其余位置保留——既支持多实例，
        又兼容装配后替换单槽字段的既有用法。
        """
        base: List[Tuple[int, Any]] = []
        if plural:
            base = [(int(pos), obj) for pos, obj in sorted(plural, key=lambda t: t[0])]
        elif self.stages:
            base = [(int(pos), obj) for pos, obj in sorted(self.stages.items())
                    if obj is not None and _module_of(obj) == module_id]
        if base:
            if single is not None:
                pos = self._pos_of(module_id)
                if pos is None:
                    pos = base[0][0]
                base = [(p, single if p == pos else obj) for p, obj in base]
            return base
        if single is not None:
            return [(self._pos_of(module_id), single)]
        return []

    def _stage_instances(self, module_id: str) -> List[Tuple[Optional[int], Any]]:
        """按模块 id 取实例序列（统一入口，供 step() / 位置表使用）。"""
        pairs = {
            "frame_manager": (self.frame, None),
            "yolo": (self.yolo, None),
            "vlm": (self.vlm, None),
            "det_filter": (self.det_filter, self.det_filters),
            "seq_filter": (self.seq_filter, self.seq_filters),
            "target_crop": (self.crop, self.target_crops),
            "annotate": (self.annotate, self.annotators),
            "plate_recog": (self.plate, self.plate_stages),
            "face_detect": (self.face, self.face_stages),
        }
        single, plural = pairs[module_id]
        return self._instances(module_id, single, plural)

    def _register(self, module_id: str, position: Optional[int], artifact: str,
                  value, count: int) -> None:
        """登记产物：有 position → 全名；本模块仅一个实例时另登记短名别名（兼容旧读法）。

        count = 本模块本链路实例数（>1 时不登记短名，避免歧义覆盖）。
        """
        if position is not None:
            self.artifacts[f"{module_id}{_POSITION_SEP}{position}.{artifact}"] = value
        if count <= 1:
            self.artifacts[f"{module_id}.{artifact}"] = value

    def _positions_by_module(self) -> Dict[str, List[int]]:
        """{module_id: [position, ...]}——本链路**已知位置**的实例（供短名解析 / 歧义判定）。

        只有位置已知的实例（stages 容器 / 多实例列表 / 单插槽 + stage_positions）才登记；
        位置未知的旧单插槽实例不进表（其产物以短名别名存在 artifacts，短名可直接命中）。
        """
        out: Dict[str, List[int]] = {}
        for mid in _CLASS_TO_MODULE.values():
            positions = [p for p, _ in self._stage_instances(mid) if p is not None]
            if positions:
                out[mid] = positions
        return out

    def _submits(self, dets: List[dict]) -> List[SubmitRequest]:
        """按识别类型（类别）构造报送请求（有效性/门控已由 det_filter/seq_filter 完成）。

        只分组 + 造语料：一类目标再多也只算一条请求（ref=full:cls{cls}）。何时报、
        报不报由链上 det_filter（有效性）与 seq_filter（节流/变化门控）决定。
        元素为 dict 记录；缺 cls_id（-1）的项跳过（与 web `_as_detections`
        「无 cls_id 跳过」语义对齐）。
        """
        by_cls: Dict[int, List[dict]] = {}
        for d in dets:
            cls = int(d.get("cls_id", -1))
            if cls < 0:
                continue
            by_cls.setdefault(cls, []).append(d)
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

    def _port(self, module_id: str, module, port: str, default):
        """旧签名兼容：等价 `_port_at(位置未知, module_id, module, port, default)`。"""
        return self._port_at(self._pos_of(module_id), module_id, module, port, default)

    def _port_at(self, position: Optional[int], module_id: str, module,
                 port: str, default):
        """解析一个输入端口的值：stage_inputs 未声明该端口 → 返回 default（现状规则，零回归）。

        连线声明键优先位置串 `"#<position>"`（P2-0），回退旧 `module_id` 键（兼容）。
        声明了则取 artifacts[上游产物名]，并做四项 fail fast 校验（抛 PortResolutionError）：
          ① 产物名（短名/全名）不在 ARTIFACT_TYPES（未知产物）；② 端口不在模块 INPUTS 声明中；
          ③ 产物类型与端口声明类型不符；④ 产物本帧未登记（不在 artifacts，如上游阶段缺失）。
        产物名解析：全名直取；短名先命名单实例别名，否则按位置解析（多实例歧义 → 报错要求写全名）。
        端口单一上游（用户决策 2）；同一产物可被多个下游引用（分支留路）。
        """
        si = self.stage_inputs or {}
        mapping = None
        if position is not None:
            mapping = si.get(f"{_POSITION_SEP}{position}")
        if not mapping:
            mapping = si.get(module_id)
        if not mapping or port not in mapping:
            return default
        art_name = mapping[port]
        base = artifact_short(art_name)
        decl = getattr(module, "INPUTS", None) or {}
        if base not in ARTIFACT_TYPES:
            raise PortResolutionError(
                f"阶段 {module_id!r} 端口 {port!r} 引用了未知产物 {art_name!r}"
                f"（可用产物: {sorted(ARTIFACT_TYPES)}）")
        if decl and port not in decl:
            raise PortResolutionError(
                f"阶段 {module_id!r} 端口 {port!r} 未在其 INPUTS 声明中"
                f"（该模块已声明端口: {sorted(decl)}）")
        want = decl.get(port)
        if want and ARTIFACT_TYPES[base] != want:
            raise PortResolutionError(
                f"阶段 {module_id!r} 端口 {port!r} 类型不符：上游产物 {art_name!r} 为 "
                f"{ARTIFACT_TYPES[base]}，端口声明为 {want}")
        full = art_name
        if full not in self.artifacts and _POSITION_SEP not in art_name:
            # 短名未命中别名（多实例 / 未知模块）→ 尝试按位置解析（歧义在此报错）
            positions = self._positions_by_module()
            if art_name.split(".", 1)[0] in positions:
                full = resolve_artifact(art_name, positions)
        if full not in self.artifacts:
            raise PortResolutionError(
                f"阶段 {module_id!r} 端口 {port!r} 请求的产物 {art_name!r} 不在本帧产物表中"
                f"（可用产物: {sorted(self.artifacts)}）")
        return self.artifacts[full]

    def step(self, frame_idx: int, ts: float, dets: Optional[List[dict]] = None,
             frame=None, force: bool = False):
        """处理一帧。返回 (submits, events)。
        dets 由拼接层在 wants_frame 为真时做 YOLO 推理提供（无 YOLO 的模式传 None）；
        直传入口逐条归一为 dict 记录（对象如 Detection 经 to_record 转换，Mapping 浅拷贝），
        并统一注入时间上下文 ts/frame_idx（契约：时间由调用方注入，见 pipe.records）——
        链路通用形态自此为 dict Record（yolo.dets 产物元素带全契约字段 kind/ts/frame_idx）。
        不传 dets 但配置了 yolo 时，传 frame 由模块自推理（车牌按 track 去重时 track=True）。
        force=True 时跳过帧决策检查——供调用方已先行 wants_frame() 门控避免重复采样。

        每帧开头重建产物表 artifacts，各阶段执行后登记其产物；各阶段输入端口在
        stage_inputs 有声明时按产物名解析（默认空 = 现状规则）。
        """
        submits: List[SubmitRequest] = []
        self.artifacts = {}
        if not force and not self.frame.wants_frame(frame_idx, ts):
            return submits, []
        fm_pos = next((p for p, _ in self._stage_instances("frame_manager")), None)
        if frame is not None:
            # frame_manager.frame（本帧原图）；有声明位置时登记全名 + 短名别名
            self._register("frame_manager", fm_pos, "frame", frame, 1)
        yolo_insts = self._stage_instances("yolo")
        yolo_pos, yolo = (yolo_insts[0] if yolo_insts else (None, None))
        if dets is None and yolo is not None:
            yolo_img = self._port_at(yolo_pos, "yolo", yolo, "image", frame)
            if yolo_img is not None:
                dets = yolo.infer(yolo_img, track=self.track_needed())
        if dets is not None:
            # 出口/入口归一：对象（Detection 等）→ dict 记录（to_record 全字段覆盖），
            # dict 浅拷贝防上游原地改动；补缺省 kind 与时间上下文（infer 已带的值不覆盖）。
            dets = [to_record(d) for d in dets]
            for d in dets:
                d.setdefault("kind", "det")
                d.setdefault("ts", ts)
                d.setdefault("frame_idx", frame_idx)
        if yolo is not None and dets is not None:
            # yolo.dets（纯推理原始检出，过滤前；元素为带全契约字段的 dict 记录）
            self._register("yolo", yolo_pos, "dets", dets, len(yolo_insts))

        # 检测筛选阶段：有效性判定（逐条条件 + 成组数量；原 YOLO 输出过滤链）——
        # 不达标的检测视为"未识别"，对**所有**下游（裁剪/标注/报送/事件）一致生效。
        # 多实例按位置升序执行、链式传递 dets（各自产物全名路径互不覆盖）。
        self.last_filter = None
        df_insts = self._stage_instances("det_filter")
        if df_insts:
            cur = dets if dets is not None else []
            ran = bool(dets)
            first_in = None
            for pos, df in df_insts:
                if ran:
                    df_in = self._port_at(pos, "det_filter", df, "dets", cur)
                    if first_in is None:
                        first_in = len(df_in)
                    cur = df.run(df_in, ts=ts, frame_idx=frame_idx)
                self._register("det_filter", pos, "dets", cur, len(df_insts))
            if ran:
                dets = cur
                self.last_filter = (first_in, len(cur))

        # 时序判定阶段：节流/变化门控（原 Pipeline check_interval/track_change_only）——
        # 门控下游可见的检测集；多实例按位置升序链式执行。
        sf_insts = self._stage_instances("seq_filter")
        if sf_insts:
            cur = dets if dets is not None else []
            ran = bool(dets)
            for pos, sf in sf_insts:
                if ran:
                    sf_in = self._port_at(pos, "seq_filter", sf, "dets", cur)
                    cur = sf.run(sf_in, ts=ts, frame_idx=frame_idx)
                self._register("seq_filter", pos, "dets", cur, len(sf_insts))
            if ran:
                dets = cur

        # 目标裁剪阶段：按上游检测框裁子图（供车牌识别/人脸检测/VLM 下游消费）。
        self.last_crops = []
        crop_insts = self._stage_instances("target_crop")
        if crop_insts:
            cur_crops = []
            for pos, cr in crop_insts:
                if frame is not None and dets is not None:
                    crop_frame = self._port_at(pos, "target_crop", cr, "frame", frame)
                    crop_dets = self._port_at(pos, "target_crop", cr, "dets", dets)
                    cur_crops = list(cr.run(crop_frame, crop_dets, ts))
                # 裁剪记录列表：转记录后元素含统一旁路键 image（旧名 img 保留为兼容别名）
                self._register("target_crop", pos, "crops",
                               [to_record(c) for c in cur_crops], len(crop_insts))
            self.last_crops = cur_crops

        vlm_insts = self._stage_instances("vlm")
        vlm = vlm_insts[0][1] if vlm_insts else None
        if yolo is not None:
            if dets is not None and vlm is not None:
                # 报送请求的唯一消费者是 VLM 阶段（门控已在上游 det_filter/seq_filter 生效）。
                # 目标裁剪阶段（crop）与 VLM 可共存：crops 供下游阶段/驱动层取裁剪子图
                # （image_source=crops），同时仍按检测按类报送 VLM（_submits 语义与门控不变）；
                # 仅有 crop、无 vlm 时不产生 submits（车牌/人脸链路既有行为不变）。
                submits = self._submits(dets)
        elif vlm is not None:
            # large_only: 帧决策即送整帧
            submits = [SubmitRequest(-1, GRAN_SCENE, "full:global")]

        # 车牌识别阶段：识别 → 事件尾（event_filter 经 publish 出事件，幸存者=告警载体）
        self.last_plates = []
        self.last_plate_input = []
        plate_events: List[dict] = []
        plate_insts = self._stage_instances("plate_recog")
        for pos, pl in plate_insts:
            if frame is None:
                continue
            plate_frame = self._port_at(pos, "plate_recog", pl, "frame", frame)
            plate_crops = self._port_at(pos, "plate_recog", pl, "crops", self.last_crops)
            self.last_plate_input = list(plate_crops)
            self.last_plates = list(pl.run(plate_frame, plate_crops, ts))
            # 产物归一（WO-0020.2）：PlateResult 对象 → dict 记录（adapter 补 kind/label/text，
            # 见 plate_recognizer._plate_record）+ 注入时间上下文；last_plates 保持对象形态
            # 供驱动层读属性（零回归）。
            plate_recs = []
            for p in self.last_plates:
                r = to_record(p)
                r.setdefault("ts", ts)
                r.setdefault("frame_idx", frame_idx)
                plate_recs.append(r)
            self._register("plate_recog", pos, "plates", plate_recs, len(plate_insts))
            for p in self.last_plates:
                if not p.plate_no:
                    continue
                rec = {"kind": "plate", "label": p.plate_no, "score": p.score,
                       "content": {"plate_no": p.plate_no, "type_name": p.type_name,
                                   "color": p.color, "bbox": list(p.bbox)}}
                plate_events += self.publish([rec], ts, frame_idx)
        # 人脸识别阶段：检测 → 人脸产物（同源于 crop 产物）；落盘/计数/事件尾留驱动层。
        self.last_faces = []
        self.last_face_input = []
        face_insts = self._stage_instances("face_detect")
        for pos, fc in face_insts:
            if frame is None:
                continue
            face_frame = self._port_at(pos, "face_detect", fc, "frame", frame)
            face_crops = self._port_at(pos, "face_detect", fc, "crops", self.last_crops)
            self.last_face_input = list(face_crops)
            self.last_faces = list(fc.run(face_frame, face_crops, ts))
            # 产物归一（WO-0020.2）：FaceResult 对象 → dict 记录（adapter 补 kind/bbox/label，
            # 见 face_pipeline._face_record；bbox=face_bbox 原图坐标）+ 注入时间上下文；
            # last_faces 保持对象形态供驱动层读属性（零回归）。产物表里的记录从此可被
            # 下游阶段（annotate 画人脸框等）统一按 bbox 消费。
            face_recs = []
            for f in self.last_faces:
                r = to_record(f)
                r.setdefault("ts", ts)
                r.setdefault("frame_idx", frame_idx)
                face_recs.append(r)
            self._register("face_detect", pos, "faces", face_recs, len(face_insts))
        # 标注阶段：在帧副本上画上游检测框（供 VLM 看"原图 + 画框"）。多实例链式：下一实例
        # 默认 image 取上一实例标注图（单实例默认仍为本帧原图，零回归）。
        # 位置在 plate_recog/face_detect 之后（呈现后于识别）：标注可经 inputs 连线消费
        # 人脸/车牌产物画框（如 face_detect.faces），否则连线产物不在本帧产物表。
        # 双模式（WO-0021，设计稿 assembly_ports.md §9）：scope=frame（默认）产帧级
        # annotated（Image）；scope=record 产逐记录带整帧图流 annotated_records（Records）
        # ——仅登记产物、主干 dets 不动（最小惊讶：与「标注不动主干」现状语义一致），
        # 下游消费显式连线取；record 实例不更新 cur_img/last_annotated（无单一标注图）。
        self.last_annotated = None
        anno_insts = self._stage_instances("annotate")
        if anno_insts and frame is not None and dets is not None:
            cur_img = frame
            last_anno = None
            for pos, an in anno_insts:
                anno_img = self._port_at(pos, "annotate", an, "image", cur_img)
                anno_dets = self._port_at(pos, "annotate", an, "dets", dets)
                out = an.run(anno_img, anno_dets, ts)
                if getattr(an, "scope", "frame") == "record":
                    if out:
                        self._register("annotate", pos, "annotated_records",
                                       out, len(anno_insts))
                    continue  # 主干与 cur_img/last_annotated 均不动
                cur_img = out
                last_anno = cur_img
                if cur_img is not None:
                    self._register("annotate", pos, "annotated", cur_img, len(anno_insts))
            self.last_annotated = last_anno
        # 链路尾通用输出：末段检测记录携带的事件（det_filter/seq_filter 等模块追加）+
        # 车牌事件尾产出。通用、不绑定模块——供驱动层读取。
        self.last_events = (collect_events(dets) if dets else []) + plate_events
        return submits, self.last_events

    def track_needed(self) -> bool:
        """是否需要 YOLO 跟踪（车牌阶段按 track 去重时需要）。"""
        if any(getattr(p, "use_track", False) for _, p in self._stage_instances("plate_recog")):
            return True
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
