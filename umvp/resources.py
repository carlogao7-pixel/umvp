"""资源估算核心模块（P1）。

两层估算：
- 静态法（static）：不加载模型，按模型文件大小/参数量 + 输入分辨率经验公式秒级粗估；
- 实测法（measured）：读 `out/fingerprints.json`（由 web_lab/calibrate.py 生成），
  真实推理标定值优先于静态公式。

聚合规则：链路 = 各模块按"每帧平均成本"聚合（不是简单相加）——
例如 VLM 受 check_interval 节流（每 N 秒才送审一次）、人脸嵌入只在检测命中后跑，
这些模块的成本要按占比折算进每帧均值。

多路外推：模型常驻权重/显存部分随路数共享（取一次），激活/图/推理并发部分随路数线性增长；
超过预算给出 WARNING 并建议 frame_skip。

用法（代码内）：
    est = ResourceEstimator(project_root=".")
    fp = est.estimate_module("yolo", {"imgsz": 640})
    pe = est.estimate_pipeline(spec, streams=16, vram_budget_mb=8192)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 数据结构（与需求文档-开发版 §3.2 一致）
# ---------------------------------------------------------------------------


@dataclass
class ResourceFingerprint:
    """单个模块（在某设备/分辨率条件下）的资源指纹。"""

    vram_mb: float = 0.0          # 显存峰值（含权重+激活+图）
    ram_mb: float = 0.0           # 内存占用（不含权重落盘的进程共享部分）
    latency_ms_cpu: float = 0.0   # 单次推理/处理 CPU 耗时（ms）
    latency_ms_gpu: float = 0.0   # 单次推理/处理 GPU 耗时（ms）
    note: str = ""                # 条件说明（分辨率、batch、设备）
    source: str = "static"        # static=公式估 / measured=实测标定

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram_mb": round(self.vram_mb, 1),
            "ram_mb": round(self.ram_mb, 1),
            "latency_ms_cpu": round(self.latency_ms_cpu, 1),
            "latency_ms_gpu": round(self.latency_ms_gpu, 1),
            "note": self.note,
            "source": self.source,
        }


@dataclass
class PipelineEstimate:
    """整条链路的聚合估算（含多路推演）。"""

    per_module: dict[str, ResourceFingerprint] = field(default_factory=dict)
    vram_mb: float = 0.0                      # 单路显存（模型常驻取 max + 激活路径 max）
    ram_mb: float = 0.0                       # 单路内存（每帧平均成本聚合）
    latency_per_frame_ms: dict[str, float] = field(default_factory=dict)  # {"cpu":…, "gpu":…}
    fps_cpu: float = 0.0
    fps_gpu: float = 0.0
    streams: int = 1
    vram_mb_N_streams: float = 0.0            # 多路总显存
    warning: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_module": {k: v.to_dict() for k, v in self.per_module.items()},
            "vram_mb": round(self.vram_mb, 1),
            "ram_mb": round(self.ram_mb, 1),
            "latency_per_frame_ms": {k: round(v, 1) for k, v in self.latency_per_frame_ms.items()},
            "fps_cpu": round(self.fps_cpu, 1),
            "fps_gpu": round(self.fps_gpu, 1),
            "streams": self.streams,
            "vram_mb_N_streams": round(self.vram_mb_N_streams, 1),
            "warning": self.warning,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# 静态估算的经验公式与基线（v1 简化，误差以实测为准）
# ---------------------------------------------------------------------------

# 模型权重文件 → 静态基线（从 models/ 目录实读文件大小自动算，这里只放"每类模型"的
# 附加激活/图开销经验系数）。
#
# 激活内存经验公式（MB）≈ 2（fp16 激活 2 字节）× 每像素张量数 × 分辨率像素 × 系数。
# 对 YOLO 一阶段检测器，激活按"主干特征图 + 头部"粗估，系数取 0.02 MB / 万像素，偏保守。


def _model_file_mb(path: Optional[str]) -> float:
    """模型文件大小（MB）；空则回退默认 yolov8n.pt 的静态值。"""
    if path:
        p = Path(path)
        if p.exists():
            return p.stat().st_size / (1024 * 1024)
    # 常见模型名 → 静态大小表（找不到文件时的兜底，避免估算失败）
    _KNOWN = {
        "yolov8n.pt": 6.5, "yolov8s.pt": 22.6, "yolov9t.pt": 4.9,
        "yolov10s.pt": 16.6, "yolo11n.pt": 5.6, "yolo11s.pt": 19.3,
        "yolo11m.pt": 40.7,
    }
    if path:
        name = Path(path).name
        if name in _KNOWN:
            return _KNOWN[name]
    return 6.5  # 默认按 yolov8n


def _yolo_activation_mb(imgsz: int) -> float:
    """YOLO 单帧激活内存（MB）经验公式。imgsz 分辨率越大激活越大。"""
    pixels = imgsz * imgsz
    # 实测基准：640 → ~90MB 激活（fp16）。按面积线性外推。
    return 90.0 * pixels / (640 * 640)


def _scrfd_activation_mb(det_size: str = "640,640") -> float:
    w, h = _parse_wh(det_size, 640, 640)
    pixels = w * h
    return 120.0 * pixels / (640 * 640)


def _merge_source(a: str, b: str) -> str:
    """两个子模块估算来源合并：全实测→measured，全静态→static，否则 mixed。"""
    if a == "measured" and b == "measured":
        return "measured"
    if a == "static" and b == "static":
        return "static"
    return "mixed"


def _parse_wh(s: str, dw: int, dh: int) -> tuple[int, int]:
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) >= 2:
            return parts[0], parts[1]
    except (ValueError, AttributeError):
        pass
    return dw, dh


# 静态耗时基线（CPU，RTX 3060 基准机；GPU 基线单独在 _STATIC 里）
_STATIC_LATENCY_CPU_MS = {
    "yolo": 1700.0,          # YOLO 单帧 CPU（CLAUDE.md 基线）
    "scrfd": 110.0,          # SCRFD 640 CPU 90–130ms，取中值
    "arcface": 10.0,         # ArcFace 单张 CPU
    "facestore_query": 1.0,  # 千级底库检索 <1ms
    "vlm": 0.0,              # 本地只做素材准备；真实推理=远端延迟，标注依赖实测
    "logic": 0.0,            # 帧管理/告警策略，纯逻辑
}

# GPU 基线：CLAUDE.md 实测 YOLO GPU ~0.02s/帧；SCRFD/ArcFace 按 CPU 的 8~20 倍加速粗估
_STATIC_LATENCY_GPU_MS = {
    "yolo": 20.0,
    "scrfd": 8.0,
    "arcface": 1.0,
    "facestore_query": 0.2,
    "vlm": 0.0,
    "logic": 0.0,
}

# 显存基线（MB）：权重(fp16) + 图。激活部分按分辨率公式另加。
_STATIC_VRAM_MB = {
    "scrfd": 200.0,   # det_10g.onnx（~21MB 权重）+ 图 + 常量
    "arcface": 150.0,  # w600k_r50.onnx（~270MB fp32 权重，fp16 减半）+ 图
    "vlm": 0.0,       # 远端推理，本地只做素材准备
    "logic": 0.0,
}


# ---------------------------------------------------------------------------
# 核心估算器
# ---------------------------------------------------------------------------


class ResourceEstimator:
    """两层资源估算器。

    构造：
        ResourceEstimator(project_root=".")  # 自动读 out/fingerprints.json（若存在）

    主要方法：
        estimate_module(module_id, params) -> ResourceFingerprint
        estimate_pipeline(spec, streams=1, vram_budget_mb=None) -> PipelineEstimate
        back_calculate(...) -> dict  # 预算反推建议
    """

    # 模块类型 → 资源类型映射（决定用哪条估算路径）
    TYPE_YOLO = "yolo"
    TYPE_SCRFD = "scrfd"
    TYPE_ARCFACE = "arcface"
    TYPE_FACESTORE = "facestore"
    TYPE_VLM = "vlm"
    TYPE_LOGIC = "logic"

    # web_lab 模块 id → 资源类型
    MODULE_TYPE = {
        "frame_manager": TYPE_LOGIC,
        "yolo": TYPE_YOLO,
        "target_crop": TYPE_LOGIC,     # 目标裁剪：纯裁剪，无模型开销
        "vlm": TYPE_VLM,
        "alarm": TYPE_LOGIC,
        "compose": TYPE_YOLO,          # compose 含 YOLO 主推理
        "face_detect": TYPE_SCRFD,
        "face_embed": TYPE_ARCFACE,
        "face_store": TYPE_FACESTORE,
        "face_monitor": TYPE_SCRFD,    # 人脸监控 = SCRFD 主推理
    }

    def __init__(self, project_root: str | Path = ".", fingerprints: Optional[dict] = None):
        self.root = Path(project_root)
        self.fingerprints: dict[str, dict] = {}
        if fingerprints is not None:
            self.fingerprints = fingerprints
        else:
            fp_path = self.root / "out" / "fingerprints.json"
            if fp_path.exists():
                try:
                    self.fingerprints = json.loads(fp_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    self.fingerprints = {}

    # ------------------------------------------------------------------ 实测
    def _measured(self, key: str) -> Optional[ResourceFingerprint]:
        """按 key（如 'yolo.yolov8n.pt.640'）查实测指纹；命中返回，未命中 None。"""
        entry = self.fingerprints.get(key)
        if not isinstance(entry, dict):
            return None
        return ResourceFingerprint(
            vram_mb=float(entry.get("vram_mb", 0) or 0),
            ram_mb=float(entry.get("ram_mb", 0) or 0),
            latency_ms_cpu=float(entry.get("latency_ms_cpu", 0) or 0),
            latency_ms_gpu=float(entry.get("latency_ms_gpu", 0) or 0),
            note=str(entry.get("note", "") or ""),
            source="measured",
        )

    def _measured_lookup_key(self, mtype: str, params: dict) -> Optional[str]:
        """根据模块类型 + 参数构造实测指纹的 key 候选；不存在返回 None。"""
        if mtype == self.TYPE_YOLO:
            model = Path(params.get("model_path") or "").name or "yolov8n.pt"
            imgsz = int(params.get("imgsz") or 640)
            return f"yolo.{model}.{imgsz}"
        if mtype == self.TYPE_SCRFD:
            det_size = str(params.get("det_size") or "640,640")
            return f"scrfd.{det_size}"
        if mtype == self.TYPE_ARCFACE:
            return "arcface"
        if mtype == self.TYPE_FACESTORE:
            return "facestore"
        return None

    # ---------------------------------------------------------------- 静态
    def _static(self, mtype: str, params: dict) -> ResourceFingerprint:
        """静态公式估算（不加载模型）。"""
        p = params or {}
        note = "静态公式估算"

        if mtype == self.TYPE_YOLO:
            model_mb = _model_file_mb(str(p.get("model_path") or ""))
            imgsz = int(p.get("imgsz") or 640)
            act = _yolo_activation_mb(imgsz)
            vram = model_mb + act  # 权重(fp16 约等于文件大小)+ 激活（图开销暂并入）
            note = f"权重{model_mb:.1f}MB + 激活(imgsz={imgsz})"
            return ResourceFingerprint(
                vram_mb=vram,
                ram_mb=vram * 2,  # CPU 上权重 fp32 + 激活 fp32，粗略按显存×2
                latency_ms_cpu=_STATIC_LATENCY_CPU_MS["yolo"] * (imgsz / 640.0) ** 2,
                latency_ms_gpu=_STATIC_LATENCY_GPU_MS["yolo"] * (imgsz / 640.0) ** 2,
                note=note,
            )

        if mtype == self.TYPE_SCRFD:
            det_size = str(p.get("det_size") or "640,640")
            w, h = _parse_wh(det_size, 640, 640)
            ratio = (w * h) / (640 * 640)
            act = _scrfd_activation_mb(det_size)
            vram = _STATIC_VRAM_MB["scrfd"] + act
            return ResourceFingerprint(
                vram_mb=vram,
                ram_mb=vram * 2,
                latency_ms_cpu=_STATIC_LATENCY_CPU_MS["scrfd"] * ratio,
                latency_ms_gpu=_STATIC_LATENCY_GPU_MS["scrfd"] * ratio,
                note=f"det_size={det_size}",
            )

        if mtype == self.TYPE_ARCFACE:
            return ResourceFingerprint(
                vram_mb=_STATIC_VRAM_MB["arcface"],
                ram_mb=_STATIC_VRAM_MB["arcface"] * 2,
                latency_ms_cpu=_STATIC_LATENCY_CPU_MS["arcface"],
                latency_ms_gpu=_STATIC_LATENCY_GPU_MS["arcface"],
                note="单张 512d 嵌入",
            )

        if mtype == self.TYPE_FACESTORE:
            # 内存 ≈ 向量数 × 512 × 4B；face_store 参数带 register_dir 时按图数粗估
            entries = int(p.get("entries") or 1000)
            ram = entries * 512 * 4 / (1024 * 1024)  # MB
            return ResourceFingerprint(
                vram_mb=0.0,
                ram_mb=ram,
                latency_ms_cpu=_STATIC_LATENCY_CPU_MS["facestore_query"],
                latency_ms_gpu=_STATIC_LATENCY_GPU_MS["facestore_query"],
                note=f"{entries} 条向量 × 512 × 4B",
            )

        # VLM / 纯逻辑
        if mtype == self.TYPE_VLM:
            return ResourceFingerprint(
                vram_mb=0.0, ram_mb=16.0,  # 素材准备缓冲
                latency_ms_cpu=0.0, latency_ms_gpu=0.0,
                note="本地仅素材准备；真实推理=远端延迟，依赖实测",
            )

        # 纯逻辑模块
        return ResourceFingerprint(
            vram_mb=0.0, ram_mb=0.0,
            latency_ms_cpu=0.0, latency_ms_gpu=0.0,
            note="纯逻辑模块，开销可忽略",
        )

    # ---------------------------------------------------------------- 单模块
    def estimate_module(self, module_id: str, params: Optional[dict] = None) -> ResourceFingerprint:
        """对单个 web_lab 模块做估算：优先实测指纹，回退静态公式。

        module_id: 'yolo' / 'face_detect' / 'compose' ...（对应 MODULE_TYPE 键）
        params:    模块参数 dict（含 model_path / imgsz / det_size 等）
        """
        params = dict(params or {})
        mtype = self.MODULE_TYPE.get(module_id, self.TYPE_LOGIC)

        # 模块用 yolo_* 前缀参数时，统一映射到实测查表用的 model_path/imgsz
        if "yolo_model" in params:
            params.setdefault("model_path", params["yolo_model"])
        if "yolo_imgsz" in params:
            params.setdefault("imgsz", params["yolo_imgsz"])

        # face_monitor = SCRFD 主推理 + ArcFace 嵌入（按检测命中折算）
        if module_id == "face_monitor":
            scrfd = self._estimate_typed(self.TYPE_SCRFD, params)
            arcface = self._estimate_typed(self.TYPE_ARCFACE, params)
            frame_skip = max(int(params.get("frame_skip") or 3), 1)
            # 每帧平均：检测每 frame_skip 帧一次；命中才嵌入，假设每帧 1 人
            hit_ratio = 1.0
            per_frame_latency_cpu = scrfd.latency_ms_cpu / frame_skip + arcface.latency_ms_cpu * hit_ratio
            per_frame_latency_gpu = scrfd.latency_ms_gpu / frame_skip + arcface.latency_ms_gpu * hit_ratio
            return ResourceFingerprint(
                vram_mb=max(scrfd.vram_mb, arcface.vram_mb),
                ram_mb=scrfd.ram_mb + arcface.ram_mb,
                latency_ms_cpu=per_frame_latency_cpu,
                latency_ms_gpu=per_frame_latency_gpu,
                note=f"SCRFD/每{frame_skip}帧 + ArcFace(命中)",
                source=_merge_source(scrfd.source, arcface.source),
            )

        return self._estimate_typed(mtype, params)

    def _estimate_typed(self, mtype: str, params: dict) -> ResourceFingerprint:
        """按资源类型估算，实测优先。

        实测条目可能缺维度（如本机 onnxruntime 无 CUDA → 人脸模型无 vram_mb/gpu 实测），
        缺的维度用静态公式补值，避免界面显示 0MB 误导。
        """
        key = self._measured_lookup_key(mtype, params)
        if key:
            measured = self._measured(key)
            if measured:
                st = self._static(mtype, params)
                missing = []
                if measured.vram_mb <= 0 and st.vram_mb > 0:
                    measured.vram_mb = st.vram_mb
                    missing.append("vram")
                if measured.ram_mb <= 0 and st.ram_mb > 0:
                    measured.ram_mb = st.ram_mb
                    missing.append("ram")
                if measured.latency_ms_gpu <= 0 and st.latency_ms_gpu > 0:
                    measured.latency_ms_gpu = st.latency_ms_gpu
                    missing.append("gpu延迟")
                if missing:
                    measured.note = f"{measured.note}（{'、'.join(missing)} 静态补值）"
                return measured
        return self._static(mtype, params)

    # ---------------------------------------------------------------- 链路聚合
    def estimate_pipeline(
        self,
        spec: dict,
        streams: int = 1,
        vram_budget_mb: Optional[float] = None,
    ) -> PipelineEstimate:
        """按 pipeline spec 聚合整链路估算。

        spec 示例（与需求文档 §4.1 一致）：
            {"name": "巡检链路", "stages": [
                {"module": "frame_manager", "params": {...}},
                {"module": "yolo", "params": {...}},
                {"module": "vlm", "params": {...}},
                {"module": "alarm", "params": {...}},
            ], "streams": 16, "budget": {"vram_mb": 8192}}
        """
        stages = spec.get("stages") or []
        pe = PipelineEstimate(streams=streams)
        if not stages:
            pe.note = "空链路"
            return pe

        # ---- 逐模块估算 + 每帧平均成本折算 ----
        per_frame_cpu = 0.0
        per_frame_gpu = 0.0
        shared_vram = 0.0   # 模型常驻部分（GPU 上取 max）
        act_vram = 0.0      # 激活/图部分（单帧路径 max）
        ram_total = 0.0

        for st in stages:
            mid = st.get("module", "")
            params = st.get("params") or {}
            fp = self.estimate_module(mid, params)
            pe.per_module[mid] = fp

            # 每帧平均成本：check_interval 节流折算
            check_interval = float(params.get("check_interval") or 0.0)
            if mid == "vlm" and check_interval > 0:
                # VLM 每 check_interval 秒送审一次；帧率按 fps 折算成"每帧占比"
                fps = float(params.get("fps") or 25.0)
                frac = 1.0 / max(check_interval * fps, 1.0)
                per_frame_cpu += fp.latency_ms_cpu * frac
                per_frame_gpu += fp.latency_ms_gpu * frac
            else:
                per_frame_cpu += fp.latency_ms_cpu
                per_frame_gpu += fp.latency_ms_gpu

            # 显存：模型常驻部分取 max（共享），激活取 max（单帧路径）
            shared_vram = max(shared_vram, fp.vram_mb * 0.6)   # 常驻 ≈ 60% 峰值
            act_vram = max(act_vram, fp.vram_mb * 0.4)
            ram_total += fp.ram_mb

        # ---- 单路指标 ----
        pe.vram_mb = shared_vram + act_vram
        pe.ram_mb = ram_total
        pe.latency_per_frame_ms = {"cpu": per_frame_cpu, "gpu": per_frame_gpu}
        pe.fps_cpu = 1000.0 / per_frame_cpu if per_frame_cpu > 0 else 0.0
        pe.fps_gpu = 1000.0 / per_frame_gpu if per_frame_gpu > 0 else 0.0

        # ---- 多路外推：模型常驻共享 + 激活/推理线性 ----
        n = max(streams, 1)
        pe.vram_mb_N_streams = shared_vram + act_vram * n
        if n > 1 and pe.vram_mb_N_streams > pe.vram_mb:
            pe.note = f"{n} 路并发：模型常驻共享，激活/推理线性增长"

        # ---- 预算反推 ----
        if vram_budget_mb and vram_budget_mb > 0:
            reco = self.back_calculate(spec, streams=streams, vram_budget_mb=vram_budget_mb)
            if reco.get("suggestions"):
                pe.warning.extend(reco["suggestions"])
            if pe.vram_mb_N_streams > vram_budget_mb:
                pe.warning.append(
                    f"超出预算 {vram_budget_mb:.0f}MB（当前 {pe.vram_mb_N_streams:.0f}MB），"
                    f"建议 frame_skip={reco['frame_skip']} 或减路数"
                )

        return pe

    # ---------------------------------------------------------------- 预算反推
    def back_calculate(
        self,
        spec: dict,
        streams: int = 1,
        vram_budget_mb: Optional[float] = None,
    ) -> dict:
        """给定显存预算，反推建议的 frame_skip（抽帧间隔）与预期 FPS。

        思路：激活/推理部分随"每秒处理帧数"线性变化，抽帧间隔放大 N 倍 ≈
        每路每秒处理帧数降为 1/N，激活摊薄为 1/N → 总显存下降。
        返回 dict: {frame_skip, fps_gpu, expected_vram_mb, suggestions}
        """
        budget = vram_budget_mb
        if not budget or budget <= 0:
            return {"frame_skip": 1, "fps_gpu": 0.0, "expected_vram_mb": 0.0, "suggestions": []}

        base = self.estimate_pipeline(spec, streams=streams)
        if base.vram_mb_N_streams <= budget:
            return {
                "frame_skip": 1,
                "fps_gpu": base.fps_gpu,
                "expected_vram_mb": base.vram_mb_N_streams,
                "suggestions": [f"当前估算 {base.vram_mb_N_streams:.0f}MB 在预算 {budget:.0f}MB 内，无需抽帧放宽"],
            }

        # 找到最小 frame_skip 使总显存 ≤ 预算（上限 32，避免无限循环）
        frame_skip = 1
        est = base
        fit = False
        while frame_skip < 32:
            frame_skip += 1
            # 激活部分按 1/frame_skip 摊薄（激活与每秒帧数成正比）；
            # 常驻/激活按 60/40 拆分，与 estimate_pipeline 一致
            shared = est.vram_mb * 0.6
            act = est.vram_mb * 0.4 * (1.0 / frame_skip)
            vram_n = shared + act * streams
            if vram_n <= budget:
                fit = True
                break

        expected_vram = shared + est.vram_mb * 0.4 / frame_skip * streams
        fps_gpu = base.fps_gpu / frame_skip
        suggestions = [
            f"建议 frame_skip={frame_skip}（抽帧间隔从 {base.vram_mb_N_streams:.0f}MB 降到 {expected_vram:.0f}MB，"
            + (f"在预算 {budget:.0f}MB 内）" if fit else f"仍超预算 {budget:.0f}MB，需减路数或换小模型）"),
            f"预期 GPU 吞吐：{fps_gpu:.1f} FPS（每路）",
        ]
        return {"frame_skip": frame_skip, "fps_gpu": fps_gpu, "expected_vram_mb": expected_vram, "suggestions": suggestions}


# ---------------------------------------------------------------------------
# 便捷函数：直接读 spec 文件估算
# ---------------------------------------------------------------------------


def estimate_pipeline_from_file(spec_path: str, project_root: str = ".") -> PipelineEstimate:
    """从 JSON spec 文件估算整条链路。"""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    est = ResourceEstimator(project_root=project_root)
    return est.estimate_pipeline(spec, streams=int(spec.get("streams") or 1),
                                 vram_budget_mb=spec.get("budget", {}).get("vram_mb"))


if __name__ == "__main__":
    import sys
    # 自检：静态估算关键模块 + 示例链路聚合
    est = ResourceEstimator(project_root=".")
    print("== 单模块静态估算 ==")
    for mid, params in [
        ("yolo", {"imgsz": 640}),
        ("yolo", {"imgsz": 1280}),
        ("face_detect", {"det_size": "640,640"}),
        ("face_embed", {}),
        ("face_store", {"entries": 1000}),
        ("vlm", {"check_interval": 3.0}),
    ]:
        fp = est.estimate_module(mid, params)
        print(f"{mid:14s} vram={fp.vram_mb:7.1f}MB ram={fp.ram_mb:7.1f}MB "
              f"cpu={fp.latency_ms_cpu:7.1f}ms gpu={fp.latency_ms_gpu:6.1f}ms [{fp.source}]")

    print("\n== 示例链路聚合（16 路）==")
    spec = {
        "name": "巡检链路",
        "stages": [
            {"module": "frame_manager", "params": {"sampling": "wall_clock", "interval_sec": 1.0}},
            {"module": "yolo", "params": {"model_path": "yolov8n.pt", "imgsz": 640, "classes": "0"}},
            {"module": "vlm", "params": {"ref": "crop:cls0", "check_interval": 3.0}},
            {"module": "alarm", "params": {"kind": "window", "target_actions": "fire"}},
        ],
        "streams": 16,
        "budget": {"vram_mb": 8192},
    }
    pe = est.estimate_pipeline(spec, streams=16, vram_budget_mb=8192)
    print(json.dumps(pe.to_dict(), ensure_ascii=False, indent=2))

    # 命令行模式：python umvp/resources.py <spec.json>
    if len(sys.argv) > 1:
        pe2 = estimate_pipeline_from_file(sys.argv[1], project_root=".")
        print(json.dumps(pe2.to_dict(), ensure_ascii=False, indent=2))
