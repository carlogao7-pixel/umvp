# -*- coding: utf-8 -*-
"""
链路运行报告导出（WO-0028）：_build_report_html 纯逻辑断言测试

覆盖（报告构建 = 输入运行快照 dict → 输出自包含 HTML 字符串，不碰服务状态）：
  1. 摘要：链路名 / 阶段序列（中文标签）/ 媒体与时间范围 / 计数 / VLM 状态 / 落盘统计。
  2. 字段表通用取自 timeline 条目 data（按数据形状识别，不绑定链路类型）：
     事件（data 含 name）/ VLM（kind=vlm 且 data 含 action）/ 车牌（data 含 plate_no）/
     人脸（kind=face）。
  3. 图片内嵌：落盘目录按前缀分组（frame_*_det / plate_ / face_）→ data URL；
     分组上限生效（超限只嵌入前 N 张并注明）；无图分组省略。
  4. HTML 转义：含 <script> 的事件 detail 不以原样出现在 HTML 里。
  5. 容错：快照缺 output_dir / timeline 为空 / 图片文件损坏时不炸、对应分组省略。

运行: conda run -n ai python tests/test_export_report.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402  （顶层无 HTTP/DB 副作用，db 连接惰性）

PASS = 0


def ok(cond, msg: str) -> None:
    global PASS
    assert cond, f"断言失败: {msg}"
    PASS += 1
    print(f"  [ok] {msg}")


def make_img(w: int = 320, h: int = 240) -> np.ndarray:
    """合成 BGR 测试图（渐变 + 色块，非纯色便于确认编码非空）。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 200, 0), -1)
    return img


def fake_snapshot(out_dir: str | None, n_frame_imgs: int = 2) -> dict:
    """构造一份假运行快照（覆盖事件/VLM/车牌/人脸四类 timeline 条目）。"""
    return {
        "name": "测试链路<script>",
        "pipeline_id": "PIPE0001",
        "stages": ["frame_manager", "yolo", "det_filter", "seq_filter", "vlm"],
        "media": {"type": "video", "name": "rescue1.MP4", "start_time": 0.0,
                  "end_time": 10.0, "start_frame": 0, "end_frame": 250},
        "stats": {"processed": 17, "detections": 42, "plates": 2, "events": 2,
                  "vlm_state": "真实调用（失败 1 次）", "vlm_fail": 1,
                  "saved_images": n_frame_imgs, "dropped_saves": 0,
                  "elapsed_sec": 12.34},
        "timeline": [
            {"kind": "frame", "stage": "frame_manager", "stage_label": "帧管理",
             "text": "时间5.20s (处理第14帧)", "ts": 5.2, "frame": 13},
            # 事件条目（data 含 name/detail，detail 带危险字符验证转义）
            {"kind": "submit", "stage": "seq_filter", "stage_label": "时序判定",
             "text": "事件 人员倒地 <b>加粗</b>", "ts": 5.2, "frame": 13,
             "data": {"name": "人员倒地", "detail": {"pos": "画面左侧", "x": "<script>alert(1)</script>"},
                      "frame_idx": 130, "conf": 0.87}},
            # VLM 条目（kind=vlm 且 data 含 action/description/content）
            {"kind": "vlm", "stage": "vlm", "stage_label": "VLM研判",
             "text": "研判 rescue — 有人倒地（真实调用）", "ts": 5.2, "frame": 13,
             "data": {"action": "rescue", "description": "有人倒地 & 挥手",
                      "content": {"alert_type": "救援救助类", "confidence": 0.9}}},
            # 模拟回填的 VLM 条目（content 为扁平兜底结构）
            {"kind": "vlm", "stage": "vlm", "stage_label": "VLM研判",
             "text": "研判 fire（模拟回填）", "ts": 6.4, "frame": 16,
             "data": {"action": "fire", "description": "",
                      "content": {"result": "fire", "description": ""}}},
            # 车牌条目（data 含 plate_no；字段覆盖 to_dict 全集 + 附加键）
            {"kind": "submit", "stage": "plate_recog", "stage_label": "车牌识别",
             "text": "识别 京A12345 蓝牌/蓝 0.98", "ts": 3.1, "frame": 8,
             "data": {"plate_no": "京A12345", "plate_type": 0, "type_name": "蓝牌",
                      "color": "蓝", "score": 0.9765, "bbox": [100, 200, 300, 260],
                      "track_id": 3}},
            # 人脸条目（kind=face；含 best_sim 附加键）
            {"kind": "face", "stage": "face_detect", "stage_label": "人脸检测",
             "text": "人脸 #1（112x112）", "ts": 7.7, "frame": 20,
             "data": {"index": 1, "face_bbox": [10, 20, 120, 130],
                      "face_size": [112, 112], "score": 0.9123,
                      "kps": [[30.5, 60.1]], "best_sim": 0.62}},
            # 去重丢弃行（kind=info，不进人脸表）
            {"kind": "info", "stage": "face_embed", "stage_label": "人脸嵌入",
             "text": "重复人脸（相似 0.91）", "ts": 8.0, "frame": 21,
             "data": {"best_sim": 0.91}},
        ],
        "output_dir": out_dir,
        "generated_at": "2026-10-27 12:00:00",
        "mode": "sync",
    }


def make_out_dir(tmp: Path, n_frame: int = 2) -> Path:
    """临时落盘目录：标注帧 / 车牌 / 人脸各若干张合成图 + 一个坏文件。"""
    d = tmp / "out"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_frame):
        cv2.imwrite(str(d / f"frame_00{i}_20261027_120000_5_20s_det.jpg"), make_img())
    cv2.imwrite(str(d / "plate_001_京A12345_3.1s.jpg"), make_img(160, 60))
    cv2.imwrite(str(d / "face_001_7.7s.jpg"), make_img(112, 112))
    cv2.imwrite(str(d / "face_002_9.9s.png"), make_img(112, 112))
    (d / "frame_009_broken_det.jpg").write_bytes(b"\xff\xff not an image")  # 坏图应被跳过
    (d / "unrelated.txt").write_text("忽略非图片文件", encoding="utf-8")
    return d


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="umvp_report_") as td:
        tmp = Path(td)
        out_dir = make_out_dir(tmp, n_frame=2)
        snap = fake_snapshot(str(out_dir))

        html = server._build_report_html(snap)

        # ---- 1. 骨架与摘要 ----
        ok(html.startswith("<!DOCTYPE html>"), "报告以 <!DOCTYPE html> 开头")
        ok("UMVP 链路运行报告" in html, "含报告标题")
        ok("帧管理 → YOLO识别 → 检测筛选 → 时序判定 → VLM研判" in html,
           "阶段序列按中文短标签展示")
        ok("视频 rescue1.MP4" in html and "0s – 10s（帧 0–250）" in html,
           "媒体与时间范围")
        ok("处理帧数" in html and ">17<" in html, "处理帧数入摘要")
        ok("真实调用（失败 1 次）" in html, "VLM 状态入摘要")
        ok("批量运行" in html, "运行方式入摘要")
        ok("2026-10-27 12:00:00" in html, "生成时间入摘要")

        # ---- 2. 字段表 ----
        ok("事件列表（1 条）" in html and "人员倒地" in html, "事件表含事件名")
        ok("画面左侧" in html and "frame_idx" in html, "事件表含 detail 与其他字段")
        ok("VLM 研判（2 条）" in html, "VLM 表条数")
        ok("救援救助类" in html and "有人倒地 &amp; 挥手" in html,
           "VLM 表含 content JSON 与转义后的 description")
        ok("车牌识别（1 条）" in html and "京A12345" in html and "蓝牌" in html,
           "车牌表含车牌号/类型")
        ok("0.9765" in html, "车牌置信度浮点格式化")
        ok("人脸截取（1 张）" in html and "人脸框" in html and "0.62" in html,
           "人脸表只收 kind=face（去重行不计）")
        ok("相似度" in html, "人脸 best_sim 列标签")

        # ---- 3. 图片内嵌与分组 ----
        ok(html.count("data:image/jpeg;base64,") == 5,
           "内嵌图片数 = 标注帧2 + 车牌1 + 人脸2（坏图跳过）")
        ok("标注帧（2 张）" in html, "标注帧分组")
        ok("车牌小图（1 张）" in html, "车牌分组")
        ok("人脸截图（2 张）" in html, "人脸分组")
        ok("unrelated" not in html, "非图片文件不入组")

        # ---- 3a. saved_files 清单过滤（目录里的历史运行文件不混入本次报告）----
        snap_own = fake_snapshot(str(out_dir))
        own_name = "frame_000_20261027_120000_5_20s_det.jpg"
        snap_own["saved_files"] = [own_name]
        html_own = server._build_report_html(snap_own)
        ok(html_own.count("data:image/jpeg;base64,") == 1, "saved_files 清单只嵌本次落盘")
        ok("标注帧（1 张）" in html_own, "清单过滤后分组计数按实际嵌入")
        ok("车牌小图" not in html_own and "人脸截图" not in html_own,
           "历史车牌/人脸图整组不出现（清单外）")

        # ---- 3b. 分组上限 ----
        html_lim = server._build_report_html(snap, max_anno=1, max_face=1)
        ok(html_lim.count("data:image/jpeg;base64,") == 3,
           "上限生效：标注帧 1 + 车牌 1 + 人脸 1")
        ok("超出上限仅嵌入前 1 张" in html_lim, "超限注明")
        ok("frame_009" not in html_lim, "坏图不出现（顺带验证上限截断后跳过）")

        # ---- 4. HTML 转义 ----
        ok("&lt;script&gt;" in html, "危险字符被转义后可见")
        ok("<script>alert(1)</script>" not in html, "原始 <script> 不出现")
        ok("<b>加粗</b>" not in html and "事件 人员倒地 &lt;b&gt;加粗&lt;/b&gt;" not in html,
           "text 原文不在正文裸渲染（表格按字段拆列展示）")

        # ---- 5. 容错 ----
        html_nodir = server._build_report_html(fake_snapshot(None))
        ok("本次运行无落盘图片" in html_nodir, "无 output_dir → 图片分组省略不炸")
        ok("本次运行无车牌识别记录" not in html_nodir, "字段表不受 output_dir 缺失影响")
        snap_empty = fake_snapshot(str(out_dir))
        snap_empty["timeline"] = []
        html_empty = server._build_report_html(snap_empty)
        ok("本次运行无事件" in html_empty and "本次运行无 VLM 研判记录" in html_empty,
           "timeline 为空 → 各表省略不炸")

        # ---- 5b. 流式快照形态（output_dir=None、mode=stream）----
        snap_stream = fake_snapshot(None)
        snap_stream["mode"] = "stream"
        snap_stream["timeline"] = [
            {"kind": "submit", "stage": "plate_recog", "stage_label": "车牌识别",
             "text": "识别 京B88888 蓝牌/蓝", "ts": 2.0, "frame": 5,
             "data": {"plate_no": "京B88888", "type_name": "蓝牌", "color": "蓝",
                      "score": 0.95, "bbox": [1, 2, 3, 4]}}]
        snap_stream["stats"].update({"saved_images": 0, "vlm_state": "未开启研判回填"})
        html_stream = server._build_report_html(snap_stream)
        ok("实时播放叠加（流式）" in html_stream, "流式运行方式标注")
        ok("京B88888" in html_stream and "无（流式运行不落盘）" in html_stream,
           "流式快照：车牌表正常、落盘统计标注不落盘")

    print(f"\n全部通过: {PASS} 断言")


if __name__ == "__main__":
    main()
