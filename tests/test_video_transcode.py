# -*- coding: utf-8 -*-
"""
视频解码适配（web_lab 按需 H.264 转码）测试：秒级（除端到端转码一小段）

背景：浏览器 <video> 不解 H.265(HEVC)，项目素材多为 HEVC/4K 会黑屏。web_lab 现按需
用 ffmpeg 转 H.264 并缓存，前端轮询进度后播放；本测试验证探测/判定/缓存键与端到端转码。

覆盖:
  1. _is_browser_playable : H.264/yuv420p + 常见音轨 直通；HEVC / 非 420 / 未知音轨 需转码
  2. _trans_cache_path    : 同源稳定；源变（size/mtime）换键；落在 out/transcoded 且 .mp4
  3. _probe_video         : 读到 h264 / hevc 编码（依赖 ffprobe，缺失则跳过）
  4. 端到端转码           : 生成 2s HEVC 小片 → _video_prepare 触发转码 → 轮询至 ready
                            → _media_path 命中且输出确为 H.264/yuv420p
  5. 直通路径             : 浏览器友好源 → status=ready / transcoded=False（不转码）

运行: conda run -n ai python tests/test_video_transcode.py
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402  web_lab/server.py（导入即初始化转码目录/探测 ffmpeg）

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


def _ffmpeg_make(path: Path, vcodec: str) -> bool:
    """用 ffmpeg 生成 2s 测试小片（指定编码）；失败返回 False。"""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
           "-i", "testsrc=size=320x240:rate=10", "-t", "2",
           "-c:v", vcodec, "-pix_fmt", "yuv420p", str(path)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main() -> None:
    print("== 1. _is_browser_playable 判定 ==")
    check("h264+yuv420p+aac 直通", server._is_browser_playable(
        {"vcodec": "h264", "pix_fmt": "yuv420p", "acodec": "aac"}))
    check("h264+yuv420p 无音轨直通", server._is_browser_playable(
        {"vcodec": "h264", "pix_fmt": "yuv420p", "acodec": None}))
    check("hevc 需转码", not server._is_browser_playable(
        {"vcodec": "hevc", "pix_fmt": "yuv420p", "acodec": "aac"}))
    check("h264+yuv444p 需转码", not server._is_browser_playable(
        {"vcodec": "h264", "pix_fmt": "yuv444p", "acodec": "aac"}))
    check("h264+ac3 音轨需转码", not server._is_browser_playable(
        {"vcodec": "h264", "pix_fmt": "yuv420p", "acodec": "ac3"}))
    check("空探测结果需转码", not server._is_browser_playable({}))

    print("== 2. _trans_cache_path 缓存键 ==")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "clip.mp4"
        src.write_bytes(b"x" * 100)
        p1 = server._trans_cache_path(src)
        check("同源稳定", p1 == server._trans_cache_path(src))
        check("落 out/transcoded 且 .mp4",
              p1.parent == server.TRANS_DIR and p1.suffix == ".mp4", str(p1.name))
        src.write_bytes(b"x" * 200)  # 大小变化
        check("源变换键", p1 != server._trans_cache_path(src))

    if not server._FFPROBE or not server._FFMPEG:
        print("  [SKIP] 缺 ffprobe/ffmpeg，跳过 3/4/5")
        print(f"\n全部通过：{PASS} 断言")
        return

    print("== 3/4/5. 探测 + 端到端转码 ==")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h264 = td / "ok_h264.mp4"
        hevc = td / "src_hevc.mp4"
        if not _ffmpeg_make(h264, "libx264") or not _ffmpeg_make(hevc, "libx265"):
            print("  [SKIP] 生成测试片失败（缺 libx264/libx265），跳过 3/4/5")
            print(f"\n全部通过：{PASS} 断言")
            return

        check("探测 h264", server._probe_video(h264).get("vcodec") == "h264")
        check("探测 hevc", server._probe_video(hevc).get("vcodec") == "hevc")

        # 5. 浏览器友好源：直接 ready、不转码
        r = server._video_prepare(str(h264))
        check("H.264 直通 ready", r.get("status") == "ready"
              and r.get("transcoded") is False, str(r.get("url")))

        # 4. HEVC 源：触发转码并轮询至 ready
        r = server._video_prepare(str(hevc))
        check("HEVC 触发转码", r.get("status") in ("transcoding", "ready"), str(r.get("status")))
        deadline = time.time() + 90
        while time.time() < deadline:
            r = server._video_prepare(str(hevc))
            if r.get("status") == "ready":
                break
            assert r.get("status") == "transcoding", f"转码异常: {r}"
            time.sleep(0.5)
        check("HEVC 转码完成", r.get("status") == "ready", str(r.get("progress")))
        check("HEVC 走 /media 地址", str(r.get("url", "")).startswith("/media/videos/"))
        mp = server._media_path(str(hevc))
        check("_media_path 命中缓存", mp is not None and mp.is_file())
        out_info = server._probe_video(mp)
        check("输出为 H.264/yuv420p",
              out_info.get("vcodec") == "h264" and out_info.get("pix_fmt") == "yuv420p",
              str(out_info.get("vcodec")))
        check("_is_browser_playable(输出)", server._is_browser_playable(out_info))

    print(f"\n全部通过：{PASS} 断言")


if __name__ == "__main__":
    main()
