# -*- coding: utf-8 -*-
"""VLM 端点环境变量注入自检（WO-0006.1，进程内可复跑）。

覆盖：
  1. 零回归：不设任何 env 时 _resolve_vlm_cfg 逐字回落（spec 给值 / spec 为空两态）；
  2. env 优先：三项各自覆盖 spec；仅设一项时其余回落；
  3. key 别名：GLM_API_KEY 生效，VLM_API_KEY 优先于 GLM_API_KEY；
  4. 真实请求 spy：url / body.model / Authorization 头（单图 + 多图 list）；
  5. 4xx 可读提示：401/403/429 抛 VlmHttpError，提示可读；
  6. 集成：链路里 4xx 提示每轮最多一条（调用 N≥2 次仍只出 1 条）。

运行: conda run -n ai python web_lab/selfcheck_vlm_env.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "umvp"))
sys.path.insert(0, str(ROOT / "web_lab"))

import server  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  [PASS] {name}" + (f" | {detail}" if detail else ""))


ENV_KEYS = ["VLM_ENDPOINT", "VLM_MODEL", "VLM_API_KEY", "GLM_API_KEY"]
DEF_URL = server.VLM_DEFAULT_ENDPOINT
DEF_MODEL = server.VLM_DEFAULT_MODEL


def clear_env() -> None:
    for k in ENV_KEYS:
        os.environ.pop(k, None)


def _spy_requests(cap: dict, status: int = 200):
    """替换 sys.modules['requests'] 为捕获 post 的桩，返回恢复函数。"""

    class _Resp:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        def json(self):
            return {"choices": [{"message": {"content": '{"result": "none", "description": ""}'}}]}

    def _post(url, json=None, headers=None, timeout=None):
        cap["url"] = url
        cap["body"] = json
        cap["headers"] = headers
        cap["timeout"] = timeout
        return _Resp()

    stub = types.ModuleType("requests")
    stub.post = _post
    orig = sys.modules.get("requests")
    sys.modules["requests"] = stub

    def restore():
        if orig is not None:
            sys.modules["requests"] = orig
        else:
            sys.modules.pop("requests", None)
    return restore


def resolver_checks() -> None:
    clear_env()
    # spec 给了值 → 原样（逐字）
    u, m, k = server._resolve_vlm_cfg(
        {"vlm_endpoint": "http://x:1/v1", "vlm_model": "m1", "vlm_key": "k1"})
    check("零回归: spec 值原样返回", (u, m, k) == ("http://x:1/v1", "m1", "k1"), str((u, m, k)))
    # spec 为空/缺省 → 内置默认（逐字）
    u, m, k = server._resolve_vlm_cfg({})
    check("零回归: 空 spec → 默认端点/模型", (u, m) == (DEF_URL, DEF_MODEL), f"{u} {m}")
    check("零回归: 空 spec → key 空", k == "", repr(k))
    u, m, k = server._resolve_vlm_cfg({"vlm_endpoint": "  ", "vlm_model": "", "vlm_key": " "})
    check("零回归: 纯空白 spec → 默认/空", (u, m, k) == (DEF_URL, DEF_MODEL, ""), f"{u}|{m}|{k!r}")
    check("零回归: 默认端点逐字", DEF_URL == "http://117.42.21.253:8000/v1/chat/completions", DEF_URL)
    check("零回归: 默认模型逐字", DEF_MODEL == "qwen3-vl-32b", DEF_MODEL)

    # env 优先（三项均覆盖 spec）
    os.environ.update(VLM_ENDPOINT="https://e/v1", VLM_MODEL="envm", VLM_API_KEY="envk")
    u, m, k = server._resolve_vlm_cfg(
        {"vlm_endpoint": "http://spec/v1", "vlm_model": "specm", "vlm_key": "speck"})
    check("env 优先: 三项均覆盖 spec", (u, m, k) == ("https://e/v1", "envm", "envk"), str((u, m, k)))
    clear_env()

    # 仅设 VLM_MODEL → endpoint/key 回落 spec
    os.environ["VLM_MODEL"] = "onlym"
    u, m, k = server._resolve_vlm_cfg({"vlm_endpoint": "http://spec/v1", "vlm_key": "speck"})
    check("仅 VLM_MODEL: endpoint/key 回落 spec",
          (u, m, k) == ("http://spec/v1", "onlym", "speck"), str((u, m, k)))
    clear_env()

    # key 别名
    os.environ["GLM_API_KEY"] = "glmk"
    _, _, k = server._resolve_vlm_cfg({"vlm_key": "speck"})
    check("GLM_API_KEY 别名生效（覆盖 spec）", k == "glmk", repr(k))
    os.environ["VLM_API_KEY"] = "vlmk"
    _, _, k = server._resolve_vlm_cfg({"vlm_key": "speck"})
    check("VLM_API_KEY 优先于 GLM_API_KEY", k == "vlmk", repr(k))
    clear_env()

    # 仅设别名、spec 也空 → 用别名
    os.environ["GLM_API_KEY"] = "glmk2"
    _, _, k = server._resolve_vlm_cfg({})
    check("仅 GLM_API_KEY: key 生效", k == "glmk2", repr(k))
    clear_env()


def spy_checks() -> None:
    clear_env()
    frame = np.zeros((100, 160, 3), np.uint8)

    cap = {}
    restore = _spy_requests(cap)
    try:
        server._vlm_real({"vlm_endpoint": "http://spec/v1", "vlm_model": "specm",
                          "vlm_key": "speck"}, {"max_resolution": (1280, 720)}, frame)
    finally:
        restore()
    check("spy 单图: url=spec", cap["url"] == "http://spec/v1", cap["url"])
    check("spy 单图: body.model=spec", cap["body"]["model"] == "specm", cap["body"]["model"])
    check("spy 单图: Authorization=Bearer speck",
          (cap["headers"] or {}).get("Authorization") == "Bearer speck", str(cap["headers"]))
    c = cap["body"]["messages"][0]["content"]
    check("spy 单图: content=[image_url,text]",
          [x["type"] for x in c] == ["image_url", "text"], str([x["type"] for x in c]))

    # env 覆盖 spec：spy 再验一次
    os.environ.update(VLM_ENDPOINT="https://env/v1", VLM_MODEL="envm", VLM_API_KEY="envk")
    cap2 = {}
    restore = _spy_requests(cap2)
    try:
        server._vlm_real({"vlm_endpoint": "http://spec/v1", "vlm_model": "specm",
                          "vlm_key": "speck"}, {"max_resolution": (1280, 720)}, frame)
    finally:
        restore()
        clear_env()
    check("spy env: url=env", cap2["url"] == "https://env/v1", cap2["url"])
    check("spy env: body.model=env", cap2["body"]["model"] == "envm", cap2["body"]["model"])
    check("spy env: Authorization=Bearer envk",
          (cap2["headers"] or {}).get("Authorization") == "Bearer envk", str(cap2["headers"]))

    # 多图 list 路径
    cap3 = {}
    restore = _spy_requests(cap3)
    try:
        imgs = [np.zeros((20, 20, 3), np.uint8) for _ in range(3)]
        server._vlm_real({}, {"max_resolution": (1280, 720)}, imgs)
    finally:
        restore()
    c3 = cap3["body"]["messages"][0]["content"]
    check("spy 多图: 3×image_url + 1×text",
          len([x for x in c3 if x["type"] == "image_url"]) == 3
          and len([x for x in c3 if x["type"] == "text"]) == 1)
    check("spy 多图: url=默认", cap3["url"] == DEF_URL, cap3["url"])

    # 无 key → 无 Authorization 头（零回归）
    cap4 = {}
    restore = _spy_requests(cap4)
    try:
        server._vlm_real({}, {"max_resolution": (1280, 720)}, np.zeros((10, 10, 3), np.uint8))
    finally:
        restore()
    check("无 key: headers 无 Authorization", "Authorization" not in (cap4["headers"] or {}),
          str(cap4["headers"]))


def hint_checks() -> None:
    clear_env()
    frame = np.zeros((10, 10, 3), np.uint8)
    for status, word in [(401, "需鉴权"), (403, "需鉴权"), (429, "限额")]:
        cap = {}
        restore = _spy_requests(cap, status=status)
        raised = None
        try:
            try:
                server._vlm_real({}, {"max_resolution": (1280, 720)}, frame)
            except server.VlmHttpError as e:
                raised = e
        finally:
            restore()
        check(f"{status}: 抛 VlmHttpError", raised is not None and raised.status == status)
        check(f"{status}: hint 可读含「{word}」",
              raised is not None and word in raised.hint, getattr(raised, "hint", ""))
        check(f"{status}: hint 非空且中文", bool(raised and raised.hint), getattr(raised, "hint", ""))


def chain_hint_once() -> None:
    """链路级：VLM 连续 401 时，可读提示每轮最多一条。"""
    clear_env()
    orig_load = server.db.load_pipeline
    orig_real = server._vlm_real
    orig_reach = server._endpoint_reachable
    orig_apply = server._apply_vlm_inputs
    calls = {"n": 0}

    def boom(p, payload, frame):
        calls["n"] += 1
        raise server.VlmHttpError(401, "VLM 端点需鉴权(HTTP 401)",
                                  "VLM 端点需鉴权：请导出 VLM_API_KEY（或 GLM_API_KEY）")

    def dup(pipe, subs):
        return (list(subs) * 2) if subs else subs  # 人为制造每轮多次调用

    spec = {"name": "wo6", "stages": [
        {"module_id": "frame_manager", "params": {"sampling": "analysis", "frame_skip": 1}},
        {"module_id": "yolo", "params": {"model_path": "models/yolov8n.pt",
                                         "conf": 0.25, "classes": "0,2"}},
        {"module_id": "annotate", "params": {"thickness": 2, "show_label": True}},
        {"module_id": "vlm", "params": {"use_real_vlm": True}},
        {"module_id": "det_filter", "params": {"filter": ""}},  # 链尾（终段合法）
    ]}
    server.db.load_pipeline = lambda pid: spec
    server._vlm_real = boom
    server._endpoint_reachable = lambda url, timeout=2.0: True
    server._apply_vlm_inputs = dup
    try:
        with tempfile.TemporaryDirectory(prefix="wo6_") as td:
            r = server.h_pipeline_run({"pipeline_id": "PIPE0001", "image": "人遮挡1.png",
                                       "output_dir": td, "image_format": "jpg",
                                       "vlm_feedback": True, "vlm_action": "fire"})
    finally:
        server.db.load_pipeline = orig_load
        server._vlm_real = orig_real
        server._endpoint_reachable = orig_reach
        server._apply_vlm_inputs = orig_apply

    tl = r.get("timeline") or []
    hint = "VLM 端点需鉴权：请导出 VLM_API_KEY"
    n_hint = sum(1 for t in tl if hint in (t.get("text") or ""))
    n_fail = sum(1 for t in tl if "调用失败" in (t.get("text") or ""))
    check("集成: 运行成功", r.get("ok") is True, str(r.get("error", "")))
    check("集成: VLM 实际被调用 ≥2 次（去重有意义）", calls["n"] >= 2, f"calls={calls['n']}")
    check("集成: 调用失败计数 = 调用次数", n_fail == calls["n"], f"fail={n_fail} calls={calls['n']}")
    check("集成: 4xx 可读提示只出现一次", n_hint == 1, f"hint={n_hint}")


def main() -> int:
    print("== 1/4 解析优先级与零回归 ==")
    resolver_checks()
    print("== 2/4 真实请求 spy（url/model/Authorization） ==")
    spy_checks()
    print("== 3/4 4xx 可读提示 ==")
    hint_checks()
    print("== 4/4 链路级 4xx 提示每轮一条 ==")
    chain_hint_once()
    print(f"\n全部通过：{PASS} 项断言")
    return 0


if __name__ == "__main__":
    sys.exit(main())
