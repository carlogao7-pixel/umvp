# -*- coding: utf-8 -*-
"""VLM 端点环境变量注入自检（WO-0006.1，进程内可复跑）。

覆盖：
  1. 零回归：不设任何 env 时 _resolve_vlm_cfg 逐字回落（spec 给值 / spec 为空两态）；
  2. env 优先：三项各自覆盖 spec；仅设一项时其余回落；
  3. key 别名：GLM_API_KEY 生效，VLM_API_KEY 优先于 GLM_API_KEY；
  4. 提供方插槽（WO-0019）：provider=glm → 端点/模型取 VLM_PROVIDERS（覆盖 spec 手填）；
     provider=custom/未知/缺省 → 回落手填/默认；env 仍压过 provider；key 不受 provider 影响；
  5. 真实请求 spy：url / body.model / Authorization 头（单图 + 多图 list）；
  6. 4xx 可读提示：401/403/429 抛 VlmHttpError，提示可读；
  7. 集成：链路里 4xx 提示每轮最多一条（调用 N≥2 次仍只出 1 条）。

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


def provider_checks() -> None:
    """提供方插槽（WO-0019）：vlm_provider 分支解析 + PARAMS schema。"""
    clear_env()
    prov = server.VLM_PROVIDERS
    glm = prov["glm"]

    # 插槽表内容逐字（锁行为：glm 端点/模型、默认插槽=内置默认）
    check("插槽表: 恰好三家（qwen_selfhost/glm/custom）",
          set(prov) == {"qwen_selfhost", "glm", "custom"}, str(sorted(prov)))
    check("插槽表: glm 端点逐字",
          glm["endpoint"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions",
          glm["endpoint"])
    check("插槽表: glm 模型逐字", glm["model"] == "glm-4v-flash", glm["model"])
    check("插槽表: qwen_selfhost = 内置默认端点/模型",
          (prov["qwen_selfhost"]["endpoint"], prov["qwen_selfhost"]["model"])
          == (DEF_URL, DEF_MODEL))
    check("插槽表: custom 无 endpoint（视为自定义回落）", "endpoint" not in prov["custom"])

    # provider=glm → endpoint/model 取插槽；key 仍按原三级（env → spec）
    u, m, k = server._resolve_vlm_cfg({"vlm_provider": "glm", "vlm_key": "speck"})
    check("provider=glm: endpoint/model 取插槽",
          (u, m) == (glm["endpoint"], glm["model"]), f"{u} {m}")
    check("provider=glm: key 不受 provider 影响", k == "speck", repr(k))

    # provider=glm 覆盖 spec 手填端点/模型
    u, m, _ = server._resolve_vlm_cfg(
        {"vlm_provider": "glm", "vlm_endpoint": "http://spec/v1", "vlm_model": "specm"})
    check("provider=glm: 覆盖 spec 手填端点/模型",
          (u, m) == (glm["endpoint"], glm["model"]), f"{u} {m}")

    # provider=custom → 回落 spec 手填；spec 空 → 内置默认
    u, m, _ = server._resolve_vlm_cfg(
        {"vlm_provider": "custom", "vlm_endpoint": "http://spec/v1", "vlm_model": "specm"})
    check("provider=custom: 回落 spec 手填", (u, m) == ("http://spec/v1", "specm"), f"{u} {m}")
    u, m, _ = server._resolve_vlm_cfg({"vlm_provider": "custom"})
    check("provider=custom: spec 空 → 内置默认", (u, m) == (DEF_URL, DEF_MODEL), f"{u} {m}")

    # provider 未知 / None（DB 空列还原）→ 回落内置默认（零回归）
    u, m, _ = server._resolve_vlm_cfg({"vlm_provider": "nope"})
    check("provider 未知: 回落内置默认", (u, m) == (DEF_URL, DEF_MODEL), f"{u} {m}")
    u, m, _ = server._resolve_vlm_cfg({"vlm_provider": None})
    check("provider=None（DB 空列）: 回落内置默认", (u, m) == (DEF_URL, DEF_MODEL), f"{u} {m}")
    u, m, _ = server._resolve_vlm_cfg({"vlm_provider": "  "})
    check("provider 纯空白: 回落内置默认", (u, m) == (DEF_URL, DEF_MODEL), f"{u} {m}")

    # env 仍压过 provider（部署级强制覆盖语义不变）
    os.environ.update(VLM_ENDPOINT="https://env/v1", VLM_MODEL="envm")
    u, m, _ = server._resolve_vlm_cfg({"vlm_provider": "glm"})
    check("env 压过 provider=glm", (u, m) == ("https://env/v1", "envm"), f"{u} {m}")
    clear_env()

    # PARAMS schema：vlm_provider select 参数（/api/schema 暴露、DB 落库路径）
    spec = server.PARAMS["vlm"]["vlm_provider"]
    check("PARAMS: vlm_provider type=select", spec.get("type") == "select", str(spec.get("type")))
    check("PARAMS: default=qwen_selfhost", spec.get("default") == "qwen_selfhost")
    opts = spec.get("options") or []
    check("PARAMS: options 三家 value 齐且有序",
          [(o["value"] if isinstance(o, dict) else o) for o in opts]
          == ["qwen_selfhost", "glm", "custom"], str(opts))
    check("PARAMS: options label 均非空",
          all((o.get("label") if isinstance(o, dict) else o) for o in opts))
    check("PARAMS: options 与插槽表 label 一致",
          all(o["label"] == prov[o["value"]]["label"] for o in opts if isinstance(o, dict)))
    check("PARAMS: vlm_provider 非 test_only（进链路 spec / DB 落库）",
          not spec.get("test_only"))
    check("db: select 列类型 VARCHAR(64)（provider 键值可承载）",
          server.db._SQL_TYPES.get("select") == "VARCHAR(64)")

    # 真实请求 spy：provider=glm → url/model 生效
    frame = np.zeros((10, 10, 3), np.uint8)
    cap = {}
    restore = _spy_requests(cap)
    try:
        server._vlm_real({"vlm_provider": "glm"}, {"max_resolution": (1280, 720)}, frame)
    finally:
        restore()
    check("spy provider=glm: url=插槽端点", cap["url"] == glm["endpoint"], cap["url"])
    check("spy provider=glm: body.model=glm-4v-flash",
          cap["body"]["model"] == glm["model"], cap["body"]["model"])


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
    print("== 1/5 解析优先级与零回归 ==")
    resolver_checks()
    print("== 2/5 提供方插槽（vlm_provider 分支 + schema） ==")
    provider_checks()
    print("== 3/5 真实请求 spy（url/model/Authorization） ==")
    spy_checks()
    print("== 4/5 4xx 可读提示 ==")
    hint_checks()
    print("== 5/5 链路级 4xx 提示每轮一条 ==")
    chain_hint_once()
    print(f"\n全部通过：{PASS} 项断言")
    return 0


if __name__ == "__main__":
    sys.exit(main())
