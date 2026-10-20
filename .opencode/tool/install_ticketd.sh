#!/usr/bin/env bash
# install_ticketd.sh —— 把工单看板装成 systemd user 常驻服务（一份代码多实例）
#
# 用法：
#   bash .opencode/tool/install_ticketd.sh [选项]
#     --name NAME       实例名（= 项目名；默认取仓库目录名），如 umvp
#     --port PORT       监听端口（默认 8010）
#     --file PATH       工单文件（默认 <仓库>/.opencode/memory/tickets.md）
#     --inbox-file PATH 收件箱文件（默认 <仓库>/.opencode/memory/inbox.md，存在即启用；
#                       传 none 显式禁用 → 看板隐藏收件箱页签）
#     --deployment-file PATH
#                       部署台账文件（默认 <仓库>/DEPLOYMENT.md，存在即启用；none 显式禁用）
#     --title "..."     看板标题（默认 "<NAME> 工单看板"）
#     --agents a,b      承接代理名单（逗号分隔；不给=内置默认）
#     --statuses a,b    状态名单（逗号分隔；不给=内置默认）
#     --no-start        只 enable 不自启（默认 enable --now）
#     --force           覆盖已存在的配置文件与 unit 模板
#     --uninstall       停止 + 取消自启（保留配置文件）
#     -h, --help        显示本帮助
#
# 幂等：重复执行不破坏已有配置（模板只写一次，包装脚本每次刷新）；换机可重建。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  awk 'NR>=2 && /^set -e/ {exit} NR>=2 {sub(/^# ?/, ""); print}' "${BASH_SOURCE[0]}"
  exit 0
}

# ---- 默认值 ----
NAME="$(basename "$REPO_ROOT")"
PORT="8010"
FILE="$REPO_ROOT/.opencode/memory/tickets.md"
INBOX_FILE="$REPO_ROOT/.opencode/memory/inbox.md"
DEPLOYMENT_FILE="$REPO_ROOT/DEPLOYMENT.md"
TITLE=""
AGENTS=""
STATUSES=""
START=1
FORCE=0
UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --name)      NAME="${2:?--name 需要值}"; shift 2 ;;
    --port)      PORT="${2:?--port 需要值}"; shift 2 ;;
    --file)      FILE="${2:?--file 需要值}"; shift 2 ;;
    --inbox-file) INBOX_FILE="${2:?--inbox-file 需要值}"; shift 2 ;;
    --deployment-file) DEPLOYMENT_FILE="${2:?--deployment-file 需要值}"; shift 2 ;;
    --title)     TITLE="${2:?--title 需要值}"; shift 2 ;;
    --agents)    AGENTS="${2:?--agents 需要值}"; shift 2 ;;
    --statuses)  STATUSES="${2:?--statuses 需要值}"; shift 2 ;;
    --no-start)  START=0; shift ;;
    --force)     FORCE=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help)   usage ;;
    *) echo "✗ 未知参数: $1（-h 查看用法）" >&2; exit 1 ;;
  esac
done

# ---- 校验 ----
if ! [[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "✗ --name 只允许字母/数字/下划线/连字符: $NAME" >&2
  exit 1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "✗ --port 应为 1-65535 的整数: $PORT" >&2
  exit 1
fi
# 文件路径展开 ~ 并绝对化（相对路径按当前目录解析；避免 cwd 坑）
abs() { python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$1"; }
FILE="$(abs "$FILE")"
# 收件箱/台账：autodetect（文件存在→绝对路径；不存在或显式 none→null 隐藏页签）
if [ "$INBOX_FILE" = "none" ] || [ -z "$INBOX_FILE" ]; then INBOX_JSON="null"
elif [ -e "$INBOX_FILE" ]; then INBOX_JSON="$(abs "$INBOX_FILE")"
else INBOX_JSON="null"; fi
if [ "$DEPLOYMENT_FILE" = "none" ] || [ -z "$DEPLOYMENT_FILE" ]; then DEP_JSON="null"
elif [ -e "$DEPLOYMENT_FILE" ]; then DEP_JSON="$(abs "$DEPLOYMENT_FILE")"
else DEP_JSON="null"; fi

CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$CONFIG_HOME/ticketd"
CONFIG_FILE="$CONFIG_DIR/$NAME.json"
UNIT_DIR="$CONFIG_HOME/systemd/user"
UNIT_FILE="$UNIT_DIR/ticketd@.service"
BIN_DIR="$HOME/.local/bin"
WRAPPER="$BIN_DIR/ticketd"

read_port() {
  python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1],encoding="utf-8")).get("port",""))
except Exception: print("")' "$1"
}

# ---- 卸载 ----
if [ "$UNINSTALL" -eq 1 ]; then
  systemctl --user disable --now "ticketd@$NAME" 2>/dev/null || \
    systemctl --user disable "ticketd@$NAME" 2>/dev/null || true
  systemctl --user stop "ticketd@$NAME" 2>/dev/null || true
  systemctl --user reset-failed "ticketd@$NAME" 2>/dev/null || true
  systemctl --user daemon-reload 2>/dev/null || true
  echo "✔ 已停止并取消自启：ticketd@$NAME"
  echo "  配置文件保留（未删除）：$CONFIG_FILE"
  echo "  如需彻底清理： rm \"$CONFIG_FILE\""
  exit 0
fi

# ---- 确定生效端口（已有配置且不覆盖时以配置为准）----
if [ -f "$CONFIG_FILE" ] && [ "$FORCE" -eq 0 ]; then
  EFF_PORT="$(read_port "$CONFIG_FILE")"
  [ -n "$EFF_PORT" ] || EFF_PORT="$PORT"
else
  EFF_PORT="$PORT"
fi

# ---- 端口冲突检测：与其它实例配置重复 ----
if [ -d "$CONFIG_DIR" ]; then
  python3 - "$CONFIG_DIR" "$NAME" "$EFF_PORT" <<'PY'
import glob, json, os, sys
cfgdir, name, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
for p in sorted(glob.glob(os.path.join(cfgdir, "*.json"))):
    n = os.path.splitext(os.path.basename(p))[0]
    if n == name:
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if d.get("port") == port:
        sys.stderr.write(f"✗ 端口 {port} 已被实例 {n} 占用（{p}）；请用 --port 换端口\n")
        sys.exit(3)
PY
fi

# ---- 端口冲突检测：端口已被占用（本实例已在跑则跳过）----
if ! systemctl --user is-active --quiet "ticketd@$NAME" 2>/dev/null; then
  python3 - "$EFF_PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.stderr.write(f"✗ 端口 {port} 已被占用；请换端口（--port）或先停占用进程\n")
    sys.exit(4)
finally:
    s.close()
PY
fi

# ---- 生成配置文件（已存在默认不覆盖）----
mkdir -p "$CONFIG_DIR" "$UNIT_DIR" "$BIN_DIR"
if [ -z "$TITLE" ]; then TITLE="${NAME^^} 工单看板"; fi
if [ -f "$CONFIG_FILE" ] && [ "$FORCE" -eq 0 ]; then
  echo "· 配置已存在，跳过（用 --force 覆盖）：$CONFIG_FILE"
  # 幂等补齐：已存在的配置缺 inbox_file/deployment_file 时补写（不动其它字段）
  python3 - "$CONFIG_FILE" "$INBOX_JSON" "$DEP_JSON" <<'PY'
import json, sys
path, inbox, dep = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    cfg = json.load(open(path, encoding="utf-8"))
except Exception:
    sys.exit(0)  # 既有配置坏 JSON 时不动（启动时自会 fail-fast）
changed = False
for key, val in (("inbox_file", inbox), ("deployment_file", dep)):
    if key not in cfg:
        cfg[key] = val
        changed = True
if changed:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"· 已补充缺失的 inbox_file/deployment_file：{path}")
PY
else
  python3 - "$CONFIG_FILE" "$NAME" "$TITLE" "$FILE" "$EFF_PORT" "$AGENTS" "$STATUSES" "$INBOX_JSON" "$DEP_JSON" <<'PY'
import json, sys
out, name, title, file, port, agents, statuses, inbox, dep = sys.argv[1:10]
cfg = {"name": name, "title": title, "file": file, "host": "127.0.0.1", "port": int(port),
       "inbox_file": None if inbox == "null" else inbox,
       "deployment_file": None if dep == "null" else dep}
if agents:
    cfg["agents"] = [a for a in agents.split(",") if a]
if statuses:
    cfg["statuses"] = [s for s in statuses.split(",") if s]
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
  echo "· 已写入配置：$CONFIG_FILE"
fi

# ---- 生成 systemd user 模板 unit（只写一次）----
if [ -f "$UNIT_FILE" ] && [ "$FORCE" -eq 0 ]; then
  echo "· 模板 unit 已存在，跳过：$UNIT_FILE"
else
  cat > "$UNIT_FILE" <<'EOF'
[Unit]
Description=ticketd 工单看板（%i）
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/ticketd --config %h/.config/ticketd/%i.json
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
  echo "· 已写入模板 unit：$UNIT_FILE"
fi

# ---- 生成/刷新包装脚本（每次刷新，自愈）----
cat > "$WRAPPER" <<EOF
#!/bin/sh
# 由 install_ticketd.sh 生成（勿手改；重跑脚本会刷新）
export PYTHONUNBUFFERED=1
exec python3 "$REPO_ROOT/.opencode/tool/ticket_web.py" "\$@"
EOF
chmod +x "$WRAPPER"
echo "· 已刷新包装脚本：$WRAPPER"

# ---- PATH 提示（不改动用户 shell 配置）----
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "⚠ $BIN_DIR 不在 PATH 中（systemd 用绝对路径不受影响；手动运行 ticketd 需先加入 PATH）" ;;
esac

# ---- 注册并（可选）启动 ----
systemctl --user daemon-reload
if [ "$START" -eq 1 ]; then
  systemctl --user enable --now "ticketd@$NAME"
else
  systemctl --user enable "ticketd@$NAME"
fi

echo
echo "✔ 实例 ticketd@$NAME 已安装"
echo "  访问地址：   http://127.0.0.1:$EFF_PORT/"
echo "  状态查看：   systemctl --user status ticketd@$NAME"
echo "  重启：       systemctl --user restart ticketd@$NAME"
echo "  停止/禁自启：systemctl --user disable --now ticketd@$NAME"
echo "  日志：       journalctl --user -u ticketd@$NAME -f"
