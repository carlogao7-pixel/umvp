#!/usr/bin/env bash
# 一键重启 UMVP 可视化测试台：停旧 web 进程 → 确保自含 MySQL → 后台拉起 server
# 用法: ./web_lab/restart.sh [--port 8001] [--host 127.0.0.1] [--no-db] [--stop-mysql]
#   默认保留自含 MySQL（只在未启动时拉起）；--stop-mysql 时连 MySQL 一起重启。
#   后台运行，日志写 out/web_lab.log；--no-db 时不启动也不依赖 MySQL。
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # web_lab/ 绝对路径
cd "$SELF"
ROOT="$(dirname "$SELF")"                  # 项目根
PY="${PY:-$(conda run -n ai which python 2>/dev/null || command -v python)}"

# 解析参数：--stop-mysql 单独吃下，其余透传给 server.py
STOP_MYSQL=0
SERVER_ARGS=()
for a in "$@"; do
  case "$a" in
    --stop-mysql) STOP_MYSQL=1 ;;
    *) SERVER_ARGS+=("$a") ;;
  esac
done

# 从透传参数里取 --port（支持 --port N 与 --port=N）
PORT=8001
PREV=""
for a in "${SERVER_ARGS[@]}"; do
  case "$a" in
    --port=*) PORT="${a#--port=}" ;;
    --port) PREV=port ;;
    *) [ "$PREV" = port ] && { PORT="$a"; PREV=""; } ;;
  esac
done

# ---- 1) 停旧 web 进程 ----
if [ "$STOP_MYSQL" -eq 1 ]; then
  "$SELF/stop.sh" --port "$PORT"
else
  "$SELF/stop.sh" --keep-mysql --port "$PORT"
fi

# ---- 2) 确保自含 MySQL 运行（同 run.sh 逻辑，非 --no-db）----
NO_DB=0
for a in "$@"; do [ "$a" = "--no-db" ] && NO_DB=1; done
if [ "$NO_DB" -eq 0 ]; then
  MYSQL_SOCK="$ROOT/.mysql/mysql.sock"
  if [ -S "$MYSQL_SOCK" ] && pgrep -f "\.mysql/mysql\.sock" >/dev/null; then
    echo "[restart] 本地 MySQL 已在运行（127.0.0.1:3307）"
  else
    mkdir -p "$ROOT/.mysql"
    # 进程已确认不在，清理残留 socket/pid/lock，防 mysqld 启动失败
    rm -f "$MYSQL_SOCK" "$ROOT/.mysql/mysql.pid" "$ROOT/.mysql/mysql.sock.lock"
    mysqld --datadir="$ROOT/.mysql/data" --socket="$MYSQL_SOCK" \
           --port=3307 --bind-address=127.0.0.1 \
           --log-error="$ROOT/.mysql/mysql.log" --daemonize
    echo "[restart] 已拉起本地 MySQL（127.0.0.1:3307，umvp/123）"
    # 等待 MySQL 就绪（最多 ~15s）；未就绪只警告不阻断（DB 不可用不影响 server 启动）
    ready=0
    for i in $(seq 1 30); do
      if ss -ltn 2>/dev/null | grep -q ":3307 "; then ready=1; break; fi
      sleep 0.5
    done
    [ "$ready" -eq 1 ] || echo "[restart] 警告：MySQL 15 秒内未就绪，presets 接口可能不可用"
  fi
fi

# ---- 3) 后台拉起 web server，日志落 out/web_lab.log ----
LOG="$ROOT/out/web_lab.log"
mkdir -p "$ROOT/out"
nohup "$PY" server.py "${SERVER_ARGS[@]}" >> "$LOG" 2>&1 &
NEW_PID=$!
echo "[restart] web_lab 已后台启动（pid=$NEW_PID，日志: out/web_lab.log）"

# 等待端口就绪（curl 探测；无 curl 时退化为等 2s）
if command -v curl >/dev/null 2>&1; then
  for i in $(seq 1 40); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then
      echo "[restart] 就绪: http://127.0.0.1:$PORT/（curl 探测通过）"
      exit 0
    fi
    kill -0 "$NEW_PID" 2>/dev/null || { echo "[restart] 进程异常退出，请查看日志: $LOG"; exit 1; }
    sleep 0.5
  done
  echo "[restart] 20 秒内未就绪，请查看日志: $LOG"
  exit 1
else
  sleep 2
  echo "[restart] curl 不可用，跳过就绪探测（进程 pid=$NEW_PID）"
fi
