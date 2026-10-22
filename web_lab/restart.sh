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

# 本地 VLM 端点配置（vlm_env.sh 不入 git；存在即注入 VLM_ENDPOINT/VLM_MODEL/VLM_API_KEY，
# setsid 拉起的 server 进程继承，env 优先级高于链路参数——见 server.py _resolve_vlm_cfg）
[ -f "$SELF/vlm_env.sh" ] && . "$SELF/vlm_env.sh"

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

# ---- 2) 确保自含 MySQL 运行（同 run.sh 逻辑，共用 mysql_lib.sh；--no-db 跳过）----
NO_DB=0
for a in "$@"; do [ "$a" = "--no-db" ] && NO_DB=1; done
if [ "$NO_DB" -eq 0 ]; then
  . "$SELF/mysql_lib.sh"
  umvp_mysql_ensure "$ROOT" restart || {
    echo "[restart] 本地 MySQL 未能就绪，已中止启动（web 未拉起）；日志: $ROOT/.mysql/mysql.log"
    exit 1
  }
fi

# ---- 3) 后台拉起 web server，日志落 out/web_lab.log ----
# setsid：让 server 独立会话/进程组，不随调用方（终端/CI/工具）退出被连带杀掉；
# stdin 接 /dev/null，避免继承调用方管道导致调用方等待。
LOG="$ROOT/out/web_lab.log"
mkdir -p "$ROOT/out"
setsid "$PY" server.py "${SERVER_ARGS[@]}" < /dev/null >> "$LOG" 2>&1 &
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
