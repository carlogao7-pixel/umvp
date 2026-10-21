#!/usr/bin/env bash
# 停止 UMVP 可视化测试台（默认连自含 MySQL 一起停；--keep-mysql 保留）
# 用法: ./web_lab/stop.sh [--port 8001] [--keep-mysql]
#   --port N        web_lab 端口（默认 8001；找不到该端口监听时按进程名兜底定位）
#   --keep-mysql    只停 web_lab，保留自含 MySQL 运行
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # web_lab/ 绝对路径
cd "$SELF"
ROOT="$(dirname "$SELF")"                  # 项目根（.mysql 数据目录在项目根）

PORT=8001
KEEP_MYSQL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --keep-mysql|--no-mysql) KEEP_MYSQL=1; shift ;;
    *) echo "[stop] 未知参数: $1（支持 --port N / --keep-mysql）"; exit 1 ;;
  esac
done

# ---- 1) 停 web_lab server：先按端口找监听 PID，找不到再按进程名兜底 ----
PID=""
if command -v lsof >/dev/null 2>&1; then
  PID="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true)"
fi
if [ -z "$PID" ] && command -v ss >/dev/null 2>&1; then
  PID="$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ (p "$") { for (i=1;i<=NF;i++) if ($i ~ /pid=/) { sub(/^.*pid=/, "", $i); sub(/,.*/, "", $i); print $i; exit } }' || true)"
fi
# 兜底：按进程名找 web_lab server（自定义端口 / 旧实例同样生效）
if [ -z "$PID" ]; then
  PID="$(pgrep -f "[s]erver\.py" | head -1 || true)"
  [ -n "$PID" ] && echo "[stop] 端口 $PORT 无监听，按进程名定位到 pid=$PID"
fi

if [ -n "$PID" ]; then
  echo "[stop] 停止 web_lab（pid=$PID，port=$PORT）..."
  kill "$PID" 2>/dev/null || true
  for i in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" 2>/dev/null || true
    echo "[stop] 进程未及时退出，已强制结束"
  fi
  echo "[stop] web_lab 已停止"
else
  echo "[stop] web_lab 未在运行（端口 $PORT 无监听）"
fi

# ---- 2) 停自含 MySQL（127.0.0.1:3307，socket .mysql/mysql.sock）----
if [ "$KEEP_MYSQL" -eq 1 ]; then
  echo "[stop] --keep-mysql：保留自含 MySQL"
  exit 0
fi

MYSQL_SOCK="$ROOT/.mysql/mysql.sock"
. "$SELF/mysql_lib.sh"
if umvp_mysql_alive "$ROOT"; then
  MYSQL_PID="$(umvp_mysql_procs "$ROOT" | head -1 || true)"
  if [ -n "$MYSQL_PID" ]; then
    echo "[stop] 停止自含 MySQL（pid=$MYSQL_PID，127.0.0.1:3307）..."
    # 优先 mysqladmin 优雅关闭（root@localhost socket → umvp socket 凭据链，
    # 见 mysql_lib.sh umvp_mysql_shutdown）；凭据均不可用才 kill -TERM 兜底
    if umvp_mysql_shutdown "$ROOT"; then
      echo "[stop] 已发出 mysqladmin 优雅关闭，等待退出..."
    else
      echo "[stop] mysqladmin 优雅关闭不可用（root/umvp 凭据连接失败或无 SHUTDOWN 权限），改用 kill -TERM（MySQL 8 下同为优雅关闭）"
    fi
    for i in $(seq 1 20); do
      kill -0 "$MYSQL_PID" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$MYSQL_PID" 2>/dev/null; then
      kill -TERM "$MYSQL_PID" 2>/dev/null || true
      sleep 3
    fi
    if kill -0 "$MYSQL_PID" 2>/dev/null; then
      kill -9 "$MYSQL_PID" 2>/dev/null || true
    fi
    echo "[stop] 自含 MySQL 已停止"
  else
    # 极端：3307 有监听/握手成功但进程名不是本实例（例如别的 mysqld），不误杀
    echo "[stop] 127.0.0.1:3307 有应答但未定位到本实例 mysqld（跳过）"
  fi
elif [ -e "$MYSQL_SOCK" ]; then
  echo "[stop] 发现残留 mysql.sock 但 3307 无应答（忽略，下次启动会自动清理）"
else
  echo "[stop] 自含 MySQL 未在运行"
fi
