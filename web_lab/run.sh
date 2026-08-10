#!/usr/bin/env bash
# 启动 UMVP 可视化测试台（零新增依赖）
# 用法: ./web_lab/run.sh [--port 8001] [--host 127.0.0.1] [--no-db]
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # web_lab/ 绝对路径
cd "$SELF"
ROOT="$(dirname "$SELF")"                  # 项目根（.mysql 数据目录在项目根）
PY="${PY:-$(conda run -n ai which python 2>/dev/null || command -v python)}"

# 除非显式 --no-db，否则确保项目自含 MySQL 实例已启动（端口 3307，仅绑定本机）
NO_DB=0
for a in "$@"; do [ "$a" = "--no-db" ] && NO_DB=1; done
if [ "$NO_DB" -eq 0 ]; then
  MYSQL_SOCK="$ROOT/.mysql/mysql.sock"
  if [ -S "$MYSQL_SOCK" ]; then
    echo "[run] 本地 MySQL 已在运行（127.0.0.1:3307）"
  else
    mkdir -p "$ROOT/.mysql"
    mysqld --datadir="$ROOT/.mysql/data" --socket="$MYSQL_SOCK" \
           --port=3307 --bind-address=127.0.0.1 \
           --log-error="$ROOT/.mysql/mysql.log" --daemonize
    echo "[run] 已拉起本地 MySQL（127.0.0.1:3307，umvp/123）"
  fi
fi

exec "$PY" server.py "$@"
