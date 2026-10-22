#!/usr/bin/env bash
# 启动 UMVP 可视化测试台（零新增依赖）
# 用法: ./web_lab/run.sh [--port 8001] [--host 127.0.0.1] [--no-db]
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"     # web_lab/ 绝对路径
cd "$SELF"
ROOT="$(dirname "$SELF")"                  # 项目根（.mysql 数据目录在项目根）
PY="${PY:-$(conda run -n ai which python 2>/dev/null || command -v python)}"

# 本地 VLM 端点配置（vlm_env.sh 不入 git；存在即注入 VLM_ENDPOINT/VLM_MODEL/VLM_API_KEY）
[ -f "$SELF/vlm_env.sh" ] && . "$SELF/vlm_env.sh"

# 除非显式 --no-db，否则确保项目自含 MySQL 实例已启动（端口 3307，仅绑定本机）
# 判活/启动逻辑与 restart.sh 共用 mysql_lib.sh（真实探活，残留 socket 自动清理）
NO_DB=0
for a in "$@"; do [ "$a" = "--no-db" ] && NO_DB=1; done
if [ "$NO_DB" -eq 0 ]; then
  . "$SELF/mysql_lib.sh"
  umvp_mysql_ensure "$ROOT" run || {
    echo "[run] 本地 MySQL 未能就绪，已中止启动；日志: $ROOT/.mysql/mysql.log"
    exit 1
  }
fi

exec "$PY" server.py "$@"
