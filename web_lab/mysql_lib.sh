#!/usr/bin/env bash
# UMVP 自含 MySQL（127.0.0.1:3307）判活 / 启动 / 等待就绪 / 优雅关闭
# —— run.sh / restart.sh / stop.sh 共用
#
# 背景（INB-0053 / WO-0011）：WSL 整体重启或 mysqld 异常退出后，$ROOT/.mysql/mysql.sock
# 会残留，仅用 [ -S socket ] 判活会误判「已在运行」而跳过启动 → 8001 正常返回 200、
# 但 3307 无监听，页面存/取命名配置全部连接失败。
#
# 判活顺序（真实探活，不看文件存在）：
#   1) mysqladmin --socket=... ping   —— 真实握手；服务在跑即返回 0（凭据错也返回 0）
#   2) ss 查 127.0.0.1:3307 监听      —— 仅在没有 mysqladmin 时作为等价判据
#   3) 本实例 mysqld 进程兜底     —— 进程已在但尚未监听（启动中），避免拉第二个实例
#      （按可执行名 mysqld + cmdline 核 datadir/socket 匹配，不误判系统 3306/无关 shell）
# 探活失败即清理残留 socket/pid/lock 后再启动，启动后再次握手确认就绪。
#
# 用法（脚本里先定位 SELF/ROOT，再 source）：
#   . "$SELF/mysql_lib.sh"
#   umvp_mysql_ensure "$ROOT" run     # 已运行则跳过；未运行则清理残留并拉起，失败返回 1
#   umvp_mysql_alive  "$ROOT"         # 0=在运行（供 stop.sh 判活）
#   umvp_mysql_shutdown "$ROOT"       # mysqladmin 优雅关闭（root→umvp 凭据链），0=已受理
# 环境变量：MYSQL_USER / MYSQL_PASSWORD（与 db.py 默认 umvp/123 一致）
#           MYSQL_ROOT_PASSWORD（root 密码，出厂空密码时不设即可）

UMVP_MYSQL_PORT=3307

# socket 路径（数据目录固定为 $ROOT/.mysql/data，端口固定 3307）
umvp_mysql_sock() { printf '%s/.mysql/mysql.sock' "$1"; }

# 端口监听判据：0=127.0.0.1:3307 有 LISTEN
umvp_mysql_listening() {
  command -v ss >/dev/null 2>&1 || return 1
  ss -ltn 2>/dev/null | awk -v p=":${UMVP_MYSQL_PORT}" \
    'NR > 1 && $4 ~ (p "$") { found = 1 } END { exit !found }'
}

# 服务可应答：有 mysqladmin 时以真实握手为准（socket 不存在/不是本实例都会失败），
# 没有 mysqladmin 时退化为查 3307 监听。两者都不认「socket 文件存在」。
umvp_mysql_serving() {
  local root="$1" sock
  sock="$(umvp_mysql_sock "$root")"
  if command -v mysqladmin >/dev/null 2>&1; then
    # ping 的 0 表示「服务在跑」，即使凭据错误也返回 0；连不上才非 0
    mysqladmin --socket="$sock" \
      -u "${MYSQL_USER:-umvp}" -p"${MYSQL_PASSWORD:-123}" \
      ping >/dev/null 2>&1
    return $?
  fi
  umvp_mysql_listening
}

# 本实例 mysqld 进程 pid 列表（按可执行名 mysqld 精确匹配，再核 cmdline 里的
# 本实例 datadir/socket —— 不会误匹配系统 3306，也不会误匹配命令行里提到 socket 的 shell）
umvp_mysql_procs() {
  local root="$1" sock datadir pid cmd
  sock="$(umvp_mysql_sock "$root")"
  datadir="$root/.mysql/data"
  for pid in $(pgrep -x mysqld 2>/dev/null); do
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    case "$cmd" in
      *"$sock"*|*"$datadir"*) printf '%s\n' "$pid" ;;
    esac
  done
}

# 本实例存活（含「进程已起、尚未监听」的启动中窗口）
umvp_mysql_alive() {
  local root="$1" pids
  umvp_mysql_serving "$root" && return 0
  pids="$(umvp_mysql_procs "$root")"
  [ -n "$pids" ]
}

# 等待就绪（最多 15s）；超时给出可诊断提示并返回 1
umvp_mysql_wait_ready() {
  local root="$1" tag="${2:-mysql}" log i
  log="$root/.mysql/mysql.log"
  for i in $(seq 1 30); do
    umvp_mysql_serving "$root" && return 0
    sleep 0.5
  done
  echo "[$tag] 警告：MySQL 15 秒内未就绪，127.0.0.1:${UMVP_MYSQL_PORT} 无监听，presets 接口将不可用"
  echo "[$tag] 诊断：ss -ltnp | grep ':${UMVP_MYSQL_PORT}' ; tail -n 50 \"$log\""
  return 1
}

# 确保自含 MySQL 运行：已运行 → 跳过；未运行 → 清理残留后拉起并确认就绪
umvp_mysql_ensure() {
  local root="$1" tag="${2:-mysql}" sock log
  sock="$(umvp_mysql_sock "$root")"
  log="$root/.mysql/mysql.log"
  mkdir -p "$root/.mysql"

  if umvp_mysql_alive "$root"; then
    echo "[$tag] 本地 MySQL 已在运行（127.0.0.1:${UMVP_MYSQL_PORT}）"
    umvp_mysql_wait_ready "$root" "$tag" || return 1
    return 0
  fi

  # 探活失败：socket/lock/pid 都是上次异常退出的残留，不清掉 mysqld 会拒绝启动
  if [ -e "$sock" ] || [ -e "$sock.lock" ]; then
    echo "[$tag] 探活失败：$sock 为残留文件（3307 无监听），清理后启动"
  fi
  rm -f "$sock" "$sock.lock" "$root/.mysql/mysql.pid"

  echo "[$tag] 启动本地 MySQL（127.0.0.1:${UMVP_MYSQL_PORT}）..."
  if ! mysqld --datadir="$root/.mysql/data" --socket="$sock" \
         --port="$UMVP_MYSQL_PORT" --bind-address=127.0.0.1 \
         --log-error="$log" --daemonize; then
    echo "[$tag] 错误：mysqld 启动失败，请查看 $log（tail -n 50 \"$log\"）"
    return 1
  fi
  if ! umvp_mysql_wait_ready "$root" "$tag"; then
    echo "[$tag] 错误：本地 MySQL 未就绪，请查看 $log"
    return 1
  fi
  echo "[$tag] 已拉起本地 MySQL（127.0.0.1:${UMVP_MYSQL_PORT}，umvp/123）"
  return 0
}

# 优雅关闭（mysqladmin shutdown）凭据链（INB-0056 / WO-0012）：
#   1) root@localhost 经 socket —— 数据目录由 `mysqld --initialize-insecure` 出厂，
#      root 空密码、仅本机可达（socket + loopback TCP 皆可，见 INB-0057），具 SHUTDOWN 权；
#      换机按同样命令重建 data 目录后该路径天然成立，无需补授。若改过 root 密码，
#      设 MYSQL_ROOT_PASSWORD 环境变量即可恢复此路径。
#   2) umvp 账号经 socket —— 仅 USAGE ON *.* + umvp.* 全权，默认无全局 SHUTDOWN，
#      预期失败；留作将来补授权限后的路径。
# 返回 0=服务端已受理优雅关闭；非 0=凭据路径均不可用（调用方自行走 kill -TERM 兜底，
# MySQL 8 下 SIGTERM 同为优雅关闭）。
umvp_mysql_shutdown() {
  local root="$1" sock
  sock="$(umvp_mysql_sock "$root")"
  command -v mysqladmin >/dev/null 2>&1 || return 1
  if [ -n "${MYSQL_ROOT_PASSWORD:-}" ]; then
    mysqladmin --socket="$sock" -uroot -p"$MYSQL_ROOT_PASSWORD" \
      shutdown >/dev/null 2>&1 && return 0
  else
    mysqladmin --socket="$sock" -uroot shutdown >/dev/null 2>&1 && return 0
  fi
  mysqladmin --socket="$sock" \
    -u "${MYSQL_USER:-umvp}" -p"${MYSQL_PASSWORD:-123}" \
    shutdown >/dev/null 2>&1 && return 0
  return 1
}
