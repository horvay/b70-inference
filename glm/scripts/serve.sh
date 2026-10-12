#!/usr/bin/env bash
# Start / stop the OpenAI-compatible GLM server (xpu/serve.py) on the B70, under scripts/guarded.sh.
#   scripts/serve.sh [start] [serve.py args...]   stop the dev server if it runs, start serve.py, wait until it is ready
#   scripts/serve.sh stop                          stop serve.py (POST /quit, then the guard's process group)
#   scripts/serve.sh status
# Port: SERVE_PORT (default 8053). Log: run/serve.log (live speed: run/live.log). RAM_START_GB as in guarded.sh.
set -u
W=$HOME/work/glm-b70
cd "$W"
PORT=${SERVE_PORT:-8053}
LOG=run/serve.log
cmd=${1:-start}
[ $# -gt 0 ] && shift

avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
running() { pgrep -f "^python3 /w/xpu/serve.py" >/dev/null; }

case "$cmd" in
status)
    if running; then curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo; else echo "not running"; fi
    exit 0 ;;
stop)
    running || { echo "not running"; exit 0; }
    curl -s -m 5 -X POST "http://127.0.0.1:$PORT/quit" >/dev/null
    for _ in $(seq 60); do running || break; sleep 1; done
    if running; then          # stuck (a broken Level Zero context can hang the exit): stop the guard's group
        for g in $(pgrep -f "guarded.sh $LOG"); do kill -TERM -- -"$g" 2>/dev/null; kill "$g" 2>/dev/null; done
        sleep 3
        pkill -KILL -f "^python3 /w/xpu/serve.py" 2>/dev/null
    fi
    for g in $(pgrep -f "guarded.sh $LOG"); do pgrep -P "$g" >/dev/null 2>&1 || kill "$g" 2>/dev/null; done
    echo "stopped"; exit 0 ;;
start) ;;
*) sed -n '2,6p' "$0"; exit 2 ;;
esac

if running; then echo "already running"; curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo; exit 0; fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "port $PORT is taken"; exit 1; fi
# one model on the B70 at a time: stop the dev server first and wait for its pinned RAM to come back
if pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null; then
    echo "stopping the dev server"
    timeout 60 python3 xpu/dev.py --quit >/dev/null 2>&1
    for _ in $(seq 60); do pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null || break; sleep 2; done
    pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null && { echo "the dev server did not stop"; exit 1; }
fi
for _ in $(seq 60); do
    [ "$(avail_gb)" -ge "${RAM_START_GB:-45}" ] && break
    sleep 2
done
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
(RAM_START_GB=${RAM_START_GB:-45} setsid nohup scripts/guarded.sh "$LOG" scripts/in_rootfs.sh --gpu \
    python3 /w/xpu/serve.py --port "$PORT" "$@" > run/serve.rc 2>&1 &)
for _ in $(seq 180); do
    sleep 2
    new=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null)
    echo "$new" | grep -q "server ready" && { echo "$new" | grep -E "glm_xpu|serve:" | tail -4; exit 0; }
    echo "$new" | grep -qE "GUARD|Traceback" && { echo "$new" | tail -8; exit 1; }
done
echo "server did not come up (see $LOG)"; exit 1
