#!/usr/bin/env bash
# Start / stop the OpenAI-compatible GLM server (xpu/serve.py) on the B70, under scripts/guarded.sh, through the b70q
# queue (~/ai/b70q) as owner glm: it starts once the card and RAM_START_GB (45) of RAM are free, and the queue stops it
# after IDLE (15m) idle on the GPU while another job waits.
#   scripts/serve.sh [start] [serve.py args...]   stop the dev server if it runs, queue serve.py, wait until it is ready
#   scripts/serve.sh stop                          stop serve.py (POST /quit, then its b70q ticket)
#   scripts/serve.sh status
# Port: SERVE_PORT (default 8053). Log: run/serve.log (live speed: run/live.log).
set -u
W=$(dirname "$(dirname "$(readlink -f "$0")")")     # this glm/ folder
cd "$W"
PORT=${SERVE_PORT:-8053}
LOG=run/serve.log
cmd=${1:-start}
[ $# -gt 0 ] && shift

# state of b70q ticket $1 (waiting / granted / done / failed / cancelled) and its place in line
ticket() {
    curl -s -m 5 --unix-socket "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/b70q.sock" "http://b70q/api/tickets/$1" |
        python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["ticket"]["state"], d.get("position") or "")' 2>/dev/null
}
running() { pgrep -f "^python3 /w/xpu/serve.py" >/dev/null; }

case "$cmd" in
status)
    if running; then curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo; else echo "not running"; fi
    exit 0 ;;
stop)
    if running; then
        curl -s -m 5 -X POST "http://127.0.0.1:$PORT/quit" >/dev/null
        for _ in $(seq 60); do running || break; sleep 1; done
    fi
    # stuck (a broken Level Zero context can hang the exit), or still waiting in line: end its b70q ticket, which stops
    # every process of it
    [ -s run/serve.ticket ] && { b70q kill "$(cat run/serve.ticket)" >/dev/null 2>&1; rm -f run/serve.ticket; }
    echo "stopped"; exit 0 ;;
start) ;;
*) sed -n '2,6p' "$0"; exit 2 ;;
esac

if running; then echo "already running"; curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo; exit 0; fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "port $PORT is taken"; exit 1; fi
# the dev server would keep the card (and its pinned RAM) until it sat idle for IDLE: stop it first
if pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null; then
    echo "stopping the dev server"
    timeout 60 python3 xpu/dev.py --quit >/dev/null 2>&1
    for _ in $(seq 60); do pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null || break; sleep 2; done
    pgrep -f "^python3 /w/xpu/devserver.py" >/dev/null && { echo "the dev server did not stop"; exit 1; }
fi
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
export RAM_START_GB=${RAM_START_GB:-45}
id=$(b70q serve glm --idle "${IDLE:-15m}" --ram-gib "$RAM_START_GB" --note "GLM serve.py :$PORT" --log "$W/run/serve.rc" -- \
    scripts/guarded.sh "$LOG" scripts/in_rootfs.sh --gpu python3 /w/xpu/serve.py --port "$PORT" "$@" | head -1)
echo "$id" > run/serve.ticket
said=""
while :; do               # its turn first: no time limit while it waits in line
    read -r state pos <<< "$(ticket "$id")"
    [ "$state" = waiting ] || break
    [ "$pos" != "$said" ] && { echo "waiting for the B70: $pos in line (b70q #$id)"; said=$pos; }
    sleep 5
done
[ "$state" = granted ] || { echo "b70q #$id ended before it started: $state (b70q log $id)"; exit 1; }
for _ in $(seq 180); do
    sleep 2
    new=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null)
    echo "$new" | grep -q "server ready" && { echo "$new" | grep -E "glm_xpu|serve:" | tail -4; exit 0; }
    echo "$new" | grep -qE "GUARD|Traceback" && { echo "$new" | tail -8; exit 1; }
done
echo "server did not come up (see $LOG)"; exit 1
