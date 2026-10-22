#!/usr/bin/env bash
# Start / stop the DeepSeek-V4.1-Flash chat server (xpu/serve.py: chat page at /, OpenAI API at /v1) on the B70, under
# scripts/guarded.sh, through the b70q queue (~/ai/b70q) as owner dsv41: it starts once the card and RAM_START_GB (34)
# of RAM are free, and the queue stops it after IDLE (15m) idle on the GPU while another job waits.
#   scripts/serve.sh [start] [serve.py args...]   queue serve.py, wait until it is ready
#   scripts/serve.sh stop                          stop serve.py (POST /quit, then its b70q ticket)
#   scripts/serve.sh status
# Port: SERVE_PORT (default 8054). Log: run/serve.log. Runs with the tested defaults: DSpark (block boost 0.3, exact
# acceptance), residency boost DSV_BOOST=0.3, stage areas DSV_STAGE_GB=0.5.
set -u
D=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$D"
PORT=${SERVE_PORT:-8054}
LOG=run/serve.log
cmd=${1:-start}
[ $# -gt 0 ] && shift

# state of b70q ticket $1 (waiting / granted / done / failed / cancelled) and its place in line
ticket() {
    curl -s -m 5 --unix-socket "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/b70q.sock" "http://b70q/api/tickets/$1" |
        python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["ticket"]["state"], d.get("position") or "")' 2>/dev/null
}
live() { [ -s run/serve.ticket ] && case "$(ticket "$(cat run/serve.ticket)")" in waiting*|granted*) return 0 ;; esac; return 1; }

case "$cmd" in
status)
    if live; then echo "b70q #$(cat run/serve.ticket): $(ticket "$(cat run/serve.ticket)")"; curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo
    else echo "not running"; fi
    exit 0 ;;
stop)
    if live; then
        curl -s -m 5 -X POST "http://127.0.0.1:$PORT/quit" >/dev/null
        for _ in $(seq 60); do live || break; sleep 1; done
        # stuck (a broken Level Zero context can hang the exit), or still waiting in line: end its ticket
        live && b70q kill "$(cat run/serve.ticket)" >/dev/null 2>&1
    fi
    rm -f run/serve.ticket; echo "stopped"; exit 0 ;;
start) ;;
*) sed -n '2,8p' "$0"; exit 2 ;;
esac

if live; then echo "already running (b70q #$(cat run/serve.ticket))"; exit 0; fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "port $PORT is taken"; exit 1; fi
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
export RAM_START_GB=${RAM_START_GB:-34}
id=$(b70q serve dsv41 --idle "${IDLE:-15m}" --ram-gib "$RAM_START_GB" --note "DSV4.1 chat server :$PORT" -- \
    env DSV_BOOST="${DSV_BOOST:-0.3}" DSV_STAGE_GB="${DSV_STAGE_GB:-0.5}" \
    scripts/guarded.sh "$LOG" scripts/in_rootfs.sh --gpu python3 -X faulthandler /w/dsv41/xpu/serve.py --port "$PORT" "$@" | head -1)
echo "$id" > run/serve.ticket
said=""
while :; do               # its turn first: no time limit while it waits in line
    read -r state pos <<< "$(ticket "$id")"
    [ "$state" = waiting ] || break
    [ "$pos" != "$said" ] && { echo "waiting for the B70: $pos in line (b70q #$id)"; said=$pos; }
    sleep 5
done
[ "$state" = granted ] || { echo "b70q #$id ended before it started: $state (b70q log $id)"; exit 1; }
for _ in $(seq 300); do
    sleep 2
    new=$(tail -c +$((off + 1)) "$LOG" 2>/dev/null)
    echo "$new" | grep -q "server ready" && { echo "$new" | grep -E "serve:|dsv:" | tail -4; exit 0; }
    echo "$new" | grep -qE "GUARD|Traceback" && { echo "$new" | tail -12; exit 1; }
    live || { echo "b70q #$id ended (b70q log $id)"; echo "$new" | tail -12; exit 1; }
done
echo "server did not come up (see $LOG)"; exit 1
