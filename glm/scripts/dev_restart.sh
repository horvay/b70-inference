#!/usr/bin/env bash
# Restart the dev server (xpu/devserver.py) under the guard, through the b70q queue (~/ai/b70q) as owner glm, and wait
# until it is ready. Args go to devserver.py. It starts once the card and RAM_START_GB (30) of RAM are free; the queue
# stops it after IDLE (15m) idle on the GPU while another job waits.
set -u
W=$HOME/work/glm-b70
cd "$W"
python3 xpu/dev.py --quit >/dev/null 2>&1
# the previous dev server's ticket, if it is still around (it ends when its server exits)
[ -s run/dev.ticket ] && { b70q kill "$(cat run/dev.ticket)" >/dev/null 2>&1; rm -f run/dev.ticket; }
ticket() {
    curl -s -m 5 --unix-socket "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/b70q.sock" "http://b70q/api/tickets/$1" |
        python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["ticket"]["state"], d.get("position") or "")' 2>/dev/null
}
off=$(stat -c %s run/dev.log 2>/dev/null || echo 0)
export RAM_START_GB=${RAM_START_GB:-30}
id=$(b70q serve glm --idle "${IDLE:-15m}" --ram-gib "$RAM_START_GB" --note "GLM devserver" --log "$W/run/dev.rc" -- \
    scripts/guarded.sh run/dev.log scripts/in_rootfs.sh --gpu python3 /w/xpu/devserver.py "$@" | head -1)
echo "$id" > run/dev.ticket
said=""
while :; do               # its turn first: no time limit while it waits in line
    read -r state pos <<< "$(ticket "$id")"
    [ "$state" = waiting ] || break
    [ "$pos" != "$said" ] && { echo "waiting for the B70: $pos in line (b70q #$id)"; said=$pos; }
    sleep 5
done
[ "$state" = granted ] || { echo "b70q #$id ended before it started: $state (b70q log $id)"; exit 1; }
for _ in $(seq 150); do
    sleep 2
    new=$(tail -c +$((off + 1)) run/dev.log)
    echo "$new" | grep -q "dev server ready" && { echo "$new" | grep "glm_xpu" | tail -3; exit 0; }
    echo "$new" | grep -qE "GUARD|Traceback" && { echo "$new" | tail -5; exit 1; }
done
echo "dev server did not come up"; exit 1
