#!/usr/bin/env bash
# One server run that profiles expert routing on game-master-style turns (route_profile.py), phase A then the held-out
# phase B (PHASES="A B G H" adds general-use prompts, G to build and H held out), then stops. Writes
# state/run/sglang/route_{A,B}.json (cumulative per-layer routing counts), profile_{A,B}.json (per-request timing)
# and profile.log. route_to_hot.py turns the counts into hot/RAM lists.
#   GM_PROMPT=<system prompt file> tools/profile_run.sh <model dir>      (TAG names the outputs; PORT, default 30250)
set -u
F=$(dirname "$(dirname "$(readlink -f "$0")")"); B70=$(dirname "$F")/b70; R=$(dirname "$F")/state/run/sglang
MODEL=$1; PORT=${PORT:-30250}; TAG=${TAG:-}; LOG=$R/profile.log
mkdir -p "$R"; : > "$LOG"; rm -f "$R/route_profile.json"
EXL3_MOE_PROFILE=/run/b70/route_profile.json "$B70" serve "$MODEL" --port "$PORT" --all-ram >> "$LOG" 2>&1 &
G=$!
t0=$(date +%s)
until curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    kill -0 $G 2>/dev/null || { echo "PROFILE: server ended before ready" | tee -a "$LOG"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "PROFILE: not ready after 15 min"; break; }
    sleep 3
done
echo "PROFILE: ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
for ph in ${PHASES:-A B}; do                 # A/B: game-master turns; G/H: general use, no system prompt
    case $ph in A|B) SYS=${GM_PROMPT:?GM_PROMPT=<system prompt file> is needed for phases A and B} ;; *) SYS=- ;; esac
    python3 "$F/tools/route_profile.py" $ph "$SYS" "$R/profile_$ph$TAG.json" "${MAX_TOKENS:-250}" 2>&1 | tee -a "$LOG"
    cp "$R/route_profile.json" "$R/route_$ph$TAG.json" 2>/dev/null || echo "PROFILE: no routing counts written" | tee -a "$LOG"
done
"$B70" stop
wait $G 2>/dev/null
echo "PROFILE: done" | tee -a "$LOG"
