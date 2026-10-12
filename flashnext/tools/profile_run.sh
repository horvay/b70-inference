#!/usr/bin/env bash
# One guarded server run that profiles expert routing on game-master-style turns (route_profile.py), phase A then the
# held-out phase B (PHASES="A B G H" adds general-use prompts, G to build and H held out), then stops. Writes run/route_A.json and run/route_B.json (cumulative per-layer routing counts),
# run/profile_A.json / profile_B.json (per-request timing), log run/profile.log. TAG names the outputs (default none).
# Pass HOT_JSON=... to serve with a different hot list.
set -u
T=$HOME/work/trellis-b70; LOG=$T/run/profile.log; : > "$LOG"; TAG=${TAG:-}
export EXL3_MOE_PROFILE=/run/trellis/route_profile.json
rm -f "$T/run/route_profile.json"
"$T/scripts/guarded.sh" "$LOG" "$T/scripts/serve_b70.sh" &
G=$!
t0=$(date +%s)
until curl -s --max-time 2 http://127.0.0.1:30250/v1/models >/dev/null 2>&1; do
    kill -0 $G 2>/dev/null || { echo "PROFILE: server ended before ready" | tee -a "$LOG"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "PROFILE: not ready after 15 min"; break; }
    sleep 3
done
echo "PROFILE: ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
GM=$HOME/work/neverending_quest/src/play/gm_voice.md
for ph in ${PHASES:-A B}; do                 # A/B: game-master turns; G/H: general use, no system prompt
    case $ph in A|B) SYS=$GM ;; *) SYS=- ;; esac
    python3 "$T/scripts/route_profile.py" $ph "$SYS" "$T/run/profile_$ph$TAG.json" "${MAX_TOKENS:-250}" 2>&1 | tee -a "$LOG"
    cp "$T/run/route_profile.json" "$T/run/route_$ph$TAG.json" 2>/dev/null || echo "PROFILE: no routing counts written" | tee -a "$LOG"
done
for p in $(ps -eo pid,args | awk '/[s]glang\.launch_server/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,stat,args | awk '$2 !~ /Z/ && /[s]glang|[b]wrap --bind \/home\/horvay\/work\/trellis-b70/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
wait $G 2>/dev/null
echo "PROFILE: done" | tee -a "$LOG"
