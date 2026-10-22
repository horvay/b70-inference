#!/usr/bin/env bash
# guarded.sh <log> <command...>: run a GPU command in its own process group, and kill the whole group at the first sign
# that it is hurting the machine:
#   - an xe fault / hang / reset line for the B70 in the kernel log (a faulting kernel wedges the xe driver if it keeps
#     going; see memory b70-gpu-faults-can-freeze-machine). Kernel lines from the run go to <log>.kernel;
#   - low RAM: MemAvailable under RAM_FLOOR_GB (default 4) or memory pressure (PSI "full" avg10) over PSI_MAX
#     (default 40; loading alone reaches ~20 while it streams 68 GB of shards, systemd-oomd acts at 50). The server pins ~35 GB of host RAM that cannot be swapped; on 2026-10-04 it and a video job
#     together ran the machine out of memory and systemd-oomd killed the whole terminal session.
# Before starting it refuses to run with less than RAM_START_GB (default 45) available. Who else may use the B70 is the
# b70q queue's business (~/ai/b70q): run this inside a queue job (b70q submit / b70-lease run), not beside one.
# The command runs with oom_score_adj 1000, so the kernel OOM killer takes it before anything of the user's.
set -u
LOG=$1; shift
RAM_FLOOR_GB=${RAM_FLOOR_GB:-4}; PSI_MAX=${PSI_MAX:-40}; RAM_START_GB=${RAM_START_GB:-45}
: > "$LOG.kernel"

avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }

psi_full() { awk '/^full/ {sub("avg10=", "", $2); printf "%d", $2}' /proc/pressure/memory 2>/dev/null; }
# the previous job's model load can leave the 10 s pressure average high for a moment: let it settle first
for _ in $(seq 60); do [ "$(psi_full)" -lt $((PSI_MAX / 2)) ] && break; sleep 2; done
a=$(avail_gb)
if [ "$a" -lt "$RAM_START_GB" ]; then
    echo "GUARD: not starting, only $a GB RAM available (need $RAM_START_GB)" | tee -a "$LOG"; exit 98
fi

START=$(stat -c %s "$LOG" 2>/dev/null || echo 0)     # only this run's GUARD lines count at the end
setsid bash -c 'echo 1000 > /proc/self/oom_score_adj; exec "$@"' guarded "$@" >> "$LOG" 2>&1 &
CMD=$!

stop_run() {      # $1: reason
    echo "GUARD: $1, stopping the run" >> "$LOG"
    {   # who was using the memory, for the post-mortem
        echo "GUARD top RAM users at $(date +%T):"
        ps -eo pid,rss,args --sort=-rss | head -9 | awk 'NR == 1 {print "  " $0; next} {printf "  %s %.1fG %s\n", $1, $2 / 1048576, substr($0, index($0, $3), 110)}'
        free -g | sed 's/^/  /'
    } >> "$LOG" 2>/dev/null
    kill -TERM -- -"$CMD" 2>/dev/null; sleep 3; kill -KILL -- -"$CMD" 2>/dev/null
}

GUARD_T0=$(date +%s)
(
    # -n 0 alone replayed old lines after a journal rotation (a 02:30 fault stopped a 02:33 start): also skip any line
    # stamped before this run started
    journalctl -k -f -n 0 --since "@$GUARD_T0" --no-pager 2>/dev/null | while IFS= read -r line; do
        ts=$(date -d "$(echo "$line" | cut -d' ' -f1-3)" +%s 2>/dev/null || echo "$GUARD_T0")
        [ "$ts" -lt "$GUARD_T0" ] && continue
        case "$line" in
            *"xe 0000:0a:00.0"*)
                echo "$line" >> "$LOG.kernel"
                case "$line" in
                    *Fault*|*"CAT error"*|*hang*|*Hang*|*reset*|*Reset*|*wedged*|*Timedout*)
                        stop_run "B70 fault in the kernel log: $line" ;;
                esac
                ;;
        esac
    done
) &
WATCH=$!

(
    while kill -0 "$CMD" 2>/dev/null; do
        a=$(avail_gb)
        p=$(awk '/^full/ {sub("avg10=", "", $2); printf "%d", $2}' /proc/pressure/memory 2>/dev/null)
        if [ "$a" -lt "$RAM_FLOOR_GB" ]; then stop_run "RAM low ($a GB available, floor $RAM_FLOOR_GB)"; break; fi
        if [ "${p:-0}" -gt "$PSI_MAX" ]; then stop_run "memory pressure ${p}% (full avg10, max $PSI_MAX), $a GB RAM available"; break; fi
        sleep 1
    done
) &
MEMWATCH=$!

wait "$CMD"; RC=$?
sleep 1
pkill -P "$WATCH" journalctl 2>/dev/null; kill "$WATCH" "$MEMWATCH" 2>/dev/null
g=$(tail -c +$((START + 1)) "$LOG" | grep '^GUARD:' | tail -1)
[ -n "$g" ] && { echo "GUARD tripped: $g"; exit 99; }
exit $RC
