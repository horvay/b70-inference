#!/usr/bin/env bash
# prof_live.sh <id> <decode|prefill> [max_tokens]: torch-profiler trace (CPU + XPU kernels) of one request against the
# running server (started with GRAPHS off to see kernels: EXL3_PROF_SCOPES=1 b70 serve ...); trace in
# state/run/sglang/prof/<id>-TP-0.trace.json.gz (tools/prof_cats.py). SGLang maps "GPU" to CUDA, so the
# XPU kernel activity must be asked for as "XPU" (the 2026-10-05 traces had driver calls only because of that).
set -u
F=$(dirname "$(dirname "$(readlink -f "$0")")"); R=$(dirname "$F")/state/run/sglang
API=${API:-http://127.0.0.1:30250}; ID=$1; KIND=$2; N=${3:-40}
mkdir -p "$R/prof"; rm -f "$R/prof/$ID"-*
curl -s -X POST $API/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/run/b70/prof","activities":["CPU","XPU"],"with_stack":false,"record_shapes":false,"profile_id":"'$ID'"}' >/dev/null
s=$(date +%s.%N)
if [ "$KIND" = prefill ]; then
    python3 - "$N" "$F/bench/long_prompt_16k.json" "$API" <<'PY'
import json, sys, time, urllib.request
p = json.load(open(sys.argv[2]))
p["messages"][0]["content"] = f"Copy {time.time()}\n" + p["messages"][0]["content"]
p["max_tokens"] = int(sys.argv[1])
urllib.request.urlopen(urllib.request.Request(sys.argv[3] + "/v1/chat/completions", json.dumps(p).encode(),
                       {"Content-Type": "application/json"}), timeout=600).read()
PY
else
    curl -s --max-time 300 $API/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"Write a long story about a lighthouse keeper. '"$(date +%s%N)"'"}],"max_tokens":'"$N"',"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
fi
echo "PROF: $ID request took $(awk -v a="$s" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}') s"
curl -s -X POST $API/stop_profile >/dev/null
for _ in $(seq 60); do ls "$R/prof/$ID"-*.trace.json.gz >/dev/null 2>&1 && break; sleep 1; done
ls -la "$R/prof/$ID"-* 2>/dev/null
