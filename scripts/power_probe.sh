#!/usr/bin/env bash
# Sample RAPL idle baseline and generation power draw.
# Note: On the dynabook G83/HS, intel-rapl:1 (psys) is unimplemented in firmware (~0.1 W),
# so only package-0 (intel-rapl:0) is valid.
set -u
R=/sys/class/powercap/intel-rapl:1/energy_uj   # psys (optional)
P=/sys/class/powercap/intel-rapl:0/energy_uj   # package-0

if [ ! -r "$P" ]; then
  echo "Cannot read $P (run with sudo or check intel-rapl permissions)." >&2
  exit 1
fi

read_e() { cat "$1" 2>/dev/null; }

# Handle 32/64-bit RAPL counter wraparound
delta() {
  local a=$1 b=$2 max=$3
  if [ "$b" -ge "$a" ]; then echo $(( b - a )); else echo $(( max - a + b )); fi
}

HAS_PSYS=0
MAX_PSYS=0
if [ -r "$R" ] && [ -r /sys/class/powercap/intel-rapl:1/max_energy_range_uj ]; then
  HAS_PSYS=1
  MAX_PSYS=$(cat /sys/class/powercap/intel-rapl:1/max_energy_range_uj)
fi
MAX_PKG=$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)

sample_for() {
  local secs=$1 label=$2
  local p0 pm dp
  p0=$(read_e "$P")
  local e0=0 em=0 de=0
  if [ "$HAS_PSYS" -eq 1 ]; then e0=$(read_e "$R"); fi
  sleep "$secs"
  pm=$(read_e "$P")
  dp=$(delta "$p0" "$pm" "$MAX_PKG")
  if [ "$HAS_PSYS" -eq 1 ]; then
    em=$(read_e "$R")
    de=$(delta "$e0" "$em" "$MAX_PSYS")
    printf "  %-12s package-0 %5.1f W   psys %5.1f W\n" "$label" \
      "$(awk -v d="$dp" -v s="$secs" 'BEGIN{print d/1e6/s}')" \
      "$(awk -v d="$de" -v s="$secs" 'BEGIN{print d/1e6/s}')"
  else
    printf "  %-12s package-0 %5.1f W   psys N/A\n" "$label" \
      "$(awk -v d="$dp" -v s="$secs" 'BEGIN{print d/1e6/s}')"
  fi
}

echo "=== Power sampling (RAPL package-0; psys ~0.1 W = firmware unimplemented) ==="
sample_for 15 "idle"

# Sample during sustained decode: send a long request, measure 60 s mid-generation
( curl -s --max-time 400 http://127.0.0.1:8080/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{"messages":[{"role":"user","content":"Write a detailed 400 word explanation of how virtual memory paging works in Linux, covering page tables, TLB, major and minor faults, and the page replacement policy."}],"max_tokens":400,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":false}}' \
   > /tmp/power_req.json ) &
CURLPID=$!
sleep 12   # skip prefill phase
sample_for 60 "generating"

wait $CURLPID 2>/dev/null
python3 - <<'PY'
import json
try:
    d = json.load(open("/tmp/power_req.json"))
    t = d.get("timings", {})
    print("  Request: gen %.2f tok/s, %d tokens" % (t.get("predicted_per_second", 0), t.get("predicted_n", 0)))
except Exception as e:
    print("  Failed to parse response JSON:", e)
PY
