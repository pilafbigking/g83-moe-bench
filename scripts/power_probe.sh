#!/usr/bin/env bash
# 测功耗：空闲基线 + 推理峰值（RAPL psys 域 = 整平台）
set -u
R=/sys/class/powercap/intel-rapl:1/energy_uj   # psys
P=/sys/class/powercap/intel-rapl:0/energy_uj   # package-0

read_e() { cat "$1" 2>/dev/null; }

# 处理计数器回绕
delta() {
  local a=$1 b=$2 max=$3
  if [ "$b" -ge "$a" ]; then echo $(( b - a )); else echo $(( max - a + b )); fi
}
MAX=$(cat /sys/class/powercap/intel-rapl:1/max_energy_range_uj)

sample_for() {
  local secs=$1 label=$2
  local p0 pm e0 em i
  p0=$(read_e $P); e0=$(read_e $R)
  sleep "$secs"
  pm=$(read_e $P); em=$(read_e $R)
  local dp de
  dp=$(delta "$p0" "$pm" "$(cat /sys/class/powercap/intel-rapl:0/max_energy_range_uj)")
  de=$(delta "$e0" "$em" "$MAX")
  printf "  %-12s package %5.1f W   psys %5.1f W\n" "$label" \
    "$(awk -v d=$dp -v s=$secs 'BEGIN{print d/1e6/s}')" \
    "$(awk -v d=$de -v s=$secs 'BEGIN{print d/1e6/s}')"
}

echo "=== 功耗测量（psys = 整平台，含 DRAM/iGPU/核）==="
sample_for 15 "空闲"

# 推理中采样：发一个长请求，期间测 60 秒
( curl -s --max-time 400 http://127.0.0.1:8080/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{"messages":[{"role":"user","content":"Write a detailed 400 word explanation of how virtual memory paging works in Linux, covering page tables, TLB, major and minor faults, and the page replacement policy."}],"max_tokens":400,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":false}}' \
   > /tmp/power_req.json ) &
CURLPID=$!
sleep 12   # 跳过 prefill 阶段
sample_for 60 "推理中"

wait $CURLPID 2>/dev/null
python3 - <<'PY'
import json
try:
    d = json.load(open("/tmp/power_req.json"))
    t = d.get("timings", {})
    print("  该请求: gen %.2f t/s, %d tokens" % (t.get("predicted_per_second", 0), t.get("predicted_n", 0)))
except Exception as e:
    print("  请求结果读取失败:", e)
PY
