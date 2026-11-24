#!/usr/bin/env bash
# Gate A: Measure physical hardware ceilings before loading models
#   1) Memory bandwidth scaling (1 -> 8 threads + large working set)
#   2) Sustained thermal / frequency stability (3-minute load)
#   3) NVMe seq & random 4K read speed (mmap major-fault cost)
set -u
OUT="${1:-$HOME/gate_a}"
SHARD="${2:-$(ls -S "$HOME"/dsv4/*00002-of-00003.gguf 2>/dev/null | head -1)}"
mkdir -p "$OUT"
SRC="$(cd "$(dirname "$0")" && pwd)"
ts() { date '+%H:%M:%S'; }

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "0. Environment snapshot"
{
  echo "Kernel    $(uname -r)"
  echo "CPU       $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ //')"
  echo "Threads   $(nproc) logical CPUs"
  echo "L3        $(lscpu | awk -F: '/L3 cache/{print $2}' | sed 's/^ //')"
  echo "MemTotal  $(free -h | awk '/Mem/{print $2}')"
  echo "Swap      $(free -h | awk '/Swap/{print $2}')"
  echo "Governor  $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "no_turbo  $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
  echo "THP       $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "lru_gen   $(cat /sys/kernel/mm/lru_gen/enabled 2>/dev/null || echo none)"
  echo "Disk free $(df -h / | awk 'NR==2{print $4}')"
} | tee "$OUT/env.txt"

say "1. Compile membw"
gcc -O3 -march=native -pthread -funroll-loops -o "$OUT/membw" "$SRC/membw.c" || { echo "Compile failed"; exit 1; }
echo "  ok  $OUT/membw"
grep -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' ' | tee "$OUT/avx512.txt"; echo

say "2. Memory bandwidth scaling (read, 2 GiB/thread)"
: > "$OUT/membw.txt"
for n in 1 2 4 8; do
  printf '  ' ; "$OUT/membw" read 2 "$n" 2>/dev/null | tee -a "$OUT/membw.txt"
done
echo "  -- Larger working set (bypass TLB/page-table cache effects) --"
printf '  ' ; "$OUT/membw" read 8 4 2>/dev/null | tee -a "$OUT/membw.txt"
printf '  ' ; "$OUT/membw" copy 2 4 2>/dev/null | tee -a "$OUT/membw.txt"

say "3. Sustained thermal / clock check (3 min)"
: > "$OUT/thermal.txt"
END=$((SECONDS + 180))
while [ $SECONDS -lt $END ]; do
  "$OUT/membw" read 1 8 >/dev/null 2>&1
done &
LOADPID=$!
while kill -0 $LOADPID 2>/dev/null; do
  T=$(for z in /sys/class/thermal/thermal_zone*; do [ -r "$z/temp" ] && cat "$z/temp"; done 2>/dev/null | sort -n | tail -1)
  F=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{s+=$1; n++} END{printf "%.0f", s/n/1000}')
  echo "$(ts)  ${F} MHz  $(( ${T:-0} / 1000 )) C" | tee -a "$OUT/thermal.txt"
  sleep 15
done
wait $LOADPID 2>/dev/null
echo "  Max temp  $(awk '{print $3}' "$OUT/thermal.txt" | sort -n | tail -1) C"
echo "  Min clock $(awk '{print $2}' "$OUT/thermal.txt" | sort -n | head -1) MHz"
echo "  Max clock $(awk '{print $2}' "$OUT/thermal.txt" | sort -n | tail -1) MHz"

say "4. NVMe read throughput (mmap major-fault cost)"
if ! command -v fio >/dev/null; then
  echo "  Installing fio..."
  sudo -n apt-get install -y -qq fio >/dev/null 2>&1 || echo "  Failed to install fio; skipping NVMe test"
fi
if command -v fio >/dev/null && [ -n "$SHARD" ] && [ -r "$SHARD" ]; then
  echo "  Target file: $(basename "$SHARD") ($(du -h "$SHARD" | cut -f1))"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  : > "$OUT/nvme.txt"
  run_fio() {
    local name=$1 rw=$2 bs=$3 qd=$4 jobs=$5
    local out bw iops
    out=$(fio --name="$name" --filename="$SHARD" --rw="$rw" --bs="$bs" \
              --iodepth="$qd" --numjobs="$jobs" --direct=1 --ioengine=libaio \
              --size=2G --runtime=25 --time_based --output-format=terse 2>/dev/null)
    # terse: field 7 = read bandwidth (KiB/s), field 8 = read IOPS
    bw=$(echo "$out" | awk -F';' '{print $7}')
    iops=$(echo "$out" | awk -F';' '{print $8}')
    printf "  %-26s %8.1f MB/s %9.0f IOPS\n" "$name" "$(awk -v b="$bw" 'BEGIN{print b/1024}')" "$iops" | tee -a "$OUT/nvme.txt"
  }
  run_fio "seq_read_1M_qd8"      read     1M  8  1
  run_fio "rand_read_4k_qd1"     randread 4k  1  1
  run_fio "rand_read_4k_qd32"    randread 4k  32 1
  run_fio "rand_read_4k_qd32x4"  randread 4k  32 4
else
  echo "  Skipped (missing fio or target shard; pass shard path as 2nd argument)"
fi

say "5. Summary"
echo "  Env     $OUT/env.txt"
echo "  Membw   $OUT/membw.txt"
echo "  Thermal $OUT/thermal.txt"
echo "  NVMe    $OUT/nvme.txt"
tail -6 "$OUT/membw.txt" 2>/dev/null | sed 's/^/  /'
