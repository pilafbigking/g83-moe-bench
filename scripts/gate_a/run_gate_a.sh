#!/usr/bin/env bash
# Gate A：在模型到位之前，把硬件的物理上限量出来
#   1) 内存带宽（单线程 → 8 线程的扩展曲线）
#   2) 持续负载下的热降频
#   3) NVMe 顺序/随机读（决定 mmap 缺页的代价）
set -u
OUT="${1:-$HOME/gate_a}"
mkdir -p "$OUT"
SRC="$(cd "$(dirname "$0")" && pwd)"
ts() { date '+%H:%M:%S'; }

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "0. 环境快照"
{
  echo "内核      $(uname -r)"
  echo "CPU       $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ //')"
  echo "核/线程   $(nproc) 逻辑核"
  echo "L3        $(lscpu | awk -F: '/L3 cache/{print $2}' | sed 's/^ //')"
  echo "内存      $(free -h | awk '/Mem/{print $2}')"
  echo "swap      $(free -h | awk '/Swap/{print $2}')"
  echo "调频      $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "turbo     $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
  echo "THP       $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "lru_gen   $(cat /sys/kernel/mm/lru_gen/enabled 2>/dev/null || echo 无)"
  echo "磁盘可用  $(df -h / | awk 'NR==2{print $4}')"
} | tee "$OUT/env.txt"

say "1. 编译 membw"
gcc -O2 -march=native -pthread -o "$OUT/membw" "$SRC/membw.c" || { echo "编译失败"; exit 1; }
echo "  ok  $OUT/membw"
grep -o 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' ' | tee "$OUT/avx512.txt"; echo

say "2. 内存带宽扩展曲线（读，2 GiB/线程）"
: > "$OUT/membw.txt"
for n in 1 2 4 8; do
  printf '  ' ; "$OUT/membw" read 2 "$n" 2>/dev/null | tee -a "$OUT/membw.txt"
done
echo "  -- 更大工作集（排除 TLB/页表缓存影响）--"
printf '  ' ; "$OUT/membw" read 8 4 2>/dev/null | tee -a "$OUT/membw.txt"
printf '  ' ; "$OUT/membw" copy 2 4 2>/dev/null | tee -a "$OUT/membw.txt"

say "3. 持续负载热降频（3 分钟，观察频率是否掉）"
: > "$OUT/thermal.txt"
END=$((SECONDS + 180))
while [ $SECONDS -lt $END ]; do
  "$OUT/membw" read 1 8 >/dev/null 2>&1
done &
LOADPID=$!
while kill -0 $LOADPID 2>/dev/null; do
  T=$(for z in /sys/class/thermal/thermal_zone*; do [ -r "$z/temp" ] && cat "$z/temp"; done 2>/dev/null | sort -n | tail -1)
  F=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{s+=$1; n++} END{printf "%.0f", s/n/1000}')
  echo "$(ts)  ${F} MHz  $(( ${T:-0} / 1000 ))°C" | tee -a "$OUT/thermal.txt"
  sleep 15
done
wait $LOADPID 2>/dev/null
echo "  最高温度 $(awk '{print $3}' "$OUT/thermal.txt" | sed 's/°C//' | sort -n | tail -1)°C"
echo "  最低频率 $(awk '{print $2}' "$OUT/thermal.txt" | sort -n | head -1) MHz"
echo "  最高频率 $(awk '{print $2}' "$OUT/thermal.txt" | sort -n | tail -1) MHz"

say "4. NVMe 读性能（mmap 缺页的代价）"
SHARD=$(ls -S "$HOME"/dsv4/*00002-of-00003.gguf 2>/dev/null | head -1)
if ! command -v fio >/dev/null; then
  echo "  安装 fio..."
  sudo -n apt-get install -y -qq fio >/dev/null 2>&1 || echo "  安装失败，跳过 NVMe 测试"
fi
if command -v fio >/dev/null && [ -n "$SHARD" ]; then
  echo "  目标文件: $(basename "$SHARD") ($(du -h "$SHARD" | cut -f1))"
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  : > "$OUT/nvme.txt"
  run_fio() {
    local name=$1 rw=$2 bs=$3 qd=$4 jobs=$5
    local out
    out=$(fio --name="$name" --filename="$SHARD" --rw="$rw" --bs="$bs" \
              --iodepth="$qd" --numjobs="$jobs" --direct=1 --ioengine=libaio \
              --size=2G --runtime=25 --time_based --output-format=terse 2>/dev/null)
    # terse: 第 7 列是带宽 KiB/s，第 8 列是 IOPS
    local bw iops
    bw=$(echo "$out" | awk -F';' '{print $7}')
    iops=$(echo "$out" | awk -F';' '{print $8}')
    printf "  %-26s %8.1f MB/s %9.0f IOPS\n" "$name" "$(awk -v b="$bw" 'BEGIN{print b/1024}')" "$iops" | tee -a "$OUT/nvme.txt"
  }
  run_fio "顺序读 1M qd8"      read     1M  8  1
  run_fio "随机读 4K qd1"      randread 4k  1  1
  run_fio "随机读 4K qd32"     randread 4k  32 1
  run_fio "随机读 4K qd32×4"   randread 4k  32 4
else
  echo "  跳过（缺 fio 或没有目标文件）"
fi

say "5. 汇总"
echo "  环境    $OUT/env.txt"
echo "  带宽    $OUT/membw.txt"
echo "  热      $OUT/thermal.txt"
echo "  NVMe    $OUT/nvme.txt"
tail -6 "$OUT/membw.txt" 2>/dev/null | sed 's/^/  /'
