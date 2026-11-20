<div align="center">

<img src="./assets/hero_banner.png" alt="A 4-bit 30B beats a 1.75-bit 284B on the same 38 GB laptop" width="100%" />

<br/><br/>

[![Hardware](https://img.shields.io/badge/Hardware-dynabook_G83%2FHS_(931g%2C_%24150)-133C55?style=flat-square)](./README.md#2-the-hardware-wall--why-dual-channel-failed)
[![RAM](https://img.shields.io/badge/RAM-38_GiB_DDR4--3200_(0_Swap)-2E6F95?style=flat-square)](./README.md#2-the-hardware-wall--why-dual-channel-failed)
[![CPU](https://img.shields.io/badge/CPU-i5--1135G7_AVX--512_Intact-475569?style=flat-square)](./README.md#2-the-hardware-wall--why-dual-channel-failed)
[![Kernel](https://img.shields.io/badge/Linux-6.8_MGLRU_0x0007-133C55?style=flat-square)](./README.md#3-how-58-gib-runs-in-38-gib-without-thrashing-mincore2-dissection)
[![Finding](https://img.shields.io/badge/Verdict-4--bit_30B_Beats_1.75--bit_284B-C8553D?style=flat-square)](./README.md#1-arena-results-4-bit-30b-vs-175-bit-284b)
[![License: MIT](https://img.shields.io/badge/License-MIT-64748B?style=flat-square)](./LICENSE)

<p>
  <a href="#1-arena-results-4-bit-30b-vs-175-bit-284b">Arena Results</a> &middot;
  <a href="#2-the-hardware-wall--why-dual-channel-failed">Hardware &amp; Bandwidth</a> &middot;
  <a href="#3-how-58-gib-runs-in-38-gib-without-thrashing-mincore2-dissection"><code>mincore(2)</code> Dissection</a> &middot;
  <a href="#4-where-this-150-laptop-still-wins-orc-and-ahai"><code>OR@C</code> &amp; <code>AHAI</code></a> &middot;
  <a href="#5-eleven-ways-our-benchmarks-lied-to-us">11 Benchmark Traps</a> &middot;
  <a href="#6-reproduce-and-limitations">Reproduce</a>
</p>

</div>

---

A used **\$150** ultraportable laptop (`dynabook G83/HS`, **931 g** weighed, `38 GiB` usable RAM, **0 B swap**) can run a **57.99 GiB**, **284.33B-parameter MoE** (`DeepSeek-V4-Flash-0731 MaxQ-v4`, 1.75 bpw, all 256 experts intact) at **2.10 tok/s** (Top-4) / **1.65 tok/s** (Top-6) with zero OOMs and zero pruning.

Then we ran an unpruned **17.28 GiB `Qwen3-30B-A3B-Instruct-2507 @ Q4_K_M`** on the **exact same machine, binary, and flags**:

> [!IMPORTANT]
> **Key Takeaway**: Sub-2-bit quantization on a giant MoE delivers **zero measurable accuracy upside** over a native 4-bit 30B MoE on the same laptop—while costing **4.3× slower generation**, **22× worse time-to-first-token**, and **6.0× higher CPU package energy per token**. You cannot quantize the high-precision dense skeleton (`4.06 GB/token` across 43 layers alone).

| Metric | 284B @ 1.75 bpw (`MaxQ-v4`) | 30B-A3B @ `Q4_K_M` | Delta |
|---|--:|--:|--:|
| Model footprint / usable RAM | 57.99 / 37 GiB (**1.57× oversubscribed**) | 17.28 / 37 GiB (fully resident) | — |
| Exact bytes moved per token (`gguf_meta.py`) | **4.95 GB** (K4) / 5.40 GB (K6) | **1.66 GB** (K8) | **3.0×** |
| Generation speed (short prompt) | 2.10 tok/s (K4) / 1.65 tok/s (K6) | **8.94 tok/s** | **4.3×** |
| Generation speed (~734 tok ctx) | ~1.7 tok/s | **6.94 tok/s** | **4.1×** |
| Prefill (~700 tok) | 7.82 tok/s (`90.1 s`) | **35.01 tok/s** (`21.0 s`) | **4.5×** |
| Time-to-first-token (20 tok prompt) | ~20 s | **0.9 s** | **22×** |
| Power (`RAPL package-0`, excl. DRAM/iGPU) | 27.7 W | **19.4 W** | — |
| Energy per token (`package-0` only) | 13.2 J/tok | **2.2 J/tok** | **6.0×** |
| GSM8K 5-shot (`n=25`) | 80.0% ± 8.2 (K4 &amp; K6) | **88.0% ± 6.6** (`strict 84.0%`) | +8 pp |
| TriviaQA 5-shot (`n=50`, raw EM / audited) | 52.0% / **64.0%** (`32/50`, K6) | **60.0%** / **72.0%** (`36/50`) | +8 pp |
| RAM residency (`mincore(2)`) | **61.3%** (`routed_experts 58.8%`) | **100%** | — |

---

## 1. Arena Results: 4-bit 30B vs 1.75-bit 284B

<div align="center">
  <img src="./assets/arena_dashboard.png" alt="4-bit 30B vs 1.75-bit 284B Dashboard" width="100%" />
</div>

### Why 1.75-bit 284B Loses on Speed and Latency

Estimating MoE traffic as `13.17B × 1.75 bpw = 2.89 GB/tok` is wrong because GGUF quantizers keep attention projections, routers, and shared experts in higher precision (`Q4_K`–`Q8_0` / `BF16`). Summing the real tensor sizes directly via [`./scripts/gguf_meta.py`](./scripts/gguf_meta.py):

```
bytes_per_token = sum(dense tensors per layer) + sum(all routed experts) * (top_k / n_expert)
```

- **Dense skeleton tax**: `DeepSeek-V4-Flash` moves **`3,874 MiB` (`3.78 GiB` / `4.06 GB`)** of dense tensors per token across 43 layers *before touching a single routed expert*—already **2.4× larger** than `Qwen3-30B-A3B`'s entire `1.66 GB/token` budget (`541 MiB` dense + `1,046 MiB` routed).
- **Why Top-K 6 → 4 boosts speed +26.8% (`1.65 → 2.10 tok/s`)**: Total bytes drop only `9%` (`5.40 → 4.95 GB`), so the remaining ~18% gain comes from cutting sparse `mul_mat_id` gather dispatches by 33%.
- **The 22× usability cliff**: A 20-token prompt stalls `~20 s` on `284B` (`18 s` fixed intercept faulting the batch's experts from NVMe + `0.10 s/tok` marginal prefill; `90.1 s` for 704 tokens) vs **`0.9 s`** on `30B`.

### Quality & Failure Modes (`n=25` / `n=50`)

- **GSM8K (5-shot, `n=25`)**: `284B @ 1.75 bpw` scores **`80.0% ± 8.2`** (`20/25`, identical at Top-4 and Top-6, and identical under `strict` and `flexible` extract) — matching `Llama-3-8B` (`79.6%`). Every single failure gets the intermediate reasoning right and collapses on the **final step** (dropping orders of magnitude like `70` for `70000`, or answering the wrong target quantity). `30B @ Q4_K_M` scores **`88.0% ± 6.6`** (`22/25`; `strict 84.0%`).
- **TriviaQA (5-shot, `rc.nocontext`, `n=50`)**: Raw `lm-eval` `exact_match` gives `52.0% ± 7.1` (`26/50`) vs `60.0% ± 7.0` (`30/50`). A line-by-line audit of all 50 responses shows **both models suffered an identical `-12 pp` (`-6` question) string-matching penalty**, lifting audited accuracy to **`64.0%` (`32/50`) vs `72.0%` (`36/50`)**.

<details>
<summary><strong>Deep Dive: Full 50-Sample TriviaQA Audit &amp; Top-K 6 vs 4 Regressions</strong></summary>

1. **284B (K6) false negatives (`26/50 → 32/50 = 64.0%`)**: `#04 ' breast cancer'` (gold `cancer`), `#08 ' Nikkei 225'` (gold `Nikkei`), `#21 ' Highway 61'` (gold `61`), `#37 ' “Stars on 45”'` (curly quotes broke EM), `#42 ' Manticore'` (missing article `A/The`), `#43 ' Cyclops'` (singular vs `Cyclopes`). *Excluded single-letter `' J'` truncation on `J.G. Ballard` (`#34`).*
2. **30B-A3B (Q4) false negatives (`30/50 → 36/50 = 72.0%`)**: `#00 ' Ross Bagdasarian Sr'` (real name of `David Seville`), `#08 ' Nikkei 225'`, `#15 ' George H'` (`George H.W. Bush` truncated at `.`; strictly excluding `#15` yields `35/50 = 70.0%`), `#21 ' Highway 61'`, `#33 ' 48 Hrs'` (official title *48 Hrs.* stripped at `.`), `#40 ' The deep sea'` (gold `Underwater`/`Undersea`). *Excluded `' H'` (`#02`) and `' J'` (`#34`).*
3. **Top-K 6 → 4 on 284B (20-prompt internal suite)**: Coding (LRU cache, SQL second-highest salary) remained identical, while objective checks dropped `9/9 → 7/9` (`1234×5678` lost carries even at 1200 tokens outputting `7000000?`, and *"the second mouse gets the cheese"* translated `"second"` into Japanese as `遅い` *slow mouse* instead of `二番目` *second*).

</details>

---

## 2. The Hardware Wall & Why Dual-Channel Failed

<div align="center">
  <img src="./assets/hardware_bandwidth_tuning.png" alt="Hardware Bandwidth Ceiling and Tuning Matrix" width="100%" />
</div>

Measured via [`./scripts/gate_a/run_gate_a.sh`](./scripts/gate_a/run_gate_a.sh) before loading any model:

- **Asymmetric rank lock**: Populating `32 GB DDR4-3200 2R` alongside the soldered `8 GB DDR4-3200 1R` did not trigger Intel Tiger Lake flex-mode interleaving. Across working sets from `0.25 GiB` to `32 GiB`, 4-thread read bandwidth stayed flat at **`18.87–21.50 GB/s`** (~84% of single-channel `25.6 GB/s` peak) with `110.5 ns` random pointer-chasing latency.
- **Un-fused AVX-512 & zero thermal throttling**: All `avx512` extensions (`vnni`, `vpopcntdq`, `vbmi2`, etc.) are active in BIOS; a 3-minute full load held **3.7–3.8 GHz at 66–77 °C** (lid open, AC power).
- **NVMe vs RAM gap (`392×`)**: Samsung `MZVLQ256HAJD` delivers `2,183.5 MB/s` seq 1M QD8, **`54.9 MB/s` (`14,063 IOPS`)** random 4K QD1, and `462.5 MB/s` (`118,396 IOPS`) random 4K QD32.
- **Why `--repack` and `-t 8` fail**: At `4` threads the single memory channel is saturated. Enabling SMT (`-t 8`) crashes speed **3.6×** (`2.11 → 0.59 tok/s`) due to barrier convoys, while `--repack` allocates `+2 GiB` of anonymous RAM—evicting `2 GiB` of experts from page cache and dropping K6 speed (`1.30 → 1.20 tok/s`).

<details>
<summary><strong>Hardware &amp; Engine Specifications (Complete Inventory)</strong></summary>

| Component | Exact Measured / Observed Spec |
|---|---|
| Machine | `dynabook G83/HS` (`931 g` measured on scale; ~$150 used) |
| CPU | Intel Core i5-1135G7 (4C/8T, Tiger Lake-U, `8 MiB` L3, `performance` governor + turbo) |
| Memory | `38 GiB` usable (`MemTotal`), `32 GB 2R` + `8 GB 1R` DDR4-3200, **0 B swap** |
| Storage | Samsung `MZVLQ256HAJD-00000` 256 GB NVMe (Linux partition `110 GiB`, `21.4 GiB` free after both models) |
| OS / Kernel | Ubuntu 24.04.5 LTS, kernel `6.8.0-139-generic`, MGLRU `/sys/kernel/mm/lru_gen/enabled = 0x0007`, THP `madvise` |
| Battery | Design capacity `53.13 Wh` (unit tested at `22%` health / `11.64 Wh`; runtime figures scaled to `53.13 Wh`) |
| Model A | `Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF` (`57.99 GiB`, 3 shards: `6 MB` metadata stub + `33.10 GiB` + `24.89 GiB`, 43 layers, 256 routed + 1 shared expert, `1.75 bpw`) |
| Model B | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` (`Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`, `17.28 GiB`, SHA256 `6c997b8af...`, 48 layers, 128 routed experts, Top-8) |
| Binary &amp; Flags | `llama-server-modchip-linux-x86_64-avx512` (b12416), `-c 4096 -np 1 -b 2048 -ub 1024 -fa on -ctk q8_0 -ctv q8_0 -nr -t 4 -tb 4 --jinja` |

</details>

---

## 3. How 58 GiB Runs in 38 GiB Without Thrashing: `mincore(2)` Dissection

With a 39% RAM deficit and 256 routed experts per layer, uniform activation over 200 tokens would touch $1 - (1 - 4/256)^{200} = 95.7\%$ of experts and collapse throughput to **0.05 tok/s**. Instead, steady-state generation added only **3,009–3,661 major page faults** over 48 tokens (on top of `11,416` load faults).

Running [`./scripts/inspect_residency.py`](./scripts/inspect_residency.py) (`mmap(PROT_READ, MAP_SHARED)` + `mincore(2)`, 3-second non-intrusive scan across all `43 × 256 = 11,008` expert slots after 20+ mixed prompts) reveals the exact mechanism:

<div align="center">
  <img src="./assets/expert_residency_heatmap.png" alt="43 x 256 Expert Residency Heatmap" width="100%" />
</div>

<br/>

<div align="center">
  <img src="./assets/mincore_analysis.png" alt="Tensor-Class Pinning and 11,008-Slot Residency Histogram" width="100%" />
</div>

> [!NOTE]
> **Three Diagnostic Verdicts from `mincore(2)`**:
> 1. **Not file-tail recency**: Layer-order correlation is `r = -0.153`, and splitting Shards 2 &amp; 3 into 40 sequential chunks shows **74% (front quarter) vs 73% (back quarter)**.
> 2. **Not whole-expert "zombie" eviction**: Only `46.0%` of the 11,008 slots sit at `0%` or `100%` (`<60%` bimodal threshold), and expert-ID correlation is `r = 0.053`.
> 3. **Page-granularity eviction + layer banding**: Each expert is `4.95 MiB` (~`1,240` pages), and Linux `fault_around` reads **`64 KiB` (16 pages = `1.3%`)** per fault. The **`32.8%` spike in the `1–25%` bin** (`3,612` slots) is the literal footprint of experts sampled briefly and reclaimed at page granularity. Because every token traverses all 43 layers equally, cold mid-band layers (`L32–34`, median residency `0.006–0.092` vs `1.000` at `L00–02` and `L38–42`) reflect differences in per-layer routing concentration (entropy) or the cyclic LRU sweep frontier.

<details>
<summary><strong>Full Layer-Band Breakdown &amp; Fault Counts</strong></summary>

| Layer Band | `attn + router` | `shared_expert` | `routed_experts` |
|---|--:|--:|--:|
| `L00–13` | 96.1% | 100.0% | 63.6% |
| `L14–27` | 95.4% | 100.0% | 53.4% |
| `L28–42` | 95.2% | 100.0% | 59.3% |

- **Cold CLI Load**: `82 s`, `65 GB` read, `37.9 GiB` RSS, `11,416` major faults (Warm server load: `53 s`).
- **Load + 48 tokens**: `14,425` major faults (Top-K 4 `-nr`, `+3,009`), `15,077` (Top-K 6 `-nr`, `+3,661`), `17,988` (Top-K 6 `--repack`, `+6,572`).

</details>

---

## 4. Where This \$150 Laptop Still Wins: `OR@C` and `AHAI`

Even though `1.75-bit 284B` loses to `4-bit 30B`, the `dynabook G83/HS` itself sits at an unusual hardware intersection: **a socketed 32 GB DDR4 slot + un-fused AVX-512 + active fan under 1 kg for \$150** (most sub-1 kg laptops solder 8–16 GB RAM and immediately OOM).

1. **Oversubscription Feasibility (`OR@C`)**: At `GSM8K ≥ 80%` and `≥ 1 tok/s`, it achieves **`OR@C = 57.99 / 37 = 1.57×`** with zero swap (`0.39 GiB/$`).
2. **Air-Gapped Heavy Autonomous Index (`AHAI`)**: Stripping out vanity parameter counts, consider a real **air-gapped, battery-powered field workload** (`α_offline = 1`) requiring a **`25 GiB` total working set** (`17.3 GiB` 30B model + `3.5 GiB` KV/compute buffers + `4.2 GiB` OS &amp; local toolchain) and reliability strictly above the `80%` 8B-smartphone ceiling (`ΔOdds = max(0, Acc/(1-Acc) - 4.0)`):

$$\text{AHAI} = \alpha_{\text{offline}} \times \beta_{\text{25G}} \times \frac{\max\left(0,\ \frac{\text{Acc}}{1 - \text{Acc}} - 4.0\right) \times \left(\frac{v_{\text{sustained}} \times T_{\text{battery}} \times 3600}{250\text{ tok/task}}\right)}{W_{\text{terminal (kg)}} \times C_{\text{total (USD)}}}$$

<div align="center">
  <img src="./assets/ahai_ranking.png" alt="AHAI Leaderboard Across Device Classes" width="100%" />
</div>

> [!NOTE]
> **Disclosure**: G83 speeds, `RAPL package-0` power, mass (**`931 g`** measured), and GSM8K (`88%` vs `80%`) are measured on-device; G83 runtime (`2.74 h`, `~87,000 tok`) is scaled to its `53.13 Wh` design battery (tested unit health is `22%`). All competitor values (`MacBook Air/Pro`, `24GB Android`, `Mini PC`, `Gaming Laptop`, `RTX 4090`) are order-of-magnitude estimates from published specs.

---

## 5. Eleven Ways Our Benchmarks Lied to Us

Every surprising anomaly during this project turned out to be a measurement bug until proven otherwise:

<details open>
<summary><strong>Debugging Log: All 11 Caught &amp; Fixed Measurement Bugs</strong></summary>

| # | Bug | False Number Produced | Fix |
|--:|---|---|---|
| 1 | Bandwidth loop `s += a[i]` float dependency chain | `11.5 GB/s` (49% of real) — measured adder latency | Switched to 8-way independent integer XOR (`23.5 GB/s`) |
| 2 | `-no-cnv` flag absent in this build | Exited in `0.02 s` | Used `-st` for single-turn CLI |
| 3 | Extrapolated swap cost from random 4K (`55 MB/s`) | Predicted `0.05 tok/s` vs actual `2.1` | Experts are contiguous; `fault_around` reads `64 KiB` |
| 4 | Regex matched explanation paragraph on translation | Flagged valid K6 output as `FAIL` | Checked only the line containing `ネズミ` |
| 5 | Regex missed LaTeX `\frac{1}{6}` | Flagged both K4 &amp; K6 as `FAIL` | Added LaTeX fraction regex |
| 6 | `ARC` 4-choice loglikelihood returned all `-inf` | Fake `0.25` accuracy | `llama-server` ignores `echo=true`; dropped logprob tasks |
| 7 | `pkill -f llama-server` matched our own SSH command | Repeated self-kill | Matched `/proc/$PID/cmdline[0]` basename |
| 8 | Estimated bytes/token via `13.17B × 1.75 bpw` | `2.89 GB/tok` (predicted 30B only `1.44×` faster) | Summed actual GGUF tensors (`4.95` vs `1.66 GB/tok`) |
| 9 | `kill` matched absolute path while server used relative path | **Three "Qwen3" runs hit the stale 284B server** | Verified `ss -tlnp` PID `/proc/$PID/cmdline` before every run |
| 10 | Fixed `220`-token cap on custom 20-prompt eval | Verbose Qwen3 truncated (`6/9`), concise 284B passed (`9/9`) | Excluded budget-sensitive custom suite from cross-model conclusions |
| 11 | Omitted `--num_fewshot` on TriviaQA (`lm-eval` defaults to 0-shot) | `0.28` (0-shot) compared against `0.60` (5-shot) | Re-ran explicitly with `--num_fewshot 5` (`0.52`) |

</details>

---

## 6. Reproduce and Limitations

All standalone scripts live in [`./scripts/`](./scripts/):

```bash
# 1. Hardware bounds (4-thread RAM sweep, 3-min thermal check, NVMe fio)
bash ./scripts/gate_a/run_gate_a.sh

# 2. Exact per-token traffic from GGUF tensor headers (only needs first 8 MB or full shards)
python3 ./scripts/gguf_meta.py /path/to/*.gguf

# 3. Non-intrusive mincore(2) 11,008-slot residency profiler + CSV export
python3 ./scripts/inspect_residency.py /path/to/DeepSeek-V4-Flash-*.gguf

# 4. Prefill & generation probe against running llama-server
python3 ./scripts/prefill_probe.py http://127.0.0.1:8080

# 5. Power sampling (RAPL package-0; psys domain unimplemented on this board)
sudo bash ./scripts/power_probe.sh
```

> [!WARNING]
> **Explicit Limitations**:
> - **No loglikelihood benchmarks**: `llama-server-modchip` (b12416) ignores `echo=true` and does not return prompt `logprobs`, making `MMLU` / `ARC` / `HellaSwag` / `GPQA` untestable here.
> - **Small sample sizes**: `GSM8K (n=25, ±8.2 pp)` and `TriviaQA (n=50, ±7.1 pp)` establish direction and order of magnitude, not leaderboard decimals.
> - **RAPL domain scope**: `package-0` (`27.7 W` / `19.4 W`) excludes DRAM and display power (`psys` reads `0.1 W`), so total wall/battery draw is higher.
> - **Single residency snapshot**: `mincore(2)` numbers reflect one post-eval warm snapshot without multi-run variance sampling.
