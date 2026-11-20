# Raw `lm-eval` samples

Every accuracy number in the main table was recomputed from these files. Nothing here is
aggregated or hand-edited; these are the `--log_samples` JSONL that `lm-eval 0.4.13` wrote
during the runs on 2026-11-19 and 2026-11-20.

| File | Model | Task | docs | EM |
|---|---|---|--:|--:|
| `284b-k4-gsm8k-n25.jsonl` | 284B, Top-4, `-nr` | GSM8K, 5-shot | 25 | 0.800 |
| `284b-k6-gsm8k-n25.jsonl` | 284B, Top-6, `-nr` | GSM8K, 5-shot | 25 | 0.800 |
| `30b-q4-gsm8k-n25.jsonl` | Qwen3-30B-A3B `Q4_K_M`, repack | GSM8K, 5-shot | 25 | 0.880 |
| `8b-q4-gsm8k-n25.jsonl` | Llama-3.1-8B `Q4_K_M`, `-nr` | GSM8K, 5-shot | 25 | 0.600 |
| `284b-k6-triviaqa-5shot-n50.jsonl` | 284B, Top-6, `-nr` | TriviaQA, 5-shot | 50 | 0.520 |
| `30b-q4-triviaqa-5shot-n50.jsonl` | Qwen3-30B-A3B `Q4_K_M`, repack | TriviaQA, 5-shot | 50 | 0.600 |
| `8b-q4-triviaqa-5shot-n50.jsonl` | Llama-3.1-8B `Q4_K_M`, `-nr` | TriviaQA, 5-shot | 50 | 0.660 |

## Notes on the files

**The GSM8K files hold 50 rows for 25 questions.** `lm-eval` writes one row per
`(doc, filter)` pair, and GSM8K registers two filters, `strict-match` and
`flexible-extract`. So 25 questions produce 50 rows. Counting rows instead of distinct
`doc_id` doubles the apparent sample size and halves the standard error.

**The TriviaQA numbers here are the raw ones.** The hand-audited figures in the top-level
README (`64.0%` and `72.0%`) are a line-by-line review of these same 50 responses and are
not stored as data anywhere, because the audit was a person reading the strings. If you
disagree with a row in that audit, the raw response is in these files so you can inspect
the exact strings yourself.

## Recompute the table

```bash
python3 - <<'EOF'
import json, glob, os, math
for p in sorted(glob.glob('eval/*.jsonl')):
    rows = [json.loads(l) for l in open(p)]
    docs = {}
    for r in rows:
        docs.setdefault(r['doc_id'], []).append(r.get('exact_match'))
    per = [max(v) for v in docs.values()]
    n = len(per); acc = sum(per) / n
    se = math.sqrt(acc * (1 - acc) / (n - 1))
    print(f"{os.path.basename(p):42s} n={n:3d}  EM={acc:.3f} +/- {se:.3f}")
EOF
```

No dependencies beyond the standard library. Each line is one JSON object with `doc`,
`target`, `resps`, `filtered_resps`, `filter` and `exact_match`.

## What is not here

- **ARC / HellaSwag / MMLU / GPQA**: `llama-server-modchip` b12416 ignores `echo=true` and
  returns no prompt `logprobs`, so the four-choice loglikelihood tasks cannot be scored at
  all. An early ARC run produced `0.25` from four `-inf` options; it isn't shipped here
  because a 0.25 row looks like a real score at a glance.
- **The 20-prompt internal suite**: budget-sensitive (a 220-token cap truncated the verbose
  model and passed the concise one), so it is not comparable across models. The grader is in
  `../scripts/auto_grade.py` if you want to run it on a single model.
- **The 0-shot TriviaQA run**: `lm-eval` defaults TriviaQA to 0-shot, and that run is the
  `0.28` that was briefly and wrongly compared against a 5-shot `0.60`. Discarded.
