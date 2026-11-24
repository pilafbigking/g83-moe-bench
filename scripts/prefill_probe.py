import json, subprocess, sys

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
long_p = ("Explain virtual memory, page cache, MGLRU, and MoE expert locality in detail. " * 45)[:3000]

for name, p, mx in [("short_~20tok", "Explain what virtual memory paging is in Linux.", 350),
                    ("long_~700tok", long_p, 350)]:
    payload = {"messages": [{"role": "user", "content": p}],
               "max_tokens": mx, "temperature": 0.0, "seed": 1234,
               "chat_template_kwargs": {"enable_thinking": False}}
    r = subprocess.run(["curl", "-s", "--max-time", "900", base + "/v1/chat/completions",
                        "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        print(name, "JSON parse failed:", r.stdout[:200])
        continue
    tm = d.get("timings", {})
    pn = tm.get("prompt_n", 0)
    pp = tm.get("prompt_per_second", 0)
    pms = tm.get("prompt_ms", 0) / 1000
    gn = tm.get("predicted_n", 0)
    gp = tm.get("predicted_per_second", 0)
    print("%-14s prompt %4d tok @ %6.2f t/s (%6.1f s) | gen %4d tok @ %5.2f t/s"
          % (name, pn, pp, pms, gn, gp))

    txt = (d.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
    if txt:
        print("   >>", txt[:180].replace("\n", " "))
