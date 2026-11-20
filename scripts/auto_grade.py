#!/usr/bin/env python3
"""对客观题自动判分。只判能客观验证的项，主观题（代码写法、翻译风格）留人工。"""
import json, os, re, sys

BASE = os.path.expanduser("~/eval")


def load(k, qid):
    p = os.path.join(BASE, "k%d" % k, qid + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["a"]


def has(a, *pats):
    return all(re.search(p, a, re.I) for p in pats)


CHECKS = [
    # (id, 描述, 判定函数)
    ("06_math", "1234×5678 = 7,006,652",
     lambda a: has(a, r"7[,\s]?006[,\s]?652")),  # 截断会被标为「未完」
    ("09_light", "299,792,458 m/s 且 299,792.458 km/s",
     lambda a: has(a, r"299[,\s]?792[,\s]?458")),
    ("10_salt", "NaCl + sodium + chlorine + ionic",
     lambda a: has(a, r"\bNaCl\b|\bNa\s*Cl\b") and has(a, r"sodium") and
               has(a, r"chlor") and has(a, r"ionic|ion-dipole|electrovalent")),
    ("14_handshake", "5 人握手 = 10",
     lambda a: bool(re.search(r"\b10\b", a)) and has(a, r"n\s*\(\s*n\s*-\s*1\s*\)|C\s*\(\s*5\s*,\s*2\s*\)|5\s*[×x*]\s*4\s*/\s*2")),
    ("15_dice", "和为 7 的概率 = 1/6",
     lambda a: has(a, r"1\s*/\s*6|6\s*/\s*36|frac\{1\}\{6\}|frac\{6\}\{36\}")),
    ("17_translate2", "数字与年份保持 1,200 与 1998",
     lambda a: has(a, r"1[.,]?200|1\.200") and has(a, r"1998")),
    ("20_gold", "Au 且密度 19.3",
     lambda a: has(a, r"\bAu\b|gold") and has(a, r"19\.3")),
]


def check_json(a):
    s = a.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        return False
    if not isinstance(d, dict) or set(d.keys()) != {"name", "version", "tags"}:
        return False
    return (isinstance(d["name"], str) and isinstance(d["version"], int)
            and isinstance(d["tags"], list) and len(d["tags"]) == 2
            and all(isinstance(t, str) for t in d["tags"]))


rows = []
for qid, desc, fn in CHECKS:
    r = [qid, desc]
    for k in (6, 4):
        a = load(k, qid)
        if a is None:
            r.append("缺")
        elif not a.strip():
            r.append("空")
        else:
            r.append("PASS" if fn(a) else "FAIL")
    rows.append(r)

# JSON 单独判
r = ["19_json", "严格 JSON，键与类型完全匹配"]
for k in (6, 4):
    a = load(k, "19_json")
    r.append("缺" if a is None else ("PASS" if check_json(a) else "FAIL"))
rows.append(r)

# 07 第二个陷阱词：只检查日语译句本身，不看解释段落
r = ["07_translate", "日语译句把 second 译成「第二」而非「慢」"]
for k in (6, 4):
    a = load(k, "07_translate")
    if a is None:
        r.append("缺")
    else:
        # 抓包含「ネズミ」的那一行（译句），只看它
        lines = [ln for ln in a.splitlines() if "ネズミ" in ln]
        seg = lines[0] if lines else ""
        ok = bool(re.search(r"二番目|2番目|2匹目|第二|second", seg)) and not re.search(r"遅い|遅れた|slow", seg)
        r.append("PASS" if ok else "FAIL")
rows.append(r)

print("| 题 | 判据 | K6 | K4 |")
print("|---|---|:-:|:-:|")
p6 = p4 = 0
for qid, desc, a, b in rows:
    print("| %s | %s | %s | %s |" % (qid, desc, a, b))
    p6 += a == "PASS"
    p4 += b == "PASS"
print()
print("**客观题通过率：K6 %d/%d，K4 %d/%d**" % (p6, len(rows), p4, len(rows)))

print("\n--- 失败项原文摘录 ---")
for qid, desc, a, b in rows:
    for k, v in ((6, a), (4, b)):
        if v == "FAIL":
            txt = load(k, qid) or ""
            print("\n[%s · K%d] %s" % (qid, k, desc))
            print(txt[:500].replace("\n", " "))
