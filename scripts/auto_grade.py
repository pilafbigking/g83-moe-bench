#!/usr/bin/env python3
"""Auto-grade objective prompts in the 20-prompt internal suite (k6 vs k4 outputs).

Reads generated answers from <eval_dir>/k6/*.json and <eval_dir>/k4/*.json
(default: ~/eval; prompts defined in ./scripts/eval_set.json). Subjective
prompts (code style, open-ended translation tone) are left for manual review.
"""
import json, os, re, sys

BASE = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/eval")


def load(k, qid):
    p = os.path.join(BASE, "k%d" % k, qid + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["a"]


def has(a, *pats):
    return all(re.search(p, a, re.I) for p in pats)


CHECKS = [
    # (id, description, predicate)
    ("06_math", "1234 x 5678 = 7,006,652",
     lambda a: has(a, r"7[,\s]?006[,\s]?652")),  # truncation is marked FAIL
    ("09_light", "299,792,458 m/s and 299,792.458 km/s",
     lambda a: has(a, r"299[,\s]?792[,\s]?458")),
    ("10_salt", "NaCl + sodium + chlorine + ionic",
     lambda a: has(a, r"\bNaCl\b|\bNa\s*Cl\b") and has(a, r"sodium") and
               has(a, r"chlor") and has(a, r"ionic|ion-dipole|electrovalent")),
    ("14_handshake", "5-person handshakes = 10",
     lambda a: bool(re.search(r"\b10\b", a)) and has(a, r"n\s*\(\s*n\s*-\s*1\s*\)|C\s*\(\s*5\s*,\s*2\s*\)|5\s*[×x*]\s*4\s*/\s*2")),
    ("15_dice", "Sum of 7 probability = 1/6",
     lambda a: has(a, r"1\s*/\s*6|6\s*/\s*36|frac\{1\}\{6\}|frac\{6\}\{36\}")),
    ("17_translate2", "Preserves 1,200 and 1998",
     lambda a: has(a, r"1[.,]?200|1\.200") and has(a, r"1998")),
    ("20_gold", "Au and density 19.3",
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


def main():
    if not (os.path.isdir(os.path.join(BASE, "k6")) or os.path.isdir(os.path.join(BASE, "k4"))):
        print(f"No k6/ or k4/ output directories found under {BASE}.")
        print("Run the prompts in ./scripts/eval_set.json and write responses to <eval_dir>/k{4,6}/<id>.json first.")
        return

    rows = []
    for qid, desc, fn in CHECKS:
        r = [qid, desc]
        for k in (6, 4):
            a = load(k, qid)
            if a is None:
                r.append("MISSING")
            elif not a.strip():
                r.append("EMPTY")
            else:
                r.append("PASS" if fn(a) else "FAIL")
        rows.append(r)

    # JSON schema check
    r = ["19_json", "Strict JSON with exact keys and types"]
    for k in (6, 4):
        a = load(k, "19_json")
        r.append("MISSING" if a is None else ("PASS" if check_json(a) else "FAIL"))
    rows.append(r)

    # 07_translate trap: inspect only the line containing the Japanese mouse translation, not explanations
    r = ["07_translate", "Japanese translates 'second' as 'second' rather than 'slow'"]
    for k in (6, 4):
        a = load(k, "07_translate")
        if a is None:
            r.append("MISSING")
        else:
            lines = [ln for ln in a.splitlines() if "ネズミ" in ln]
            seg = lines[0] if lines else ""
            ok = bool(re.search(r"二番目|2番目|2匹目|第二|second", seg)) and not re.search(r"遅い|遅れた|slow", seg)
            r.append("PASS" if ok else "FAIL")
    rows.append(r)

    print("| Prompt | Criterion | K6 | K4 |")
    print("|---|---|:-:|:-:|")
    p6 = p4 = 0
    for qid, desc, a, b in rows:
        print("| %s | %s | %s | %s |" % (qid, desc, a, b))
        p6 += a == "PASS"
        p4 += b == "PASS"
    print()
    print("**Objective pass rate: K6 %d/%d, K4 %d/%d**" % (p6, len(rows), p4, len(rows)))

    print("\n--- Excerpts for FAIL items ---")
    for qid, desc, a, b in rows:
        for k, v in ((6, a), (4, b)):
            if v == "FAIL":
                txt = load(k, qid) or ""
                print("\n[%s · K%d] %s" % (qid, k, desc))
                print(txt[:500].replace("\n", " "))


if __name__ == "__main__":
    main()
