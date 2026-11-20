#!/usr/bin/env python3
"""
mincore(2) 张量级/专家级 Page Cache 驻留探测（G83 / DeepSeek-V4-Flash-MaxQ-v4）
- 按张量类别统计
- 按层区间统计（前/中/后）
- 按文件顺序 40 个等分块统计（区分"尾部 recency" vs "热集驻留"）★关键
- 43 x 256 专家槽位直方图 + CSV（画热力图）
用法: python3 inspect_residency.py <gguf分片...>
"""
import ctypes, os, re, statistics, struct, sys
from collections import defaultdict

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PROT_READ, MAP_SHARED = 1, 1
MAP_FAILED = ctypes.c_void_p(-1).value
PAGE_SIZE = os.sysconf("SC_PAGESIZE")

libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.mincore.restype = ctypes.c_int

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
INT_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def skip_val(f, vt):
    if vt in SCALAR_SIZES:
        b = f.read(SCALAR_SIZES[vt])
        if vt in INT_TYPES:
            return struct.unpack(INT_TYPES[vt], b)[0]
        return None
    if vt == 8:
        return read_str(f)
    if vt == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        for _ in range(cnt):
            skip_val(f, et)
        return None
    raise ValueError(f"Unknown GGUF vt: {vt}")


def parse_gguf(path):
    fsize = os.path.getsize(path)
    alignment = 32
    tensors = []
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", f"{path} not GGUF"
        ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            k = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            v = skip_val(f, vt)
            if k == "general.alignment" and isinstance(v, int):
                alignment = v
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            gtype, offset = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, gtype, offset))
        pos = f.tell()
        data_start = ((pos + alignment - 1) // alignment) * alignment

    tensors.sort(key=lambda x: x[3])
    spans = []
    for i, (name, dims, gtype, off) in enumerate(tensors):
        s = data_start + off
        e = (data_start + tensors[i + 1][3]) if i + 1 < len(tensors) else fsize
        spans.append((name, dims, s, e))
    return fsize, spans, len(tensors)


def classify(name):
    if "token_embd" in name or name.startswith("output"):
        return "0_embd_head"
    if "shexp" in name or "shared_expert" in name:
        return "2_shared_exp"
    if "_exps" in name:
        return "3_routed_exp"
    if "attn_" in name or "ffn_norm" in name or "attn_norm" in name or "ffn_gate_inp" in name:
        return "1_attn_router"
    return "4_other"


def mincore_bm(path, fsize):
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, fsize, PROT_READ, MAP_SHARED, fd, 0)
        if addr == MAP_FAILED:
            raise OSError(ctypes.get_errno(), "mmap failed")
        n = (fsize + PAGE_SIZE - 1) // PAGE_SIZE
        vec = (ctypes.c_ubyte * n)()
        if libc.mincore(addr, fsize, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        res = bytes(b & 1 for b in vec)
        libc.munmap(addr, fsize)
        return res
    finally:
        os.close(fd)


def main():
    paths = sorted(sys.argv[1:])
    if not paths:
        print("用法: python3 inspect_residency.py <gguf分片...>")
        sys.exit(1)

    cat = defaultdict(lambda: [0, 0])
    zone = defaultdict(lambda: [0, 0])
    slots = defaultdict(lambda: [0, 0])
    chunks_all = []
    tot_res = tot_pg = 0
    dims_reported = False
    nl_ex = 0

    for p in paths:
        fsize, spans, nt = parse_gguf(p)
        bm = mincore_bm(p, fsize)
        res, tot = sum(bm), len(bm)
        tot_res += res
        tot_pg += tot
        print(f"[分片] {os.path.basename(p)[-20:]}  {res*PAGE_SIZE/2**30:6.2f} / {fsize/2**30:6.2f} GiB  ({100*res/tot:5.1f}%)  {nt} 张量")

        # 40 等分块的驻留率 —— 区分尾偏 vs 热集
        nch = 40
        csize = tot // nch
        ck = [100.0 * sum(bm[i*csize:(i+1)*csize]) / csize for i in range(nch)] if csize else []
        chunks_all.append(ck)

        for name, dims, s, e in spans:
            p0, p1 = s // PAGE_SIZE, max(s // PAGE_SIZE + 1, (e + PAGE_SIZE - 1) // PAGE_SIZE)
            if p1 > len(bm):
                p1 = len(bm)
            if p1 <= p0:
                continue
            r, t = sum(bm[p0:p1]), p1 - p0
            c = classify(name)
            cat[c][0] += r
            cat[c][1] += t
            m = re.search(r"blk\.(\d+)\.", name)
            if not m:
                continue
            L = int(m.group(1))
            zb = "L00-13" if L <= 13 else ("L14-27" if L <= 27 else "L28+")
            zone[(zb, c)][0] += r
            zone[(zb, c)][1] += t
            if c == "3_routed_exp":
                ne = dims[-1] if (len(dims) >= 3 and dims[-1] > 1) else 256
                if not dims_reported:
                    print(f"  [样本] {name}  dims={dims}  -> 按 {ne} 个专家切分")
                    dims_reported = True
                step = (e - s) // ne
                for x in range(ne):
                    a = s + x * step
                    ap0 = a // PAGE_SIZE
                    ap1 = max(ap0 + 1, (a + step + PAGE_SIZE - 1) // PAGE_SIZE)
                    ap1 = min(ap1, len(bm))
                    slots[(L, x)][0] += sum(bm[ap0:ap1])
                    slots[(L, x)][1] += ap1 - ap0
                nl_ex = max(nl_ex, L + 1)

    print(f"\n=== 总体驻留 {tot_res*PAGE_SIZE/2**30:.2f} / {tot_pg*PAGE_SIZE/2**30:.2f} GiB ({100*tot_res/tot_pg:.1f}%) ===")

    print("\n--- 按类别 ---")
    for c in sorted(cat):
        r, t = cat[c]
        if t:
            print(f"  {c:<14s} {r*PAGE_SIZE/2**30:6.2f} / {t*PAGE_SIZE/2**30:6.2f} GiB  ({100*r/t:5.1f}%)")

    print("\n--- 层区间 x 类别 ---")
    for zb in ("L00-13", "L14-27", "L28+"):
        parts = []
        for c in ("0_embd_head", "1_attn_router", "2_shared_exp", "3_routed_exp"):
            r, t = zone[(zb, c)]
            if t:
                parts.append(f"{c.split('_',1)[1]}={100*r/t:5.1f}%")
        if parts:
            print(f"  {zb}:  " + "  ".join(parts))

    print("\n--- 文件顺序 40 等分块驻留率（★判定 recency vs 热集）---")
    for si, ck in enumerate(chunks_all):
        line = " ".join(f"{v:3.0f}" for v in ck)
        print(f"  片{si+1}: {line}")

    # 专家直方图
    bins = [(0.0, "0%   全在盘"), (0.001, "1-25% "), (0.25, "25-75%"), (0.75, "75-99%"), (1.0, "=100% 全常驻")]
    cnt = defaultdict(int)
    ratios = []
    for (L, x), (r, t) in slots.items():
        f = r / t if t else 0.0
        ratios.append(f)
        if f <= 0.0:
            cnt["0%   全在盘"] += 1
        elif f < 0.25:
            cnt["1-25% "] += 1
        elif f < 0.75:
            cnt["25-75%"] += 1
        elif f < 1.0:
            cnt["75-99%"] += 1
        else:
            cnt["=100% 全常驻"] += 1

    n = len(ratios)
    print(f"\n--- 专家槽位驻留直方图（{n} 个槽位，{nl_ex} 层）---")
    for _, label in bins:
        v = cnt[label]
        bar = "#" * int(50 * v / max(1, n))
        print(f"  {label:<12s} {v:6d} ({100*v/max(1,n):5.1f}%) {bar}")
    if ratios:
        mean = sum(ratios) / n
        print(f"  平均驻留率 {100*mean:.1f}%   完全常驻 {cnt['=100% 全常驻']}   完全在盘 {cnt['0%   全在盘']}")

    out = os.path.expanduser("~/expert_residency_map.csv")
    with open(out, "w") as f:
        f.write("layer,expert,resident_pages,total_pages,ratio\n")
        for (L, x) in sorted(slots):
            r, t = slots[(L, x)]
            f.write(f"{L},{x},{r},{t},{r/t:.4f}\n")
    print(f"\n逐槽位 CSV: {out}")

    # ---- 自动判决（V2.1 计划只写了甲/乙两分支，这里补第三支）----
    layers = sorted({L for (L, _) in slots})
    lmean = [statistics.mean(v for (L2, _), (r, t) in slots.items() if L2 == L and (v := r / t) >= 0)
             for L in layers]
    n = len(layers)
    mx, my = (n - 1) / 2, statistics.mean(lmean)
    num = sum((i - mx) * (lmean[i] - my) for i in range(n))
    den = (sum((i - mx) ** 2 for i in range(n)) * sum((v - my) ** 2 for v in lmean)) ** 0.5 or 1
    corr = num / den

    front = sum(sum(c[:len(c) // 4]) / max(1, len(c) // 4) for c in chunks_all) / max(1, len(chunks_all))
    back = sum(sum(c[-len(c) // 4:]) / max(1, len(c) // 4) for c in chunks_all) / max(1, len(chunks_all))
    bimodal = (cnt["=100% 全常驻"] + cnt["0%   全在盘"]) / max(1, n_slots if (n_slots := len(ratios)) else 1)

    print("\n=== 自动判决 ===")
    print(f"  层序 vs 驻留率 相关 r = {corr:+.3f}（接近 +1 才是文件尾偏）")
    print(f"  文件前 1/4 块 {front:.0f}%  vs 后 1/4 块 {back:.0f}%  (差 {back-front:+.0f} pt)")
    print(f"  双峰占比（0% 或 100%）{100*bimodal:.1f}%")
    if corr > 0.6 and (back - front) > 25:
        print("  → 情况甲：文件尾部 recency 主导。浅层与中层是同一次加载被挤出去的。")
    elif bimodal > 0.6:
        print("  → 情况乙：专家粒度淘汰（全热或全冷），僵尸专家假说成立。")
    else:
        print("  → 情况丙：**层结构主导 + 页粒度部分淘汰**。")
        print("     既不是纯 recency，也不是僵尸专家；冷热按层成段分布，层内按页零散残留。")


if __name__ == "__main__":
    main()
