#!/usr/bin/env python3
"""
读 GGUF 元数据 + 张量清单，算出「每 token 实际要搬多少字节」。

不需要完整文件：GGUF 的 header / KV / tensor info 都在文件开头，前 8 MB 足够。
用法:
  python3 gguf_meta.py <file.gguf> [--total-bytes N]
"""
import os, re, struct, sys
from collections import defaultdict

SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
          4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
          10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

TYPENAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
            13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
            18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
            22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
            27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0"}


def rd_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def rd_val(f, vt):
    if vt in SCALAR:
        fmt, sz = SCALAR[vt]
        return struct.unpack(fmt, f.read(sz))[0]
    if vt == 8:
        return rd_str(f)
    if vt == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        return [rd_val(f, et) for _ in range(cnt)]
    raise ValueError(f"unknown gguf type {vt}")


def main():
    argv = sys.argv[1:]
    total = None
    if "--total-bytes" in argv:
        i = argv.index("--total-bytes")
        total = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = sorted({a for a in argv if not a.startswith("--")})

    kv = {}
    tensors = []
    total_bytes = 0
    all_spans = []
    for path in args:
      with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
        for _ in range(nkv):
            k = rd_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            kv[k] = rd_val(f, vt)
        tensors = []
        for _ in range(nt):
            name = rd_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            gt, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, gt, off))
        pos = f.tell()
        align = kv.get("general.alignment", 32)
        dstart = ((pos + align - 1) // align) * align
        fsize = total if (total and len(args) == 1) else os.path.getsize(path)
        tensors.sort(key=lambda x: x[3])
        for i, (name, dims, gt, off) in enumerate(tensors):
            s0 = dstart + off
            e0 = (dstart + tensors[i + 1][3]) if i + 1 < len(tensors) else fsize
            all_spans.append((name, dims, gt, s0, max(0, e0 - s0)))
        total_bytes += fsize
        print(f"[片] {os.path.basename(path)[-24:]}  {fsize/2**30:6.2f} GiB  {nt} 张量")

    print(f"合计 {total_bytes/2**30:.2f} GiB · {len(all_spans)} 张量 · GGUF v{ver}")
    spans = all_spans
    fsize = total_bytes

    arch = kv.get("general.architecture", "?")
    print(f"\n--- 关键元数据 (arch={arch}) ---")
    keys = ["block_count", "expert_count", "expert_used_count", "expert_shared_count",
            "embedding_length", "feed_forward_length", "attention.head_count",
            "attention.head_count_kv", "context_length", "vocab_size"]
    for k in keys:
        kk = f"{arch}.{k}"
        cands = [x for x in kv if x.endswith("." + k)]
        for c in cands:
            v = kv[c]
            if isinstance(v, list):
                v = f"[{len(v)} values]"
            print(f"  {c:<44s} {v}")

    bytype = defaultdict(int)
    for _, _, gt, _, sz in spans:
        bytype[TYPENAME.get(gt, gt)] += sz
    print("\n--- 按量化类型 ---")
    for t, b in sorted(bytype.items(), key=lambda x: -x[1]):
        print(f"  {str(t):<8s} {b/1e9:8.2f} GB  ({100*b/max(1,fsize):5.1f}%)")

    # 每层构成（取第 0 层）
    L0 = [(n, d, g, sz) for n, d, g, s, sz in spans if n.startswith("blk.0.")]
    if L0:
        print("\n--- blk.0 张量 ---")
        for n, d, g, sz in sorted(L0, key=lambda x: -x[3]):
            print(f"  {n:<34s} {str(d):<22s} {TYPENAME.get(g, g):<7s} {sz/2**20:8.2f} MiB")

    # 每 token 字节估算
    nb = kv.get(f"{arch}.block_count")
    ne = kv.get(f"{arch}.expert_used_count")
    if nb and ne:
        per_layer_dense = 0
        per_layer_exp_all = 0
        for n, d, g, s, sz in spans:
            m = re.match(r"blk\.\d+\.(.+)", n)
            if not m:
                continue
            base = m.group(1)
            if base.endswith("_exps.weight") or base.endswith("_exps.bias"):
                per_layer_exp_all += sz
            elif base.startswith("attn_") or base.startswith("ffn_") or base.startswith("hc_"):
                per_layer_dense += sz
        n_ex = kv.get(f"{arch}.expert_count", 256)
        routed = per_layer_exp_all * ne / max(1, n_ex)
        per_tok = routed + per_layer_dense
        print("\n--- 每 token 字节（实算）---")
        print(f"  每层稠密（attn/router/shared/hc） : {per_layer_dense/2**20:8.2f} MiB")
        print(f"  每层全部专家（{n_ex} 个）          : {per_layer_exp_all/2**20:8.2f} MiB")
        print(f"  top-{ne} 路由部分                 : {routed/2**20:8.2f} MiB")
        print(f"  合计 / token                    : {(per_tok)/2**20:8.2f} MiB = {per_tok/1e9:.2f} GB")
        print(f"  模型 token/秒 预测（0.48 × 21.5 / bytes）: {0.48*21.5/(per_tok/1e9):.2f} tok/s")


if __name__ == "__main__":
    main()
