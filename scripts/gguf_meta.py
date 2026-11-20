#!/usr/bin/env python3
"""Parse GGUF headers and tensor tables to compute exact per-token byte traffic.

Only the GGUF headers (first ~8 MB of single/first shard, or full shards for sizes)
are needed; no tensor weights are read into memory.

Usage:
  python3 gguf_meta.py <file.gguf...> [--total-bytes N] [--top-k K]
"""
import argparse
import os
import re
import struct
from collections import defaultdict

SCALAR = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<B", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}

TYPENAME = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}

RE_EXPS = re.compile(r"^blk\.\d+\..*_exps\.(weight|bias)$")
RE_TID2EID = re.compile(r"^blk\.\d+\..*_tid2eid\.")


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


def parse_shards(paths, total_bytes_override=None):
    kv = {}
    all_spans = []
    total_bytes = 0
    ver = 0
    for path in sorted(set(paths)):
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
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
            fsize = (
                total_bytes_override
                if (total_bytes_override and len(paths) == 1)
                else os.path.getsize(path)
            )
            tensors.sort(key=lambda x: x[3])
            for i, (name, dims, gt, off) in enumerate(tensors):
                s0 = dstart + off
                e0 = (dstart + tensors[i + 1][3]) if i + 1 < len(tensors) else fsize
                all_spans.append((name, dims, gt, s0, max(0, e0 - s0)))
            total_bytes += fsize
            print(f"[shard] {os.path.basename(path):<42s} {fsize/2**30:6.2f} GiB  {nt:4d} tensors")
    return ver, kv, all_spans, total_bytes


def summarize(ver, kv, spans, total_bytes, top_k_override=None):
    print(f"Total: {total_bytes/2**30:.2f} GiB ({total_bytes/1e9:.2f} GB) · {len(spans)} tensors · GGUF v{ver}")

    arch = kv.get("general.architecture", "?")
    print(f"\n--- Key Metadata (arch={arch}) ---")
    keys = [
        "block_count",
        "expert_count",
        "expert_used_count",
        "expert_shared_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "context_length",
        "vocab_size",
    ]
    for k in keys:
        for c in sorted(x for x in kv if x.endswith("." + k)):
            v = kv[c]
            if isinstance(v, list):
                v = f"[{len(v)} values]"
            print(f"  {c:<44s} {v}")

    bytype = defaultdict(int)
    for _, _, gt, _, sz in spans:
        bytype[TYPENAME.get(gt, f"type_{gt}")] += sz
    print("\n--- By Quantization Type ---")
    for t, b in sorted(bytype.items(), key=lambda x: -x[1]):
        print(f"  {str(t):<10s} {b/1e9:7.2f} GB  ({100*b/max(1, total_bytes):5.1f}%)")

    l0 = [(n, d, g, sz) for n, d, g, _, sz in spans if n.startswith("blk.0.")]
    if l0:
        print("\n--- blk.0 Tensors ---")
        for n, d, g, sz in sorted(l0, key=lambda x: -x[3]):
            print(f"  {n:<34s} {str(d):<22s} {TYPENAME.get(g, f'type_{g}'):<8s} {sz/2**20:7.2f} MiB")

    has_output_weight = any(n == "output.weight" for n, _, _, _, _ in spans)
    blk_dense = 0
    blk_exps = 0
    output_head = 0
    lookup_tables = 0

    for n, _, _, _, sz in spans:
        if n.startswith("token_embd."):
            if has_output_weight:
                lookup_tables += sz
            else:
                output_head += sz  # tied embeddings read full matrix at output projection
        elif n == "output.weight" or n == "output.bias" or n.startswith("output_norm.") or n.startswith("output_hc_"):
            output_head += sz
        elif RE_EXPS.match(n):
            blk_exps += sz
        elif RE_TID2EID.match(n):
            lookup_tables += sz
        elif n.startswith("blk."):
            blk_dense += sz
        else:
            output_head += sz

    n_ex = kv.get(f"{arch}.expert_count", 0)
    k_used = top_k_override if top_k_override is not None else kv.get(f"{arch}.expert_used_count", 0)
    routed = (blk_exps * k_used / n_ex) if (n_ex and k_used) else 0.0
    dense_total = blk_dense + output_head
    per_tok = dense_total + routed

    print("\n--- Exact Per-Token Traffic (blocks + output head, excl. 1-row lookups) ---")
    print(f"  Per-token block dense (attn/router/shared/indexer): {blk_dense/1e9:6.3f} GB  ({blk_dense/2**20:7.1f} MiB)")
    print(f"  Output head (output.weight + norm/hc)             : {output_head/1e9:6.3f} GB  ({output_head/2**20:7.1f} MiB)")
    print(f"  Total dense + output head                         : {dense_total/1e9:6.3f} GB  ({100*dense_total/max(1, per_tok):5.1f}%)")
    if n_ex:
        print(f"  All routed experts ({n_ex} experts)                 : {blk_exps/1e9:6.3f} GB  ({blk_exps/2**20:7.1f} MiB)")
        print(f"  Routed portion (top-{k_used} / {n_ex})                   : {routed/1e9:6.3f} GB  ({100*routed/max(1, per_tok):5.1f}%)")
    print(f"  TOTAL moved per token                             : {per_tok/1e9:6.3f} GB  ({per_tok/2**20:7.1f} MiB)")
    print(f"  Excluded 1-row lookup tables (token_embd/tid2eid) : {lookup_tables/1e9:6.3f} GB  ({lookup_tables/2**20:7.1f} MiB)")


def main():
    ap = argparse.ArgumentParser(description="Compute exact per-token bytes moved from GGUF tensor tables.")
    ap.add_argument("files", nargs="+", help="One or more GGUF shard paths")
    ap.add_argument("--total-bytes", type=int, default=None, help="Override single-file byte size")
    ap.add_argument("--top-k", type=int, default=None, help="Override expert_used_count for what-if top-K")
    args = ap.parse_args()
    ver, kv, spans, total_bytes = parse_shards(args.files, args.total_bytes)
    summarize(ver, kv, spans, total_bytes, args.top_k)


if __name__ == "__main__":
    main()
