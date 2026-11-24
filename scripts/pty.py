#!/usr/bin/env python3
"""
PTY — Portable Token Yield
==========================

Replaces the previous AHAI index, which had four hand-picked constants
(4.0 odds baseline, 250 tok/task, 25 GiB working set) and two binary gates
that handed non-participants a score of 0.

PTY has no performance constants. It is a ratio of two measured/published quantities:

    PTY(T) = (v * 3600 * T) / [ C_used * (m + deficit/e) ]

  v       sustained generation, tok/s     (measured or BW-grounded est., 30B-A3B class)
  T       mission duration, hours         (scenario, reported at 8 h + sweep 2-24 h)
  C_used  used street price, USD          (Nov 2026 secondary market; new MSRP in C_new ref)
  m       device mass, kg                 (weighed / spec)
  P       inference power, W              (RAPL package-0 for G83; whole-kit est. for peers)
  e       Li-ion specific energy, kWh/kg  (physical constant, 0.25)

Unit: tok / ($ * kg) — tokens produced per dollar of hardware per kilogram
carried. Capability does not multiply the score; it defines the class a device
competes in (GSM8K >= 80% class).

All non-G83 rows use Nov 2026 US secondary-market estimates (eBay / Swappa /
RefurbMe ranges; new MSRP listed separately as C_new for reference) and
order-of-magnitude speeds for a Qwen3-30B-A3B Q4-class workload (1.92 GB/tok),
not measurements performed on those machines. The dynabook G83/HS 40GB price
($220) sums ~$100 for the discontinued 2021 corporate 8 GiB base unit (original
MSRP ~$1,500) plus ~$120 for a Nov 2026 used 32 GB DDR4-3200 SO-DIMM.
"""
import argparse
import json

E_DEFAULT = 0.25  # kWh/kg, Li-ion pack level (cell ~0.3, pack ~0.22-0.25)
G83_30B_NAME = "dynabook G83/HS 40GB (2021, 30B)"
G83_284B_NAME = "dynabook G83/HS 40GB (2021, 284B)"


def pty(v, C, m, P, T, e=E_DEFAULT, E_int=0.0):
    """E_int = internal battery, kWh. Spare cells are carried only for the
    deficit between what the mission requires and what the device holds."""
    deficit = max(0.0, P * T - E_int * 1000) / 1000.0  # kWh
    carried = m + deficit / e                          # kg
    return (v * 3600 * T) / (C * carried), carried


# All rows priced used (C, Nov 2026 secondary market); C_new holds retail MSRP
# for reference where applicable (None for discontinued 2021 corporate dynabook).
DEVICES = [
    dict(name=G83_30B_NAME,
         C=220,  C_new=None, m=0.931, P=19.4,  v=8.94, gsm8k=0.880, measured=True,  E_int=0.05313),
    dict(name="MacBook Air 13 M4 24GB (est.)",
         C=950,  C_new=1399, m=1.240, P=20.0,  v=45.0, gsm8k=0.880, measured=False, E_int=0.0538),
    dict(name="MacBook Pro 14 M1 Pro 32GB (2021, est.)",
         C=850,  C_new=2399, m=1.600, P=25.0,  v=40.0, gsm8k=0.880, measured=False, E_int=0.0696),
    dict(name="Ryzen 7840HS mini PC 32GB (est.)",
         C=500,  C_new=650,  m=1.400, P=35.0,  v=22.0, gsm8k=0.880, measured=False, E_int=0.0),
    dict(name="MacBook Pro 14 M4 Pro 48GB (est.)",
         C=2100, C_new=2399, m=1.600, P=30.0,  v=75.0, gsm8k=0.880, measured=False, E_int=0.0724),
    dict(name=G83_284B_NAME,
         C=220,  C_new=None, m=0.931, P=27.7,  v=2.10, gsm8k=0.800, measured=True,  E_int=0.05313),
    dict(name="Gaming laptop RTX 4090 32GB (est.)",
         C=1900, C_new=2800, m=2.500, P=140.0, v=70.0, gsm8k=0.880, measured=False, E_int=0.0999),
]


def sweep_power(g, devices, power_w):
    """Run 25-cell (T, e) sweep for G83-30B at power_w and print ranks/ratios vs M4 Air."""
    print(f"\nsensitivity (used pricing, G83 at P={power_w:.1f} W) — rank and ratio vs M4 Air")
    print(f"{'T (h)':>6s} " + " ".join(f"{'e=' + f'{e:.2f}':>9s}" for e in (0.15, 0.20, 0.25, 0.30, 0.35)))
    mba_ratios = []
    m1p_ratios = []
    ranks = []
    for TT in (2, 4, 8, 12, 24):
        cells = []
        for e in (0.15, 0.20, 0.25, 0.30, 0.35):
            scored = []
            for d in devices:
                pw = power_w if d is g else d["P"]
                sc = pty(d["v"], d["C"], d["m"], pw, TT, e, d.get("E_int", 0))[0]
                scored.append((sc, d["name"]))
            scored.sort(reverse=True)
            names = [n for _, n in scored]
            rank = names.index(G83_30B_NAME) + 1
            by_name = dict((n, sc) for sc, n in scored)
            r_mba = by_name[G83_30B_NAME] / by_name["MacBook Air 13 M4 24GB (est.)"]
            r_m1p = by_name[G83_30B_NAME] / by_name["MacBook Pro 14 M1 Pro 32GB (2021, est.)"]
            ranks.append(rank)
            mba_ratios.append(r_mba)
            m1p_ratios.append(r_m1p)
            cells.append(f"#{rank}/{r_mba:.2f}x")
        print(f"{TT:6d} " + " ".join(f"{c:>9s}" for c in cells))
    counts = {r: ranks.count(r) for r in sorted(set(ranks))}
    print(f"  ranks across 25 cells: {counts} | vs M4 Air: {min(mba_ratios):.2f}x - {max(mba_ratios):.2f}x "
          f"(median {sorted(mba_ratios)[12]:.2f}x) | vs M1 Pro: {min(m1p_ratios):.2f}x - {max(m1p_ratios):.2f}x "
          f"(median {sorted(m1p_ratios)[12]:.2f}x)")


def main():
    ap = argparse.ArgumentParser(description="Compute Portable Token Yield (PTY) table on all-used pricing.")
    ap.add_argument("--json", nargs="?", const="pty_results.json", default=None,
                    help="Optional JSON output path (off by default).")
    ap.add_argument("-T", type=float, default=8.0, help="Mission duration in hours (default: 8.0).")
    args = ap.parse_args()

    T = args.T
    print(f"PTY (all-used pricing incl. 32GB SO-DIMM, new MSRP ref) at T={T:g} h, e={E_DEFAULT} kWh/kg\n")
    print(f"{'device':41s} {'used$':>6s} {'new$':>6s} {'kg':>6s} {'W':>5s} {'tok/s':>6s} "
          f"{'carried':>8s} {'PTY(used)':>9s} {'vs #1':>7s} {'PTY(new)':>9s}")
    rows = []
    for d in DEVICES:
        s, carried = pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))
        s_new = pty(d["v"], d["C_new"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0] if d["C_new"] else None
        rows.append((s, d, carried, s_new))
    rows.sort(key=lambda r: -r[0])
    top = rows[0][0]
    for s, d, carried, s_new in rows:
        ratio_str = f"{s / top:6.2f}x"
        new_str = f"{d['C_new']:6d}" if d["C_new"] else f"{'—':>6s}"
        pty_new_str = f"{s_new:9.0f}" if s_new is not None else f"{'—':>9s}"
        print(f"{d['name']:41s} {d['C']:6d} {new_str} {d['m']:6.3f} {d['P']:5.1f} "
              f"{d['v']:6.2f} {carried:8.3f} {s:9.0f} {ratio_str:>7s} {pty_new_str}")

    g = next(d for d in DEVICES if d["name"] == G83_30B_NAME)
    sweep_power(g, DEVICES, 19.4)
    sweep_power(g, DEVICES, 30.0)

    # ---- term-by-term attribution against used peers
    sg, cg = pty(g["v"], g["C"], g["m"], g["P"], T, E_int=g["E_int"])
    for label, match in [("MacBook Air M4 24GB (used $950, est.)", "MacBook Air"),
                         ("same-year 2021 M1 Pro 32GB (used $850, est.)", "M1 Pro")]:
        peer = next(d for d in DEVICES if match in d["name"])
        sp, cp = pty(peer["v"], peer["C"], peer["m"], peer["P"], T, E_int=peer["E_int"])
        print(f"\nattribution — G83-30B vs {label}")
        print(f"  total ratio               {sg/sp:5.2f}x  ({sg:.0f} vs {sp:.0f})")
        print(f"  from used price ({peer['C']}/{g['C']})  {peer['C']/g['C']:5.2f}x")
        print(f"  from mass ({cp:.3f}/{cg:.3f})   {cp/cg:5.2f}x")
        print(f"  from speed ({g['v']:.2f}/{peer['v']:.0f})     {g['v']/peer['v']:5.2f}x")

    # ---- whole-system wall power & sunk-RAM / break-even summary
    mac = next(d for d in DEVICES if "MacBook Air" in d["name"])
    sm = pty(mac["v"], mac["C"], mac["m"], mac["P"], T, E_int=mac["E_int"])[0]
    sg30, cg30 = pty(g["v"], g["C"], g["m"], 30.0, T, E_int=g["E_int"])
    sunk19, _ = pty(g["v"], 100, g["m"], 19.4, T, E_int=g["E_int"])
    sunk30, _ = pty(g["v"], 100, g["m"], 30.0, T, E_int=g["E_int"])
    be19 = (g["v"] * 3600 * T) / (sm * cg)
    be30 = (g["v"] * 3600 * T) / (sm * cg30)
    print(f"\nreference notes (T={T:g} h, e={E_DEFAULT}):")
    print(f"  G83-30B at P=30.0 W wall est.: carried {cg30:.3f} kg, PTY {sg30:.0f} ({sg30/sm:.2f}x vs M4 Air)")
    print(f"  Sunk-RAM baseline ($100 bare only): PTY {sunk19:.0f} @ 19.4 W, PTY {sunk30:.0f} @ 30 W")
    print(f"  Break-even G83 kit price to tie M4 Air ({sm:.0f}): ${be19:.0f} @ 19.4 W, ${be30:.0f} @ 30 W")

    if args.json:
        out = [
            dict(
                name=d["name"],
                PTY_used=round(pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0]),
                PTY_new=(round(pty(d["v"], d["C_new"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0])
                         if d["C_new"] else None),
                carried_kg=round(pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[1], 3),
                price_used=d["C"],
                price_new_ref=d["C_new"],
                mass=d["m"],
                power=d["P"],
                tok_s=d["v"],
                measured=d["measured"],
            )
            for d in DEVICES
        ]
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
