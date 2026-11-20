#!/usr/bin/env python3
"""
PTY — Portable Token Yield
==========================

Replaces the previous AHAI index, which had four hand-picked constants
(4.0 odds baseline, 250 tok/task, 25 GiB working set) and two binary gates
that handed non-participants a score of 0.

PTY has no performance constants. It is a ratio of two measured/published quantities:

    PTY(T) = (v * 3600 * T) / [ C * (m + deficit/e) ]

  v  sustained generation, tok/s          (measured or BW-grounded est., 30B-A3B class)
  T  mission duration, hours              (scenario, reported at 8 h + sweep 2-24 h)
  C  kit price, USD                       (paid / street price)
  m  device mass, kg                      (weighed / spec)
  P  inference power, W                   (RAPL package-0 for G83; whole-kit est. for peers)
  e  Li-ion specific energy, kWh/kg       (physical constant, 0.25)

Unit: tok / ($ * kg) — tokens produced per dollar of hardware per kilogram
carried. Capability does not multiply the score; it defines the class a device
competes in (GSM8K >= 80% class).

All non-G83 rows are order-of-magnitude estimates for a Qwen3-30B-A3B Q4-class
workload (1.92 GB/tok) grounded in each system's memory bandwidth and cooling envelope,
not measurements performed on those machines. For G83-30B we also print a 30 W
whole-system power sensitivity check alongside the 19.4 W RAPL package-0 reading.
"""
import argparse
import json

E_DEFAULT = 0.25  # kWh/kg, Li-ion pack level (cell ~0.3, pack ~0.22-0.25)


def pty(v, C, m, P, T, e=E_DEFAULT, E_int=0.0):
    """E_int = internal battery, kWh. Spare cells are carried only for the
    deficit between what the mission requires and what the device holds."""
    deficit = max(0.0, P * T - E_int * 1000) / 1000.0  # kWh
    carried = m + deficit / e                          # kg
    return (v * 3600 * T) / (C * carried), carried


# Non-G83 v/P values are order-of-magnitude estimates for 30B-A3B Q4 (~1.92 GB/tok):
#   - MacBook Air 13 M4 24GB: ~120 GB/s unified RAM, fanless sustained ~45 tok/s
#   - MacBook Pro 14 M4 Pro 48GB: ~273 GB/s unified RAM, active fans ~75 tok/s
#   - Ryzen mini PC 32GB: dual-channel DDR5-5600 (~65-75 GB/s) + PSU, ~22 tok/s
#   - Gaming laptop RTX 4090 32GB: 16 GB VRAM + DDR5 spill, ~70 tok/s
DEVICES = [
    dict(name="dynabook G83/HS (used, 30B)",
         C=150,  m=0.931, P=19.4, v=8.94,  gsm8k=0.880, measured=True,  E_int=0.05313),
    dict(name="MacBook Air 13 M4 24GB (est.)",
         C=1199, m=1.24,  P=20.0, v=45.0,  gsm8k=0.880, measured=False, E_int=0.0538),
    dict(name="Ryzen mini PC 32GB (est.)",
         C=600,  m=1.40,  P=35.0, v=22.0,  gsm8k=0.880, measured=False, E_int=0.0),
    dict(name="dynabook G83/HS (used, 284B)",
         C=150,  m=0.931, P=27.7, v=2.10,  gsm8k=0.800, measured=True,  E_int=0.05313),
    dict(name="MacBook Pro 14 M4 Pro 48GB (est.)",
         C=1999, m=1.60,  P=30.0, v=75.0,  gsm8k=0.880, measured=False, E_int=0.0724),
    dict(name="Gaming laptop RTX 4090 32GB (est.)",
         C=2400, m=2.50,  P=140.0, v=70.0, gsm8k=0.880, measured=False, E_int=0.0999),
]


def main():
    ap = argparse.ArgumentParser(description="Compute Portable Token Yield (PTY) table and sensitivity.")
    ap.add_argument("--json", nargs="?", const="pty_results.json", default=None,
                    help="Optional JSON output path (off by default).")
    ap.add_argument("-T", type=float, default=8.0, help="Mission duration in hours (default: 8.0).")
    args = ap.parse_args()

    T = args.T
    print(f"PTY at T={T:g} h, e={E_DEFAULT} kWh/kg\n")
    print(f"{'device':38s} {'$':>6s} {'kg':>6s} {'W':>5s} {'tok/s':>6s} "
          f"{'carried':>8s} {'PTY':>7s} {'vs G83':>8s}")
    rows = []
    for d in DEVICES:
        s, carried = pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))
        rows.append((s, d, carried))
    rows.sort(key=lambda r: -r[0])
    base = next(s for s, d, _ in rows if "30B" in d["name"])
    for s, d, carried in rows:
        ratio = base / s
        ratio_str = f"1/{ratio:.2f}x" if ratio > 1.005 else "1.00x"
        print(f"{d['name']:38s} {d['C']:6d} {d['m']:6.3f} {d['P']:5.1f} "
              f"{d['v']:6.2f} {carried:8.3f} {s:7.0f} {ratio_str:>8s}")

    # ---- sensitivity across mission duration T and cell specific energy e
    print("\nsensitivity — rank of G83-30B and lead over runner-up")
    print(f"{'T (h)':>6s} " + " ".join(f"e={e:>5.2f}" for e in (0.15, 0.20, 0.25, 0.30, 0.35)))
    leads = []
    for TT in (2, 4, 8, 12, 24):
        cells = []
        for e in (0.15, 0.20, 0.25, 0.30, 0.35):
            scored = sorted(
                ((pty(d["v"], d["C"], d["m"], d["P"], TT, e, d.get("E_int", 0))[0], d["name"])
                 for d in DEVICES),
                reverse=True,
            )
            names = [n for _, n in scored]
            rank = names.index("dynabook G83/HS (used, 30B)") + 1
            lead = scored[0][0] / max(1e-9, scored[1][0])
            leads.append(lead)
            cells.append(f"{rank}/{lead:.2f}x")
        print(f"{TT:6d} " + " ".join(f"{c:>7s}" for c in cells))
    print(f"  lead range across 25 scenarios: {min(leads):.2f}x - {max(leads):.2f}x "
          f"(median {sorted(leads)[len(leads)//2]:.2f}x)")

    # ---- term-by-term attribution vs MacBook Air M4
    g = next(d for d in DEVICES if "30B" in d["name"])
    mac = next(d for d in DEVICES if "MacBook Air" in d["name"])
    sg, cg = pty(g["v"], g["C"], g["m"], g["P"], T, E_int=g["E_int"])
    sm, cm = pty(mac["v"], mac["C"], mac["m"], mac["P"], T, E_int=mac["E_int"])
    print("\nattribution — G83-30B vs MacBook Air M4 (est.)")
    print(f"  total ratio            {sg/sm:5.2f}x  ({sg:.0f} vs {sm:.0f})")
    print(f"  from price  (1199/150) {mac['C']/g['C']:5.2f}x")
    print(f"  from mass   ({cm:.2f}/{cg:.2f}) {cm/cg:5.2f}x")
    print(f"  from speed  ({g['v']:.2f}/{mac['v']:.0f}) {g['v']/mac['v']:5.2f}x")

    # ---- whole-system power sensitivity (RAPL package-0=19.4W -> wall est.=30.0W)
    print("\nwhole-system power sensitivity — G83-30B at P=30.0 W (est. wall incl. DRAM/display)")
    sg30, cg30 = pty(g["v"], g["C"], g["m"], 30.0, T, E_int=g["E_int"])
    print(f"  T={T:g} h: carried {cg30:.3f} kg, PTY {sg30:.0f} -> {sg30/sm:.2f}x vs MacBook Air M4")
    wall_leads = []
    for TT in (2, 4, 8, 12, 24):
        for e in (0.15, 0.20, 0.25, 0.30, 0.35):
            g_wall = pty(g["v"], g["C"], g["m"], 30.0, TT, e, g["E_int"])[0]
            runner = max(
                pty(d["v"], d["C"], d["m"], d["P"], TT, e, d.get("E_int", 0))[0]
                for d in DEVICES if d is not g
            )
            wall_leads.append(g_wall / runner)
    print(f"  at P=30W across 25 (T, e) cells: min {min(wall_leads):.2f}x, "
          f"max {max(wall_leads):.2f}x, median {sorted(wall_leads)[len(wall_leads)//2]:.2f}x")

    if args.json:
        out = [
            dict(
                name=d["name"],
                PTY=round(pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0]),
                carried_kg=round(pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[1], 3),
                price=d["C"],
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
