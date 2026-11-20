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
not measurements performed on those machines. Note that the dynabook G83/HS is
discontinued (original 2021 corporate MSRP was ~$1,500; used units trade ~$150).
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


# All rows priced used (C, Nov 2026 secondary market); C_new holds retail MSRP
# for reference where applicable (None for discontinued 2021 corporate dynabook).
DEVICES = [
    dict(name="dynabook G83/HS (2021, 30B)",
         C=150,  C_new=None, m=0.931, P=19.4,  v=8.94, gsm8k=0.880, measured=True,  E_int=0.05313),
    dict(name="MacBook Air 13 M4 24GB (est.)",
         C=950,  C_new=1399, m=1.240, P=20.0,  v=45.0, gsm8k=0.880, measured=False, E_int=0.0538),
    dict(name="MacBook Pro 14 M1 Pro 32GB (2021, est.)",
         C=850,  C_new=2399, m=1.600, P=25.0,  v=40.0, gsm8k=0.880, measured=False, E_int=0.0696),
    dict(name="Ryzen 7840HS mini PC 32GB (est.)",
         C=500,  C_new=650,  m=1.400, P=35.0,  v=22.0, gsm8k=0.880, measured=False, E_int=0.0),
    dict(name="MacBook Pro 14 M4 Pro 48GB (est.)",
         C=2100, C_new=2399, m=1.600, P=30.0,  v=75.0, gsm8k=0.880, measured=False, E_int=0.0724),
    dict(name="dynabook G83/HS (2021, 284B)",
         C=150,  C_new=None, m=0.931, P=27.7,  v=2.10, gsm8k=0.800, measured=True,  E_int=0.05313),
    dict(name="Gaming laptop RTX 4090 32GB (est.)",
         C=1900, C_new=2800, m=2.500, P=140.0, v=70.0, gsm8k=0.880, measured=False, E_int=0.0999),
]


def main():
    ap = argparse.ArgumentParser(description="Compute Portable Token Yield (PTY) table on all-used pricing.")
    ap.add_argument("--json", nargs="?", const="pty_results.json", default=None,
                    help="Optional JSON output path (off by default).")
    ap.add_argument("-T", type=float, default=8.0, help="Mission duration in hours (default: 8.0).")
    args = ap.parse_args()

    T = args.T
    print(f"PTY (all-used pricing, new MSRP ref) at T={T:g} h, e={E_DEFAULT} kWh/kg\n")
    print(f"{'device':41s} {'used$':>6s} {'new$':>6s} {'kg':>6s} {'W':>5s} {'tok/s':>6s} "
          f"{'carried':>8s} {'PTY(used)':>9s} {'vs G83':>8s} {'PTY(new)':>9s}")
    rows = []
    for d in DEVICES:
        s, carried = pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))
        s_new = pty(d["v"], d["C_new"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0] if d["C_new"] else None
        rows.append((s, d, carried, s_new))
    rows.sort(key=lambda r: -r[0])
    base = next(s for s, d, _, _ in rows if "30B" in d["name"])
    for s, d, carried, s_new in rows:
        ratio = base / s
        ratio_str = f"1/{ratio:.2f}x" if ratio > 1.005 else "1.00x"
        new_str = f"{d['C_new']:6d}" if d["C_new"] else f"{'—':>6s}"
        pty_new_str = f"{s_new:9.0f}" if s_new is not None else f"{'—':>9s}"
        print(f"{d['name']:41s} {d['C']:6d} {new_str} {d['m']:6.3f} {d['P']:5.1f} "
              f"{d['v']:6.2f} {carried:8.3f} {s:9.0f} {ratio_str:>8s} {pty_new_str}")

    # ---- sensitivity across mission duration T and cell specific energy e (used pricing)
    print("\nsensitivity (used pricing) — rank of G83-30B and lead over runner-up")
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
            rank = names.index("dynabook G83/HS (2021, 30B)") + 1
            lead = scored[0][0] / max(1e-9, scored[1][0])
            leads.append(lead)
            cells.append(f"{rank}/{lead:.2f}x")
        print(f"{TT:6d} " + " ".join(f"{c:>7s}" for c in cells))
    print(f"  lead range across 25 scenarios: {min(leads):.2f}x - {max(leads):.2f}x "
          f"(median {sorted(leads)[len(leads)//2]:.2f}x)")

    # ---- term-by-term attribution against used peers
    g = next(d for d in DEVICES if "30B" in d["name"])
    sg, cg = pty(g["v"], g["C"], g["m"], g["P"], T, E_int=g["E_int"])
    for label, match in [("MacBook Air M4 24GB (used $950, est.)", "MacBook Air"),
                         ("same-year 2021 M1 Pro 32GB (used $850, est.)", "M1 Pro")]:
        peer = next(d for d in DEVICES if match in d["name"])
        sp, cp = pty(peer["v"], peer["C"], peer["m"], peer["P"], T, E_int=peer["E_int"])
        print(f"\nattribution — G83-30B vs {label}")
        print(f"  total ratio            {sg/sp:5.2f}x  ({sg:.0f} vs {sp:.0f})")
        print(f"  from used price ({peer['C']}/{g['C']}) {peer['C']/g['C']:5.2f}x")
        print(f"  from mass ({cp:.2f}/{cg:.2f})     {cp/cg:5.2f}x")
        print(f"  from speed ({g['v']:.2f}/{peer['v']:.0f})     {g['v']/peer['v']:5.2f}x")

    # ---- whole-system power sensitivity (RAPL package-0=19.4W -> wall est.=30.0W)
    mac = next(d for d in DEVICES if "MacBook Air" in d["name"])
    sm = pty(mac["v"], mac["C"], mac["m"], mac["P"], T, E_int=mac["E_int"])[0]
    sg30, cg30 = pty(g["v"], g["C"], g["m"], 30.0, T, E_int=g["E_int"])
    print("\nwhole-system power sensitivity — G83-30B at P=30.0 W (est. wall incl. DRAM/display)")
    print(f"  T={T:g} h: carried {cg30:.3f} kg, PTY {sg30:.0f} -> {sg30/sm:.2f}x vs used MacBook Air M4")
    wall_ratios = []
    wins = 0
    for TT in (2, 4, 8, 12, 24):
        for e in (0.15, 0.20, 0.25, 0.30, 0.35):
            g_wall = pty(g["v"], g["C"], g["m"], 30.0, TT, e, g["E_int"])[0]
            best_peer = max(
                pty(d["v"], d["C"], d["m"], d["P"], TT, e, d.get("E_int", 0))[0]
                for d in DEVICES if d is not g
            )
            r = g_wall / best_peer
            wall_ratios.append((r, TT, e))
            if r >= 1.0:
                wins += 1
    wall_vals = sorted(r for r, _, _ in wall_ratios)
    worst = min(wall_ratios, key=lambda x: x[0])
    print(f"  at P=30W across 25 (T, e) cells: wins {wins}/25, "
          f"range {min(wall_vals):.2f}x - {max(wall_vals):.2f}x (median {wall_vals[len(wall_vals)//2]:.2f}x; "
          f"worst {worst[0]:.2f}x at T={worst[1]}h, e={worst[2]:.2f})")

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
