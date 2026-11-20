#!/usr/bin/env python3
"""
PTY — Portable Token Yield
==========================

Replaces the previous AHAI index, which had four hand-picked constants
(4.0 odds baseline, 250 tok/task, 25 GiB working set) and two binary gates
that handed non-participants a score of 0.

PTY has no performance constants. It is a ratio of two measured quantities:

    PTY(T) = (v * 3600 * T) / [ C * (m + P*T/e) ]

  v  sustained generation, tok/s          (measured, same model class)
  T  mission duration, hours              (scenario, reported at 2/8/24 h)
  C  kit price, USD                       (paid price)
  m  device mass, kg                      (weighed)
  P  inference power, W                   (RAPL package or published TDP)
  e  Li-ion specific energy, kWh/kg       (physical constant, 0.25)

Unit: tok / ($ * kg) — tokens produced per dollar of hardware per kilogram
carried. Every factor is a published or measured number, so the score can be
taken apart by hand. Capability does not multiply the score; it defines the
class a device competes in, the way a weight class does in boxing.

The only judgement left is T (how long the mission is) and the equal weighting
of "one dollar" against "one kilogram" in relative terms. Both are stated.
"""
import json

E_DEFAULT = 0.25  # kWh/kg, Li-ion pack level (cell ~0.3, pack ~0.22-0.25)


def pty(v, C, m, P, T, e=E_DEFAULT, E_int=0.0):
    """E_int = internal battery, kWh. Spare cells are only carried for the
    deficit between what the mission needs and what the device already holds."""
    deficit = max(0.0, P * T - E_int * 1000) / 1000.0   # kWh
    carried = m + deficit / e                            # kg
    return (v * 3600 * T) / (C * carried), carried


# ---------------------------------------------------------------- device table
# v/P for non-G83 devices are order-of-magnitude estimates from published
# specs; everything for G83 is measured on the machine.
DEVICES = [
    dict(name="dynabook G83/HS (used, 30B)",
         C=150,  m=0.931, P=19.4, v=8.94,  gsm8k=0.880, measured=True, E_int=0.05313),
    dict(name="dynabook G83/HS (used, 284B)",
         C=150,  m=0.931, P=27.7, v=2.10,  gsm8k=0.800, measured=True, E_int=0.05313),
    dict(name="MacBook Air 13 M4 24GB",
         C=1199, m=1.24,  P=20,   v=11.0,  gsm8k=0.880, measured=False, E_int=0.0538),
    dict(name="MacBook Pro 14 M4 Pro 48GB",
         C=1999, m=1.60,  P=28,   v=14.0,  gsm8k=0.880, measured=False, E_int=0.0724),
    dict(name="Ryzen mini PC 32GB",
         C=600,  m=1.40,  P=30,   v=6.0,   gsm8k=0.880, measured=False, E_int=0.0),
    dict(name="Gaming laptop RTX 4090 32GB",
         C=2400, m=2.50,  P=140,  v=35.0,  gsm8k=0.880, measured=False, E_int=0.0999),
]


def main():
    T = 8
    print(f"PTY at T={T} h, e={E_DEFAULT} kWh/kg\n")
    print(f"{'device':38s} {'$':>6s} {'kg':>6s} {'W':>5s} {'tok/s':>6s} "
          f"{'carried':>8s} {'PTY':>7s} {'vs G83':>7s}")
    rows = []
    for d in DEVICES:
        s, carried = pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))
        rows.append((s, d, carried))
    rows.sort(key=lambda r: -r[0])
    base = next(s for s, d, _ in rows if "30B" in d["name"])
    for s, d, carried in rows:
        print(f"{d['name']:38s} {d['C']:6d} {d['m']:6.3f} {d['P']:5.0f} "
              f"{d['v']:6.2f} {carried:8.3f} {s:7.0f} {base/s:6.1f}x")

    # ---- sensitivity: does the ranking survive reasonable changes?
    print("\nsensitivity — rank of G83-30B under different scenarios")
    print(f"{'T (h)':>6s} " + " ".join(f"e={e:>5.2f}" for e in (0.15, 0.20, 0.25, 0.30, 0.35)))
    for TT in (2, 4, 8, 12, 24):
        cells = []
        for e in (0.15, 0.20, 0.25, 0.30, 0.35):
            scored = sorted(
                ((pty(d["v"], d["C"], d["m"], d["P"], TT, e, d.get("E_int", 0))[0], d["name"])
                 for d in DEVICES), reverse=True)
            names = [n for _, n in scored]
            rank = names.index("dynabook G83/HS (used, 30B)") + 1
            top = scored[0][0]
            runner = scored[1][0]
            cells.append(f"{rank}/{top/max(1e-9, runner):.1f}x")
        print(f"{TT:6d} " + " ".join(f"{c:>7s}" for c in cells))
    print("  cell = rank of G83-30B / lead over 2nd place")

    # ---- how much of the win is price, how much is mass, how much is speed?
    print("\nattribution — G83-30B vs MacBook Air M4")
    g = next(d for d in DEVICES if d["name"].startswith("dynabook") and "30B" in d["name"])
    mac = next(d for d in DEVICES if d["name"].startswith("MacBook Air"))
    sg = pty(g["v"], g["C"], g["m"], g["P"], T, E_int=g["E_int"])[0]
    sm = pty(mac["v"], mac["C"], mac["m"], mac["P"], T, E_int=mac["E_int"])[0]
    print(f"  total ratio            {sg/sm:5.2f}x")
    print(f"  from price   (1199/150) {mac['C']/g['C']:5.2f}x")
    print(f"  from mass    {pty(mac['v'], mac['C'], mac['m'], mac['P'], T, E_int=mac['E_int'])[1]/pty(g['v'], g['C'], g['m'], g['P'], T, E_int=g['E_int'])[1]:5.2f}x"
          f"  (carried {pty(g['v'], g['C'], g['m'], g['P'], T, E_int=g['E_int'])[1]:.2f} vs {pty(mac['v'], mac['C'], mac['m'], mac['P'], T, E_int=mac['E_int'])[1]:.2f} kg)")
    print(f"  from speed   (8.94/11)  {g['v']/mac['v']:5.2f}x")

    json.dump([dict(name=d["name"], PTY=round(pty(d["v"], d["C"], d["m"], d["P"], T, E_int=d.get("E_int", 0))[0]),
                    price=d["C"], mass=d["m"], power=d["P"], tok_s=d["v"],
                    measured=d["measured"]) for d in DEVICES],
              open("pty_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
