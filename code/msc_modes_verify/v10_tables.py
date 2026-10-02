#!/usr/bin/env python
"""Check by the second implementation, part 10: post-mode monotonicity from the pole expansion, and tables."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); V = _os.path.join(_R, 'data', 'msc_modes_verify')
A = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'analysis.json')))
P = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes_verify', 'poles.jsonl'))]
mono = []
for r in P:
    nus, res = np.array(r["nu"]), np.array(r["res"])
    gp = lambda t: -np.sum(res * nus * np.exp(-nus * t))
    up = np.logspace(np.log10(r["mode"] * 1.0001), np.log10(15 * r["mfpt"]), 4000)
    dn = np.linspace(0.6 * r["mode"], r["mode"] * 0.9999, 400)
    mono.append(dict(key=r["key"], decreasing_after_mode=bool(all(gp(t) < 0 for t in up)), increasing_before_mode=bool(all(gp(t) > 0 for t in dn))))
json.dump(mono, open(_os.path.join(_R, 'data', 'msc_modes_verify', 'monotonicity_poles.json'), "w"), indent=1)
print("pole-expansion cases:", len(mono), "all decreasing after mode to 15 MFPT:", all(m["decreasing_after_mode"] for m in mono),
      "all increasing on [0.6,1) mode:", all(m["increasing_before_mode"] for m in mono))
L = []
for d in ("1", "2", "3"):
    L.append(f"\n**Table V{d}. d = {d}, corner to corner, q = 0.8: exact discrete argmax (C stepper, this verification).**\n")
    L.append("| N | mode | MFPT (exact) | mode/MFPT | q·mode/N² | P(T ≤ mode) |\n|---|---|---|---|---|---|")
    for x in A["cc_q0.8"][d]:
        L.append(f"| {x['N']} | {x['mode']} | {x['mfpt']:.4f} | {x['ratio']:.5f} | {x['qmode_over_N2']:.4f} | {x['cdf_at_mode']:.4f} |")
for d in (2, 3):
    L.append(f"\n**Table W{d}. d = {d}, corner to corner, continuous time, unit rate: pole expansion (this check) vs Talbot (main implementation).**\n")
    L.append("| N | mode | mode/MFPT | mode/N² | X | rel. diff. to main-implementation mode | L1 error | 3-pole error | P(T ≤ mode) | median/MFPT | 99 % band / mode |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for x in A["poles_cc"]:
        if x["d"] == d:
            rd = f"{x['rel_diff_mode']:+.1e}" if "rel_diff_mode" in x else "–"
            L.append(f"| {x['N']} | {x['mode']:.6g} | {x['ratio']:.5f} | {x['mode_over_N2']:.5f} | {x['X']:.3f} | {rd} | {100*x['L1_err']:+.3f} % | {x['pole3_err']:+.1e} | {x['cdf_at_mode']:.4f} | {x['median_over_mfpt']:.4f} | [{x['band99'][0]:.3f}, {x['band99'][1]:.3f}] |")
open(_os.path.join(_R, 'data', 'msc_modes_verify', 'tables.md'), "w").write("\n".join(L) + "\n")
print("tables written")
