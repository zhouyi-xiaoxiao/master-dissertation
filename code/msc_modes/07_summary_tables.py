"""
07_summary_tables.py -- assemble Markdown summary tables from the saved results
(no computation).  Output: data/msc_modes/summary_tables.md
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
D = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')) if l.strip()]
L = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')) if l.strip()]
FIT = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))
Q = 0.8
out = []


def sel(rows, **kw):
    return sorted([r for r in rows if all(r.get(k) == v for k, v in kw.items())], key=lambda r: r["N"])


def dmode(d, geo, N, q=Q):
    m = [r for r in D if (r["d"], r["geo"], r["N"], r["q"]) == (d, geo, N, q)]
    return m[0] if m else None


def loc_exp(rows, i, key="mode"):
    if i == 0 or i == len(rows) - 1:
        return float("nan")
    a, b = rows[i - 1], rows[i + 1]
    return math.log(b[key] / a[key]) / math.log(b["N"] / a["N"])


for d, pick in ((2, [5, 10, 20, 35, 50, 100, 200, 400, 1280, 4001, 10240, 163840, 1048576, 16777216]),
                (3, [5, 10, 20, 35, 40, 50, 100, 200, 320, 640, 1280, 4096])):
    rows = sel(L, d=d, geo="CC")
    out.append(f"\n**Table M{d}. Corner-to-corner, d = {d}: exact modes and closed-form predictions.** "
               "Continuous-time values are for unit jump rate (multiply times by 1/q for activity q); "
               "'discrete' is the argmax of the exact PMF at q = 0.8.\n")
    out.append("| N | q·mode/N² | mode/MFPT | discrete mode (q=0.8) | q·mode_disc − mode_cont | X = μ₁·MFPT | L0 err | L1 err | 2-pole err | 3-pole err |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        if r["N"] not in pick:
            continue
        dm = dmode(d, "CC", r["N"]) or (dmode(d, "CC", 401) if r["N"] == 400 and False else None)
        ds = f"{dm['mode']}" if dm else "–"
        off = f"{Q * dm['mode'] - r['mode']:+.2f}" if dm else "–"
        e = lambda k: f"{100 * (r[k] / r['mode'] - 1):+.3f}%" if abs(r[k] / r['mode'] - 1) > 5e-6 else f"{(r[k] / r['mode'] - 1):+.1e}"
        out.append(f"| {r['N']} | {r['mode'] / r['N'] ** 2:.5f} | {r['mode'] / r['mfpt']:.5f} | {ds} | {off} | "
                   f"{r['p'][0] * r['mfpt']:.3f} | {e('pred_L0')} | {e('pred_L1')} | {e('pred_2pole')} | {e('pred_3pole')} |")

rows = sel(L, d=1, geo="CC")
out.append("\n**Table M1. End-to-end chain, d = 1.**\n")
out.append("| N | mode_cont/(N−½)² (unit rate) | mode/MFPT (cont.) | discrete mode (q=0.8) | discrete mode/MFPT |")
out.append("|---|---|---|---|---|")
for r in rows:
    if r["N"] in (5, 10, 20, 50, 100, 200, 1000, 10240, 100000, 1000000) or r["N"] == 57:
        out.append(f"| {r['N']} | {r['mode'] / (r['N'] - 0.5) ** 2:.8f} | {r['mode'] / r['mfpt']:.8f} | {r['mode_discrete_q0.8']} | "
                   f"{r['mode_discrete_q0.8'] / r['mfpt_discrete_q0.8']:.8f} |")

out.append("\n**Table G. Other start/target placements (continuous time, unit rate).** C2M: corner start, centre target; "
           "M2C: centre start, corner target.\n")
out.append("| d | geometry | N | mode/N² | mode/MFPT | L1 err | 3-pole err | sign of a₁ |")
out.append("|---|---|---|---|---|---|---|---|")
for d in (2, 3):
    for geo in ("C2M", "M2C"):
        for r in sel(L, d=d, geo=geo):
            if r["N"] in (11, 41, 161, 641, 1281, 10241, 655361):
                l1 = f"{100 * (r['pred_L1'] / r['mode'] - 1):+.3f}%" if r.get("pred_L1") else "n/a"
                p3 = f"{(r['pred_3pole'] / r['mode'] - 1):+.1e}" if r.get("pred_3pole") == r.get("pred_3pole") and r.get("pred_3pole") else "n/a"
                out.append(f"| {d} | {geo} | {r['N']} | {r['mode'] / r['N'] ** 2:.5f} | {r['mode'] / r['mfpt']:.5f} | {l1} | {p3} | "
                           f"{'−' if r['res'][1] < 0 else '+'} |")

out.append("\n**Table Q. Activity dependence and parity (exact discrete PMFs).** 'two-step' is the first index of the "
           "maximising pair of (f(t)+f(t+1))/2.\n")
out.append("| d | N | q | argmax f | two-step argmax | q·argmax | continuous-time mode (unit rate) |")
out.append("|---|---|---|---|---|---|---|")
for d, Ns in ((1, (51, 201)), (2, (21, 51, 101)), (3, (11, 21, 31))):
    for N in Ns:
        c = [r for r in L if (r["d"], r["geo"], r["N"]) == (d, "CC", N)]
        cm = f"{c[0]['mode']:.2f}" if c else "–"
        for q in (0.3, 0.5, 0.8, 0.9, 1.0):
            r = dmode(d, "CC", N, q)
            if r is None:
                # q=0.8 for these N may not have been run for all sizes
                continue
            out.append(f"| {d} | {N} | {q} | {r['mode']} | {r['mode_two_step_avg']} | {q * r['mode']:.1f} | {cm} |")

if os.path.exists(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json')):
    PS = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'profile_stats.json')))
    out.append("\n**Table S. How sharp and how 'typical' is the mode? (corner-to-corner, continuous time).** "
               "g_max·MFPT = 1 for a pure exponential law; the 99% band is the time window in which g ≥ 0.99 g_max.\n")
    out.append("| d | N | g_max·MFPT | P(T ≤ mode) | S(MFPT) | 99% band / mode | 50% band / mode |")
    out.append("|---|---|---|---|---|---|---|")
    for s in PS:
        out.append(f"| {s['d']} | {s['N']} | {s['g_max_times_mfpt']:.5f} | {1 - s['S_at_mode']:.5f} | {s['S_at_mfpt']:.5f} | "
                   f"[{s['t_lo_0.99'] / s['mode']:.3f}, {s['t_hi_0.99'] / s['mode']:.3f}] | "
                   f"[{s['t_lo_0.5'] / s['mode']:.3f}, {s['t_hi_0.5'] / s['mode']:.2f}] |")
if os.path.exists(_os.path.join(_R, 'data', 'msc_modes', 'medians.json')):
    M = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'medians.json')))
    out.append("\n**Table Md. Mode, median and mean (corner-to-corner, continuous time).**\n")
    out.append("| d | N | mode/MFPT | median/MFPT | P(T ≤ mode) | S(MFPT) |")
    out.append("|---|---|---|---|---|---|")
    for m in M:
        out.append(f"| {m['d']} | {m['N']} | {m['mode_over_mfpt']:.5f} | {m['median_over_mfpt']:.5f} | {m['P_T_le_mode']:.5f} | {m['S_at_mfpt']:.5f} |")
if os.path.exists(_os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')):
    T = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')))
    out.append("\n**Table U. Full-support check of the exact discrete PMF (iterated until S < 1e-12).**\n")
    out.append("| d | geometry | N | q | steps | 1 − mass | mode | local maxima | local minima | mean from PMF | CV | P(T ≤ mode) | median |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in T:
        out.append(f"| {r['d']} | {r['geo']} | {r['N']} | {r['q']} | {r['t_end']} | {1 - r['mass']:.1e} | {r['mode']} | "
                   f"{r['n_local_maxima_full_support']} | {r['n_local_minima_full_support']} | {r['mean_from_pmf']:.4f} | {r['cv']:.4f} | "
                   f"{r['P_T_le_mode']:.4f} | {r['median']} |")
open(_os.path.join(_R, 'data', 'msc_modes', 'summary_tables.md'), "w").write("\n".join(out) + "\n")
print("\n".join(out))
