#!/usr/bin/env python
"""Supplementary Section S10 (appC_tables): LaTeX rows of the extended tables, copied from the result files.

The numerical tables of sections/appC_tables.tex are not typed by hand.  This script reads the
source tables, converts every entry to LaTeX WITHOUT changing a digit (only typography changes:
thin spaces as thousands separators, 'e' notation -> powers of ten, a per cent sign moved into
the column heading, the two residual columns of the fit tables expressed in units of 1e-2 by an
exact decimal shift, the half-widths of fitted parameters printed in brackets without the '+/-'
sign), and writes one fragment per table to  data/article/appC_tables_fragments/<name>.tex .
Only a subset of the rows of Tables T1 and T2 is printed (SUBSET below); in T1 the ratio is printed to five
decimals, computed from the exact mode and the exact mean N(N-1)/q; three large-N rows of
T1 are added from the 50-digit check data/msc_modes_verify/bigN_1d_mp.json, with the
mean N(N-1)/q evaluated exactly.

It also cross-checks the Markdown tables against the underlying JSON/CSV files and computes the
few derived numbers quoted in the text.  Everything goes to data/article/appC_tables_checks.json .

Inputs (relative to the root of the working tree):
    data/msc_modes/tables.md, summary_tables.md, full_tail.json, fit_results.json
    data/msc_modes_verify/bigN_1d_mp.json
    data/msc_defects/tables/sweep_uniform.md, sweep_smart.md, structured_deterministic.md
    data/msc_defects/summary_sweep_N35.csv, structured_deterministic.csv
    data/msc_defects/sweep/sweep_N35_smart_p*.csv
    data/msc_defects_verify/v06_periodic.csv, v06_det.csv, v03_summary.csv

Usage:
    python appC_tables_build.py              write fragments + checks
    python appC_tables_build.py --assemble   also replace the  %%FRAG:<name>%%  placeholders
                                             of sections/appC_tables.tex by the fragments
    python appC_tables_build.py --check      verify that every fragment occurs verbatim in
                                             sections/appC_tables.tex (exit status 1 otherwise)
Runtime: a few seconds, < 200 MB.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import csv
import glob
import json
import os
import re
import sys
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ART = _R                      # the repository root
ROOT = _R         # 
FRAG = _os.path.join(_R, 'data', 'article', 'appC_tables_fragments')
SECTION = _os.path.join(_R, 'sections', 'appC_tables.tex')
OUTJSON = _os.path.join(_R, 'data', 'article', 'appC_tables_checks.json')

os.makedirs(FRAG, exist_ok=True)
checks = {}
fragments = {}


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def md_table(text, title_start):
    """Rows (list of lists of stripped cells) of the Markdown table that follows the line
    starting with `title_start`; header and separator removed.  title_start=None: the text
    is one bare table."""
    lines = text.splitlines()
    i0 = 0
    if title_start is not None:
        hits = [i for i, l in enumerate(lines) if l.startswith(title_start)]
        if len(hits) != 1:
            raise RuntimeError(f"title {title_start!r}: {len(hits)} matches")
        i0 = hits[0] + 1
    rows = []
    started = False
    for l in lines[i0:]:
        if l.startswith("|"):
            started = True
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
        elif started:
            break
    header, body = rows[0], rows[2:]
    return header, body


def group(intpart):
    """Thin-space thousands separators for integer parts with five or more digits."""
    if len(intpart) < 5:
        return intpart
    out = []
    while intpart:
        out.insert(0, intpart[-3:])
        intpart = intpart[:-3]
    return r"\,".join(out)


def tex_num(s):
    """One number as it is printed in a source table -> LaTeX math (no surrounding $).
    Accepts thousands blanks ('14 100.8'), signs, decimals and 'e' notation."""
    s = s.strip().replace("−", "-").replace(" ", "").replace(" ", "")
    m = re.fullmatch(r"([+-]?)(\d+)(\.\d+)?(?:[eE]([+-]?\d+))?", s)
    if not m:
        raise ValueError(f"not a number: {s!r}")
    sign, ip, frac, ex = m.groups()
    body = sign + group(ip) + (frac or "")
    if ex is not None:
        body += r"\times10^{" + str(int(ex)) + "}"
    return body


def cell(s):
    return "$" + tex_num(s) + "$"


def pm_cell(s):
    """'14 711 ± 118' -> $14\\,711\\pm118$ ;  plain numbers pass through."""
    if "±" in s:
        a, b = s.split("±")
        return "$" + tex_num(a) + r"\pm" + tex_num(b) + "$"
    return cell(s)


def write_fragment(name, rows):
    txt = "\n".join(rows) + "\n"
    fragments[name] = txt
    with open(os.path.join(FRAG, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write(txt)


def close(a, b, rel=5e-5, absol=0.0):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), absol)


# ------------------------------------------------------------------------------------------
# 1.  T1-T3: exact modes and means
# ------------------------------------------------------------------------------------------
tables_md = read(_os.path.join(_R, 'data', 'msc_modes', 'tables.md'))
SUBSET = {
    1: [5, 9, 13, 17, 21, 25, 29, 35, 50, 70, 100, 140, 200, 400, 1000],
    2: [5, 9, 13, 17, 21, 25, 29, 35, 50, 70, 100, 140, 200, 301, 401],
    3: None,                       # all rows
}
T = {}
for d in (1, 2, 3):
    hdr, body = md_table(tables_md, f"**Table T{d}.")
    assert hdr[0] == "N" and len(hdr) in (4, 8), hdr        # the exact columns come first
    T[d] = body
    rows = []
    for r in body:
        N = int(r[0])
        if SUBSET[d] is not None and N not in SUBSET[d]:
            continue
        cells = list(r[:4])
        if d == 1:      # ratio to five decimals, from the exact mode and the exact mean N(N-1)/q (q = 4/5)
            cells[3] = f"{float(Fraction(int(r[1])) / Fraction(N * (N - 1) * 5, 4)):.5f}"
        rows.append(" & ".join(cell(c) for c in cells) + r" \\")
    if SUBSET[d] is not None:
        got = [int(r[0]) for r in body if int(r[0]) in SUBSET[d]]
        assert got == SUBSET[d], (d, got)
    write_fragment(f"T{d}", rows)
    # consistency of the source table itself (exact columns)
    for r in body:
        mode, mfpt, ratio = int(r[1]), float(r[2]), float(r[3])
        assert abs(mode / mfpt - ratio) < 6e-5 + 1e-5, (d, r)       # 4 d.p., MFPT printed to 6 s.f.
    checks[f"T{d}_rows_in_source"] = len(body)

# exact ratios quoted in the text (from the source tables)
checks["exact_ratio_2d_min_max_N5_200"] = [min(float(r[3]) for r in T[2] if int(r[0]) <= 200),
                                           max(float(r[3]) for r in T[2] if int(r[0]) <= 200)]
checks["exact_ratio_3d_min_max"] = [min(float(r[3]) for r in T[3]), max(float(r[3]) for r in T[3])]
t2 = {int(r[0]): int(r[1]) for r in T[2]}
checks["q_tmode_over_N2_2d"] = {str(N): 0.8 * t2[N] / N ** 2 for N in (5, 35, 100, 200, 401)}

# large-N rows of T1: exact discrete modes re-established in 50-digit arithmetic
big = json.load(open(_os.path.join(_R, 'data', 'msc_modes_verify', 'bigN_1d_mp.json')))
rows = []
bigrec = []
for b in big:
    N, mode = int(b["N"]), int(b["recheck"])
    assert b["recheck_is_argmax"] is True
    mfpt = Fraction(N * (N - 1) * 5, 4)                    # N(N-1)/q with q = 4/5
    assert mfpt.denominator == 1
    ratio = Fraction(mode) / mfpt
    r4 = f"{float(ratio):.4f}"
    r8 = f"{float(ratio):.8f}"
    bigrec.append({"N": N, "mode": mode, "mfpt": int(mfpt), "ratio_4dp": r4, "ratio_8dp": r8})
    rows.append(" & ".join([cell(str(N)), cell(str(mode)), cell(str(int(mfpt))), cell(f"{float(ratio):.5f}")]) + r" \\")
checks["T1_large_N_rows"] = bigrec
write_fragment("T1big", rows)

# ------------------------------------------------------------------------------------------
# 2.  fit tables (2D and 3D mode)
# ------------------------------------------------------------------------------------------
FORMS = {
    "A N^2 (pure quadratic)": r"$A N^{2}$",
    "A N^3 (pure cubic)": r"$A N^{3}$",
    "A N^a": r"$A N^{a}$",
    "A N^2 (ln N)^b": r"$A N^{2}(\ln N)^{b}$",
    "A N^a (ln N)^b": r"$A N^{a}(\ln N)^{b}$",
    "N^2 (A ln ln N + B)": r"$N^{2}(A\ln\ln N+B)$",
    "N^2 (A ln N + B)": r"$N^{2}(A\ln N+B)$",
}
PNAME = {"lnA": r"\ln A", "a": "a", "b": "b", "A": "A", "B": "B"}


def tex_params(s):
    out = []
    for item in s.split(", "):
        m = re.fullmatch(r"(\w+)=([-+0-9.eE]+) \(±([0-9.eE+-]+)\)", item)
        if not m:
            raise ValueError(item)
        name, val, hw = m.groups()
        out.append("${" + PNAME[name] + "=" + tex_num(val) + r"\,(" + tex_num(hw) + ")}$")
    return r" \newline ".join(out)


def in_units_of_1e_minus_2(s):
    """'8.34e-02' -> '8.34', '5.72e-04' -> '0.0572' (exact decimal shift, no rounding)."""
    v = Decimal(s).scaleb(2)
    t = format(v, "f")
    assert Decimal(t) == Decimal(s) * 100 and len(t.replace(".", "").lstrip("0")) == 3, (s, t)
    return "$" + t + "$"


def pct(s):
    assert s.endswith("%")
    return "$" + tex_num(s[:-1]) + "$"


fitres = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')))["fits"]
FITS = [
    ("fit2d", "**2D mode (continuous time, unit rate): fit on N = 10..200, extrapolated**",
     "2d_mode_fit_10_200_extrapolated", 4),
    ("fit3d", "**3D mode (continuous time, unit rate): fit on N = 10..100, extrapolated**",
     "3d_mode_fit_10_100_extrapolated", 3),
    ("fit2d_discrete", "**2D mode, exact discrete PMF, q = 0.8, N = 11..200 (short range)**",
     "2d_mode_discrete_short_range", 0),
    ("fit3d_discrete", "**3D mode, exact discrete PMF, q = 0.8, N = 11..100 (short range)**",
     "3d_mode_discrete_short_range", 0),
]
for name, title, key, nh in FITS:
    hdr, body = md_table(tables_md, title)
    assert len(hdr) == 6 + nh, hdr
    rows = []
    for r in body:
        jr = fitres[key][r[0]]
        assert jr["form"] == r[1]
        assert f"{jr['rms_ln_residual']:.2e}" == r[3] and f"{jr['max_abs_ln_residual']:.2e}" == r[4]
        assert f"{jr['dAICc']:.1f}" == r[5]
        cells = [FORMS[r[1]], r"\appCrr " + tex_params(r[2]), in_units_of_1e_minus_2(r[3]),
                 in_units_of_1e_minus_2(r[4]), cell(r[5])]
        ext = [pct(c) for c in r[6:]]
        if nh:
            assert [f"{100 * e:+.2f}%" for e in jr["holdout_rel_error"]] == r[6:]
        ext += [""] * (4 - len(ext))
        rows.append(" & ".join(cells + ext) + r" \tabularnewline")
    write_fragment(name, rows)
    checks[f"{name}_n_points"] = fitres[key][body[0][0]]["n_points"]
    if nh:
        checks[f"{name}_holdout_N"] = fitres[key][body[0][0]]["holdout_N"]

# ------------------------------------------------------------------------------------------
# 3.  full-support check (Table U) and activity / parity (Table Q)
# ------------------------------------------------------------------------------------------
summary_md = read(_os.path.join(_R, 'data', 'msc_modes', 'summary_tables.md'))
GEO = {"CC": r"\CC", "C2M": r"\CtM", "M2C": r"\MtC"}

hdr, body = md_table(summary_md, "**Table U.")
assert len(hdr) == 13, hdr
full = json.load(open(_os.path.join(_R, 'data', 'msc_modes', 'full_tail.json')))
assert len(full) == len(body)
rows = []
for r, j in zip(body, full):
    assert (int(r[0]), r[1], int(r[2]), float(r[3])) == (j["d"], j["geo"], j["N"], j["q"]), r
    assert int(r[4]) == j["t_end"] and int(r[6]) == j["mode"] and int(r[12]) == j["median"]
    assert int(r[7]) == j["n_local_maxima_full_support"] and int(r[8]) == j["n_local_minima_full_support"]
    assert f"{1 - j['mass']:.1e}" == r[5], (r[5], 1 - j["mass"])
    assert f"{j['mean_from_pmf']:.4f}" == r[9] and f"{j['cv']:.4f}" == r[10]
    assert f"{j['P_T_le_mode']:.4f}" == r[11]
    assert j["S_end"] < 1e-12
    cells = [r[0], GEO[r[1]], cell(r[2]), cell(r[3]), cell(r[4]), cell(r[5]), cell(r[6]), r[7], r[8],
             cell(r[9]), cell(r[10]), cell(r[11]), cell(r[12])]
    rows.append(" & ".join(cells) + r" \\")
write_fragment("unimodal", rows)
# mean recomputed from the PMF against the exact mean: N(N-1)/q for d = 1 (all rows), and the
# exact corner-to-corner means of Tables T2/T3 (printed to six significant figures) for d = 2, 3
dev1 = [abs(j["mean_from_pmf"] / (j["N"] * (j["N"] - 1) / j["q"]) - 1) for j in full if j["d"] == 1]
checks["unimodal_d1_max_rel_dev_mean_from_pmf_vs_N(N-1)/q"] = max(dev1)
checks["unimodal_d1_mean_equals_N(N-1)/q_to_4_decimals"] = all(
    f"{j['mean_from_pmf']:.4f}" == f"{j['N'] * (j['N'] - 1) / j['q']:.4f}" for j in full if j["d"] == 1)
devT = []
for j in full:
    if j["d"] in (2, 3) and j["geo"] == "CC" and j["q"] == 0.8:
        ref = [float(r[2]) for r in T[j["d"]] if int(r[0]) == j["N"]]
        if ref:
            devT.append(abs(j["mean_from_pmf"] / ref[0] - 1))
checks["unimodal_d23_cc_q0.8_rows_compared_with_T2_T3"] = len(devT)
checks["unimodal_d23_cc_q0.8_max_rel_dev_mean_vs_T2_T3_(6 s.f.)"] = max(devT)
checks["unimodal_runs"] = len(body)
checks["unimodal_all_one_maximum_no_minimum"] = all(r[7] == "1" and r[8] == "0" for r in body)
checks["unimodal_max_abs_one_minus_mass"] = max(abs(1 - j["mass"]) for j in full)
checks["unimodal_max_S_end"] = max(j["S_end"] for j in full)

hdr, body = md_table(summary_md, "**Table Q.")
assert len(hdr) == 7, hdr
rows = []
maxdiff = 0.0
maxdiff_q_lt_1 = 0.0
for r in body:
    assert abs(float(r[2]) * int(r[3]) - float(r[5])) < 0.051, r
    diff = float(r[5]) - float(r[6])
    maxdiff = max(maxdiff, abs(diff))
    if float(r[2]) < 1:
        maxdiff_q_lt_1 = max(maxdiff_q_lt_1, abs(diff))
    rows.append(" & ".join([r[0]] + [cell(c) for c in r[1:]]) + r" \\")
write_fragment("activity", rows)
checks["activity_rows"] = len(body)
checks["activity_max_abs_(q*argmax - continuous mode)_all"] = maxdiff
checks["activity_max_abs_(q*argmax - continuous mode)_q<1"] = maxdiff_q_lt_1
checks["activity_two_step_differs_by_more_than_1_step"] = [
    r for r in body if abs(int(r[3]) - int(r[4])) > 1]

# ------------------------------------------------------------------------------------------
# 4.  defect sweeps (sample A), uniform and corner-thinned placement
# ------------------------------------------------------------------------------------------
def sweep_rows(fname, scheme):
    hdr, body = md_table(read(os.path.join(_R, 'data', 'msc_defects', 'tables', fname)), None)
    assert len(hdr) == 10, hdr
    with open(_os.path.join(_R, 'data', 'msc_defects', 'summary_sweep_N35.csv')) as fh:
        summ = {round(float(r["p"]), 2): r for r in csv.DictReader(fh) if r["scheme"] == scheme}
    rows = []
    for r in body:
        p = float(r[0])
        s = summ[round(p, 2)]
        # cross-check the printed table against the summary CSV
        if p > 0:
            mf, mfci = [float(x.replace(" ", "")) for x in r[2].split("±")]
            mo, moci = [float(x.replace(" ", "")) for x in r[3].split("±")]
            ra, raci = [float(x) for x in r[4].split("±")]
            assert abs(mf - float(s["mfpt"])) <= 0.5 and abs(mfci - float(s["mfpt_ci"])) <= 0.5, r
            assert abs(mo - float(s["mode"])) <= 0.5 and abs(moci - float(s["mode_ci"])) <= 0.5, r
            assert abs(ra - float(s["ratio"])) <= 5.1e-5 and abs(raci - float(s["ratio_ci"])) <= 5.1e-5, r
            assert abs(float(r[1]) - float(s["acceptance"])) <= 5.1e-4, r
            assert int(float(s["K"])) == 400
            rom = re.fullmatch(r"([0-9.]+) \[([0-9.]+)–([0-9.]+)\]", r[5])
            assert abs(float(rom.group(1)) - float(s["ratio_of_means"])) <= 5.1e-5, r
            romtex = ("$" + rom.group(1) + r"\ [" + rom.group(2) + r",\," + rom.group(3) + "]$")
            acc = cell(r[1])
        else:
            romtex = cell(r[5])
            acc = "--"
        assert abs(float(r[6]) - float(s["cv"])) <= 5.1e-4, r
        assert abs(float(r[9]) - float(s["hom_factor"])) <= 5.1e-4, r
        rows.append({"p": p, "main": " & ".join([cell(r[0]), acc, pm_cell(r[2]), pm_cell(r[3]),
                                                 pm_cell(r[4]), romtex, cell(r[6])]) + r" \\",
                     "factors": (pm_cell(r[7]), pm_cell(r[8])), "theta": cell(r[9]), "ptex": cell(r[0]),
                     "raw": r})
    return rows


uni = sweep_rows("sweep_uniform.md", "uniform")
thin = sweep_rows("sweep_smart.md", "smart")
assert [u["p"] for u in uni] == [t["p"] for t in thin]
assert [u["theta"] for u in uni] == [t["theta"] for t in thin]
write_fragment("sweep_uniform", [u["main"] for u in uni])
write_fragment("sweep_thinned", [t["main"] for t in thin])
write_fragment("sweep_factors", [
    " & ".join([u["ptex"], u["factors"][0], u["factors"][1], t["factors"][0], t["factors"][1],
                u["theta"]]) + r" \\" for u, t in zip(uni, thin)])
clean_ratio = float(uni[0]["raw"][4])


def col(rows, k):
    return {r["p"]: float(r["raw"][k].split("±")[0].split("[")[0].replace(" ", "")) for r in rows}


checks["sweep_clean_ratio"] = clean_ratio
checks["sweep_uniform_mean_of_ratios_min_over_p>0"] = min(v for p, v in col(uni, 4).items() if p > 0)
vals = [col(thin, 4)[p] for p in sorted(col(thin, 4))]
checks["sweep_thinned_mean_of_ratios_by_p"] = vals
checks["sweep_thinned_mean_of_ratios_non_monotone_steps"] = [
    [sorted(col(thin, 4))[i], sorted(col(thin, 4))[i + 1]] for i in range(len(vals) - 1)
    if vals[i + 1] < vals[i]]
rom_u = col(uni, 5)
checks["sweep_uniform_ratio_of_means_below_clean_at_p"] = sorted(p for p, v in rom_u.items()
                                                               if p > 0 and v < clean_ratio)
checks["sweep_uniform_ratio_of_means_min"] = min(v for p, v in rom_u.items() if p > 0)
checks["sweep_uniform_mean_of_ratios_range_p<=0.14"] = [
    min(v for p, v in col(uni, 4).items() if 0 < p <= 0.14),
    max(v for p, v in col(uni, 4).items() if 0 < p <= 0.14)]
# per-placement ratios of the corner-thinned sample for p <= 0.2
pp = []
for f in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_defects', 'sweep', 'sweep_N35_smart_p0.*.csv'))):
    pval = float(os.path.basename(f).split("_p")[1][:-4])
    if pval <= 0.2 + 1e-9:
        with open(f) as fh:
            pp += [float(r["ratio"]) for r in csv.DictReader(fh)]
checks["sweep_thinned_per_placement_ratio_p<=0.2"] = {"n": len(pp), "min": min(pp), "max": max(pp)}
with open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v03_summary.csv')) as fh:
    v03 = list(csv.DictReader(fh))
zs = [abs(float(r[k])) for r in v03 for k in ("z_mfpt", "z_mode", "z_ratio")]
checks["sweep_sample_A_vs_sample_B"] = {"comparisons": len(zs), "max_abs_z": max(zs),
                                        "rows": len(v03), "K_sample_B": sorted({int(float(r["K"])) for r in v03})}
checks["sweep_acceptance_at_p=0.30_0.35"] = {
    "uniform": [uni[-2]["raw"][1], uni[-1]["raw"][1]], "corner_thinned": [thin[-2]["raw"][1], thin[-1]["raw"][1]]}

# ------------------------------------------------------------------------------------------
# 5.  deterministic geometries
# ------------------------------------------------------------------------------------------
hdr, body = md_table(read(_os.path.join(_R, 'data', 'msc_defects', 'tables', 'structured_deterministic.md')), None)
assert len(hdr) == 10 and len(body) == 29, (hdr, len(body))
with open(_os.path.join(_R, 'data', 'msc_defects', 'structured_deterministic.csv')) as fh:
    det = {r["name"]: r for r in csv.DictReader(fh)}
with open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_det.csv')) as fh:
    det2 = {r["name"]: r for r in csv.DictReader(fh)}
clean = det["clean"]
rows = []
maxrel_second = 0.0
for r in body:
    name = r[0].strip("`")
    j = det[name]
    mf = float(r[3].replace(" ", ""))
    mo = int(r[4].replace(" ", ""))
    assert int(r[1]) == int(j["n_blocked"]) and int(r[2]) == int(j["n_cluster"]), r
    assert abs(mf - float(j["mfpt"])) <= 0.051 and mo == int(float(j["mode"])), r
    assert abs(float(r[5]) - float(j["ratio"])) <= 5.1e-5 and abs(float(r[8]) - float(j["cv"])) <= 5.1e-4, r
    assert abs(float(r[6]) - float(j["mfpt"]) / float(clean["mfpt"])) <= 5.1e-4, r
    assert abs(float(r[7]) - float(j["mode"]) / float(clean["mode"])) <= 5.1e-4, r
    assert abs(float(r[9]) - float(j["median"]) / float(j["mfpt"])) <= 5.1e-4, r
    # second implementation, written separately: identical mode, MFPT to round-off
    k = det2[name]
    assert int(float(k["mode"])) == mo and int(k["blocked"]) == int(r[1]), (name,)
    maxrel_second = max(maxrel_second, abs(float(k["mfpt"]) / float(j["mfpt"]) - 1))
    texname = r"\texttt{" + name.replace("_", r"\_") + "}"
    rows.append(" & ".join([texname] + [cell(c) for c in r[1:]]) + r" \\")
write_fragment("deterministic", rows)
checks["deterministic_rows"] = len(body)
checks["deterministic_second_implementation_max_rel_diff_mfpt"] = maxrel_second
ratios = {r[0].strip("`"): float(r[5]) for r in body}
checks["deterministic_ratio_min"] = min(ratios.items(), key=lambda kv: kv[1])
checks["deterministic_ratio_max"] = max(ratios.items(), key=lambda kv: kv[1])
with open(_os.path.join(_R, 'data', 'msc_defects_verify', 'v06_periodic.csv')) as fh:
    per = list(csv.DictReader(fh))
for a in (2, 3, 4):
    sub = [r for r in per if int(r["a"]) == a]
    checks[f"periodic_a{a}_phases"] = len(sub)
    checks[f"periodic_a{a}_mfpt_factor_min_max"] = [min(float(r["mfpt_fac"]) for r in sub),
                                                    max(float(r["mfpt_fac"]) for r in sub)]
    checks[f"periodic_a{a}_mode_factor_min_max"] = [min(float(r["mode_fac"]) for r in sub),
                                                    max(float(r["mode_fac"]) for r in sub)]

# ------------------------------------------------------------------------------------------
# 6.  values quoted from Table M2
# ------------------------------------------------------------------------------------------
hdr, body = md_table(summary_md, "**Table M2.")
m2 = {int(r[0]): r for r in body}
checks["exact_ratio_2d_N4001_unit_rate"] = m2[4001][2]
checks["exact_ratio_2d_N16777216_unit_rate"] = m2[16777216][2]
checks["q_tmode_over_N2_2d_N16777216"] = m2[16777216][1]

checks["_generated_by"] = "code/article/appC_tables_build.py"
checks["_fragments"] = sorted(fragments)
with open(OUTJSON, "w", encoding="utf-8") as fh:
    json.dump(checks, fh, indent=1, ensure_ascii=False)
print("fragments:", ", ".join(sorted(fragments)))
print("checks   ->", os.path.relpath(OUTJSON, ROOT))

# ------------------------------------------------------------------------------------------
# 7.  assemble / check the section file
# ------------------------------------------------------------------------------------------
if "--assemble" in sys.argv:
    txt = read(SECTION)
    for name, frag in fragments.items():
        tag = f"%%FRAG:{name}%%\n"
        if tag in txt:
            txt = txt.replace(tag, frag)
            print("inserted", name)
    with open(SECTION, "w", encoding="utf-8") as fh:
        fh.write(txt)

# fragments that are written for the repository but are not printed in the article (the density-sweep tables of
# sample A alone; the pooled table of both samples is in Supplementary Section S9)
REPOSITORY_ONLY = {"sweep_uniform", "sweep_thinned", "sweep_factors"}
if "--check" in sys.argv:
    txt = read(SECTION)
    missing = [name for name, frag in fragments.items() if name not in REPOSITORY_ONLY and frag not in txt]
    left = re.findall(r"%%FRAG:(\w+)%%", txt)
    if missing or left:
        print("NOT IN SECTION FILE:", missing, "UNFILLED PLACEHOLDERS:", left)
        sys.exit(1)
    print(f"all {len(fragments) - len(REPOSITORY_ONLY)} printed fragments occur verbatim in sections/appC_tables.tex "
          f"({len(REPOSITORY_ONLY)} further fragments are kept for the repository only)")
