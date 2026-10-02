"""summarise_ext.py -- collect the certificates of the C engine (certs/c128_*.jsonl) into certs/SUMMARY_EXT.json,
numbers_ext.tex and the tables tab_ext_*.tex / text_ext_q1.tex used by sec_ext*.tex.  Pure bookkeeping: every
number is read from the certificate files; the statements are re-checked by the separately written audit_ext.py."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, math, os, re, sys
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_rigorous_certified'))
DN = {1: "One", 2: "Two", 3: "Three"}
QN = {"4/5": "F", "1/2": "H", "1/1": "U"}
QTEX = {"4/5": "4/5", "1/2": "1/2", "1/1": "1"}
KEYS = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T", "undecided_adjacent_pairs")
EXC = {(1, "4/5", 4), (1, "1/2", 3), (1, "1/2", 4)}


def load(name):
    p = os.path.join(CERT, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ranges(ns, sep="--"):
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}{sep}{ns[j]}")
        i = j + 1
    return ", ".join(out)


def up(x, nd):
    f = Fraction(x)
    return f"{float(Fraction(-((-f.numerator * 10 ** nd) // f.denominator), 10 ** nd)):.{nd}f}"


def down(x, nd):
    f = Fraction(x)
    return f"{float(Fraction((f.numerator * 10 ** nd) // f.denominator, 10 ** nd)):.{nd}f}"


def sci_floor(fr):
    e = 0
    while fr * 10 ** e < 100:
        e += 1
    m = (fr * 10 ** e).__floor__()
    return f"{m / 100:.2f}\\times10^{{{2 - e}}}"


def intersect(a, b):
    return all(not (int(x) > int(b["F_enclosures"][t][1]) or int(y) < int(b["F_enclosures"][t][0]))
               for t, (x, y) in a["F_enclosures"].items() if t in b["F_enclosures"])


S = {"modes": {}}
macros = {}
regress_bad, regress_n = 0, 0
for d in (1, 2, 3):
    for q in ("4/5", "1/2", "1/1"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        rows = {c["N"]: c for c in load(f"c128_{tag}.jsonl")}
        if not rows:
            continue
        gen = {c["N"]: c for c in load(f"modes_{tag}.jsonl")}
        nmax = 1
        while nmax + 1 in rows:
            nmax += 1
        extra = sorted(N for N in rows if N > nmax)
        same = sorted(N for N in gen if N in rows and all(rows[N][k] == gen[N][k] for k in KEYS) and intersect(gen[N], rows[N]))
        bad = sorted(N for N in gen if N in rows and N not in same)
        missing = sorted(N for N in gen if N not in rows)
        regress_bad += len(bad) + len(missing)
        regress_n += len(same)
        new = sorted(set(rows) - set(gen))
        xl = {c["N"]: c for c in load(f"xcheck_limb_{tag}.jsonl")}
        xp = {c["N"]: c for c in load(f"xcheck_packed_{tag}.jsonl")}
        xe = {c["N"]: c for c in load(f"xcheck_exact_{tag}.jsonl")}
        xl_ok = sorted(N for N, c in xl.items() if c["same_verdicts"] and c["enclosures_intersect"])
        xp_ok = sorted(N for N, c in xp.items() if c["same_verdicts"] and c["enclosures_intersect"])
        x_bad = sorted(set(xl) - set(xl_ok)) + sorted(set(xp) - set(xp_ok))
        widths, gapratio = [], []
        for N, c in rows.items():
            ts = str(c["mode"])
            w = int(c["F_enclosures"][ts][1]) - int(c["F_enclosures"][ts][0])
            widths.append(w)
            if c["mode_certified"]:
                gaps = [int(c[k]) for k in ("gap_left", "gap_right") if k in c]
                gapratio.append((Fraction(min(gaps), max(w, 1)), N))
        non_cert = sorted(N for N, c in rows.items() if not c["mode_certified"])
        non_uni = sorted(N for N, c in rows.items() if not c["unimodal_certified"])
        multi = sorted(N for N, c in rows.items() if c["certified_local_maxima_upto_T"] >= 2)
        tail_ne = sorted(N for N, c in rows.items() if c["T_tail"] != c["mode"])
        gam = {N: Fraction(c["one_minus_gamma_lower"]) for N, c in rows.items()}
        s = dict(d=d, q=q, N_max=nmax, extra_N=extra, count=len(rows), new_N=ranges(new), new_count=len(new),
                 tierA_identical_count=len(same), tierA_mismatch_N=bad, tierA_missing_N=missing,
                 not_certified_N=non_cert, non_unimodal_N=non_uni, multimodal_N=multi,
                 undecided_N=sorted(set(non_uni) - set(multi)), tail_ne_mode_N=tail_ne,
                 max_tail_over_mode=max(c["T_tail"] / c["mode"] for c in rows.values()),
                 t0_is_d_times_N_minus_1=all(c["t0"] == d * (N - 1) for N, c in rows.items()),
                 max_local_maxima=max(c["certified_local_maxima_upto_T"] for c in rows.values()),
                 max_T=max(c["T_tail"] for c in rows.values()), max_tsw=max(c["ts"] for c in rows.values()),
                 max_stored_sites=max(c["stored_sites"] for c in rows.values()),
                 xcheck_limb_N=xl_ok, xcheck_packed_N=xp_ok, xcheck_mismatch_N=x_bad,
                 xcheck_exact={str(N): c for N, c in sorted(xe.items())},
                 xcheck_seconds=dict(limb=round(sum(c["seconds"] for c in xl.values())), packed=round(sum(c["seconds"] for c in xp.values()))),
                 max_xcheck_seconds=max([c["seconds"] for c in list(xl.values()) + list(xp.values())], default=0),
                 min_gap_over_enclosure_width=float(min(gapratio)[0]), min_gap_at_N=min(gapratio)[1],
                 max_enclosure_width_at_mode=max(widths),
                 min_one_minus_gamma=str(min(gam.values())), min_one_minus_gamma_at_N=min(gam, key=gam.get),
                 seconds=round(sum(c.get("seconds", 0) for c in rows.values())),
                 max_seconds=max(c.get("seconds", 0) for c in rows.values()), max_calls=max(c.get("fp128_calls", 1) for c in rows.values()),
                 modes={str(N): c["mode"] for N, c in sorted(rows.items())})
        S["modes"][tag] = s
        key = f"{DN[d]}{QN[q]}"
        macros[f"XNmax{key}"] = str(nmax)
        macros[f"XExtra{key}"] = ",\\ ".join(str(x) for x in extra) if extra else "none"
        macros[f"XGam{key}"] = sci_floor(min(gam.values()))
        macros[f"XAlso{key}"] = (" and $N\\in\\{" + ",\\ ".join(str(x) for x in extra) + "\\}$") if extra else ""   # text-mode phrase
allm = list(S["modes"].values())
macros["XTotalCerts"] = str(sum(v["count"] for v in allm))
macros["XTotalNew"] = str(sum(v["new_count"] for v in allm))
macros["XRegressBad"] = str(regress_bad)
macros["XRegressN"] = str(regress_n)
macros["XXLimb"] = str(sum(len(v["xcheck_limb_N"]) for v in allm))
macros["XXPacked"] = str(sum(len(v["xcheck_packed_N"]) for v in allm))
macros["XXBad"] = str(sum(len(v["xcheck_mismatch_N"]) for v in allm))
macros["XMaxWidth"] = str(max(v["max_enclosure_width_at_mode"] for v in allm))
macros["XMinMargin"] = "$10^{%d}$" % int(math.floor(math.log10(min(v["min_gap_over_enclosure_width"] for v in allm))))
macros["XMaxTsw"] = str(max(v["max_tsw"] for v in allm))
macros["XMaxT"] = str(max(v["max_T"] for v in allm))
macros["XMaxSites"] = str(max(v["max_stored_sites"] for v in allm))
macros["XCpu"] = str(sum(v["seconds"] for v in allm))
macros["XMaxSeconds"] = str(int(math.ceil(max(v["max_seconds"] for v in allm))))
macros["XMaxCalls"] = str(max(v["max_calls"] for v in allm))
macros["XMaxXSeconds"] = str(int(math.ceil(max(v["max_xcheck_seconds"] for v in allm))))
# ---------------- second run of every certificate (reproducibility, conservation of probability)
rc_n, rc_bad, rc_slack, rc_missing = 0, 0, 0.0, 0
for tag, sm in S["modes"].items():
    rc = {c["N"]: c for c in load(f"c128_recheck_{tag}.jsonl")}
    Ns = {int(N) for N in sm["modes"]}
    rc_missing += len(Ns - set(rc))
    for N in Ns & set(rc):
        rc_n += 1
        rc_bad += not (rc[N]["identical_to_first_run"] and rc[N]["conservation_ok"])
        rc_slack = max(rc_slack, rc[N]["relative_slack"])
    sm["recheck"] = dict(count=len(Ns & set(rc)), missing=sorted(Ns - set(rc)),
                         all_identical=all(rc[N]["identical_to_first_run"] for N in Ns & set(rc)),
                         all_conservation_ok=all(rc[N]["conservation_ok"] for N in Ns & set(rc)),
                         max_relative_slack=max([rc[N]["relative_slack"] for N in Ns & set(rc)], default=None),
                         seconds=round(sum(rc[N]["seconds"] for N in Ns & set(rc))))
macros["XRecheckN"] = str(rc_n)
macros["XRecheckBad"] = str(rc_bad)
macros["XRecheckMissing"] = str(rc_missing)
if rc_n:
    e = int(math.floor(math.log10(rc_slack)))
    macros["XMaxSlack"] = f"{math.ceil(rc_slack / 10 ** e * 10 + 1e-9) / 10:.1f}\\times10^{{{e}}}"
macros["XRecheckCpu"] = str(sum(v["recheck"]["seconds"] for v in S["modes"].values()))

for name, key in (("c128_refcheck.json", "XRefSmall"), ("c128_refcheck_medium.json", "XRefMedium")):
    p = os.path.join(CERT, name)
    if os.path.exists(p):
        j = json.load(open(p))
        S[name] = dict(cases=j["cases"], mismatches=j["mismatches"])
        macros[key] = str(j["cases"])
        macros[key + "Bad"] = str(j["mismatches"])

# ---------------- comparison with the floating-point study (data/msc_modes)
float_study_path = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')
agree, disagree, uncovered = [], [], []
if os.path.exists(float_study_path):
    for line in open(float_study_path):
        r = json.loads(line)
        qs = {0.8: "4/5", 0.5: "1/2", 1.0: "1/1"}.get(r.get("q"))
        if r.get("geo") != "CC" or qs is None:
            continue
        m = S["modes"].get(f"d{r['d']}_q{qs.replace('/', '-')}", {}).get("modes", {}).get(str(r["N"]))
        if m is not None:
            (agree if m == r["mode"] else disagree).append((r["d"], qs, r["N"], r["mode"], m))
        else:
            uncovered.append((r["d"], qs, r["N"]))
S["comparison_with_float_study_modes"] = dict(agree=len(set(agree)), agree_list=[list(x) for x in sorted(set(agree))], disagree=[list(x) for x in disagree],
                                             float_study_cases_not_covered=[list(x) for x in sorted(set(uncovered))])
macros["XFloatStudyUncovered"] = str(len(set(uncovered)))
macros["XFloatStudyAgree"] = str(len(set(agree)))
macros["XFloatStudyDisagree"] = str(len(disagree))

cfp = _os.path.join(_R, 'data', 'msc_rigorous_certified', 'closedform_summary_c128.json')
cf = json.load(open(cfp)) if os.path.exists(cfp) else {}
S["closed_form"] = cf
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))

# ---------------- 1D law
kap = (Fraction(const["kappa_1 = 2 tau_star"]["lo_dec"]), Fraction(const["kappa_1 = 2 tau_star"]["hi_dec"]))
S["law_1d"] = {}
for q in ("4/5", "1/2"):
    tag = f"d1_q{q.replace('/', '-')}"
    if tag not in S["modes"]:
        continue
    nmax = S["modes"][tag]["N_max"]
    rows = [r for r in load(f"closedform_c128_{tag}.jsonl") if "delta_1D_lo" in r]
    qq = Fraction(q)
    sel = [r for r in rows if 10 <= r["N"] <= nmax]
    assert [r["N"] for r in sel] == list(range(10, nmax + 1))
    dl = min(Fraction(r["delta_1D_lo"]) for r in sel)
    dh = max(Fraction(r["delta_1D_hi"]) for r in sel)
    nd = 3                                    # outward rounding to nd decimals, nd as small as possible with length < 1
    while True:
        hi_r = Fraction(math.floor(dh * 10 ** nd) + 1, 10 ** nd)
        lo_r = Fraction(math.ceil(dl * 10 ** nd) - 1, 10 ** nd)
        if hi_r - lo_r < 1 or nd >= 9:
            break
        nd += 1
    floor_ok = all((kap[0] * Fraction(2 * r["N"] - 1, 2) ** 2 / qq + hi_r).__floor__() == r["mode"]
                   == (kap[1] * Fraction(2 * r["N"] - 1, 2) ** 2 / qq + hi_r).__floor__() for r in sel)
    rr = []
    for r in sel:
        N = r["N"]
        ratio = Fraction(r["mode"]) * qq / (N * (N - 1))
        rr.append(max(abs(ratio - kap[0]), abs(ratio - kap[1])) * N * (N - 1))
    ext = {r["N"]: [r["delta_1D_lo"], r["delta_1D_hi"]] for r in rows if r["N"] > nmax}
    ext_ok = all(lo_r < Fraction(a) and Fraction(b) < hi_r for a, b in ext.values())
    S["law_1d"][tag] = dict(N_min=10, N_max=nmax, delta_lo_rounded=str(lo_r), delta_hi_rounded=str(hi_r),
                            interval_length_lt_1=bool(hi_r - lo_r < 1), floor_formula_verified=bool(floor_ok),
                            delta_min=float(dl), delta_max=float(dh), max_NNm1_times_abs_ratio_minus_kappa=float(max(rr)),
                            extra_N=ext, extra_N_within_bounds=ext_ok)
    macros[f"XDeltaLo{QN[q]}"] = f"{float(lo_r):.{nd}f}"
    macros[f"XDeltaHi{QN[q]}"] = f"{float(hi_r):.{nd}f}"
    macros[f"XRatioConst{QN[q]}"] = f"{math.ceil(max(rr) * 100 + 1e-9) / 100:.2f}"

# ---------------- q-scaling
S["q_scaling"] = {}
for d in (1, 2, 3):
    a, b = S["modes"].get(f"d{d}_q4-5"), S["modes"].get(f"d{d}_q1-2")
    if not a or not b:
        continue
    top = min(a["N_max"], b["N_max"])
    diff = {N: Fraction(4, 5) * a["modes"][str(N)] - Fraction(1, 2) * b["modes"][str(N)] for N in range(2, top + 1)}
    lo, hi = min(diff.values()), max(diff.values())
    S["q_scaling"][f"d{d}"] = dict(N_max=top, min=str(lo), max=str(hi), argmin=min(diff, key=diff.get), argmax=max(diff, key=diff.get))
    macros[f"XQsLo{DN[d]}"] = f"{float(lo):.1f}"
    macros[f"XQsHi{DN[d]}"] = f"{float(hi):.1f}"
    macros[f"XQsN{DN[d]}"] = str(top)

# ---------------- closed-form macros
for tag, s in cf.items():
    d, q = s["d"], s["q"]
    key = f"eps[10,{s['N_max']}]"
    if key in s:
        macros[f"XEps{DN[d]}{QN[q]}"] = up(s[key]["max_abs_rel_err_upper"], 4 if d == 1 else 5)
        macros[f"XEpsAt{DN[d]}{QN[q]}"] = str(s[key]["at_N"])
        lo_all = min([Fraction(s[key]["min_abs_rel_err_lower"])] + ([Fraction(v[0]) for v in s.get("extra", {}).values()] if d == 1 else []))
        macros[f"XEpsLo{DN[d]}{QN[q]}"] = down(str(lo_all), 4 if d == 1 else 6)       # d = 1: lower bound valid for the isolated N too
        macros[f"XEpsN{DN[d]}{QN[q]}"] = str(s["N_max"])


def w(name, text, src):
    """write a generated LaTeX fragment; src = the files its numbers are read from (paths relative to ROOT)."""
    open(os.path.join(ROOT, name), "w").write("% generated by scripts/c128/summarise_ext.py -- do not edit\n% src: " + src + "\n" + text)


def nlist(ns, cap=12):
    ns = sorted(ns)
    return ranges(ns) if ns else "--"


lines = [r"\begin{tabular}{cc|p{3.2cm}|r|r|p{3.1cm}|p{2.5cm}|r}",
         r"$d$ & $q$ & certified $N$ (C engine) & number & new & re-derived by \texttt{fplimb} & by \texttt{verify\_packed} & time (s) \\ \hline"]
for tag, s in S["modes"].items():
    rng = f"$2$--${s['N_max']}$" + (f", ${', '.join(str(x) for x in s['extra_N'])}$" if s["extra_N"] else "")
    lines.append(f"{s['d']} & ${QTEX[s['q']]}$ & {rng} & {s['count']} & {s['new_count']} & {nlist(s['xcheck_limb_N'])} & "
                 f"{nlist(s['xcheck_packed_N'])} & {s['seconds']} \\\\")
lines.append(r"\end{tabular}")
w("tab_ext_ranges.tex", "\n".join(lines) + "\n", "certs/SUMMARY_EXT.json, key modes (from certs/c128_*.jsonl, certs/xcheck_limb_*.jsonl, "
  "certs/xcheck_packed_*.jsonl; times: sums of the field seconds, wall-clock)")

sel = {1: [600, 700, 800, 900, 1000, 1200, 1500, 2000], 2: [160, 180, 200, 220, 250, 280, 300, 350, 401, 450, 500, 600],
       3: [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]}
lines = [r"\begin{tabular}{c|r|r|r|r|r|r|r|r}",
         r"$d$ & $N$ & $t^*$ ($q=4/5$) & $\tau_N=q\,\mathbb E T$ & $t^*/\mathbb E T$ & $t_{\rm cf}$ & $(t_{\rm cf}-t^*)/t^*$ & $t^*$ ($q=1/2$) & $t^*$ ($q=1$) \\ \hline"]
for d in (1, 2, 3):
    cfr = {r["N"]: r for r in load(f"closedform_c128_d{d}_q4-5.jsonl")}
    mh = S["modes"].get(f"d{d}_q1-2", {}).get("modes", {})
    mu = S["modes"].get(f"d{d}_q1-1", {}).get("modes", {})
    for N in sel[d]:
        if N not in cfr:
            continue
        r = cfr[N]
        tau = float(Fraction(r["mfpt_lo"]) * Fraction(4, 5))
        lines.append(f"{d} & {N} & {r['mode']} & {tau:.4f} & {float(Fraction(r['mode_over_mfpt_lo'])):.6f} & {float(Fraction(r['t_cf_lo'])):.3f} & "
                     f"${float(Fraction(r['rel_err_lo'])):+.5f}$ & {mh.get(str(N), '--')} & {mu.get(str(N), '--')} \\\\")
    if d < 3:
        lines.append(r"\hline")
lines.append(r"\end{tabular}")
w("tab_ext_modes.tex", "\n".join(lines) + "\n", "certs/closedform_c128_d*_q4-5.jsonl (t*, tau_N, t_cf), certs/c128_d*_q1-2.jsonl, certs/c128_d*_q1-1.jsonl")


def signed(lo, hi, nd=6):
    a = down(lo, nd) if not lo.startswith("-") else "-" + up(lo[1:], nd)
    b = up(hi, nd) if not hi.startswith("-") else "-" + down(hi[1:], nd)
    return f"[{a},\\ {b}]"


lines = [r"\begin{tabular}{cc|c|c|c|c}",
         r"$d$ & $q$ & range of $N$ & bound on $|t_{\rm cf}-t^*|/t^*$ & worst $N$ & interval containing $(t_{\rm cf}-t^*)/t^*$ \\ \hline"]
for tag, s in cf.items():
    if s["q"] == "1/1":
        continue
    for key in [k for k in s if k.startswith("eps[")]:
        a, b = key[4:-1].split(",")
        if a == "2" or int(a) >= int(b):
            continue
        e = s[key]
        lines.append(f"{s['d']} & ${QTEX[s['q']]}$ & ${a}\\le N\\le {b}$ & ${up(e['max_abs_rel_err_upper'], 6)}$ & {e['at_N']} & "
                     f"${signed(*e['signed_range'])}$ \\\\")
    for N, (lo, hi) in s.get("extra", {}).items():
        m = max(abs(Fraction(lo)), abs(Fraction(hi)))
        lines.append(f"{s['d']} & ${QTEX[s['q']]}$ & $N={N}$ & ${up(str(m), 6)}$ & {N} & ${signed(lo, hi)}$ \\\\")
lines.append(r"\end{tabular}")
w("tab_ext_cf.tex", "\n".join(lines) + "\n", "certs/closedform_summary_c128.json (from certs/closedform_c128_d*_q*.jsonl, written by scripts/closed_form.py c128)")

# ---------------- appendix: the new modes for d = 2, 3
for d in (2, 3):
    for q in ("4/5", "1/2"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        if tag not in S["modes"]:
            continue
        gen = {c["N"] for c in load(f"modes_{tag}.jsonl")}
        items = [(int(N), t) for N, t in S["modes"][tag]["modes"].items() if int(N) not in gen]
        per = 7 if max(t for _, t in items) >= 100000 else 8
        lines = [r"\begin{tabular}{" + "|".join(["rr"] * per) + "}", " & ".join([r"$N$ & $t^*$"] * per) + r" \\ \hline"]
        nrow = -(-len(items) // per)
        for r in range(nrow):
            cells = []
            for c in range(per):
                i = c * nrow + r
                cells.append(f"{items[i][0]} & {items[i][1]}" if i < len(items) else " & ")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\end{tabular}")
        w(f"tab_ext_all_{tag}.tex", "\n".join(lines) + "\n", f"field mode of certs/c128_{tag}.jsonl")

# ---------------- text of the q = 1 theorem (extended)
import verify_exact as ve


def tie_description(d, N):
    f, T = ve.exact_pmf(d, N, Fraction(1))
    eq = [t for t in range(1, T + 1) if f[t] == f[t + 1] and f[t] > 0]
    core = [("<" if f[t] < f[t + 1] else ">") for t in range(1, T + 1) if f[t] != f[t + 1]]
    weak = "".join(core) == "<" * core.count("<") + ">" * core.count(">")
    return eq, weak


txt = [r"\begin{itemize}"]
S["q1"] = {}
for d in (1, 2, 3):
    sq = S["modes"].get(f"d{d}_q1-1")
    if not sq:
        continue
    genx = {c["N"]: c for c in load(f"packedexact_d{d}_q1-1.jsonl")}
    parts = []
    allN = sorted(int(N) for N in sq["modes"])
    ties = {}
    for N in sq["not_certified_N"]:
        if N in genx and len(genx[N]["maximisers_of_lower_bound"]) > 1:
            ties[N] = genx[N]["maximisers_of_lower_bound"]
        elif str(N) in sq["xcheck_exact"]:
            ties[N] = sq["xcheck_exact"][str(N)]["maximisers_of_lower_bound"]
    if sq["not_certified_N"]:
        covered = all(N in ties and len(ties[N]) == 2 for N in sq["not_certified_N"])
        desc = "; ".join(f"$N={N}$: $t\\in\\{{{', '.join(str(t) for t in ties[N])}\\}}$" for N in sq["not_certified_N"] if N in ties)
        parts.append(f"the maximiser is unique except for $N\\in\\{{{', '.join(str(N) for N in sq['not_certified_N'])}\\}}$, "
                     f"where the maximum is attained at exactly two times ({desc}; exact arithmetic, Theorem~\\ref{{thm:modesU}})"
                     + ("" if covered else " [SOME UNDECIDED]"))
    else:
        parts.append("the maximiser is unique for every $N$ of the range")
    uni = sorted(set(allN) - set(sq["non_unimodal_N"]))
    parts.append(f"$f$ is strictly unimodal for $N\\in{{}}$\\{{{ranges(uni)}\\}}" if uni else "$f$ is strictly unimodal for no $N$ of the range")
    if sq["multimodal_N"]:
        parts.append(f"$f$ has at least two strict local maxima, hence is \\emph{{not}} unimodal, for $N\\in{{}}$\\{{{ranges(sq['multimodal_N'])}\\}} "
                     f"(the largest certified number of strict local maxima is {sq['max_local_maxima']})")
    und_desc = {}
    for N in sq["undecided_N"]:
        if N > 12:
            parts.append(f"for $N={N}$ strict unimodality is UNDECIDED")
            und_desc[N] = "undecided"
            continue
        eq, weak = tie_description(d, N)
        und_desc[N] = dict(equal_consecutive_values_at_t=eq, weakly_unimodal=weak)
        parts.append(f"for $N={N}$, $f$ is {'unimodal in the weak sense but not strictly' if weak else 'NOT weakly unimodal'}: "
                     + ", ".join(f"$f({t})=f({t + 1})$" for t in eq))
    # equalities of consecutive values: every relation left undecided by the C certificate must be settled exactly
    rows_q1 = {c["N"]: c for c in load(f"c128_d{d}_q1-1.jsonl")}
    eqs = {}
    for N, c in sorted(rows_q1.items()):
        if c["undecided_adjacent_pairs"]:
            if N in genx:
                eqs[N] = genx[N]["equal_adjacent_pairs_from_t0"]
            elif N <= 12:
                eqs[N] = tie_description(d, N)[0]
            else:
                eqs[N] = None
    if any(v is None or not v for v in eqs.values()):
        parts.append("SOME RELATION UNDECIDED")
    elif eqs:
        lst = " and ".join(f"$N={N}$ (" + ", ".join(f"$f({t})=f({t + 1})$" for t in v) + ")" for N, v in eqs.items())
        parts.append(f"equal consecutive non-zero values occur only for {lst} (exact arithmetic, Theorem~\\ref{{thm:modesU}}): "
                     f"for every other pair $(N,t)$ with $N$ in the range and $t\\ge d(N-1)-1$, $f(t)\\ne f(t+1)$")
    else:
        parts.append("$f(t)\\ne f(t+1)$ for every $N$ of the range and every $t\\ge d(N-1)-1$")
    if sq["tail_ne_mode_N"]:
        parts.append(f"the tail time $T$ exceeds $t^*$ for {len(sq['tail_ne_mode_N'])} of the {sq['count']} values of $N$ "
                     f"(largest ratio $T/t^*={sq['max_tail_over_mode']:.2f}$)")
    S["q1"][f"d{d}"] = dict(strictly_unimodal=ranges(uni), multimodal=ranges(sq["multimodal_N"]), undecided=und_desc,
                            two_maximisers={str(k): v for k, v in ties.items()}, adjacent_equalities={str(N): v for N, v in eqs.items()})
    txt.append(f"\\item[$d={d}$:] " + "; ".join(parts) + ".")
txt.append(r"\end{itemize}")
w("text_ext_q1.tex", "\n".join(txt) + "\n", "certs/c128_d*_q1-1.jsonl, certs/packedexact_d*_q1-1.jsonl, certs/xcheck_packed_d2_q1-1.jsonl; "
  "exact ties re-derived by scripts/verify_exact.py")

# ---------------- Conjecture conj:last on the certified ranges (q in {4/5, 1/2})
n_all = n_nonexc = n_small = n_large = 0
for d in (1, 2, 3):
    for q in ("4/5", "1/2"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        px = {c["N"]: c for c in load(f"packedexact_{tag}.jsonl")}
        for c in load(f"c128_{tag}.jsonl"):
            n_all += 1
            if (d, q, c["N"]) in EXC:
                mx = px[c["N"]]["maximisers_of_lower_bound"]
                assert px[c["N"]]["T_tail"] == c["T_tail"]
                n_small += c["T_tail"] == min(mx)
                n_large += c["T_tail"] == max(mx)
            else:
                n_nonexc += 1
                n_small += c["T_tail"] == c["mode"] and c["mode_certified"]
                n_large += c["T_tail"] == c["mode"] and c["mode_certified"]
S["conjecture_last"] = dict(certificates=n_all, non_exceptional=n_nonexc, T_equals_smallest_maximiser=n_small, T_equals_largest_maximiser=n_large)
macros["XLastCerts"] = str(n_all)
macros["XLastNonExc"] = str(n_nonexc)
macros["XLastSmall"] = str(n_small)
macros["XLastLarge"] = str(n_large)

json.dump(S, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'SUMMARY_EXT.json'), "w"), indent=1)
used = set()
for f in os.listdir(ROOT):
    if f.endswith(".tex") and (f.startswith("sec_ext") or f == "R5_certified_computations.tex"):
        used |= set(re.findall(r"\\(X[A-Z][A-Za-z]+)(?![A-Za-z])", open(os.path.join(ROOT, f)).read()))
missing = sorted(used - set(macros))
for k in missing:
    macros[k] = "??"
if missing:
    print("WARNING: macros without data:", missing)
def macro_src(k):
    if k.startswith("XEps"):
        return "certs/closedform_summary_c128.json (from certs/closedform_c128_d*_q*.jsonl, written by scripts/closed_form.py c128)"
    if k.startswith(("XDelta", "XRatioConst")):
        return "certs/SUMMARY_EXT.json, key law_1d (from certs/closedform_c128_d1_q*.jsonl and certs/constants.json)"
    if k.startswith("XQs"):
        return "certs/SUMMARY_EXT.json, key q_scaling (field mode of certs/c128_d*_q4-5.jsonl and certs/c128_d*_q1-2.jsonl)"
    if k.startswith("XFloatStudy"):
        return "certs/SUMMARY_EXT.json, key comparison_with_float_study_modes (floating-point study: data/msc_modes/discrete_modes.jsonl)"
    if k.startswith("XRef"):
        return "certs/c128_refcheck.json, certs/c128_refcheck_medium.json"
    if k.startswith(("XRecheck", "XMaxSlack")):
        return "certs/SUMMARY_EXT.json, key modes.*.recheck (from certs/c128_recheck_*.jsonl)"
    if k.startswith("XLast"):
        return "certs/SUMMARY_EXT.json, key conjecture_last (from certs/c128_d*_q4-5.jsonl, certs/c128_d*_q1-2.jsonl, certs/packedexact_d1_q*.jsonl)"
    return "certs/SUMMARY_EXT.json, key modes (from certs/c128_*.jsonl, certs/xcheck_*.jsonl; times: field seconds, wall-clock)"


with open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'numbers_ext.tex'), "w") as fh:
    fh.write("% generated by scripts/c128/summarise_ext.py -- do not edit\n")
    last = None
    for k in sorted(macros, key=lambda k: (macro_src(k), k)):
        if macro_src(k) != last:
            last = macro_src(k)
            fh.write(f"% src: {last}\n")
        fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("modes",)} for k, v in S["modes"].items()}, indent=1))
print(json.dumps(S["law_1d"], indent=1))
print(json.dumps(S["q_scaling"], indent=1))
print(json.dumps(macros, indent=1))
