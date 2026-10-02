"""summarise.py -- collect all certificates into certs/SUMMARY.json, numbers.tex and the data tables
(tab_*.tex, text_*.tex) used by note R5.  Pure bookkeeping: every number is read from the certificate
files; the statements of the theorems are re-checked by the separately written scripts/audit.py."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, json, math, os, re
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
DN = {1: "One", 2: "Two", 3: "Three"}
QN = {"4/5": "F", "1/2": "H", "1/1": "U"}
QTEX = {"4/5": "4/5", "1/2": "1/2", "1/1": "1"}
VERDICT = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T")


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


def strip(c):
    return {k: v for k, v in c.items() if k not in ("seconds", "engine")}


def up(x, nd):          # round a decimal string up / down to nd decimals
    f = Fraction(x)
    return f"{float(Fraction(-((-f.numerator * 10 ** nd) // f.denominator), 10 ** nd)):.{nd}f}"


def down(x, nd):
    f = Fraction(x)
    return f"{float(Fraction((f.numerator * 10 ** nd) // f.denominator, 10 ** nd)):.{nd}f}"


def sci_floor(fr):      # 'm.mm \times 10^{e}' rounded down (for positive rationals)
    e = 0
    while fr * 10 ** e < 100:
        e += 1
    m = (fr * 10 ** e).__floor__()
    return f"{m / 100:.2f}\\times10^{{{2 - e}}}"


S = {"modes": {}}
macros = {}
for d in (1, 2, 3):
    for q in ("4/5", "1/2", "1/1"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        rows = {c["N"]: c for c in load(f"modes_{tag}.jsonl")}
        if not rows:
            continue
        nmax = 1
        while nmax + 1 in rows:
            nmax += 1
        extra = sorted(N for N in rows if N > nmax)
        # --- separately written check (Python integers, packed)
        pk = {c["N"]: c for c in load(f"packed_{tag}.jsonl")}
        pk_ok = sorted(N for N in pk if N in rows and pk[N].get("agrees_with_generator") and pk[N].get("same_verdicts_as_generator"))
        pk_bad = sorted(N for N in pk if N in rows and not (pk[N].get("agrees_with_generator") and pk[N].get("same_verdicts_as_generator")))
        # --- exact arithmetic (packed --exact, and the Fraction program)
        px = {c["N"]: c for c in load(f"packedexact_{tag}.jsonl")}
        px_ok = sorted(N for N in px if N in rows and px[N].get("agrees_with_generator"))
        px_bad = sorted(N for N in px if N in rows and not px[N].get("agrees_with_generator"))
        ex = {r["N"]: r for r in load(f"exactcheck_{tag}.jsonl")}
        ex_ok = sorted(N for N in ex if ex[N]["agrees_with_certificate"])
        ex_bad = sorted(N for N in ex if not ex[N]["agrees_with_certificate"])
        # the two exact programs must agree with each other
        ex_px_conflict = sorted(N for N in ex if N in px and not (
            ex[N]["argmax"] == px[N]["maximisers_of_lower_bound"] and ex[N]["T_tail"] == px[N]["T_tail"]
            and ex[N]["strictly_unimodal"] == px[N]["unimodal_certified"]
            and ex[N]["strict_local_maxima_upto_T"] == px[N]["certified_local_maxima_upto_T"]))
        # --- reference engine (Python integers in numpy object arrays): identical JSON line
        ref = {c["N"]: c for c in load(f"refcheck_{tag}.jsonl")}
        ref_same = sorted(N for N in ref if N in rows and strip(ref[N]) == strip(rows[N]))
        ref_diff = sorted(N for N in ref if N in rows and strip(ref[N]) != strip(rows[N]))
        # --- exact information on ties
        ties_top = {}          # N -> list of maximisers (exact), when not unique
        flat = {}              # N -> times t with f(t) = f(t+1) > 0 (exact)
        for N, c in px.items():
            if len(c["maximisers_of_lower_bound"]) > 1:
                ties_top[N] = c["maximisers_of_lower_bound"]
            if c["equal_adjacent_pairs_from_t0"]:
                flat[N] = c["equal_adjacent_pairs_from_t0"]
        for N, r in ex.items():
            if not r["mode_unique"]:
                assert ties_top.get(N, r["argmax"]) == r["argmax"]
                ties_top[N] = r["argmax"]
        widths, gapratio, fvals = [], [], []
        for N, c in rows.items():
            ts = str(c["mode"])
            w = int(c["F_enclosures"][ts][1]) - int(c["F_enclosures"][ts][0])
            widths.append(w)
            fvals.append(int(c["F_enclosures"][ts][0]))
            if c["mode_certified"]:
                gaps = [int(c[k]) for k in ("gap_left", "gap_right") if k in c]
                gapratio.append((Fraction(min(gaps), max(w, 1)), N))
        non_cert = sorted(N for N, c in rows.items() if not c["mode_certified"])
        non_uni = sorted(N for N, c in rows.items() if not c["unimodal_certified"])
        multi = sorted(N for N, c in rows.items() if c["certified_local_maxima_upto_T"] >= 2)
        tail_ne = sorted(N for N, c in rows.items() if c["T_tail"] != c["mode"])
        gam = [Fraction(pk[N]["one_minus_gamma_lower"]) for N in pk_ok]
        s = dict(d=d, q=q, N_max=nmax, extra_N=extra, count=len(rows),
                 not_certified_N=non_cert, non_unimodal_N=non_uni, multimodal_N=multi,
                 undecided_N=sorted(set(non_uni) - set(multi)), tail_ne_mode_N=tail_ne,
                 max_tail_over_mode=max(c["T_tail"] / c["mode"] for c in rows.values()),
                 t0_is_d_times_N_minus_1=all(c["t0"] == d * (N - 1) for N, c in rows.items()),
                 max_local_maxima=max(c["certified_local_maxima_upto_T"] for c in rows.values()),
                 packed_verified_N=ranges(pk_ok), packed_verified_count=len(pk_ok), packed_mismatch_N=pk_bad,
                 packed_verified_all=(set(pk_ok) == set(rows)),
                 packed_missing_N=ranges(sorted(set(rows) - set(pk_ok))),
                 exact_N=ranges(sorted(set(px_ok) | set(ex_ok))), exact_count=len(set(px_ok) | set(ex_ok)),
                 packed_exact_N=ranges(px_ok), fraction_exact_N=ranges(ex_ok),
                 exact_mismatch_N=sorted(set(px_bad) | set(ex_bad)), exact_programs_conflict_N=ex_px_conflict,
                 reference_identical_N=ranges(ref_same), reference_identical_count=len(ref_same), reference_mismatch_N=ref_diff,
                 exact_ties_at_maximum={str(N): v for N, v in sorted(ties_top.items())},
                 exact_flat_steps={str(N): v for N, v in sorted(flat.items())},
                 min_gap_over_enclosure_width=float(min(gapratio)[0]), min_gap_at_N=min(gapratio)[1],
                 max_enclosure_width_at_mode=max(widths), min_F_at_mode=float(min(fvals)), max_F_at_mode=float(max(fvals)),
                 min_one_minus_gamma=(str(min(gam)) if gam else None),
                 max_error_bound_b=max((pk[N]["error_bound_b"] for N in pk_ok), default=None),
                 seconds=dict(generator=round(sum(c.get("seconds", 0) for c in rows.values())),
                              packed=round(sum(c.get("seconds", 0) for c in pk.values())),
                              packed_exact=round(sum(c.get("seconds", 0) for c in px.values())),
                              reference=round(sum(c.get("seconds", 0) for c in ref.values()))),
                 modes={str(N): c["mode"] for N, c in sorted(rows.items())})
        S["modes"][tag] = s
        macros[f"Nmax{DN[d]}{QN[q]}"] = str(nmax)
        macros[f"Extra{DN[d]}{QN[q]}"] = ",\\ ".join(str(x) for x in extra) if extra else "none"
        if gam:
            macros[f"Gam{DN[d]}{QN[q]}"] = sci_floor(min(gam))

# ---------------- comparison with the floating-point study (data/msc_modes)
float_study_path = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')
agree, disagree = [], []
if os.path.exists(float_study_path):
    for line in open(float_study_path):
        r = json.loads(line)
        qs = {0.8: "4/5", 0.5: "1/2", 1.0: "1/1"}.get(r.get("q"))
        if r.get("geo") != "CC" or qs is None:
            continue
        m = S["modes"].get(f"d{r['d']}_q{qs.replace('/', '-')}", {}).get("modes", {}).get(str(r["N"]))
        if m is not None:
            (agree if m == r["mode"] else disagree).append((r["d"], qs, r["N"], r["mode"], m))
S["comparison_with_float_study_modes"] = dict(agree=len(agree), agree_list=[list(x) for x in sorted(agree)],
                                             disagree=[list(x) for x in disagree])
allm = list(S["modes"].values())
macros["MinMargin"] = "$10^{%d}$" % int(math.floor(math.log10(min(v["min_gap_over_enclosure_width"] for v in allm))))
macros["MaxWidth"] = str(max(v["max_enclosure_width_at_mode"] for v in allm))
macros["FminExp"] = str(int(math.floor(math.log10(min(v["min_F_at_mode"] for v in allm)))))
macros["FmaxExp"] = str(int(math.floor(math.log10(max(v["max_F_at_mode"] for v in allm)))))
macros["FloatStudyAgree"] = str(len(agree))
macros["FloatStudyDisagree"] = str(len(disagree))
macros["TotalCerts"] = str(sum(v["count"] for v in allm))
macros["TotalRef"] = str(sum(v["reference_identical_count"] for v in allm))
macros["TotalPacked"] = str(sum(v["packed_verified_count"] for v in allm))
macros["TotalExact"] = str(sum(v["exact_count"] for v in allm))
macros["TotalMismatch"] = str(sum(len(v["packed_mismatch_N"]) + len(v["exact_mismatch_N"]) + len(v["reference_mismatch_N"])
                                  + len(v["exact_programs_conflict_N"]) for v in allm))
bmax = max((v["max_error_bound_b"] or 0) for v in allm)
macros["MaxB"] = str(bmax)

cf = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'closedform_summary.json'))) if os.path.exists(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'closedform_summary.json')) else {}
S["closed_form"] = cf
S["constants"] = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))
S["mfpt2d_remainder"] = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'mfpt2d_remainder.json')))

# ---------------- 1D law:  t* - kappa_1 (N - 1/2)^2 / q  lies in an interval of length < 1
kap = (Fraction(S["constants"]["kappa_1 = 2 tau_star"]["lo_dec"]), Fraction(S["constants"]["kappa_1 = 2 tau_star"]["hi_dec"]))
S["law_1d"] = {}
for q in ("4/5", "1/2"):
    tag = f"d1_q{q.replace('/', '-')}"
    nmax = S["modes"][tag]["N_max"]
    rows = [r for r in load(f"closedform_{tag}.jsonl") if "delta_1D_lo" in r]
    qq = Fraction(q)
    for Nlo in (10, 2):
        sel = [r for r in rows if Nlo <= r["N"] <= nmax]
        assert [r["N"] for r in sel if r["N"] >= 5] == list(range(max(Nlo, 5), nmax + 1))
        dl = min(Fraction(r["delta_1D_lo"]) for r in sel)
        dh = max(Fraction(r["delta_1D_hi"]) for r in sel)
        hi_r = Fraction(math.floor(dh * 1000) + 1, 1000)        # dh rounded up to 3 decimals (strictly above)
        lo_r = Fraction(math.ceil(dl * 1000) - 1, 1000)         # dl rounded down (strictly below)
        floor_ok = all((kap[0] * Fraction(2 * r["N"] - 1, 2) ** 2 / qq + hi_r).__floor__() == r["mode"]
                       == (kap[1] * Fraction(2 * r["N"] - 1, 2) ** 2 / qq + hi_r).__floor__() for r in sel)
        rr = []
        for r in sel:
            N = r["N"]
            ratio = Fraction(r["mode"]) * qq / (N * (N - 1))
            rr.append(max(abs(ratio - kap[0]), abs(ratio - kap[1])) * N * (N - 1))
        ext = {r["N"]: [r["delta_1D_lo"], r["delta_1D_hi"]] for r in rows if r["N"] > nmax}
        S["law_1d"][f"{tag}_N>={Nlo}"] = dict(N_min=Nlo, N_max=nmax, delta_lo_rounded=str(lo_r), delta_hi_rounded=str(hi_r),
                                               interval_length_lt_1=bool(hi_r - lo_r < 1), floor_formula_verified=bool(floor_ok),
                                               delta_min=float(dl), delta_max=float(dh),
                                               max_NNm1_times_abs_ratio_minus_kappa=float(max(rr)), extra_N=ext)
    k = S["law_1d"][f"{tag}_N>=10"]
    macros[f"DeltaLo{QN[q]}"] = f"{float(Fraction(k['delta_lo_rounded'])):.3f}"
    macros[f"DeltaHi{QN[q]}"] = f"{float(Fraction(k['delta_hi_rounded'])):.3f}"
    macros[f"DeltaHiS{QN[q]}"] = f"{float(Fraction(k['delta_hi_rounded'])):+.3f}"
    macros[f"RatioConst{QN[q]}"] = f"{math.ceil(k['max_NNm1_times_abs_ratio_minus_kappa'] * 100 + 1e-9) / 100:.2f}"

# ---------------- q-scaling: (4/5) t*(4/5) - (1/2) t*(1/2) on the common contiguous range (exact rationals)
S["q_scaling"] = {}
for d in (1, 2, 3):
    a, b = S["modes"][f"d{d}_q4-5"], S["modes"][f"d{d}_q1-2"]
    top = min(a["N_max"], b["N_max"])
    diff = {N: Fraction(4, 5) * a["modes"][str(N)] - Fraction(1, 2) * b["modes"][str(N)] for N in range(2, top + 1)}
    lo, hi = min(diff.values()), max(diff.values())
    S["q_scaling"][f"d{d}"] = dict(N_max=top, min=str(lo), max=str(hi), argmin=min(diff, key=diff.get), argmax=max(diff, key=diff.get))
    macros[f"QsLo{DN[d]}"] = f"{float(lo):.1f}"
    macros[f"QsHi{DN[d]}"] = f"{float(hi):.1f}"
    macros[f"QsN{DN[d]}"] = str(top)

# ---------------- closed-form macros
for tag, s in cf.items():
    d, q = s["d"], s["q"]
    key = f"eps[10,{s['N_max']}]"
    if key in s:
        macros[f"Eps{DN[d]}{QN[q]}"] = up(s[key]["max_abs_rel_err_upper"], 4 if d == 1 else 5)
        macros[f"EpsAt{DN[d]}{QN[q]}"] = str(s[key]["at_N"])
        macros[f"EpsLo{DN[d]}{QN[q]}"] = down(s[key]["min_abs_rel_err_lower"], 4 if d == 1 else 6)
        macros[f"EpsN{DN[d]}{QN[q]}"] = str(s["N_max"])
rem = S["mfpt2d_remainder"]
macros["RemLo"] = down(rem["self_contained"]["min_remainder_lower"], 4)
macros["RemHiA"] = up(rem["self_contained"]["max_remainder_upper"], 5)
macros["RemNA"] = str(rem["self_contained"]["N_max"])
if rem.get("via_single_sum"):
    macros["RemHiB"] = up(rem["via_single_sum"]["max_remainder_upper"], 7)
    macros["RemNB"] = str(rem["via_single_sum"]["N_max"])


# ---------------- tables
def w(name, text, src):
    """write a generated LaTeX fragment; src = the files its numbers are read from (paths relative to this directory's parent)."""
    open(os.path.join(ROOT, name), "w").write("% generated by scripts/summarise.py -- do not edit\n% src: " + src + "\n" + text)


lines = [r"\begin{tabular}{cc|p{2.7cm}|c|p{3.3cm}|p{1.5cm}|p{3.9cm}}",
         r"$d$ & $q$ & certified $N$ (generator \texttt{fplimb}) & number & verified with Python integers (\texttt{verify\_packed}) & exact arithmetic & bit-identical (\texttt{fpcore}) \\ \hline"]
for tag, s in S["modes"].items():
    rng = f"$2$--${s['N_max']}$" + (f", ${', '.join(str(x) for x in s['extra_N'])}$" if s["extra_N"] else "")
    lines.append(f"{s['d']} & ${QTEX[s['q']]}$ & {rng} & {s['count']} & "
                 f"{'all' if s['packed_verified_all'] else (s['packed_verified_N'] or 'none')} & "
                 f"{s['exact_N'] or 'none'} & {s['reference_identical_N'] or 'none'} \\\\")
lines.append(r"\end{tabular}")
w("tab_ranges.tex", "\n".join(lines) + "\n", "certs/SUMMARY.json, key modes (from certs/modes_*.jsonl, certs/packed_*.jsonl, "
  "certs/packedexact_*.jsonl, certs/exactcheck_*.jsonl, certs/refcheck_*.jsonl)")

sel = {1: [2, 3, 5, 10, 20, 35, 50, 100, 200, 300, 400, 500, 600, 800, 1000],
       2: [2, 3, 5, 10, 20, 35, 50, 80, 100, 120, 140, 160, 180, 200],
       3: [2, 3, 5, 10, 15, 20, 30, 35, 40, 50, 60]}
for d in (1, 2, 3):
    cfr = {r["N"]: r for r in load(f"closedform_d{d}_q4-5.jsonl")}
    mh = S["modes"].get(f"d{d}_q1-2", {}).get("modes", {})
    mu = S["modes"].get(f"d{d}_q1-1", {}).get("modes", {})
    lines = [r"\begin{tabular}{r|r|r|r|r|r|r|r}",
             r"$N$ & $t^*$ ($q=4/5$) & $\tau_N=q\,\mathbb E T$ & $t^*/\mathbb E T$ & $t_{\rm cf}$ & $(t_{\rm cf}-t^*)/t^*$ & $t^*$ ($q=1/2$) & $t^*$ ($q=1$) \\ \hline"]
    for N in sel[d]:
        if N not in cfr:
            continue
        r = cfr[N]
        tau = float(Fraction(r["mfpt_lo"]) * Fraction(4, 5))
        lines.append(f"{N} & {r['mode']} & {tau:.4f} & {float(Fraction(r['mode_over_mfpt_lo'])):.6f} & {float(Fraction(r['t_cf_lo'])):.3f} & "
                     f"${float(Fraction(r['rel_err_lo'])):+.5f}$ & {mh.get(str(N), '--')} & {mu.get(str(N), '--')} \\\\")
    lines.append(r"\end{tabular}")
    w(f"tab_modes_d{d}.tex", "\n".join(lines) + "\n", f"certs/closedform_d{d}_q4-5.jsonl (t*, tau_N, t_cf), certs/modes_d{d}_q1-2.jsonl, "
      f"certs/modes_d{d}_q1-1.jsonl")


def signed(lo, hi, nd=6):
    a = down(lo, nd) if not lo.startswith("-") else "-" + up(lo[1:], nd)
    b = up(hi, nd) if not hi.startswith("-") else "-" + down(hi[1:], nd)
    return f"[{a},\\ {b}]"


# ---------------- appendix: all certified modes for d = 2, 3
for d in (2, 3):
    for q in ("4/5", "1/2"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        md = S["modes"][tag]["modes"]
        items = [(int(N), t) for N, t in md.items()]
        per = 8
        lines = [r"\begin{tabular}{" + "|".join(["rr"] * per) + "}", " & ".join([r"$N$ & $t^*$"] * per) + r" \\ \hline"]
        nrow = -(-len(items) // per)
        for r in range(nrow):
            cells = []
            for c in range(per):
                i = c * nrow + r
                cells.append(f"{items[i][0]} & {items[i][1]}" if i < len(items) else " & ")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\end{tabular}")
        w(f"tab_all_{tag}.tex", "\n".join(lines) + "\n", f"field mode of certs/modes_{tag}.jsonl")
md1 = {q: S["modes"][f"d1_q{q.replace('/', '-')}"]["modes"] for q in ("4/5", "1/2")}
w("text_small_1d.tex", "; ".join(f"$q={QTEX[q]}$: " + ", ".join(md1[q][str(N)].__str__() for N in range(2, 10)) for q in ("4/5", "1/2")) + "\n",
  "field mode of certs/modes_d1_q4-5.jsonl and certs/modes_d1_q1-2.jsonl")

lines = [r"\begin{tabular}{cc|c|c|c|c}",
         r"$d$ & $q$ & range of $N$ & bound on $|t_{\rm cf}-t^*|/t^*$ & worst $N$ & interval containing $(t_{\rm cf}-t^*)/t^*$ \\ \hline"]
for tag, s in cf.items():
    if s["q"] == "1/1" and s["d"] == 1:
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
w("tab_cf.tex", "\n".join(lines) + "\n", "certs/closedform_summary.json (from certs/closedform_d*_q*.jsonl, written by scripts/closed_form.py)")

# ---------------- text of the q = 1 part of the mode theorem
import sys
sys.path.insert(0, HERE)
import verify_exact as ve
MR = lambda ns: ranges(ns, sep=r"\text{--}")          # ranges inside math mode


def tie_description(d, N, q):
    """Exact description of a case that is neither strictly unimodal nor certified multimodal (small N)."""
    f, T = ve.exact_pmf(d, N, Fraction(q))
    eq = [t for t in range(1, T + 1) if f[t] == f[t + 1] and f[t] > 0]
    rel = [("<" if f[t] < f[t + 1] else ">" if f[t] > f[t + 1] else "=") for t in range(1, T + 1)]
    core = [r for r in rel if r != "="]
    weak = "".join(core) == "<" * core.count("<") + ">" * core.count(">")
    return eq, weak


txt = [r"\begin{itemize}"]
S["q1_tie_cases"] = {}
for d in (1, 2, 3):
    sq = S["modes"].get(f"d{d}_q1-1")
    if not sq:
        continue
    parts = []
    ties = sq["exact_ties_at_maximum"]
    allN = sorted(int(N) for N in sq["modes"])
    if sq["not_certified_N"]:
        covered = all(str(N) in ties for N in sq["not_certified_N"])
        desc = "; ".join(f"$N={N}$: $t\\in\\{{{', '.join(str(t) for t in ties[str(N)])}\\}}$" for N in sq["not_certified_N"] if str(N) in ties)
        parts.append(f"the maximiser is unique except for $N\\in\\{{{', '.join(str(N) for N in sq['not_certified_N'])}\\}}$, "
                     f"where the maximum is attained at exactly two times ({desc})" + ("" if covered else " [SOME UNDECIDED]"))
    else:
        parts.append("the maximiser is unique for every $N$ of the range")
    uni = sorted(set(allN) - set(sq["non_unimodal_N"]))
    parts.append(f"$f$ is strictly unimodal for $N\\in{{}}$\\{{{ranges(uni)}\\}}" if uni else "$f$ is strictly unimodal for no $N$ of the range")
    if sq["multimodal_N"]:
        parts.append(f"$f$ has at least two strict local maxima, hence is \\emph{{not}} unimodal, for $N\\in{{}}$\\{{{ranges(sq['multimodal_N'])}\\}} "
                     f"(the largest certified number of strict local maxima is {sq['max_local_maxima']})")
    for N in sq["undecided_N"]:
        if N > 12:
            parts.append(f"for $N={N}$ strict unimodality is UNDECIDED")
            continue
        eq, weak = tie_description(d, N, "1")
        S["q1_tie_cases"][f"d{d}_N{N}"] = dict(equal_consecutive_values_at_t=eq, weakly_unimodal=weak)
        parts.append(f"for $N={N}$, $f$ is {'unimodal in the weak sense but not strictly' if weak else 'NOT weakly unimodal'}: "
                     + ", ".join(f"$f({t})=f({t + 1})$" for t in eq))
    # equalities of consecutive values: every relation left undecided by the rounded certificate must be settled exactly
    rows_q1 = {c["N"]: c for c in load(f"modes_d{d}_q1-1.jsonl")}
    px_q1 = {c["N"]: c for c in load(f"packedexact_d{d}_q1-1.jsonl")}
    eqs = {}
    for N, c in sorted(rows_q1.items()):
        if c["undecided_adjacent_pairs"]:
            if N in px_q1:
                eqs[N] = px_q1[N]["equal_adjacent_pairs_from_t0"]
            elif N <= 12:
                eqs[N] = tie_description(d, N, "1")[0]
            else:
                eqs[N] = None
    S["q1_adjacent_equalities"] = S.get("q1_adjacent_equalities", {})
    S["q1_adjacent_equalities"][f"d{d}"] = {str(N): v for N, v in eqs.items()}
    if any(v is None or not v for v in eqs.values()):
        parts.append("SOME RELATION UNDECIDED")
    elif eqs:
        lst = " and ".join(f"$N={N}$ (" + ", ".join(f"$f({t})=f({t + 1})$" for t in v) + ")" for N, v in eqs.items())
        parts.append(f"equal consecutive non-zero values occur only for {lst} (exact arithmetic): for every other pair $(N,t)$ "
                     f"with $N$ in the range and $t\\ge d(N-1)-1$, $f(t)\\ne f(t+1)$")
    else:
        parts.append("$f(t)\\ne f(t+1)$ for every $N$ of the range and every $t\\ge d(N-1)-1$")
    if sq["tail_ne_mode_N"]:
        parts.append(f"the tail time $T$ exceeds $t^*$ for {len(sq['tail_ne_mode_N'])} of the {sq['count']} values of $N$ "
                     f"(largest ratio $T/t^*={sq['max_tail_over_mode']:.2f}$)")
    txt.append(f"\\item[$d={d}$:] " + "; ".join(parts) + ".")
txt.append(r"\end{itemize}")
w("text_q1.tex", "\n".join(txt) + "\n", "certs/modes_d*_q1-1.jsonl, certs/packed_d*_q1-1.jsonl, certs/packedexact_d*_q1-1.jsonl, "
  "certs/exactcheck_d*_q1-1.jsonl; exact ties re-derived by scripts/verify_exact.py")

lines = [r"\begin{tabular}{cc|r|r|r|r|r|r|r}",
         r"$d$ & $q$ & certificates & \texttt{verify\_packed} & exact & \texttt{fpcore} & \multicolumn{3}{c}{time (s): generator, packed, exact} \\ \hline"]
for tag, sm in S["modes"].items():
    t = sm["seconds"]
    lines.append(f"{sm['d']} & ${QTEX[sm['q']]}$ & {sm['count']} & {sm['packed_verified_count']} & {sm['exact_count']} & "
                 f"{sm['reference_identical_count']} & {t['generator']} & {t['packed']} & {t['packed_exact']} \\\\")
lines.append(r"\end{tabular}")
w("tab_verif.tex", "\n".join(lines) + "\n", "certs/SUMMARY.json, key modes (counts; sums of the field seconds, wall-clock, of the certificate files)")

json.dump(S, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'SUMMARY.json'), "w"), indent=1)
used = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'R5_certified_computations.tex')).read()))
expected = {m for m in used if re.match(r"(Nmax|Extra|Gam(One|Two|Three)|Eps|Delta(Lo|Hi)|RatioConst|Rem|Min|Max|Fm|FloatStudy|Total|Qs)", m)}
missing = sorted(expected - set(macros))
for k in missing:
    macros[k] = "??"
if missing:
    print("WARNING: macros without data:", missing)
def macro_src(k):
    if k.startswith("Eps"):
        return "certs/closedform_summary.json (from certs/closedform_d*_q*.jsonl, written by scripts/closed_form.py)"
    if k.startswith(("Delta", "RatioConst")):
        return "certs/SUMMARY.json, key law_1d (from certs/closedform_d1_q*.jsonl and certs/constants.json)"
    if k.startswith("Qs"):
        return "certs/SUMMARY.json, key q_scaling (field mode of certs/modes_d*_q4-5.jsonl and certs/modes_d*_q1-2.jsonl)"
    if k.startswith("Rem"):
        return "certs/mfpt2d_remainder.json (written by scripts/mfpt2d_remainder.py)"
    if k.startswith("FloatStudy"):
        return "certs/SUMMARY.json, key comparison_with_float_study_modes (floating-point study: data/msc_modes/discrete_modes.jsonl)"
    if k.startswith("Total"):
        return "certs/SUMMARY.json, key modes (counts over certs/modes_*.jsonl, certs/packed_*.jsonl, certs/packedexact_*.jsonl, certs/exactcheck_*.jsonl, certs/refcheck_*.jsonl)"
    return "certs/SUMMARY.json, key modes (from certs/modes_d*_q*.jsonl and certs/packed_d*_q*.jsonl)"


with open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'numbers.tex'), "w") as fh:
    fh.write("% generated by scripts/summarise.py -- do not edit\n")
    last = None
    for k in sorted(macros, key=lambda k: (macro_src(k), k)):
        if macro_src(k) != last:
            last = macro_src(k)
            fh.write(f"% src: {last}\n")
        fh.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
hide = ("modes",)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in hide} for k, v in S["modes"].items()}, indent=1))
print(json.dumps(S["law_1d"], indent=1))
print(json.dumps(macros, indent=1))
