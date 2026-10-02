"""audit_ext.py -- re-check every numerical assertion of the extended-range theorems (Section 'A second engine and
extended ranges' of note R5, and the extended closed-form / one-dimensional statements) against the raw certificate
files certs/c128_*.jsonl.  The numbers printed in the theorems are macros in numbers_ext.tex (written by
summarise_ext.py); this script reads them back and tests each assertion with exact rational arithmetic and, for the
closed form, with mpmath.iv (separately from closed_form.py, which uses Arb).  It never repairs anything.

Output: certs/AUDIT_EXT.json; exit status 1 if any assertion fails.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, re, sys
from fractions import Fraction
from mpmath import iv, libmp
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
iv.prec = 200
DN = {1: "One", 2: "Two", 3: "Three"}
QN = {"4/5": "F", "1/2": "H", "1/1": "U"}
KEYS = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T", "undecided_adjacent_pairs")
EXC = {(1, "4/5", 4), (1, "1/2", 3), (1, "1/2", 4)}
report, failures = [], 0


def check(name, ok, detail=""):
    global failures
    report.append(dict(assertion=name, status="PASS" if ok else "FAIL", detail=detail))
    failures += not ok
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""), flush=True)


def load(name):
    p = os.path.join(CERT, name)
    return {json.loads(l)["N"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def frac(s):
    n, dn = s.split("/")
    return Fraction(int(n), int(dn))


def ivq(fr):
    return iv.mpf(fr.numerator) / iv.mpf(fr.denominator)


def rat(x):
    return tuple(Fraction(*libmp.to_rational(e)) for e in x._mpi_)


def rng(ns):
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}--{ns[j]}")
        i = j + 1
    return ", ".join(out)


macros = {m.group(1): m.group(2) for m in
          re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'numbers_ext.tex')).read(), re.M)}
M = lambda k: macros[k]


def Nset(d, q):
    key = f"{DN[d]}{QN[q]}"
    top = int(M(f"XNmax{key}"))
    ex = M(f"XExtra{key}")
    extra = [] if ex == "none" else [int(x) for x in re.findall(r"\d+", ex)]
    return list(range(2, top + 1)) + extra, top


def intersect(a, b):
    return all(not (int(x) > int(b["F_enclosures"][t][1]) or int(y) < int(b["F_enclosures"][t][0]))
               for t, (x, y) in a["F_enclosures"].items() if t in b["F_enclosures"])


# ====================================================================== validation files (V1)
for name, key in (("c128_refcheck.json", "XRefSmall"), ("c128_refcheck_medium.json", "XRefMedium")):
    j = json.load(open(os.path.join(CERT, name)))
    check(f"(V1) {name}: {M(key)} cases, C output byte-identical with the Python mirror in every case",
          j["cases"] == int(M(key)) and j["mismatches"] == 0 and all(r["identical"] for r in j["results"]))

# ====================================================================== certificates
total, total_new, reg_n, recheck_n = 0, 0, 0, 0
allrows = {}
for q in ("4/5", "1/2", "1/1"):
    for d in (1, 2, 3):
        tag = f"d{d}_q{q.replace('/', '-')}"
        rows, gen = load(f"c128_{tag}.jsonl"), load(f"modes_{tag}.jsonl")
        allrows[tag] = rows
        Ns, top = Nset(d, q)
        total += len(rows)
        total_new += len(set(rows) - set(gen))
        check(f"[{tag}] a C certificate exists for every N of the stated set ({len(Ns)} values, 2..{top} + extras) and for no other N",
              set(rows) == set(Ns), f"file has {len(rows)} lines")
        check(f"[{tag}] every line: engine c128, P = 112, symmetry-reduced, t0 = d(N-1), t_sw < 10^5, 1 - gamma > 0, tail margin > 0",
              all(c["engine"] == "c128" and c["P"] == 112 and c["reduced"] and c["t0"] == d * (N - 1) and c["ts"] < 10 ** 5
                  and Fraction(c["one_minus_gamma_lower"]) > 0 and int(c["tail_margin"]) > 0 for N, c in rows.items()),
              f"max t_sw = {max(c['ts'] for c in rows.values())}, max T = {max(c['T_tail'] for c in rows.values())}")
        ok = [N for N in gen if N in rows and all(rows[N][k] == gen[N][k] for k in KEYS) and intersect(gen[N], rows[N])]
        reg_n += len(ok)
        check(f"[{tag}] (V2) identical t0, t*, T, verdicts, peak and undecided counts as the Tier-A certificate, intersecting enclosures, for every Tier-A N",
              set(ok) == set(gen), f"{len(ok)} of {len(gen)}")
        for eng in ("limb", "packed"):
            x = load(f"xcheck_{eng}_{tag}.jsonl")
            if x:
                check(f"[{tag}] (V3) cross-check by {eng}: identical verdicts and intersecting enclosures; every run shorter than 20 minutes",
                      all(c["same_verdicts"] and c["enclosures_intersect"] and all(c[k] == rows[N][k] for k in KEYS[:6]) and c["seconds"] < 1200 for N, c in x.items()),
                      f"N = {rng(x)}; longest run {max(c['seconds'] for c in x.values()):.0f} s")
        rc = load(f"c128_recheck_{tag}.jsonl")
        recheck_n += len(set(rc) & set(Ns))
        check(f"[{tag}] (V4),(V5) second run of every certificate: identical sha256 digest, conservation of probability holds",
              set(rc) >= set(Ns) and all(rc[N]["identical_to_first_run"] and rc[N]["sha256_second_run"] == rows[N]["digest"]
                                         and rc[N]["conservation_ok"] and int(rc[N]["slack_lower"]) >= 0 and int(rc[N]["slack_upper"]) >= 0 for N in Ns),
              f"{len(set(rc) & set(Ns))} of {len(Ns)}; largest relative slack {max([rc[N]['relative_slack'] for N in Ns if N in rc], default=0):.2e}")
        if q != "1/1":
            normal = [N for N in Ns if (d, q, N) not in EXC]
            check(f"[{tag}] strictly unimodal with unique certified mode, no undecided relation, for every non-exceptional N",
                  all(rows[N]["mode_certified"] and rows[N]["unimodal_certified"] and rows[N]["certified_local_maxima_upto_T"] == 1
                      and rows[N]["undecided_adjacent_pairs"] == 0 for N in normal))
            check(f"[{tag}] tail time T equals the mode for every non-exceptional N",
                  all(rows[N]["T_tail"] == rows[N]["mode"] for N in normal))
            gmin = min(Fraction(rows[N]["one_minus_gamma_lower"]) for N in Ns)
            check(f"[{tag}] the printed lower bound for min(1-gamma) is valid", bool(re.match(r"([0-9.]+)\\times10\^\{(-?\d+)\}", M(f"XGam{DN[d]}{QN[q]}")))
                  and Fraction(re.match(r"([0-9.]+)", M(f"XGam{DN[d]}{QN[q]}")).group(1)) * Fraction(10) ** int(re.search(r"\{(-?\d+)\}", M(f"XGam{DN[d]}{QN[q]}")).group(1)) <= gmin,
                  f"min(1-gamma) >= {float(gmin):.3g}")
            for (dd, qq, N) in sorted(EXC):
                if (dd, qq) == (d, q):
                    check(f"[{tag}] exceptional N={N}: one undecided relation, strict unimodality not certified (exact tie, Theorem modes (d))",
                          not rows[N]["unimodal_certified"] and rows[N]["undecided_adjacent_pairs"] == 1)
check(f"totals: {M('XTotalCerts')} C certificates, {M('XTotalNew')} of them outside the Tier-A sets, {M('XRegressN')} regression cases, {M('XRegressBad')} disagreements; "
      f"{M('XRecheckN')} second runs, {M('XRecheckBad')} failures",
      total == int(M("XTotalCerts")) and total_new == int(M("XTotalNew")) and reg_n == int(M("XRegressN")) and M("XRegressBad") == "0"
      and recheck_n == int(M("XRecheckN")) == total and M("XRecheckBad") == "0" and M("XRecheckMissing") == "0")

sx = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'SUMMARY_EXT.json')))["comparison_with_float_study_modes"]
float_study_path = _os.path.join(_R, 'data', 'msc_modes', 'discrete_modes.jsonl')
n_ag, n_dis, n_unc = set(), 0, set()
for line in open(float_study_path):
    r = json.loads(line)
    qs = {0.8: "4/5", 0.5: "1/2", 1.0: "1/1"}.get(r.get("q"))
    if r.get("geo") != "CC" or qs is None:
        continue
    c = allrows[f"d{r['d']}_q{qs.replace('/', '-')}"].get(r["N"])
    if c is None:
        n_unc.add((r["d"], qs, r["N"]))
    elif c["mode"] == r["mode"]:
        n_ag.add((r["d"], qs, r["N"]))
    else:
        n_dis += 1
check(f"comparison with the floating-point study: every corner-to-corner case of the floating-point study with q in {{4/5, 1/2, 1}} lies in the certified sets and has the same mode ({M('XFloatStudyAgree')} cases)",
      len(n_ag) == int(M("XFloatStudyAgree")) and n_dis == 0 == int(M("XFloatStudyDisagree")) and not n_unc and M("XFloatStudyUncovered") == "0",
      f"agree {len(n_ag)}, disagree {n_dis}, not covered {sorted(n_unc)}")

# ====================================================================== q-scaling
for d in (1, 2, 3):
    A, B = allrows[f"d{d}_q4-5"], allrows[f"d{d}_q1-2"]
    top, lo, hi = int(M(f"XQsN{DN[d]}")), Fraction(M(f"XQsLo{DN[d]}")), Fraction(M(f"XQsHi{DN[d]}"))
    vals = [Fraction(4, 5) * A[N]["mode"] - Fraction(1, 2) * B[N]["mode"] for N in range(2, top + 1)]
    check(f"q-scaling, d = {d}: {lo} <= (4/5) t*(4/5) - (1/2) t*(1/2) <= {hi} for all 2 <= N <= {top}, both bounds attained",
          min(vals) == lo and max(vals) == hi and top == min(Nset(d, "4/5")[1], Nset(d, "1/2")[1]))

# ====================================================================== q = 1
txt = open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'text_ext_q1.tex')).read()
q1 = {}
for d in (1, 2, 3):
    tag = f"d{d}_q1-1"
    rows = allrows[tag]
    Ns, top = Nset(d, "1/1")
    uni = [N for N in Ns if rows[N]["unimodal_certified"]]
    multi = [N for N in Ns if rows[N]["certified_local_maxima_upto_T"] >= 2]
    und = sorted(set(Ns) - set(uni) - set(multi))
    notc = [N for N in Ns if not rows[N]["mode_certified"]]
    px = load(f"packedexact_{tag}.jsonl")
    xe = load(f"xcheck_exact_{tag}.jsonl")
    ex = {N: (px[N] if N in px else xe.get(N)) for N in set(notc) | set(und)}
    check(f"[{tag}] every N without a certified unique mode is settled in exact arithmetic (two maximisers, 'mode' is the smaller)",
          all(ex[N] is not None and len(ex[N]["maximisers_of_lower_bound"]) == 2 and rows[N]["mode"] == min(ex[N]["maximisers_of_lower_bound"]) for N in notc),
          f"N = {notc}")
    check(f"[{tag}] every N is classified: strictly unimodal / >= 2 certified strict local maxima / tie settled in exact arithmetic",
          all(ex[N] is not None and ex[N]["equal_adjacent_pairs_from_t0"] for N in und) and "UNDECIDED" not in txt,
          f"unimodal: {rng(uni) or 'none'}; multimodal: {rng(multi) or 'none'}; tie only: {rng(und) or 'none'}")
    check(f"[{tag}] the generated text of the theorem lists these sets",
          (not uni or "{" + rng(uni) + "\\}" in txt) and (not multi or "{" + rng(multi) + "\\}" in txt) and all(f"$N={N}$" in txt for N in und))
    undp = {N: rows[N]["undecided_adjacent_pairs"] for N in Ns if rows[N]["undecided_adjacent_pairs"]}
    eqx = {N: (px[N]["equal_adjacent_pairs_from_t0"] if N in px else None) for N in undp}
    check(f"[{tag}] every relation left undecided by the C certificates is settled exactly (Tier-A exact file); the text lists every "
          f"equality f(t) = f(t+1) > 0 and states f(t) != f(t+1) otherwise",
          all(eqx[N] for N in undp) and all(f"$N={N}$ (" + ", ".join(f"$f({t})=f({t + 1})$" for t in eqx[N]) + ")" in txt for N in undp)
          and txt.count("f(t)\\ne f(t+1)") >= 1, f"equalities: {eqx}")
    q1[tag] = dict(strictly_unimodal=rng(uni), multimodal=rng(multi), tie_only=rng(und), adjacent_equalities={str(N): v for N, v in eqx.items()})

# ====================================================================== Conjecture conj:last: counts printed in its status paragraph
nl = dict(all=0, nonexc=0, small=0, large=0)
for d in (1, 2, 3):
    for q in ("4/5", "1/2"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        px = load(f"packedexact_{tag}.jsonl")
        for N, c in allrows[tag].items():
            nl["all"] += 1
            if (d, q, N) in EXC:
                mx = px[N]["maximisers_of_lower_bound"]
                assert px[N]["T_tail"] == c["T_tail"]
                nl["small"] += c["T_tail"] == min(mx)
                nl["large"] += c["T_tail"] == max(mx)
            else:
                nl["nonexc"] += 1
                nl["small"] += c["mode_certified"] and c["T_tail"] == c["mode"]
                nl["large"] += c["mode_certified"] and c["T_tail"] == c["mode"]
check(f"Conjecture last: {M('XLastCerts')} certificates with q in {{4/5, 1/2}}, {M('XLastNonExc')} non-exceptional; T = smallest maximiser in "
      f"{M('XLastSmall')}, T = largest maximiser in {M('XLastLarge')} (exceptional triples from the exact files)",
      (nl["all"], nl["nonexc"], nl["small"], nl["large"]) == tuple(int(M(k)) for k in ("XLastCerts", "XLastNonExc", "XLastSmall", "XLastLarge"))
      and nl["large"] == nl["all"], str(nl))

# ====================================================================== closed form, recomputed with mpmath.iv
mf = {d: {N: (frac(r["tau_lo"]), frac(r["tau_hi"])) for N, r in load(f"mfpt_d{d}.jsonl").items()} for d in (1, 2, 3)}
check("MFPT enclosures: 0 < lower <= upper, relative width < 1e-28; for d = 1 they contain N(N-1)",
      all(0 < lo <= hi and (hi - lo) / lo < Fraction(1, 10 ** 28) for d in mf for lo, hi in mf[d].values())
      and all(lo <= N * (N - 1) <= hi for N, (lo, hi) in mf[1].items()),
      f"d=1: {len(mf[1])}, d=2: {len(mf[2])}, d=3: {len(mf[3])} values of N")


def rel_err(d, N, q, tstar):
    lo, hi = mf[d][N]
    tau = iv.mpf([ivq(lo).a, ivq(hi).b])
    X = tau / d * (1 - iv.cos(iv.pi / N))
    tcf = tau / ivq(q) * iv.log(2 * d * X) / (X + 2 * d - 1)
    return rat((tcf - tstar) / tstar)


cf_out = {}
for q in ("4/5", "1/2"):
    for d in (1, 2, 3):
        tag = f"d{d}_q{q.replace('/', '-')}"
        rows = allrows[tag]
        key = f"{DN[d]}{QN[q]}"
        top, eps = int(M(f"XEpsN{key}")), Fraction(M(f"XEps{key}"))
        check(f"[{tag}] every N in 10..{top} has a certified unique mode and an MFPT enclosure; {top} is the top of the contiguous certified range",
              all(rows[N]["mode_certified"] and N in mf[d] for N in range(10, top + 1)) and top == int(M(f"XNmax{key}")))
        errs = {N: rel_err(d, N, Fraction(q), rows[N]["mode"]) for N in range(10, top + 1)}
        if d == 1:
            lo = Fraction(M(f"XEpsLo{key}"))
            check(f"[{tag}] {lo} t* <= t_cf - t* <= {eps} t* for all 10 <= N <= {top}  (mpmath.iv)",
                  all(lo <= a and b <= eps for a, b in errs.values()),
                  f"range [{float(min(a for a, b in errs.values())):.6f}, {float(max(b for a, b in errs.values())):.6f}]")
        else:
            worst = max(errs, key=lambda N: max(abs(errs[N][0]), abs(errs[N][1])))
            check(f"[{tag}] |t* - t_cf| <= {eps} t* for all 10 <= N <= {top}  (mpmath.iv)",
                  all(-eps <= a and b <= eps for a, b in errs.values()) and worst == int(M(f"XEpsAt{key}")),
                  f"worst N = {worst}: {float(max(abs(errs[worst][0]), abs(errs[worst][1]))):.6f}")
        stored = load(f"closedform_c128_{tag}.jsonl")
        check(f"[{tag}] the Arb enclosures stored by closed_form.py intersect the mpmath.iv ones for every N",
              all(not (Fraction(stored[N]["rel_err_lo"]) > b or Fraction(stored[N]["rel_err_hi"]) < a) for N, (a, b) in errs.items()))
        cf_out[tag] = {str(N): [float(a), float(b)] for N, (a, b) in errs.items() if N in (10, 20, 50, 100, 200, 300, 600, 1000, top)}
        Ns, _ = Nset(d, q)
        for N in [n for n in Ns if n > top]:
            a, b = rel_err(d, N, Fraction(q), rows[N]["mode"])
            cf_out[tag][str(N)] = [float(a), float(b)]
            if d == 1:
                check(f"[{tag}] isolated N = {N}: the same two-sided bound holds", Fraction(M(f"XEpsLo{key}")) <= a and b <= eps, f"[{float(a):.6f}, {float(b):.6f}]")
            else:
                check(f"[{tag}] isolated N = {N}: |t* - t_cf| <= {eps} t*", -eps <= a and b <= eps, f"[{float(a):.6f}, {float(b):.6f}]")

# the sub-range table tab_ext_cf.tex: every row re-checked
tab = open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'tab_ext_cf.tex')).read()
nrow, okrow = 0, True
for m in re.finditer(r"^(\d) & \$([0-9/]+)\$ & \$(\d+)\\le N\\le (\d+)\$ & \$([0-9.]+)\$ & (\d+) & \$\[(-?[0-9.]+),\\ (-?[0-9.]+)\]\$", tab, re.M):
    d, q, a, b = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    bound, lo, hi = Fraction(m.group(5)), Fraction(m.group(7)), Fraction(m.group(8))
    rows = allrows[f"d{d}_q{q.replace('/', '-')}"]
    nrow += 1
    for N in range(a, b + 1):
        x, y = rel_err(d, N, Fraction(q), rows[N]["mode"])
        okrow = okrow and lo <= x and y <= hi and max(abs(x), abs(y)) <= bound
niso = 0
for m in re.finditer(r"^(\d) & \$([0-9/]+)\$ & \$N=(\d+)\$ & \$([0-9.]+)\$ & (\d+) & \$\[(-?[0-9.]+),\\ (-?[0-9.]+)\]\$", tab, re.M):
    d, q, N = int(m.group(1)), m.group(2), int(m.group(3))
    bound, lo, hi = Fraction(m.group(4)), Fraction(m.group(6)), Fraction(m.group(7))
    x, y = rel_err(d, N, Fraction(q), allrows[f"d{d}_q{q.replace('/', '-')}"][N]["mode"])
    niso += 1
    okrow = okrow and lo <= x and y <= hi and max(abs(x), abs(y)) <= bound
check(f"Table of sub-range bounds (tab_ext_cf.tex): all {nrow} range rows hold for every N of their range, and all {niso} single-N rows hold  (mpmath.iv)",
      okrow and nrow > 0)

# ====================================================================== 1D law
cj = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))
kap = (2 * frac(cj["tau_star"]["lo"]), 2 * frac(cj["tau_star"]["hi"]))
law = {}
for q in ("4/5", "1/2"):
    tag = f"d1_q{q.replace('/', '-')}"
    rows = allrows[tag]
    Ns, top = Nset(1, q)
    qq = Fraction(q)
    dlo, dhi, c = Fraction(M(f"XDeltaLo{QN[q]}")), Fraction(M(f"XDeltaHi{QN[q]}")), Fraction(M(f"XRatioConst{QN[q]}"))
    y = lambda N, kk: kk * Fraction(2 * N - 1, 2) ** 2 / qq
    check(f"[{tag}] {dlo} < t* - kappa_1 (N-1/2)^2/q < {dhi} for all 10 <= N <= {top}, and the interval is shorter than 1",
          all(dlo < rows[N]["mode"] - y(N, kap[1]) and rows[N]["mode"] - y(N, kap[0]) < dhi for N in range(10, top + 1)) and dhi - dlo < 1,
          f"length {float(dhi - dlo):.3f}")
    check(f"[{tag}] t* = floor(kappa_1 (N-1/2)^2/q + ({dhi})) for all 10 <= N <= {top} (both endpoints of the kappa_1 enclosure)",
          all((y(N, kk) + dhi).__floor__() == rows[N]["mode"] for N in range(10, top + 1) for kk in kap))
    check(f"[{tag}] |t*/E T - kappa_1| <= {c}/(N(N-1)) for all 10 <= N <= {top}",
          all(abs(rows[N]["mode"] * qq / (N * (N - 1)) - kk) * N * (N - 1) <= c for N in range(10, top + 1) for kk in kap))
    extra = [N for N in Ns if N > top]
    if extra:
        check(f"[{tag}] the isolated N = {extra} satisfy the same two-sided bound and the floor formula",
              all(dlo < rows[N]["mode"] - y(N, kap[1]) and rows[N]["mode"] - y(N, kap[0]) < dhi
                  and (y(N, kk) + dhi).__floor__() == rows[N]["mode"] for N in extra for kk in kap))
    law[tag] = dict(delta_lo=str(dlo), delta_hi=str(dhi), ratio_const=str(c), N_max=top)

json.dump(dict(failures=failures, assertions=len(report), q1=q1, closed_form_samples=cf_out, law_1d=law, report=report),
          open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'AUDIT_EXT.json'), "w"), indent=1)
print(f"\n{len(report)} assertions, {failures} failures")
sys.exit(1 if failures else 0)
