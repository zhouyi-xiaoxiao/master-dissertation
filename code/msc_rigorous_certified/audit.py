"""audit.py -- re-check every numerical assertion of the theorems of note R5 against the certificate files.

The numbers printed in the theorems are macros in numbers.tex (written by summarise.py) or literals in
note R5.  This script reads those numbers back and tests each assertion directly on the raw certificate
files, with exact rational arithmetic (fractions.Fraction) and, for the transcendental quantities, with
mpmath.iv interval arithmetic (separately from closed_form.py / mfpt2d_remainder.py, which use Arb).
It never repairs anything: an assertion that fails is reported as FAIL and the exit status is 1.

Output: certs/AUDIT.json, certs/MANIFEST.sha256
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, hashlib, json, os, re, sys
from fractions import Fraction
from mpmath import iv, libmp
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import verify_exact as ve
iv.prec = 200
DN = {1: "One", 2: "Two", 3: "Three"}
QN = {"4/5": "F", "1/2": "H", "1/1": "U"}
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


def rat(x):          # exact rational endpoints of an mpmath interval
    return tuple(Fraction(*libmp.to_rational(e)) for e in x._mpi_)


def rng(ns, sep="--"):
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}{sep}{ns[j]}")
        i = j + 1
    return ", ".join(out)


macros = {m.group(1): m.group(2) for m in
          re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'numbers.tex')).read(), re.M)}
tex = open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'R5_certified_computations.tex')).read()


def M(name):
    return macros[name]


def Nset(d, q):
    top = int(M(f"Nmax{DN[d]}{QN[q]}"))
    ex = M(f"Extra{DN[d]}{QN[q]}") if f"Extra{DN[d]}{QN[q]}" in macros else "none"
    extra = [] if ex == "none" else [int(x) for x in re.findall(r"\d+", ex)]
    return list(range(2, top + 1)) + extra, top


def exact_f(N, q, tmax):
    """d = 1: f(1..tmax) exactly, from f(t) = (q/2) (Q^(t-1))_{N-2,0}; separately written 10-line implementation."""
    n = N - 1
    v = [Fraction(0)] * n
    v[0] = Fraction(1)
    f = [None]
    for t in range(1, tmax + 1):
        f.append(q / 2 * v[n - 1])
        w = [Fraction(0)] * n
        for j in range(n):
            w[j] += (1 - q) * v[j]
            if j == 0:
                w[0] += q / 2 * v[0]              # cancelled move at the reflecting end
            else:
                w[j - 1] += q / 2 * v[j]
            if j + 1 < n:
                w[j + 1] += q / 2 * v[j]          # (the move from n-1 to the target is absorbed)
        v = w
    return f


# ====================================================================== Theorem modes (q = 4/5, 1/2)
EXC = {(1, "4/5", 4), (1, "1/2", 3), (1, "1/2", 4)}
coverage = {}
for q in ("4/5", "1/2", "1/1"):
    for d in (1, 2, 3):
        tag = f"d{d}_q{q.replace('/', '-')}"
        rows = load(f"modes_{tag}.jsonl")
        Ns, top = Nset(d, q)
        check(f"[{tag}] a certificate exists for every N of the stated set ({len(Ns)} values, 2..{top} + extras)",
              all(N in rows for N in Ns) and set(rows) == set(Ns), f"file has {len(rows)} lines")
        check(f"[{tag}] t0 = d(N-1) for every N", all(rows[N]["t0"] == d * (N - 1) for N in Ns))
        pk, px, ex, rf = load(f"packed_{tag}.jsonl"), load(f"packedexact_{tag}.jsonl"), load(f"exactcheck_{tag}.jsonl"), load(f"refcheck_{tag}.jsonl")
        keys = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T")
        pk_ok = [N for N in Ns if N in pk and pk[N].get("agrees_with_generator") and all(pk[N][k] == rows[N][k] for k in keys)]
        pk_bad = [N for N in pk if N in rows and N not in pk_ok]
        check(f"[{tag}] verify_packed.py: no disagreement with the generator", not pk_bad, f"verified N = {rng(pk_ok)}")
        check(f"[{tag}] exact programs: no disagreement with the generator, nor with each other",
              all(c.get("agrees_with_generator") for c in px.values()) and all(r["agrees_with_certificate"] for r in ex.values())
              and all(ex[N]["argmax"] == px[N]["maximisers_of_lower_bound"] and ex[N]["T_tail"] == px[N]["T_tail"]
                      and ex[N]["strictly_unimodal"] == px[N]["unimodal_certified"] for N in ex if N in px),
              f"exact N = {rng(set(px) | set(ex))}")
        strip = lambda c: {k: v for k, v in c.items() if k not in ("seconds", "engine")}
        check(f"[{tag}] fpcore.py: output identical to the generator's wherever run", all(strip(rf[N]) == strip(rows[N]) for N in rf),
              f"N = {rng(rf)}")
        coverage[tag] = dict(certified=rng(Ns), packed=rng(pk_ok), exact=rng(set(px) | set(ex)), fpcore=rng(rf),
                             not_packed=rng(set(Ns) - set(pk_ok)))
        if q != "1/1":
            normal = [N for N in Ns if (d, q, N) not in EXC]
            check(f"[{tag}] (b) strictly unimodal with unique certified mode, for every non-exceptional N",
                  all(rows[N]["mode_certified"] and rows[N]["unimodal_certified"] and rows[N]["certified_local_maxima_upto_T"] == 1
                      and rows[N]["undecided_adjacent_pairs"] == 0 for N in normal))
            check(f"[{tag}] (c) tail time T equals the mode, for every non-exceptional N",
                  all(rows[N]["T_tail"] == rows[N]["mode"] for N in normal))
            check(f"[{tag}] (c) a ratio gamma < 1 is recorded for every packed-verified N",
                  all(Fraction(pk[N]["one_minus_gamma_lower"]) > 0 for N in pk_ok),
                  f"min(1-gamma) >= {float(min(Fraction(pk[N]['one_minus_gamma_lower']) for N in pk_ok)):.3g}" if pk_ok else "")
            for (dd, qq, N) in sorted(EXC):
                if (dd, qq) == (d, q):
                    check(f"[{tag}] exceptional N={N}: the rounded programs do not certify strict unimodality",
                          not rows[N]["unimodal_certified"] and rows[N]["undecided_adjacent_pairs"] == 1)

# Corollary (q-scaling)
for d in (1, 2, 3):
    A, B = load(f"modes_d{d}_q4-5.jsonl"), load(f"modes_d{d}_q1-2.jsonl")
    top, lo, hi = int(M(f"QsN{DN[d]}")), Fraction(M(f"QsLo{DN[d]}")), Fraction(M(f"QsHi{DN[d]}"))
    vals = [Fraction(4, 5) * A[N]["mode"] - Fraction(1, 2) * B[N]["mode"] for N in range(2, top + 1)]
    check(f"Corollary q-scaling, d = {d}: {lo} <= (4/5) t*(4/5) - (1/2) t*(1/2) <= {hi} for all 2 <= N <= {top}, both bounds attained",
          min(vals) == lo and max(vals) == hi and top == min(Nset(d, "4/5")[1], Nset(d, "1/2")[1]))

# the three exceptional chains, recomputed here from scratch in exact arithmetic
f = exact_f(4, Fraction(4, 5), 40)
check("(d) (1,4/5,4): 0 = f(2) < f(3) = f(4) < f(5) > f(6) > ... (checked to t = 40)",
      f[1] == f[2] == 0 < f[3] == f[4] < f[5] and all(f[t] > f[t + 1] for t in range(5, 40)), f"f(3)=f(4)={f[3]}, f(5)={f[5]}")
f = exact_f(3, Fraction(1, 2), 40)
check("(d) (1,1/2,3): 0 = f(1) < f(2) < f(3) = f(4) > f(5) > ... (checked to t = 40)",
      f[1] == 0 < f[2] < f[3] == f[4] and all(f[t] > f[t + 1] for t in range(4, 40)), f"f(3)=f(4)={f[3]}")
f = exact_f(4, Fraction(1, 2), 40)
check("(d) (1,1/2,4): strictly increasing on 2 <= t <= 7, f(7) = f(8), strictly decreasing afterwards (to t = 40)",
      f[1] == f[2] == 0 and all(f[t] < f[t + 1] for t in range(2, 7)) and f[7] == f[8] and all(f[t] > f[t + 1] for t in range(8, 40)),
      f"f(7)=f(8)={f[7]}")
for (d, q, N, argmax, T, eq) in ((1, "4/5", 4, [5], 5, [3]), (1, "1/2", 3, [3, 4], 4, [3]), (1, "1/2", 4, [7, 8], 8, [7])):
    tag = f"d{d}_q{q.replace('/', '-')}"
    px, ex, rows = load(f"packedexact_{tag}.jsonl"), load(f"exactcheck_{tag}.jsonl"), load(f"modes_{tag}.jsonl")
    check(f"(d) ({d},{q},{N}): both exact programs give maximisers {argmax}, tail time {T}; the generator's T is {T}",
          N in px and N in ex and px[N]["maximisers_of_lower_bound"] == argmax == ex[N]["argmax"] and px[N]["T_tail"] == T == ex[N]["T_tail"]
          and px[N]["equal_adjacent_pairs_from_t0"] == eq and px[N]["certified_local_maxima_upto_T"] == (1 if len(argmax) == 1 else 0)
          and rows[N]["T_tail"] == T)

# ====================================================================== Theorem modesU (q = 1)
q1 = {}
txt = open(_os.path.join(_R, 'proofs', 'R5_certified_computations', 'text_q1.tex')).read()
for d in (1, 2, 3):
    tag = f"d{d}_q1-1"
    rows, px, ex = load(f"modes_{tag}.jsonl"), load(f"packedexact_{tag}.jsonl"), load(f"exactcheck_{tag}.jsonl")
    Ns, top = Nset(d, "1/1")
    notc = [N for N in Ns if not rows[N]["mode_certified"]]
    exact_arg = {N: (px[N]["maximisers_of_lower_bound"] if N in px else ex[N]["argmax"]) for N in notc if N in px or N in ex}
    check(f"[{tag}] every N without a certified unique mode is settled exactly: maximum attained twice, 'mode' is the smaller time",
          all(N in exact_arg and len(exact_arg[N]) == 2 and rows[N]["mode"] == min(exact_arg[N]) for N in notc),
          f"N = {notc}: {exact_arg}")
    uni = [N for N in Ns if rows[N]["unimodal_certified"]]
    multi = [N for N in Ns if rows[N]["certified_local_maxima_upto_T"] >= 2]
    und = sorted(set(Ns) - set(uni) - set(multi))
    und_ok = True
    for N in und:                      # recomputed here with the exact rational program: weakly unimodal with a tie
        fx, Tx = ve.exact_pmf(d, N, Fraction(1)) if N <= 12 else (None, None)
        if fx is None:
            und_ok = False
            continue
        core = [("<" if fx[t] < fx[t + 1] else ">") for t in range(1, Tx + 1) if fx[t] != fx[t + 1]]
        ties_ = [t for t in range(1, Tx + 1) if fx[t] == fx[t + 1] and fx[t] > 0]
        und_ok = und_ok and bool(ties_) and "".join(core) == "<" * core.count("<") + ">" * core.count(">")
        und_ok = und_ok and all(f"$f({t})=f({t + 1})$" in txt for t in ties_)
    check(f"[{tag}] every N is classified: strictly unimodal / >= 2 certified strict local maxima / weakly unimodal with an exact tie",
          und_ok, f"unimodal: {rng(uni) or 'none'}; multimodal: {rng(multi) or 'none'}; tie only: {rng(und) or 'none'}")
    check(f"[{tag}] the generated text of the theorem lists these sets",
          (not uni or "{" + rng(uni) + "\\}" in txt) and (not multi or "{" + rng(multi) + "\\}" in txt)
          and all(f"$N={N}$" in txt for N in und))
    undp = {N: rows[N]["undecided_adjacent_pairs"] for N in Ns if rows[N]["undecided_adjacent_pairs"]}
    eqx = {}
    for N in undp:                     # every undecided relation must be settled exactly (equal values listed in the text)
        if N in px:
            eqx[N] = px[N]["equal_adjacent_pairs_from_t0"]
        elif N <= 12:
            fx, Tx = ve.exact_pmf(d, N, Fraction(1))
            eqx[N] = [t for t in range(1, Tx + 1) if fx[t] == fx[t + 1] and fx[t] > 0]
    check(f"[{tag}] every relation left undecided is settled exactly; the text lists every equality f(t) = f(t+1) > 0 and states "
          f"f(t) != f(t+1) otherwise",
          all(N in eqx and len(eqx[N]) >= 1 for N in undp) and all(f"$N={N}$ (" + ", ".join(f"$f({t})=f({t + 1})$" for t in eqx[N]) + ")" in txt
                                                                  for N in undp)
          and ("f(t)\\ne f(t+1)" in txt) and "UNDECIDED" not in txt,
          f"equalities: {eqx}")
    q1[tag] = dict(strictly_unimodal=rng(uni), multimodal=rng(multi), tie_only=rng(und), two_maximisers={str(k): v for k, v in exact_arg.items()},
                   max_certified_local_maxima=max(rows[N]["certified_local_maxima_upto_T"] for N in Ns))

# ====================================================================== Theorem cf (closed form), recomputed with mpmath.iv
mf = {d: {N: (frac(r["tau_lo"]), frac(r["tau_hi"])) for N, r in load(f"mfpt_d{d}.jsonl").items()} for d in (1, 2, 3)}
check("MFPT enclosures: 0 < lower <= upper, relative width < 1e-28, and for d = 1 they contain N(N-1)",
      all(0 < lo <= hi and (hi - lo) / lo < Fraction(1, 10 ** 28) for d in mf for lo, hi in mf[d].values())
      and all(lo <= N * (N - 1) <= hi for N, (lo, hi) in mf[1].items()))


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
        rows = load(f"modes_{tag}.jsonl")
        key = f"{DN[d]}{QN[q]}"
        top = int(M(f"EpsN{key}"))
        eps = Fraction(M(f"Eps{key}"))
        errs = {N: rel_err(d, N, Fraction(q), rows[N]["mode"]) for N in range(10, top + 1)}
        check(f"[{tag}] all N in 10..{top} have a certified unique mode and an MFPT enclosure",
              all(rows[N]["mode_certified"] for N in range(10, top + 1)) and top <= int(M(f"Nmax{key}")))
        if d == 1:
            lo = Fraction(M(f"EpsLo{key}"))
            check(f"[{tag}] (c) {lo} t* <= t_cf - t* <= {eps} t* for all 10 <= N <= {top}  (mpmath.iv)",
                  all(lo <= a and b <= eps for a, b in errs.values()),
                  f"range [{float(min(a for a, b in errs.values())):.6f}, {float(max(b for a, b in errs.values())):.6f}]")
        else:
            worst = max(errs, key=lambda N: max(abs(errs[N][0]), abs(errs[N][1])))
            check(f"[{tag}] |t* - t_cf| <= {eps} t* for all 10 <= N <= {top}  (mpmath.iv)",
                  all(-eps <= a and b <= eps for a, b in errs.values()),
                  f"worst N = {worst}: {float(max(abs(errs[worst][0]), abs(errs[worst][1]))):.6f}")
        stored = load(f"closedform_{tag}.jsonl")
        check(f"[{tag}] the Arb enclosures stored by closed_form.py intersect the mpmath.iv ones for every N",
              all(not (Fraction(stored[N]["rel_err_lo"]) > b or Fraction(stored[N]["rel_err_hi"]) < a) for N, (a, b) in errs.items()))
        cf_out[tag] = {str(N): [float(a), float(b)] for N, (a, b) in errs.items() if N in (10, 20, 50, 100, top)}
        Ns, _ = Nset(d, q)
        for N in [n for n in Ns if n > top]:
            a, b = rel_err(d, N, Fraction(q), rows[N]["mode"])
            cf_out[tag][str(N)] = [float(a), float(b)]
            check(f"[{tag}] isolated N = {N}: relative error of the closed form in [{float(a):.6f}, {float(b):.6f}]", a <= b)

# ====================================================================== Theorem tau and Theorem 1dlaw
cj = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))
tlo, thi = frac(cj["tau_star"]["lo"]), frac(cj["tau_star"]["hi"])


def B_iv(fr):
    tau = ivq(fr)
    b = [(2 * n + 1) * (iv.mpf((2 * n + 1) ** 2) / 4 - 3 * tau / 2) * iv.exp(-iv.mpf(n * (n + 1)) / tau) for n in (1, 2, 3)]
    base = iv.mpf(1) / 4 - 3 * tau / 2
    return base - (b[0] - b[1] + b[2]), base - (b[0] - b[1])


m = re.search(r"\n([0-9.]+)<\\tau\^\*<([0-9.]+),", tex)
k = re.search(r"\\kappa_1:=2\\tau\^\*\\in\\bigl\(([0-9.]+),\\ ([0-9.]+)\\bigr\)", tex)
check("Theorem tau: B(tau_-) > 0 and B(tau_+) < 0 at the stored rationals (mpmath.iv, 200 bits)",
      B_iv(tlo)[0].a > 0 and B_iv(thi)[1].b < 0 and 0 < tlo < thi < Fraction(1, 4))
check("Theorem tau: the decimal bounds printed for tau* and kappa_1 contain [tau_-, tau_+] and [2 tau_-, 2 tau_+]",
      bool(m and k) and Fraction(m.group(1)) <= tlo and thi <= Fraction(m.group(2))
      and Fraction(k.group(1)) <= 2 * tlo and 2 * thi <= Fraction(k.group(2)),
      f"width of the kappa_1 enclosure {float(2 * (thi - tlo)):.2e}")
kap = (2 * tlo, 2 * thi)
law = {}
for q in ("4/5", "1/2"):
    tag = f"d1_q{q.replace('/', '-')}"
    rows = load(f"modes_{tag}.jsonl")
    Ns, top = Nset(1, q)
    qq = Fraction(q)
    dlo, dhi, c = Fraction(M(f"DeltaLo{QN[q]}")), Fraction(M(f"DeltaHi{QN[q]}")), Fraction(M(f"RatioConst{QN[q]}"))
    y = lambda N, kk: kk * Fraction(2 * N - 1, 2) ** 2 / qq
    ok_a = all(dlo < rows[N]["mode"] - y(N, kap[1]) and rows[N]["mode"] - y(N, kap[0]) < dhi for N in range(10, top + 1))
    check(f"[{tag}] {dlo} < t* - kappa_1 (N-1/2)^2/q < {dhi} for all 10 <= N <= {top}, and the interval is shorter than 1",
          ok_a and dhi - dlo < 1, f"length {float(dhi - dlo):.3f}")
    check(f"[{tag}] t* = floor(kappa_1 (N-1/2)^2/q + ({dhi})) for all 10 <= N <= {top} (both endpoints of the kappa_1 enclosure)",
          all((y(N, kk) + dhi).__floor__() == rows[N]["mode"] for N in range(10, top + 1) for kk in kap))
    check(f"[{tag}] |t*/E T - kappa_1| <= {c}/(N(N-1)) for all 10 <= N <= {top}",
          all(abs(rows[N]["mode"] * qq / (N * (N - 1)) - kk) * N * (N - 1) <= c for N in range(10, top + 1) for kk in kap))
    extra = [N for N in Ns if N > top]
    if extra:
        check(f"[{tag}] the isolated N = {extra} satisfy the same two-sided bound",
              all(dlo < rows[N]["mode"] - y(N, kap[1]) and rows[N]["mode"] - y(N, kap[0]) < dhi for N in extra))
    law[tag] = dict(delta_lo=str(dlo), delta_hi=str(dhi), ratio_const=str(c), N_max=top)

# ====================================================================== Theorem C2 and Theorem rem (mpmath.iv)
g14 = iv.gamma(iv.mpf(1) / 4)
varpi = g14 ** 2 / (2 * iv.sqrt(2 * iv.pi))
C2 = 8 / iv.pi * (iv.euler + iv.log(4 * iv.sqrt(2) / varpi) - iv.mpf(1) / 2 - iv.pi / 4)
C2b = 8 / iv.pi * (iv.euler + 4 * iv.log(2) + iv.log(iv.pi) / 2 - 2 * iv.log(g14)) - 2 - 4 / iv.pi
c2lit = re.search(r"C_2-([0-9.]+)\\bigr\|<10\^\{-50\}", tex)
a, b = rat(C2)
check("Theorem C2: |C_2 - printed decimal| < 1e-50, and the two closed forms agree  (mpmath.iv)",
      bool(c2lit) and abs(a - Fraction(c2lit.group(1))) < Fraction(1, 10 ** 50) and abs(b - Fraction(c2lit.group(1))) < Fraction(1, 10 ** 50)
      and not (C2b.a > C2.b or C2b.b < C2.a), f"C_2 in [{iv.mpf(C2).a}, ...]"[:60])
NA = int(M("RemNA"))
rem = {}
for N in range(2, NA + 1):
    lo, hi = mf[2][N]
    rem[N] = rat(iv.mpf([ivq(lo).a, ivq(hi).b]) - 8 / iv.pi * N * N * iv.log(N) - C2 * N * N)
check(f"Theorem rem: {M('RemLo')} < r_N < {M('RemHiA')} for all 2 <= N <= {NA}  (mpmath.iv)",
      all(Fraction(M("RemLo")) < x and yv < Fraction(M("RemHiA")) for x, yv in rem.values()),
      f"r_2 = {float(rem[2][0]):.8f}, r_{NA} = {float(rem[NA][1]):.8f}")
check(f"Theorem rem: r_N strictly increasing on 2 <= N <= {NA}  (mpmath.iv)", all(rem[N][1] < rem[N + 1][0] for N in range(2, NA)))

rj = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'mfpt2d_remainder.json')))
C0 = varpi ** 4 / (9 * iv.pi ** 2)
if rj.get("via_single_sum") and "RemHiB" in macros:
    v = rj["via_single_sum"]
    check(f"Proposition rem (conditional on the single-sum formula): stored bound for {v['N_min']} <= N <= {v['N_max']} is below "
          f"{M('RemHiB')}, which is below C_0 = varpi^4/(9 pi^2); monotone; single sum agrees with the linear system on the overlap",
          Fraction(v["max_remainder_upper"]) < Fraction(M("RemHiB")) < rat(C0)[0] and v["strictly_increasing_in_N"]
          and v["N_min"] == NA + 1 and v["N_max"] == int(M("RemNB")) and rj["single_sum_agrees_with_linear_system_on_overlap"],
          f"C_0 >= {float(rat(C0)[0]):.10f}")

if rj.get("via_single_sum"):
    v = rj["via_single_sum"]
    check(f"Proposition rem: junction r_{NA} < r_{NA + 1} (r_{NA}: mpmath.iv from the linear-system enclosure; r_{NA + 1}: stored lower "
          f"bound from the single sum, the minimum of the conditional range)",
          v["at_N"] == v["N_min"] == NA + 1 and rem[NA][1] < Fraction(v["min_remainder_lower"]),
          f"sup r_{NA} <= {float(rem[NA][1]):.12f} < {v['min_remainder_lower']}")

# ====================================================================== remark on log-concavity
f = exact_f(3, Fraction(4, 5), 8)
g = exact_f(4, Fraction(4, 5), 8)
check("Remark: q = 4/5 is not log-concave: f(3)^2 < f(2) f(4) for N = 3 and f(4)^2 < f(3) f(5) for N = 4 (exact)",
      f[3] ** 2 < f[2] * f[4] and g[4] ** 2 < g[3] * g[5])
ok3 = True
for q in (Fraction(4, 5), Fraction(3, 4), Fraction(77, 100), Fraction(1), Fraction(1, 2)):
    f = exact_f(3, q, 42)
    det = 1 - Fraction(3, 2) * q + q * q / 4
    ok3 &= all(f[t] ** 2 - (f[t - 1] if t > 1 else 0) * f[t + 1] == q ** 4 / 16 * det ** (t - 2) for t in range(2, 41))
    ok3 &= [t for t in range(2, 41) if f[t] ** 2 < f[t - 1] * f[t + 1]] == ([t for t in range(3, 41, 2)] if (3 - q) ** 2 < 5 else [])
check("Remark lc, N = 3: f(t)^2 - f(t-1) f(t+1) = (q^4/16) det(Q)^(t-2) for 2 <= t <= 40, failures exactly at odd t >= 3 iff q > 3 - sqrt 5 "
      "(exact, five values of q)", ok3)
g = exact_f(4, Fraction(4, 5), 402)
check("Remark lc, N = 4, q = 4/5: f(t)^2 < f(t-1) f(t+1) for t = 4 and for no other 2 <= t <= 400 (exact)",
      [t for t in range(2, 401) if g[t] ** 2 < g[t - 1] * g[t + 1]] == [4])

# ====================================================================== manifest
lines = []
for p in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_rigorous_certified', '*')) + glob.glob(_os.path.join(_R, 'code', 'msc_rigorous_certified', '*')) + glob.glob(_os.path.join(_R, 'code', 'msc_rigorous_certified_c128', '*'))):
    if os.path.isfile(p) and os.path.basename(p) not in ("MANIFEST.sha256", "AUDIT.json", "AUDIT_EXT.json"):
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, ROOT)}")
open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'MANIFEST.sha256'), "w").write("\n".join(lines) + "\n")
out = dict(failures=failures, assertions=len(report), coverage=coverage, q1=q1, closed_form_samples=cf_out, law_1d=law, report=report)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'AUDIT.json'), "w"), indent=1)
print(f"\n{len(report)} assertions, {failures} failures")
sys.exit(1 if failures else 0)
