"""closed_form.py -- certified relative error of the closed form (Theorem 21 of note R5)

    t_cf = MFPT * ln(2d X) / (X + 2d - 1),   X = mu_1 * MFPT,   mu_1 = (q/d)(1 - cos(pi/N)),

against the certified integer mode t* (certs/modes_*.jsonl), using the rigorous rational enclosure of
tau_N = q*MFPT (certs/mfpt_d*.jsonl).  Note X = (tau_N/d)(1 - cos(pi/N)) does not depend on q.

Trusted base: Arb ball arithmetic (python-flint) for cos, log, pi; Python integers/Fractions elsewhere.
Every enclosure is recomputed with mpmath.iv and the two are required to overlap.
For d = 1 the script also certifies  delta_N = t* - (kappa_1/q)(N - 1/2)^2  (kappa_1 from certs/constants.json).

usage: closed_form.py            -> certs/closedform_d{d}_q{q}.jsonl, certs/closedform_summary.json
       closed_form.py c128       -> the same for the certificates certs/c128_*.jsonl of the C engine (extended ranges):
                                    certs/closedform_c128_d{d}_q{q}.jsonl, certs/closedform_summary_c128.json
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, json, os, sys
from fractions import Fraction
from flint import arb, ctx, fmpq
from mpmath import iv, libmp

HERE = os.path.dirname(os.path.abspath(__file__))
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
ctx.prec = 256
iv.prec = 256
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "modes"
OUTTAG = "" if PREFIX == "modes" else PREFIX + "_"
STARTS = (10, 20, 35, 50, 100) if PREFIX == "modes" else (10, 20, 35, 50, 60, 100, 160, 200, 300, 600)


def frac(s):
    n, dn = s.split("/")
    return Fraction(int(n), int(dn))


def ball(lo, hi):
    """Arb ball containing the rational interval [lo, hi]."""
    mid, rad = (lo + hi) / 2, (hi - lo) / 2
    return arb(fmpq(mid.numerator, mid.denominator)) + arb(0, 1) * arb(fmpq(rad.numerator, rad.denominator))


def bounds(x):
    """Exact rational lower/upper bounds of an Arb ball."""
    out = []
    for e in (x.lower(), x.upper()):
        m, ex = e.man_exp()
        out.append(Fraction(int(m)) * Fraction(2) ** int(ex))
    return out


def dec(fr, digits, up):
    sc = 10 ** digits
    n = fr.numerator * sc
    v = -((-n) // fr.denominator) if up else n // fr.denominator
    sgn = "-" if v < 0 else ""
    s = str(abs(v)).rjust(digits + 1, "0")
    return sgn + s[:-digits] + "." + s[-digits:]


def tcf_arb(d, N, q, tlo, thi):
    tau = ball(tlo, thi)
    X = tau / d * (1 - (arb.pi() / N).cos())
    return tau / arb(fmpq(q.numerator, q.denominator)) * (2 * d * X).log() / (X + 2 * d - 1), X


def tcf_iv(d, N, q, tlo, thi):
    tau = iv.mpf([iv.mpf(tlo.numerator) / tlo.denominator, iv.mpf(thi.numerator) / thi.denominator])
    tau = iv.mpf([tau.a, tau.b])
    X = tau / d * (1 - iv.cos(iv.pi / N))
    return tau / (iv.mpf(q.numerator) / q.denominator) * iv.log(2 * d * X) / (X + 2 * d - 1)


mf = {}
for d in (1, 2, 3):
    p = os.path.join(CERT, f"mfpt_d{d}.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            mf[(d, r["N"])] = (frac(r["tau_lo"]), frac(r["tau_hi"]))
kap = None
cpath = _os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')
if os.path.exists(cpath):
    cj = json.load(open(cpath))
    kap = (frac(cj["tau_star"]["lo"]) * 2, frac(cj["tau_star"]["hi"]) * 2)

summary = {}
for path in sorted(glob.glob(os.path.join(CERT, PREFIX + "_d*_q*.jsonl"))):
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda c: c["N"])
    tag = os.path.basename(path)[len(PREFIX) + 1:-len(".jsonl")]
    outp = os.path.join(CERT, f"closedform_{OUTTAG}{tag}.jsonl")
    recs = []
    for c in rows:
        d, N = c["d"], c["N"]
        if (d, N) not in mf or not c["mode_certified"]:
            continue
        q = frac(c["q"])
        tlo, thi = mf[(d, N)]
        ts = c["mode"]
        t, X = tcf_arb(d, N, q, tlo, thi)
        e = (t - ts) / ts
        elo, ehi = bounds(e)
        tl, th = bounds(t)
        ti = tcf_iv(d, N, q, tlo, thi)
        ei = (ti - ts) / ts
        # the two independent enclosures must overlap
        ia, ib = (Fraction(*libmp.to_rational(x)) for x in ei._mpi_)
        assert not (elo > ib or ehi < ia), (d, N, elo, ehi, ei)
        Xl, Xh = bounds(X)
        mlo, mhi = tlo / q, thi / q
        rec = dict(d=d, N=N, q=c["q"], mode=ts, unimodal_certified=c["unimodal_certified"],
                   mfpt_lo=dec(mlo, 12, False), mfpt_hi=dec(mhi, 12, True),
                   mode_over_mfpt_lo=dec(Fraction(ts) / mhi, 15, False), mode_over_mfpt_hi=dec(Fraction(ts) / mlo, 15, True),
                   X_lo=dec(Xl, 12, False), X_hi=dec(Xh, 12, True),
                   t_cf_lo=dec(tl, 9, False), t_cf_hi=dec(th, 9, True),
                   rel_err_lo=dec(elo, 15, False), rel_err_hi=dec(ehi, 15, True),
                   abs_rel_err_upper=dec(max(abs(elo), abs(ehi)), 15, True),
                   abs_rel_err_lower=dec(Fraction(0) if elo <= 0 <= ehi else min(abs(elo), abs(ehi)), 15, False))
        rec["_e"] = (elo, ehi)
        if d == 1 and kap is not None:
            a = Fraction(2 * N - 1, 2) ** 2 / q
            dl, dh = ts - kap[1] * a, ts - kap[0] * a
            rec["delta_1D_lo"], rec["delta_1D_hi"] = dec(dl, 12, False), dec(dh, 12, True)
            rec["_d"] = (dl, dh)
        recs.append(rec)
    with open(outp, "w") as fh:
        for r in recs:
            fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    if not recs:
        continue
    d = recs[0]["d"]
    allN = sorted(c["N"] for c in rows)
    have = {r["N"] for r in recs}
    # contiguous certified range: every N from 10 on must have a certified mode AND an MFPT enclosure
    top = 9
    while top + 1 in have:
        top += 1
    extra = sorted(N for N in have if N > top)
    Ns = [N for N in sorted(have) if N <= top]
    s = dict(d=d, q=recs[0]["q"], N_min=min(have), N_max=top, extra_N=extra,
             skipped_N_mode_not_certified=[c["N"] for c in rows if not c["mode_certified"]],
             skipped_N_no_mfpt_enclosure=[N for N in allN if (d, N) not in mf],
             extra={str(r["N"]): [dec(r["_e"][0], 12, False), dec(r["_e"][1], 12, True)] for r in recs if r["N"] in extra})
    for (a, b) in [(a, top) for a in STARTS if a < top] + [(2, 9)]:
        sel = [r for r in recs if a <= r["N"] <= b]
        if not sel:
            continue
        worst = max(sel, key=lambda r: max(abs(r["_e"][0]), abs(r["_e"][1])))
        best_lo = min(sel, key=lambda r: Fraction(0) if r["_e"][0] <= 0 <= r["_e"][1] else min(abs(r["_e"][0]), abs(r["_e"][1])))
        s[f"eps[{a},{b}]"] = dict(max_abs_rel_err_upper=worst["abs_rel_err_upper"], at_N=worst["N"],
                                  min_abs_rel_err_lower=best_lo["abs_rel_err_lower"], at_N_min=best_lo["N"],
                                  signed_range=[dec(min(r["_e"][0] for r in sel), 12, False), dec(max(r["_e"][1] for r in sel), 12, True)])
    if d == 1 and "_d" in recs[0]:
        for (a, b) in ((2, top), (10, top)):
            sel = [r for r in recs if a <= r["N"] <= b]
            if sel:
                s[f"delta_1D[{a},{b}]"] = [dec(min(r["_d"][0] for r in sel), 9, False), dec(max(r["_d"][1] for r in sel), 9, True)]
    summary[tag] = s
json.dump(summary, open(os.path.join(CERT, "closedform_summary.json" if PREFIX == "modes" else f"closedform_summary_{PREFIX}.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith(("eps[10", "N_", "extra", "skipped", "delta"))} for k, v in summary.items()}, indent=1))
