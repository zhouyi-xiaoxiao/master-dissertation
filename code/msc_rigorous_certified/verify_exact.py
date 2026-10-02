"""verify_exact.py -- separately written exact check of mode certificates for small N.

No numpy, no stencil, no rounding: the chain is built site by site from the model rules and
iterated with exact rationals (fractions.Fraction).  For each certificate line it re-derives
  t0, the mode, the tail time T, strict unimodality, the number of strict local maxima on [1,T]
and checks that the three stored enclosures contain the exact values.

usage: verify_exact.py certs/modes_d2_q4-5.jsonl Nmax     -> certs/exactcheck_d2_q4-5.jsonl
"""
import json, sys
from fractions import Fraction
from itertools import product


def exact_pmf(d, N, q):
    a, x0 = (N - 1,) * d, (0,) * d
    out = {}                                   # out[y][z] = P(y -> z) for y != a
    for y in product(range(N), repeat=d):
        if y == a:
            continue
        row = {y: 1 - q}
        for ax in range(d):
            for sgn in (-1, 1):
                z = list(y)
                z[ax] += sgn
                z = tuple(z) if 0 <= z[ax] < N else y      # move off the lattice is cancelled
                row[z] = row.get(z, 0) + q / (2 * d)
        out[y] = {z: p for z, p in row.items() if p != 0}
    v = {x0: Fraction(1)}                      # v_t(y) = P(X_{t-1} = y, T > t-1)
    f = [None]                                 # f[t] = P(T = t)
    while True:
        f.append(sum(p * out[y].get(a, 0) for y, p in v.items()))
        w = {}
        for y, p in v.items():
            for z, pz in out[y].items():
                if z != a:
                    w[z] = w.get(z, 0) + p * pz
        tail = all(w.get(y, 0) < v.get(y, 0) for y in out)   # v_{t+1} < v_t at every non-target site
        if tail:
            f.append(sum(p * out[y].get(a, 0) for y, p in w.items()))
            return f, len(f) - 2               # f[1..T+1], T
        v = w


def analyse(f, T):
    t0 = next(t for t in range(1, T + 2) if f[t] > 0)
    fmax = max(f[1:T + 1])
    argmax = [t for t in range(1, T + 1) if f[t] == fmax]
    ts = argmax[0]
    unimodal = (len(argmax) == 1 and all(f[t] == 0 for t in range(1, t0))
                and all(f[t] < f[t + 1] for t in range(t0, ts)) and all(f[t] > f[t + 1] for t in range(ts, T + 1)))
    g = [Fraction(0)] + f[1:]
    nmax = sum(1 for t in range(1, T + 1) if g[t] > g[t + 1] and (t == 1 or g[t - 1] < g[t]))
    return t0, argmax, unimodal, nmax


if __name__ == "__main__":
    import os
    path, Nmax = sys.argv[1], int(sys.argv[2])
    outp = path.replace("modes_", "exactcheck_")
    results = []
    bad = 0
    for line in open(path):
        c = json.loads(line)
        if c["N"] > Nmax:
            continue
        qn, qd = (int(x) for x in c["q"].split("/"))
        q = Fraction(qn, qd)
        f, T = exact_pmf(c["d"], c["N"], q)
        t0, argmax, unimodal, nmax = analyse(f, T)
        unique = len(argmax) == 1
        # The certificate's claims are one-sided; each claim made must be true.
        ok = (t0 == c["t0"] and c["mode"] in argmax and T <= c["T_tail"]
              and (unique and argmax[0] == c["mode"] if c["mode_certified"] else True)
              and (unimodal if c["unimodal_certified"] else True)
              and nmax >= c["certified_local_maxima_upto_T"])
        if T == c["T_tail"]:      # same horizon: the certified count of peaks cannot exceed the exact one
            ok = ok and c["certified_local_maxima_upto_T"] <= nmax
        scale = Fraction(c["D"], c["cm"]) * 2 ** c["P"]          # 2^P s_t = 2^P (D/cm) f(t)
        for t, (lo, hi) in c["F_enclosures"].items():
            if int(t) < len(f):
                ok = ok and int(lo) <= scale * f[int(t)] <= int(hi)
        bad += not ok
        fs = f[argmax[0]]
        results.append(dict(d=c["d"], N=c["N"], q=c["q"], t0=t0, argmax=argmax, mode_unique=unique, T_tail=T,
                            strictly_unimodal=unimodal, strict_local_maxima_upto_T=nmax,
                            f_mode=f"{fs.numerator}/{fs.denominator}" if len(str(fs.denominator)) < 400 else "(too long)",
                            certificate=dict(mode=c["mode"], mode_certified=c["mode_certified"], unimodal_certified=c["unimodal_certified"],
                                             T_tail=c["T_tail"], local_maxima=c["certified_local_maxima_upto_T"]),
                            agrees_with_certificate=bool(ok)))
        print(f"d={c['d']} q={c['q']} N={c['N']}: t0={t0} argmax={argmax} T={T} unimodal={unimodal} nmax={nmax} "
              f"f(mode)~{float(fs):.12g} | cert: mode={c['mode']} certified={c['mode_certified']} unimodal={c['unimodal_certified']} "
              f"T={c['T_tail']} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    with open(outp, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print("ALL OK" if bad == 0 else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)
