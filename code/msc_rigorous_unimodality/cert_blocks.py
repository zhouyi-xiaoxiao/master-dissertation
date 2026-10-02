"""
cert_blocks.py -- certified log-concavity of the increment phi_d(t) = u_d(t) - u_d(t-1) of the FREE propagator
u_d(t) = P_q^t(x0, a) of the d-dimensional box for small N (Computer-assisted Proposition 9.12 of note R4).

Geometries: CC  (corner -> opposite corner),  CEN (corner -> centre, N odd; the same u serves centre -> corner).
Activity q = 4/5 (rational QN/QD).

Spectral form (Lemma 1.1):  with nu_k = 1 - q + (q/d) sum_i cos(k_i pi/N)  and one-dimensional weights
    CC :  w_k = (-1)^k c_k^2 cos^2(k pi/(2N)),                    k = 0..N-1
    CEN:  w_k = (-1)^(k/2) c_k^2 cos(k pi/(2N))  (k even), 0 (k odd)       (c_0^2 = 1/N, c_k^2 = 2/N)
    a(t) := phi_d(t+1) = sum over multisets K = {k_1<=...<=k_d} of  mult(K) prod_i w_{k_i} (nu_K - 1) nu_K^t ,  t >= 0.
a(t) = 0 for t < D-1 (D = graph distance), a(D-1) > 0.
Certified:  a(t) > 0 for all t >= D-1  and  a(t)^2 - a(t-1) a(t+1) >= 0 for all t >= D.
  tail (Lemma 9.5, list form): the two leading modes are K1 = {0,..,0,k*}, K2 = {0,..,0,k*,k*} (k* = 1 for CC, 2 for CEN);
     every other mode has |nu| < nu_K2; coefficients C_K1 > 0 > C_K2; and at t0:  2 L_{t0} > Tot_{t0} and
     C_1 nu_1^t0 > sum_{others} |C_i| |nu_i|^t0.
  window D-1 <= t < t0: Arb spectral sums; when N^d <= EXACT_SITES the values are ALSO computed in exact integer
     arithmetic (lazy walk on the box) and compared (value in ball, exact sign of the Casorati determinant).

Usage: python cert_blocks.py            (runs the list CASES)  ->  data/cert_blocks.jsonl
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import itertools
import json
import os
import sys
import time
from math import factorial

import numpy as np
from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
QN, QD = 4, 5
PREC = 600
EXACT_SITES = 20000

CASES = []
for N in range(3, 9):
    CASES += [("CC", N, 2), ("CC", N, 3)]
for N in range(9, 25, 2):
    CASES += [("CEN", N, 2), ("CEN", N, 3)]
for N in (5, 7):
    CASES += [("CEN", N, 3), ("CEN", N, 4), ("CEN", N, 5)]
CASES += [("CEN", 3, 4), ("CEN", 3, 5), ("CEN", 3, 6), ("CEN", 3, 7)]


def ipow(x, e):
    r = arb(1)
    b = x
    while e:
        if e & 1:
            r = r * b
        e >>= 1
        if e:
            b = b * b
    return r


def modes(geom, N, d):
    """list of (label, C, nu) as arb; the mode {0,...,0} (nu = 1, coefficient exactly 0) is omitted."""
    q = arb(QN) / QD
    pi = arb.pi()
    one = {}
    for k in range(N):
        ck2 = arb(1) / N if k == 0 else arb(2) / N
        if geom == "CC":
            w = ck2 * (pi * k / (2 * N)).cos() ** 2 * (1 if k % 2 == 0 else -1)
        else:
            if k % 2:
                continue
            w = ck2 * (pi * k / (2 * N)).cos() * (1 if (k // 2) % 2 == 0 else -1)
        one[k] = ((pi * k / N).cos(), w)
    out = []
    for ks in itertools.combinations_with_replacement(sorted(one), d):
        if all(k == 0 for k in ks):
            continue
        cnt = {}
        for k in ks:
            cnt[k] = cnt.get(k, 0) + 1
        mult = factorial(d)
        for v in cnt.values():
            mult //= factorial(v)
        s = arb(0)
        w = arb(mult)
        for k in ks:
            s += one[k][0]
            w *= one[k][1]
        nu = 1 - q + q * s / d
        out.append((ks, w * (nu - 1), nu))
    return out


def exact_a(geom, N, d, tmax):
    """integers Phi_t with phi(t) = Phi_t / Dn^t (Dn = 2 d QD), t = 0..tmax, by exact stepping of the free walk"""
    shape = (N,) * d
    Dn = 2 * d * QD
    nblk = np.zeros(shape, dtype=object)
    for ax in range(d):
        i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
        i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
    hold = (QD - QN) * 2 * d + QN * nblk
    v = np.zeros(shape, dtype=object)
    v[(0,) * d] = 1
    a = (N - 1,) * d if geom == "CC" else ((N - 1) // 2,) * d
    U = [int(v[a])]
    for t in range(1, tmax + 1):
        w = hold * v
        for ax in range(d):
            sa = [slice(None)] * d; sb = [slice(None)] * d
            sa[ax] = slice(1, None); sb[ax] = slice(None, -1)
            w[tuple(sa)] += QN * v[tuple(sb)]
            w[tuple(sb)] += QN * v[tuple(sa)]
        v = w
        U.append(int(v[a]))
    Phi = [0] + [U[t] - Dn * U[t - 1] for t in range(1, tmax + 1)]
    return Phi, Dn


def certify(geom, N, d):
    t_start = time.time()
    ctx.prec = PREC
    ms = modes(geom, N, d)
    kstar = 1 if geom == "CC" else 2
    D = d * (N - 1) if geom == "CC" else d * (N - 1) // 2
    rec = dict(geom=geom, N=N, d=d, q="%d/%d" % (QN, QD), D=D, n_modes=len(ms), prec=PREC)
    K1 = tuple([0] * (d - 1) + [kstar])
    K2 = tuple([0] * (d - 2) + [kstar, kstar])
    i1 = [i for i, m in enumerate(ms) if m[0] == K1][0]
    i2 = [i for i, m in enumerate(ms) if m[0] == K2][0]
    C1, nu1 = ms[i1][1], ms[i1][2]
    C2, nu2 = ms[i2][1], ms[i2][2]
    hyp = bool(C1 > 0) and bool(C2 < 0) and bool(nu1 > nu2) and bool(nu2 > 0)
    others = [m for i, m in enumerate(ms) if i not in (i1, i2)]
    for m in others:
        hyp = hyp and bool(nu2 > abs(m[2]))
    rec["tail_hypotheses"] = hyp
    if not hyp:
        rec["status"] = "FAIL: tail hypotheses"
        return rec

    def tail_ok(t0):
        # pair form: a(t)^2 - a(t-1)a(t+1) = - sum_{i<j} C_i C_j (nu_i nu_j)^(t-1) (nu_i - nu_j)^2
        den = (nu1 * nu2).sqrt()
        lst = [(C1, nu1), (C2, nu2)] + [(m[1], m[2]) for m in others]
        p = [abs(c) * ipow(abs(nu) / den, t0 - 1) for c, nu in lst]
        s0 = arb(0); s1 = arb(0); s2 = arb(0)
        for pk, (c, nu) in zip(p, lst):
            s0 += pk; s1 += pk * nu; s2 += pk * nu * nu
        tot = s0 * s2 - s1 * s1
        L = p[0] * p[1] * (nu1 - nu2) ** 2
        if not bool(2 * L - tot > 0):
            return False
        rest = abs(C2) * ipow(nu2 / nu1, t0)
        for m in others:
            rest += abs(m[1]) * ipow(abs(m[2]) / nu1, t0)
        return bool(C1 - rest > 0)

    t0 = max(D + 1, 8)
    while not tail_ok(t0):
        t0 = int(t0 * 1.15) + 2
        if t0 > 200000:
            rec["status"] = "FAIL: no tail time"
            return rec
    rec["t0"] = t0
    # window: a(t) for t = D-2 .. t0  (normalised by nu1^t)
    lst = [(m[1], m[2] / nu1) for m in ms]
    cur = [c * ipow(r, D - 2) for c, r in lst] if D >= 2 else None
    vals = {}
    for t in range(D - 2, t0 + 1):
        s = arb(0)
        for x in cur:
            s += x
        vals[t] = s
        cur = [x * r for x, (c, r) in zip(cur, lst)]
    exact = (N ** d <= EXACT_SITES)
    if exact:
        Phi, Dn = exact_a(geom, N, d, t0 + 2)       # a(t) = phi(t+1) = Phi[t+1] / Dn^(t+1)
    n_notpos = 0; n_notlc = 0; n_eq = 0; cross_ok = True
    min_margin = None; arg = None
    for t in range(D - 1, t0):
        a_t, a_m, a_p = vals[t], vals[t - 1], vals[t + 1]
        pos = bool(a_t > 0)
        Cd = a_t * a_t - a_m * a_p
        lc = bool(Cd > 0)
        if exact:
            e_t, e_m, e_p = Phi[t + 1], Phi[t], Phi[t + 2]
            # a(t) nu1^(-t) * ... compare:  Phi[t+1] / Dn^(t+1) / nu1^t  vs ball
            ex = arb(e_t) / ipow(arb(Dn), t + 1) / ipow(nu1, t)
            if not ex.overlaps(a_t):
                cross_ok = False
            if not pos and e_t > 0:
                pos = True
            # exact Casorati sign: e_t^2 - e_m e_p   (common factor Dn^(2t+2))
            sg = e_t * e_t - e_m * e_p
            if sg < 0:
                cross_ok = False
            elif sg == 0:
                n_eq += 1
                lc = True
            else:
                lc = True
        if t == D - 1:
            lc = True            # a(D-2) = 0: the inequality is trivial
            if exact and Phi[D - 1] != 0:
                cross_ok = False
        if not pos:
            n_notpos += 1
        if not lc:
            n_notlc += 1
        elif t > D - 1:
            mg = float((Cd / (a_t * a_t)).mid())
            if min_margin is None or mg < min_margin:
                min_margin, arg = mg, t
    # support: a(D-2) must vanish (exactly, combinatorially: no path of length D-1); check ball contains 0
    rec.update(window=[D - 1, t0 - 1], n_not_positive=n_notpos, n_not_logconcave=n_notlc, exact_equalities=n_eq,
               exact_crosscheck=exact, exact_crosscheck_ok=cross_ok, a_before_support_contains_zero=bool(vals[D - 2].contains(0)) if D >= 2 else None,
               min_rel_margin=min_margin, argmin_t=arg, seconds=round(time.time() - t_start, 1))
    ok = n_notpos == 0 and n_notlc == 0 and cross_ok
    rec["status"] = "LOGCONCAVE (certified)" if ok else "NOT CERTIFIED"
    return rec


def main():
    out = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_blocks.jsonl')
    cases = CASES
    if len(sys.argv) > 1:
        cases = [(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))]
    for geom, N, d in cases:
        rec = certify(geom, N, d)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
