"""verify_packed.py -- separately written checker of the mode / unimodality certificates.

TRUSTED BASE: Python integers only (+, -, *, //, <<, >>, &, comparisons, int.to_bytes / int.from_bytes).
No numpy, no floating point, no code shared with fpcore.py / fplimb.py.

Model (Section 2 of note R5).  Lazy walk on {0..N-1}^d, each of the 2d directions with probability
q/(2d), a move off the box is cancelled, target a = (N-1,..,N-1) absorbing, start x0 = 0.
Q = M/D on the non-target sites, M = cm*(A + B) + cs*I, (cm, cs, D) integers with q/(2d) = cm/D, 1-q = cs/D.
v_1 = e_x0, v_{t+1} = Q v_t, f(t) = P(T = t) = (cm/D) * sigma_t, sigma_t = sum of v_t over the d neighbours of a.

Data layout.  The whole vector (z_i), i = x_1 + N x_2 + N^2 x_3, is ONE Python integer
Z = sum_i z_i 2^(w i)  ("fields" of w bits).  A step of the walk is a handful of shifts, ANDs and additions.

Method (different from the generator, which iterates a lower and an upper vector with floor/ceil division):
  exact phase   (t <= ts): X_t = D^(t-1) v_t exactly; field width grows as needed.
  rounded phase (t >= ts): ONE vector C_t and ONE integer b_t with   C_t <= 2^P v_t <= C_t + b_t  entrywise:
        C_ts = floor(2^P v_ts), b_ts = 1;   C_{t+1} = floor( (M C_t) * R / 2^K ),  R = floor(2^K / D),  b_{t+1} = b_t + 2.
     Proof.  W = M C_t >= 0 has entries <= D 2^P, and  W/D - 2 < W/D - D 2^(P-K) - 1 < floor(W R / 2^K) <= W/D
     because 2^K/D - 1 < R <= 2^K/D and D 2^(P-K) < 1.  Since Q >= 0 has row sums <= 1:
     C_{t+1} <= Q C_t <= 2^P v_{t+1} <= Q C_t + b_t <= C_{t+1} + 2 + b_t.
  tail time T: first t with v_{t+1} < v_t certified at EVERY non-target site (C_{t+1} + b_{t+1} < C_t, or
     X_{t+1} < D X_t in the exact phase).  By the tail lemma f is then strictly decreasing on [T, infinity),
     and f(T+s) <= gamma^s f(T) with gamma = max_y v_{T+1}(y)/v_T(y) < 1.

usage:  verify_packed.py certs/modes_d2_q4-5.jsonl [Nmin Nmax [budget_s [stride]]]  -> certs/packed_d2_q4-5.jsonl
        verify_packed.py d qn qd Nmin Nmax [budget_s]                 (no certificate to compare with)
        option --exact : never leave the exact phase (ties are then decided exactly; small N only)
Restartable (N already present in the output file are skipped).
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
from fractions import Fraction
from math import gcd

P = 112                 # fixed-point scale of the rounded phase (the generator's value, so enclosures are comparable)
K = P + 8               # precision of the reciprocal R = floor(2^K / D)
WB = (P + K) // 8 + 1   # bytes per field in the rounded phase: 8*WB > P + K
G = 44                  # leave the exact phase at the first t with floor(2^P sigma_t) >= 2^G


def coeffs(d, qn, qd):
    g = gcd(gcd(qn, 2 * d * (qd - qn)), 2 * d * qd)
    return qn // g, 2 * d * (qd - qn) // g, 2 * d * qd // g


class Box:
    """Masks and stencil for vectors packed with wb bytes per site."""

    def __init__(self, d, N, wb):
        self.d, self.N, self.wb, self.w, self.n = d, N, wb, 8 * wb, N ** d
        w, n = self.w, self.n
        ff, zz = b"\xff" * wb, bytes(wb)

        def mask(pred):         # all-ones field at every site whose index satisfies pred
            return int.from_bytes(b"".join(ff if pred(i) else zz for i in range(n)), "little")

        self.not_hi = [mask(lambda i, s=N ** ax: (i // s) % N != N - 1) for ax in range(d)]
        self.wall = [mask(lambda i, s=N ** ax: (i // s) % N in (0, N - 1)) for ax in range(d)]
        self.shift = [w * N ** ax for ax in range(d)]
        self.not_target = (1 << (w * (n - 1))) - 1                   # all bits of the non-target fields
        self.one_nt = self.not_target // ((1 << w) - 1)              # the integer 1 in every non-target field
        self.sign_nt = self.one_nt << (w - 1)                        # top bit of every non-target field
        self.nb = [n - 1 - N ** ax for ax in range(d)]               # the d neighbours of the target

    def apply_M(self, Z, cm, cs):
        """M Z.  Up-moves: fields not on the upper wall shift up; down-moves: shift down and discard what
        lands on the upper wall (it came from a lower wall of the next row); cancelled moves stay."""
        S = 0
        for ax in range(self.d):
            sh, nh = self.shift[ax], self.not_hi[ax]
            S += ((Z & nh) << sh) + ((Z >> sh) & nh) + (Z & self.wall[ax])
        return (cm * S + cs * Z) & self.not_target                   # mass entering the target is absorbed

    def nbsum(self, Z):
        m = (1 << self.w) - 1
        return sum((Z >> (self.w * i)) & m for i in self.nb)

    def unpack(self, Z):
        b, wb = Z.to_bytes(self.n * self.wb, "little"), self.wb
        return [int.from_bytes(b[i * wb:(i + 1) * wb], "little") for i in range(self.n)]

    def pack(self, vals):
        return int.from_bytes(b"".join(v.to_bytes(self.wb, "little") for v in vals), "little")

    def less_everywhere(self, A, Bv, slack, bound):
        """True iff A(y) + slack < Bv(y) at every non-target site y.  Requires 0 <= A, Bv <= bound entrywise.
        Field y of V is 2^(w-1) + (Bv(y) - A(y) - slack - 1), a number in [0, 2^w), so there is no carry
        between fields and the top bit of the field is set iff Bv(y) - A(y) - slack - 1 >= 0."""
        assert bound + slack + 1 < (1 << (self.w - 1))
        V = Bv + self.sign_nt - A - (slack + 1) * self.one_nt
        return (V & self.sign_nt) == self.sign_nt


def enclose(s, Dpow, exact):
    """(lo, hi) with lo <= 2^P s / Dpow <= hi: the exact rational twice (exact mode) or floor and ceiling."""
    if exact:
        return Fraction(s << P, Dpow), Fraction(s << P, Dpow)
    return (s << P) // Dpow, -((-(s << P)) // Dpow)


def certify(d, N, qn, qd, exact_only=False, tmax=10 ** 8):
    cm, cs, D = coeffs(d, qn, qd)
    assert D < (1 << (K - P)) and N >= 2
    lo, hi, rel = [None], [None], [None]     # lo[t] <= 2^P sigma_t <= hi[t];  rel[t]: f(t) vs f(t+1)
    # ---------------------------------------------------------------- exact phase
    box = Box(d, N, 8)
    X, t, Dpow, s_prev, ts = 1, 1, 1, None, None          # X_1 = e_x0 (site index 0); Dpow = D^(t-1)
    while True:
        s = box.nbsum(X)
        l_, h_ = enclose(s, Dpow, exact_only)
        lo.append(l_)
        hi.append(h_)
        if t > 1:
            a = D * s_prev                                  # f(t-1) vs f(t)  <=>  D s_{t-1} vs s_t
            rel.append("<" if a < s else ">" if a > s else "=")
        s_prev = s
        if not exact_only and lo[t] >> G:
            ts = t
            break
        while (Dpow * D).bit_length() > box.w - 2:          # widen the fields: entries of X_{t+1} are <= D^t
            vals = box.unpack(X)
            box = Box(d, N, box.wb + max(1, box.wb // 4))
            X = box.pack(vals)
        Xn = box.apply_M(X, cm, cs)
        if box.less_everywhere(Xn, D * X, 0, Dpow * D):     # X_{t+1} < D X_t on all non-target sites
            xs, xn = box.unpack(X), box.unpack(Xn)
            assert all(b < D * a for a, b in zip(xs[:-1], xn[:-1]))          # direct re-check, site by site
            gam = max(Fraction(b, D * a) for a, b in zip(xs[:-1], xn[:-1]))
            sn = box.nbsum(Xn)
            l_, h_ = enclose(sn, Dpow * D, exact_only)
            lo.append(l_)
            hi.append(h_)
            rel.append("<" if D * s < sn else ">" if D * s > sn else "=")
            return finish(d, N, qn, qd, lo, hi, rel, t, None, 0, gam)
        X, t, Dpow = Xn, t + 1, Dpow * D
        if t > tmax:
            raise RuntimeError("tmax reached")
    # ---------------------------------------------------------------- rounded phase
    vals = [(x << P) // Dpow for x in box.unpack(X)]
    box = Box(d, N, WB)
    C, b = box.pack(vals), 1                                # C <= 2^P v_ts <= C + 1
    R = (1 << K) // D
    low = ((1 << (box.w * box.n)) - 1) // ((1 << box.w) - 1) * ((1 << (box.w - K)) - 1)   # low w-K bits of every field
    while True:
        Cn, bn = ((box.apply_M(C, cm, cs) * R) >> K) & low, b + 2
        sn = box.nbsum(Cn)
        lo.append(sn)
        hi.append(sn + d * bn)
        rel.append("<" if hi[t] < lo[t + 1] else ">" if lo[t] > hi[t + 1] else "?")
        if rel[t] == ">" and box.less_everywhere(Cn, C, bn, 1 << P):          # C_{t+1} + b_{t+1} < C_t
            cs_, cn_ = box.unpack(C), box.unpack(Cn)
            assert all(y + bn < x for x, y in zip(cs_[:-1], cn_[:-1]))       # direct re-check, site by site
            gam = max(Fraction(y + bn, x) for x, y in zip(cs_[:-1], cn_[:-1]))
            return finish(d, N, qn, qd, lo, hi, rel, t, ts, bn, gam)
        C, b, t = Cn, bn, t + 1
        if t > tmax:
            raise RuntimeError("tmax reached")


def finish(d, N, qn, qd, lo, hi, rel, T, ts, bT, gam):
    """Verdicts from lo[1..T+1], hi[1..T+1], rel[1..T] and the tail time T (Proposition 'what a certificate proves')."""
    tstar = max(range(1, T + 1), key=lambda t: (lo[t], -t))
    mode_ok = all(lo[tstar] > hi[t] for t in range(1, T + 1) if t != tstar)
    t0 = next(t for t in range(1, T + 2) if hi[t] > 0)
    zeros_ok = all(rel[t] == "=" for t in range(1, t0 - 1))
    inc_ok = all(rel[t] == "<" for t in range(max(t0 - 1, 1), tstar))
    dec_ok = all(rel[t] == ">" for t in range(tstar, T))
    down = lambda t: t == T or rel[t] == ">"                # f(T) > f(T+1) by the tail lemma
    peaks = sum(1 for t in range(1, T + 1) if down(t) and (t == 1 or rel[t - 1] == "<"))
    g = 1 - gam                                             # 1 - gamma > 0, an exact rational
    e = 0
    while g * 10 ** e < 100:
        e += 1
    fl = lambda x: x.numerator // x.denominator             # floor / ceiling (x: int or Fraction)
    ce = lambda x: -((-x.numerator) // x.denominator)
    return dict(d=d, N=N, q=f"{qn}/{qd}", engine="packed", P=P, K=K, G=G, ts=ts, t0=t0, mode=tstar, T_tail=T,
                mode_certified=bool(mode_ok), unimodal_certified=bool(mode_ok and zeros_ok and inc_ok and dec_ok),
                certified_local_maxima_upto_T=peaks,
                maximisers_of_lower_bound=[t for t in range(1, T + 1) if lo[t] == lo[tstar]],
                undecided_adjacent_pairs=sum(1 for t in range(1, T) if rel[t] == "?"),
                equal_adjacent_pairs_from_t0=[t for t in range(t0, T) if rel[t] == "="],
                error_bound_b=bT, one_minus_gamma_lower=f"{(g * 10 ** e).__floor__()}e-{e}",
                F_enclosures={str(t): [str(fl(lo[t])), str(ce(hi[t]))]
                              for t in (tstar - 1, tstar, tstar + 1) if 1 <= t <= T + 1})


def compare(recheck, cert, exact):
    """(no_contradiction, same_verdicts).  Both programs are sound, so: equal t0; equal modes when both certify
    the mode; never 'strictly unimodal' against '>= 2 certified peaks'; intersecting enclosures (same scale).
    In exact mode this program knows the truth, so every claim of the certificate must hold exactly."""
    ok = recheck["t0"] == cert["t0"]
    if recheck["mode_certified"] and cert["mode_certified"]:
        ok = ok and recheck["mode"] == cert["mode"]
    ok = ok and not (recheck["unimodal_certified"] and cert["certified_local_maxima_upto_T"] >= 2)
    ok = ok and not (cert["unimodal_certified"] and recheck["certified_local_maxima_upto_T"] >= 2)
    if exact:
        ok = ok and cert["mode"] in recheck["maximisers_of_lower_bound"]
        ok = ok and (recheck["mode_certified"] or not cert["mode_certified"])
        ok = ok and (recheck["unimodal_certified"] or not cert["unimodal_certified"])
        ok = ok and recheck["T_tail"] <= cert["T_tail"]
    if cert.get("P") == P:
        for t, (a, b) in cert["F_enclosures"].items():
            if t in recheck["F_enclosures"]:
                lo_, hi_ = (int(x) for x in recheck["F_enclosures"][t])
                ok = ok and not (int(a) > hi_ or int(b) < lo_)
    same = all(recheck[k] == cert[k] for k in ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified",
                                             "certified_local_maxima_upto_T"))
    return bool(ok), bool(same)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--exact"]
    exact_only = "--exact" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    if args[0].endswith(".jsonl"):
        certs = {}
        for line in open(args[0]):
            c = json.loads(line)
            certs[c["N"]] = c
        any_c = next(iter(certs.values()))
        d = any_c["d"]
        qn, qd = (int(x) for x in any_c["q"].split("/"))
        Nmin = int(args[1]) if len(args) > 1 else min(certs)
        Nmax = int(args[2]) if len(args) > 2 else max(certs)
        rest = args[3:]
    else:
        certs = {}
        d, qn, qd, Nmin, Nmax = (int(x) for x in args[:5])
        rest = args[5:]
    budget = float(rest[0]) if rest else 1000.0
    stride = int(rest[1]) if len(rest) > 1 else 1
    out = os.path.join(_R, 'data', 'msc_rigorous_certified', ("packedexact" if exact_only else "packed") + f"_d{d}_q{qn}-{qd}.jsonl")
    done = {json.loads(l)["N"] for l in open(out)} if os.path.exists(out) else set()
    start, bad = time.time(), 0
    for N in range(Nmin, Nmax + 1, stride):
        if N in done or (certs and N not in certs):
            continue
        if time.time() - start > budget:
            print("budget exhausted before N =", N, flush=True)
            break
        t0 = time.time()
        c = certify(d, N, qn, qd, exact_only)
        if exact_only:
            c["engine"] = "packed-exact"
        if N in certs:
            c["agrees_with_generator"], c["same_verdicts_as_generator"] = compare(c, certs[N], exact_only)
            bad += not c["agrees_with_generator"]
        c["seconds"] = round(time.time() - t0, 2)
        with open(out, "a") as fh:
            fh.write(json.dumps(c) + "\n")
        print(f"d={d} q={qn}/{qd} N={N} mode={c['mode']} T={c['T_tail']} cert={c['mode_certified']} "
              f"unimodal={c['unimodal_certified']} peaks={c['certified_local_maxima_upto_T']} "
              f"agree={c.get('agrees_with_generator')} same={c.get('same_verdicts_as_generator')} {c['seconds']}s", flush=True)
    sys.exit(1 if bad else 0)
