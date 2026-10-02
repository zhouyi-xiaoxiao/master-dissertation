"""
cert_enclosure.py -- RIGOROUS cone certificate of unimodality with truncated integer (fixed-point) arithmetic.

Same certificate as cert_unimodal.py (Lemma 6.1), but instead of exact integers that grow by log2(D) bits per step,
it propagates a lower and an upper bound in fixed-point arithmetic with 120 fractional bits:

    L_t(x) <= 2^119 * rho_t(x) <= U_t(x)        (integers, 3 limbs of 40 bits in numpy int64 arrays)

    L_{t+1}(x) = floor( (stay(x) L_t(x) + move * sum_{y~x} L_t(y)) / D ),   U_{t+1}(x) = ceil( ... U_t ... ),

and L, U := 0 at the target.  All weights are nonnegative integers, floor/ceil are exact integer operations, so the
enclosure holds by induction; no floating point is used anywhere.

Decisions (all on integers):
   F^L(t+1) = move * sum_{y~a} L_t(y),  F^U(t+1) = move * sum_{y~a} U_t(y)     ( f(t+1) = F(t+1) / (D 2^119) )
   sign(f(t+1)-f(t)) = '+'  if F^L(t+1) > F^U(t);   '-' if F^U(t+1) < F^L(t);   '0' if F^U(t+1) = F^U(t) = 0;
                       '?'  otherwise.
   CONE at T:  U_{T+1}(x) <= L_T(x) for every site x  ==>  rho_{T+1} <= rho_T  ==>  f nonincreasing from T+1 on.
Verdict "unimodal" iff the cone holds at some T and the signs of f(j)-f(j-1), j = 1..T+1, are
   0...0  +...+  [at most one '?']  -...-          (a single undetermined sign between the blocks is harmless).
Verdict "not unimodal" iff the determined signs contain + ... - ... +.

Usage:  python cert_enclosure.py GEO d qn qd N1 N2 ...      GEO in {CC, C2M, M2C}
Records -> data/certenc_<GEO>_d<d>_q<qn>-<qd>.jsonl (existing N are skipped).
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')

LB = 40                       # bits per limb
BASE = 1 << LB
MASK = BASE - 1
NL = 3                        # limbs -> 120 bits
ONE = 1 << (LB * NL - 1)      # fixed-point representation of 1.0  (2^119)


def geometry(geo, N, d):
    c = (N - 1) // 2
    if geo == "CC":
        return (0,) * d, (N - 1,) * d
    if geo == "C2M":
        assert N % 2 == 1
        return (0,) * d, (c,) * d
    if geo == "M2C":
        assert N % 2 == 1
        return (c,) * d, (N - 1,) * d
    raise ValueError(geo)


class Enclosure:
    def __init__(self, N, d, qn, qd, start, target):
        self.N, self.d = N, d
        shape = (N,) * d
        move = qn
        base = 2 * d * (qd - qn)
        g = math.gcd(move, base) if base else move
        self.move = move // g
        self.D = 2 * d * qd // g
        nblk = np.zeros(shape, dtype=np.int64)
        for ax in range(d):
            i0 = [slice(None)] * d; i0[ax] = 0; nblk[tuple(i0)] += 1
            i1 = [slice(None)] * d; i1[ax] = N - 1; nblk[tuple(i1)] += 1
        self.stay = (base + qn * nblk) // g
        assert int(self.stay.max()) + 2 * d * self.move == self.D, "row sums must equal D at interior/wall sites"
        # safety of int64 arithmetic: limb sums < (stay + 2d*move) * BASE = D * BASE, plus remainder (D-1)*BASE
        assert 2 * self.D * BASE < (1 << 62)
        self.start, self.target = tuple(start), tuple(target)
        self.sl = []
        for ax in range(d):
            lo = [slice(None)] * d; hi = [slice(None)] * d
            lo[ax] = slice(0, N - 1); hi[ax] = slice(1, N)
            self.sl.append((tuple(lo), tuple(hi)))
        self.nbrs = []
        for ax in range(d):
            for dx in (-1, 1):
                y = list(self.target); y[ax] += dx
                if 0 <= y[ax] < N:
                    self.nbrs.append(tuple(y))
        self.L = np.zeros((NL,) + shape, dtype=np.int64)
        self.U = np.zeros((NL,) + shape, dtype=np.int64)
        for A in (self.L, self.U):
            A[(NL - 1,) + self.start] = ONE >> (LB * (NL - 1))
        self.t = 0

    def _apply(self, A, up):
        """one step on a limb array (normalised limbs in [0, BASE)); returns normalised limbs."""
        S = A * self.stay[None]
        mA = A * self.move if self.move != 1 else A
        for lo, hi in self.sl:
            S[(slice(None),) + hi] += mA[(slice(None),) + lo]
            S[(slice(None),) + lo] += mA[(slice(None),) + hi]
        D = self.D
        out = np.empty_like(S)
        rem = np.zeros(S.shape[1:], dtype=np.int64)
        for i in range(NL - 1, 0, -1):
            tcur = S[i] + rem * BASE
            out[i] = tcur // D
            rem = tcur - out[i] * D
        t0 = S[0] + rem * BASE
        if up:
            out[0] = -((-t0) // D)
        else:
            out[0] = t0 // D
        # normalise
        for i in range(NL - 1):
            carry = out[i] >> LB
            out[i] &= MASK
            out[i + 1] += carry
        out[(slice(None),) + self.target] = 0
        return out

    def step(self):
        self.Lprev, self.Uprev = self.L, self.U
        self.L = self._apply(self.L, False)
        self.U = self._apply(self.U, True)
        self.t += 1

    @staticmethod
    def to_int(A, idx):
        v = 0
        for i in range(NL - 1, -1, -1):
            v = (v << LB) + int(A[(i,) + idx])
        return v

    def flux_bounds(self):
        """integers F^L, F^U with F^L <= D 2^119 f(t+1) <= F^U, from the CURRENT state (time t)."""
        fl = self.move * sum(self.to_int(self.L, y) for y in self.nbrs)
        fu = self.move * sum(self.to_int(self.U, y) for y in self.nbrs)
        return fl, fu

    def cone(self):
        """True iff U_t(x) <= L_{t-1}(x) for all x (rigorous: rho_t <= rho_{t-1})."""
        U, L = self.U, self.Lprev
        ok = np.ones(U.shape[1:], dtype=bool)
        undecided = np.ones(U.shape[1:], dtype=bool)
        for i in range(NL - 1, -1, -1):
            lt = U[i] < L[i]
            gt = U[i] > L[i]
            ok &= ~(undecided & gt)
            undecided &= ~(lt | gt)
            if not ok.all():
                return False
        return True


def certify(geo, N, d, qn, qd, Tmax=None):
    t0 = time.time()
    start, target = geometry(geo, N, d)
    en = Enclosure(N, d, qn, qd, start, target)
    if Tmax is None:
        Tmax = 400 * N * N * d * qd // qn + 1000
    signs = []          # sign of f(j)-f(j-1), j = 1, 2, ...
    FLp, FUp = 0, 0     # bounds for f(0) = 0
    T_cone = None
    max_width = 0
    seen_minus = False
    while en.t < Tmax:
        FL, FU = en.flux_bounds()            # bounds for f(t+1), t = en.t
        assert FL <= FU
        max_width = max(max_width, FU - FL)
        if FU == 0 and FUp == 0:
            s = "0"
        elif FL > FUp:
            s = "+"
        elif FU < FLp:
            s = "-"
            seen_minus = True
        else:
            s = "?"
        signs.append(s)                       # sign of f(t+1) - f(t)
        FLp, FUp = FL, FU
        en.step()
        # cone at T = en.t - 1 :  rho_{T+1} <= rho_T ; only tested once f has started to decrease
        if seen_minus and en.cone():
            T_cone = en.t - 1
            # signs recorded so far: j = 1..T+1 (len = T+1); the cone gives f(u+2) <= f(u+1) for all u >= T
            break
    runs = []
    for j, s in enumerate(signs, start=1):
        if runs and runs[-1][0] == s:
            runs[-1][2] = j
        else:
            runs.append([s, j, j])
    pat = "".join(r[0] for r in runs)
    n_unknown = signs.count("?")
    core = pat.lstrip("0")
    ok_pattern = core in ("", "+", "-", "+-", "+?-") and (n_unknown <= 1)
    unimodal = None
    if T_cone is not None and ok_pattern:
        unimodal = True
    det = [s for s in signs if s in "+-"]
    state = 0
    for s in det:
        if state == 0 and s == "+":
            state = 1
        elif state == 1 and s == "-":
            state = 2
        elif state == 2 and s == "+":
            unimodal = False
            break
    plus = [r for r in runs if r[0] == "+"]
    rec = dict(geo=geo, d=d, N=N, q="%d/%d" % (qn, qd), start=list(start), target=list(target),
               steps=en.t, T_cone=T_cone, sign_runs=runs if len(runs) <= 40 else runs[:20] + [["..."]] + runs[-20:],
               n_unknown=n_unknown, unimodal=unimodal,
               mode_last_increase=(plus[-1][2] if plus else None),
               fixed_point_bits=LB * NL - 1, max_flux_enclosure_width_units=int(max_width),
               seconds=round(time.time() - t0, 2))
    return rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    geo, d, qn, qd = args[0], int(args[1]), int(args[2]), int(args[3])
    Ns = [int(v) for v in args[4:]]
    out = os.path.join(DATA, "certenc_%s_d%d_q%d-%d.jsonl" % (geo, d, qn, qd))
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)["N"])
            except Exception:
                pass
    for N in Ns:
        if N in done:
            continue
        rec = certify(geo, N, d, qn, qd)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(geo, "d", d, "N", N, "q", rec["q"], "unimodal", rec["unimodal"], "last increase at", rec["mode_last_increase"],
              "T_cone", rec["T_cone"], "unknown", rec["n_unknown"], "sec", rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
