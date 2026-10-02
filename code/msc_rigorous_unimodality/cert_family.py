"""
cert_family.py -- computer-assisted proof of Lemma 9.10 of note R4 ("every twisted group is log-concave").

Statement certified (y = 1/2, i.e. q = 4/5).  For an integer n and real (beta, delta) put
    A_n(beta, delta) = ( lam_l )_{l=0..n-1},     lam_l = y + 2 cos((l + beta) delta),
    H_j(beta, delta) = h_j(lam_0, ..., lam_{n-1})   (complete homogeneous symmetric polynomial; H_0 = 1),
    C_j = H_j^2 - H_{j-1} H_{j+1}.
CLAIM(n, region):  H_j > 0 and C_j > 0 for every j >= 1 and every (beta, delta) in the region.
Regions:  F1: n = 16, beta in [0, 1],      delta in [DLO, DHI]
          F2: n = 17, beta in [0, 0.2585], delta in [DLO, DHI]
with [DLO, DHI] = [0.19322, 0.19635], which contains [pi/(16 + 16/62), pi/16].

Method for one box  B = {|beta - beta_c| <= w1, |delta - delta_c| <= w2}:
 (T) tail (Lemma 9.5 with interval parameters, on a grid of sub-boxes covering B): an index j0 such that for all
     parameters in B:  lam_0 > lam_1 > max_{k>=2}|lam_k|, lam_1 > 0, all differences lam_k - lam_l (k<l) > 0,
     L_{j0} > sum of the other |terms| (so C_j > 0 for j >= j0) and c_0 lam_0^J > sum_{k>=1} |c_k| |lam_k|^J at
     J = j0+n-1 (so H_j > 0 for j >= j0).  Here H_j = sum_k c_k lam_k^{j+n-1}, c_k = 1/prod_{l != k}(lam_k - lam_l).
 (W) window 1 <= j < j0: Taylor models.  With beta = beta_c + r1 u, delta = delta_c + r2 v, the functions H_j and C_j
     are entire in (u, v).  On the closed unit polydisc |lam_l| <= mu_l (explicit), hence |H_j| <= M_j := h_j(mu) and
     |C_j| <= M_j^2 + M_{j-1} M_{j+1}; by Cauchy's estimates the Taylor coefficients are bounded by these numbers, so
     for |u|, |v| <= t < 1 the Taylor polynomial of total degree p has remainder <= bound * R(t, p),
     R(t,p) = sum_{k>p} (k+1) t^k = ((p+2) t^{p+1} - (p+1) t^{p+2}) / (1-t)^2.
     The Taylor polynomial (coefficients = Arb balls, computed by truncated power-series arithmetic from the exact
     centre) is bounded below on B by  c_00 - sum_{(a,b) != (0,0)} |c_ab| t1^a t2^b.
     Certificate: lower bound of H_j and of C_j on B strictly positive for 1 <= j < j0.

All arithmetic is Arb ball arithmetic (python-flint); every comparison is a certified comparison of balls.

Usage:  python cert_family.py F1|F2 [box_lo box_hi]      -> data/cert_family_<F>.jsonl  (one line per box)
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys
import time
from fractions import Fraction as Fr

from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')

PREC = 320
Y = Fr(1, 2)
DLO, DHI = Fr(19322, 100000), Fr(19635, 100000)
P_DEG = 14            # total degree of the Taylor polynomials
R1 = Fr(1, 2)         # polydisc radius in beta
T_RATIO = Fr(1, 10)   # box half-width / polydisc radius (both variables)
J0_CANDIDATES = (30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 200)
SUB_BETA, SUB_DELTA = 10, 4   # sub-boxes for the tail test


def A(x):
    """Fraction/int -> exact arb"""
    if isinstance(x, Fr):
        return arb(x.numerator) / arb(x.denominator)
    return arb(x)


def ball(mid, rad):
    """arb ball [mid - rad, mid + rad] for exact Fractions mid, rad"""
    return A(mid) + A(rad) * arb(0, 1) if rad != 0 else A(mid)


# ----------------------------------------------------------------------------------------------------------
# tail test on a (sub-)box
# ----------------------------------------------------------------------------------------------------------
def tail_box(n, bb, dd, j0):
    """certified: for all (beta, delta) in the balls bb, dd the hypotheses of the tail lemma hold at j0.
    returns (ok, info)"""
    y = A(Y)
    lam = [y + 2 * ((l + bb) * dd).cos() for l in range(n)]
    # differences d[k][l] = lam_k - lam_l = 4 sin(((k+l)/2 + beta) delta) sin((l-k) delta/2), k < l
    d = [[None] * n for _ in range(n)]
    for k in range(n):
        for l in range(k + 1, n):
            v = 4 * ((A(Fr(k + l, 2)) + bb) * dd).sin() * ((l - k) * dd / 2).sin()
            if not (v > 0):
                return False, "difference %d,%d not certified positive" % (k, l)
            d[k][l] = v
    if not (lam[1] > 0):
        return False, "lam_1 > 0 fails"
    for k in range(2, n):
        if not (lam[1] > abs(lam[k])):
            return False, "lam_1 > |lam_%d| fails" % k
    # |c_k| = 1 / prod_{l != k} |lam_k - lam_l|
    cabs = []
    for k in range(n):
        pr = arb(1)
        for l in range(n):
            if l < k:
                pr *= d[l][k]
            elif l > k:
                pr *= d[k][l]
        cabs.append(1 / pr)
    J = j0 + n - 2
    # q_k = |lam_k| / lam_0 (k>=1) and |lam_k| / lam_1 (k >= 2), certified upper bounds raised to the power J
    up0 = [None] * n   # (|lam_k|/lam_0)^J
    up1 = [None] * n   # (|lam_k|/lam_1)^J
    for k in range(2, n):
        up0[k] = ipow((abs(lam[k]) / lam[0]).upper(), J)
        up1[k] = ipow((abs(lam[k]) / lam[1]).upper(), J)
    rest = arb(0)
    for k in range(n):
        for l in range(k + 1, n):
            if (k, l) == (0, 1):
                continue
            if k == 0:
                w = up1[l]                       # (lam_0 |lam_l|)^J / (lam_0 lam_1)^J
            elif k == 1:
                w = up0[l]                       # (lam_1 |lam_l|)^J / (lam_0 lam_1)^J
            else:
                w = up0[k] * up1[l]
            rest += (cabs[k] * cabs[l] * d[k][l] ** 2).upper() * w
    L = (cabs[0] * cabs[1] * d[0][1] ** 2).lower()
    if not (L - rest > 0):
        return False, "L > rest fails (ratio %.3g)" % float((rest / L).mid())
    # positivity for j >= j0:  |c_0| > sum_{k>=1} |c_k| (|lam_k|/lam_0)^(j0+n-1)
    J2 = j0 + n - 1
    s = (cabs[1] * ipow((lam[1] / lam[0]).upper(), J2)).upper()
    for k in range(2, n):
        s += cabs[k].upper() * ipow((abs(lam[k]) / lam[0]).upper(), J2)
    if not (cabs[0].lower() - s > 0):
        return False, "tail positivity fails"
    return True, float((rest / L).upper())


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


# ----------------------------------------------------------------------------------------------------------
# truncated bivariate power series (total degree <= p), coefficients = arb
# ----------------------------------------------------------------------------------------------------------
class SeriesRing:
    def __init__(self, p):
        self.p = p
        self.idx = [(a, b) for b in range(p + 1) for a in range(p + 1 - b)]
        self.pos = {ab: i for i, ab in enumerate(self.idx)}
        table = [[] for _ in self.idx]
        for i, (a1, b1) in enumerate(self.idx):
            for j, (a2, b2) in enumerate(self.idx):
                if a1 + a2 + b1 + b2 <= p:
                    table[self.pos[(a1 + a2, b1 + b2)]].append((i, j))
        self.table = table
        self.size = len(self.idx)

    def zero(self):
        return [arb(0)] * self.size

    def one(self):
        z = self.zero()
        z[self.pos[(0, 0)]] = arb(1)
        return z

    def mul(self, X, Yv):
        out = []
        for lst in self.table:
            s = arb(0)
            for i, j in lst:
                s += X[i] * Yv[j]
            out.append(s)
        return out

    @staticmethod
    def add(X, Yv):
        return [a + b for a, b in zip(X, Yv)]

    @staticmethod
    def scale(X, c):
        return [a * c for a in X]

    def lower_bound(self, X, t1, t2):
        """c_00 - sum_{(a,b) != 0} |c_ab| t1^a t2^b   (arb)"""
        s = arb(0)
        for (a, b), c in zip(self.idx, X):
            if a == 0 and b == 0:
                continue
            s += abs(c).upper() * t1 ** a * t2 ** b
        return X[self.pos[(0, 0)]] - s


def lambda_series(ring, l, beta_c, delta_c, r1, r2):
    """Taylor series (total degree <= p) of lam_l = y + 2 cos((l + beta_c + r1 u)(delta_c + r2 v)) in (u, v)."""
    p = ring.p
    a = (l + A(beta_c)) * A(delta_c)
    eta = ring.zero()
    eta[ring.pos[(1, 0)]] = A(r1) * A(delta_c)
    eta[ring.pos[(0, 1)]] = (l + A(beta_c)) * A(r2)
    if p >= 2:
        eta[ring.pos[(1, 1)]] = A(r1) * A(r2)
    cosser = ring.one()
    sinser = ring.zero()
    power = ring.one()
    fact = arb(1)
    for k in range(1, p + 1):
        power = ring.mul(power, eta)
        fact = fact * k
        if k % 2 == 0:
            sgn = 1 if (k // 2) % 2 == 0 else -1
            cosser = ring.add(cosser, ring.scale(power, sgn / fact))
        else:
            sgn = 1 if ((k - 1) // 2) % 2 == 0 else -1
            sinser = ring.add(sinser, ring.scale(power, sgn / fact))
    ser = ring.add(ring.scale(cosser, 2 * a.cos()), ring.scale(sinser, -2 * a.sin()))
    ser[ring.pos[(0, 0)]] = ser[ring.pos[(0, 0)]] + A(Y)
    return ser


def mu_bound(l, beta_c, delta_c, r1, r2):
    """certified upper bound of |lam_l| on the closed polydisc |u|, |v| <= 1 (complex)"""
    a = (l + A(beta_c)) * A(delta_c)
    eps = A(r1) * A(delta_c) + (l + A(beta_c)) * A(r2) + A(r1) * A(r2)
    val = abs(A(Y) + 2 * a.cos()) + 2 * abs(a.cos()) * (eps.cosh() - 1) + 2 * abs(a.sin()) * eps.sinh()
    return val.upper()


def taylor_models(n, beta_c, delta_c, r1, r2, jmax, ring):
    """Taylor series (in the normalised variables u, v) of H_0..H_jmax at the centre, and the bounds
    M_j >= sup |H_j| over the closed unit polydisc."""
    lam = [lambda_series(ring, l, beta_c, delta_c, r1, r2) for l in range(n)]
    mu = [mu_bound(l, beta_c, delta_c, r1, r2) for l in range(n)]
    H = [ring.one()] + [ring.zero() for _ in range(jmax)]
    M = [arb(1)] + [arb(0)] * jmax
    for l in range(n):
        for j in range(1, jmax + 1):
            H[j] = ring.add(H[j], ring.mul(lam[l], H[j - 1]))
            M[j] = M[j] + mu[l] * M[j - 1]
    return H, M


def certify_box(n, beta_c, w1, delta_c, w2, ring, verbose=False):
    t0 = time.time()
    ctx.prec = PREC
    r1 = R1
    t = T_RATIO
    assert w1 <= r1 * t
    r2 = w2 / t
    t1 = A(w1 / r1)
    t2 = A(w2 / r2)
    tt = A(t)
    p = ring.p
    rec = dict(n=n, beta_c=str(beta_c), w1=str(w1), delta_c=str(delta_c), w2=str(w2), p=p, r1=str(r1), r2=str(r2),
               t=str(t), prec=PREC)
    # ---- (T) tail time valid on the whole box (grid of sub-boxes)
    j0 = None
    worst_ratio = None
    for cand in J0_CANDIDATES:
        ok = True
        wr = 0.0
        hb, hd = w1 / SUB_BETA, w2 / SUB_DELTA
        for ib in range(SUB_BETA):
            bc = beta_c - w1 + hb * (2 * ib + 1)
            bb = ball(bc, hb)
            for idl in range(SUB_DELTA):
                dc = delta_c - w2 + hd * (2 * idl + 1)
                dd = ball(dc, hd)
                good, info = tail_box(n, bb, dd, cand)
                if not good:
                    ok = False
                    break
                wr = max(wr, info)
            if not ok:
                break
        if ok:
            j0 = cand
            worst_ratio = wr
            break
    if j0 is None:
        rec["status"] = "FAIL: no tail index"
        return rec
    rec["j0"] = j0
    rec["tail_rest_over_L_max"] = worst_ratio
    # ---- (W) Taylor models
    H, M = taylor_models(n, beta_c, delta_c, r1, r2, j0, ring)
    Rrem = ((p + 2) * tt ** (p + 1) - (p + 1) * tt ** (p + 2)) / (1 - tt) ** 2
    c00 = ring.pos[(0, 0)]
    minH = None
    minC = None
    argC = None
    maxkappa = 0.0
    okall = True
    fail = None
    for j in range(1, j0):
        hj = H[j][c00]
        lbH = ring.lower_bound(H[j], t1, t2) - M[j] * Rrem
        relH = (lbH / hj)
        if not (lbH > 0):
            okall = False
            fail = fail or ("H", j, float(relH.mid()))
        TC = ring.add(ring.mul(H[j], H[j]), ring.scale(ring.mul(H[j - 1], H[j + 1]), arb(-1)))
        MC = M[j] * M[j] + M[j - 1] * M[j + 1]
        lbC = ring.lower_bound(TC, t1, t2) - MC * Rrem
        relC = lbC / (hj * hj)
        if not (lbC > 0):
            okall = False
            fail = fail or ("C", j, float(relC.mid()))
        fH = float(relH.mid())
        fC = float(relC.mid())
        if minH is None or fH < minH:
            minH = fH
        if minC is None or fC < minC:
            minC, argC = fC, j
        maxkappa = max(maxkappa, float((M[j] / hj).mid()))
    rec.update(min_lowerbound_H_rel=minH, min_lowerbound_C_over_H2=minC, argmin_j=argC, max_kappa=maxkappa,
               remainder_factor=float(Rrem.upper()), seconds=round(time.time() - t0, 1))
    rec["centre_margin_at_j0m1"] = float(((H[j0 - 1][c00] ** 2 - H[j0 - 2][c00] * H[j0][c00]) / H[j0 - 1][c00] ** 2).mid())
    if fail:
        rec["first_failure"] = fail
    rec["status"] = "CERTIFIED" if okall else "NOT CERTIFIED"
    return rec


def boxes(family):
    if family == "F1":
        n, bhi = 16, Fr(1)
    elif family == "F2":
        n, bhi = 17, Fr(2585, 10000)
    else:
        raise ValueError(family)
    w1 = R1 * T_RATIO                      # 0.05
    nb = -(-bhi // (2 * w1))               # ceil
    nd = 2
    w2 = (DHI - DLO) / (2 * nd)
    out = []
    for ib in range(int(nb)):
        bc = w1 * (2 * ib + 1)
        for idl in range(nd):
            dc = DLO + w2 * (2 * idl + 1)
            out.append((n, bc, w1, dc, w2))
    return out


def region_ok():
    """certified: [pi/(16 + 16/62), pi/16] is contained in [DLO, DHI]"""
    ctx.prec = PREC
    lo = arb.pi() / (arb(16) + arb(16) / 62)
    hi = arb.pi() / 16
    return bool(lo > A(DLO)) and bool(hi < A(DHI))


def main():
    assert region_ok()
    fam = sys.argv[1]
    bl = boxes(fam)
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(bl)
    ring = SeriesRing(P_DEG)
    out = os.path.join(DATA, "cert_family_%s.jsonl" % fam)
    for i in range(lo, min(hi, len(bl))):
        n, bc, w1, dc, w2 = bl[i]
        rec = certify_box(n, bc, w1, dc, w2, ring)
        rec["box_index"] = i
        rec["family"] = fam
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
