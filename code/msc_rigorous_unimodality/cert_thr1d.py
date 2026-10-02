"""
cert_thr1d.py -- EXACT certificates for the one-dimensional thresholds q*(N)
(note R4, proofs/R4_unimodality/, Section 7.1: Proposition 7.3 and
Computer-assisted Theorem 7.4; also Conjecture 10.2).

Model (d = 1): lazy walk on {1..N}, each direction with probability q/2, a move
off the lattice is cancelled; target a = N; start x0 (x0 = 1: "end to end";
x0 = (N+1)/2, N odd: "centre to corner").  The killed walk lives on {1..N-1}:

    Q(1,1) = 1 - q/2,  Q(i,i) = 1 - q (i >= 2),  Q(i,i+-1) = q/2,
    rho_t = Q^t e_{x0},   f(t+1) = (q/2) rho_t(N-1).

Exact arithmetic.  For q = qn/qd put g = gcd(qn, 2(qd-qn), 2qd-qn, 2qd) and
    OFF = qn/g, DIAG = 2(qd-qn)/g, DIAG1 = (2qd-qn)/g, DEN = 2qd/g.
Then R_t := DEN^t rho_t is an integer vector, R_{t+1} = M R_t with the integer
Jacobi matrix M = (OFF; DIAG1, DIAG, ..., DIAG), and
    F_{t+1} := DEN^{t+1} f(t+1) = OFF * R_t(N-1)        (F_0 = 0).
    sign(f(j) - f(j-1)) = sign(F_j - DEN F_{j-1}),  j >= 1.
Cone at time T (Lemma 6.1):  R_{T+1} <= DEN R_T componentwise  ==>  f is
nonincreasing from T+1 on.  Verdicts:
    unimodal      : cone found at T and the signs s_1..s_{T+1} are (>=0)*(<=0)*
    not unimodal  : the signs contain  + ... - ... +   (exact witness returned)

Sub-commands
    bound  N1 N2      certify the PMF at q = (2N-4)/(2N-3) for N1 <= N <= N2 (end to end), pure-Python integers
    boundf N1 N2      the same with the second implementation (python-flint integer polynomials)
    bisect N [den]    bracket q*(N) (end to end) between k/den and (k+1)/den   (default den = 10^6)
    m2c               centre -> corner, d = 1: q = 4/5, q = 1 - 1/M, brackets, q*(5) = (4+sqrt2)/7 in Z[sqrt2]
    m2cbound N1 N2    centre -> corner, d = 1, odd N in [N1, N2]: certificate at q = (N-3)/(N-1)
Records are appended to data/cert_thr1d_<cmd>.jsonl (existing N are skipped).
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository

import json
import os
import sys
import time
from fractions import Fraction
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')


def scaled(qn, qd):
    off, diag, diag1, den = qn, 2 * (qd - qn), 2 * qd - qn, 2 * qd
    g = gcd(gcd(off, diag), gcd(diag1, den))
    return off // g, diag // g, diag1 // g, den // g


def certify(N, x0, qn, qd, tmax=None, want_pmf=0):
    """Exact cone certificate for start x0 (1-based), target N, q = qn/qd.
    Returns dict(verdict, T_cone, mode, runs, witness, steps)."""
    off, diag, diag1, den = scaled(qn, qd)
    m = N - 1                      # sites 1..N-1  -> indices 0..m-1
    R = [0] * m
    R[x0 - 1] = 1
    if tmax is None:
        tmax = 60 * N * N * qd // qn + 2000
    Fprev = 0                      # F_0
    runs = []                      # [sign, first_j, last_j]
    state = 0                      # 0: nothing, 1: seen +, 2: seen + then -
    first_plus = last_plus = first_minus_after_plus = None
    witness = None
    T_cone = None
    pmf = []
    t = 0                          # R = R_t
    while t < tmax:
        # F_{t+1} and its sign
        F = off * R[m - 1]
        j = t + 1
        if want_pmf and j <= want_pmf:
            pmf.append((j, F, den))
        dlt = F - den * Fprev
        s = (dlt > 0) - (dlt < 0)
        if runs and runs[-1][0] == s:
            runs[-1][2] = j
        else:
            runs.append([s, j, j])
        if s > 0:
            if state == 2:
                witness = dict(t_minus=first_minus_after_plus, t_plus_again=j)
                return dict(verdict=False, T_cone=None, mode=None, runs=runs, witness=witness, steps=j)
            state = 1
            last_plus = j
        elif s < 0 and state == 1:
            state = 2
            first_minus_after_plus = j
        Fprev = F
        # next vector
        Rn = [0] * m
        if m == 1:
            Rn[0] = diag1 * R[0]
        else:
            Rn[0] = diag1 * R[0] + off * R[1]
            for i in range(1, m - 1):
                Rn[i] = diag * R[i] + off * (R[i - 1] + R[i + 1])
            Rn[m - 1] = diag * R[m - 1] + off * R[m - 2]
        # cone at T = t :  R_{t+1} <= den R_t   (cheap pre-check at the last site)
        if Rn[m - 1] <= den * R[m - 1]:
            if all(Rn[i] <= den * R[i] for i in range(m)):
                T_cone = t
                # sign of f(T+2) - f(T+1) is <= 0 by the cone; signs s_1..s_{T+1} recorded
                nz = [r[0] for r in runs if r[0] != 0]
                ch = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
                ok = ch == 0 or (ch == 1 and nz[0] > 0)
                mode = last_plus if last_plus is not None else 1
                return dict(verdict=bool(ok), T_cone=T_cone, mode=mode, runs=runs, witness=None,
                            steps=j, bits=R[m - 1].bit_length(), pmf=pmf)
        R = Rn
        t += 1
    return dict(verdict=None, T_cone=None, mode=None, runs=runs, witness=None, steps=t)


def certify_flint(N, x0, qn, qd, tmax=None):
    """Same certificate as certify(), second implementation: the vector R_t is stored as an
    integer polynomial (python-flint fmpz_poly) and one step is the polynomial product with
    OFF + DIAG x + OFF x^2, plus the wall correction at site 1.  Exact integers throughout."""
    from flint import fmpz, fmpz_poly
    off, diag, diag1, den = scaled(qn, qd)
    m = N - 1
    v = [0] * m
    v[x0 - 1] = 1
    R = fmpz_poly(v)
    K = fmpz_poly([off, diag, off])
    if tmax is None:
        tmax = 60 * N * N * qd // qn + 2000
    Fprev = fmpz(0)
    runs = []
    state = 0
    last_plus = first_minus_after_plus = None
    t = 0
    while t < tmax:
        top = R[m - 1]
        F = off * top
        j = t + 1
        dlt = F - den * Fprev
        s = (dlt > 0) - (dlt < 0)
        if runs and runs[-1][0] == s:
            runs[-1][2] = j
        else:
            runs.append([s, j, j])
        if s > 0:
            if state == 2:
                return dict(verdict=False, T_cone=None, mode=None, runs=runs,
                            witness=dict(t_minus=first_minus_after_plus, t_plus_again=j), steps=j)
            state = 1
            last_plus = j
        elif s < 0 and state == 1:
            state = 2
            first_minus_after_plus = j
        Fprev = F
        P = R * K
        Rn = fmpz_poly(P.coeffs()[1:m + 1]) + (diag1 - diag) * R[0]
        if Rn[m - 1] <= den * top:
            Dp = den * R - Rn
            if all(c >= 0 for c in Dp.coeffs()):
                nz = [r[0] for r in runs if r[0] != 0]
                ch = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
                ok = ch == 0 or (ch == 1 and nz[0] > 0)
                mode = last_plus if last_plus is not None else 1
                return dict(verdict=bool(ok), T_cone=t, mode=mode, runs=runs, witness=None, steps=j,
                            bits=int(top).bit_length())
        R = Rn
        t += 1
    return dict(verdict=None, T_cone=None, mode=None, runs=runs, witness=None, steps=t)


def pmf_exact(N, x0, q, tmax):
    """f(1..tmax) as Fractions, straight from the definition (independent of certify)."""
    q = Fraction(q)
    m = N - 1
    rho = [Fraction(0)] * m
    rho[x0 - 1] = Fraction(1)
    out = [Fraction(0)]
    for _ in range(tmax):
        out.append(q / 2 * rho[m - 1])
        new = [Fraction(0)] * m
        for i in range(m):
            stay = (1 - q / 2) if i == 0 else (1 - q)
            new[i] = stay * rho[i]
            if i > 0:
                new[i] += q / 2 * rho[i - 1]
            if i < m - 1:
                new[i] += q / 2 * rho[i + 1]
        rho = new
    return out


def short(runs, k=12):
    return runs if len(runs) <= 2 * k else runs[:k] + [["..."]] + runs[-k:]


def load_done(path, key="N"):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)[key])
            except Exception:
                pass
    return done


def cmd_bound(N1, N2, engine="py"):
    out = os.path.join(DATA, "cert_thr1d_bound.jsonl" if engine == "py" else "cert_thr1d_boundf.jsonl")
    done = load_done(out)
    for N in range(N1, N2 + 1):
        if N in done:
            continue
        t0 = time.time()
        qn, qd = 2 * N - 4, 2 * N - 3
        r = (certify if engine == "py" else certify_flint)(N, 1, qn, qd)
        rec = dict(N=N, q=f"{qn}/{qd}", unimodal=r["verdict"], T_cone=r["T_cone"], mode=r["mode"],
                   steps=r["steps"], witness=r["witness"], sign_runs=short(r["runs"]),
                   bits=r.get("bits"), seconds=round(time.time() - t0, 2))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("bound", engine, "N", N, "q", rec["q"], "unimodal", rec["unimodal"], "mode", rec["mode"], "T_cone", rec["T_cone"],
              "witness", rec["witness"], "sec", rec["seconds"], flush=True)


def cmd_bisect(N, den=10 ** 6):
    out = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_bisect.jsonl')
    done = load_done(out)
    if N in done:
        return
    t0 = time.time()
    # upper end: the Proposition 4.4 bound (2N-4)/(2N-3); test it first
    rb = certify(N, 1, 2 * N - 4, 2 * N - 3)
    lo, hi = den // 2, (den * (2 * N - 4)) // (2 * N - 3) + 1      # hi/den > bound: not unimodal (Prop. 4.4)
    rlo = certify(N, 1, lo, den)
    assert rlo["verdict"] is True
    rhi = certify(N, 1, hi, den)
    assert rhi["verdict"] is False, (N, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = certify(N, 1, mid, den)
        assert r["verdict"] is not None
        if r["verdict"]:
            lo, rlo = mid, r
        else:
            hi, rhi = mid, r
    rec = dict(N=N, den=den, lo=lo, hi=hi, unimodal_at_bound=rb["verdict"], bound=f"{2*N-4}/{2*N-3}",
               bound_float=(2 * N - 4) / (2 * N - 3),
               lo_T_cone=rlo["T_cone"], lo_mode=rlo["mode"], hi_witness=rhi["witness"],
               bound_witness=rb["witness"], bound_T_cone=rb["T_cone"], seconds=round(time.time() - t0, 2))
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("bisect N", N, "q* in [", lo / den, ",", hi / den, ")  bound", rec["bound_float"], "unimodal at bound:",
          rb["verdict"], "hi witness", rhi["witness"], "sec", rec["seconds"], flush=True)


# ---------- exact arithmetic in Z[sqrt2] for q*(5) = (4 + sqrt2)/7 (centre -> corner) ----------
class Z2:
    """a + b*sqrt(2), a, b integers."""
    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a, self.b = a, b

    def __add__(self, o):
        return Z2(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        return Z2(self.a - o.a, self.b - o.b)

    def __mul__(self, o):
        return Z2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    def sign(self):
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if a > 0 and b > 0:
            return 1
        if a < 0 and b < 0:
            return -1
        # opposite signs: compare a^2 with 2 b^2
        c = a * a - 2 * b * b
        if a > 0:      # a > 0 > b : positive iff a^2 > 2 b^2
            return (c > 0) - (c < 0)
        return (c < 0) - (c > 0)

    def __float__(self):
        return self.a + self.b * 2 ** 0.5


def certify_m2c_N5_sqrt2(tmax=400):
    """N = 5, start 3, target 5, q = (4+sqrt2)/7:  q/2 = (4+sqrt2)/14, 1-q = (6-2sqrt2)/14, 1-q/2 = (10-sqrt2)/14."""
    off, diag, diag1, den = Z2(4, 1), Z2(6, -2), Z2(10, -1), Z2(14, 0)
    m = 4
    R = [Z2(0)] * m
    R[2] = Z2(1)
    Fprev = Z2(0)
    signs = []
    for t in range(tmax):
        F = off * R[m - 1]
        signs.append((F - den * Fprev).sign())
        Fprev = F
        Rn = [None] * m
        Rn[0] = diag1 * R[0] + off * R[1]
        for i in range(1, m - 1):
            Rn[i] = diag * R[i] + off * (R[i - 1] + R[i + 1])
        Rn[m - 1] = diag * R[m - 1] + off * R[m - 2]
        if all((den * R[i] - Rn[i]).sign() >= 0 for i in range(m)):
            nz = [s for s in signs if s != 0]
            ch = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
            return dict(T_cone=t, signs=signs, unimodal=bool(ch == 0 or (ch == 1 and nz[0] > 0)))
        R = Rn
    return dict(T_cone=None, signs=signs, unimodal=None)


def cmd_m2c():
    out = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_m2c.json')
    res = {}
    F45 = Fraction(4, 5)
    # (1) q = 4/5, odd N = 3..31
    rows = []
    for N in range(3, 33, 2):
        c = (N + 1) // 2
        r = certify(N, c, 4, 5)
        row = dict(N=N, unimodal=r["verdict"], T_cone=r["T_cone"], mode=r["mode"], witness=r["witness"])
        if r["verdict"] is False:
            w = r["witness"]
            f = pmf_exact(N, c, F45, w["t_plus_again"])
            tm = w["t_plus_again"] - 1          # f(tm-1) > f(tm) < f(tm+1) ?
            # the dip: last index before the rise
            assert f[tm] < f[tm + 1]
            k = tm
            while f[k - 1] == f[k]:
                k -= 1
            assert f[k - 1] > f[k]
            row["exact_dip"] = [[k - 1, str(f[k - 1])], [tm, str(f[tm])], [tm + 1, str(f[tm + 1])]]
        rows.append(row)
        print("m2c q=4/5 N", N, row, flush=True)
    res["q=4/5"] = rows
    # (2) the elementary threshold q = 1 - 1/M, M = (N-1)/2 >= 3 : unimodal there?
    rows = []
    for N in range(7, 41, 2):
        M = (N - 1) // 2
        c = (N + 1) // 2
        r = certify(N, c, M - 1, M)
        f = pmf_exact(N, c, Fraction(M - 1, M), M + 3)
        rows.append(dict(N=N, M=M, q=f"{M-1}/{M}", unimodal=r["verdict"], T_cone=r["T_cone"], mode=r["mode"],
                         witness=r["witness"], tie=bool(f[M] == f[M + 1]), fM2_minus_fM1=str(f[M + 2] - f[M + 1])))
        print("m2c q=1-1/M N", N, rows[-1], flush=True)
    res["q=1-1/M"] = rows
    # (3) brackets with denominator 10^6
    den = 10 ** 6
    rows = []
    for N in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31):
        c = (N + 1) // 2
        M = (N - 1) // 2
        lo = den // 2
        hi = den if N == 3 else None
        # an upper end that is certainly not unimodal
        top = den
        r = certify(N, c, top, den)
        assert r["verdict"] is False, N
        hi = top
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = certify(N, c, mid, den)
            assert r["verdict"] is not None
            if r["verdict"]:
                lo = mid
            else:
                hi = mid
        rows.append(dict(N=N, lo=lo, hi=hi, den=den, elementary_bound=(1 - 1 / M) if M >= 2 else None))
        print("m2c bracket N", N, lo / den, hi / den, "elementary 1-1/M =", (1 - 1 / M) if M >= 2 else None, flush=True)
    res["brackets"] = rows
    # (4) N = 5 at q = (4+sqrt2)/7, exact in Z[sqrt2]
    r5 = certify_m2c_N5_sqrt2()
    res["N5_sqrt2"] = dict(q="(4+sqrt2)/7", q_float=(4 + 2 ** 0.5) / 7, T_cone=r5["T_cone"], unimodal=r5["unimodal"],
                           signs=r5["signs"])
    print("m2c N=5, q=(4+sqrt2)/7:", res["N5_sqrt2"], flush=True)
    # (5) N = 3: hand formulas
    q = Fraction(4, 5)
    f = pmf_exact(3, 2, q, 8)
    res["N3_q4/5"] = [str(v) for v in f]
    json.dump(res, open(out, "w"), indent=1)
    print("written", out)


def cmd_m2cbound(N1, N2):
    """centre -> corner, d = 1, odd N: certify the PMF at q = 1 - 1/M = (N-3)/(N-1), M = (N-1)/2 >= 3."""
    out = _os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_thr1d_m2cbound.jsonl')
    done = load_done(out)
    for N in range(N1, N2 + 1):
        if N % 2 == 0 or N < 7 or N in done:
            continue
        t0 = time.time()
        M = (N - 1) // 2
        r = certify_flint(N, M + 1, M - 1, M)
        rec = dict(N=N, M=M, q=f"{M-1}/{M}", unimodal=r["verdict"], T_cone=r["T_cone"], mode=r["mode"],
                   steps=r["steps"], witness=r["witness"], sign_runs=short(r["runs"]),
                   bits=r.get("bits"), seconds=round(time.time() - t0, 2))
        if N <= 81:      # cross-check with the pure-Python engine
            r2 = certify(N, M + 1, M - 1, M)
            assert (r2["verdict"], r2["T_cone"], r2["mode"], r2["witness"]) == (r["verdict"], r["T_cone"], r["mode"], r["witness"])
            rec["crosscheck_python_engine"] = True
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("m2cbound N", N, "q", rec["q"], "unimodal", rec["unimodal"], "mode", rec["mode"], "T_cone", rec["T_cone"],
              "witness", rec["witness"], "sec", rec["seconds"], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bound":
        cmd_bound(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "boundf":
        cmd_bound(int(sys.argv[2]), int(sys.argv[3]), engine="flint")
    elif cmd == "bisect":
        for N in sys.argv[2:]:
            cmd_bisect(int(N))
    elif cmd == "m2c":
        cmd_m2c()
    elif cmd == "m2cbound":
        cmd_m2cbound(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
