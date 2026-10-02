"""
verify_family.py -- separately written sanity checks for Section 9.3 of note R4 (twisted groups, Lemmas 9.9, 9.10, Theorem 9.11).

Y1  group decomposition: for several N the residue classes k mod G (G = floor(N/16), resp. floor(N/32) for the
    folded spectrum) partition the index set; class g is A_n(beta, delta) with beta = g/G (beta = 1 for g = 0),
    delta = pi/rho, rho = N/G resp. N/(2G), n = ceil(rho - beta); every (n, beta, delta) lies in region F1 or F2
    or is the special group (n = 15, rho = 16, beta = 1) = spectrum of the path with 15 vertices.  (exact integers/Fractions)
Y2  product identity (mp, 60 digits): prod over groups of prod(1 - lam z) equals prod_k (1 - (y + 2cos(k pi/N)) z) at
    random z, and the convolution of the groups' h-sequences equals h_j of the full spectrum (N = 992, 1000; j <= 60).
Y3  Taylor models: at random real points of random boxes the value H_j (computed directly, mp 80 digits) lies within
    M_j R(t,p) of the Taylor polynomial; at random COMPLEX points of the unit polydisc |H_j| <= M_j.
Y4  negative controls: the certificate procedure fails where it must (n = 12, delta = pi/12, beta near 0.9: e_2 < 0).
Y5  independent brute force (mp): for 300 random (beta, delta) in F1 and 100 in F2, H_j > 0 and C_j > 0 for j <= 600.
Y6  direct certificates (cert_lc1d) for sizes beyond the direct range agree with Theorem 9.11
    (reads data/cert_lc1d_*_samples.jsonl if present).
Y7  for five further values of N (up to 40009): every residue-class group is certified DIRECTLY at its own parameter
    point (Arb, no Taylor models): window j < 100 and the tail criterion at j0 = 100.
"""
from __future__ import annotations
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, math, os, random, sys
from fractions import Fraction as Fr
import mpmath as mp
from flint import arb, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cert_family as F  # noqa: E402

DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
RESULTS = []
random.seed(20261001)


def record(name, ok, detail=""):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    print(("PASS " if ok else "FAIL ") + name + (" :: " + str(detail) if detail else ""), flush=True)
    with open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'verify_family.json'), "w") as fh:
        json.dump(RESULTS, fh, indent=1)


# ------------------------------------------------------------------ Y1
def groups_E2E(N):
    G = N // 16
    rho = Fr(N, G)
    out = []
    for g in range(G):
        ks = [k for k in range(g if g else G, N, G) if 1 <= k <= N - 1]
        beta = Fr(g, G) if g else Fr(1)
        out.append((g, beta, rho, ks))
    return G, rho, out


def groups_FOLD(N):
    M = (N - 1) // 2
    G = N // 32
    rho = Fr(N, 2 * G)
    out = []
    for g in range(G):
        ks = [k for k in range(g if g else G, M + 1, G) if 1 <= k <= M]
        beta = Fr(g, G) if g else Fr(1)
        out.append((g, beta, rho, ks))
    return G, rho, out


ok = True
det = []
PI_LO, PI_HI = Fr(314159265358979, 10 ** 14), Fr(314159265358980, 10 ** 14)
for kind, Ns in (("E2E", [992, 993, 1000, 1001, 1007, 1008, 1024, 1601, 1600, 4001, 4096, 100003, 65536]),
                 ("FOLD", [1985, 1987, 2001, 2015, 2017, 3001, 4001, 100003])):
    for N in Ns:
        G, rho, grp = (groups_E2E if kind == "E2E" else groups_FOLD)(N)
        ok = ok and G >= 62 and Fr(16) <= rho < Fr(16) + Fr(16, G) and rho <= Fr(16) + Fr(16, 62)
        allk = sorted(k for g in grp for k in g[3])
        top = N - 1 if kind == "E2E" else (N - 1) // 2
        ok = ok and allk == list(range(1, top + 1))
        cnt = {15: 0, 16: 0, 17: 0}
        for g, beta, rho_, ks in grp:
            n = len(ks)
            ncalc = math.ceil(rho - beta)
            ok = ok and n == ncalc
            # the indices are g + G l (l = 0..n-1) for g >= 1 and G (l+1) for g = 0
            ok = ok and ks == [(g if g else G) + G * l for l in range(n)]
            # region: delta = pi/rho in [DLO, DHI]  (pi between PI_LO and PI_HI)
            dlo, dhi = PI_LO / rho, PI_HI / rho
            if n == 16:
                ok = ok and (0 < beta <= 1) and F.DLO <= dlo and dhi <= F.DHI
            elif n == 17:
                ok = ok and (0 < beta <= Fr(2585, 10000)) and F.DLO <= dlo and dhi <= F.DHI
            elif n == 15:
                ok = ok and rho == 16 and beta == 1 and kind == "E2E"
            else:
                ok = False
            cnt[n] = cnt.get(n, 0) + 1
        det.append("%s N=%d: G=%d rho=%.4f groups n=15/16/17: %d/%d/%d" % (kind, N, G, float(rho), cnt[15], cnt[16], cnt[17]))
record("Y1 residue-class decomposition into twisted groups; all groups in F1, F2 or the special group", ok, " | ".join(det))

# ------------------------------------------------------------------ Y2
mp.mp.dps = 60
y = mp.mpf(1) / 2
ok = True
for N in (992, 1000):
    G, rho, grp = groups_E2E(N)
    for _ in range(3):
        z = mp.mpf(random.uniform(-0.3, 0.3))
        full = mp.mpf(1)
        for k in range(1, N):
            full *= (1 - (y + 2 * mp.cos(k * mp.pi / N)) * z)
        prod = mp.mpf(1)
        for g, beta, rho_, ks in grp:
            n = len(ks)
            for l in range(n):
                prod *= (1 - (y + 2 * mp.cos((l + mp.mpf(beta.numerator) / beta.denominator) * mp.pi * rho.denominator / rho.numerator)) * z)
        ok = ok and abs(full - prod) < mp.mpf(10) ** (-45) * abs(full)
record("Y2 product over groups = characteristic polynomial of the path (mp 60 digits, N=992, 1000)", ok)

# ------------------------------------------------------------------ Y3
ctx.prec = F.PREC
ring = F.SeriesRing(F.P_DEG)
ok_real = True
ok_cplx = True
worst = 0.0
for fam in ("F1", "F2"):
    bl = F.boxes(fam)
    for bi in random.sample(range(len(bl)), 4 if fam == "F1" else 2):
        n, bc, w1, dc, w2 = bl[bi]
        r1 = F.R1
        r2 = w2 / F.T_RATIO
        jmax = 60
        H, M = F.taylor_models(n, bc, dc, r1, r2, jmax, ring)
        tt = F.A(F.T_RATIO)
        p = ring.p
        Rrem = ((p + 2) * tt ** (p + 1) - (p + 1) * tt ** (p + 2)) / (1 - tt) ** 2
        from flint import acb
        for _ in range(5):
            u = Fr(random.randint(-1000, 1000), 10000)
            v = Fr(random.randint(-1000, 1000), 10000)
            beta = F.A(bc) + F.A(r1) * F.A(u)
            delta = F.A(dc) + F.A(r2) * F.A(v)
            lam = [arb(1) / 2 + 2 * ((l + beta) * delta).cos() for l in range(n)]
            h = [arb(1)] + [arb(0)] * jmax
            for a in lam:
                for j in range(1, jmax + 1):
                    h[j] = h[j] + a * h[j - 1]
            for j in (1, 2, 5, 17, 40, 60):
                tay = arb(0)
                for (a_, b_), c in zip(ring.idx, H[j]):
                    tay += c * F.A(u) ** a_ * F.A(v) ** b_
                bound = M[j] * Rrem
                err = abs(tay - h[j])
                ok_real = ok_real and bool(err < bound)
                worst = max(worst, float((err / bound).upper()))
        for _ in range(5):
            zu = complex(random.uniform(-1, 1), random.uniform(-1, 1)); zu = zu / max(1.0, abs(zu) * 1.0000001)
            zv = complex(random.uniform(-1, 1), random.uniform(-1, 1)); zv = zv / max(1.0, abs(zv) * 1.0000001)
            if random.random() < 0.5:
                zu = zu / (abs(zu) * 1.0000001); zv = zv / (abs(zv) * 1.0000001)
            beta = acb(F.A(bc)) + acb(F.A(r1)) * acb(zu.real, zu.imag)
            delta = acb(F.A(dc)) + acb(F.A(r2)) * acb(zv.real, zv.imag)
            lam = [acb(arb(1) / 2) + 2 * ((l + beta) * delta).cos() for l in range(n)]
            h = [acb(1)] + [acb(0)] * jmax
            for a in lam:
                for j in range(1, jmax + 1):
                    h[j] = h[j] + a * h[j - 1]
            for j in (1, 5, 17, 40, 60):
                ok_cplx = ok_cplx and bool(abs(h[j]) < M[j])
record("Y3a Taylor models: |H_j(point) - Taylor polynomial| <= M_j R(t,p) at random real points of 6 boxes", ok_real,
       "max observed error / bound = %.2e" % worst)
record("Y3b Cauchy bound: |H_j| <= M_j at random complex points of the unit polydisc", ok_cplx)

# ------------------------------------------------------------------ Y4 negative controls
ctx.prec = F.PREC
neg = []
# (a) n = 12, delta ~ pi/12, beta ~ 0.9 : e_2 < 0, first inequality fails
rec = F.certify_box(12, Fr(9, 10), Fr(1, 20), Fr(2618, 10000), Fr(313, 400000), ring)
neg.append(rec["status"])
# (b) n = 9, delta ~ pi/9, beta ~ 0.75: e_2 < 0, fails
rec2 = F.certify_box(9, Fr(3, 4), Fr(1, 20), Fr(3491, 10000), Fr(313, 400000), ring)
neg.append(rec2["status"])
record("Y4 negative controls are NOT certified", all(s != "CERTIFIED" for s in neg), str(neg))

# ------------------------------------------------------------------ Y5 brute force
def brute(n, beta, delta, jmax=600):
    mp.mp.dps = 50 + int(0.45 * (jmax + n))
    lam = [mp.mpf(1) / 2 + 2 * mp.cos((l + beta) * delta) for l in range(n)]
    h = [mp.mpf(1)] + [mp.mpf(0)] * (jmax + 1)
    for a in lam:
        for j in range(1, jmax + 2):
            h[j] += a * h[j - 1]
    worst = None
    for j in range(1, jmax + 1):
        if h[j] <= 0:
            return False, j
        c = (h[j] ** 2 - h[j - 1] * h[j + 1]) / h[j] ** 2
        if c <= 0:
            return False, j
        if worst is None or c < worst:
            worst = c
    return True, float(worst)


ok = True
wmin = 1.0
for fam, n, bhi, cnt in (("F1", 16, 1.0, 300), ("F2", 17, 0.2585, 100)):
    for _ in range(cnt):
        beta = mp.mpf(random.uniform(0, bhi))
        delta = mp.mpf(random.uniform(float(F.DLO), float(F.DHI)))
        good, info = brute(n, beta, delta)
        ok = ok and good
        if good:
            wmin = min(wmin, info)
record("Y5 brute force (mp): 400 random points of F1/F2, H_j > 0 and C_j > 0 for 1 <= j <= 600", ok,
       "min C_j/H_j^2 over j <= 600: %.2e" % wmin)

# ------------------------------------------------------------------ Y6
det = []
ok = True
found = False
for name in ("cert_lc1d_E2E_1-2_samples.jsonl", "cert_lc1d_FOLD_1-2_samples.jsonl"):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            found = True
            ok = ok and r["status"].startswith("LOGCONCAVE")
            det.append("%s N=%d: %s (n0=%d)" % (r["geom"], r["N"], r["status"], r["n0"]))
record("Y6 direct certificates beyond the direct range agree with Theorem 9.11", ok and found, " | ".join(det) if det else "no sample file yet")

# ------------------------------------------------------------------ Y7 direct group-by-group certificates for specific N
def group_direct(n, beta_fr, rho_fr, jwin=100):
    """Arb, point parameters (no Taylor models): H_j > 0, C_j > 0 for 1 <= j < jwin, and the tail criterion of
    Lemma 9.5 at j0 = jwin.  Independent of cert_family.certify_box (only the definition of the group is shared)."""
    ctx.prec = 256
    beta = F.A(beta_fr)
    delta = arb.pi() / F.A(rho_fr)
    lam = [arb(1) / 2 + 2 * ((l + beta) * delta).cos() for l in range(n)]
    h = [arb(1)] + [arb(0)] * (jwin + 1)
    for a in lam:
        for j in range(1, jwin + 2):
            h[j] = h[j] + a * h[j - 1]
    for j in range(1, jwin):
        if not (h[j] > 0) or not (h[j] * h[j] - h[j - 1] * h[j + 1] > 0):
            return False
    # tail at j0 = jwin : exponents J = jwin + n - 2 (pairs), jwin + n - 1 (positivity)
    for k in range(n - 1):
        if not (lam[k] > lam[k + 1]):
            return False
    if not (lam[1] > 0):
        return False
    c = []
    for k in range(n):
        pr = arb(1)
        for l in range(n):
            if l != k:
                pr *= (lam[k] - lam[l])
        c.append(abs(1 / pr))
    for k in range(2, n):
        if not (lam[1] > abs(lam[k])):
            return False
    J = jwin + n - 2
    den = (lam[0] * lam[1]).sqrt()
    p = [c[k] * F.ipow(abs(lam[k]) / den, J) for k in range(n)]
    s0 = arb(0); s1 = arb(0); s2 = arb(0)
    for pk, lk in zip(p, lam):
        s0 += pk; s1 += pk * lk; s2 += pk * lk * lk
    tot = s0 * s2 - s1 * s1
    L = p[0] * p[1] * (lam[0] - lam[1]) ** 2
    if not (2 * L - tot > 0):
        return False
    rest = arb(0)
    for k in range(1, n):
        rest += c[k] * F.ipow(abs(lam[k]) / lam[0], J + 1)
    return bool(c[0] - rest > 0)


ok = True
det = []
for kind, N in (("E2E", 1237), ("E2E", 5003), ("E2E", 40009), ("FOLD", 2003), ("FOLD", 30011)):
    G, rho, grp = (groups_E2E if kind == "E2E" else groups_FOLD)(N)
    good = 0
    for g, beta, rho_, ks in grp:
        n = len(ks)
        if n == 15:
            good += 1          # the path with 15 vertices (direct certificate, N = 16)
            continue
        if group_direct(n, beta, rho):
            good += 1
    ok = ok and good == len(grp)
    det.append("%s N=%d: %d/%d groups certified directly" % (kind, N, good, len(grp)))
record("Y7 direct (parameter-point) Arb certificate of every group of specific N: window j<100 and tail lemma at j0=100", ok, " | ".join(det))

print("ALL PASS" if all(r["ok"] for r in RESULTS) else "SOME FAILED")
