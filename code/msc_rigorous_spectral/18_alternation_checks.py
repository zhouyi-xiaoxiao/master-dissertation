"""Sanity checks for Section 5.2 of note R2: Proposition 5.9 (monotone signs across an eigenvalue of pure parity),
Theorem 5.10 (the residues alternate iff d = 1 or N = 2) and Remark 5.11 (sign pattern = sign of one polynomial of degree p - D).

 (A) Remark 5.11 on random weighted graphs (50-digit arithmetic): deg N_x = p - D exactly, leading coefficient (-1)^p (L^D)_{xa},
     sign r_j = (-1)^j sign N_x(sigma_j), sign N_x(Lam_i) = (-1)^i sign omega_i, #(non-alternating pairs) <= p - D,
     #(sign changes of (-1)^i omega_i) <= p - D, and the counting criterion (4).
 (B) Proposition 5.9 on random chains with an involution R, Ra = x0 (all eigenvalues simple, hence of pure parity),
     and on boxes with a generic target a and x0 = Ra.
 (C) Theorem 5.10 for the corner-to-corner box: the spectral claims of the proof (values, index, purity and parity of the eigenvalues used)
     for many (d, N), the residue signs (float64) where feasible, comparison with the certified patterns of script 13,
     and the counting criterion of Remark 5.11(4).
Output: ../data/18_alternation_checks.json ; every line is PASS/FAIL, the script exits with status 1 on any failure.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, itertools, math
from collections import Counter, defaultdict
import numpy as np
import mpmath as mp
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
from t2lib import build_L, site_index
mp.mp.dps = 50
rng = np.random.default_rng(20261001)
OUT = {"checks": [], "box": [], "criterion": []}
NFAIL = 0
def check(name, ok, info=""):
    global NFAIL
    OUT["checks"].append({"name": name, "ok": bool(ok), "info": info})
    if not ok: NFAIL += 1
    print(("PASS " if ok else "FAIL ") + name + ("   " + info if info else ""), flush=True)

def graph_dist(L, x, a):
    n = L.shape[0]
    dist = {x: 0}; frontier = [x]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(n):
                if v != u and L[u, v] < 0 and v not in dist:
                    dist[v] = dist[u] + 1; nxt.append(v)
        frontier = nxt
    return dist[a]

def visible_data(L, a, x, tol=mp.mpf(10) ** -30):
    """distinct visible eigenvalues Lam_i, weights w_i = <e_a,E_i e_a>, cross weights omega_i = <e_x,E_i e_a> (mp, 50 digits)."""
    n = L.shape[0]
    E, Q = mp.eigsy(mp.matrix(L.tolist()))
    order = sorted(range(n), key=lambda i: E[i])
    Lam, W, Om = [], [], []
    for i in order:
        wa = Q[a, i] ** 2; om = Q[x, i] * Q[a, i]
        if Lam and abs(E[i] - Lam[-1]) < tol * (1 + abs(E[i])):
            W[-1] += wa; Om[-1] += om
        else:
            Lam.append(E[i]); W.append(wa); Om.append(om)
    keep = [i for i in range(len(Lam)) if W[i] > tol]
    return [Lam[i] for i in keep], [W[i] for i in keep], [Om[i] for i in keep]

def zeros_of_G(Lam, W):
    p = len(Lam) - 1
    G = lambda s: mp.fsum(W[i] / (Lam[i] - s) for i in range(p + 1))
    sig = []
    for j in range(p):
        lo, hi = Lam[j], Lam[j + 1]
        a_, b_ = lo, hi
        for _ in range(170):                      # bisection on the strictly increasing G
            m = (a_ + b_) / 2
            if G(m) < 0: a_ = m
            else: b_ = m
        sig.append((a_ + b_) / 2)
    return sig

def residues(Lam, W, Om, sig):
    p = len(Lam) - 1
    out = []
    for s in sig:
        Gx = mp.fsum(Om[i] / (Lam[i] - s) for i in range(p + 1))
        dG = mp.fsum(W[i] / (Lam[i] - s) ** 2 for i in range(p + 1))
        out.append(-Gx / (s * dG))
    return out

def sgn(v, tol=mp.mpf(10) ** -25):
    return 0 if abs(v) < tol else (1 if v > 0 else -1)

# ------------------------------------------------------------------------------------------------ (A)
def random_graph(n, extra):
    """random tree plus `extra` random edges, random weights in (0.2, 1.2); returns L = D - A (symmetric, L 1 = 0)."""
    A = np.zeros((n, n))
    perm = rng.permutation(n)
    for i in range(1, n):
        j = perm[rng.integers(0, i)]
        A[perm[i], j] = A[j, perm[i]] = 0.2 + rng.random()
    for _ in range(extra):
        i, j = rng.integers(0, n, 2)
        if i != j: A[i, j] = A[j, i] = 0.2 + rng.random()
    return np.diag(A.sum(1)) - A

nA = 0; okA = dict(deg=True, lead=True, sign=True, lam=True, pairs=True, V=True, crit=True)
statsA = Counter()
graphs = [random_graph(int(rng.integers(5, 12)), int(rng.integers(0, 4))) for _ in range(60)]
for N_ in (4, 6, 9):                                 # paths (d = 1 boxes): p = D, alternation
    graphs.append(build_L(N_, 1))
graphs += [build_L(3, 2), build_L(4, 2), build_L(2, 3)]
pairs_fixed = {}                                     # opposite ends / opposite corners: cases with p = D
for N_ in (4, 6, 9): pairs_fixed[len(graphs) - 6 + (4, 6, 9).index(N_)] = (N_ - 1, 0)
pairs_fixed[len(graphs) - 3] = (8, 0); pairs_fixed[len(graphs) - 2] = (15, 0); pairs_fixed[len(graphs) - 1] = (7, 0)
for gi, L in enumerate(graphs):
    n = L.shape[0]
    for rep in range(3):
        a, x = (int(v) for v in rng.choice(n, 2, replace=False))
        if rep == 0 and gi in pairs_fixed: a, x = pairs_fixed[gi]
        D = graph_dist(L, x, a)
        Lam, W, Om = visible_data(L, a, x)
        p = len(Lam) - 1
        sig = zeros_of_G(Lam, W)
        r = residues(Lam, W, Om, sig)
        # polynomial N_x(s) = sum_i omega_i prod_{l != i} (Lam_l - s): coefficients (highest first)
        coeffs = [mp.mpf(0)] * (p + 1)
        for i in range(p + 1):
            poly = [mp.mpf(1)]
            for l in range(p + 1):
                if l != i:
                    poly = [(-poly[t] if t < len(poly) else 0) + (Lam[l] * poly[t - 1] if t >= 1 else 0) for t in range(len(poly) + 1)]
            # poly holds coefficients of prod (Lam_l - s), highest degree first, length p+1
            for t in range(p + 1): coeffs[t] += Om[i] * poly[t]
        scale = max(abs(c) for c in coeffs)
        high = coeffs[:D]                            # degrees p, p-1, ..., p-D+1 must vanish
        lead = coeffs[D]                             # coefficient of s^{p-D}
        LD = np.linalg.matrix_power(L, D)[x, a]
        okA["deg"] &= all(abs(c) < mp.mpf(10) ** -30 * scale for c in high) and abs(lead) > mp.mpf(10) ** -20 * scale
        okA["lead"] &= abs(lead - (-1) ** p * LD) < 1e-9 * abs(LD) and (np.sign(LD) == (-1) ** D)
        Nx = lambda s: mp.fsum(Om[i] * mp.fprod(Lam[l] - s for l in range(p + 1) if l != i) for i in range(p + 1))
        sr = [sgn(v) for v in r]
        okA["sign"] &= all(sr[j] == (-1) ** j * sgn(Nx(sig[j]), mp.mpf(10) ** -40) for j in range(p) if sr[j] != 0)
        okA["lam"] &= all(sgn(Nx(Lam[i]), mp.mpf(10) ** -40) == (-1) ** i * sgn(Om[i], mp.mpf(10) ** -40) for i in range(p + 1))
        nonalt = sum(1 for j in range(p - 1) if sr[j] * sr[j + 1] > 0)
        okA["pairs"] &= nonalt <= p - D
        eps = [(-1) ** i * sgn(Om[i]) for i in range(1, p + 1)]
        nz = [e for e in eps if e != 0]
        V = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
        okA["V"] &= V <= p - D
        alternates = all(sr[j] == (-1) ** j for j in range(p))
        mminus = sum(1 for e in eps[:-1] if e <= 0)
        if alternates: okA["crit"] &= 2 * mminus <= p - D
        if p == D: okA["crit"] &= alternates
        statsA["cases"] += 1; statsA["alternating"] += int(alternates); statsA["p==D"] += int(p == D)
        nA += 1
check(f"Remark 5.11: deg N_x = p - D exactly ({nA} cases: random graphs, paths, small boxes; random start/target)", okA["deg"])
check("Remark 5.11: leading coefficient = (-1)^p (L^D)_{xa}, sign (-1)^{p+D}", okA["lead"])
check("Remark 5.11: sign r_j = (-1)^j sign N_x(sigma_j) for every pole", okA["sign"])
check("Remark 5.11: sign N_x(Lam_i) = (-1)^i sign omega_i", okA["lam"])
check("Remark 5.11(2): number of non-alternating consecutive pairs <= p - D", okA["pairs"])
check("Remark 5.11(3): sign changes of ((-1)^i omega_i) <= p - D", okA["V"])
check("Remark 5.11(1),(4): p = D implies alternation; alternation implies 2 #{i <= p-1: (-1)^i omega_i <= 0} <= p - D", okA["crit"], str(dict(statsA)))

# ------------------------------------------------------------------------------------------------ (B)
def sym_chain(m, extra):
    """random chain on 2m (+1 fixed point sometimes) vertices commuting with the involution R: i <-> i+m."""
    fixed = int(rng.integers(0, 3))
    n = 2 * m + fixed
    R = list(range(m, 2 * m)) + list(range(0, m)) + list(range(2 * m, n))
    A = np.zeros((n, n))
    perm = rng.permutation(n)
    edges = [(perm[i], perm[rng.integers(0, i)]) for i in range(1, n)]
    edges += [tuple(rng.integers(0, n, 2)) for _ in range(extra)]
    for i, j in edges:
        if i == j: continue
        wgt = 0.2 + rng.random()
        for (u, v) in ((i, j), (R[i], R[j])):
            if u != v:
                A[u, v] += wgt; A[v, u] += wgt
    return np.diag(A.sum(1)) - A, R

def prop59_case(L, R, a):
    """returns (#pure eigenvalues tested, #violations, sign string)"""
    n = L.shape[0]; x0 = R[a]
    E, Q = np.linalg.eigh(L)
    Rm = np.zeros((n, n)); Rm[np.arange(n), R] = 1.0
    # group eigenvalues, parity content of each group as seen from a:  w^+- = ||E v_+-||^2
    vp = np.zeros(n); vp[a] += 0.5; vp[x0] += 0.5
    vm = np.zeros(n); vm[a] += 0.5; vm[x0] -= 0.5
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(E[j + 1] - E[i]) < 1e-9: j += 1
        B = Q[:, i:j + 1]
        groups.append((E[i:j + 1].mean(), float(np.sum((B.T @ vp) ** 2)), float(np.sum((B.T @ vm) ** 2)))); i = j + 1
    vis = [(l, wp, wm) for (l, wp, wm) in groups if wp + wm > 1e-12]
    if any(1e-12 < wp + wm < 1e-7 for (l, wp, wm) in groups): return None       # undecidable in float64
    Lam = np.array([v[0] for v in vis]); Wp = np.array([v[1] for v in vis]); Wm = np.array([v[2] for v in vis])
    p = len(Lam) - 1
    if p < 2 or np.min(np.diff(Lam)) < 1e-6: return None
    G = lambda s: np.sum((Wp + Wm) / (Lam - s))
    sig = []
    for j in range(p):
        gap = Lam[j + 1] - Lam[j]; e = 1e-10 * gap
        while G(Lam[j] + e) > 0: e /= 8
        lo = Lam[j] + e; e = 1e-10 * gap
        while G(Lam[j + 1] - e) < 0: e /= 8
        sig.append(brentq(G, lo, Lam[j + 1] - e, xtol=1e-300, rtol=1e-15))
    sig = np.array(sig)
    r = np.array([2 * np.sum(Wm / (Lam - s)) / (s * np.sum((Wp + Wm) / (Lam - s) ** 2)) for s in sig])
    r2 = np.array([-2 * np.sum(Wp / (Lam - s)) / (s * np.sum((Wp + Wm) / (Lam - s) ** 2)) for s in sig])
    if np.max(np.abs(r - r2)) > 1e-6 * max(1.0, np.max(np.abs(r))): return None
    if np.min(np.abs(r)) < 1e-9: return None
    tested = viol = 0
    for i in range(1, p):
        even = Wm[i] < 1e-12; odd = Wp[i] < 1e-12
        if even:
            tested += 1; viol += int(r[i - 1] > 0 and r[i] < 0)
            viol += int(not (np.sum(Wm / (Lam - sig[i - 1])) < np.sum(Wm / (Lam - sig[i]))))
        elif odd:
            tested += 1; viol += int(r[i - 1] < 0 and r[i] > 0)
            viol += int(not (np.sum(Wp / (Lam - sig[i - 1])) < np.sum(Wp / (Lam - sig[i]))))
    return tested, viol, "".join("+" if v > 0 else "-" for v in r)

tot = viol = ncase = 0; nskip = 0
for _ in range(400):
    m = int(rng.integers(2, 9))
    L, R = sym_chain(m, int(rng.integers(0, 5)))
    a = int(rng.integers(0, 2 * m))
    res = prop59_case(L, R, a)
    if res is None: nskip += 1; continue
    tot += res[0]; viol += res[1]; ncase += 1
check(f"Proposition 5.9 on random chains with an involution ({ncase} chains, {tot} pure eigenvalues tested, {nskip} skipped as float-undecidable)", viol == 0 and tot > 1000, f"violations {viol}")
tot = viol = ncase = 0
for (N, d) in ((3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (3, 3), (4, 3)):
    L = build_L(N, d); n = N ** d
    Rl = [site_index(tuple(N + 1 - c for c in np.array(np.unravel_index(i, (N,) * d)) + 1), N, d) for i in range(n)]
    for a in range(n):
        if Rl[a] == a: continue
        res = prop59_case(L, Rl, a)
        if res is None: continue
        tot += res[0]; viol += res[1]; ncase += 1
check(f"Proposition 5.9 on boxes, every target a with Ra != a, x0 = Ra ({ncase} pairs, {tot} pure eigenvalues tested)", viol == 0 and tot > 500, f"violations {viol}")

# ------------------------------------------------------------------------------------------------ (C)
def box_spectrum(N, d):
    """distinct values V = sum_i sin^2(k_i u)/sin^2(u) (mp), with the list of |k| parities, the signed and unsigned corner weights
    (n-normalised: prod eps_k cos^2(theta_k/2), times the number of permutations)."""
    u = mp.pi / (2 * N)
    sig1 = [mp.sin(k * u) ** 2 / mp.sin(u) ** 2 for k in range(N)]
    w1 = [(1 if k == 0 else 2) * mp.cos(k * u) ** 2 for k in range(N)]
    groups = {}
    for k in itertools.combinations_with_replacement(range(N), d):
        cnt = Counter(k); mult = math.factorial(d)
        for v in cnt.values(): mult //= math.factorial(v)
        V = mp.fsum(sig1[j] for j in k)
        key = mp.nstr(V, 32)
        g = groups.setdefault(key, [V, mp.mpf(0), mp.mpf(0), set(), []])
        w = mult * mp.fprod(w1[j] for j in k)
        g[1] += w; g[2] += w * (-1) ** sum(k); g[3].add(sum(k) % 2); g[4].append(k)
    vals = sorted(groups.values(), key=lambda g: g[0])
    for i in range(len(vals) - 1):
        assert vals[i + 1][0] - vals[i][0] > mp.mpf(10) ** -20, "grouping ambiguous"
    return vals

def thm510c_claim(N, d):
    """Theorem 5.10(c): (index i of the purely even eigenvalue, excluded signs of (r_{i-1}, r_i)); None if not covered"""
    if d == 2 and N >= 3: return (3, "even", "+-")
    if d >= 4 and N >= 4: return (5, "even", "+-")
    if d == 3 and N >= 7: return (7, "even", "+-")
    return None

def thm510b_check(N, d, vals):
    """Theorem 5.10(b): returns (ok, i, parity) after checking every spectral claim of its proof (values of the drops, the
    inequalities between alpha and beta, index range, purity and common parity of Lam_i, Lam_{i+1}; for N = 4: all eigenvalues pure)."""
    p = len(vals) - 1; D = d * (N - 1)
    u = mp.pi / (2 * N); x = mp.sin(u) ** 2
    sg = [mp.sin(k * u) ** 2 / mp.sin(u) ** 2 for k in range(N)]
    top = d * sg[N - 1]
    al = mp.sin(3 * u) / mp.sin(u); be = mp.sin(5 * u) / mp.sin(u)
    ok = abs(al - (3 - 4 * x)) < mp.mpf(10) ** -40 and abs(be - (5 - 20 * x + 16 * x ** 2)) < mp.mpf(10) ** -40
    Dm = [sg[N - 1] - sg[N - 1 - m] for m in range(N)]
    ok &= all(abs((Dm[m] - Dm[m - 1]) - mp.sin((2 * m + 1) * u) / mp.sin(u)) < mp.mpf(10) ** -40 and Dm[m] > Dm[m - 1] for m in range(1, N))
    if N >= 5:
        ok &= (0 < al < be < 2 * al); drops = [0, al, 2 * al, al + be]; i = p - 3; par = D % 2
    elif N == 3:
        ok &= abs(al - 2) < mp.mpf(10) ** -40 and abs(be - 1) < mp.mpf(10) ** -40; drops = [0, 2, 3, 4]; i = p - 3; par = D % 2
    else:  # N == 4
        ok &= abs(al - (1 + mp.sqrt(2))) < mp.mpf(10) ** -40 and abs(be - al) < mp.mpf(10) ** -40 and abs(Dm[3] - Dm[2] - 1) < mp.mpf(10) ** -40
        drops = [0, al, 2 * al, 2 * al + 1, 3 * al]; i = p - 4; par = (D - 1) % 2
        ok &= all(len(g[3]) == 1 for g in vals)                    # every eigenvalue is pure when N = 4
    for j, dr in enumerate(drops):
        ok &= abs(vals[p - j][0] - (top - dr)) < mp.mpf(10) ** -30
    ok &= (1 <= i) and (i + 1 <= p - 1) and vals[i][3] == {par} and vals[i + 1][3] == {par}
    ok &= (d * Dm[N - 1] > drops[-1])                              # Lam_0 lies below the eigenvalues used
    return ok, i, par

C13 = {(r["d"], r["N"]): r for r in json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '13_certified_residue_signs.json')))["rows"]}
spec_ok = True; nspec = 0; specb_ok = True; nspecb = 0; sign_ok = True; nsign = 0; cert_ok = True; ncert = 0; n2_ok = True
crit_fail = []
cases = [(2, N) for N in range(2, 61)] + [(3, N) for N in range(2, 25)] + [(4, N) for N in range(2, 11)] + [(5, N) for N in range(2, 8)] + \
        [(6, N) for N in range(2, 6)] + [(d, 3) for d in range(7, 13)] + [(d, 2) for d in range(7, 13)] + [(7, 4), (8, 4)]
for (d, N) in cases:
    vals = box_spectrum(N, d)
    p = len(vals) - 1; D = d * (N - 1)
    claim = thm510c_claim(N, d)
    row = {"d": d, "N": N, "p": p, "D": D}
    if N == 2:
        n2_ok &= (p == d == D)
    bclaim = None
    if d >= 2 and N >= 3:
        okb, ib, parb = thm510b_check(N, d, vals)
        specb_ok &= okb; nspecb += 1; bclaim = (ib, parb)
        row.update(thm510b_index=ib, thm510b_parity=parb, thm510b_ok=bool(okb))
        if not okb: print("  SPECTRAL CLAIM (b) FAILS", d, N)
    if claim is not None:
        i, par, pat = claim
        g = vals[i]
        pure = (g[3] == {0}) if par == "even" else (g[3] == {1})
        alt_pair = ("+" if (i - 1) % 2 == 0 else "-") + ("+" if i % 2 == 0 else "-")
        okc = pure and 1 <= i <= p - 1 and alt_pair == pat
        # the value of Lam_i / Lam_1 claimed in the proof
        t = 4 * mp.cos(mp.pi / (2 * N)) ** 2
        want = t if d == 2 else (mp.mpf(4) if d >= 4 else 2 * t)
        okc &= abs(g[0] - want) < mp.mpf(10) ** -30
        spec_ok &= okc; nspec += 1
        row.update(claim_index=i, claim_parity=par, excluded=pat, claim_ok=bool(okc))
        if not okc: print("  SPECTRAL CLAIM (c) FAILS", d, N, claim, g[0], g[3])
    # counting criterion of Remark 5.11(4)
    eps = [(-1) ** i * sgn(vals[i][2]) for i in range(1, p + 1)]
    mminus = sum(1 for e in eps[:-1] if e <= 0)
    excluded = 2 * mminus > p - D
    row.update(m_minus=mminus, criterion_excludes_alternation=bool(excluded))
    OUT["criterion"].append({"d": d, "N": N, "p": p, "D": D, "m_minus": mminus, "excludes": bool(excluded)})
    if N >= 3 and d >= 2 and not excluded: crit_fail.append((d, N))
    if N == 2: n2_ok &= (mminus == 0)
    # residue signs in float64 where the spectrum is small and well separated
    if p <= 140:
        Lam = np.array([float(g[0]) for g in vals]); W = np.array([float(g[1]) for g in vals]); SW = np.array([float(g[2]) for g in vals])
        if np.min(np.diff(Lam)) > 1e-7:
            G = lambda s: np.sum(W / (Lam - s))
            sg = ""
            small = False
            for j in range(p):
                gap = Lam[j + 1] - Lam[j]; e = 1e-10 * gap
                while G(Lam[j] + e) > 0: e /= 8
                lo = Lam[j] + e; e = 1e-10 * gap
                while G(Lam[j + 1] - e) < 0: e /= 8
                s = brentq(G, lo, Lam[j + 1] - e, xtol=1e-300, rtol=1e-15)
                r = -np.sum(SW / (Lam - s)) / (s * np.sum(W / (Lam - s) ** 2))
                small |= abs(r) < 1e-9
                sg += "+" if r > 0 else "-"
            if not small:
                nsign += 1
                alt = "".join("+" if j % 2 == 0 else "-" for j in range(p))
                row["signs_float"] = sg
                if N == 2: sign_ok &= (sg == alt)
                else: sign_ok &= (sg != alt)
                if claim is not None: sign_ok &= (sg[claim[0] - 1:claim[0] + 1] != claim[2])
                if bclaim is not None:
                    ib, parb = bclaim; tr = [1 if c == "+" else -1 for c in sg[ib - 1:ib + 2]]
                    sign_ok &= (tr[0] <= tr[1] <= tr[2]) if parb == 0 else (tr[0] >= tr[1] >= tr[2])
                    row["thm510b_triple_float"] = sg[ib - 1:ib + 2]
                changes = sum(1 for i in range(p - 1) if sg[i] != sg[i + 1])
                sign_ok &= (changes >= D - 1) and (p - 1 - changes <= p - D)
                row.update(sign_changes_float=changes, negative_float=sg.count("-"))
                if (d, N) in C13:
                    ncert += 1; cert_ok &= (C13[(d, N)]["signs"] == sg)
    OUT["box"].append(row)
check(f"Theorem 5.10(b): drops Delta_m, alpha, beta, the values of Lam_p..Lam_(p-3) (Lam_(p-4) for N=4), index range, purity and common parity of Lam_i, Lam_(i+1); N=4: every eigenvalue pure ({nspecb} pairs (d,N) with d>=2, N>=3: d=2 N<=60, d=3 N<=24, d=4 N<=10, d=5 N<=7, d=6 N<=5, N=3 d<=12, d=7,8 N=4)", specb_ok and nspecb >= 90)
check(f"Theorem 5.10(c): index, purity, parity and value of the eigenvalue used in the proof ({nspec} pairs (d,N))", spec_ok)
check("Theorem 5.10(a): N = 2 gives p = d = D (d = 2..12)", n2_ok)
check(f"Theorem 5.10: float residue signs ({nsign} cases with p <= 140): alternation iff N = 2; the triple of (b) is monotone; the pair of (c) never has the excluded signs; >= D-1 sign changes", sign_ok)
check(f"float sign patterns agree with the certified patterns of script 13 ({ncert} cases)", cert_ok and ncert >= 15)
# the statistics quoted in Observation 14.2 (floating point; asserted here so that the text cannot drift from the data)
FL = {(r["d"], r["N"]): r for r in OUT["box"] if "signs_float" in r}
obs_ok = (len(FL) == nsign) and all(r["p"] <= 140 for r in FL.values())
obs_ok &= all((2, N) in FL for N in range(2, 17)) and all((3, N) in FL for N in range(2, 10)) and all((d, 3) in FL for d in range(2, 12))
obs_ok &= all(0.33 * r["p"] <= r["negative_float"] <= 0.6 * r["p"] for r in FL.values())
obs_ok &= all(FL[(2, N)]["D"] - 1 < FL[(2, N)]["sign_changes_float"] < FL[(2, N)]["p"] - 1 for N in range(7, 17))
obs_ok &= all(FL[(3, N)]["D"] - 1 < FL[(3, N)]["sign_changes_float"] < FL[(3, N)]["p"] - 1 for N in range(4, 10))
obs_ok &= all(FL[(d, 3)]["sign_changes_float"] == FL[(d, 3)]["D"] - 1 == 2 * d - 1 for d in range(2, 12))
check(f"Observation 14.2 (float): {len(FL)} pairs with p <= 140 incl. d=2 N<=16, d=3 N<=9; negative residues between 0.33 p and 0.6 p; "
      "D-1 < S < p-1 for d=2, 7<=N<=16 and d=3, 4<=N<=9; S = D-1 = 2d-1 for N=3, 2<=d<=11", obs_ok)
ncrit = sum(1 for (d, N) in cases if d >= 2 and N >= 3)
check("Remark 5.11(4): the counting criterion is consistent (it never excludes alternation for N = 2, where alternation holds) and excludes "
      f"alternation in {ncrit - len(crit_fail)} of the {ncrit} examined pairs with d >= 2, N >= 3",
      n2_ok and (3, 4) in crit_fail and (4, 4) in crit_fail and ncrit - len(crit_fail) >= 80, f"not excluded by the criterion: {sorted(crit_fail)}")
OUT["criterion_summary"] = {"examined": ncrit, "excluded": ncrit - len(crit_fail)}
OUT["criterion_not_excluding"] = sorted(crit_fail)
OUT["n_fail"] = NFAIL
json.dump(OUT, open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '18_alternation_checks.json'), "w"), indent=1, default=str)
print("ALL OK" if NFAIL == 0 else f"{NFAIL} FAILURES")
sys.exit(1 if NFAIL else 0)
