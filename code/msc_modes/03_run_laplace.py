"""
03_run_laplace.py -- exact continuous-time first-passage densities for large N
via the renewal ratio F^(s) = P^(a,s|x0)/P^(a,s|a) (reflecting-box propagator
reduced to a (d-1)-fold sum with the closed-form 1D lattice resolvent) and
fixed-Talbot inversion.  Unit jump rate (q = 1): for jump rate q the density is
q g(q t), so mode_q = mode_1/q and MFPT_q = MFPT_1/q exactly.

For every (d, geometry, N) we record
  * the exact mode t* (root of g'(t) located by bracketing + zoom),
  * a unimodality check on a wide logarithmic grid,
  * the exact MFPT,
  * the lowest poles nu_0 < nu_1 < nu_2 of F^ (zeros of P^(a,s|a) on the
    negative real axis) and their residues a_j, and the two-/three-pole
    predictions of the mode,
  * closed-form small-target predictions
        L0:  t* = ln(A mu MFPT)/mu
        L1:  t* = MFPT ln(A X)/(X + W - 1),   X = mu MFPT.

Usage: python 03_run_laplace.py GROUP [GROUP ...]   (restartable; appends to
data/laplace_modes.jsonl)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time, math, itertools
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
import fptlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_modes')
OUT = _os.path.join(_R, 'data', 'msc_modes', 'laplace_modes.jsonl')
MT = 20   # Talbot nodes (validated: rel. error ~1e-13 near the mode in double precision)


def halfoct(lo, hi):
    out, x = [], float(lo)
    while x <= hi * 1.0001:
        out.append(int(round(x)))
        x *= math.sqrt(2.0)
    return sorted(set(out))


def odd(v):
    return [n if n % 2 == 1 else n + 1 for n in v]


N2_CC = sorted(set([5, 7] + halfoct(10, 10240) + [20480, 40960, 81920, 163840, 327680, 655360, 1048576]
                   + [35, 50, 100, 200, 400]))
N2_ODD = sorted(set(odd([5, 7] + halfoct(10, 10240) + [40960, 163840, 655360])))
N3_CC = sorted(set([5, 7] + halfoct(10, 2896) + [4096] + [35, 50, 100, 200]))
N3_ODD = sorted(set(odd([5, 7] + halfoct(10, 1448))))
N1 = sorted(set([5, 7] + halfoct(10, 10240) + [100, 200, 1000, 100000, 1000000]))

GROUPS = {
    "2d_cc": [(2, "CC", N) for N in N2_CC],
    "2d_cc_huge": [(2, "CC", N) for N in (2097152, 4194304, 8388608, 16777216)],
    "2d_c2m": [(2, "C2M", N) for N in N2_ODD],
    "2d_m2c": [(2, "M2C", N) for N in N2_ODD],
    "3d_cc": [(3, "CC", N) for N in N3_CC],
    "3d_c2m": [(3, "C2M", N) for N in N3_ODD],
    "3d_m2c": [(3, "M2C", N) for N in N3_ODD],
    "1d": [(1, "CC", N) for N in N1],
    # sizes used in the q-scan of 02_run_discrete.py and the N = 4001 lattice
    "extra_pts": [(2, "CC", 21), (2, "CC", 51), (2, "CC", 101), (3, "CC", 11), (3, "CC", 21), (3, "CC", 31),
                  (1, "CC", 51), (1, "CC", 201), (2, "CC", 4001)],
}


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line)
                done.add((r["d"], r["geo"], r["N"]))
    return done


# --------------------------------------------------------------------------
def target_poles(L, kmax=6):
    """smallest distinct reflecting eigenvalues whose eigenfunctions do not
    vanish at the target (poles of P^(a,s|a)), with total weight
    W = N^d * sum phi_k(a)^2 over the degenerate set."""
    N, d = L.N, L.d
    k = np.arange(kmax + 1)
    ek = L.ek[: kmax + 1]
    wt1 = []
    for i in range(d):
        m = (k * (2 * L.target[i] + 1)) % (4 * N)
        c = np.cos(np.pi * m / (2.0 * N))
        wt1.append(np.where(k == 0, 1.0, 2.0) * c * c)
    amp1 = []
    for i in range(d):
        m1 = (k * (2 * L.target[i] + 1)) % (4 * N)
        m2 = (k * (2 * L.start[i] + 1)) % (4 * N)
        amp1.append(np.where(k == 0, 1.0, 2.0) * np.cos(np.pi * m1 / (2.0 * N)) * np.cos(np.pi * m2 / (2.0 * N)))
    vals = {}
    for ks in itertools.product(range(min(kmax, N - 1) + 1), repeat=d):
        if sum(ks) == 0:
            continue
        e = sum(ek[j] for j in ks)
        W = float(np.prod([wt1[i][ks[i]] for i in range(d)]))
        Aamp = float(np.prod([amp1[i][ks[i]] for i in range(d)]))
        key = None
        for kk in vals:
            if abs(kk - e) <= 1e-9 * e:
                key = kk
                break
        if key is None:
            key = e
            vals[key] = [0.0, 0.0]
        vals[key][0] += W
        vals[key][1] += Aamp
    items = sorted(vals.items())
    poles = [(e, W, Aamp) for e, (W, Aamp) in items if W > 1e-10]
    return poles


def pole_data(L, tau):
    """nu_0,nu_1,nu_2 (zeros of P^_aa) and residues a_j of F^."""
    poles = target_poles(L)
    p = [pp[0] for pp in poles[:3]]
    L.prepare(1.3 * p[2])
    vol = float(L.N) ** L.d

    def Paa(s):
        return L._P(np.array([s], dtype=complex), "aa")[0].real

    def D(s):
        return vol * s * Paa(s)

    eta = 1e-9
    nus = []
    nus.append(-brentq(D, -p[0] * (1 - eta), -1e-3 / tau, xtol=1e-300, rtol=1e-15, maxiter=300))
    for j in (0, 1):
        nus.append(-brentq(Paa, -p[j + 1] * (1 - eta), -p[j] * (1 + eta), xtol=1e-300, rtol=1e-15, maxiter=300))
    # residues by a small circle (trapezoid rule, spectrally accurate)
    res = []
    marks = sorted([0.0] + p + nus)
    for nu in nus:
        gap = min(abs(nu - m) for m in marks if abs(m - nu) > 1e-14 * nu)
        rc = 0.35 * gap
        n = 32
        z = rc * np.exp(2j * np.pi * (np.arange(n) + 0.5) / n)
        vals = L.Fhat(-nu + z)
        res.append(float((vals * z).mean().real))
    return poles[:3], nus, res


def multi_exp_mode(nus, amps, t0):
    """mode of sum_j a_j exp(-nu_j t) near t0 (root of derivative)."""
    nus = np.array(nus); amps = np.array(amps)

    def dg(t):
        return float((-nus * amps * np.exp(-nus * t)).sum())
    lo, hi = 0.5 * t0, 2.0 * t0
    if dg(lo) * dg(hi) > 0:
        return float("nan")
    return brentq(dg, lo, hi, rtol=1e-14)


def find_mode_talbot(L, t0, tau):
    """bracket + zoom on g'(t) (Talbot inversion of s F^(s))."""
    js = np.arange(-12, 33)
    tg = t0 * 2.0 ** (js / 4.0)
    tg = tg[tg <= 15.0 * tau]
    L.prepare(2.0 * MT * MT / (5.0 * tg.min()) * 1.05)
    g = F.talbot_invert(L.Fhat, tg, M=MT)
    dg = F.talbot_invert(L.Fhat, tg, M=MT, deriv=1)
    gmax = g.max()
    ok = g > 1e-7 * gmax
    idx = np.where(ok)[0]
    sc = [i for i in idx[:-1] if ok[i + 1] and dg[i] > 0 and dg[i + 1] <= 0]
    sc_up = [i for i in idx[:-1] if ok[i + 1] and dg[i] < 0 and dg[i + 1] >= 0]
    if len(sc) == 0:
        return None
    # if several, take the one with the largest g
    i = max(sc, key=lambda ii: max(g[ii], g[ii + 1]))
    tl, tr = tg[i], tg[i + 1]
    for _ in range(16):
        tt = np.linspace(tl, tr, 9)
        dd = F.talbot_invert(L.Fhat, tt, M=MT, deriv=1)
        j = np.where((dd[:-1] > 0) & (dd[1:] <= 0))[0]
        if len(j) == 0:
            break
        j = j[0]
        tl, tr, dl, dr = tt[j], tt[j + 1], dd[j], dd[j + 1]
        if (tr - tl) / tr < 1e-11:
            break
    tstar = tl + (tr - tl) * dl / (dl - dr)
    gstar = float(F.talbot_invert(L.Fhat, [tstar], M=MT)[0])
    # half-maximum crossing on the rise (coarse, from the grid, log-interp)
    rise = float("nan")
    below = np.where((g[:-1] < 0.5 * gstar) & (g[1:] >= 0.5 * gstar))[0]
    if len(below):
        k = below[0]
        w = (0.5 * gstar - g[k]) / (g[k + 1] - g[k])
        rise = float(tg[k] * (tg[k + 1] / tg[k]) ** w)
    return {"mode": float(tstar), "g_max": gstar, "n_sign_changes_down": len(sc), "n_sign_changes_up": len(sc_up),
            "grid_tmin": float(tg.min()), "grid_tmax": float(tg.max()), "bracket_rel_width": float((tr - tl) / tr),
            "t_half_rise": rise}


def run_point(d, geo, N):
    t00 = time.time()
    s0, a0 = F.GEOMETRIES[geo][0](N, d), F.GEOMETRIES[geo][1](N, d)
    rec = {"d": d, "geo": geo, "N": N, "rate": 1.0, "start": list(s0), "target": list(a0)}
    if d == 1:
        ch = F.Chain1D(N, 1.0)
        tau = ch.mfpt()
        t0 = (N - 0.5) ** 2 / 3.0
        ts = brentq(lambda t: ch.dg_cont([t])[0], 0.5 * t0, 1.6 * t0, rtol=1e-14)
        rec.update({"mode": float(ts), "g_max": float(ch.g_cont([ts])[0]), "mfpt": tau,
                    "method": "closed-form spectral sum (Chain1D)", "n_sign_changes_down": 1, "n_sign_changes_up": 0})
        # unimodality on a grid
        tg = t0 * 2.0 ** (np.arange(-20, 25) / 4.0)
        dg = ch.dg_cont(tg); g = ch.g_cont(tg)
        ok = g > 1e-9 * g.max()
        rec["n_sign_changes_down"] = int(((dg[:-1] > 0) & (dg[1:] <= 0) & ok[:-1] & ok[1:]).sum())
        # Laplace/Talbot cross-check for moderate N
        if N <= 20000:
            L = F.LaplaceFP(N, 1, 1.0, s0, a0)
            r = find_mode_talbot(L, t0, tau)
            rec["mode_talbot_check"] = r["mode"]
            rec["mfpt_laplace_check"] = float(L.mfpt())
        # exact discrete-time mode at q = 0.8 from the closed form (argmax over integers).
        # Delta(t) = f(t+1) - f(t) = -sum_j W_j mu_j lambda_j^(t-1) is evaluated directly
        # (well conditioned, unlike a numerical difference of f); the integer argmax is the
        # smallest integer m with Delta(m) <= 0.
        q = 0.8
        chq = F.Chain1D(N, q)
        pos = chq.lam > 0
        # log1p(-mu), not log(1 - mu): forming 1 - mu in double precision loses the digits that decide the
        # integer mode once mu ~ N^-2 is below about 1e-9 (with log(1 - mu) the located maximum moves by 353 and
        # 385 703 steps at N = 1e5 and 1e6; code/article/s3_methods_log1p_demo.py)
        Wp, mup, lap = chq.W[pos], chq.mu[pos], np.log1p(-chq.mu[pos])

        def delta(t):
            return float(-(Wp * mup * np.exp((t - 1.0) * lap)).sum())
        if t0 / q > 400:
            tr = brentq(delta, 0.5 * t0 / q, 1.6 * t0 / q, rtol=1e-15)
            rec["mode_discrete_q0.8"] = int(math.ceil(tr))
            rec["mode_discrete_q0.8_real_root_of_increment"] = float(tr)
            rec["neg_eigen_bound_at_mode"] = chq.neg_bound(tr)
        else:
            cand = np.arange(1, int(3 * t0 / q) + 10)
            fv = chq.f_discrete(cand)
            rec["mode_discrete_q0.8"] = int(cand[np.argmax(fv)])
        rec["mfpt_discrete_q0.8"] = N * (N - 1) / q
        rec["wall_s"] = time.time() - t00
        return rec
    L = F.LaplaceFP(N, d, 1.0, s0, a0)
    tau = float(L.mfpt())
    rec["mfpt"] = tau
    poles, nus, res = pole_data(L, tau)
    mu, W, Aamp = poles[0]
    rec.update({"p": [pp[0] for pp in poles], "W": [pp[1] for pp in poles], "A_u": [-pp[2] for pp in poles],
                "nu": nus, "res": res})
    X = mu * tau
    A = -Aamp
    # closed-form predictions (only meaningful when the slowest target pole also carries the arrival, A>0)
    if A > 1e-9:
        rec["pred_L0"] = math.log(A * X) / mu
        rec["pred_L1"] = tau * math.log(A * X) / (X + W - 1.0)
    else:
        rec["pred_L0"] = None
        rec["pred_L1"] = None
    # pole-based predictions
    t2 = float("nan")
    if res[1] < 0:
        t2 = math.log(-res[1] * nus[1] / (res[0] * nus[0])) / (nus[1] - nus[0])
    rec["pred_2pole"] = t2
    guess = t2 if np.isfinite(t2) else (rec["pred_L0"] or 1.0 / mu)
    t3 = multi_exp_mode(nus, res, guess)
    rec["pred_3pole"] = t3
    if not np.isfinite(guess) or guess <= 0:
        guess = 3.0 / mu
    r = find_mode_talbot(L, guess if not np.isfinite(t3) else t3, tau)
    if r is None:
        rec["mode"] = None
    else:
        rec.update(r)
    rec["method"] = f"renewal ratio + fixed Talbot (M={MT})"
    rec["wall_s"] = time.time() - t00
    return rec


def main():
    done = load_done()
    for g in sys.argv[1:]:
        for (d, geo, N) in GROUPS[g]:
            if (d, geo, N) in done:
                continue
            rec = run_point(d, geo, N)
            rec["group"] = g
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            done.add((d, geo, N))
            m = rec.get("mode")
            print(f"[{g}] d={d} {geo} N={N}: mode={m} mfpt={rec['mfpt']:.6g} ratio={(m / rec['mfpt']) if m else None} "
                  f"L0={rec.get('pred_L0')} L1={rec.get('pred_L1')} 2p={rec.get('pred_2pole')} 3p={rec.get('pred_3pole')} "
                  f"nsc={rec.get('n_sign_changes_down')} wall={rec['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
