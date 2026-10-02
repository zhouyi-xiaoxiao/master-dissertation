"""Theorem 7.4, computer-assisted part: exact location of the mode(s) of the simple random walk (q = 1) for 3 <= N <= NMAX.

For each N and each parity class D in {N-1 (near), N (far)} the class sequence t -> f(t), t = D mod 2, is unimodal
(Theorem 3.10): Psi_D(t) = Phi_D(t+2) - Phi_D(t) is > 0 before its unique zero t0 > D-2 and < 0 after, where
  Phi_D(t) = sum_{k=1}^{N-1} sin(th_k) sin(D th_k) cos(th_k)^(t-1),  th_k = k pi/(2N-1),   f(t) = 2 Phi_D(t)/(2N-1).
We certify in ball arithmetic  Psi_D(t_D - 2) > 0 > Psi_D(t_D)  (so the class mode is t_D, unique), then compare
f(t_near) with f(t_far).  Output: global mode, its class, and t* - tau1,  tau1 = c L^2 - b L + 1 + s2.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import core
from core import arb
from lib1d import pmf_stepper

core.set_prec(256)
PI = arb.pi()
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 600
d16 = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '16_q1_theorem.json')))
const = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '03_constants.json')))
c = arb(const["c_mid_80_digits"][:60], "1e-55")
b = arb(d16["b"]); s2 = arb(d16["s2"])


def Phi(N, D, t):
    W = 2 * N - 1
    s = arb(0)
    for k in range(1, N):
        th = k * PI / W
        s += th.sin() * (D * th).sin() * th.cos() ** (t - 1)
    return s


def exact_modes(N):
    """all global maximisers and the class modes, by exact dyadic arithmetic (integer path counts)."""
    n = N - 1
    T = 3 * N * N
    v = [0] * n
    v[0] = 1
    cnt = [0] * (T + 1)                 # f(t) = cnt[t] / 2^t
    for t in range(1, T + 1):
        cnt[t] = v[n - 1]
        w = [0] * n
        for j in range(n):
            if j > 0:
                w[j] += v[j - 1]
            if j < n - 1:
                w[j] += v[j + 1]
        w[0] += v[0]
        v = w
    val = [cnt[t] << (T - t) for t in range(T + 1)]      # f(t) * 2^T, exact integers
    res = {}
    for name, D in (("near", N - 1), ("far", N)):
        idx = list(range(D, T - 1, 2))
        m = max(val[t] for t in idx)
        ms = [t for t in idx if val[t] == m]
        # class unimodality (Thm 3.10) guarantees no later maximum provided the sequence has started to decrease
        assert val[idx[-1]] < m and ms[-1] < idx[-3]
        res[name] = (ms, m)
    return res


rows = []
allok = True
lo_min, hi_max = 1e9, -1e9
t00 = time.time()
for N in range(3, NMAX + 1):
    L = N - 0.5
    tau1 = c * arb(2 * N - 1) ** 2 / 4 - b * arb(2 * N - 1) / 2 + 1 + s2
    rec = dict(N=N)
    ok = True
    cls = {}
    if N <= 14:
        ex = exact_modes(N)
        cls = {k: v[0] for k, v in ex.items()}
        if ex["near"][1] > ex["far"][1]:
            modes, mcls = ex["near"][0], "near"
        elif ex["near"][1] < ex["far"][1]:
            modes, mcls = ex["far"][0], "far"
        else:
            modes, mcls = sorted(ex["near"][0] + ex["far"][0]), "both"
        rec["method"] = "exact"
    else:
        T = int(0.9 * L * L) + 12
        f = pmf_stepper(N, 1.0, 1, T + 4)
        for name, D in (("near", N - 1), ("far", N)):
            idx = np.arange(D, T, 2)
            tg = int(idx[np.argmax(f[idx])])
            found = None
            for cand in (tg, tg - 2, tg + 2, tg - 4, tg + 4):
                if cand < D:
                    continue
                right = Phi(N, D, cand + 2) - Phi(N, D, cand) < 0
                left = True if cand == D else (Phi(N, D, cand) - Phi(N, D, cand - 2) > 0)
                if right and left:
                    found = cand
                    break
            if found is None:
                ok = False
            cls[name] = [found]
        modes, mcls = None, None
        if ok:
            vn = Phi(N, N - 1, cls["near"][0]); vf = Phi(N, N, cls["far"][0])
            if vn > vf:
                modes, mcls = cls["near"], "near"
            elif vf > vn:
                modes, mcls = cls["far"], "far"
            else:
                ok = False
        rec["method"] = "ball"
    if ok:
        offs = [float((arb(m) - tau1).mid()) for m in modes]
        rec.update(modes=modes, cls=mcls, t_near=cls["near"], t_far=cls["far"], modes_minus_tau1=[round(o, 6) for o in offs],
                   L_times_excess=round(float(L * max(max(-o, o - 2, 0.0) for o in offs)), 4))
        if mcls == "near":
            lo_min = min(lo_min, min(offs)); hi_max = max(hi_max, max(offs))
    rec["certified"] = ok
    allok &= ok
    rows.append(rec)
    if N <= 16 or N % 50 == 0:
        print(rec, flush=True)
notnear = [r["N"] for r in rows if r.get("cls") != "near"]
multi = [r["N"] for r in rows if r.get("modes") and len(r["modes"]) > 1]
excess = max(r.get("L_times_excess", 0) for r in rows)
print(f"all certified: {allok};  N with mode not in near class: {notnear};  N with several modes: {multi};  "
      f"range of (mode - tau1): [{lo_min:.4f}, {hi_max:.4f}];  max L*excess beyond [0,2]: {excess}   [{time.time()-t00:.0f}s]")
json.dump(dict(NMAX=NMAX, all_certified=allok, not_near=notnear, several_modes=multi, min_off=lo_min, max_off=hi_max,
               max_L_excess=excess, rows=rows),
          open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '17_q1_small_N.json'), "w"), indent=1)
