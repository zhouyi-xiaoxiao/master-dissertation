"""mfpt_enclosure.py -- rigorous rational enclosures of tau_N = q * MFPT (corner to opposite corner).

Lemma 19 of note R5.  K = 2d I - (A + B) on the non-target sites (K = 2d (I - Q) at q = 1) has a
non-negative inverse, and tau = q * (E_x T)_x solves K tau = 2d 1.  Hence for ANY vector z with
R = K z > 0 entrywise:     2d z / max(R)  <=  tau  <=  2d z / min(R)   entrywise.

A float64 guess (DCT solve + two refinement steps) is turned into an integer vector z at scale 2^S;
R = K z is then computed EXACTLY in Python integers.  Only the last step is trusted.

usage: mfpt_enclosure.py d Nmin Nmax      -> certs/mfpt_d{d}.jsonl   (restartable)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, math, os, sys, time
from fractions import Fraction
import numpy as np
from scipy.fft import dctn, idctn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fpcore import apply_M

S = 96   # fixed-point scale of the integer guess


def neumann_solve(rhs, d, N):
    """Float solution u (up to a constant) of (2d I - A_full - B) u = rhs, sum(rhs) = 0."""
    k = np.arange(N)
    lam1 = 2.0 - 2.0 * np.cos(np.pi * k / N)
    lam = sum(np.reshape(lam1, [N if a == ax else 1 for a in range(d)]) for ax in range(d))
    h = dctn(rhs, type=2, norm="ortho")
    lam[(0,) * d] = 1.0
    h /= lam
    h[(0,) * d] = 0.0
    return idctn(h, type=2, norm="ortho")


def K_apply_exact(z, d, N):
    """Exact K z on the non-target sites (z object array with z[target] = 0)."""
    R = 2 * d * z - apply_M(z, d, N, 1, 0)
    return R


def enclosure(d, N):
    target = (N - 1,) * d
    shape = (N,) * d
    z = np.zeros(shape, dtype=object)
    z[...] = 0
    for it in range(4):
        # exact residual of the current integer guess, as floats:  rho = 2d - K z / 2^S
        R = K_apply_exact(z, d, N)
        rho_int = (2 * d << S) - R
        rho = np.array([float(Fraction(int(x), 1 << S)) for x in rho_int.ravel()]).reshape(shape)
        rho[target] = 0.0
        rho[target] = -rho.sum()                     # solvability of the Neumann problem
        u = neumann_solve(rho, d, N)
        c = u - u[target]                            # correction with c[target] = 0
        dz = np.array([int(Fraction(float(x)) * (1 << S)) for x in c.ravel()], dtype=object).reshape(shape)
        z = z + dz
        z[target] = 0
    R = K_apply_exact(z, d, N)
    mask = np.ones(shape, dtype=bool)
    mask[target] = False
    Rmin, Rmax = min(R[mask]), max(R[mask])
    assert Rmin > 0
    z0 = int(z[(0,) * d])
    lo, hi = Fraction(2 * d * z0, int(Rmax)), Fraction(2 * d * z0, int(Rmin))
    return lo, hi


def dec(fr, digits, up):
    """Decimal string with `digits` decimals, rounded down (up=False) or up (up=True)."""
    sc = 10 ** digits
    n = fr.numerator * sc
    v = -((-n) // fr.denominator) if up else n // fr.denominator
    s = str(v).rjust(digits + 1, "0")
    return s[:-digits] + "." + s[-digits:]


if __name__ == "__main__":
    d, Nmin, Nmax = (int(x) for x in sys.argv[1:4])
    out = os.path.join(_R, 'data', 'msc_rigorous_certified', f"mfpt_d{d}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["N"] for l in open(out)}
    for N in range(Nmin, Nmax + 1):
        if N in done:
            continue
        t0 = time.time()
        lo, hi = enclosure(d, N)
        rec = dict(d=d, N=N, tau_lo=f"{lo.numerator}/{lo.denominator}", tau_hi=f"{hi.numerator}/{hi.denominator}",
                   tau_lo_dec=dec(lo, 30, False), tau_hi_dec=dec(hi, 30, True),
                   rel_width=float((hi - lo) / lo), seconds=round(time.time() - t0, 2))
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(d, N, rec["tau_lo_dec"], rec["tau_hi_dec"], f"{rec['rel_width']:.2e}", rec["seconds"], flush=True)
