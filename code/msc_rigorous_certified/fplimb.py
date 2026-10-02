"""fplimb.py -- fast engine for the rounded phase: the SAME integer recurrences as fpcore.certify,

    L_{t+1} = floor(M L_t / D),   U_{t+1} = ceil(M U_t / D),

carried out on numpy int64 "limbs" (base 2^58).  Every intermediate int64 value is < 2^63
(see the bound in the comments), so the arithmetic is exact and the integers produced are
bit-identical to those of the reference implementation (compare the sha256 digests).
Trusted base: numpy int64 add / multiply / shift / and / divmod without overflow, Python integers.
"""
import numpy as np
from fpcore import coeffs, neighbours_of_target, exact_phase, to_fixed_point, digest

B = 58
MASK = (1 << B) - 1


class Limbs:
    """Vector of non-negative integers < 2^(B*nl - 4), stored as nl padded int64 arrays."""

    def __init__(self, d, N, nl):
        self.d, self.N, self.nl = d, N, nl
        self.p = [np.zeros((N + 2,) * d, dtype=np.int64) for _ in range(nl)]
        self.inner = (slice(1, N + 1),) * d

    def load(self, X):                       # X: object array of Python ints
        for j in range(self.nl):
            part = (X >> (B * j)) if j == self.nl - 1 else ((X >> (B * j)) & MASK)
            self.p[j][self.inner] = part.astype(np.int64)
        self.ghosts()

    def dump(self):                          # back to Python ints
        X = np.zeros((self.N,) * self.d, dtype=object)
        X[...] = 0
        for j in range(self.nl):
            X += self.p[j][self.inner].astype(object) * (1 << (B * j))
        return X

    def ghosts(self):
        """Ghost layer = copy of the adjacent wall layer (a cancelled move leaves the walker in place)."""
        N = self.N
        for a in self.p:
            for ax in range(self.d):
                am = np.moveaxis(a, ax, 0)
                am[0] = am[1]
                am[N + 1] = am[N]

    def at(self, idx):                       # Python int at a lattice site (0-based index)
        pidx = tuple(i + 1 for i in idx)
        return sum(int(self.p[j][pidx]) << (B * j) for j in range(self.nl))


def step(src, dst, cm, cs, D, ceil, work):
    """dst = floor(M src / D)  (ceil=False)  or  ceil(M src / D)  (ceil=True).

    Bounds.  Limbs of src are <= 2^58 - 1 (top limb <= 2^54), the stencil weights sum to D <= 15, so
    S <= 15*2^58 + 14; with the incoming remainder r <= D-1:  S + r*2^58 < 30*2^58 < 2^63.
    Long division by the small divisor D from the top limb down is exact; the quotient digits are
    < 2^59 and are normalised by one carry pass.
    """
    d, N, nl = src.d, src.N, src.nl
    S, W2, R = work
    inner = src.inner
    for j in range(nl - 1, -1, -1):
        Pj = src.p[j]
        first = True
        for ax in range(d):
            lo = tuple(slice(0, N) if a == ax else slice(1, N + 1) for a in range(d))
            hi = tuple(slice(2, N + 2) if a == ax else slice(1, N + 1) for a in range(d))
            if first:
                np.add(Pj[lo], Pj[hi], out=S)
                first = False
            else:
                S += Pj[lo]
                S += Pj[hi]
        if cm != 1:
            S *= cm
        if cs:
            np.multiply(Pj[inner], cs, out=W2)
            S += W2
        if ceil and j == 0:
            S += D - 1
        if j != nl - 1:
            R <<= B
            S += R
        np.divmod(S, D, out=(dst.p[j][inner], R))
    for j in range(nl - 1):                  # carry normalisation
        a = dst.p[j][inner]
        np.right_shift(a, B, out=W2)
        a &= MASK
        dst.p[j + 1][inner] += W2
    tgt = (N,) * d                           # padded index of the target
    for j in range(nl):
        dst.p[j][tgt] = 0
    dst.ghosts()


def all_less(A, Bv, work):
    """True iff A < Bv at every non-target site (lexicographic comparison of the limbs)."""
    nl, inner = A.nl, A.inner
    lt = A.p[nl - 1][inner] < Bv.p[nl - 1][inner]
    eq = A.p[nl - 1][inner] == Bv.p[nl - 1][inner]
    for j in range(nl - 2, -1, -1):
        lt |= eq & (A.p[j][inner] < Bv.p[j][inner])
        eq &= (A.p[j][inner] == Bv.p[j][inner])
    lt[(A.N - 1,) * A.d] = True
    return bool(lt.all())


def certify(d, N, qn, qd, P=112, G=48, tmax=10 ** 8, progress=None):
    cm, cs, D = coeffs(d, qn, qd)
    assert D <= 15 and P % B <= 54 and P > B
    nl = P // B + 1
    nb = neighbours_of_target(d, N)
    target = (N - 1,) * d
    kind, X, ts, ck = exact_phase(d, N, qn, qd, P, G, tmax)
    if kind == 'done':
        return X
    Lo, Uo = to_fixed_point(X, ts, D, P)
    L, U, Ln, Un = (Limbs(d, N, nl) for _ in range(4))
    L.load(Lo)
    U.load(Uo)
    del X, Lo, Uo
    work = tuple(np.zeros((N,) * d, dtype=np.int64) for _ in range(3))
    sL = sum(L.at(i) for i in nb)
    sU = sum(U.at(i) for i in nb)
    ck.rounded_step(ts, sL, sU, first=True)
    t = ts
    while True:
        step(L, Ln, cm, cs, D, False, work)
        step(U, Un, cm, cs, D, True, work)
        sLn = sum(Ln.at(i) for i in nb)
        sUn = sum(Un.at(i) for i in nb)
        ck.rounded_step(t + 1, sLn, sUn)
        # tail test U_{t+1} < L_t entrywise; cheap necessary condition first (target neighbours)
        if sUn < sL and all_less(Un, L, work):
            Lobj, Unobj = L.dump(), Un.dump()
            diff = Lobj - Unobj
            diff[target] = diff.max()
            return ck.finish(t, ts, diff.min(), digest(Lobj, Unobj))
        L, Ln = Ln, L
        U, Un = Un, U
        sL, sU = sLn, sUn
        t += 1
        if progress and t % progress == 0:
            print(f"   ... t={t}", flush=True)
        if t > tmax:
            raise RuntimeError("tmax reached")


if __name__ == "__main__":
    import json, sys, time
    d, N, qn, qd = (int(x) for x in sys.argv[1:5])
    t0 = time.time()
    c = certify(d, N, qn, qd)
    c["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(c))
