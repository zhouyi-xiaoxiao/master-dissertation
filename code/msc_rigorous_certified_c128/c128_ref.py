"""c128_ref.py -- pure-Python (integers only) mirror of fp128.c, written to validate the C arithmetic.

It follows the mathematical description, not the C code: no 58-bit or 42-bit limbs, no overflow considerations
(Python integers are unbounded), dictionaries of coordinates instead of index arrays.  For the same (d,N,q,red) it must
produce a file that is byte-identical with the output of fp128; any wrap-around, mis-rounded division or indexing error
in the C program would show up as a difference.

usage: c128_ref.py d N qn qd red out.bin
"""
import itertools, struct, sys
from math import gcd

P, G = 112, 48
MAGIC = 0x3832315046


# ---- (B) floating intervals: (m, e) with m = 0 or 2^62 <= m < 2^63
def fsum(terms, Dv, up):                     # terms: list of (c, (m, e));  result <= (sum c x)/Dv (up=False) or >= (up=True)
    terms = [(c, m, e) for c, (m, e) in terms if c and m]
    if not terms:
        return (0, 0)
    E = max(e for c, m, e in terms)
    acc = 0
    for c, m, e in terms:
        s = E - e
        v = -((-m) >> s) if up else m >> s           # ceil / floor of m / 2^s
        acc += c * v
    num = acc << 58
    q = -((-num) // Dv) if up else num // Dv
    s = q.bit_length() - 63
    assert s > 0
    m = -((-q) >> s) if up else q >> s
    if m == 1 << 63:
        m, s = m >> 1, s + 1
    return (m, E - 58 + s)


def less(x, y):                              # exact comparison of two numbers of the format
    if y[0] == 0:
        return False
    if x[0] == 0:
        return True
    return (x[1], x[0]) < (y[1], y[0])


def run(d, N, qn, qd, red):
    g = gcd(gcd(qn, 2 * d * (qd - qn)), 2 * d * qd)
    cm, cs, D = qn // g, 2 * d * (qd - qn) // g, 2 * d * qd // g
    # stored sites, in the order of the index x_1 + N x_2 + N^2 x_3
    sites = [x[::-1] for x in itertools.product(range(N), repeat=d)]
    if red:
        sites = [x for x in sites if all(x[k] <= x[k + 1] for k in range(d - 1))]
    a = (N - 1,) * d
    rep = (lambda y: tuple(sorted(y))) if red else (lambda y: y)

    def neighbours(x):                       # the 2d sites reached from x (x itself for a cancelled move)
        out = []
        for k in range(d):
            for s in (-1, 1):
                y = list(x)
                y[k] += s
                out.append(x if not 0 <= y[k] <= N - 1 else rep(tuple(y)))
        return out

    nbs = {x: neighbours(x) for x in sites}
    if red:
        tn = [((N - 2,) + (N - 1,) * (d - 1), d)]
    else:
        tn = [(tuple(N - 1 - (j == k) for j in range(d)), 1) for k in range(d)]
    x0 = (0,) * d

    # ---- (A) fixed point
    L = {x: 0 for x in sites}
    U = dict(L)
    L[x0] = U[x0] = 1 << P

    def step_fixed(L, U):
        Ln, Un = {}, {}
        for x in sites:
            if x == a:
                Ln[x] = Un[x] = 0
                continue
            Ln[x] = (cm * sum(L[y] for y in nbs[x]) + cs * L[x]) // D
            Un[x] = -((-(cm * sum(U[y] for y in nbs[x]) + cs * U[x])) // D)
        return Ln, Un

    lo = {x: (0, 0) for x in sites}
    hi = dict(lo)
    lo[x0] = hi[x0] = (1 << 62, -62)

    def step_float(lo, hi):
        lon, hin = {}, {}
        for x in sites:
            if x == a:
                lon[x] = hin[x] = (0, 0)
                continue
            lon[x] = fsum([(cm, lo[y]) for y in nbs[x]] + [(cs, lo[x])], D, False)
            hin[x] = fsum([(cm, hi[y]) for y in nbs[x]] + [(cs, hi[x])], D, True)
        return lon, hin

    sig = lambda lo, hi: (fsum([(c, lo[y]) for y, c in tn], 1, False), fsum([(c, hi[y]) for y, c in tn], 1, True))

    FL = [None, sum(c * L[y] for y, c in tn)]
    FU = [None, sum(c * U[y] for y, c in tn)]
    relf = [None]
    factive = FL[1] >> G == 0
    tsw = 0 if factive else 1
    slo, shi = sig(lo, hi)
    t = 1
    while True:
        Ln, Un = step_fixed(L, U)
        FL.append(sum(c * Ln[y] for y, c in tn))
        FU.append(sum(c * Un[y] for y, c in tn))
        r = "?"
        if factive:
            lo, hi = step_float(lo, hi)
            slon, shin = sig(lo, hi)
            if less(shi, slon):
                r = "<"
            elif less(shin, slo):
                r = ">"
            elif shi[0] == 0 and shin[0] == 0:
                r = "="
            slo, shi = slon, shin
            if FL[t + 1] >> G:
                factive, tsw = False, t + 1
        relf.append(r)
        if all(Un[x] < L[x] for x in sites if x != a):
            break
        L, U, t = Ln, Un, t + 1
    T = t
    n = len(sites)
    b = struct.pack("<15q", MAGIC, d, N, qn, qd, cm, cs, D, P, G, red, n, sites.index(a), T, tsw)
    b += b"".join(v.to_bytes(16, "little") for v in FL[1:T + 2])
    b += b"".join(v.to_bytes(16, "little") for v in FU[1:T + 2])
    b += "".join(relf[1:T + 1]).encode()
    b += b"".join(L[x].to_bytes(16, "little") for x in sites)
    b += b"".join(Un[x].to_bytes(16, "little") for x in sites)
    return b


if __name__ == "__main__":
    d, N, qn, qd, red = (int(x) for x in sys.argv[1:6])
    open(sys.argv[6], "wb").write(run(d, N, qn, qd, red))
