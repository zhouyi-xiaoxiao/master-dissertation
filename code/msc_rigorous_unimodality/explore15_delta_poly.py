"""explore15: W_n(y) = ((y I + Adj(P_m))^n)(1,m);  Delta_n(y) = W_n^2 - W_{n-1} W_{n+1}.
Is Delta_n a polynomial in Y = y^2 - 2/m with nonnegative coefficients?  (exact, sympy)"""
import sympy as sp
y, Y = sp.symbols('y Y')
def Wseq(m, J):
    M = sp.zeros(m, m)
    for i in range(m):
        M[i, i] = y
        if i + 1 < m:
            M[i, i + 1] = 1; M[i + 1, i] = 1
    v = sp.zeros(m, 1); v[0] = 1
    out = []
    for n in range(m - 1 + J + 2):
        out.append(sp.expand(v[m - 1]))
        v = (M * v).applyfunc(sp.expand)
    return out
for m in range(2, 9):
    J = 9
    W = Wseq(m, J)
    res = []
    for j in range(1, J + 1):
        n = m - 1 + j
        D = sp.expand(W[n] ** 2 - W[n - 1] * W[n + 1])
        # substitute y^2 = Y + 2/m
        P = sp.Poly(D, y)
        assert all(mon[0] % 2 == 0 for mon in P.monoms())
        DY = sp.expand(sum(c * (Y + sp.Rational(2, m)) ** (mon[0] // 2) for mon, c in zip(P.monoms(), P.coeffs())))
        coeffs = sp.Poly(DY, Y).all_coeffs()
        res.append(all(c >= 0 for c in coeffs))
        if not res[-1]:
            print("  m=%d j=%d coeffs:" % (m, j), coeffs)
    print("m=%d: Delta_n nonneg-coefficient polynomial in Y for j=1..%d: %s" % (m, J, res), flush=True)
