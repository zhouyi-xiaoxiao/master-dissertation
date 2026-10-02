"""explore17: lambda = (eta +- sqrt(r_k), k=1..M) plus (m-2M) copies of eta.   s_j = h_j(lambda).
Test: is Delta_j = s_j^2 - s_{j-1}s_{j+1} a polynomial in Y = eta^2 - eta_*^2 with nonneg coefficients,
where eta_*^2 = 2*sum(r)/(m(m-1))  (e_2 = 0)?   exact rationals (sympy)."""
import random
import sympy as sp
eta, Y, z = sp.symbols('eta Y z')
rng = random.Random(5)
def test(m, rs, J):
    M = len(rs)
    E = (1 - eta * z) ** (m - 2 * M)
    for r in rs:
        E *= ((1 - eta * z) ** 2 - r * z ** 2)
    E = sp.expand(E)
    ser = sp.series(1 / E, z, 0, J + 2).removeO()
    s = [sp.expand(ser.coeff(z, j)) for j in range(J + 2)]
    eta2 = sp.Rational(2) * sum(rs) / (m * (m - 1))
    allok = True
    for j in range(1, J + 1):
        D = sp.expand(s[j] ** 2 - s[j - 1] * s[j + 1])
        P = sp.Poly(D, eta)
        DY = sp.expand(sum(c * (Y + eta2) ** (mon[0] // 2) for mon, c in zip(P.monoms(), P.coeffs())))
        co = sp.Poly(DY, Y).all_coeffs()
        ok = all(c >= 0 for c in co)
        allok &= ok
        if not ok:
            return False, j, co
    return True, None, None
for trial in range(40):
    m = rng.randint(2, 8)
    M = rng.randint(1, m // 2)
    rs = [sp.Rational(rng.randint(1, 30), rng.randint(1, 10)) for _ in range(M)]
    ok, j, co = test(m, rs, 7)
    print(m, M, rs, ok, j, (co if not ok else ''), flush=True)
