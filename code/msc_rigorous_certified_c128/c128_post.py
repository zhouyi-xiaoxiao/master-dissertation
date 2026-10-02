"""c128_post.py -- turn the binary output of fp128 (C engine) into a certificate line, with Python integers only.

What is trusted from the C program: the integer sequences FL_t <= 2^P sigma_t <= FU_t (t = 1..T+1), the early
relations relf_t, and the two vectors L_T, U_{T+1}.  What is done here, in Python integers:
  * the relation rel_t between f(t) and f(t+1): '<' if FU_t < FL_{t+1}, '>' if FL_t > FU_{t+1}, else relf_t;
    (a conflict between a decided fixed-point relation and a decided floating relation would disprove soundness
    and raises an error);
  * the tail condition U_{T+1}(y) < L_T(y) at every stored non-target site is re-checked, and FL_T, FU_{T+1}
    are re-derived from the two vectors;
  * the verdicts of Proposition 'what a certificate proves' by the code of the generator (fpcore.Checker.finish);
  * a rational gamma < 1 with U_{T+1} <= gamma L_T, hence v_{T+1} <= gamma v_T.

usage: c128_post.py out.bin            (prints the JSON certificate line)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import hashlib, json, os, struct, sys
from fractions import Fraction
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_rigorous_certified'))
from fpcore import Checker

MAGIC = 0x3832315046


def read_bin(path):
    b = open(path, "rb").read()
    magic, d, N, qn, qd, cm, cs, D, P, G, red, n, tgt, T, tsw = struct.unpack_from("<15q", b, 0)
    assert magic == MAGIC
    off = 120

    def u128(cnt):
        nonlocal off
        out = [int.from_bytes(b[off + 16 * i:off + 16 * i + 16], "little") for i in range(cnt)]
        off += 16 * cnt
        return out

    FL, FU = u128(T + 1), u128(T + 1)
    relf = b[off:off + T].decode("ascii")
    off += T
    L, U = u128(n), u128(n)
    assert off == len(b), "trailing bytes"
    return dict(d=d, N=N, qn=qn, qd=qd, cm=cm, cs=cs, D=D, P=P, G=G, reduced=red, n=n, tgt=tgt, T=T, tsw=tsw,
                FL=FL, FU=FU, relf=relf, L=L, U=U, sha256=hashlib.sha256(b).hexdigest())


def certificate(path):
    r = read_bin(path)
    d, N, T, n, tgt = r["d"], r["N"], r["T"], r["n"], r["tgt"]
    ck = Checker(d, N, r["qn"], r["qd"], r["P"], r["G"])
    assert (ck.cm, ck.cs, ck.D) == (r["cm"], r["cs"], r["D"]) and r["P"] == 112
    full = N ** d
    assert tgt == n - 1 and n == (len([1 for i in range(full) if sorted_ok(i, N, d)]) if r["reduced"] else full)
    FL, FU = [None] + r["FL"], [None] + r["FU"]                 # index t = 1..T+1
    assert all(0 <= a <= b <= d << r["P"] for a, b in zip(r["FL"], r["FU"]))
    rel, from_float = [None], 0
    for t in range(1, T + 1):
        fx = "<" if FU[t] < FL[t + 1] else ">" if FL[t] > FU[t + 1] else "?"
        fl = r["relf"][t - 1]
        assert fl in "<>=?"
        if fx != "?" and fl != "?" and fx != fl:
            raise RuntimeError(f"conflict between fixed-point and floating relation at t={t}")
        if fl == "=":
            assert FU[t] == 0 and FU[t + 1] == 0               # '=' is only issued for exact zeros
        rel.append(fx if fx != "?" else fl)
        from_float += fx == "?" and fl != "?"
    ck.FL, ck.FU, ck.rel = FL, FU, rel
    # the tail condition, re-checked here; and the link between the vectors and the recorded sums
    L, U = r["L"], r["U"]
    assert L[tgt] == 0 and U[tgt] == 0
    assert all(U[i] < L[i] for i in range(n) if i != tgt), "tail condition fails"
    if r["reduced"]:
        nbs = [(n - 2, d)]                                      # representative (N-2,N-1,..,N-1), multiplicity d
    else:
        nbs = [(full - 1 - N ** k, 1) for k in range(d)]
    assert FL[T] == sum(c * L[i] for i, c in nbs) and FU[T + 1] == sum(c * U[i] for i, c in nbs)
    margin = min(L[i] - U[i] for i in range(n) if i != tgt)
    ib = max((i for i in range(n) if i != tgt), key=lambda i: Fraction(U[i], L[i]))
    g = 1 - Fraction(U[ib], L[ib])                              # 1 - gamma > 0
    e = 0
    while g * 10 ** e < 100:
        e += 1
    c = ck.finish(T, r["tsw"], margin, r["sha256"])
    c.update(engine="c128", reduced=bool(r["reduced"]), stored_sites=n, relations_from_float_phase=from_float,
             one_minus_gamma_lower=f"{(g * 10 ** e).__floor__()}e-{e}")
    return c


def sorted_ok(i, N, d):
    x = []
    for _ in range(d):
        x.append(i % N)
        i //= N
    return all(x[k] <= x[k + 1] for k in range(d - 1))


if __name__ == "__main__":
    print(json.dumps(certificate(sys.argv[1])))
