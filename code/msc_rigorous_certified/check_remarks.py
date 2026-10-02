"""check_remarks.py -- exact and interval checks of remarks of note R5 (proofs/R5_certified_computations/)
(none of them is a computer-assisted theorem; each one is either proved in the text or a count read off the
certificate files).  Output: certs/remark_checks.json.

 (a) Remark rem:lc, N = 3: for d = 1, N = 3 and rational q the identity
         f(t)^2 - f(t-1) f(t+1) = (q^4/16) det(Q)^(t-2),   det(Q) = 1 - 3q/2 + q^2/4,   t >= 2,
     checked in exact arithmetic for t <= 80 and several q on both sides of the threshold 3 - sqrt(5); and the set of
     t <= 80 where log-concavity fails.
 (b) Remark rem:lc, N = 4, q = 4/5: the set of t <= H at which f(t)^2 < f(t-1) f(t+1) (exact arithmetic).
 (c) Remark rem:cfX: the continuum value ln(pi^2)/((pi^2/2 + 1) kappa_1) - 1, with mpmath.iv from the certified
     enclosure of kappa_1 (certs/constants.json).
 (d) Conjecture conj:last: counts over the C certificates with q in {4/5, 1/2} (certs/c128_d*_q4-5.jsonl,
     certs/c128_d*_q1-2.jsonl): number of certificates, of non-exceptional ones, of those with T = t* (t* the smallest
     maximiser), and with T = the largest maximiser (exceptional triples: exact data of Theorem modes (d), re-read from
     certs/packedexact_*.jsonl).
 (e) Theorem rem / Proposition rem: the junction sup r_301 < inf r_302 from certs/mfpt2d_remainder.json (outward-rounded
     decimals written by scripts/mfpt2d_remainder.py).
 (g) Remarks rem:cfshape, rem:cfX and the paragraph after Corollary cor:qscaling: signs of (t_cf - t*)/t*, the values of N
     where it fails to decrease (d = 3), its value at N = 2 and at the largest N (certs/closedform_*.jsonl,
     certs/closedform_c128_*.jsonl); the largest mode on the ranges of Corollary cor:qscaling and the largest value of
     (4/5) t*(4/5) - t*(1) for d = 1, N <= 200 (certs/modes_d1_*.jsonl).
 (f) Section sec:ext: ratios of the recorded wall-clock times (field 'seconds', same loaded machine) of the generator
     (certs/modes_*.jsonl) and of the Python-integer checker (certs/packed_*.jsonl) to those of the C engine
     (certs/c128_*.jsonl), on the upper parts of the ranges of Table tab:ranges.
usage: check_remarks.py
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
from fractions import Fraction
from mpmath import iv, mpf
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
CERT = _os.path.join(_R, 'data', 'msc_rigorous_certified')
out = {}


def pmf_1d(N, q, tmax):
    """exact f(1..tmax) for d = 1 from the definition (sites 0..N-2, target N-1, reflecting end at 0)."""
    n = N - 1
    v = [Fraction(0)] * n
    v[0] = Fraction(1)
    f = [Fraction(0)]                                   # f[0] := 0
    for t in range(1, tmax + 1):
        f.append(v[n - 1] * q / 2)                      # P(T = t) = P(X_{t-1} = N-2, T > t-1) * q/2
        w = [Fraction(0)] * n
        for y in range(n):
            p = v[y]
            if not p:
                continue
            stay = 1 - q + (q / 2 if y == 0 else 0)     # rest, or the cancelled move at the reflecting end
            w[y] += p * stay
            if y >= 1:
                w[y - 1] += p * q / 2
            if y + 1 <= n - 1:
                w[y + 1] += p * q / 2                   # the move from N-2 to N-1 is the absorption
        v = w
    return f


# ---------------- (a) N = 3
qs = [Fraction(1, 2), Fraction(3, 4), Fraction(76, 100), Fraction(7639, 10000), Fraction(7640, 10000), Fraction(77, 100),
      Fraction(4, 5), Fraction(9, 10), Fraction(1)]
H3 = 80
resa = {}
for q in qs:
    f = pmf_1d(3, q, H3 + 1)
    det = 1 - Fraction(3, 2) * q + q * q / 4
    ident = all(f[t] ** 2 - f[t - 1] * f[t + 1] == q ** 4 / 16 * det ** (t - 2) for t in range(2, H3 + 1))
    fails = [t for t in range(1, H3 + 1) if f[t] ** 2 < f[t - 1] * f[t + 1]]
    below = (3 - q) ** 2 > 5                            # q < 3 - sqrt 5  <=>  (3-q)^2 > 5  (0 < q <= 1)
    resa[str(q)] = dict(det_Q=str(det), identity_holds_2_le_t_le_80=ident, log_concave_up_to_80=not fails,
                        q_below_3_minus_sqrt5=below, failures_are_exactly_odd_t_ge_3=(fails == list(range(3, H3 + 1, 2))))
    assert ident and det != 0                          # det(Q) = 0 only at the irrational q = 3 - sqrt 5
    assert (det > 0) == below == (not fails)
    assert det > 0 or fails == list(range(3, H3 + 1, 2))
out["remark_lc_N3"] = dict(horizon=H3, results=resa)

# ---------------- (b) N = 4, q = 4/5
H4 = 400
f = pmf_1d(4, Fraction(4, 5), H4 + 1)
fails4 = [t for t in range(1, H4 + 1) if f[t] ** 2 < f[t - 1] * f[t + 1]]
out["remark_lc_N4_q4-5"] = dict(horizon=H4, log_concavity_fails_at_t=fails4)
assert fails4 == [4], fails4

# ---------------- (c) continuum value of t_cf/t* - 1 in one dimension
c = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'constants.json')))["kappa_1 = 2 tau_star"]
iv.dps = 40
kap = iv.mpf([c["lo_dec"], c["hi_dec"]])
val = iv.log(iv.pi ** 2) / ((iv.pi ** 2 / 2 + 1) * kap) - 1
out["remark_cfX_continuum_value"] = dict(formula="ln(pi^2)/((pi^2/2+1) kappa_1) - 1", interval=str(val))

# ---------------- (d) Conjecture conj:last on the certified ranges
EXC = {(1, "4/5", 4): [5], (1, "1/2", 3): [3, 4], (1, "1/2", 4): [7, 8]}     # maximisers, Theorem modes (d)
n_all = n_nonexc = n_T_small = n_T_large = 0
exc_seen = {}
for d in (1, 2, 3):
    for q in ("4/5", "1/2"):
        tag = f"d{d}_q{q.replace('/', '-')}"
        px = {}
        p = os.path.join(CERT, f"packedexact_{tag}.jsonl")
        if os.path.exists(p):
            px = {json.loads(l)["N"]: json.loads(l) for l in open(p)}
        for line in open(os.path.join(CERT, f"c128_{tag}.jsonl")):
            r = json.loads(line)
            n_all += 1
            key = (d, q, r["N"])
            if key in EXC:
                mx = px[r["N"]]["maximisers_of_lower_bound"]
                assert mx == EXC[key] and px[r["N"]]["T_tail"] == r["T_tail"], (key, mx)
                exc_seen[str(key)] = dict(maximisers=mx, T=r["T_tail"])
                n_T_small += r["T_tail"] == min(mx)
                n_T_large += r["T_tail"] == max(mx)
            else:
                n_nonexc += 1
                assert r["mode_certified"] and r["unimodal_certified"]
                n_T_small += r["T_tail"] == r["mode"]
                n_T_large += r["T_tail"] == r["mode"]
out["conjecture_last"] = dict(certificates=n_all, non_exceptional=n_nonexc, T_equals_smallest_maximiser=n_T_small,
                              T_equals_largest_maximiser=n_T_large, exceptional=exc_seen,
                              source="certs/c128_d*_q4-5.jsonl, certs/c128_d*_q1-2.jsonl, certs/packedexact_d1_q*.jsonl")

# ---------------- (e) junction of Theorem rem and Proposition rem
rem = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'mfpt2d_remainder.json')))
a, b = rem["self_contained"], rem["via_single_sum"]
assert a["at_N_max"] == a["N_max"] and b["at_N"] == b["N_min"] == a["N_max"] + 1
out["remainder_junction"] = dict(N=a["N_max"], sup_r_N_upper=a["max_remainder_upper"], inf_r_N_plus_1_lower=b["min_remainder_lower"],
                                 strictly_increasing_across=Fraction(a["max_remainder_upper"]) < Fraction(b["min_remainder_lower"]))

# ---------------- (f) speed of the C engine relative to the programs of Section 3 (wall-clock times)
import statistics


def rows(name):
    return {json.loads(l)["N"]: json.loads(l) for l in open(os.path.join(CERT, name))}


sp = {}
for d, q, lo, hi in ((1, "4-5", 400, 600), (2, "4-5", 100, 160), (3, "4-5", 40, 60), (1, "1-2", 200, 300), (2, "1-2", 60, 80), (3, "1-2", 20, 32)):
    c, g, p = rows(f"c128_d{d}_q{q}.jsonl"), rows(f"modes_d{d}_q{q}.jsonl"), rows(f"packed_d{d}_q{q}.jsonl")
    Ns = [N for N in range(lo, hi + 1) if N in c and N in g and N in p and c[N]["seconds"] > 0]
    rg = [g[N]["seconds"] / c[N]["seconds"] for N in Ns]
    rp = [p[N]["seconds"] / c[N]["seconds"] for N in Ns]
    sp[f"d{d}_q{q}_N{lo}-{hi}"] = dict(cases=len(Ns), generator_over_c=[round(min(rg), 1), round(statistics.median(rg), 1), round(max(rg), 1)],
                                       checker_over_c=[round(min(rp), 1), round(statistics.median(rp), 1), round(max(rp), 1)])
allg = [v["generator_over_c"] for v in sp.values()]
allp = [v["checker_over_c"] for v in sp.values()]
out["speed_ratios_wall_clock"] = dict(note="[min, median, max] over the N of each range; wall-clock times on a heavily loaded machine",
                                      ranges=sp, generator_over_c_min=min(x[0] for x in allg), generator_over_c_max=max(x[2] for x in allg),
                                      generator_over_c_median_range=[min(x[1] for x in allg), max(x[1] for x in allg)],
                                      checker_over_c_min=min(x[0] for x in allp), checker_over_c_max=max(x[2] for x in allp),
                                      checker_over_c_median_range=[min(x[1] for x in allp), max(x[1] for x in allp)])

# ---------------- (g) shape of the closed-form error; numbers quoted after Corollary cor:qscaling
shape = {}
for pre in ("closedform", "closedform_c128"):
    for d in (2, 3):
        for q in ("4-5", "1-2"):
            r = rows(f"{pre}_d{d}_q{q}.jsonl")
            Ns = sorted(r)
            lo = {N: Fraction(r[N]["rel_err_lo"]) for N in Ns}
            hi = {N: Fraction(r[N]["rel_err_hi"]) for N in Ns}
            shape[f"{pre}_d{d}_q{q}"] = dict(
                N_range=[Ns[0], Ns[-1]], count=len(Ns),
                positive_N=[N for N in Ns if lo[N] > 0], negative_N_count=sum(1 for N in Ns if hi[N] < 0),
                first_negative_N=min([N for N in Ns if hi[N] < 0], default=None),
                negative_for_all_N_from=(min(N for N in Ns if all(hi[M] < 0 for M in Ns if M >= N)) if hi[Ns[-1]] < 0 else None),
                sign_undetermined_N=[N for N in Ns if lo[N] <= 0 <= hi[N]],
                not_decreasing_at_N_ge_10=[N for i, N in enumerate(Ns[:-1]) if N >= 10 and not hi[Ns[i + 1]] < lo[N]],
                value_at_N2=[float(lo[2]), float(hi[2])], value_at_Nmax=[float(lo[Ns[-1]]), float(hi[Ns[-1]])])
out["closed_form_shape"] = shape
m12 = rows("modes_d1_q1-2.jsonl")
m45, m1 = rows("modes_d1_q4-5.jsonl"), rows("modes_d1_q1-1.jsonl")
big = max((r["mode"], d, N) for d in (1, 2, 3) for q, top in (("4-5", None), ("1-2", None))
          for N, r in rows(f"modes_d{d}_q{q}.jsonl").items() if N <= {1: 300, 2: 80, 3: 32}[d])
gap = max((Fraction(4, 5) * m45[N]["mode"] - m1[N]["mode"], N) for N in m1 if N in m45 and N <= 200)
out["qscaling_text"] = dict(largest_mode_on_corollary_ranges=dict(mode=big[0], d=big[1], N=big[2]),
                            largest_4_5_tstar_4_5_minus_tstar_1_d1_N_le_200=dict(value=str(gap[0]), N=gap[1]))

json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'remark_checks.json'), "w"), indent=1)
print(json.dumps(out, indent=1))
