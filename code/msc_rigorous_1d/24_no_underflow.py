"""No-underflow certificate for the float64 pass of script 08 (Proposition 3.18, Lemma "no underflow").

Lemma.  For 1 <= j <= n = N-1 and t >= j-1,
     (Q^t)_{1,j} >= (q/2)^(j-1) (Q^(t-j+1))_{1,1} >= (q/2)^(j-1) (1 - q/2) lam_1^t  * 4 cos^2(th_1/2)/(2N-1),
(Chapman-Kolmogorov with nonnegative terms; (Q^s)_{11} = sum_m lam_m^s phi_m(1)^2 >= lam_1^s phi_1(1)^2 for even s,
 (Q^s)_{11} >= Q_11 (Q^(s-1))_{11} for odd s; phi_1(1)^2 = 4 cos^2(th_1/2)/(2N-1)), and (Q^t)_{1,j} = 0 for t < j-1.

Hence every nonzero entry met by script 08 (q = 4/5, t <= steps = t_end + 2) is at least
     vmin(N) := (2/5)^(N-2) (3/5) lam_1^(t_end+2) * 4 cos^2(th_1/2)/(2N-1).
We certify vmin(N) > 1e-242 for 2 <= N <= 600 in ball arithmetic; then all intermediate products (constants 0.2, 0.4)
exceed 1e-243 > 2^-1022, no operation underflows, and the standard relative error model (1+u)^(4t) of script 08 applies.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, json
from flint import arb, ctx

ctx.prec = 200
PI = arb.pi()
here = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '08_modes_q45.jsonl'))]
q = arb(4) / 5
worst = None
ok = True
for r in rows:
    N = r["N"]; steps = r["t_end"] + 2
    th1 = PI / (2 * N - 1)
    lam1 = 1 - q * (1 - th1.cos())
    vmin = (q / 2) ** max(N - 2, 0) * (1 - q / 2) * lam1 ** steps * 4 * (th1 / 2).cos() ** 2 / (2 * N - 1)
    good = bool(vmin > arb("1e-242"))
    ok &= good
    lg = float(vmin.log().lower()) / 2.302585092994046
    worst = lg if worst is None or lg < worst else worst
print(f"N = {rows[0]['N']}..{max(r['N'] for r in rows)} ({len(rows)} values): all vmin > 1e-242: {ok};  min log10(vmin) = {worst:.2f}")
json.dump(dict(n_values=len(rows), all_above_1e_242=ok, min_log10_vmin=worst),
          open(_os.path.join(_R, 'data', 'msc_rigorous_1d', '24_no_underflow.json'), "w"), indent=1)
