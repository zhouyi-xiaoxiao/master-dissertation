"""explore19: are the 'twisted groups' S(delta,beta) = {y + 2cos((l+beta)delta) : (l+beta)delta < pi} log-concave
(h_j(S) positive and h_j^2 >= h_{j-1}h_{j+1})?  float/mp exploration only."""
import sys, math
import mpmath as mp
mp.mp.dps = 60

def group(y, rho, beta):
    delta = mp.pi / rho
    lam = []
    l = 0
    while (l + beta) * delta < mp.pi - mp.mpf(10) ** (-40):
        lam.append(y + 2 * mp.cos((l + beta) * delta)); l += 1
    return lam

def hseq(lam, jmax):
    """h_j via recurrence of 1/prod(1-lam z) (convolution one factor at a time, mp)"""
    h = [mp.mpf(1)] + [mp.mpf(0)] * jmax
    for a in lam:
        for j in range(1, jmax + 1):
            h[j] = h[j] + a * h[j - 1]
    return h

def scan(y, rho, beta, jmax=None):
    lam = group(y, rho, beta)
    n = len(lam)
    if jmax is None:
        jmax = int(1.0 * n * n) + 40
    mp.mp.dps = 40 + int(0.6 * (jmax + n))
    lam = group(y, rho, beta)
    h = hseq(lam, jmax + 1)
    worst = None; argw = None; neg = None
    for j in range(1, jmax + 1):
        if h[j] <= 0 and neg is None:
            neg = j
        c = h[j] * h[j] - h[j - 1] * h[j + 1]
        rel = c / (h[j] * h[j])
        if worst is None or rel < worst:
            worst, argw = rel, j
    e1 = sum(lam); p2 = sum(a * a for a in lam); e2 = (e1 * e1 - p2) / 2
    return n, float(e2), float(worst), argw, neg

y = mp.mpf(1) / 2
if len(sys.argv) > 1:
    y = mp.mpf(sys.argv[1])
for rho in (9, 10, 11, 12, 12.5, 13, 14, 16, 20, 20.4, 24, 30):
    out = []
    for beta in (0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
        n, e2, worst, argw, neg = scan(y, mp.mpf(rho), mp.mpf(beta))
        out.append("b=%.2f n=%d e2=%.2f min=%.2e@%d%s" % (beta, n, e2, worst, argw, " NEG@%d" % neg if neg else ""))
    print("rho=%s: " % rho + " | ".join(out), flush=True)
