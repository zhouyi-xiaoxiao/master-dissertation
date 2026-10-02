"""explore14: is  j -> h_j(lambda_1..lambda_m)  (complete homogeneous symmetric polynomial) log-concave
whenever e_2(lambda) >= 0 (and e_1 > 0)?  Random real lambda with some negatives.  mp arithmetic."""
import random, mpmath as mp
mp.mp.dps = 200

def hseq(lam, J):
    poly = [mp.mpf(1)]
    for l in lam:
        new = poly + [mp.mpf(0)]
        for i in range(len(poly)):
            new[i + 1] -= l * poly[i]
        poly = new
    s = [mp.mpf(1)]
    for j in range(1, J + 1):
        s.append(-mp.fsum(poly[i] * s[j - i] for i in range(1, min(j, len(poly) - 1) + 1)))
    return s

rng = random.Random(1)
stats = {}
bad_examples = []
for trial in range(4000):
    m = rng.randint(3, 9)
    npos = rng.randint(1, m - 1)
    lam = [mp.mpf(rng.uniform(0.0, 1.0)) for _ in range(npos)] + [-mp.mpf(rng.uniform(0.0, 1.0)) * rng.choice([0.05, 0.2, 0.6]) for _ in range(m - npos)]
    e1 = mp.fsum(lam)
    e2 = (e1 ** 2 - mp.fsum(l * l for l in lam)) / 2
    if e1 <= 0:
        continue
    s = hseq(lam, 80)
    lc = all(s[j] > 0 for j in range(81)) and all(s[j] ** 2 >= s[j - 1] * s[j + 1] * (1 - mp.mpf(10) ** (-60)) for j in range(1, 80))
    key = (bool(e2 >= 0), bool(lc))
    stats[key] = stats.get(key, 0) + 1
    if (e2 >= 0) != lc and len(bad_examples) < 5:
        firstbad = next((j for j in range(1, 80) if s[j] ** 2 < s[j - 1] * s[j + 1] or s[j] <= 0), None)
        bad_examples.append(([float(l) for l in lam], float(e2), lc, firstbad))
print(stats)
for b in bad_examples:
    print(b)
