#!/usr/bin/env python
"""Section 8.3: separate computation of the second-order coefficient of the low-density expansion of the
conductivity of the site-diluted square lattice,

        Sigma/Sigma_0 = 1 - l1 p + a2 p^2 + O(p^3),        D_eff/D_0 = (Sigma/Sigma_0)/P_inf,

by the cluster (inclusion-exclusion) expansion on an Nt x Nt torus of unit resistors under a uniform mean field E
along x_1:   loss(B) := Nt^2 (1 - Sigma(B)/Sigma_0)  for a set B of removed sites,

        l1 = loss({0}),        l2(r) = loss({0, r}),        a2 = -(1/2) sum_{r != 0} [ l2(r) - 2 l1 ].

No random numbers and no fit.  The published value of a2 (Ernst, Nieuwenhuizen and van Velthoven, J. Phys. A 20,
5335 (1987)) is quoted in the article as 1.2858; this script does not use it.

Method (exact for each torus).  Removing a set R of bonds lowers the dissipated power by  e_R^T f_R  with
f_R = (I - Gamma)^+ e_R,  Gamma = D_R G D_R^T,  where D is the bond-node incidence matrix, G the pseudo-inverse of the
torus Laplacian and e the bond field of the uniform solution (E on x_1-bonds, 0 on x_2-bonds); this is the identity
of Proposition s7_defects_density:prop-dilute(b) of the article written for a periodic network.  (I - Gamma) is
singular along the directions that shift the potential of an isolated site; e_R is orthogonal to them, so the
pseudo-inverse gives the loss.  G is obtained from one FFT.
Checks inside the script: l1 against the stored single-site values of data/msc_defects/homogenization.json
(computed there by a sparse solve of the diluted torus), and l2 for six separations against a direct dense solve of
the diluted torus.

Output: data/article/s7_defects_density_second_order.json
Run   : python s7_defects_density_second_order.py            (about 1 minute)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
OUT = _os.path.join(_R, 'data', 'article', 's7_defects_density_second_order.json')
HOM = _os.path.join(_R, 'data', 'msc_defects', 'homogenization.json')


def green(Nt):
    kx = 2 * np.pi * np.arange(Nt) / Nt
    lam = (2 - 2 * np.cos(kx))[:, None] + (2 - 2 * np.cos(kx))[None, :]
    lam[0, 0] = 1.0
    inv = 1.0 / lam
    inv[0, 0] = 0.0
    return np.real(np.fft.ifft2(inv))          # G(x) for the Laplacian sum_y [h(x) - h(y)]; sum_x G(x) = 0


def bonds_of(site, Nt):
    """the four bonds of a site as (tail, head, e) with e = 1 on x_1-bonds, 0 on x_2-bonds; orientation tail -> head
    is the positive coordinate direction"""
    x, y = site
    return [(((x - 1) % Nt, y), (x, y), 1.0), ((x, y), ((x + 1) % Nt, y), 1.0),
            ((x, (y - 1) % Nt), (x, y), 0.0), ((x, y), (x, (y + 1) % Nt), 0.0)]


def loss(sites, G, Nt):
    bl = []
    for s in sites:
        for b in bonds_of(s, Nt):
            if (b[0], b[1]) not in [(c[0], c[1]) for c in bl]:
                bl.append(b)
    m = len(bl)
    g = lambda a, b: G[(a[0] - b[0]) % Nt, (a[1] - b[1]) % Nt]
    Gam = np.empty((m, m))
    for i, (ti, hi, _) in enumerate(bl):
        for j, (tj, hj, _) in enumerate(bl):
            # (D G D^T)_{ij} with (D phi)_b = phi(head) - phi(tail)
            Gam[i, j] = g(hi, hj) - g(hi, tj) - g(ti, hj) + g(ti, tj)
    e = np.array([b[2] for b in bl])
    f = np.linalg.lstsq(np.eye(m) - Gam, e, rcond=1e-10)[0]
    return float(e @ f)


def loss_direct(sites, Nt):
    """dense check: minimise sum over kept bonds of (e_b + (D phi)_b)^2 over periodic phi"""
    idx = lambda x, y: (x % Nt) * Nt + (y % Nt)
    removed = set(idx(*s) for s in sites)
    rows = []
    ev = []
    for x in range(Nt):
        for y in range(Nt):
            for (dx, dy, e) in ((1, 0, 1.0), (0, 1, 0.0)):
                a, b = idx(x, y), idx(x + dx, y + dy)
                if a in removed or b in removed:
                    continue
                r = np.zeros(Nt * Nt)
                r[b] += 1.0
                r[a] -= 1.0
                rows.append(r)
                ev.append(e)
    D = np.array(rows)
    ev = np.array(ev)
    phi = np.linalg.lstsq(D, -ev, rcond=None)[0]
    W = float(np.sum((ev + D @ phi) ** 2))
    return Nt * Nt - W


t0 = time.time()
out = {"generated_by": "code/article/s7_defects_density_second_order.py", "rows": []}
# check against a direct dense solve
Nt = 12
G = green(Nt)
chk = []
for r in [(1, 0), (0, 1), (1, 1), (2, 0), (3, 5), (6, 6)]:
    a, b = loss([(0, 0), r], G, Nt), loss_direct([(0, 0), r], Nt)
    chk.append({"Nt": Nt, "r": list(r), "loss_green": a, "loss_direct": b, "abs_diff": abs(a - b)})
out["check_two_sites_against_dense_solve"] = chk
out["check_max_abs_diff"] = max(c["abs_diff"] for c in chk)

for Nt in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
    G = green(Nt)
    l1 = loss([(0, 0)], G, Nt)
    S = 0.0
    nearest = {}
    # symmetry: r -> (-x, y) and (x, -y) leave l2 unchanged
    for x in range(0, Nt // 2 + 1):
        for y in range(0, Nt // 2 + 1):
            if x == 0 and y == 0:
                continue
            mult = (1 if x in (0, Nt // 2) and Nt % 2 == 0 or x == 0 else 2) * \
                   (1 if y in (0, Nt // 2) and Nt % 2 == 0 or y == 0 else 2)
            l2 = loss([(0, 0), (x, y)], G, Nt)
            S += mult * (l2 - 2 * l1)
            if (x, y) in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2)):
                nearest["%d,%d" % (x, y)] = l2 - 2 * l1
    a2 = -0.5 * S
    row = {"Nt": Nt, "l1": l1, "a2": a2, "D2_coefficient": a2 - l1 + 1.0, "pair_terms_l2_minus_2l1": nearest}
    out["rows"].append(row)
    print(Nt, "l1 = %.6f" % l1, "a2 = %.6f" % a2, "D2 = %.6f" % (a2 - l1 + 1.0), "%.0fs" % (time.time() - t0), flush=True)

# check of l1 against the single-site values stored by the main implementation (sparse solves of the diluted torus)
stored = {int(x["L"]): x["one_minus_sigma_times_L2"] for x in json.load(open(HOM))["single_site"]}
out["check_l1_against_stored_single_site"] = {
    "source": "data/msc_defects/homogenization.json (single_site)",
    "sizes": sorted(stored),
    "max_abs_diff": max(abs(q["l1"] - stored[q["Nt"]]) for q in out["rows"] if q["Nt"] in stored)}
# a2 (Nt) - a2 (infinity) scales as 1/Nt^2: the product below is constant over the four largest tori
out["Nt2_times_(a2_minus_extrapolated)_last4"] = None

# extrapolation in 1/Nt^2 from the three largest tori (the single-site loss converges like 1/Nt^2)
r = out["rows"]
x = np.array([1.0 / q["Nt"] ** 2 for q in r[-3:]])
y = np.array([q["a2"] for q in r[-3:]])
c = np.polyfit(x, y, 1)
out["a2_extrapolated_linear_in_Nt^-2_last3"] = float(c[1])
out["a2_largest_torus"] = r[-1]["a2"]
out["D2_coefficient_from_extrapolated_a2"] = float(c[1]) - np.pi + 1.0
out["Nt2_times_(a2_minus_extrapolated)_last4"] = [q["Nt"] ** 2 * (q["a2"] - float(c[1])) for q in r[-4:]]
out["published_ENvV1987_as_quoted_in_article"] = {"a2": 1.2858, "D2": -0.8558}
out["extrapolated_minus_published"] = {"a2": float(c[1]) - 1.2858, "D2": float(c[1]) - np.pi + 1.0 + 0.8558}
out["wall_s"] = round(time.time() - t0, 1)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
