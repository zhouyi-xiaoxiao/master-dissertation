"""ensemble.py -- defect placement generators and the per-configuration analysis record."""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import fptcore as fc


# ------------------------------------------------------------------ placements
def generate_defects_smart(N, p, seed):
    """Corner-thinned placement law (Section 8 of the article): weighted random placement with reduced
    weight near start and target.
    Returns defect_mask (True = blocked)."""
    np.random.seed(seed)

    if p == 0:
        return np.zeros((N, N), dtype=bool)

    M = int(round(p * N * N))
    defect_mask = np.zeros((N, N), dtype=bool)

    # Create weight matrix - lower weight near start and target
    weights = np.ones((N, N))

    # Reduce weight near start (0,0)
    radius = max(3, N//8)
    for i in range(radius):
        for j in range(radius):
            dist = np.sqrt(i**2 + j**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)

    # Reduce weight near target (N-1, N-1)
    for i in range(N-radius, N):
        for j in range(N-radius, N):
            dist = np.sqrt((N-1-i)**2 + (N-1-j)**2)
            weights[i, j] *= np.exp(-2 * (radius - dist) / radius)

    # Never place at start or target
    weights[0, 0] = 0
    weights[N-1, N-1] = 0

    # Flatten and normalize
    weights_flat = weights.flatten()
    weights_flat = weights_flat / weights_flat.sum()

    # Sample defect positions
    all_positions = np.arange(N * N)
    if M > 0 and M < N * N - 2:
        selected = np.random.choice(all_positions, M, replace=False, p=weights_flat)
        for idx in selected:
            i, j = idx // N, idx % N
            if not (i == 0 and j == 0) and not (i == N-1 and j == N-1):
                defect_mask[i, j] = True

    return defect_mask


def place_uniform(N, M, rng, d=2):
    """Exactly M blocked sites, uniform over all sites except the two corners s and t."""
    tot = N ** d
    cand = np.arange(1, tot - 1)  # flat index 0 = start corner, tot-1 = target corner
    blk = rng.choice(cand, M, replace=False)
    mask = np.zeros(tot, bool); mask[blk] = True
    return mask.reshape((N,) * d)


# ------------------------------------------------------------------ analysis record
def local_blocked(defect_mask, site, radius):
    """Number of blocked sites within Euclidean distance `radius` of `site` and the number of
    lattice sites in that neighbourhood (excluding `site` itself)."""
    N = defect_mask.shape[0]
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    near = ((ii - site[0]) ** 2 + (jj - site[1]) ** 2) <= radius ** 2
    near[site] = False
    return int(defect_mask[near].sum()), int(near.sum())


def analyze(defect_mask, q, start=None, target=None, tetali=True, radii=(3, 6, 10)):
    N = defect_mask.shape[0]
    d = defect_mask.ndim
    start = start or (0,) * d
    target = target or (N - 1,) * d
    om = ~defect_mask
    summ, ch = fc.fpt_summary(om, q, start, target)
    rec = {k: summ[k] for k in ('n_cluster', 'mfpt', 'sd', 'mode', 'f_mode', 'median', 'q10', 'q90',
                                'S_mode', 'S_mfpt', 'lam1', 'tau1', 'A1', 'ratio', 'cv', 't_cert',
                                'lam2', 'w1')}
    mu2, tau_rel = fc.reflecting_relaxation(ch)
    rec['mu2'] = mu2; rec['tau_rel'] = tau_rel
    rec['n_blocked'] = int(defect_mask.sum())
    rec['n_open'] = int(om.sum())
    rec['pocket_free'] = int(rec['n_open'] == rec['n_cluster'])
    rec['chem_dist'] = fc.chemical_distance(ch['cluster'], start, target)
    if tetali:
        tp = fc.tetali_parts(ch, q, d=d)
        rec.update(G_tt=tp['G_tt'], G_ss=tp['G_ss'], G_st=tp['G_st'], R_st=tp['R_st'],
                   mfpt_tetali=tp['mfpt_tetali'])
    if d == 2:
        for R in radii:
            bt, nt = local_blocked(defect_mask, target, R)
            bs, ns = local_blocked(defect_mask, start, R)
            rec[f'blk_t_r{R}'] = bt; rec[f'blk_s_r{R}'] = bs; rec[f'nsite_r{R}'] = nt
        # degree (number of open neighbours) of start and target
        def deg(site):
            c = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = site[0] + dx, site[1] + dy
                if 0 <= x < N and 0 <= y < N and om[x, y]:
                    c += 1
            return c
        rec['deg_s'] = deg(start); rec['deg_t'] = deg(target)
    return rec


def draw_connected(N, p, scheme, seedseq, max_attempts=100000):
    """Draw placements until start and target are connected. Returns (mask, attempts)."""
    rng = np.random.default_rng(seedseq)
    M = int(round(p * N * N))
    s, t = (0, 0), (N - 1, N - 1)
    for attempt in range(1, max_attempts + 1):
        if scheme == 'uniform':
            mask = place_uniform(N, M, rng)
        elif scheme == 'smart':
            mask = generate_defects_smart(N, p, int(rng.integers(0, 2**31 - 1)))
        else:
            raise ValueError(scheme)
        if M == 0 or fc.connected(~mask, s, t):
            return mask, attempt
    raise RuntimeError('no connected configuration found')


SCHEME_ID = {'uniform': 1, 'smart': 2}


def worker(args):
    N, q, p, scheme, rep = args
    ss = np.random.SeedSequence([20261001, N, SCHEME_ID[scheme], int(round(p * 10000)), rep])
    mask, attempts = draw_connected(N, p, scheme, ss)
    rec = analyze(mask, q)
    rec.update(N=N, q=q, p=p, scheme=scheme, rep=rep, attempts=attempts)
    return rec


def worker_large(args):
    """Time-stepping route (any N): exact MFPT/SD, certified mode, median, sparse Tetali parts."""
    N, q, p, scheme, rep = args
    ss = np.random.SeedSequence([20261001, N, SCHEME_ID[scheme], int(round(p * 10000)), rep])
    mask, attempts = draw_connected(N, p, scheme, ss)
    summ, ch = fc.timestep_summary(~mask, q, (0, 0), (N - 1, N - 1))
    tp = fc.tetali_parts_sparse(ch, q)
    rec = dict(summ)
    rec.update(G_tt=tp['G_tt'], G_ss=tp['G_ss'], G_st=tp['G_st'], mfpt_tetali=tp['mfpt_tetali'],
               N=N, q=q, p=p, scheme=scheme, rep=rep, attempts=attempts,
               n_blocked=int(mask.sum()))
    return rec
