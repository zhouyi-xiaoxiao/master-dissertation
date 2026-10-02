"""V6: figures of the check (vector PDF + PNG), built only from verify/results/*.json."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(_R, 'data', 'msc_proof_verify', f)))
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK = '#333333'; MUTED = '#777777'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
                     'pdf.fonttype': 42, 'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.markersize': 4})

# ---------------- Fig V1: exact agreement and float64 behaviour
a = R('v1_exact.json')['corner'] + (R('v1b_exact_large.json') if os.path.exists(_os.path.join(_R, 'data', 'msc_proof_verify', 'v1b_exact_large.json')) else [])
fl = R('v2_identities.json')['float64']
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
keys = [('rel_D', 'double sum (D)'), ('rel_form_a', 'form (a)'), ('rel_S', 'form (S)'), ('rel_Hodd', 'odd half-sum'), ('rel_Heven', 'even half-sum')]
floor = 1e-63
for (k, lab), c, m in zip(keys, PAL, 'osD^v'):
    ax[0].semilogy([r['N'] for r in a], [max(r[k], floor) for r in a], m, color=c, label=lab, markersize=3.5, linestyle='none', alpha=0.9)
ax[0].axvline(22.5, color=MUTED, lw=0.8, ls=':')
ax[0].text(11, 1e-47, '50-digit arithmetic', color=MUTED, ha='center', fontsize=8)
ax[0].text(37, 1e-56, '60-digit arithmetic', color=MUTED, ha='center', fontsize=8)
ax[0].text(1, 1e-57, 'points on the bottom line:\ndeviation is exactly 0\nat working precision', color=MUTED, fontsize=7, va='center')
ax[0].set_ylim(1e-64, 1e-38); ax[0].set_xlabel('lattice side $N$'); ax[0].set_ylabel('relative deviation from exact rational solve')
ax[0].set_title('(a) Closed forms vs exact rational solve, $N=2\\ldots%d$' % max(r['N'] for r in a), fontsize=9, loc='left')
ax[0].grid(True, axis='y'); ax[0].legend(ncol=3, fontsize=7.5, loc='upper center')
Ns = [r['N'] for r in fl]
ax[1].loglog(Ns, [r['rel_S'] for r in fl], 'o-', color=PAL[0], label='sparse direct solve vs form (S)')
e35 = [(r['N'], abs(r['T_form_a_float64'] - r['T_formS']) / r['T_formS']) for r in fl if np.isfinite(r['T_form_a_float64'])]
ax[1].loglog([e[0] for e in e35], [max(e[1], 1e-17) for e in e35], 's-', color=PAL[1], label='form (a) vs form (S)')
ax[1].axvline(403, color=MUTED, lw=0.8)
ax[1].text(380, 2e-16, 'form (a) not finite\nin float64 for $N\\geq403$', color=INK, ha='right', fontsize=8)
ax[1].set_xlabel('lattice side $N$'); ax[1].set_ylabel('relative difference (float64)')
ax[1].set_title('(b) Double precision, $q=0.8$', fontsize=9, loc='left'); ax[1].grid(True); ax[1].legend(fontsize=7.5, loc='upper left')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(_R, 'out', 'figures', 'msc_proof_verify', f'figV1_exact_agreement.{ext}'), dpi=220)
plt.close(fig)

# ---------------- Fig V2: asymptotics
A = R('v3_asymptotics.json')
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
orders = ['2', '0', '-2', '-4', '-6', '-8']
x = np.arange(len(orders)); w = 0.26
for i, (fam, lab) in enumerate([('2^j (top 12)', '$N=2^j$, 12 sizes up to $2^{17}$'), ('2^j (top 9)', '$N=2^j$, 9 sizes'), ('3^j (all 9, odd N)', '$N=3^j$ (odd $N$), 9 sizes up to $3^{10}$')]):
    v = [abs(float(A['blind_even_power_fit'][fam][o]['minus_closed'])) for o in orders]
    ax[0].bar(x + (i - 1) * w, v, w * 0.9, color=PAL[i], label=lab)
ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(['$C_{2}$', '$C_{0}$', '$C_{-2}$', '$C_{-4}$', '$C_{-6}$', '$C_{-8}$'])
ax[0].set_ylabel('|blind fit $-$ closed form|'); ax[0].grid(True, axis='y'); ax[0].legend(fontsize=7.5, loc='upper left')
ax[0].set_title('(a) Coefficients recovered without using the closed forms', fontsize=9, loc='left')
T = A['truncation_table']
Ns = [r['N'] for r in T]
for t in range(1, 8):
    ax[1].loglog(Ns, [r[f'rel_err_{t}_terms'] for r in T], 'o-', color=PAL[(t - 1) % 8], label=f'{t} term' + ('s' if t > 1 else ''))
ax[1].set_xlabel('lattice side $N$'); ax[1].set_ylabel('relative error vs exact $qT_N$')
ax[1].set_title('(b) Truncated expansion (3.7)', fontsize=9, loc='left')
ax[1].grid(True); ax[1].legend(fontsize=7, ncol=2, loc='lower left')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(_R, 'out', 'figures', 'msc_proof_verify', f'figV2_asymptotics.{ext}'), dpi=220)
print('figures written')
