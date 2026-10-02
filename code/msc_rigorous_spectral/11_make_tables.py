"""Generate the LaTeX tables of Section 13 from the certified data (09_certified_part3.json, 07_certified_poles.json)."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, json
HERE = os.path.dirname(__file__)
D = json.load(open(_os.path.join(_R, 'data', 'msc_rigorous_spectral', '09_certified_part3.json')))
def f(v, nd=5):
    return "--" if v is None else f"{v:.{nd}f}"
def table(rows, Ns, cap, label):
    L = []
    L.append(r"\begin{table}[ht]\centering\footnotesize")
    L.append(r"\begin{tabular}{rrccccc}\toprule")
    L.append(r"$N$ & $\Xb$ & $[X_-,X_+]$ & $\sigma_0\tau$ & bounds (i) & $\sigma_0m$ & bounds (ii)\\\midrule")
    for r in rows:
        if r["N"] not in Ns or "explicitN_bounds" not in r: continue
        b = r["explicitN_bounds"]
        L.append(f"{r['N']} & {r['X']:.4f} & [{r['X_lo']:.3f}, {r['X_hi']:.3f}] & {r['sigma0_tau']:.6f} & [{b['t_lo']:.6f}, {b['t_hi']:.6f}] & {r['sigma0_m']:.5f} & "
                 f"[{b['s0m_lo']:.5f}, {b['s0m_hi']:.5f}]" + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\\[6pt]\begin{tabular}{rcccccc}\toprule")
    L.append(r"$N$ & $\delta=\sigma_1/\Lambda_1-1$ & bounds (iii) $[\delta_-,\delta_+]$ & $r_0$ & bounds (v) & $r_1$ & bounds (vi)\\\midrule")
    for r in rows:
        if r["N"] not in Ns or "explicitN_bounds" not in r: continue
        b = r["explicitN_bounds"]
        L.append(f"{r['N']} & {r['delta']:.5f} & "
                 + (f"[{f(b.get('d_lo'))}, {f(b.get('d_hi'))}]" if b.get('d_hi') is not None else "--") + " & "
                 + f"{r['r0']:.5f} & [{f(b.get('r0_lo_N'))}, {f(b.get('r0_hi_N'))}] & {r['r1']:.5f} & "
                 + (f"[{f(b.get('r1_lo'))}, {f(b.get('r1_hi'))}]" if b.get('r1_lo') is not None else "--") + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\caption{" + cap + r"}\label{" + label + r"}\end{table}")
    return "\n".join(L)
t2 = table(D["rows2"], (5, 10, 20, 50, 100, 200, 300),
           r"$d=2$, corner to corner: certified values (ball arithmetic, all digits shown are correct up to rounding) and the bounds of Corollary~\ref{cor:explicitN}, which depend on $N$ only ($X_\pm$, $M_\pm$ from Corollary~\ref{cor:Xbar}; $Z$ is the certified upper end $6.0268120396919401\ldots$ of $Z_2$ from Lemma~\ref{lem:epstein}, not the rounded value $6.02682$, with which the bounds change in the last digit shown at most, e.g.\ the lower bound for $r_1$ at $N=5$ becomes $-19.90509$). A dash means that the hypothesis of the corresponding bound is not satisfied.", "tab:cc2")
t3 = table(D["rows3"], (4, 6, 10, 20, 30, 50),
           r"$d=3$, corner to corner: certified values and the bounds of Corollary~\ref{cor:explicitN} (functions of $N$ only), with $X_\pm$, $M_\pm$ from Corollary~\ref{cor:Xbar}, i.e.\ from Theorem~\ref{thm:d3}(c), and $Z$ the certified upper end of $Z_3$ from Lemma~\ref{lem:epstein}. A dash means that the hypothesis of the corresponding bound is not satisfied.", "tab:cc3")
open(_os.path.join(_R, 'out', 'tex', 'msc_rigorous_spectral', 'tab_cc2.tex'), "w").write(t2 + "\n")
open(_os.path.join(_R, 'out', 'tex', 'msc_rigorous_spectral', 'tab_cc3.tex'), "w").write(t3 + "\n")
# exact-X table for d = 3 (bounds of Theorem 7.10 with the exact Xbar)
def tableX(rows, Ns, cap, label):
    L = [r"\begin{table}[ht]\centering\footnotesize", r"\begin{tabular}{rrccccc}\toprule",
         r"$N$ & $\Xb$ & $1-\sigma_0\tau$ & bounds (i) on $1-\sigma_0\tau$ & $\delta$ & $[\delta_-,\delta_+]$ & $r_1$ and bounds (vi)\\\midrule"]
    for r in rows:
        if r["N"] not in Ns or "exactX_bounds" not in r: continue
        b = r["exactX_bounds"]
        L.append(f"{r['N']} & {r['X']:.3f} & {1-r['sigma0_tau']:.3e} & [{1-b['t_hi']:.3e}, {1-b['t_lo']:.3e}] & {r['delta']:.5f} & "
                 + (f"[{f(b.get('d_lo'))}, {f(b.get('d_hi'))}]" if b.get('d_hi') is not None else "--") + " & "
                 + f"{r['r1']:.5f} " + (f"[{f(b.get('r1_lo'))}, {f(b.get('r1_hi'))}]" if b.get('r1_lo') is not None else "") + r"\\")
    L += [r"\bottomrule\end{tabular}", r"\caption{" + cap + r"}\label{" + label + r"}\end{table}"]
    return "\n".join(L)
tx = tableX(D["rows3"], (4, 6, 10, 20, 30, 50), r"$d=3$: bounds of Theorem~\ref{thm:cc} evaluated with the exact (certified) $\Xb$ and with $Z$ the certified upper end of $Z_3$ (Lemma~\ref{lem:epstein}).", "tab:cc3x")
open(_os.path.join(_R, 'out', 'tex', 'msc_rigorous_spectral', 'tab_cc3x.tex'), "w").write(tx + "\n")
print(t2); print(t3); print(tx)
