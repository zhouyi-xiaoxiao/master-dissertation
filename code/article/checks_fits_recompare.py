#!/usr/bin/env python
"""Supplementary Sections S6.3 and S10.2 (scaling fits): comparison of the fits of the main implementation with those of the second
implementation that checked them.

Reads the two stored fit files and, for every fit window and every model fitted by both, compares the parameters,
the AICc differences to the best model of the window (dAICc) and the ranking of the models by AICc.

Input : data/msc_modes/fit_results.json            (field fits; main implementation)
        data/msc_modes_verify/analysis.json        (field fits; second implementation)
Output: data/checks/modes_scaling_fits_recheck.json
Run   : python checks_fits_recompare.py          (under a second; no computation beyond reading the two files)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _R
FIRST = _os.path.join(_R, 'data', 'msc_modes', 'fit_results.json')
CHECK = _os.path.join(_R, 'data', 'msc_modes_verify', 'analysis.json')
OUT = _os.path.join(_R, 'data', 'checks', 'modes_scaling_fits_recheck.json')

WINDOWS = [   # (window in the second implementation, window in the main implementation)
    ("discrete_d1_short_range", "1d_mode_discrete_short_range"),
    ("discrete_d2_short_range", "2d_mode_discrete_short_range"),
    ("discrete_d3_short_range", "3d_mode_discrete_short_range"),
    ("cont_d2_mode_fit_10-200", "2d_mode_fit_10_200_extrapolated"),
    ("cont_d2_mfpt_fit_10-200", "2d_mfpt_fit_10_200_extrapolated"),
    ("cont_d3_mode_fit_10-100", "3d_mode_fit_10_100_extrapolated"),
    ("cont_d3_mfpt_fit_10-100", "3d_mfpt_fit_10_100_extrapolated"),
]


def norm(form):
    """model label without spaces and without a trailing '(pure ...)' remark"""
    form = form.split(" (pure")[0]
    return form.replace(" ", "")


first = json.load(open(FIRST))["fits"]
check = json.load(open(CHECK))["fits"]
rows, max_d, max_rel_p, same_rank = [], 0.0, 0.0, True
for wc, wf in WINDOWS:
    fc = check[wc]["fits"]
    ff = {norm(v["form"]): v for v in first[wf].values() if isinstance(v, dict) and "dAICc" in v}
    common = [m for m in fc if norm(m) in ff]
    rank_c = sorted(common, key=lambda m: fc[m]["aicc"])
    rank_f = sorted(common, key=lambda m: ff[norm(m)]["AICc"])
    same_rank &= rank_c == rank_f
    for m in common:
        a, b = fc[m], ff[norm(m)]
        pf = list(b["params"].values())
        rel = max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(a["params"], pf))
        d = a["dAICc"] - b["dAICc"]
        max_d = max(max_d, abs(d)); max_rel_p = max(max_rel_p, rel)
        rows.append({"window_second_implementation": wc, "window_main_implementation": wf, "model": m,
                     "dAICc_second_implementation": a["dAICc"], "dAICc_main_implementation": b["dAICc"],
                     "dAICc_difference": d, "params_max_rel_diff": rel})
out = {"provenance": "computed by code/article/checks_fits_recompare.py from data/msc_modes/fit_results.json and "
                     "data/msc_modes_verify/analysis.json (fits of the main implementation and of its check)",
       "n_windows": len(WINDOWS), "n_models_compared": len(rows),
       "max_abs_dAICc_difference": max_d, "params_max_rel_diff": max_rel_p,
       "ranking_by_AICc_identical_in_every_window": same_rank, "rows": rows}
json.dump(out, open(OUT, "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
