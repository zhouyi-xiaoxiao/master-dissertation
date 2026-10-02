"""summarize_lc1d.py -- reads data/cert_lc1d_*.jsonl, data/cert_family_F*.jsonl, data/cert_blocks.jsonl and checks that
the ranges claimed in Section 9 of note R4 are complete and certified.  -> data/summary_lc1d.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')

def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, pattern))):
        for line in open(f):
            rows.append(json.loads(line))
    return rows

out = {}
def rng(rows, geom, ymode, lo, hi, name):
    sel = {}
    for r in rows:
        if r["geom"] == geom and r["ymode"] == ymode:
            sel[r["m"]] = r           # last record wins
    missing = [m for m in range(lo, hi + 1) if m not in sel]
    bad = [m for m in range(lo, hi + 1) if m in sel and not sel[m]["status"].startswith("LOGCONCAVE")]
    ok = not missing and not bad
    rs = [sel[m] for m in range(lo, hi + 1) if m in sel]
    s = dict(range_m=[lo, hi], complete=not missing, all_certified=not bad, missing=missing[:10], not_certified=bad[:10],
             n_records=len(rs),
             max_n0_over_N2=max(r["n0_over_N2"] for r in rs), min_n0_over_N2=min(r["n0_over_N2"] for r in rs),
             total_window_checks=sum(r["n_checked"] for r in rs),
             min_rel_margin=min(r["min_rel_margin"] for r in rs if r["min_rel_margin"] is not None),
             all_exact_crosschecks_ok=all(r["exact_crosscheck_ok"] for r in rs),
             exact_equalities=sorted(set(r["exact_equalities"] for r in rs)),
             cpu_seconds=round(sum(r["seconds"] for r in rs), 1))
    out[name] = s
    print(name, json.dumps(s))
    return ok

rows = load("cert_lc1d_*.jsonl")
ok = True
ok &= rng(rows, "E2E", "1/2", 9, 1000, "E2E y=1/2 (q=4/5), N=10..1001")
ok &= rng(rows, "E2E", "thr", 3, 500, "E2E threshold q=q_1(N), N=4..501")
ok &= rng(rows, "FOLD", "1/2", 13, 1000, "FOLD y=1/2 (q=4/5), odd N=27..2001")
ok &= rng(rows, "FOLD", "thr", 3, 500, "FOLD threshold, odd N=7..1001")
samples = [r for r in rows if (r["geom"], r["ymode"]) in (("E2E", "1/2"), ("FOLD", "1/2")) and r["m"] > 1000]
out["samples beyond the direct range"] = [dict(geom=r["geom"], N=r["N"], status=r["status"], n0=r["n0"]) for r in samples]
print("samples:", out["samples beyond the direct range"])
fam = {}
for F in ("F1", "F2"):
    rs = load("cert_family_%s.jsonl" % F)
    idx = sorted(set(r["box_index"] for r in rs))
    last = {r["box_index"]: r for r in rs}
    need = 20 if F == "F1" else 6
    good = idx == list(range(need)) and all(last[i]["status"] == "CERTIFIED" for i in idx)
    ok &= good
    fam[F] = dict(boxes=len(idx), all_certified=good, j0=sorted(set(last[i]["j0"] for i in idx)),
                  min_lowerbound_C_over_H2=min(last[i]["min_lowerbound_C_over_H2"] for i in idx),
                  min_lowerbound_H_rel=min(last[i]["min_lowerbound_H_rel"] for i in idx),
                  max_kappa=max(last[i]["max_kappa"] for i in idx),
                  max_tail_ratio=max(last[i]["tail_rest_over_L_max"] for i in idx),
                  cpu_seconds=round(sum(last[i]["seconds"] for i in idx), 1))
    print(F, json.dumps(fam[F]))
out["family"] = fam
blocks = load("cert_blocks.jsonl")
lastb = {(r["geom"], r["N"], r["d"]): r for r in blocks}
needb = [("CC", N, d) for N in range(3, 9) for d in (2, 3)] + [("CEN", N, d) for N in range(9, 25, 2) for d in (2, 3)] \
        + [("CEN", N, d) for N in (5, 7) for d in (3, 4, 5)] + [("CEN", 3, d) for d in (4, 5, 6, 7)]
goodb = all(k in lastb and lastb[k]["status"].startswith("LOGCONCAVE") for k in needb)
ok &= goodb
out["blocks"] = dict(cases=len(needb), all_certified=goodb, max_t0=max(lastb[k]["t0"] for k in needb),
                     min_rel_margin=min(lastb[k]["min_rel_margin"] for k in needb),
                     all_exact_crosschecks=all(lastb[k]["exact_crosscheck"] and lastb[k]["exact_crosscheck_ok"] for k in needb))
print("blocks", json.dumps(out["blocks"]))
out["ALL_OK"] = bool(ok)
json.dump(out, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'summary_lc1d.json'), "w"), indent=1)
print("ALL OK" if ok else "INCOMPLETE")
