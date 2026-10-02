"""run_modes.py -- produce (engine=limb) or re-derive with a separately written engine (engine=ref) mode certificates.

usage: run_modes.py d qn qd Nmin Nmax [limb|ref] [budget_seconds] [step]
Appends one JSON line per N to certs/modes_d{d}_q{qn}-{qd}.jsonl      (limb engine)
                            or certs/refcheck_d{d}_q{qn}-{qd}.jsonl   (reference engine, Python ints)
Restartable: N already present in the output file are skipped.  Stops when the time budget is spent.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fpcore, fplimb

d, qn, qd, Nmin, Nmax = (int(x) for x in sys.argv[1:6])
engine = sys.argv[6] if len(sys.argv) > 6 else "limb"
budget = float(sys.argv[7]) if len(sys.argv) > 7 else 1000.0
stride = int(sys.argv[8]) if len(sys.argv) > 8 else 1
name = ("modes" if engine == "limb" else "refcheck") + f"_d{d}_q{qn}-{qd}.jsonl"
out = os.path.join(_R, 'data', 'msc_rigorous_certified', name)
done = set()
if os.path.exists(out):
    for line in open(out):
        done.add(json.loads(line)["N"])
start = time.time()
for N in range(Nmin, Nmax + 1, stride):
    if N in done:
        continue
    if time.time() - start > budget:
        print("budget exhausted before N =", N, flush=True)
        break
    t0 = time.time()
    c = (fplimb if engine == "limb" else fpcore).certify(d, N, qn, qd)
    c["engine"] = engine
    c["seconds"] = round(time.time() - t0, 2)
    with open(out, "a") as fh:
        fh.write(json.dumps(c) + "\n")
    print(f"d={d} q={qn}/{qd} N={N} mode={c['mode']} T={c['T_tail']} cert={c['mode_certified']} "
          f"unimodal={c['unimodal_certified']} nmax={c['certified_local_maxima_upto_T']} {c['seconds']}s", flush=True)
