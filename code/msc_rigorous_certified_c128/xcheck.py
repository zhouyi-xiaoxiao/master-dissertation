"""xcheck.py -- cross-check certificates of the C engine in the extended ranges with the programs of Section 3.

usage: xcheck.py limb   d qn qd N [N ...]     (generator fplimb.py: numpy int64 limbs + exact Python-integer early phase)
       xcheck.py packed d qn qd N [N ...]     (verify_packed.py: Python integers only)
       xcheck.py exact  d qn qd N [N ...]     (verify_packed.py --exact: no rounding; decides ties; small N only)
Each N is one computation; nothing is written to the Tier-A files.
Output: certs/xcheck_{engine}_d{d}_q{qn}-{qd}.jsonl with the full line of the checking program and the comparison:
 same_verdicts  : t0, mode, T_tail, mode_certified, unimodal_certified, certified_local_maxima_upto_T identical
 enclosures_intersect : the enclosures of 2^P sigma_t at t*-1, t*, t*+1 (same scale 2^112) intersect
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
sys.path.insert(0, _os.path.join(_R, 'code', 'msc_rigorous_certified'))
KEYS = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T")

engine, d, qn, qd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
Ns = [int(x) for x in sys.argv[5:]]
tag = f"d{d}_q{qn}-{qd}"
outp = os.path.join(_R, 'data', 'msc_rigorous_certified', f"xcheck_{engine}_{tag}.jsonl")
done = {json.loads(l)["N"] for l in open(outp)} if os.path.exists(outp) else set()
if engine == "limb":
    import fplimb
    run = lambda N: fplimb.certify(d, N, qn, qd)
elif engine == "exact":                      # no rounding at all: decides ties (small N only)
    import verify_packed
    run = lambda N: verify_packed.certify(d, N, qn, qd, exact_only=True)
else:
    import verify_packed
    run = lambda N: verify_packed.certify(d, N, qn, qd)
bad = 0
for N in Ns:
    if N in done:
        continue
    recheck = {json.loads(l)["N"]: json.loads(l) for l in open(os.path.join(_R, 'data', 'msc_rigorous_certified', f"c128_{tag}.jsonl"))}.get(N)
    if recheck is None:
        print("no c128 certificate yet for N =", N, flush=True)
        continue
    t0 = time.time()
    c = run(N)
    c["checker"] = engine
    c["seconds"] = round(time.time() - t0, 2)
    if engine == "exact":                    # the exact program knows the truth: every claim of the C certificate must hold
        c["consistent_with_c128"] = bool(c["t0"] == recheck["t0"] and recheck["mode"] in c["maximisers_of_lower_bound"]
                                         and (c["mode_certified"] or not recheck["mode_certified"])
                                         and (c["unimodal_certified"] or not recheck["unimodal_certified"])
                                         and c["T_tail"] <= recheck["T_tail"]
                                         and c["certified_local_maxima_upto_T"] >= recheck["certified_local_maxima_upto_T"])
    c["same_verdicts"] = all(c[k] == recheck[k] for k in KEYS)
    c["enclosures_intersect"] = all(not (int(a) > int(recheck["F_enclosures"][t][1]) or int(b) < int(recheck["F_enclosures"][t][0]))
                                    for t, (a, b) in c["F_enclosures"].items() if t in recheck["F_enclosures"])
    bad += not (c["same_verdicts"] and c["enclosures_intersect"])
    with open(outp, "a") as fh:
        fh.write(json.dumps(c) + "\n")
    print(engine, tag, "N =", N, "mode", c["mode"], "T", c["T_tail"], "same", c["same_verdicts"], "intersect", c["enclosures_intersect"],
          c["seconds"], "s", flush=True)
sys.exit(1 if bad else 0)
