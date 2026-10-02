"""c128_validate_ref.py -- byte-for-byte comparison of fp128 (C, optimised and -DCHECK/UBSan builds) with the
pure-Python mirror c128_ref.py on a list of small cases, reduced and unreduced.  Output: certs/c128_refcheck.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import c128_ref

cases = []
for qn, qd in ((4, 5), (1, 2), (1, 1)):
    cases += [(1, N, qn, qd) for N in list(range(2, 41)) + [50, 64, 80]]
    cases += [(2, N, qn, qd) for N in list(range(2, 17)) + [20, 24]]
    cases += [(3, N, qn, qd) for N in list(range(2, 9)) + [10]]
extra_q = [(1, 12, 2, 3), (2, 9, 2, 3), (3, 5, 2, 3), (1, 12, 3, 4), (1, 10, 1, 3), (2, 6, 1, 3), (1, 12, 3, 5)]   # other q with D <= 15
cases += extra_q
out, bad, t0 = [], 0, time.time()
for (d, N, qn, qd) in cases:
    for red in (1, 0):
        if d == 1 and red == 0:
            continue
        ref = c128_ref.run(d, N, qn, qd, red)
        res = {}
        for exe in ("fp128", "fp128_check"):
            p = os.path.join(WORK, f"val_{exe}.bin")
            subprocess.run([os.path.join(WORK, exe), str(d), str(N), str(qn), str(qd), str(red), p], check=True, stdout=subprocess.DEVNULL)
            res[exe] = open(p, "rb").read() == ref
            os.remove(p)
        ok = all(res.values())
        bad += not ok
        out.append(dict(d=d, N=N, q=f"{qn}/{qd}", reduced=red, bytes=len(ref), sha256=hashlib.sha256(ref).hexdigest(), identical=ok))
        print(d, N, f"{qn}/{qd}", "red" if red else "full", len(ref), "OK" if ok else "DIFFERENT", flush=True)
json.dump(dict(cases=len(out), mismatches=bad, seconds=round(time.time() - t0, 1), results=out),
          open(_os.path.join(_R, 'data', 'msc_rigorous_certified', 'c128_refcheck.json'), "w"), indent=0)
print(len(out), "cases,", bad, "mismatches")
sys.exit(1 if bad else 0)
