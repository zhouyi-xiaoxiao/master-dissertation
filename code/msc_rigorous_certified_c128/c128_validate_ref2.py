"""c128_validate_ref2.py -- byte-for-byte comparison of fp128 with the pure-Python mirror on a few medium-size cases
(thousands of time steps, thousands of sites).  Output: certs/c128_refcheck_medium.json"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import c128_ref

cases = [(1, 300, 4, 5, 1), (2, 60, 4, 5, 1), (2, 40, 4, 5, 0), (3, 20, 4, 5, 1), (3, 12, 4, 5, 0),
         (2, 50, 1, 2, 1), (3, 16, 1, 2, 1), (1, 200, 1, 1, 1), (2, 38, 1, 1, 1), (3, 16, 1, 1, 1)]
outp = _os.path.join(_R, 'data', 'msc_rigorous_certified', 'c128_refcheck_medium.json')
out, bad = [], 0
for (d, N, qn, qd, red) in cases:
    t0 = time.time()
    ref = c128_ref.run(d, N, qn, qd, red)
    p = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified', 'val2.bin')
    subprocess.run([_os.path.join(_R, 'out', 'work', 'msc_rigorous_certified', 'fp128'), str(d), str(N), str(qn), str(qd), str(red), p], check=True, stdout=subprocess.DEVNULL)
    ok = open(p, "rb").read() == ref
    os.remove(p)
    bad += not ok
    out.append(dict(d=d, N=N, q=f"{qn}/{qd}", reduced=red, bytes=len(ref), sha256=hashlib.sha256(ref).hexdigest(), identical=ok,
                    seconds_python=round(time.time() - t0, 1)))
    print(out[-1], flush=True)
    json.dump(dict(cases=len(out), mismatches=bad, results=out), open(outp, "w"), indent=0)
print(len(out), "cases,", bad, "mismatches")
sys.exit(1 if bad else 0)
