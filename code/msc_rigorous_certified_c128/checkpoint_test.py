"""checkpoint_test.py -- the checkpoint/restart path of the C engine (not part of any proof).

For each case the binary work/fp128 (sha256 compared with certs/c128_build.json) is run with a time budget of
`budget` seconds, so that it saves its state and exits with code 75 again and again; it is restarted until it
finishes.  The sha256 digest of the complete output file must equal the field 'digest' of the certificate
(certs/c128_*.jsonl), which was produced by an uninterrupted run.   Output: certs/c128_checkpoint_test.json;
the binary outputs stay in work/ckpttest_*.bin.
usage: checkpoint_test.py [budget_seconds]
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK, CERT = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified'), _os.path.join(_R, 'data', 'msc_rigorous_certified')
BUDGET = sys.argv[1] if len(sys.argv) > 1 else "0.5"
CASES = [(1, 1500, 4, 5), (3, 50, 4, 5), (2, 120, 4, 5)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


build = json.load(open(os.path.join(CERT, "c128_build.json")))
exe = os.path.join(WORK, "fp128")
binary_ok = sha(exe) == build["binary_sha256"]
res = []
for d, N, qn, qd in CASES:
    cert = next(json.loads(l) for l in open(os.path.join(CERT, f"c128_d{d}_q{qn}-{qd}.jsonl")) if json.loads(l)["N"] == N)
    out = os.path.join(WORK, f"ckpttest_d{d}_N{N}.bin")
    # an existing out.bin is overwritten; a checkpoint left by an interrupted run of this test is simply continued
    # (that is the path under test, and the digest comparison below still applies).  No file is deleted.
    t0, calls, interruptions = time.time(), 0, 0
    while True:
        calls += 1
        rc = subprocess.run([exe, str(d), str(N), str(qn), str(qd), "1", out, BUDGET], stdout=subprocess.DEVNULL).returncode
        if rc == 0:
            break
        assert rc == 75, rc
        interruptions += 1
    dig = sha(out)                                         # the output stays in work/ (overwritten by the next run)
    res.append(dict(d=d, N=N, q=f"{qn}/{qd}", budget_seconds=float(BUDGET), interruptions=interruptions, calls=calls,
                    sha256=dig, identical_to_certificate=(dig == cert["digest"]), seconds=round(time.time() - t0, 2)))
    print(res[-1], flush=True)
json.dump(dict(binary_sha256_matches_build=binary_ok, cases=res,
               all_identical=all(r["identical_to_certificate"] for r in res)),
          open(os.path.join(CERT, "c128_checkpoint_test.json"), "w"), indent=1)
print("binary matches build:", binary_ok, " all identical:", all(r["identical_to_certificate"] for r in res))
