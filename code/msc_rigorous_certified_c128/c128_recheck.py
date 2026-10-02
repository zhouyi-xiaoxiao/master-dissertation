"""c128_recheck.py -- second run of every C certificate, with two checks done in Python integers:

 (V4) reproducibility: the binary output file of the second run has the same sha256 digest as the file of the first run
      (field 'digest' of the certificate), i.e. the two runs produced the same integers F^-_t, F^+_t (all t), the same
      early relations, the same T and the same final vectors.  Guards against transient hardware faults.
 (V5) conservation of probability: 1 = sum_y v_t(y) + sum_{s<t} f(s) and f = (cm/D) sigma give, for the enclosures,
          D sum_y mu(y) L_T(y) + cm sum_{s=1}^{T-1} F^-_s   <=   D 2^P   <=   D sum_y mu(y) U_{T+1}(y) + cm sum_{s=1}^{T} F^+_s
      (mu(y) = number of lattice sites represented by the stored site y).  This ties the whole sequence F^+-_t to the
      final vectors; a wrap-around or an indexing error anywhere in the run would break it.  The slack of the two
      inequalities (in units of 2^-P) is recorded.
Neither check is part of a proof.

usage: c128_recheck.py [workers]        -> certs/c128_recheck_d{d}_q{q}.jsonl   (restartable)
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, hashlib, json, os, subprocess, sys, time
from itertools import product
from math import factorial
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK, CERT = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified'), _os.path.join(_R, 'data', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import c128_post


def multiplicities(d, N):
    """orbit sizes of the stored (sorted) sites, in storage order (index x_1 + N x_2 + N^2 x_3)."""
    out = []
    for x in product(range(N), repeat=d):
        x = x[::-1]
        if all(x[k] <= x[k + 1] for k in range(d - 1)):
            m = factorial(d)
            for v in set(x):
                m //= factorial(x.count(v))
            out.append(m)
    return out


def one(job):
    d, N, qn, qd, digest = job
    t0 = time.time()
    out = os.path.join(WORK, f"recheck_d{d}_q{qn}-{qd}_N{N}.bin")
    calls = 0
    while True:
        calls += 1
        rc = subprocess.run([os.path.join(WORK, "fp128"), str(d), str(N), str(qn), str(qd), "1", out, "900"], stdout=subprocess.DEVNULL).returncode
        if rc == 0:
            break
        if rc != 75:
            return dict(d=d, N=N, q=f"{qn}/{qd}", error=f"fp128 exit code {rc}")
    r = c128_post.read_bin(out)
    os.remove(out)
    mu = multiplicities(d, N)
    T, P, D, cm = r["T"], r["P"], r["D"], r["cm"]
    assert len(mu) == r["n"] and sum(mu) == N ** d
    lower = D * sum(m * v for m, v in zip(mu, r["L"])) + cm * sum(r["FL"][:T - 1])        # FL[s-1] = F^-_s
    upper = D * sum(m * v for m, v in zip(mu, r["U"])) + cm * sum(r["FU"][:T])
    one_ = D << P
    return dict(d=d, N=N, q=f"{qn}/{qd}", T_tail=T, sha256_second_run=r["sha256"], identical_to_first_run=(r["sha256"] == digest),
                conservation_ok=bool(lower <= one_ <= upper), slack_lower=str(one_ - lower), slack_upper=str(upper - one_),
                relative_slack=float(max(one_ - lower, upper - one_)) / float(one_), fp128_calls=calls, seconds=round(time.time() - t0, 2))


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    jobs = []
    for f in sorted(glob.glob(os.path.join(CERT, "c128_d*_q*.jsonl"))):
        tag = os.path.basename(f)[len("c128_"):-len(".jsonl")]
        outp = os.path.join(CERT, f"c128_recheck_{tag}.jsonl")
        done = {json.loads(l)["N"] for l in open(outp)} if os.path.exists(outp) else set()
        for line in open(f):
            c = json.loads(line)
            if c["N"] not in done:
                qn, qd = (int(x) for x in c["q"].split("/"))
                jobs.append((c["T_tail"] * c["stored_sites"], (c["d"], c["N"], qn, qd, c["digest"])))
    jobs.sort(reverse=True)
    bad = 0
    with Pool(workers) as pool:
        for c in pool.imap_unordered(one, [j for _, j in jobs]):
            if "error" in c:
                print("ERROR", c, flush=True)
                bad += 1
                continue
            tag = f"d{c['d']}_q{c['q'].replace('/', '-')}"
            with open(os.path.join(CERT, f"c128_recheck_{tag}.jsonl"), "a") as fh:
                fh.write(json.dumps(c) + "\n")
            ok = c["identical_to_first_run"] and c["conservation_ok"]
            bad += not ok
            print(tag, "N =", c["N"], "identical", c["identical_to_first_run"], "conservation", c["conservation_ok"],
                  f"slack {c['relative_slack']:.2e}", c["seconds"], "s", flush=True)
    print("RECHECK-DONE", "failures:", bad)
    sys.exit(1 if bad else 0)
