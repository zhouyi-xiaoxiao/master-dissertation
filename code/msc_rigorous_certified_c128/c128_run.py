"""c128_run.py -- run the C engine fp128 over a set of N, post-process with Python integers, append certificate lines.

usage: c128_run.py d qn qd NSPEC [workers [budget_s]]
   NSPEC: e.g. 2-160  or  601-1000  or  180,200,401  (ranges and lists may be mixed with commas)
Output: certs/c128_d{d}_q{qn}-{qd}.jsonl  (one line per N; N already present are skipped).
Each invocation of fp128 is limited to budget_s seconds (default 900); if the budget is exhausted the program writes a
checkpoint and is invoked again, so no single computation exceeds the limit.
If a Tier-A certificate (certs/modes_*.jsonl, generator fplimb.py) exists for the same N, the two are compared:
t0, mode, T, all verdicts must be identical and the enclosures of 2^P sigma_t at t*-1, t*, t*+1 must intersect.
"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, os, subprocess, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _os.path.join(_R, 'proofs', 'R5_certified_computations')
WORK, CERT = _os.path.join(_R, 'out', 'work', 'msc_rigorous_certified'), _os.path.join(_R, 'data', 'msc_rigorous_certified')
sys.path.insert(0, HERE)
import c128_post

KEYS = ("t0", "mode", "T_tail", "mode_certified", "unimodal_certified", "certified_local_maxima_upto_T", "undecided_adjacent_pairs")


def one(job):
    d, N, qn, qd, budget = job
    t0 = time.time()
    out = os.path.join(WORK, f"c128_d{d}_q{qn}-{qd}_N{N}.bin")
    calls = 0
    while True:
        calls += 1
        rc = subprocess.run([os.path.join(WORK, "fp128"), str(d), str(N), str(qn), str(qd), "1", out, str(budget)],
                            stdout=subprocess.DEVNULL).returncode
        if rc == 0:
            break
        if rc != 75:
            return dict(d=d, N=N, q=f"{qn}/{qd}", error=f"fp128 exit code {rc}")
    c_seconds = time.time() - t0
    c = c128_post.certificate(out)
    os.remove(out)
    c["fp128_calls"] = calls
    c["seconds"] = round(time.time() - t0, 2)
    c["seconds_engine"] = round(c_seconds, 2)
    return c


def parse(spec):
    Ns = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            Ns += list(range(int(a), int(b) + 1))
        else:
            Ns.append(int(part))
    return Ns


if __name__ == "__main__":
    d, qn, qd = (int(x) for x in sys.argv[1:4])
    Ns = parse(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    budget = float(sys.argv[6]) if len(sys.argv) > 6 else 900.0
    tag = f"d{d}_q{qn}-{qd}"
    outp = os.path.join(CERT, f"c128_{tag}.jsonl")
    done = {json.loads(l)["N"] for l in open(outp)} if os.path.exists(outp) else set()
    gen = {}
    gp = os.path.join(CERT, f"modes_{tag}.jsonl")
    if os.path.exists(gp):
        gen = {json.loads(l)["N"]: json.loads(l) for l in open(gp)}
    jobs = [(d, N, qn, qd, budget) for N in sorted(set(Ns) - done, reverse=True)]
    bad = 0
    with Pool(workers) as pool:
        for c in pool.imap_unordered(one, jobs):
            if "error" in c:
                print("ERROR", c, flush=True)
                bad += 1
                continue
            N = c["N"]
            if N in gen:
                gcert = gen[N]
                same = all(c[k] == gcert[k] for k in KEYS)
                inter = all(not (int(a) > int(c["F_enclosures"][t][1]) or int(b) < int(c["F_enclosures"][t][0]))
                            for t, (a, b) in gcert["F_enclosures"].items() if t in c["F_enclosures"])
                c["same_verdicts_as_generator"], c["enclosures_intersect_generator"] = bool(same), bool(inter)
                bad += not (same and inter)
            with open(outp, "a") as fh:
                fh.write(json.dumps(c) + "\n")
            print(f"{tag} N={N} mode={c['mode']} T={c['T_tail']} cert={c['mode_certified']} unimodal={c['unimodal_certified']} "
                  f"peaks={c['certified_local_maxima_upto_T']} und={c['undecided_adjacent_pairs']} "
                  f"same={c.get('same_verdicts_as_generator')} inter={c.get('enclosures_intersect_generator')} "
                  f"calls={c['fp128_calls']} {c['seconds']}s", flush=True)
    sys.exit(1 if bad else 0)
