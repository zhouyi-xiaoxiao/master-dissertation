"""summarize_certs.py -- collect all exact certificates into data/summary_certificates.json and print the
ranges that enter Computer-assisted Theorems 7.1 and 8.2."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_unimodality')
groups = {}
for fn in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'cert_*.jsonl'))):
    for line in open(fn):
        r = json.loads(line)
        key = (r.get("geo", "?"), r["d"], r["q"])
        groups.setdefault(key, {})[r["N"]] = r
summary = []
for key in sorted(groups):
    recs = groups[key]
    Ns = sorted(recs)
    uni = [N for N in Ns if recs[N]["unimodal"] is True]
    non = [N for N in Ns if recs[N]["unimodal"] is False]
    und = [N for N in Ns if recs[N]["unimodal"] is None]
    cone_eq = [N for N in uni if recs[N]["T_cone"] == recs[N]["mode"] - 1]
    s = dict(geo=key[0], d=key[1], q=key[2], N_list=Ns, n=len(Ns), unimodal=uni, not_unimodal=non, undecided=und,
             max_steps=max(recs[N]["steps"] for N in Ns), max_bits=max(recs[N]["bits_last"] for N in Ns),
             total_seconds=round(sum(recs[N]["seconds"] for N in Ns), 1),
             cone_time_equals_mode_minus_1=len(cone_eq), modes={N: recs[N]["mode"] for N in uni},
             T_cone={N: recs[N]["T_cone"] for N in Ns})
    summary.append(s)
    def rng(v):
        if not v: return "-"
        out = []; a = b = v[0]
        for x in v[1:]:
            if x == b + 1 or (x == b + 2 and all(y % 2 == 1 for y in v)): b = x
            else: out.append((a, b)); a = b = x
        out.append((a, b))
        return ",".join(str(a) if a == b else f"{a}..{b}" for a, b in out)
    print(f"{key[0]:4s} d={key[1]} q={key[2]:9s} N: {rng(Ns):12s} unimodal: {rng(uni):14s} NOT: {non}  undecided: {und}  "
          f"cone=mode-1 in {len(cone_eq)}/{len(uni)}  max steps {s['max_steps']}  max bits {s['max_bits']}  cpu {s['total_seconds']}s")
json.dump(summary, open(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'summary_certificates.json'), "w"))
# pairs
for fn in sorted(glob.glob(_os.path.join(_R, 'data', 'msc_rigorous_unimodality', 'pairs_d*_N*.json'))):
    P = json.load(open(fn))
    non = [p for p in P["pairs"] if p["cont"].startswith("NOT")]
    art = [p for p in P["pairs"] if p["q12"] is False and p["cont"].startswith("unimodal")]
    und = [p for p in P["pairs"] if p["cont"].startswith("UNDEC")]
    print(f"pairs d={P['d']} N={P['N']}: {P['n_pairs']} pairs; q=1/2 unimodal: {sum(1 for p in P['pairs'] if p['q12'] is True)}; "
          f"continuous NOT unimodal (proved): {len(non)}; q=1/2 multimodal but continuous unimodal: {len(art)}; undecided: {len(und)}; "
          f"max #extrema among non-unimodal: {max([len(p.get('float_extrema', [])) for p in non] or [0])}")
