"""Compare two sets of window certificates of note R3 block by block (constants c_plus, c_minus and all Boolean flags).

Usage: python 83_compare_cert_runs.py DIR_A [DIR_B]
  DIR_A, DIR_B  directories that contain certificate files 52_*.jsonl, 72_*.jsonl, 76_*.jsonl, 78_*.jsonl (paths as
                given, relative to the current directory); DIR_B defaults to the directory of the stored certificates.
Typical use: move a stored certificate file aside, regenerate it with its script, and compare the two directories.
Every file of DIR_B is compared with the file of the same name in DIR_A; the script stops with an error if no pair of
files is found."""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
if len(sys.argv) < 2:
    sys.exit(__doc__)
a_dir = os.path.abspath(sys.argv[1])
b_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else _os.path.join(_R, 'data', 'msc_rigorous_asymptotics')
keys = ['c_plus', 'c_minus', 'ok_plus', 'ok_minus', 'ok', 'ok_A', 'ok_B', 'ok_C', 'ok_D']
tot_files = tot_blocks = tot_diff = 0
for fb in sorted(glob.glob(os.path.join(b_dir, '[57][2678]_*.jsonl'))):
    fa = os.path.join(a_dir, os.path.basename(fb))
    if not os.path.exists(fa):
        print(os.path.basename(fb), 'no counterpart in', a_dir); continue
    A = {(r['Na'], r['Nb']): r for r in map(json.loads, open(fa))}
    B = {(r['Na'], r['Nb']): r for r in map(json.loads, open(fb))}
    diffs = [(k, kk, A[k].get(kk), B[k].get(kk)) for k in B if k in A for kk in keys if kk in B[k] and A[k].get(kk) != B[k].get(kk)]
    miss = len(set(A) ^ set(B))
    tot_files += 1; tot_blocks += len(B); tot_diff += len(diffs) + miss
    print('%-28s blocks %4d  differences %d  unmatched blocks %d' % (os.path.basename(fb), len(B), len(diffs), miss), diffs[:3])
if tot_files == 0:
    sys.exit('error: no certificate file of %s has a counterpart in %s' % (b_dir, a_dir))
print('TOTAL files %d, blocks %d, differences %d' % (tot_files, tot_blocks, tot_diff))
