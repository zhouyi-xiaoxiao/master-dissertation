"""v07c -- additional placements for the size study (rep indices 1000+): python v07c_more.py N p K workers"""
import os as _os
_R = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # root of the repository
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
import v07_size as v7
HERE = os.path.dirname(os.path.abspath(__file__)); D = _os.path.join(_R, 'data', 'msc_defects_verify', 'v07_size')
if __name__ == '__main__':
    N = int(sys.argv[1]); p = float(sys.argv[2]); K = int(sys.argv[3]); w = int(sys.argv[4])
    fn = os.path.join(D, f'N{N}_p{p:.2f}_extra.csv')
    if not os.path.exists(fn):
        with Pool(w) as pool:
            df = pd.DataFrame(pool.map(v7.work, [(N, p, 1000 + r) for r in range(K)], chunksize=1))
        df.to_csv(fn, index=False)
    print(N, p, K, 'done', flush=True)
