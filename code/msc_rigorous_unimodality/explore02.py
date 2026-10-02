"""float exploration: corner-to-corner at q=1 (and others), d=2,3 larger N: runs of sign of Delta f.
Uses float stepping with symmetric update; reports sign runs with a relative tolerance to flag ambiguous."""
import sys, time, numpy as np
def runs_float(N,d,q,start,target,T):
    shape=(N,)*d; w=q/(2*d)
    nblk=np.zeros(shape)
    sl=[]
    for ax in range(d):
        i0=[slice(None)]*d; i0[ax]=0; nblk[tuple(i0)]+=1
        i1=[slice(None)]*d; i1[ax]=N-1; nblk[tuple(i1)]+=1
        lo=[slice(None)]*d; hi=[slice(None)]*d; lo[ax]=slice(0,N-1); hi[ax]=slice(1,N); sl.append((tuple(lo),tuple(hi)))
    stay=(1-q)+w*nblk
    rho=np.zeros(shape); rho[start]=1.0
    f=np.zeros(T+1)
    for t in range(1,T+1):
        new=stay*rho
        for lo,hi in sl:
            new[hi]+=w*rho[lo]; new[lo]+=w*rho[hi]
        f[t]=new[target]; new[target]=0.0; rho=new
    return f
def describe(f,tol=1e-12):
    df=np.diff(f); sc=np.maximum(f[1:],f[:-1])
    sg=np.where(np.abs(df)<=tol*sc,0,np.sign(df)).astype(int)
    runs=[]
    for i,s in enumerate(sg):
        if not runs or runs[-1][0]!=s: runs.append([s,i,i])
        else: runs[-1][2]=i
    return runs
if __name__=="__main__":
    d=int(sys.argv[1]); q=float(sys.argv[2]); Ns=[int(x) for x in sys.argv[3:]]
    for N in Ns:
        t0=time.time()
        T=int((12 if d==2 else 8)*N*N/q)
        f=runs_float(N,d,q,(0,)*d,(N-1,)*d,T)
        r=describe(f)
        desc=' '.join('%s[%d..%d]'%('+' if s>0 else ('-' if s<0 else '0'),a,b) for s,a,b in r[:10])
        print(d,N,q,'nruns',len(r),desc,'mode',int(np.argmax(f)),'%.1fs'%(time.time()-t0),flush=True)
