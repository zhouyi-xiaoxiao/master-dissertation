/* fp128.c -- certified enclosures of the corner-to-corner first-passage PMF (second engine).
 *
 * TRUSTED BASE: C99 unsigned integer arithmetic (uint64_t and unsigned __int128: + - * / % << >> & |, comparisons).
 * No floating point is used for any certified quantity (clock()/time only for the time budget).
 *
 * Model (Section 2 of note R5).  Lazy walk on {0..N-1}^d, each of the 2d directions with probability q/(2d),
 * a move off the box is cancelled, target a = (N-1,..,N-1) absorbing, start x0 = 0.
 * Q = M/D on the non-target sites, M = cm*(A + B) + cs*I;  v_1 = e_x0, v_{t+1} = Q v_t,
 * sigma_t = sum of v_t over the d neighbours of a,  f(t) = (cm/D) sigma_t.
 *
 * Symmetry reduction (option red=1).  v_t is invariant under permutations of the coordinates, and so are the
 * integer vectors below (the same integer operations are performed at symmetric sites).  Only the sites with
 * x_1 <= x_2 <= .. <= x_d are stored; a neighbour is replaced by its sorted representative.
 *
 * (A) fixed point, all t >= 1 (Lemma "enclosure by directed rounding" with t_s = 1):
 *        L_1 = U_1 = 2^P e_x0,   L_{t+1} = floor(M L_t / D),   U_{t+1} = ceil(M U_t / D),   P = 112,
 *     so that L_t <= 2^P v_t <= U_t <= 2^P entrywise;  FL_t = sum_{y ~ a} L_t(y) <= 2^P sigma_t <= FU_t.
 * (B) floating intervals, early times only (while FL_t < 2^G, G = 48): lo_t <= v_t <= hi_t entrywise with
 *     lo, hi in {0} u {m 2^e : 2^62 <= m < 2^63}; every rounding is downwards for lo and upwards for hi, and 0
 *     is produced only from exact zeros.  It yields a certified relation relf_t in {'<','>','=','?'} between
 *     sigma_t and sigma_{t+1}  ('=' only if both are exactly 0), which the fixed-point data cannot give when
 *     sigma_t is far below 2^-P.
 * Tail time T: first t with U_{t+1}(y) < L_t(y) at every stored non-target site.
 *
 * Output file (little endian): header, FL_t and FU_t for t = 1..T+1 (16 bytes each), relf_t for t = 1..T (1 byte each),
 * then L_T and U_{T+1} at all stored sites (16 bytes each).  The verdicts are formed from this file by c128_post.py
 * (Python integers), which also re-checks U_{T+1} < L_T.
 *
 * usage: fp128 d N qn qd red out.bin [budget_seconds]
 *   exit 0: finished;  exit 75: budget exhausted, state saved in out.bin.ckpt (run again to continue).
 * build: cc -O2 -o fp128 fp128.c        (add -DCHECK for internal consistency checks of the division routines)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;

#define P 112
#define G 48
#define MAXD 3

static int d, N, qn, qd, cm, cs, D, reduced;
static long n, tgt;                 /* number of stored sites, stored index of the target                  */
static int32_t *nb;                 /* nb[2d*r+k]: stored index of the k-th neighbour of r (r itself if the  */
                                    /* move is cancelled; the target's slot always holds 0)                 */
static long snb[MAXD];              /* stored sites adjacent to the target ...                              */
static int scoef[MAXD], nsnb;       /* ... and how many of the d neighbours of a each of them represents     */

static void die(const char *s) { fprintf(stderr, "fp128: %s\n", s); exit(2); }

static int igcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

static void build(void)
{
    long full = 1;
    for (int k = 0; k < d; k++) full *= N;
    int32_t *map = malloc(full * sizeof(int32_t));
    if (!map) die("malloc");
    n = 0;
    for (long i = 0; i < full; i++) {
        int x[MAXD]; long j = i; int keep = 1;
        for (int k = 0; k < d; k++) { x[k] = (int)(j % N); j /= N; }
        if (reduced) for (int k = 0; k + 1 < d; k++) if (x[k] > x[k + 1]) keep = 0;
        map[i] = keep ? (int32_t)n++ : -1;
    }
    nb = malloc((size_t)n * 2 * d * sizeof(int32_t));
    if (!nb) die("malloc");
    for (long i = 0; i < full; i++) {
        long r = map[i];
        if (r < 0) continue;
        int x[MAXD]; long j = i;
        for (int k = 0; k < d; k++) { x[k] = (int)(j % N); j /= N; }
        for (int k = 0; k < d; k++) for (int s = 0; s < 2; s++) {
            int y[MAXD];
            for (int m = 0; m < d; m++) y[m] = x[m];
            y[k] += s ? 1 : -1;
            long res;
            if (y[k] < 0 || y[k] > N - 1) res = r;                       /* cancelled move: the walker stays */
            else {
                if (reduced)                                             /* sort ascending (d <= 3)          */
                    for (int a = 0; a < d; a++) for (int b = 0; b + 1 < d; b++)
                        if (y[b] > y[b + 1]) { int t = y[b]; y[b] = y[b + 1]; y[b + 1] = t; }
                long e = 0;
                for (int m = d - 1; m >= 0; m--) e = e * N + y[m];
                res = map[e];
                if (res < 0) die("internal: neighbour is not a representative");
            }
            nb[2 * d * r + 2 * k + s] = (int32_t)res;
        }
    }
    tgt = map[full - 1];
    if (reduced) {                      /* the d neighbours of a form one orbit, represented by (N-2,N-1,..,N-1) */
        snb[0] = map[full - 2]; scoef[0] = d; nsnb = 1;
    } else {
        long s = 1;
        for (int k = 0; k < d; k++) { snb[k] = map[full - 1 - s]; scoef[k] = 1; s *= N; }
        nsnb = d;
    }
    free(map);
}

/* ---------------------------------------------------------------- (A) fixed point */

#define M58 (((u64)1 << 58) - 1)

/* floor(W / D) for 0 <= W < 2^116, 1 <= D <= 15, with 64-bit divisions only:
 * W = w1 2^58 + w0,  w1 = D q1 + r1  =>  floor(W/D) = q1 2^58 + floor((r1 2^58 + w0)/D),  r1 2^58 + w0 < 2^62. */
static inline u128 divfloor(u128 W)
{
    u64 w1 = (u64)(W >> 58), w0 = (u64)W & M58;
    u64 q1 = w1 / (u64)D, r1 = w1 % (u64)D;
    u64 c = (r1 << 58) | w0;
    u128 res = ((u128)q1 << 58) + c / (u64)D;
#ifdef CHECK
    if ((W >> 116) != 0) die("CHECK: W >= 2^116");
    if (res != W / (u128)D) die("CHECK: divfloor");
#endif
    return res;
}

static void step_fixed(const u128 *L, const u128 *U, u128 *Ln, u128 *Un)
{
    const int k2 = 2 * d;
    for (long r = 0; r < n; r++) {
        if (r == tgt) { Ln[r] = 0; Un[r] = 0; continue; }            /* mass entering the target is absorbed */
        const int32_t *p = nb + (size_t)k2 * r;
        u128 sl = 0, su = 0;
        for (int k = 0; k < k2; k++) { sl += L[p[k]]; su += U[p[k]]; }
        u128 wl = (u128)cm * sl + (u128)cs * L[r];
        u128 wu = (u128)cm * su + (u128)cs * U[r];
        Ln[r] = divfloor(wl);
        Un[r] = divfloor(wu + (u128)(D - 1));
#ifdef CHECK
        if (Un[r] > ((u128)1 << P) || Ln[r] > Un[r]) die("CHECK: enclosure order / bound");
#endif
    }
}

static void fsum_fixed(const u128 *L, const u128 *U, u128 *fl_, u128 *fu_)
{
    u128 a = 0, b = 0;
    for (int k = 0; k < nsnb; k++) { a += (u128)scoef[k] * L[snb[k]]; b += (u128)scoef[k] * U[snb[k]]; }
    *fl_ = a; *fu_ = b;
}

/* ---------------------------------------------------------------- (B) floating intervals */

typedef struct { u64 m; int32_t e; } flo;      /* the number m 2^e;  m = 0 (then e = 0) or 2^62 <= m < 2^63 */

#define M42 (((u64)1 << 42) - 1)

/* floor(W / Dv) for 0 <= W < 2^126, 1 <= Dv <= 15, by long division in base 2^42 (64-bit divisions only). */
static inline u128 divsmall(u128 W, int Dv)
{
    u64 a2 = (u64)(W >> 84), a1 = (u64)(W >> 42) & M42, a0 = (u64)W & M42;
    u64 q2 = a2 / (u64)Dv, r = a2 % (u64)Dv;
    u64 t = (r << 42) | a1;
    u64 q1 = t / (u64)Dv; r = t % (u64)Dv;
    t = (r << 42) | a0;
    u64 q0 = t / (u64)Dv;
    u128 res = ((u128)q2 << 84) + ((u128)q1 << 42) + q0;
#ifdef CHECK
    if ((W >> 126) != 0) die("CHECK: W >= 2^126");
    if (res != W / (u128)Dv) die("CHECK: divsmall");
#endif
    return res;
}

static inline int bitlen(u128 q)            /* number of binary digits of q > 0 */
{
    u64 hi = (u64)(q >> 64), lo = (u64)q;
    return hi ? 128 - __builtin_clzll(hi) : 64 - __builtin_clzll(lo);
}

/* A number in the format, <= (sum_j c_j x_j)/Dv if up = 0 and >= it if up = 1; it is 0 iff the sum is 0.
 * Requires 0 <= c_j, sum_j c_j <= 15, 1 <= Dv <= 15. */
static flo fsum(int k, const int *c, const flo *x, int Dv, int up)
{
    int any = 0; int32_t E = 0;
    for (int j = 0; j < k; j++) if (c[j] && x[j].m) { if (!any || x[j].e > E) E = x[j].e; any = 1; }
    flo z = {0, 0};
    if (!any) return z;
    u128 acc = 0;                               /* acc 2^E is a lower (upper) bound of sum_j c_j x_j        */
    for (int j = 0; j < k; j++) if (c[j] && x[j].m) {
        int64_t s = (int64_t)E - x[j].e;        /* >= 0                                                      */
        u64 v;
        if (s >= 64) v = up ? 1 : 0;
        else {
            v = x[j].m >> s;
            if (up && s > 0 && (x[j].m & (((u64)1 << s) - 1))) v++;
        }
        acc += (u128)c[j] * v;
    }                                           /* 2^62 <= acc <= 15 * 2^63 < 2^67                           */
    u128 num = acc << 58;                       /* < 2^125                                                   */
    u128 q = up ? divsmall(num + (u128)(Dv - 1), Dv) : divsmall(num, Dv);     /* q >= 2^120/15 > 2^116       */
    int s = bitlen(q) - 63;                     /* > 0                                                       */
    u64 m = (u64)(q >> s);
    if (up && (q & (((u128)1 << s) - 1))) m++;
    if (m == ((u64)1 << 63)) { m >>= 1; s++; }
    z.m = m; z.e = (int32_t)(E - 58 + s);
#ifdef CHECK
    if (s <= 0 || m < ((u64)1 << 62) || m >= ((u64)1 << 63)) die("CHECK: fsum normalisation");
#endif
    return z;
}

static inline int flt(flo a, flo b)             /* a < b (exact comparison of two numbers of the format)      */
{
    if (b.m == 0) return 0;
    if (a.m == 0) return 1;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}

static void step_float(const flo *lo, const flo *hi, flo *lon, flo *hin)
{
    const int k2 = 2 * d;
    int c[2 * MAXD + 1];
    flo x[2 * MAXD + 1];
    for (int k = 0; k < k2; k++) c[k] = cm;
    c[k2] = cs;
    flo z = {0, 0};
    for (long r = 0; r < n; r++) {
        if (r == tgt) { lon[r] = z; hin[r] = z; continue; }
        const int32_t *p = nb + (size_t)k2 * r;
        for (int k = 0; k < k2; k++) x[k] = lo[p[k]];
        x[k2] = lo[r];
        lon[r] = fsum(k2 + 1, c, x, D, 0);
        for (int k = 0; k < k2; k++) x[k] = hi[p[k]];
        x[k2] = hi[r];
        hin[r] = fsum(k2 + 1, c, x, D, 1);
    }
}

static void sigma_float(const flo *lo, const flo *hi, flo *slo, flo *shi)
{
    flo x[MAXD];
    for (int k = 0; k < nsnb; k++) x[k] = lo[snb[k]];
    *slo = fsum(nsnb, scoef, x, 1, 0);
    for (int k = 0; k < nsnb; k++) x[k] = hi[snb[k]];
    *shi = fsum(nsnb, scoef, x, 1, 1);
}

/* ---------------------------------------------------------------- tail test */

static long bad = 0;        /* a site at which the last test failed (the start site 0 is never the target) */

static int tail_ok(const u128 *L, const u128 *Un)       /* U_{t+1}(y) < L_t(y) at every stored non-target site? */
{
    if (!(Un[bad] < L[bad])) return 0;
    for (long r = n - 1; r >= 0; r--)
        if (r != tgt && !(Un[r] < L[r])) { bad = r; return 0; }
    return 1;
}

/* ---------------------------------------------------------------- records, checkpoint, output */

static u128 *FLr, *FUr; static char *relf; static long cap;

static void grow(long t)
{
    if (t + 2 < cap) return;
    if (!cap) cap = 4096;
    while (t + 2 >= cap) cap *= 2;
    FLr = realloc(FLr, cap * sizeof(u128)); FUr = realloc(FUr, cap * sizeof(u128)); relf = realloc(relf, cap);
    if (!FLr || !FUr || !relf) die("realloc");
}

typedef struct { int64_t magic, d, N, qn, qd, cm, cs, D, p, g, reduced, n, tgt, T, tsw; } header;
#define MAGIC 0x3832315046LL   /* "FP128" */

static void wr(const void *p, size_t sz, size_t cnt, FILE *f) { if (fwrite(p, sz, cnt, f) != cnt) die("write error"); }
static void rd(void *p, size_t sz, size_t cnt, FILE *f) { if (fread(p, sz, cnt, f) != cnt) die("read error"); }

int main(int argc, char **argv)
{
    if (argc < 7) die("usage: fp128 d N qn qd red out.bin [budget_seconds]");
    d = atoi(argv[1]); N = atoi(argv[2]); qn = atoi(argv[3]); qd = atoi(argv[4]); reduced = atoi(argv[5]);
    const char *out = argv[6];
    double budget = argc > 7 ? atof(argv[7]) : 1e30;
    if (d < 1 || d > MAXD || N < 2 || qn < 1 || qn > qd) die("bad parameters");
    int g = igcd(igcd(qn, 2 * d * (qd - qn)), 2 * d * qd);
    cm = qn / g; cs = 2 * d * (qd - qn) / g; D = 2 * d * qd / g;
    if (D > 15 || 2 * d * cm + cs != D) die("D > 15 is not supported");
    build();

    u128 *L = calloc(n, sizeof(u128)), *U = calloc(n, sizeof(u128)), *Ln = calloc(n, sizeof(u128)), *Un = calloc(n, sizeof(u128));
    flo *lo = calloc(n, sizeof(flo)), *hi = calloc(n, sizeof(flo)), *lon = calloc(n, sizeof(flo)), *hin = calloc(n, sizeof(flo));
    if (!L || !U || !Ln || !Un || !lo || !hi || !lon || !hin) die("calloc");

    long t = 1, tsw = 0; int factive; flo slo, shi; double used = 0;
    char ck[4096], tmp[4096];
    snprintf(ck, sizeof ck, "%s.ckpt", out); snprintf(tmp, sizeof tmp, "%s.ckpt.tmp", out);
    FILE *f = fopen(ck, "rb");
    if (f) {                                                    /* resume */
        int64_t h[8];
        rd(h, sizeof(int64_t), 8, f);
        if (h[0] != MAGIC || h[1] != d || h[2] != N || h[3] != qn || h[4] != qd || h[5] != reduced) die("checkpoint mismatch");
        t = h[6]; factive = (int)(h[7] & 1); tsw = h[7] >> 1;
        rd(&bad, sizeof(long), 1, f); rd(&used, sizeof(double), 1, f);
        grow(t + 1);
        rd(FLr, sizeof(u128), t + 1, f); rd(FUr, sizeof(u128), t + 1, f); rd(relf, 1, t + 1, f);
        rd(L, sizeof(u128), n, f); rd(U, sizeof(u128), n, f);
        if (factive) { rd(lo, sizeof(flo), n, f); rd(hi, sizeof(flo), n, f); rd(&slo, sizeof(flo), 1, f); rd(&shi, sizeof(flo), 1, f); }
        fclose(f);
    } else {
        L[0] = U[0] = (u128)1 << P;                             /* v_1 = e_x0 (x0 has stored index 0) */
        lo[0].m = hi[0].m = (u64)1 << 62; lo[0].e = hi[0].e = -62;
        grow(1);
        fsum_fixed(L, U, &FLr[1], &FUr[1]);
        factive = (FLr[1] >> G) == 0;
        if (!factive) tsw = 1;
        sigma_float(lo, hi, &slo, &shi);
    }
    time_t start = time(NULL);
    long T = -1;
    for (;;) {
        grow(t + 1);
        step_fixed(L, U, Ln, Un);                               /* time t -> t+1 */
        fsum_fixed(Ln, Un, &FLr[t + 1], &FUr[t + 1]);
        relf[t] = '?';
        if (factive) {
            flo slon, shin;
            step_float(lo, hi, lon, hin);
            sigma_float(lon, hin, &slon, &shin);
            if (flt(shi, slon)) relf[t] = '<';
            else if (flt(shin, slo)) relf[t] = '>';
            else if (shi.m == 0 && shin.m == 0) relf[t] = '=';
            slo = slon; shi = shin;
            flo *x = lo; lo = lon; lon = x; x = hi; hi = hin; hin = x;
            if ((FLr[t + 1] >> G) != 0) { factive = 0; tsw = t + 1; }
        }
        if (tail_ok(L, Un)) { T = t; break; }
        u128 *y = L; L = Ln; Ln = y; y = U; U = Un; Un = y;
        t++;
        if ((t & 63) == 0 && difftime(time(NULL), start) > budget) {
            FILE *o = fopen(tmp, "wb");
            if (!o) die("cannot write checkpoint");
            int64_t h[8] = {MAGIC, d, N, qn, qd, reduced, t, (int64_t)factive | ((int64_t)tsw << 1)};
            used += difftime(time(NULL), start);
            wr(h, sizeof(int64_t), 8, o); wr(&bad, sizeof(long), 1, o); wr(&used, sizeof(double), 1, o);
            wr(FLr, sizeof(u128), t + 1, o); wr(FUr, sizeof(u128), t + 1, o); wr(relf, 1, t + 1, o);
            wr(L, sizeof(u128), n, o); wr(U, sizeof(u128), n, o);
            if (factive) { wr(lo, sizeof(flo), n, o); wr(hi, sizeof(flo), n, o); wr(&slo, sizeof(flo), 1, o); wr(&shi, sizeof(flo), 1, o); }
            if (fclose(o)) die("checkpoint close");
            if (rename(tmp, ck)) die("checkpoint rename");
            printf("CHECKPOINT d=%d N=%d q=%d/%d t=%ld seconds_so_far=%.0f\n", d, N, qn, qd, t, used);
            return 75;
        }
    }
    used += difftime(time(NULL), start);
    FILE *o = fopen(out, "wb");
    if (!o) die("cannot write output");
    header h = {MAGIC, d, N, qn, qd, cm, cs, D, P, G, reduced, n, tgt, T, tsw};
    wr(&h, sizeof h, 1, o);
    wr(FLr + 1, sizeof(u128), T + 1, o); wr(FUr + 1, sizeof(u128), T + 1, o); wr(relf + 1, 1, T, o);
    wr(L, sizeof(u128), n, o); wr(Un, sizeof(u128), n, o);
    if (fclose(o)) die("output close");
    remove(ck);
    printf("DONE d=%d N=%d q=%d/%d reduced=%d sites=%ld T=%ld tsw=%ld seconds=%.0f\n", d, N, qn, qd, reduced, n, T, tsw, used);
    return 0;
}
