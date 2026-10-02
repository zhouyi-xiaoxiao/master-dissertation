/* s3_methods_certificate_chain.c -- certificate of the global mode for the one-dimensional chain at a size
 * beyond the time-stepping runs of route A (Supplementary Section S2.3, Proposition s3_methods:prop-certificate(b)).
 *
 * Chain {0,...,N-1}, reflecting end 0 (cancelled move), target N-1, activity q.  Q is the transition matrix
 * with the target removed (sites 0..N-2), r = (q/2) e_{N-2}.  With u_k = Q^k e_0 and v_k = Q^k r,
 *
 *      f(2k+1) = u_k . v_k ,     f(2k+2) = u_{k+1} . v_k        (sums of non-negative terms),
 *      f(t')  <= ||u_k|| ||v_k||   for every t' >= 2k+1         (Cauchy-Schwarz; ||Q||_2 <= 1).
 *
 * The program steps until ||u_k|| ||v_k|| < (1 - MARGIN) max_{t <= 2k} f(t) (tested every CHK values of k) and
 * prints one JSON record: the smallest maximiser on {1,...,t_cert-1}, the number of local maxima and of exact
 * ties there, whether f decreases monotonically from the mode to t_cert, the relative gap between the maximum
 * and the larger of its two neighbours, and a round-off estimate (f(2k+1) evaluated a second time as
 * u_{k+1} . v_{k-1} for the odd t within WIN steps of the expected mode).
 *
 * At large N the maximum is so flat that neighbouring values of f differ by less than the round-off of the
 * stepping (N = 10240: true relative gap 1.7e-16; accumulated round-off of the stepped values 1e-10).  The
 * stepper is therefore used only where its margins exceed its worst-case round-off: for t >= t_cert (bound
 * above; relative margin MARGIN) and for t < t_cert outside the stored window of +-STORE steps around the
 * expected mode, for which the program reports the largest f relative to the maximum (N = 10240: 1 - 1.6e-7).
 * Inside the window the order of the values is decided in multi-precision arithmetic from the closed form by
 * s3_methods_certificate_chain_check.py, which also contains the a-priori round-off bound (all operations
 * below act on non-negative numbers, so the relative error grows by at most about five units of round-off per
 * step).  The "plateau" (first and last t with f(t) >= (1 - TOL) max f, TOL = 1e-12) is reported for
 * information only.
 *
 * Build:  cc -O3 -o s3_methods_certificate_chain s3_methods_certificate_chain.c -lm
 * Usage:  ./s3_methods_certificate_chain N q expected_mode [probe-file]     e.g.  ... 10240 0.8 43679969
 *         With a fourth argument the computed f(t) is also written, as lines "t f(t)", for every t within
 *         PROBE_NEAR steps of the expected mode, every PROBE_STEP-th t of the stored window and every
 *         PROBE_FAR-th t of the whole run.  s3_methods_certificate_chain_check.py compares these values with
 *         the closed form in multi-precision arithmetic; this measures the accumulated round-off of the
 *         stepping and how much it varies with t.  The option changes no arithmetic and no other output.
 * Cost :  about t_cert * N floating-point updates (N = 10240: 6e11, a few minutes); memory 4 N doubles.
 * No random numbers.  No other code is used.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHK 16
#define MARGIN 1e-6
#define WIN 2000
#define STORE 20000
#define TOL 1e-12
#define PROBE_NEAR 60
#define PROBE_STEP 500
#define PROBE_FAR 4000000

static void applyQ(const double *restrict x, double *restrict out, long n, double q, long lo, long hi)
{
    /* out <- Q x on sites lo..hi (0 <= lo <= hi <= n-1); x vanishes outside [lo-1, hi+1] by construction */
    const double a = 1.0 - q, c = 0.5 * q;
    long j0 = lo < 1 ? 1 : lo, j1 = hi > n - 2 ? n - 2 : hi;
    if (lo == 0)
        out[0] = (1.0 - c) * x[0] + c * x[1];                /* reflecting end: the cancelled move stays */
    for (long j = j0; j <= j1; j++)
        out[j] = a * x[j] + c * (x[j - 1] + x[j + 1]);
    if (hi == n - 1)
        out[n - 1] = a * x[n - 1] + c * x[n - 2];            /* site N-2: the move to the target is absorbed */
}

static double dot(const double *restrict x, const double *restrict y, long lo, long hi)
{
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    long j = lo;
    for (; j + 3 <= hi; j += 4) {
        s0 += x[j] * y[j];
        s1 += x[j + 1] * y[j + 1];
        s2 += x[j + 2] * y[j + 2];
        s3 += x[j + 3] * y[j + 3];
    }
    for (; j <= hi; j++)
        s0 += x[j] * y[j];
    return (s0 + s1) + (s2 + s3);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s N q expected_mode [probe-file]\n", argv[0]);
        return 2;
    }
    FILE *probe = argc > 4 ? fopen(argv[4], "w") : NULL;
    if (argc > 4 && !probe)
        return 4;
    long N = atol(argv[1]);
    double q = atof(argv[2]);
    long expect = atol(argv[3]);
    long n = N - 1;                                           /* number of non-target sites */
    double *u = calloc(n + 2, sizeof(double)), *un = calloc(n + 2, sizeof(double));
    double *v = calloc(n + 2, sizeof(double)), *vn = calloc(n + 2, sizeof(double)), *vp = calloc(n + 2, sizeof(double));
    if (!u || !un || !v || !vn || !vp)
        return 3;
    clock_t c0 = clock();
    u[0] = 1.0;
    v[n - 1] = 0.5 * q;
    double fmax = -1.0, fprev = 0.0, fprev2 = 0.0, fbefore = 0.0, fafter = -1.0;
    long tmode = 0, nties = 0, nlocmax = 0, t = 0, tcert = 0, k = 0;
    int monotone = 1, rising = 1, have_after = 0;
    double noise = 0.0, bound = 0.0, fmax_outside = -1.0;
    double *fs = calloc(2 * STORE + 1, sizeof(double));       /* f(t) for |t - expect| <= STORE */
    if (!fs)
        return 3;
    /* f(t) for t = 1, 2, ...; bookkeeping of local maxima: t-1 is a local maximum if f(t-2) < f(t-1) > f(t) */
    for (k = 0;; k++) {
        long ulo = 0, uhi = k < n - 1 ? k : n - 1;            /* support of u_k */
        long vlo = n - 1 - k > 0 ? n - 1 - k : 0;            /* support of v_k: vlo .. n-1 */
        for (int half = 0; half < 2; half++) {
            double f;
            if (half == 0) {                                   /* t = 2k+1 */
                f = (vlo <= uhi) ? dot(u, v, vlo, uhi) : 0.0;
            } else {                                           /* t = 2k+2: advance u, then u_{k+1} . v_k */
                long nhi = k + 1 < n - 1 ? k + 1 : n - 1;
                applyQ(u, un, n, q, ulo, nhi);
                double *tmp = u; u = un; un = tmp;
                uhi = nhi;
                f = (vlo <= uhi) ? dot(u, v, vlo, uhi) : 0.0;
                /* second evaluation of f(2k+1) = u_{k+1} . v_{k-1} near the expected mode */
                if (k >= 1 && labs(2 * k + 1 - expect) <= WIN) {
                    long plo = n - k > 0 ? n - k : 0;         /* support of v_{k-1} */
                    double f2 = (plo <= uhi) ? dot(u, vp, plo, n - 1) : 0.0;
                    if (fprev > 0) {
                        double d = fabs(f2 - fprev) / fprev;
                        if (d > noise) noise = d;
                    }
                }
            }
            t++;
            if (probe && (labs(t - expect) <= PROBE_NEAR || t % PROBE_FAR == 0
                          || (labs(t - expect) <= STORE && (t - expect) % PROBE_STEP == 0)))
                fprintf(probe, "%ld %.17g\n", t, f);
            if (labs(t - expect) <= STORE)
                fs[t - expect + STORE] = f;
            else if (f > fmax_outside)
                fmax_outside = f;
            if (f > fmax) {
                fmax = f; tmode = t; fbefore = fprev; have_after = 0;
            } else {
                if (f == fmax && f > 0) nties++;
                if (t == tmode + 1) { fafter = f; have_after = 1; }
            }
            if (t >= 3 && fprev > fprev2 && fprev > f) nlocmax++;
            fprev2 = fprev; fprev = f;
        }
        /* advance v:  v_{k+1} = Q v_k  (keep v_k as "previous" for the round-off estimate) */
        {
            long nlo = n - 2 - k > 0 ? n - 2 - k : 0;
            applyQ(v, vn, n, q, nlo, n - 1);
            double *tmp = vp; vp = v; v = vn; vn = tmp;
        }
        if ((k + 1) % CHK == 0) {
            long kk = k + 1;                                   /* u, v now hold u_{kk}, v_{kk} */
            long lo2 = n - 1 - kk > 0 ? n - 1 - kk : 0, hi2 = kk < n - 1 ? kk : n - 1;
            double nu = sqrt(dot(u, u, 0, hi2)), nv = sqrt(dot(v, v, lo2, n - 1));
            bound = nu * nv;
            if (bound < (1.0 - MARGIN) * fmax) { tcert = 2 * kk + 1; break; }
        }
    }
    /* monotone decrease from the mode to t_cert: exactly one local maximum and it is the mode */
    monotone = (nlocmax == 1);
    (void)rising;
    double gap = -1.0;
    if (have_after) {
        double nb = fbefore > fafter ? fbefore : fafter;
        gap = (fmax - nb) / fmax;
    }
    /* plateau: stored t with f(t) >= (1 - TOL) fmax */
    long plo = 0, phi = 0, pn = 0;
    for (long i = 0; i <= 2 * STORE; i++) {
        long tt = expect - STORE + i;
        if (tt < 1 || tt >= tcert) continue;
        if (fs[i] >= (1.0 - TOL) * fmax) {
            if (pn == 0) plo = tt;
            phi = tt; pn++;
        }
    }
    int plateau_inside = (pn > 0 && plo > expect - STORE && phi < expect + STORE && phi - plo + 1 == pn);
    int outside_below = (fmax_outside < (1.0 - TOL) * fmax);
    double wall = (double)(clock() - c0) / CLOCKS_PER_SEC;
    printf("{\"d\": 1, \"geo\": \"CC\", \"N\": %ld, \"q\": %.17g, \"mode\": %ld, \"f_max\": %.17g, "
           "\"t_cert\": %ld, \"t_cert_over_mode\": %.12g, \"bound_at_t_cert\": %.17g, \"margin_rel\": %.6g, "
           "\"certified\": true, \"n_local_maxima_below_t_cert\": %ld, \"n_ties_exact\": %ld, "
           "\"single_local_maximum_and_monotone_from_mode_to_t_cert\": %s, \"gap_rel_to_larger_neighbour\": %.6g, "
           "\"roundoff_rel_near_mode\": %.6g, \"mode_expected\": %ld, \"mode_equals_expected\": %s, "
           "\"plateau_tol\": %.3g, \"plateau_first_t\": %ld, \"plateau_last_t\": %ld, \"plateau_n\": %ld, "
           "\"plateau_contiguous_and_inside_stored_window\": %s, \"stored_window_half_width\": %d, "
           "\"f_max_outside_stored_window_over_f_max\": %.17g, \"all_t_outside_plateau_below_(1-tol)_f_max\": %s, "
           "\"steps_t\": %ld, \"cpu_s\": %.1f, \"generated_by\": \"code/article/s3_methods_certificate_chain.c\"}\n",
           N, q, tmode, fmax, tcert, (double)tcert / (double)tmode, bound, 1.0 - bound / fmax,
           nlocmax, nties, monotone ? "true" : "false", gap, noise, expect, tmode == expect ? "true" : "false",
           TOL, plo, phi, pn, plateau_inside ? "true" : "false", STORE,
           fmax_outside > 0 ? fmax_outside / fmax : 0.0,
           (plateau_inside && outside_below) ? "true" : "false",
           t, wall);
    if (probe)
        fclose(probe);
    free(u); free(un); free(v); free(vn); free(vp); free(fs);
    return 0;
}
