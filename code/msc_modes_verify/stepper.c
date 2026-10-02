/*
 * Second implementation (check): exact first-passage PMF of the lazy nearest-neighbour
 * walk on {0..N-1}^d with cancelled-move ("stay") reflection and absorbing sites.
 *
 *   rho_{t+1}(n) = (1-q) rho_t(n) + (q/2d) * sum_{2d directions} rho_t(n') ,
 *   n' = neighbour in that direction, or n itself when the neighbour is off-lattice.
 *   f(t+1) = total mass that lands on absorbing sites at step t+1 (then removed).
 *
 * This is the transpose-free form of rho_{t+1} = Q rho_t because Q is symmetric.
 *
 * usage: stepper d N q geom stop_factor tmin tmax outfile
 *   geom: CC  (start corner 0, target opposite corner)
 *         C2M (start corner 0, target centre; N odd)
 *         M2C (start centre, target corner N-1)
 *         EXIT(start centre, all boundary sites absorbing; N odd)
 *   stop_factor > 0 : stop when t >= max(tmin, stop_factor * running_argmax)
 *   stop_factor <= 0: stop when survival < 1e-13 (full support)
 *   tmax: hard cap on the number of steps
 * output: binary float64 array f[1..T] in outfile; one text line on stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int d, N;
static long nsite;
static double q;
static double *rho, *nw;
static unsigned char *absorb;

static inline long idx3(int i, int j, int k) { return ((long)i * N + j) * N + k; }

static void step1(void) {
    double c = q / 2.0, s = 1.0 - q;
    for (int i = 0; i < N; i++) {
        double l = rho[i > 0 ? i - 1 : i], r = rho[i < N - 1 ? i + 1 : i];
        nw[i] = s * rho[i] + c * (l + r);
    }
}
static void step2(void) {
    double c = q / 4.0, s = 1.0 - q;
    for (int i = 0; i < N; i++) {
        const double *row = rho + (long)i * N;
        const double *up = rho + (long)(i > 0 ? i - 1 : i) * N;
        const double *dn = rho + (long)(i < N - 1 ? i + 1 : i) * N;
        double *out = nw + (long)i * N;
        out[0] = s * row[0] + c * (up[0] + dn[0] + row[0] + row[1]);
        for (int j = 1; j < N - 1; j++)
            out[j] = s * row[j] + c * (up[j] + dn[j] + row[j - 1] + row[j + 1]);
        out[N - 1] = s * row[N - 1] + c * (up[N - 1] + dn[N - 1] + row[N - 2] + row[N - 1]);
    }
}
static void step3(void) {
    double c = q / 6.0, s = 1.0 - q;
    for (int i = 0; i < N; i++) {
        int im = i > 0 ? i - 1 : i, ip = i < N - 1 ? i + 1 : i;
        for (int j = 0; j < N; j++) {
            int jm = j > 0 ? j - 1 : j, jp = j < N - 1 ? j + 1 : j;
            const double *row = rho + idx3(i, j, 0);
            const double *a1 = rho + idx3(im, j, 0), *a2 = rho + idx3(ip, j, 0);
            const double *b1 = rho + idx3(i, jm, 0), *b2 = rho + idx3(i, jp, 0);
            double *out = nw + idx3(i, j, 0);
            out[0] = s * row[0] + c * (a1[0] + a2[0] + b1[0] + b2[0] + row[0] + row[1]);
            for (int k = 1; k < N - 1; k++)
                out[k] = s * row[k] + c * (a1[k] + a2[k] + b1[k] + b2[k] + row[k - 1] + row[k + 1]);
            out[N - 1] = s * row[N - 1] + c * (a1[N - 1] + a2[N - 1] + b1[N - 1] + b2[N - 1] + row[N - 2] + row[N - 1]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 9) { fprintf(stderr, "usage\n"); return 2; }
    d = atoi(argv[1]); N = atoi(argv[2]); q = atof(argv[3]);
    const char *geom = argv[4];
    double stop_factor = atof(argv[5]);
    long tmin = atol(argv[6]), tmax = atol(argv[7]);
    const char *outfile = argv[8];
    nsite = 1; for (int a = 0; a < d; a++) nsite *= N;
    rho = calloc(nsite, sizeof(double)); nw = calloc(nsite, sizeof(double));
    absorb = calloc(nsite, 1);
    long corner0 = 0, cornerN = nsite - 1, centre = 0;
    { long m = 1; for (int a = 0; a < d; a++) { centre += m * ((N - 1) / 2); m *= N; } }
    long start;
    if (!strcmp(geom, "CC")) { start = corner0; absorb[cornerN] = 1; }
    else if (!strcmp(geom, "C2M")) { start = corner0; absorb[centre] = 1; }
    else if (!strcmp(geom, "M2C")) { start = centre; absorb[cornerN] = 1; }
    else if (!strcmp(geom, "EXIT")) {
        start = centre;
        for (long s = 0; s < nsite; s++) {
            long r = s; int onb = 0;
            for (int a = 0; a < d; a++) { int c = r % N; r /= N; if (c == 0 || c == N - 1) onb = 1; }
            absorb[s] = onb;
        }
    } else { fprintf(stderr, "bad geom\n"); return 2; }
    /* list of absorbing sites */
    long nabs = 0; for (long s = 0; s < nsite; s++) nabs += absorb[s];
    long *alist = malloc(nabs * sizeof(long));
    { long c = 0; for (long s = 0; s < nsite; s++) if (absorb[s]) alist[c++] = s; }
    rho[start] = 1.0;
    long cap = 1 << 20; double *f = malloc(cap * sizeof(double));
    double cum = 0.0, comp = 0.0; /* Kahan */
    double fmax = 0.0; long amax = 0; long t = 0;
    while (t < tmax) {
        if (d == 1) step1(); else if (d == 2) step2(); else step3();
        double ft = 0.0;
        for (long c = 0; c < nabs; c++) { ft += nw[alist[c]]; nw[alist[c]] = 0.0; }
        double *tmp = rho; rho = nw; nw = tmp;
        t++;
        if (t >= cap) { cap *= 2; f = realloc(f, cap * sizeof(double)); }
        f[t - 1] = ft;
        double y = ft - comp, tt = cum + y; comp = (tt - cum) - y; cum = tt;
        if (ft > fmax) { fmax = ft; amax = t; }
        if (stop_factor > 0) { if (amax > 0 && t >= tmin && (double)t >= stop_factor * (double)amax) break; }
        else if ((t & 255) == 0 && t >= tmin) {   /* full support: stop on the remaining transient mass */
            double sr = 0.0; for (long s2 = 0; s2 < nsite; s2++) sr += rho[s2];
            if (sr < 1e-13) break;
        }
    }
    /* survival from remaining mass (independent of cum) */
    double srem = 0.0; for (long s = 0; s < nsite; s++) srem += rho[s];
    FILE *fp = fopen(outfile, "wb"); fwrite(f, sizeof(double), t, fp); fclose(fp);
    printf("{\"d\":%d,\"N\":%d,\"q\":%.17g,\"geom\":\"%s\",\"T\":%ld,\"argmax\":%ld,\"fmax\":%.17g,\"cum\":%.17g,\"S_remaining\":%.17g}\n",
           d, N, q, geom, t, amax, fmax, cum, srem);
    return 0;
}
