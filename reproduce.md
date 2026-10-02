# How to reproduce every figure and table

All commands are run from the root of this repository. Scripts locate their inputs and outputs relative to their
own position, so the working directory does not matter, and no script needs an argument unless one is shown.

```
python --version                 # 3.10 or later
pip install -r requirements.txt  # numpy, scipy, mpmath, sympy, pandas, matplotlib, python-flint
```

The results in `data/` were produced with Python 3.14.6, numpy 2.5.3, scipy 1.18.1, mpmath 1.3.0, sympy 1.14.0,
pandas 3.0.6, matplotlib 3.11.2 and TeX Live 2025 on a laptop. Times below are wall-clock times on that machine.

There are four levels of reproduction, from cheapest to most complete.

1. **Build the PDFs** from the LaTeX sources and the stored figures (Section 1).
2. **Rebuild every figure and table from the stored results, and re-check every displayed formula**
   (Sections 2–4; minutes; the certificate of the global modes, Section 4, takes about 20 minutes more).
3. **Regenerate the stored results themselves** from nothing (Section 5; a few hours on four to five cores).
4. **Re-verify the certificates of the proofs** of Section 7 of the article (Section 8; a few minutes for the
   quick checks, hours for the longest certificates).

What is random: the Monte Carlo checks and the placements of defects, all with fixed seeds, so that every run gives
the same numbers. Everything else is deterministic. Regenerated JSON files agree with the stored ones to round-off
(wall-clock fields excepted); regenerated PDF figures differ from the stored ones only in metadata.

---

## 1. The article and its Supplementary Material

```
sh build_pdfs.sh                 # -> main.pdf (the same document as paper.pdf) and supplement.pdf
```

The two documents cite each other through the package `xr-hyper`; each reads the labels of the other from a copy of
its `.aux` file in `xref/`, which `build_pdfs.sh` refreshes (two rounds). `xref/` is part of the repository, so a single
`latexmk -pdf main.tex` also resolves every reference to the Supplementary Material. Standard `article` class;
packages amsmath, amssymb, amsthm, graphicx, booktabs, array, longtable, placeins, microtype, xcolor, natbib,
xr-hyper, hyperref, lmodern, geometry. The build has no error and no undefined reference or citation.

## 2. Figures

All figures are written straight into `figures/` by scripts under `code/article/`, at the text width of the article
and in its notation. Figures 1–4 are in the article, Figures S1–S17 in the Supplementary Material.

| Fig. | File in `figures/` | Command | Reads |
|---|---|---|---|
| 1, S6, S8 | `s5_modes_mode_mean.pdf`, `s5_modes_ratio.pdf`, `s5_modes_exponents.pdf` | `python code/article/s5_modes_figures.py` | `data/msc_modes/discrete_modes.jsonl`, `laplace_modes.jsonl`, `fit_results.json` |
| 2, S2, S7, S9, S10 | `s6_mechanism_density.pdf`, `s3_methods_parity.pdf`, `s5_modes_compensated.pdf`, `s6_mechanism_accuracy.pdf`, `s6_mechanism_geometries.pdf` | `python code/article/s3s6_modes_figures.py` | `data/msc_modes/discrete_modes.jsonl`, `laplace_modes.jsonl`, `fit_results.json`, `profiles.npz`, `profile_stats.json`, `pmf/` |
| 3, S12 | `s7_defects_density_sweep.pdf`, `s7_defects_density_size.pdf` | `python code/article/s7_defects_density_pooled.py` then `python code/article/s7_defects_density_figures.py` | `data/msc_defects/sweep/`, `data/msc_defects_verify/v03_sweep/`, `v07_summary.csv`, `v07b_summary.csv`, `v04b_theta.json`, `data/msc_defects/homogenization.json` |
| 4, S11, S14, S15, S16 | `s8_defects_placement_structured.pdf`, `s7_defects_density_pmf.pdf`, `s8_defects_placement_maps.pdf`, `s8_defects_placement_twotime.pdf`, `s8_defects_placement_shape.pdf` | `python code/article/s7s8_defects_figures.py` (about 20 s; recomputes one uniform placement and the spectrum of the clean 35 × 35 lattice) | `data/msc_defects/structured_pmfs.npz`, `structured_masks.npz`, `structured_deterministic.csv`, `sensitivity_N35.csv`, `summary_structured_local.csv`, `structured_local.csv`, `sweep/`, `size_clean.csv`, `beyond_1d_random.csv`; `data/msc_defects_verify/v09b_twotime.csv`; `data/article/s8_defects_placement_checks.json` |
| S1 | `s3_methods_validation.pdf` | `python code/article/s3_methods_validation.py` (about 4 min on four cores: 4 × 10⁶ simulated walkers; `--reuse-mc` loads the stored arrival times instead) | computes everything itself; `data/msc_modes/` for the bookkeeping part |
| S3, S4, S5 | `s4_mfpt_verification.pdf`, `s4_mfpt_landscape.pdf`, `s4_mfpt_asymptotics.pdf` | `python code/article/s4_mfpt_figures.py` | `data/msc_proof/verify_float.csv`, `asymptotics.csv`, `data/msc_proof_verify/v5_extra.json` |
| S13, S17 | `s8_defects_placement_mechanism.pdf`, `s8_defects_placement_beyond.pdf` | `python code/article/s8_defects_placement_figures.py` | `data/msc_defects/sweep/`, `summary_sweep_N35.csv`, `summary_beyond.json`, `beyond_1d_single.csv` |

All figure scripts except those of Figs S1 and 4, S11, S14–S16 take a few seconds. The figure scripts of the three
analysis modules (`code/msc_modes/05_make_figures.py`, `code/msc_defects/09_figures.py`,
`code/msc_*_verify/*figures.py`) write the working figures of the analysis to `out/figures/<module>/`; the article
does not use them.

## 3. Tables

The rows of the tables are generated by scripts and copied into the LaTeX sources; for the extended tables of
Supplementary Section S10 the copy is verified automatically.

| Table | Content | Command | Output read by the article |
|---|---|---|---|
| 3, 8, S1 | laws; status of the results; notation | — (no data) | — |
| 1, S3, S4 | coefficients of the expansion; exact means; truncation errors | `python code/article/s4_mfpt_checks.py` (re-computation, 6 s) | `data/article/s4_mfpt_checks.json`; `data/msc_proof/verify_exact.csv`, `asymptotic_series.json`, `asymptotic_truncation_errors.csv`; `data/msc_proof_verify/v3_asymptotics.json` |
| 2, S5 | exact modes; scaling fits | `python code/article/s5_modes_tables.py` (about 1 min). Without an option the re-stepping with a second time stepper of the two largest cases (d = 2, N = 401 and d = 3, N = 100) is copied from the stored output; `--big` re-steps them as well (about 9 minutes more) | `data/article/s5_modes_tables.json` (`tab_modes_rows`, `tab_fits`, `stepper_recheck`); `data/msc_modes/tables.md`; certificate of the unmarked rows: `data/article/s3_methods_certificate.jsonl` |
| 4, 5, S6 | accuracy of the closed-form approximation of the mode; shape statistics; other placements | `python code/article/s6_mechanism_tables.py` (10 s) | `data/article/s6_mechanism_rows.tex`, `s6_mechanism_tables.json` |
| 6 | constants of the windows of the closed form (Section 7) | — (copied from the certificate tables) | `data/msc_rigorous_asymptotics/80_tables.json`; regenerated by Section 8 below |
| 7, S11, S12 | localised placements; first-order slopes; deterministic geometries | `python code/article/s8_defects_placement_checks.py` (1 s) | `data/article/s8_defects_placement_checks.json`; `data/msc_defects/summary_first_order.json`, `summary_structured_local.csv`, `structured_deterministic.csv` |
| S2 | cross-checks of the numerical routes | `python code/article/s3_methods_validation.py` (new checks); `python code/article/s3_methods_certificate.py --summary` (certificate row; see Section 4) | `data/article/s3_methods_validation.json`, `s3_methods_certificate.json`; `data/msc_modes/validation.json`, `fit_results.json`; `data/msc_modes_verify/validate.json`, `analysis.json`; `data/msc_proof_verify/v2_identities.json` |
| S7, S8, S9, S10 | validation of the defect computations; pooled sweep; homogenised coefficients; size dependence | `python code/article/s7_defects_density_pooled.py` (2 s) | `data/article/s7_defects_density_pooled.json`, `s7_defects_density_tables.txt`, `s7_defects_density_pooled.csv`; `data/msc_defects/validation.json`; `data/msc_defects_verify/v01_validation.json` |
| S13–S19 | extended tables (Supplementary Section S10) | `python code/article/appC_tables_build.py` (writes the rows), `python code/article/appC_tables_build.py --check` (verifies that the rows in `sections/appC_tables.tex` are verbatim copies) | `data/article/appC_tables_fragments/*.tex`, `appC_tables_checks.json` |

The per-sample tables of the density sweep (sample A: `data/msc_defects/tables/sweep_uniform.md`, `sweep_smart.md`;
sample B: `data/msc_defects_verify/v03_summary.csv`) are not printed; LaTeX rows for sample A are kept in
`data/article/appC_tables_fragments/sweep_*.tex`. In the file names of the analysis, `smart` (sample A) and
`cornerthinned` (sample B) denote the corner-thinned placement law of the article.

## 4. Checks of the mathematics and of the numerical statements

Every displayed formula of the article and of the Supplementary Material is evaluated numerically in the form in which
it is printed.

```
python code/article/s2_model_checks.py              # Section 2: structure, two spectra, pseudo-inverse     (8 s)
python code/article/closed_form_checks.py                  # closed-form approximation against exact modes; zero-based formulas (1 s)
python code/article/s4_mfpt_checks.py               # Section 4 against exact rational solves                (6 s)
python code/article/s4_mfpt_literature_check.py     # Eqs (57), (64) of Tan and Tan (2020) against Theorems 4.2, 4.3 (a few s)
python code/article/appA_proofs_check.py            # Supplementary Section S3: 51 checks                    (8 s)
python code/article/appB_expansion_check.py         # Supplementary Section S4: 38 checks, incl. every inequality used (3 s)
python code/article/s7_defects_density_dipole_check.py   # Proposition 8.2                                   (1 s)
python code/article/s7_defects_density_second_order.py   # second-order coefficient of the dilute expansion (5 s)
python code/article/derived_checks.py                    # derived numbers of Sections 3, 5, 8 and S2, S9         (2 s)
python code/article/s6b_rigorous_checks.py          # Section 7 and the computer-assisted parts of Section 4 against Sections 3-6 (1 s)
python code/article/s3_methods_inversion_other_sizes.py  # inversion of a generating function at three sizes, Section S2.5 (a few s)
python code/article/s3_methods_log1p_demo.py        # log1p(-mu) against log(1 - mu) in route B, Section S2.1 (a few s per size)
python code/article/reference_checks.py              # looks up every DOI of refs.bib on Crossref (needs network)
```

Each prints its checks and writes `data/article/<name>.json`; `n_failed` must be 0 where the file has that key.

Certificate that the discrete-time modes are global maximisers (Proposition 3.1; Supplementary Section S2.3):

```
python code/article/s3_methods_certificate.py small     # runs with N^d * t_end < 2e9             (about 2 min)
python code/article/s3_methods_certificate.py mid       #                                          (about 5 min)
python code/article/s3_methods_certificate.py big       # d = 2, N >= 241 and d = 3, N >= 60       (about 13 min)
python code/article/s3_methods_certificate.py --summary # -> data/article/s3_methods_certificate.json
python code/article/s3_methods_certificate_roundoff.py  # a-priori round-off bound for the 96 runs not certified in exact
                                                        # arithmetic -> data/article/s3_methods_certificate_roundoff.json (about 1 min)

cc -O3 -o s3_methods_certificate_chain code/article/s3_methods_certificate_chain.c -lm
./s3_methods_certificate_chain 10240 0.8 43679969 data/article/s3_methods_certificate_chain_N10240_probes.txt \
    > data/article/s3_methods_certificate_chain_N10240.json      # chain, N = 10 240: 6.1e7 steps, 5 to 8 min
python code/article/s3_methods_certificate_chain_check.py        # -> data/article/s3_methods_certificate_chain.json (30 s)
```

`s3_methods_certificate.py` is restartable: it skips the runs already present in
`data/article/s3_methods_certificate.jsonl`, so delete that file to recompute all 193 runs. Memory stays below
200 MB. The chain at N = 10 240 has three parts (Supplementary Section S2.3): the compiled stepper covers the times
beyond t_cert and those outside a window of ± 20 000 steps around the mode, where its margins exceed an a-priori bound
on its round-off; `s3_methods_certificate_chain_check.py` evaluates the closed form in 60-digit arithmetic at every
time of the window, checks the two margins against the round-off bound, and compares the stepped values written to the
probe file with the closed form. The check script alone can be re-run from the stored stepper output.

## 5. Regenerating the stored results

Each module has a first implementation (`code/<module>/`) and a second implementation (`code/<module>_verify/`),
written separately within this project, that shares no code with it and was used to check it. The run scripts of
`msc_modes` are restartable: they skip cases already present in the `.jsonl` files, so delete
`data/msc_modes/discrete_modes.jsonl` and `laplace_modes.jsonl` (and the corresponding files of the other modules)
to recompute from nothing. Scripts that take a number of worker processes as their last argument are shown with the
value used.

### 5.1 Mean first-passage time formulas (`msc_proof`) — about half an hour

```
python code/msc_proof/check_identities.py           # every intermediate identity            -> data/msc_proof/identity_checks.json
python code/msc_proof/verify_mfpt.py                # exact rational solves N = 2..40, float64 tests (about 12 min)
                                                    #                                       -> verify_exact.csv, verify_float.csv, verify_summary.json
python code/msc_proof/general_pairs.py              # arbitrary start and target (about 10 min) -> general_pairs*.{csv,json}
python code/msc_proof/asymptotics.py                # 50-digit values up to N = 524288       -> asymptotics.csv, asymptotics_summary.json
python code/msc_proof/asymptotic_series.py          # closed-form coefficients (sympy)       -> asymptotic_series.json
python code/msc_proof/make_figures.py               # -> out/figures/msc_proof/, asymptotic_truncation_errors.csv, landscape_check.json

python code/msc_proof_verify/v1_exact.py            # second exact solver, N <= 22
python code/msc_proof_verify/v1b_exact_large.py     # N up to 44
python code/msc_proof_verify/v2_identities.py
python code/msc_proof_verify/v3_asymptotics.py      # blind recovery of the coefficients from 70-digit values
python code/msc_proof_verify/v5_extra.py
python code/msc_proof_verify/v6_figures.py
```

### 5.2 Exact distributions and modes (`msc_modes`) — about an hour

```
python code/msc_modes/01_validate.py                                   # cross-validation        -> data/msc_modes/validation.json
python code/msc_modes/02_run_discrete.py 1d 2d_cc 2d_other qscan exit 2d_cc_big 3d_cc 3d_other 3d_cc_big
                                                                       # time stepping           -> discrete_modes.jsonl, pmf/*.npz
python code/msc_modes/03_run_laplace.py 1d 2d_cc 2d_c2m 2d_m2c 3d_cc 3d_c2m 3d_m2c 2d_cc_huge extra_pts
                                                                       # Laplace inversion, poles -> laplace_modes.jsonl
                                                                       # (group 1d also gives the discrete modes of the
                                                                       #  chain up to N = 1e6 from the closed form)
python code/msc_modes/03b_profiles.py
python code/msc_modes/03c_medians.py
python code/msc_modes/06_full_tail_unimodality.py                      # full-support runs       -> full_tail.json
python code/msc_modes/04_fit_scaling.py                                # fits, constants         -> fit_results.json, tables.md
python code/msc_modes/05_make_figures.py                               # -> out/figures/msc_modes/
python code/msc_modes/07_summary_tables.py                             # -> summary_tables.md
```

Largest jobs: time stepping for d = 3, N = 100 (10⁶ sites, about 5 minutes, under 100 MB) and Laplace inversion for
d = 2, N = 16 777 216 (about 3 minutes, about 1 GB). No random numbers.

Check by the second implementation (a compiled time stepper and a pole expansion):

```
cc -O3 -o code/msc_modes_verify/stepper code/msc_modes_verify/stepper.c -lm
python code/msc_modes_verify/v01_discrete.py small 1d 2d qscan geom exit full 3d 2d_big 3d_big n40_long
python code/msc_modes_verify/v02_validate.py
python code/msc_modes_verify/v03_constants.py
python code/msc_modes_verify/v04_poles.py farcheck 2d 3d conv
python code/msc_modes_verify/v05_misc.py
python code/msc_modes_verify/v06_analysis.py
python code/msc_modes_verify/v07_extra.py
python code/msc_modes_verify/v08_1d_bigN_mp.py       # local maximality of the modes at N = 1e5 and 1e6 in 50-digit arithmetic
python code/msc_modes_verify/v10_tables.py
python code/msc_modes_verify/v09_figures.py
python code/article/checks_fits_recompare.py         # fits of both implementations compared -> data/checks/modes_scaling_fits_recheck.json
```

### 5.3 Inert defects (`msc_defects`) — about an hour on five worker processes

```
python code/msc_defects/00_validate.py                    # -> data/msc_defects/validation.json
python code/msc_defects/02_density_sweep.py 35 400 5      # N, placements per density, workers        -> sweep/*.csv
python code/msc_defects/03_homogenization.py              # conductivity on tori                      -> homogenization.json
python code/msc_defects/04_structured.py 5                # localised and deterministic placements
python code/msc_defects/05_size_dependence.py 5           # about 20 minutes
python code/msc_defects/06_sensitivity.py 35 5            # single-defect maps; also with 15 and 25 in place of 35
python code/msc_defects/07_beyond_inert.py 5              # barriers, permeable obstacles, traps
python code/msc_defects/08_analysis.py                    # summaries                                 -> summary_*.{csv,json}
python code/msc_defects/09_figures.py                     # -> out/figures/msc_defects/
python code/msc_defects/10_tables.py                      # -> tables/*.md
```

Check by the second implementation (separately written code, different seeds; about an hour on three to four workers):

```
python code/msc_defects_verify/v01_clean_validation.py
python code/msc_defects_verify/v03_sweep.py 400 3
python code/msc_defects_verify/v04_homog.py
python code/msc_defects_verify/v04b_theta_precise.py
python code/msc_defects_verify/v03b_analyse.py
python code/msc_defects_verify/v03c_perconfig.py
python code/msc_defects_verify/v05_sensitivity.py 15 3    # and 25 3, 35 3
python code/msc_defects_verify/v06_structured.py
python code/msc_defects_verify/v07_size.py
python code/msc_defects_verify/v07c_more.py 70 0.1 400 4  # N, p, placements, workers; repeated for the sizes and densities of Table S10
python code/msc_defects_verify/v07b_mfpt_largeN.py 2
python code/msc_defects_verify/v07d_analyse.py
python code/msc_defects_verify/v08_beyond.py
python code/msc_defects_verify/v09_mechanism.py
python code/msc_defects_verify/v09b_twotime.py
python code/msc_defects_verify/v11_shape_test.py
python code/msc_defects_verify/v10_figures.py
```

### 5.4 Article-level scripts — after 5.1–5.3

```
python code/article/closed_form_checks.py
python code/article/s2_model_checks.py
python code/article/s3_methods_validation.py
python code/article/s3_methods_certificate.py            # all groups, about 20 min; then the compiled chain run of Section 4
python code/article/s3_methods_certificate_chain_check.py
python code/article/s3_methods_certificate_roundoff.py   # about 1 min
python code/article/s3_methods_log1p_demo.py
python code/article/s3_methods_inversion_other_sizes.py
python code/article/s4_mfpt_checks.py
python code/article/s4_mfpt_literature_check.py
python code/article/s4_mfpt_figures.py
python code/article/s5_modes_tables.py --big
python code/article/s5_modes_figures.py
python code/article/s6_mechanism_tables.py
python code/article/s3s6_modes_figures.py
python code/article/s7_defects_density_pooled.py
python code/article/s7_defects_density_dipole_check.py
python code/article/s7_defects_density_second_order.py
python code/article/s7_defects_density_figures.py
python code/article/s8_defects_placement_checks.py
python code/article/s8_defects_placement_figures.py
python code/article/s7s8_defects_figures.py
python code/article/derived_checks.py
python code/article/appA_proofs_check.py
python code/article/appB_expansion_check.py
python code/article/appC_tables_build.py --check
sh build_pdfs.sh
```

## 6. Where each number comes from

Every quantitative statement in `main.tex`, `supplement.tex` and `sections/*.tex` is followed by a comment
`% src: <path> (<key>)` naming the file (and, where useful, the key or table inside it) that contains the number, with
paths relative to the repository root. `data/checks/…` are small files that collect numbers of the checks with their
provenance. A few comments name a published article and the equation that was evaluated.

## 7. What was re-run in the layout of this repository

The published tree itself was tested: a copy of this repository, laid out exactly as published, was made, and 63
scripts were run in it, one at a time — the Python scripts of `code/article/` except `reference_checks.py` (which
queries the Crossref service over the network) and `s3_methods_validation.py` (Monte Carlo and inversion runs of
several minutes), with `s3_methods_certificate.py` in its `--summary` mode; the validation, summary, table and figure
scripts of `code/msc_modes/`, `code/msc_defects/` and `code/msc_proof/` that work from the stored results (among
them `01_validate.py`, `04_fit_scaling.py`, `05_make_figures.py`, `03_homogenization.py`, `08_analysis.py`,
`09_figures.py`, `10_tables.py`, `check_identities.py`, `make_figures.py`); the corresponding scripts of the three
`code/*_verify/` directories (among them `v02_validate.py`, `v06_analysis.py`, `v03b_analyse.py`,
`v03c_perconfig.py`, `v05_sensitivity.py 15 2`, `v09_mechanism.py`, `v1_exact.py`, `v2_identities.py`,
`v3_asymptotics.py`); and the scripts of the proofs listed in Section 8 that regenerate their output in place.
62 ran without error straight away; `v02_validate.py` needs the compiled time stepper and ran without error
once it had been compiled with the command of Section 5. Every file the scripts rewrote was then compared with the
published one: 1088 files are byte-identical, 15 JSON files agree within a relative tolerance of 10⁻⁸ once run
times are ignored, and the 20 regenerated figures of `figures/` differ only in the creation date and document
identifier embedded in the PDF. In addition, one file of the second density sweep
(`data/msc_defects_verify/v03_sweep/N35_cornerthinned_p0.10.csv`, 400 placements) was deleted in the copy and
regenerated with `PGRID=0.10 python code/msc_defects_verify/v03_sweep.py 400 2 cornerthinned`; the regenerated file is
byte-identical to the published one. The long computations of Section 5 (time stepping, Laplace inversion, the
density sweeps, the 50-digit sums of `asymptotics.py` and the exact rational solves of `verify_mfpt.py`) were not
repeated in this test; they write the stored files that the scripts above read.
Nineteen further scripts (among them
`code/article/s3_methods_certificate_roundoff.py`, `code/article/s3_methods_log1p_demo.py`,
`code/msc_rigorous_spectral/20_certify_rounded_constants.py`, `code/msc_rigorous_certified_c128/audit_ext.py`,
`code/msc_rigorous_asymptotics/83_compare_cert_runs.py` and `code/msc_modes_verify/v08_1d_bigN_mp.py`) were run
in a second fresh copy: all ran without error, every file they rewrote is byte-identical to the published one
except three JSON files that agree within 10⁻⁸ once run times are ignored, and five regenerated figures that differ
only in the creation date and document identifier.
Seven further scripts (`code/article/checks_1d_exact_formula.py`, `code/msc_modes/07_summary_tables.py`,
`code/msc_rigorous_certified/check_packed.py`, and `verify_certificate_code.py`, `verify_lemmas.py`,
`check_lc_threshold.py` and `table_windows.py` of `code/msc_rigorous_unimodality/`) were run in a third fresh copy:
all ran without error, and every file they rewrote is byte-identical to the published one except
`data/msc_rigorous_unimodality/table_windows.json`, whose floating-point eigenvalues agree within 10⁻⁸.

## 8. Re-verifying the proofs

Section 7 of the article states theorems on the first-passage law and its mode; their complete proofs are in
`proofs/` (the companion notes R0–R5, PDF and LaTeX source), and the short ones also in Supplementary Section S8. The
computer-assisted proofs rest on certificates in `data/msc_rigorous_*`, written and checked by the scripts in
`code/msc_rigorous_*`, with three trusted bases: (T1) Python integers and fractions; (T3) Arb ball arithmetic through
python-flint 0.9.0, together with the transcription of the formulas into the scripts; (T5), for the extended ranges of
exact modes only, the C program `code/msc_rigorous_certified_c128/fp128.c` (Apple clang 21, arm64, `-O2`), whose
output is turned into verdicts with Python integers. The theorem statements of the article name the base of each
proof.

Several certificate scripts are **resumable**: they skip the entries already present in their output file. To
re-verify such a certificate, move the stored file aside, run the script, and compare the regenerated file with the
stored one (the run-time fields excepted). Some commands below regenerate only part of a stored file (the range of N
given on the command line): compare the regenerated records with the matching records of the stored file, then put
the stored file back. `cert_family.py` appends to its output file, so move
`data/msc_rigorous_unimodality/cert_family_F1.jsonl` (20 records) and `cert_family_F2.jsonl` (6 records) aside before
running it. The commands below were run in this way in a fresh copy of the repository, one at a time, on a loaded
laptop (times are wall-clock); every regenerated record equals the corresponding stored record.

### 8.1 Quick re-verification

```
python code/article/s6b_rigorous_checks.py                 # Section 7 against Sections 3-6; Cor. 7.22 recomputed          (1 s)
python code/msc_rigorous_1d/03_constant_enclosure.py        # the constant c to 70 digits; alpha, beta (kappa, rho of R1)    (<1 s)
python code/msc_rigorous_1d/39_hand_constants.py            # the constants derived by hand in R1                           (2 s)
python code/msc_rigorous_spectral/run_all.py                # all 21 scripts of R2, its certificates included                (80 s)
python code/msc_rigorous_asymptotics/50_cert_constants.py   # lattice constants of R3 (Prop. 4.5, 4.6, Cor. 4.7 of R3)       (17 s)

# windows of the closed form (Thms 7.18 and 7.20); move data/msc_rigorous_asymptotics/<output>.jsonl aside first
python code/msc_rigorous_asymptotics/52_cert_theoremB.py 2 10      # continuous time, d = 2: 220 blocks   (1 s; d = 3: 389 blocks, 18 s)
python code/msc_rigorous_asymptotics/72_cert_discrete.py 2 10      # discrete time, q <= 1/2              (1 s; d = 3: 18 s)
python code/msc_rigorous_asymptotics/76_cert_q_large.py 2 0.8 10   # q <= 0.8 (likewise 0.9, 0.95, 0.99)  (2 s; d = 3: 20 s)
python code/msc_rigorous_asymptotics/78_cert_allq.py 2             # every q < 1                          (2 s; d = 3: 20 s)
python code/msc_rigorous_asymptotics/80_tables.py                  # Table 6 of the article and the full tables of R3   (1 s)

python code/msc_rigorous_unimodality/cert_family.py F1      # Taylor-model certificate of R4, Lemma 9.10 (11 s; F2: 4 s); move cert_family_F*.jsonl aside first
python code/msc_rigorous_unimodality/summarize_lc1d.py      # completeness of the ranges behind Thm 7.7     (<1 s)
python code/msc_rigorous_certified/audit.py                 # every number of the theorems of R5 against the certificates
python code/msc_rigorous_certified_c128/audit_ext.py        # the same for the extended ranges              (3 s)
python code/article/checks_1d_exact_formula.py              # counts behind t* = ceil(tau) in one dimension (R0, Section 4)
                                                            #   -> data/checks/r0_1d_exact_formula_counts.json (a few s)

# exact modes (Thm 7.15): the Python-integer checker re-derives the certificates for the range of N given; move
# data/msc_rigorous_certified/packed_d*_q4-5.jsonl aside first (stored: N = 2-200 for d = 2, 2-60 for d = 3,
# 2-1000 for d = 1), compare the regenerated records with the stored ones for that range, then restore the stored file
python code/msc_rigorous_certified/verify_packed.py data/msc_rigorous_certified/modes_d2_q4-5.jsonl 2 60   # N = 2-60, 59 records (50 s)
python code/msc_rigorous_certified/verify_packed.py data/msc_rigorous_certified/modes_d3_q4-5.jsonl 2 20   # N = 2-20, 19 records (11 s)
python code/msc_rigorous_certified/verify_packed.py data/msc_rigorous_certified/modes_d1_q4-5.jsonl 2 300  # N = 2-300, 299 records (110 s)
```

### 8.2 The long computations (recorded times)

These were not repeated for this repository; the times are those recorded when the certificates were made (R0,
Section 9).

* All Python-integer mode certificates (1539, over the Python-integer ranges of R5): the checker
  `code/msc_rigorous_certified/verify_packed.py`, longest single certificate 1794 s (d = 3, N = 60). The certificates
  themselves are produced by `python code/msc_rigorous_certified/run_modes.py d qn qd Nmin Nmax limb` (restartable;
  repeat until the range is complete).
* The extended ranges (3824 certificates): compile the C program with
  `cc -O2 -o out/work/msc_rigorous_certified/fp128 code/msc_rigorous_certified_c128/fp128.c`, then
  `python code/msc_rigorous_certified_c128/c128_run.py d qn qd NSPEC workers`; 28 623 s for the first pass and
  30 080 s for the second, longest certificate 1501 s. Use at most two workers on a laptop.
* Unimodality: `code/msc_rigorous_unimodality/cert_lc1d.py` (log-concavity of the one-dimensional free increment,
  about 26 CPU-minutes for each of the two families; the commands are listed in R4, Appendix A), `cert_thr1d.py`
  (thresholds in one dimension: 1.84 h with python-flint for N ≤ 300, 0.44 h with Python integers for N ≤ 200,
  0.57 h for the centre geometry), `cert_unimodal.py` (finite ranges: 88 and 20 CPU-minutes).
* One dimension: the window certificates of R1 (scripts 12, 13, 14, 15, 18, 32, 35 of `code/msc_rigorous_1d/`) take
  seconds to minutes each (script 14: 4–6 s per configuration).

### 8.3 The companion notes

```
cd proofs/R0_rigorous_results && latexmk -pdf R0_rigorous_results.tex     # likewise R1 ... R5
```

Run logs other than those in `data/*/logs/` and `data/checks/`, and working notes, are not part of the repository.
