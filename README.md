# Mean and most probable first-passage times of lazy random walks in reflecting hypercubic lattices

**Exact results, modal scaling and inert defects**

Author: Xiaoxiao Zhouyi, School of Engineering Mathematics and Technology, University of Bristol, UK
(zhouyixiaoxiao@gmail.com).

This repository contains the article (`paper.pdf`, 41 pages), its Supplementary Material (`supplement.pdf`,
56 pages), the LaTeX sources of both, the companion notes with the complete proofs of the computer-assisted
theorems (`proofs/`), and all code and data needed to re-run every number, table and figure.

## What the paper shows

The article studies the first-passage time of a lazy nearest-neighbour random walk on the reflecting lattice
{0, …, N−1}^d (d = 1, 2, 3) to a single absorbing site, and compares its mean T with its most probable value, the
mode t\*.

* **Means.** T = N(N−1)/q in one dimension; in two dimensions a complete proof of the single-sum formula for the
  corner-to-corner mean for every N ≥ 2, an equivalent form that does not overflow in floating point, a single sum
  for any start and target, the large-N expansion with closed-form coefficients and (computer-assisted) a remainder
  bound valid for every N; in three dimensions a double sum and, with computer assistance,
  q T = 4.32046… N³ − 3.887… N² + O(1), the leading constant being a sum of lattice Green's function values. Through
  the commute-time identity the two-dimensional formulas are resistances of the resistor grid, which are in print;
  Section 4.2 of the article says which formula is in which paper.
* **Modes, computed exactly.** First-passage distributions by time stepping and by Laplace inversion, cross-checked
  by a separately written pole expansion; every mode obtained by time stepping on the lattice without defects is
  certified to be the global maximiser of its distribution (in exact arithmetic for corner-to-corner transport at
  q ∈ {1/2, 4/5, 1}, otherwise with an a-priori bound on round-off). t\*/T → 0.33328 in one dimension; it falls slowly in two dimensions (0.27 at N = 5, 0.056 at
  N = 1.7 × 10⁷) and like ln N / N in three (0.0118 at N = 100). **The separation between mean and mode grows with
  dimension.** A closed form obtained by two-pole asymptotics, t\* ≈ T ln(2dX)/(X + 2d − 1) with X = μ₁qT,
  reproduces the exact continuous-time mode, and the exact discrete-time mode at q = 0.8, within 1 % (d = 2) and
  0.3 % (d = 3) at every computed size N ≥ 10. A result of companion note R5 (`proofs/R5_certified_computations/`, not
  stated in the article): for the certified discrete-time modes at q = 1 (10 ≤ N ≤ 160 and N = 180, 200 for d = 2,
  10 ≤ N ≤ 60 for d = 3) the error is at most 0.9712 % and 0.4534 %.
* **Theorems on the mode** (Section 7 of the article; complete proofs in Supplementary Section S8 and in `proofs/`).
  The first-passage law of the three point-target placements studied is unimodal in continuous time in every
  dimension, and in discrete time for q ≤ 4/5 from corner to corner (for the centre placements when d ≥ 2; a threshold
  attained on the chain with N = 4); unimodality fails near q = 1 on the chain, at q = 1 for some sizes in two
  dimensions, and for other placements. In one dimension t\*/T → 0.33328426549… for every activity q, with the mode located to
  within about one step for every N ≥ 3 when q ≤ 4/5. For corner-to-corner transport in two and three dimensions the
  scaled error of the closed
  form, X μ₁ (t\*_c − q t_cf), stays in an explicit bounded window for every N ≥ 12 (d = 2) and N ≥ 10 (d = 3), the
  leading laws t\* ∼ (4/π²q) N² ln ln N and (6/π²q) N² ln N hold with explicit remainders (in continuous time and, in
  discrete time, for q ≤ 0.99 once N exceeds an explicit size and for every q < 1 once N exceeds an
  explicit q-dependent size; q = 1 is open), and the closed form is within 0.971 % (d = 2) and 0.275 % (d = 3) of the
  certified exact mode on finite ranges with N ≥ 10 at q = 4/5 and 1/2, and
  within 0.3 % for every N ≥ 10 in three dimensions at these activities. Several of these proofs are computer-assisted: lemmas proved by hand reduce them to
  finitely many inequalities, checked in exact integer arithmetic or ball arithmetic; the trusted bases are stated
  and the certificates are in this repository. The proofs were checked within the project
  (`notes/VERIFICATION.md`), not by external referees.
* **Inert defects.** Each configuration of blocked sites is solved exactly. Uniformly random defects multiply mean
  and mode by nearly the same factor D₀/D_eff(p) at low density (N = 35, p ≤ 0.2), with D_eff/D₀ = 1 − (π − 1)p + O(p²)
  (a published law), so the ratio stays within 3.3 % of its clean value (0.177 at N = 35) up to p = 0.14; at p = 0.2
  the mean of the per-placement ratios is 8.9 % above it (the ratio of ensemble means 4.7 %);
  the rise shrinks with N but is not shown to vanish. Where the defects sit decides the direction of any change
  (forty blocked sites near the target: 0.075; near the start: 0.233).

## Contents

| Path | Content |
|---|---|
| `paper.pdf`, `supplement.pdf` | the article and its Supplementary Material (Sections S1–S10) |
| `main.tex`, `supplement.tex`, `macros.tex`, `sections/`, `figures/`, `refs.bib`, `xref/`, `build_pdfs.sh` | LaTeX sources; every number carries a comment `% src: <file>` naming the result file or script it comes from (paths relative to this directory); `xref/` holds the cross-reference labels that the two documents read from each other |
| `code/article/` | scripts that build the tables and figures of the article and re-check every displayed formula |
| `code/msc_modes/`, `code/msc_defects/`, `code/msc_proof/` | the three parts of the analysis: exact distributions and modes; inert defects; mean first-passage time formulas and their expansion |
| `code/*_verify/` | second implementations, written separately within this project, used to check each part |
| `code/msc_rigorous_*/` | scripts of the proofs about the mode: certificates (Python integers, Arb ball arithmetic, a C program), their checkers, sanity checks, and the floating-point exploration scripts (`explore*.py`) that note R4 cites or its sanity checks use; one folder per companion note (`1d`, `spectral`, `asymptotics`, `unimodality`, and `certified` with `certified_c128` for R5) |
| `proofs/` | the companion notes R0–R5 (PDF and LaTeX source): R0 a dependency-ordered account of all results on the mode, R1 one dimension, R2 spectral structure, R3 the mode in two and three dimensions, R4 unimodality, R5 certified computations |
| `data/<module>/` | the result files written by `code/<module>/` (the numbers of the article are read from these files); `data/msc_rigorous_certified/` holds the mode certificates (12 MB) |
| `data/checks/` | numbers of the checks that the article and the companion notes quote, collected in small files with their provenance |
| `notes/VERIFICATION.md` | how the results were checked |
| `reproduce.md` | exact commands for every figure and table, for the full pipeline and for the certificates |
| `out/` | empty folders into which the scripts write working figures, logs and scratch files that the article does not use |
| `LICENSE` | MIT (code), CC BY 4.0 (text, figures, data) |

## How to reproduce

Requirements: Python 3.10 or later with numpy, scipy, mpmath, sympy, pandas and matplotlib
(`pip install -r requirements.txt`; the versions used are listed in `reproduce.md`), python-flint 0.9.0 for the
ball-arithmetic certificates, a C compiler for three of the checks, and a TeX distribution with `latexmk` for the
PDFs.

```
# the two PDFs (main.pdf = paper.pdf, supplement.pdf)
sh build_pdfs.sh

# re-check every formula and rebuild the tables and figures from the stored results (minutes)
python code/article/appA_proofs_check.py
python code/article/appB_expansion_check.py
python code/article/s4_mfpt_checks.py
python code/article/appC_tables_build.py --check
python code/article/s5_modes_figures.py

# re-check Section 7 against the computations, and re-verify some certificates of the proofs (minutes)
python code/article/s6b_rigorous_checks.py
python code/msc_rigorous_1d/03_constant_enclosure.py
python code/msc_rigorous_certified/audit.py
```

`reproduce.md` gives the script behind each figure and table, the order and approximate cost of the full pipeline
(which regenerates the files in `data/` from nothing), the commands that re-verify the certificates of the proofs,
and what is random (fixed seeds) and what is deterministic. All computations run on a laptop; no job needs more than
about 1 GB of memory.

## Status of the results

The article distinguishes these kinds of statement, and so should any use of it: theorems and propositions
(complete proofs in the article, its Supplementary Material or the companion notes); theorems with computer
assistance (the trusted base and the certificates are named); exact numerics (valid at the sizes computed); formal
asymptotics (the closed form for placements other than corner to corner, and the dilute-defect law); conjectures
(among them the 1 % accuracy of the closed form in two dimensions for every N, and homogenisation at large N); and
refuted statements. Section 9 of the article lists the limitations.

## Citation

> X. Zhouyi, *Mean and most probable first-passage times of lazy random walks in reflecting hypercubic lattices:
> exact results, modal scaling and inert defects* (2026), https://github.com/zhouyi-xiaoxiao/master-dissertation

```bibtex
@misc{Zhouyi2026,
  author       = {Zhouyi, Xiaoxiao},
  title        = {Mean and most probable first-passage times of lazy random walks in reflecting
                  hypercubic lattices: exact results, modal scaling and inert defects},
  year         = {2026},
  howpublished = {\url{https://github.com/zhouyi-xiaoxiao/master-dissertation}}
}
```

## Licence

Code: MIT. Text, figures and data: CC BY 4.0. See `LICENSE`.
