# How the results were checked

This note describes how the numbers, formulas and proofs of the article and its Supplementary Material were
checked. All checks were made within this project and on the same machine: they are checks by separately written
code and by line-by-line reading, not peer review and not a replication by a third party. The article states every
result with the status that survived these checks, never a stronger one.

## Current submission verification

The reproducible checks performed on 2 October 2026 are recorded in `submission/SCIENTIFIC-CHECKS.md`. In particular, the all-q certificate was regenerated rather than resumed: its 171 certified d = 2 blocks and 387 certified d = 3 blocks agree with the released data, excluding execution times. The earlier checks described below remain provenance records, not additional computations claimed for this submission run.

## 1. Every displayed formula is evaluated

The scripts of `code/article/` evaluate every displayed formula of the article and of the Supplementary Material in
the form in which it is printed (zero-based sites), against exact rational solves of the defining linear system
built site by site from the walker rule, and against double-precision sparse solves for larger lattices:

| script | what it checks | outcome |
|---|---|---|
| `s2_model_checks.py` | Section 2: exact structure, transform identity, interlacing, pseudo-inverse formula, two spectra | all identities to round-off |
| `appA_proofs_check.py` | Supplementary Section S3: every identity of the proofs of the mean formulas | 51 checks, 0 failed |
| `appB_expansion_check.py` | Supplementary Section S4: every identity, bound and coefficient of the proof of the large-N expansion | 38 checks, 0 failed |
| `s4_mfpt_checks.py`, `s4_mfpt_literature_check.py` | Section 4 against exact solves; the published resistance formulas of Tan and Tan (2020) evaluated as printed | agreement to 10⁻³⁸ or better |
| `s3_methods_certificate.py`, `s3_methods_certificate_roundoff.py`, `s3_methods_log1p_demo.py` | Section 3 and Supplementary Section S2: the certificate of the global mode for all 193 runs of time stepping, the a-priori round-off bound for the 96 runs not certified in exact arithmetic, and the log1p pitfall of route B | 193 of 193 certified; 96 of 96 within the round-off bound |
| `s5_modes_tables.py`, `s6_mechanism_tables.py`, `closed_form_checks.py` | Sections 5–6: the closed-form PMF of the chain, the modes re-stepped with a separately written stepper, the closed form of the mode | identical modes |
| `s6b_rigorous_checks.py` | Section 7 against Sections 3–6: all 97 corner-to-corner discrete-time modes at q ∈ {0.5, 0.8, 1} against the certified modes (the 16 runs at q = 0.3 and 0.9 lie in no certified range and rest on the round-off bound); the one-dimensional windows; the corollary on the closed form recomputed in interval arithmetic | 97 of 97 agree |
| `s7_defects_density_*.py`, `s8_defects_placement_checks.py`, `derived_checks.py` | Section 8 and Supplementary Section S9: every derived number of the defect sections from the stored per-placement results | — |
| `appC_tables_build.py --check` | the rows of the extended tables are verbatim copies of the result files | — |

## 2. Second implementations of the three parts of the analysis

The analysis has three parts: exact distributions and modes (`code/msc_modes/`), inert defects
(`code/msc_defects/`), and the mean first-passage time formulas with their large-N expansion (`code/msc_proof/`).
Each part was checked by a second implementation (`code/msc_modes_verify/`, `code/msc_defects_verify/`,
`code/msc_proof_verify/`), written separately from the model definition, sharing no code with the first one and
using its own random seeds where seeds are used.

* **Distributions and modes.** A compiled time stepper, a pole expansion (root finding on the resolvent sum) in
  place of Talbot inversion, sparse solves for the means, 30-digit arithmetic for the constants. In 148 exact
  discrete cases computed by both implementations no integer mode differs, and 64 continuous-time modes agree to
  7 × 10⁻¹³ (d = 2, up to N = 16 777 216) and 1.1 × 10⁻¹⁰ (d = 3, up to N = 4096). Fitted exponents, extrapolation
  errors, shape statistics and the constants of the exit geometry were reproduced to the printed digits.
  *Not repeated:* the scan of 186 continuous-time densities for a single maximum and the Erlang-mixture check of the
  inversion; the corner-to-centre and centre-to-corner placements were checked in discrete time only.
* **Inert defects.** The exact machinery was written again from the model definition (sparse solves for the means;
  the mode by time stepping with the Cauchy–Schwarz certificate and, as a second route, by a dense
  eigendecomposition; a Monte Carlo simulation of the walker rule). 400 placements of the density sweep, regenerated
  from their seeds, the clean lattice and all 28 deterministic geometries give identical modes and means that agree to about 10⁻¹¹. The
  rest of the density sweep was checked against a second sample of 10 400 placements drawn with different seeds,
  which agrees within sampling error (largest standardised difference 2.1 in 78 comparisons); the article pools the
  two samples.
* **Mean formulas.** The walk was built again from its rules, with a separate exact rational solver and separate code
  for every closed form. The single sum, its overflow-free form, both half-range sums and the double sum agree with
  exact rational solves for N = 2, …, 44; the large-N coefficients were recovered by a linear solve that uses no
  closed form; the constants printed by Essam and Wu (2009) and Izmailian and Huang (2010) were reproduced.
  *Limit:* the exact rational solves stop at N = 44.

The Monte Carlo simulation of the literal walker rule (4 × 10⁶ walkers at N = 35, Supplementary Section S2)
confirms the exact distribution without any linear algebra.

## 3. The proofs about the mode

The complete proofs of Sections 4 and 7 of the article are in the companion notes `proofs/R0`–`R5`. Several are
computer-assisted, with three trusted bases: (T1) Python integers and fractions; (T3) Arb ball arithmetic through
python-flint 0.9.0, together with the transcription of the formulas into the scripts; (T5) a C program with 64- and
128-bit integer arithmetic, whose output is turned into verdicts with Python integers. Floating-point arithmetic is
used only to find candidates, never to decide.

* **Two checks of each note.** Each of R1–R5 was examined in two ways: a line-by-line reading of every proof, with
  the displayed identities re-derived, every citation checked and the certificate scripts re-run on a separate copy;
  and a numerical search for counter-examples, with separately written code that tested every statement with its
  explicit constants (for example about 1.1 million numerical checks of R2 and 5363 certificate comparisons of R5
  with separately written engines; a second implementation of the Taylor-model certificate of Lemma 9.10 of R4
  certifies all 26 boxes, `data/checks/r4_family_second_implementation.json`). No statement labelled theorem was
  found false. The code of these two checks (the line-by-line reports and the counter-example searches) is not part
  of this repository, so these checks cannot be re-run from it; the certificates and the programs that check them
  are in the repository.
* **Re-run.** Every sanity script of the five notes (52 runs) was run again serially; all ended with exit status 0
  and every printed verdict reports success. Of the 339 data files of the notes, 333 were byte-identical before and
  after; six differ in the last digits of floating-point eigen-decompositions (four), in timing fields (one), or by
  added rows with the same verdicts (one).
* **Final reading.** R0 was read once more in full. The proofs written after the two checks and the deductions that
  combine two notes (identified by provenance markers in companion note R0) were read line by line and their numerical content
  was re-checked with separately written code; so were the main certificates (exact modes and unimodality for small
  N, the closed-form windows against the certified modes, the constant c and the spectral constants) and three
  lemmas chosen at random (this code is not part of the repository either). No statement was found false and no
  status had to be lowered. The counts behind the exact-mode formula of the chain that R0 quotes (the hypothesis of
  the formula holds for 1495 of 1499 sizes at q = 4/5 and for 989 of 998 at q = 1/2, and the formula then gives the
  certified mode) are recomputed from the released certificates by `code/article/checks_1d_exact_formula.py`
  (`data/checks/r0_1d_exact_formula_counts.json`).
* **Certificates regenerated for this repository.** In a fresh copy of the repository every resumable certificate
  was moved aside and regenerated: the window certificates of R3 (220, 389, 217, 386, 220 and 389 records, identical
  to the stored ones), the tables of R3, the Python-integer checker of the exact modes for d = 2, N ≤ 60, d = 3,
  N ≤ 20 and d = 1, N ≤ 300 (59 of 59, 19 of 19 and 299 of 299 certificates), the constants of R1 and R2, and the
  audits of R5.
* **What has not been checked.** The certificate of the discrete-time window for every q < 1 was regenerated in the current submission verification. Its implementation has not been independently re-implemented. For 2242 of the 3824 certificates of the C program, that program, run twice with identical
  checksums, is the only witness. No program has been verified in a proof assistant.

## 4. The published tree

The repository itself was tested in its published layout: a copy was made and the scripts listed in `reproduce.md`,
Section 7, were run in it; every file they rewrote was compared with the published one (byte-identical, or for JSON
files equal within a relative tolerance of 10⁻⁸ once run times are ignored; regenerated PDF figures differ only in
the creation date). When the repository was assembled it was also checked that every `% src:` path of the LaTeX
sources (article, Supplementary Material and companion notes) exists, that every
path expression of every script points to an existing location, that every script compiles, and that no file
contains a local path.
