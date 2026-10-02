# Scientific checks for the submission package

Executed on 2 October 2026 using Python 3 with NumPy 2.5.3, SciPy 1.18.1 and python-flint 0.9.0. The machine-readable ledger and path-normalized output copies are in `verification/`; its README describes the normalization. Original literal logs remain in the private delivery evidence. Run commands from the source root; substitute a Python interpreter with these dependencies for the recorded local interpreter path.

## Executed checks

- R0: exact one-dimensional formula counts, recomputed from released certified modes.
- R1: constant enclosure and hand-constant calculations.
- R2: all 21 scripts in `code/msc_rigorous_spectral/run_all.py`; no script failures.
- R3: `78_cert_allq.py 2` and `78_cert_allq.py 3`, with resumable output files moved aside first. The regenerated files have 220 and 389 records; all 171 d = 2 blocks beginning at N >= 59 and all 387 d = 3 blocks beginning at N >= 12 certify, including terminal unbounded blocks. Every field agrees with the released data except elapsed execution time. The 49 and 2 lower-size records are outside those certified ranges, not failed claims of the theorem. R3 table generation also completed.
- R4: all 20 F1 and 6 F2 Taylor-model boxes regenerated and CERTIFIED. Every non-runtime field matches the released certificates.
- R5: standard and extended audits completed. A fresh Python-integer packed calculation for d = 3, q = 4/5, N = 2,...,20 produced 19 certified modes and unimodality verdicts, all equal to the released certificates. The source package retains the complete released N = 2,...,60 file, not only this regenerated subset. The standard audit was repeated after restoring the complete file.
- Article: Supplementary S3 and S4 formula checks, extended-table check and Section 7 consistency checks completed with exit status 0. The matching Section 7 check reports agreement for 97 certified corner-to-corner modes and all 36 tested one-dimensional modes inside their windows; the window determines 34 modes and agrees in all 34.

## PDF and reference checks

The final PDFs contain 40 main-text pages and 56 supplementary pages, all A4. Both final TeX logs have no unresolved references or citations and no overfull boxes. The PDF objects contain 44 links from paper.pdf to supplement.pdf and 292 links in the reverse direction; all 336 file targets and named destinations resolve. All 67 unique cited bibliography keys exist. `pdf_audit.py` repeats these checks using pypdf. The document contact sheets and representative full-resolution pages were inspected, including the q < 1 table, proof pages and reference lists; no clipping or overlapping text was observed.

## Interpretation and scope

The q < 1 result is a computer-assisted theorem under its stated conditions: N >= max(N*, N1(q)), with N* = 59 (d = 2) or 12 (d = 3), and sin(pi/N) <= (1-q)/q. Its regeneration verifies execution and consistency of the released ball-arithmetic implementation; it is not an independent reimplementation or a proof-assistant verification. Analytical reductions are given in the proof notes.

The provenance marks in R0 identify the development and combination of proofs, not a mathematical provisional status due merely to lack of external review. No conjecture is promoted by the computational checks. The q = 1 general mode laws, the stated uncovered finite-size ranges near q = 1, the proposed all-size two-dimensional relative-error bound, and defect homogenisation beyond the stated evidence remain open or conjectural as identified in the manuscript.

The full 3824-certificate C-engine computation was not regenerated in this run. Its released outputs were audited; 2242 certificates have only that engine as their computational witness, as recorded in R0. Historical additional checks are described in `notes/VERIFICATION.md`; their unreleased checking programs are not represented as newly reproduced evidence.

## Reproduction

Use `submission/run_checks.sh` in a disposable source copy for the current check suite (the optional packed N = 2,...,20 regeneration is recorded separately in the command ledger). Use `reproduce.md` for the full workflow and `build_pdfs.sh` for the interlinked main and supplementary PDFs. The latter requires a local TeX installation with latexmk; successful completion synchronizes `paper.pdf` with `main.pdf` and rejects unresolved references. No network service is required for the scientific checks. To regenerate a resumable certificate rather than reuse it, preserve and move aside its existing output file first. Recorded commands and path-normalized stdout/stderr copies are supplied in `verification/`.
