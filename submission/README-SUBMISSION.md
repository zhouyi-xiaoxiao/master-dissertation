# Scientific verification and reproducibility materials

The manuscript and Supplementary Material are ../paper.pdf and ../supplement.pdf. The full source, figures, data and companion proof notes remain in the repository layout.

- SCIENTIFIC-CHECKS.md: executed checks, numerical scope, and remaining mathematical limitations.
- run_checks.sh: repeat the scientific checks in a disposable source copy; select an interpreter with the PYTHON environment variable.
- pdf_audit.py: check cross-document PDF links and bibliography keys, using Python with pypdf.
- verification/: path-normalized copies of execution records; see its README for provenance.
- ../build_pdfs.sh: build both PDFs and synchronize paper.pdf with main.pdf.
- ../reproduce.md: full scientific workflow and trusted arithmetic bases.

From the repository root, run sh submission/run_checks.sh, or python submission/pdf_audit.py . for the PDF audit. Required dependencies are described in requirements.txt and reproduce.md. No private filesystem path is needed.

These public files document scientific verification. Editorial covering letters and journal-specific declarations are not part of this public verification bundle. No journal submission is asserted.

## Mathematical scope

For the discrete-time window, q < 1 and N >= max(N*, N1(q)), where N* is 59 in two dimensions and 12 in three dimensions, and sin(pi/N) <= (1-q)/q remain essential. The corresponding q = 1 statement remains open. Finite-range claims are not extended to all sizes. The 2D all-size 1% closed-form bound and large-system defect homogenisation remain conjectural. Certificates retain their Python-integer, Arb and C-arithmetic trusted bases.
