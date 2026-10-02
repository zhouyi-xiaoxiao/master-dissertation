#!/bin/sh
# build_pdfs.sh -- compile the article (main.tex -> main.pdf, published as paper.pdf) and its Supplementary Material
# (supplement.tex -> supplement.pdf).  The two documents cite each other through xr-hyper; each reads the labels of the
# other from a copy of its .aux file in xref/ (without the \bibcite lines, which belong to each document's own
# bibliography).  Two rounds make every cross-reference resolve.  Needs latexmk and pdflatex on PATH.
set -e
cd "$(dirname "$0")"
mkdir -p xref
for round in 1 2; do
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement.tex
  grep -v '^\\bibcite' supplement.aux > xref/supplement.aux
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
  grep -v '^\\bibcite' main.aux > xref/main.aux
done
for f in main supplement; do
  printf '%s: ' "$f"
  grep -E 'Output written' "$f.log" | sed 's/.*(\([0-9]*\) pages.*/\1 pages/' | tr -d '\n'
  printf ', undefined references or citations: %s\n' "$(grep -cE "(Reference|Citation) \`[^']*' on page [0-9]* undefined" "$f.log" || true)"
done
