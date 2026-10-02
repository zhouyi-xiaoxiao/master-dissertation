#!/bin/sh
# Run in a disposable copy: resumable outputs must be preserved then moved aside.
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
backup="out/submission-prerun-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup"
for f in data/msc_rigorous_asymptotics/78_cert_allq_d2.jsonl data/msc_rigorous_asymptotics/78_cert_allq_d3.jsonl data/msc_rigorous_unimodality/cert_family_F1.jsonl data/msc_rigorous_unimodality/cert_family_F2.jsonl; do
 if [ -f "$f" ]; then cp "$f" "$backup/$(basename "$f")"; rm "$f"; fi
done
"$PYTHON" code/article/checks_1d_exact_formula.py
"$PYTHON" code/msc_rigorous_1d/03_constant_enclosure.py
"$PYTHON" code/msc_rigorous_1d/39_hand_constants.py
"$PYTHON" code/msc_rigorous_spectral/run_all.py
"$PYTHON" code/msc_rigorous_asymptotics/78_cert_allq.py 2
"$PYTHON" code/msc_rigorous_asymptotics/78_cert_allq.py 3
"$PYTHON" code/msc_rigorous_asymptotics/80_tables.py
"$PYTHON" code/msc_rigorous_unimodality/cert_family.py F1
"$PYTHON" code/msc_rigorous_unimodality/cert_family.py F2
"$PYTHON" code/msc_rigorous_certified/audit.py
"$PYTHON" code/msc_rigorous_certified_c128/audit_ext.py
"$PYTHON" code/article/appA_proofs_check.py
"$PYTHON" code/article/appB_expansion_check.py
"$PYTHON" code/article/appC_tables_build.py --check
"$PYTHON" code/article/s6b_rigorous_checks.py
