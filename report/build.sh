#!/usr/bin/env bash
# report_src.md (+ results/summary.json) -> report.md -> report.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
python3 report/build.py
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex --resource-path=.:..
echo "wrote report/report.pdf"
