#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$DIR/output}"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo " Running ERG_Lopez_et_al Replication Pipeline..."
echo " Output directory: $OUTPUT_DIR"
echo "========================================================================"

echo "[1/2] Running automated statistical parity audit..."
python3 "$DIR/audit_parity.py" --out-json "$OUTPUT_DIR/audit_report.json"

echo "[2/2] Rendering Figures 4 & 11 (PNG 400 DPI, vector PDF, SVG)..."
python3 "$DIR/render_figures.py" --figures all --output-dir "$OUTPUT_DIR/figures"

echo "========================================================================"
echo " All steps finished successfully!"
echo " Audit Report: $OUTPUT_DIR/audit_report.json"
echo " Figures:      $OUTPUT_DIR/figures/"
echo "========================================================================"
