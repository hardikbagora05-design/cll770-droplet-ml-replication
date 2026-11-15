#!/usr/bin/env bash
# Reproduces every number and figure in the report (~12 min on a 2-core CPU)
set -e
cd "$(dirname "$0")"
python src/01_replicate.py
python src/02_mass_conservation.py
python src/03_model_comparison.py
python src/04_uncertainty.py
python src/05_importance_learning_curve.py
python src/06_figures.py
python src/07_tables.py
python src/08_equations.py
echo "Done. See results/ and figures/"
