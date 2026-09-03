# Experiment Record

## Reproducibility Check — 2026-09-03

Purpose:
Confirm that `experiments.compare_stability` produces deterministic results
for seeds 0–9 before modifying stability metrics.

Command:

powershell
python -m experiments.compare_stability

Configuration:

- Seeds: 0–9
- Epsilon: 0.3
- Episodes per seed: 5000
- Baseline and HOPE both tested
- Python: 3.11
- NumPy: 2.4.0
- Gymnasium: <version>

Results:

    Run 1 saved as baseline_repro_2026-09-03_run1.csv
    Run 2 saved as baseline_repro_2026-09-03_run2.csv
    Run 2 matched Run 1 exactly.

CSV files:

- summary_20260903-102204.csv
- summary_20260903-102825.csv

Conclusion:
Current experiment is reproducible under the recorded environment and seeds.
