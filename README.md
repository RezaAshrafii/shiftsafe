# ShiftSafe

ShiftSafe is a small, evidence-first project for reliable sensor and time-series AI under distribution shift.

The project starts with a deterministic quality gate. It checks whether a dataset is structurally ready for the next experiment and produces a machine-readable summary plus a short Markdown report.

## What exists now

- Explicit `DataContract` input model.
- Missing-column, missing-cell, duplicate-row and target checks.
- Timestamp parseability and monotonic-order checks.
- `CONTINUE`, `REWORK`, or `STOP` decision.
- Leakage-aware temporal and whole-group train/test splits.
- Python API and a small CSV CLI.
- Unit tests.

برای توضیح ساده فازها:

- [توضیح فارسی فاز ۱: قرارداد و گیت کیفیت](docs/PHASE_1_FA.md)
- [توضیح فارسی فاز ۲: تفکیک بدون نشت داده](docs/PHASE_2_FA.md)
- [توضیح فارسی فاز ۳: baseline و معیارهای پایه](docs/PHASE_3_FA.md)

This first release does **not** claim model safety, legal compliance, causal impact, fairness certification, ROI, or production readiness.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Run the CLI:

```powershell
shiftsafe data/example.csv --target target --time-column timestamp `
  --json-out reports/run.json --markdown-out reports/run.md
```

## Research direction

The research question is:

> Can uncertainty-calibrated predictions remain useful for decisions when sensor or time-series data shifts across time, sites, devices, or operating regimes?

Planned milestones are conformal prediction, prediction intervals, calibration diagnostics, drift stress tests, abstention, and decision-cost evaluation.

## Product direction

The first commercial offer is a limited, local-first validation review for one dataset and one AI pilot. The repository is intentionally small so a solo researcher can demonstrate the method before building infrastructure.

## License

MIT. See `LICENSE`.
