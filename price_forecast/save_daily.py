#!/usr/bin/env python3
"""Copy one run's outputs into daily_forecasts/<date>/ and daily_forecasts/latest/, and append
the forecasts to daily_forecasts/history.csv (one row per market x horizon per run).

    python save_daily.py OUT_DIR DEST_ROOT [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

FILES = ["oil_gas_price_forecast.xlsx", "forecast.csv", "report.html"]
HISTORY_COLS = ["run_date", "as_of", "market", "horizon_calendar_days", "target_date", "last_price",
                "forecast_price", "p10", "p90", "p2_5", "p97_5", "prob_up_%", "backtest_MAPE_%", "model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("dest_root")
    ap.add_argument("--date", default=pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d"))
    args = ap.parse_args()
    out, root = Path(args.out_dir), Path(args.dest_root)
    for folder in (root / args.date, root / "latest"):
        folder.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            if (out / name).exists():
                shutil.copy2(out / name, folder / name)

    fc = pd.read_csv(out / "forecast.csv")
    fc.insert(0, "run_date", args.date)
    fc = fc[[c for c in HISTORY_COLS if c in fc.columns]]
    hist_path = root / "history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = hist[hist["run_date"] != args.date]  # a re-run on the same day replaces that day
        fc = pd.concat([hist, fc], ignore_index=True)
    fc.to_csv(hist_path, index=False)
    print(f"saved {args.date}: {len(fc)} rows in {hist_path}")


if __name__ == "__main__":
    main()
