#!/usr/bin/env python3
"""Forecast Brent, WTI and Henry Hub 7-15 days ahead and report out-of-sample accuracy.

Examples
    python run_forecast.py                              # live FRED data (+EIA if EIA_API_KEY set)
    python run_forecast.py --eia-key YOUR_KEY           # adds futures curve + inventories
    python run_forecast.py --source github              # EIA spot prices via GitHub mirror, no key
    python run_forecast.py --source auto                # live if reachable, newest prices from either
    python run_forecast.py --source csv --csv my.csv --markets brent ttf
    python run_forecast.py --source synthetic --quick   # offline pipeline test
"""

from __future__ import annotations

import argparse
import time

from ogforecast import data
from ogforecast.features import trading_days
from ogforecast.model import run_market
from ogforecast.report import write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "live", "github", "csv", "synthetic"], default="live")
    ap.add_argument("--csv", help="CSV file for --source csv")
    ap.add_argument("--eia-key", help="EIA API key (or set EIA_API_KEY)")
    ap.add_argument("--start", default="2005-01-01", help="first date of history to use")
    ap.add_argument("--markets", nargs="+", default=["brent", "wti", "henry_hub"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[7, 15],
                    help="forecast horizons in calendar days (default: 7 15)")
    ap.add_argument("--backtest-years", type=float, default=8)
    ap.add_argument("--refit-every", type=int, default=63, help="trading days between model refits")
    ap.add_argument("--quick", action="store_true", help="short backtest, fewer refits (for testing)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.quick:
        args.backtest_years, args.refit_every = 4, 126

    print(f"Loading data ({args.source}) ...")
    if args.source == "auto":
        df = data.load_auto(args.start, args.eia_key)
    elif args.source == "live":
        df = data.load_live(args.start, args.eia_key)
    elif args.source == "github":
        df = data.load_github(args.start)
    elif args.source == "csv":
        if not args.csv:
            ap.error("--source csv needs --csv PATH")
        df = data.load_csv(args.csv)
    else:
        df = data.load_synthetic(args.start)
    source_used = df.attrs.get("source", args.source)  # "auto" may fall back to one source
    df = df.loc[args.start:]
    print(f"  {len(df):,} business days, {df.index[0].date()} -> {df.index[-1].date()}, "
          f"{df.shape[1]} series: {', '.join(df.columns)}")

    results = []
    for market in args.markets:
        if market not in df or df[market].notna().sum() < 1500:
            print(f"  ! {market}: not enough history, skipped")
            continue
        for cal in args.horizons:
            h = trading_days(cal)
            t0 = time.time()
            res = run_market(df, market, cal, h, args.backtest_years, args.refit_every)
            results.append(res)
            f = res.forecast
            print(f"  {market:>10} +{cal:>2}d ({h:>2} td) -> {f['forecast_price']:8.2f}  "
                  f"80%: {f['p10']:.2f}-{f['p90']:.2f}  P(up) {f['prob_up_%']:.0f}%  "
                  f"backtest MAPE {f['backtest_MAPE_%']:.1f}%  [{res.chosen}, {time.time() - t0:.0f}s]")

    if not results:
        raise SystemExit("Nothing to forecast.")
    path = write_outputs(df, results, args.out, source_used)
    print("\nBacktest accuracy (out-of-sample):")
    cols = ["MAPE_%", "within_5%", "direction_hit_%", "theil_U",
            "DM_pvalue_vs_RW", "coverage80_%", "coverage95_%"]
    for r in results:
        print(f"\n== {r.market} {r.calendar_days}d ==")
        print(r.metrics[cols].astype(float).round(3).to_string())
    print(f"\nReport: {path.resolve()}")


if __name__ == "__main__":
    main()
