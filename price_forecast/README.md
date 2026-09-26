# Oil & Gas Short-Horizon Price Forecaster (7–15 days)

Forecasts **Brent, WTI and Henry Hub natural gas** 7 and 15 calendar days ahead (any horizon can be
requested). For each forecast it gives a point price, 80% and 95% ranges, the probability of a rise,
and the accuracy the same model achieved in an out-of-sample walk-forward backtest.

## Quick start

```bash
pip install -r requirements.txt
python run_forecast.py                         # live FRED data
EIA_API_KEY=xxxx python run_forecast.py        # + futures curve & EIA inventories (recommended)
python run_forecast.py --source github         # EIA spot prices via GitHub mirror (no key, prices only)
python run_forecast.py --horizons 7 10 15      # any horizons, in calendar days
python run_forecast.py --source csv --csv my_prices.csv --markets brent ttf   # your own data
python run_forecast.py --source synthetic --quick                              # offline test
```

Output goes to `outputs/`: `oil_gas_price_forecast.xlsx` (summary table: 7/15-day forecast and % accuracy per market, with a Details sheet), `report.html` (open in a browser), `forecast.csv/json`,
`backtest_metrics.csv` (full backtest), `backtest_metrics_last2y.csv`, and every daily backtest
forecast (`backtest_<market>_<h>d.csv`) for your own checks. You can get a free EIA key at
https://www.eia.gov/opendata/register.php. A full run takes about 2 minutes.

## Data (all free)

| Driver | Series | Source | Why it matters at 1–3 weeks |
|---|---|---|---|
| Brent, WTI, Henry Hub spot | DCOILBRENTEU, DCOILWTICO, DHHNGSP | FRED | targets, momentum, mean reversion |
| WTI futures months 1/2/4 | RCLC1/2/4 | EIA | backwardation = tight prompt market |
| HH futures months 1/2 | RNGC1/2 | EIA | gas curve slope, winter premium |
| US crude stocks (ex-SPR), production, refinery utilisation | WCESTUS1, WCRFPUS2, WPULEUS3 | EIA weekly | surprise vs 5-year seasonal norm |
| Working gas in storage (L48) | NW2_EPG0_SWO_R48_BCF | EIA weekly | storage deficit/surplus drives HH |
| Oil volatility (OVX), VIX | OVXCLS, VIXCLS | FRED | risk regime, width of ranges |
| Dollar index, 10y breakeven, 10y yield, S&P 500 | DTWEXBGS, T10YIE, DGS10, SP500 | FRED | macro / risk-on-off |

Weekly EIA data is re-dated to its **release** day (WPSR Wednesday, gas storage Thursday), so the
model never "knows" an inventory figure before the market did. Your own series (Platts Dubai,
TTF, JKM, Argus, Indian APM gas price, etc.) can be added by CSV; any column can be a target.

## Method

1. **Features (~50):** returns over 1–126 days, realised volatility (10/21/63 d), distance from
   50/200-day averages, 1-year z-score, RSI, drawdown, cross-market moves, Brent–WTI spread, futures
   curve slope, inventory anomaly vs the 5-year same-week average, macro drivers, seasonality.
2. **Target:** log price change over *h* trading days (7 calendar days ≈ 5 trading days, 15 ≈ 11).
3. **Models:** random walk (no-change benchmark), ridge regression, shallow gradient-boosted trees,
   and their ensemble.
4. **Walk-forward backtest:** 8 years, expanding window, refit every quarter. At each refit the model
   trains only on outcomes already known on that date. Tests check that no future data leaks in.
5. **Model choice:** the model with the lowest backtest error is used only if it beats the random walk
   (Theil U < 0.995, Diebold–Mariano p < 0.20). Otherwise the forecast is "no change" and the value of
   the tool is the calibrated range.
6. **Ranges:** volatility-scaled conformal intervals. Backtest errors are divided by the volatility at
   the time of each forecast, and their quantiles are re-scaled by today's volatility. The ranges widen
   automatically in turbulent markets, and their coverage is measured out-of-sample.

## Accuracy metrics reported

| Metric | Meaning |
|---|---|
| MAPE % | average absolute % error of the forecast price, the "typical error" |
| within ±5% / ±10% | share of forecasts that landed within 5% / 10% of the actual price |
| direction_hit % | share of up/down moves called correctly (50% = coin toss) |
| theil_U | model RMSE ÷ random-walk RMSE (< 1 = better than no-change) |
| DM p-value | whether the gain over the random walk is statistically significant |
| coverage80/95 % | how often the actual price fell inside the 80%/95% range; should be ≈ 80/95 |

## Expected real-market accuracy

The numbers below are **my estimates**. They are based on typical 2015–2025 volatility (Brent ≈ 30%,
WTI ≈ 33%, Henry Hub ≈ 55–65% annualised) and on the published forecasting literature (e.g.
Alquist, Kilian & Vigfusson 2013; Baumeister & Kilian 2014–15). They are **not** results of this code
on live data. Running `--source live` measures the actual figures and prints them in the report.

| Market | Horizon | Typical error (MAPE) | Within ±5% | Direction called |
|---|---|---|---|---|
| Brent | 7 d | 2.5–4% | ~75–85% | 50–56% |
| Brent | 15 d | 4–6% | ~55–65% | 50–56% |
| WTI | 7 d | 3–4.5% | ~70–80% | 50–56% |
| WTI | 15 d | 4.5–6.5% | ~50–60% | 50–56% |
| Henry Hub | 7 d | 5–8% | ~40–50% | 50–57% |
| Henry Hub | 15 d | 8–12% | ~30–40% | 50–57% |

In shock periods (2008, 2014–15, 2020, 2022) errors are 2–3× larger; the ranges widen with them.

**Be realistic about this:** at 1–3 week horizons, oil and gas prices are close to a random walk.
Good models typically improve on the no-change forecast by only 0–5% in RMSE, and the gain is often
not statistically significant. The most reliable output is therefore the **range** and the
**probability of a rise**, not the point price. The gas model gains most from the storage and
curve-slope data, so supply an EIA key. Weather forecasts (HDD/CDD) are the strongest missing
driver for Henry Hub; add them via CSV if you have them. The tool cannot anticipate discrete events
(OPEC+ decisions, sanctions, hurricanes, outages).

## Tests

```bash
python -m pytest -q tests
```

The tests check for look-ahead in both features and backtest training, check the target definition,
and check that the forecast ranges are consistent.
