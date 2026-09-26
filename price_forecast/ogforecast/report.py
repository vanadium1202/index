"""Write forecasts and accuracy tables as CSV/JSON and a self-contained HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import HorizonResult

UNITS = {"brent": "$/bbl", "wti": "$/bbl", "henry_hub": "$/MMBtu"}
LABELS = {"brent": "Brent", "wti": "WTI", "henry_hub": "Henry Hub gas"}

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e1da;--card:#ffffff;
--accent:#0f6e8c;--band1:rgba(15,110,140,.28);--band2:rgba(15,110,140,.12);--good:#1f7a3a;--bad:#a8321e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;
--muted:#a3a19a;--line:#33322f;--card:#1f1f1d;--accent:#4fb3d4;--band1:rgba(79,179,212,.32);
--band2:rgba(79,179,212,.14);--good:#6fcf8a;--bad:#f08a70}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;
margin:0;padding:24px 16px}
main{max-width:1080px;margin:0 auto}h1{font-size:1.6rem;margin:0 0 4px}h2{margin-top:36px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin:14px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-weight:600;color:var(--muted);font-size:.85rem}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
svg text{fill:var(--muted);font-size:11px}.warn{border-left:4px solid var(--bad)}
"""


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


def _fan_svg(df: pd.DataFrame, market: str, results: list[HorizonResult]) -> str:
    hist = df[market].dropna().iloc[-60:]
    W, H, L, R, T, B = 460, 240, 44, 12, 14, 26
    fcs = sorted((r.forecast for r in results), key=lambda f: f["horizon_trading_days"])
    n_hist = len(hist)
    n_tot = n_hist + max(f["horizon_trading_days"] for f in fcs)
    lo = min(hist.min(), min(f["p2_5"] for f in fcs))
    hi = max(hist.max(), max(f["p97_5"] for f in fcs))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def x(i):
        return L + (W - L - R) * i / (n_tot - 1)

    def y(v):
        return T + (H - T - B) * (hi - v) / (hi - lo)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hist.values))
    last_i, last_v = n_hist - 1, hist.values[-1]
    anchor = [(last_i, last_v, last_v, last_v, last_v, last_v)]
    for f in fcs:
        anchor.append((last_i + f["horizon_trading_days"], f["p2_5"], f["p10"],
                       f["forecast_price"], f["p90"], f["p97_5"]))

    def poly(lo_k, hi_k):
        up = [f"{x(a[0]):.1f},{y(a[hi_k]):.1f}" for a in anchor]
        dn = [f"{x(a[0]):.1f},{y(a[lo_k]):.1f}" for a in reversed(anchor)]
        return " ".join(up + dn)

    mid = " ".join(f"{x(a[0]):.1f},{y(a[3]):.1f}" for a in anchor)
    ticks = "".join(
        f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:,.1f}</text>'
        f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--line)"/>'
        for v in np.linspace(lo + pad, hi - pad, 4)
    )
    labels = "".join(
        f'<text x="{x(a[0]):.1f}" y="{H - 8}" text-anchor="end">+{f["horizon_calendar_days"]}d</text>'
        for a, f in zip(anchor[1:], fcs)
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
        f'aria-label="{LABELS.get(market, market)} last 60 trading days and forecast fan">'
        f"{ticks}"
        f'<polygon points="{poly(1, 5)}" fill="var(--band2)"/>'
        f'<polygon points="{poly(2, 4)}" fill="var(--band1)"/>'
        f'<polyline points="{pts}" fill="none" stroke="var(--fg)" stroke-width="1.5"/>'
        f'<polyline points="{mid}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="4 3"/>'
        f'<text x="{L - 6}" y="{T - 2}" text-anchor="end">{UNITS.get(market, "")}</text>'
        f"{labels}</svg>"
    )


def _metrics_table(res: HorizonResult) -> str:
    cols = ["MAPE_%", "within_5%", "within_10%", "direction_hit_%", "theil_U",
            "DM_pvalue_vs_RW", "coverage80_%", "coverage95_%"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["model", *cols])
    body = []
    for m, row in res.metrics.iterrows():
        cls = ' class="good"' if m == res.chosen else ""
        cells = "".join(f"<td>{_fmt(row[c], 3 if c in ('theil_U', 'DM_pvalue_vs_RW') else 1)}</td>" for c in cols)
        body.append(f"<tr><td{cls}>{m}{' ✓' if m == res.chosen else ''}</td>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def write_outputs(df: pd.DataFrame, results: list[HorizonResult], out_dir: str, source: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fc = pd.DataFrame([{k: v for k, v in r.forecast.items() if k != "all_models_change_%"} for r in results])
    fc.to_csv(out / "forecast.csv", index=False)
    (out / "forecast.json").write_text(json.dumps([r.forecast for r in results], indent=2, default=float))
    metrics = pd.concat(
        {(r.market, r.calendar_days): r.metrics for r in results}, names=["market", "horizon_days", "model"]
    )
    metrics.to_csv(out / "backtest_metrics.csv")
    recent = pd.concat(
        {(r.market, r.calendar_days): r.metrics_recent for r in results},
        names=["market", "horizon_days", "model"],
    )
    recent.to_csv(out / "backtest_metrics_last2y.csv")
    for r in results:
        r.backtest.to_csv(out / f"backtest_{r.market}_{r.calendar_days}d.csv")

    rows = []
    for f in fc.to_dict("records"):
        chg = f["expected_change_%"]
        rows.append(
            "<tr>"
            f"<td>{LABELS.get(f['market'], f['market'])}</td><td>{f['horizon_calendar_days']} d</td>"
            f"<td>{f['target_date']}</td><td>{_fmt(f['last_price'])}</td>"
            f"<td><b>{_fmt(f['forecast_price'])}</b></td>"
            f"<td class=\"{'good' if chg >= 0.05 else 'bad' if chg <= -0.05 else ''}\">{chg:+.1f}%</td>"
            f"<td>{_fmt(f['p10'])} – {_fmt(f['p90'])}</td><td>{_fmt(f['p2_5'])} – {_fmt(f['p97_5'])}</td>"
            f"<td>{_fmt(f['prob_up_%'], 0)}%</td><td>±{_fmt(f['backtest_MAPE_%'], 1)}%</td>"
            f"<td>{_fmt(f['backtest_within_5%'], 0)}%</td><td>{f['model']}</td></tr>"
        )
    markets = list(dict.fromkeys(r.market for r in results))
    charts = "".join(
        f'<div class="card"><b>{LABELS.get(m, m)}</b>'
        f"{_fan_svg(df, m, [r for r in results if r.market == m])}</div>"
        for m in markets
    )
    detail = "".join(
        f"<h3>{LABELS.get(r.market, r.market)} – {r.calendar_days} calendar days "
        f"({r.h} trading days)</h3><div class=\"card\">{_metrics_table(r)}"
        f"<p class=\"muted\">Backtest {r.backtest.index[0].date()} → {r.backtest.index[-1].date()}, "
        f"{int(r.metrics.iloc[0]['n_forecasts'])} daily out-of-sample forecasts. "
        f"Last-2-year MAPE for the chosen model: {_fmt(r.metrics_recent.loc[r.chosen, 'MAPE_%'], 1)}%.</p></div>"
        for r in results
    )
    warn = ""
    if source == "synthetic":
        warn = ('<div class="card warn"><b>Synthetic data.</b> This report was produced on a simulated market '
                "to test the pipeline. Its prices and accuracy numbers are not real. Run with "
                "<code>--source live</code> for real forecasts.</div>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oil &amp; Gas Forecast</title><style>{CSS}</style></head><body><main>
<h1>Oil &amp; gas price forecast</h1>
<p class="muted">As of {fc['as_of'].max()} · data source: {html.escape(source)} ·
bands are out-of-sample calibrated (volatility-scaled conformal)</p>{warn}
<div class="card"><table><tr><th>Market</th><th>Horizon</th><th>Target date</th><th>Last</th>
<th>Forecast</th><th>Change</th><th>80% range</th><th>95% range</th><th>P(up)</th>
<th>Typical error</th><th>Hit ±5%</th><th>Model</th></tr>{''.join(rows)}</table></div>
<div class="charts">{charts}</div>
<h2>How accurate is it?</h2>
<p><b>MAPE</b> = mean absolute % error of the forecast price. <b>theil_U</b> &lt; 1 means better than
the random walk (no-change) benchmark; <b>DM_pvalue</b> &lt; 0.05 means that gain is statistically
significant. <b>direction_hit</b> = share of moves whose sign was called correctly (50% = coin toss).
<b>coverage80/95</b> should be close to 80/95 if the ranges are honest. ✓ = model used for the forecast
(the random walk is kept unless another model beats it).</p>
{detail}
</main></body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
