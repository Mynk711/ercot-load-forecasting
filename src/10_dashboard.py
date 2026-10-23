"""Step 10: Forecast performance dashboard (interactive HTML).

Builds figures/dashboard.html from the day-ahead (48h) results:
  - headline metrics (MAPE, bias, peak-day MAPE, interval coverage)
  - actual vs forecast over 2025, with the 90% interval, zoomable
  - monthly MAPE and bias
  - error by hour of day
  - daily error vs temperature (weekdays vs weekends)
  - worst-forecast days
Open the file in a browser. Needs: pip install plotly
"""
import pandas as pd
import plotly.graph_objects as go
from config import PROCESSED_DIR, FIG_DIR


def mape(a, p):
    return float(((a - p).abs() / a).mean() * 100)


def bias(a, p):
    return float(((p - a) / a).mean() * 100)


def main():
    p = pd.read_csv(PROCESSED_DIR / "predictions_2025_48h.csv", index_col=0, parse_dates=True)
    iv_path = PROCESSED_DIR / "intervals_2025.csv"
    iv = pd.read_csv(iv_path, index_col=0, parse_dates=True) if iv_path.exists() else None

    p["ape"] = (p["model"] - p["load"]).abs() / p["load"] * 100
    p["pe"] = (p["model"] - p["load"]) / p["load"] * 100
    daily = p.groupby(p.index.date).agg(
        mape=("ape", "mean"), avg_temp=("temp", "mean"), max_temp=("temp", "max"),
        actual_peak=("load", "max"), pred_peak=("model", "max"))
    daily.index = pd.to_datetime(daily.index)
    peak_mape = mape(daily["actual_peak"], daily["pred_peak"])

    kpis = [("Hourly MAPE", f"{mape(p['load'], p['model']):.2f}%"),
            ("Bias", f"{bias(p['load'], p['model']):+.2f}%"),
            ("Peak-day MAPE", f"{peak_mape:.2f}%"),
            ("vs. seasonal naive", f"{(1 - mape(p['load'], p['model']) / mape(p['load'], p['baseline'])) * 100:.0f}% lower error"),
            ("vs. regression", f"{(1 - mape(p['load'], p['model']) / mape(p['load'], p['regression'])) * 100:.0f}% lower error")]
    if iv is not None:
        cov = ((iv["load"] >= iv["low_cal"]) & (iv["load"] <= iv["high_cal"])).mean() * 100
        kpis.append(("90% interval coverage", f"{cov:.1f}%"))

    figs = []
    layout = dict(template="plotly_white", height=420, margin=dict(l=60, r=30, t=60, b=40))

    # 1. Actual vs forecast, whole year, zoomable.
    f = go.Figure()
    if iv is not None:
        f.add_trace(go.Scatter(x=iv.index, y=iv["high_cal"] / 1000, line=dict(width=0),
                               showlegend=False, hoverinfo="skip"))
        f.add_trace(go.Scatter(x=iv.index, y=iv["low_cal"] / 1000, line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(70,130,180,0.2)", name="90% interval"))
    f.add_trace(go.Scatter(x=p.index, y=p["load"] / 1000, name="Actual", line=dict(color="black", width=1.5)))
    f.add_trace(go.Scatter(x=p.index, y=p["model"] / 1000, name="Day-ahead forecast", line=dict(width=1.2)))
    f.update_layout(title="Actual vs day-ahead forecast, 2025 (drag to zoom)", yaxis_title="GW",
                    xaxis=dict(rangeslider=dict(visible=True),
                               range=[pd.Timestamp("2025-07-14"), pd.Timestamp("2025-07-21")]),
                    **{**layout, "height": 500})
    figs.append(f)

    # 2. Monthly MAPE and bias.
    monthly = p.groupby(p.index.to_period("M")).apply(
        lambda g: pd.Series({"MAPE": mape(g["load"], g["model"]), "Bias": bias(g["load"], g["model"])}))
    months = monthly.index.strftime("%b")
    f = go.Figure([go.Bar(x=months, y=monthly["MAPE"], name="MAPE %"),
                   go.Scatter(x=months, y=monthly["Bias"], name="Bias %", mode="lines+markers")])
    f.update_layout(title="Monthly MAPE and bias", yaxis_title="%", **layout)
    figs.append(f)

    # 3. Error by hour, all three forecasts.
    by_hour = p.groupby(p.index.hour).apply(
        lambda g: pd.Series({n: mape(g["load"], g[n]) for n in ["baseline", "regression", "model"]}))
    f = go.Figure([go.Bar(x=by_hour.index, y=by_hour[n], name=label)
                   for n, label in [("baseline", "Seasonal naive"), ("regression", "Regression"), ("model", "XGBoost")]])
    f.update_layout(title="MAPE by hour of day", xaxis_title="Hour (hour beginning, Central)",
                    yaxis_title="MAPE %", barmode="group", **layout)
    figs.append(f)

    # 4. Daily error vs temperature.
    wk = daily.index.dayofweek >= 5
    f = go.Figure([
        go.Scatter(x=daily.loc[~wk, "avg_temp"], y=daily.loc[~wk, "mape"], mode="markers", name="Weekday",
                   text=daily.index[~wk].strftime("%b %d"), hovertemplate="%{text}: %{y:.1f}% at %{x:.0f}°F"),
        go.Scatter(x=daily.loc[wk, "avg_temp"], y=daily.loc[wk, "mape"], mode="markers", name="Weekend",
                   text=daily.index[wk].strftime("%b %d"), hovertemplate="%{text}: %{y:.1f}% at %{x:.0f}°F"),
    ])
    f.update_layout(title="Daily forecast error vs average temperature", xaxis_title="Average temperature (°F)",
                    yaxis_title="Daily MAPE %", **layout)
    figs.append(f)

    # 5. Worst days table.
    worst = daily.sort_values("mape", ascending=False).head(10)
    f = go.Figure(go.Table(
        header=dict(values=["Date", "Day", "Daily MAPE %", "Max temp °F", "Actual peak GW", "Forecast peak GW"],
                    fill_color="#e8eef5", align="left"),
        cells=dict(values=[worst.index.strftime("%Y-%m-%d"), worst.index.strftime("%a"),
                           worst["mape"].round(1), worst["max_temp"].round(0),
                           (worst["actual_peak"] / 1000).round(1), (worst["pred_peak"] / 1000).round(1)],
                   align="left")))
    f.update_layout(title="10 worst-forecast days", **{**layout, "height": 380})
    figs.append(f)

    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{k}</div></div>' for k, v in kpis)
    parts = [fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
             for i, fig in enumerate(figs)]
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ERCOT Day-Ahead Load Forecast Performance, 2025</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; background: #f6f8fa; color: #1f2328; }}
header {{ padding: 24px 32px 8px; }}
h1 {{ margin: 0 0 4px; font-size: 24px; }}
p.sub {{ margin: 0; color: #57606a; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 16px 32px; }}
.kpi {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 18px; min-width: 150px; }}
.kpi .v {{ font-size: 22px; font-weight: 600; }}
.kpi .l {{ color: #57606a; font-size: 13px; margin-top: 2px; }}
.card {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; margin: 0 32px 16px; padding: 8px; }}
footer {{ color: #57606a; font-size: 13px; padding: 8px 32px 32px; }}
</style></head><body>
<header><h1>ERCOT Day-Ahead Load Forecast Performance, 2025</h1>
<p class="sub">XGBoost day-ahead model on a 2025 holdout year. Uses observed weather, so operational error would be higher.</p></header>
<div class="kpis">{kpi_html}</div>
{''.join(f'<div class="card">{part}</div>' for part in parts)}
<footer>Data: ERCOT hourly load archives, NOAA weather via Meteostat. Code: github.com/Mynk711/ercot-load-forecasting</footer>
</body></html>"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {out}. Open it in a browser.")


if __name__ == "__main__":
    main()
