"""Step 9: Forecast uncertainty with quantile regression and calibrated prediction intervals.

Instead of one number per hour, predicts a 90% prediction interval: a low (5th
percentile) and high (95th percentile) estimate of day-ahead load. A good 90%
interval should contain actual load about 90% of the time ("coverage").

Steps:
  1. Quantile gradient boosting (5th, 50th, 95th percentiles), trained on
     2019-2023 (Winter Storm Uri excluded), on the same ratio-to-last-week target
     and day-ahead (48h) features as the hourly model.
  2. Calibration (conformalized quantile regression): measure how far 2024
     actuals fell outside the raw intervals, and widen intervals by that amount
     so 2024 coverage would have been 90%.
  3. Test both raw and calibrated intervals on 2025.

Outputs: data/processed/intervals_2025.csv, data/processed/uncertainty_metrics.csv,
figures/intervals_july.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from config import PROCESSED_DIR, FIG_DIR

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "is_holiday", "hour_sin", "hour_cos",
            "temp", "temp_sq", "cdd", "hdd", "temp_roll24", "temp_lag24", "temp_lag168", "lag48_ratio"]
LOW_Q, HIGH_Q, TARGET_COVERAGE = 0.05, 0.95, 0.90
URI = ("2021-02-10", "2021-02-21")


def quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=500,
                                         learning_rate=0.05, max_depth=7, random_state=42)


def coverage(actual, low, high):
    return float(((actual >= low) & (actual <= high)).mean() * 100)


def main():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", index_col="timestamp", parse_dates=True)
    df = df[~((df.index >= URI[0]) & (df.index < URI[1]))]
    train = df.loc[:"2023-12-31 23:00"]
    calib = df.loc["2024-01-01":"2024-12-31 23:00"]
    test = df.loc["2025-01-01":].copy()
    target = train["load"] / train["load_lag168"]

    models = {}
    for q in [LOW_Q, 0.5, HIGH_Q]:
        print(f"Training {int(q * 100)}th percentile model ...")
        models[q] = quantile_model(q).fit(train[FEATURES], target)

    def predict(frame):
        p = {q: m.predict(frame[FEATURES]) * frame["load_lag168"].values for q, m in models.items()}
        low = np.minimum(p[LOW_Q], p[HIGH_Q])
        high = np.maximum(p[LOW_Q], p[HIGH_Q])
        return pd.Series(low, frame.index), pd.Series(p[0.5], frame.index), pd.Series(high, frame.index)

    # Calibration on 2024: how far outside the interval did actuals land (relative to load)?
    c_low, _, c_high = predict(calib)
    rel_score = np.maximum((c_low - calib["load"]) / calib["load_lag168"],
                           (calib["load"] - c_high) / calib["load_lag168"])
    n = len(rel_score)
    k = min(1.0, np.ceil((n + 1) * TARGET_COVERAGE) / n)
    adjust = float(np.quantile(rel_score, k))
    print(f"Calibration adjustment: {adjust * 100:+.2f}% of last week's load")

    low, mid, high = predict(test)
    test["low_raw"], test["median"], test["high_raw"] = low, mid, high
    test["low_cal"] = low - adjust * test["load_lag168"]
    test["high_cal"] = high + adjust * test["load_lag168"]

    daily_max_idx = test.groupby(test.index.date)["load"].idxmax().values
    peak_hours = test.loc[daily_max_idx]
    top_days = test.groupby(test.index.date)["load"].max()
    top_days = top_days[top_days >= top_days.quantile(0.9)].index
    top_peak_hours = peak_hours[pd.Index(peak_hours.index.date).isin(top_days)]

    rows = []
    for name, lo, hi in [("raw", "low_raw", "high_raw"), ("calibrated", "low_cal", "high_cal")]:
        rows.append({
            "intervals": name,
            "coverage_all_hours_%": coverage(test["load"], test[lo], test[hi]),
            "coverage_daily_peak_hours_%": coverage(peak_hours["load"], peak_hours[lo], peak_hours[hi]),
            "coverage_top10pct_peak_days_%": coverage(top_peak_hours["load"], top_peak_hours[lo], top_peak_hours[hi]),
            "avg_width_gw": float((test[hi] - test[lo]).mean() / 1000),
            "avg_width_%_of_load": float(((test[hi] - test[lo]) / test["load"]).mean() * 100),
        })
    m = pd.DataFrame(rows).set_index("intervals").round(2)
    med_mape = float(((test["median"] - test["load"]).abs() / test["load"]).mean() * 100)
    print(f"\n2025 90% prediction intervals (target coverage {TARGET_COVERAGE * 100:.0f}%)")
    print(m.to_string())
    print(f"Median (50th percentile) forecast MAPE: {med_mape:.2f}%")

    m.to_csv(PROCESSED_DIR / "uncertainty_metrics.csv")
    test[["load", "low_raw", "median", "high_raw", "low_cal", "high_cal", "temp"]].to_csv(
        PROCESSED_DIR / "intervals_2025.csv")

    week = test.loc["2025-07-14":"2025-07-20"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(week.index, week["low_cal"] / 1000, week["high_cal"] / 1000,
                    alpha=0.25, label="90% interval (calibrated)")
    ax.plot(week.index, week["median"] / 1000, lw=1.5, label="Median forecast")
    ax.plot(week.index, week["load"] / 1000, color="black", lw=2, label="Actual")
    ax.set_ylabel("ERCOT load (GW)")
    ax.set_title("Day-ahead forecast with 90% prediction interval, week of July 14, 2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intervals_july.png", dpi=150)
    print("Figure saved to figures/intervals_july.png")


if __name__ == "__main__":
    main()
