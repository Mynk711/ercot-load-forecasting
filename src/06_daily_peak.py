"""Step 6: Daily peak demand model, day-ahead.

Predicts each day's highest hourly ERCOT load using daily weather, calendar, and
peaks known by the morning before (2 and 7 days earlier). Like the hourly model,
the target is the ratio to the peak 7 days earlier, so load growth doesn't
need to be extrapolated.

Compares: seasonal naive (peak 7 days ago), weather-normalized linear regression
(degree days + calendar), and XGBoost. Also compares against the daily max of the
hourly day-ahead model from step 4.
Train 2019-2024 (Winter Storm Uri excluded), test 2025.
"""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LinearRegression
from config import PROCESSED_DIR, BALANCE_POINT_F

URI = ("2021-02-10", "2021-02-20")


def make_model():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=42), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03, max_depth=4, random_state=42), "HistGradientBoosting"


def mape(a, p):
    return float(((a - p).abs() / a).mean() * 100)


def bias(a, p):
    return float(((p - a) / a).mean() * 100)


def build_daily():
    load = pd.read_csv(PROCESSED_DIR / "load_hourly.csv", index_col="timestamp", parse_dates=True)["ERCOT"]
    temp = pd.read_csv(PROCESSED_DIR / "weather_hourly.csv", index_col="timestamp", parse_dates=True)["temp_wavg"]
    d = pd.DataFrame({
        "peak": load.resample("D").max(),
        "peak_hour": load.resample("D").apply(lambda s: s.idxmax().hour if s.notna().any() else None),
        "tmax": temp.resample("D").max(),
        "tmin": temp.resample("D").min(),
        "tmean": temp.resample("D").mean(),
    })
    d["cdd"] = (d["tmean"] - BALANCE_POINT_F).clip(lower=0)
    d["hdd"] = (BALANCE_POINT_F - d["tmean"]).clip(lower=0)
    d["cdd_max"] = (d["tmax"] - BALANCE_POINT_F).clip(lower=0)   # afternoon heat drives summer peaks
    d["hdd_min"] = (BALANCE_POINT_F - d["tmin"]).clip(lower=0)   # morning cold drives winter peaks
    d["cdd_prev"] = d["cdd"].shift(1)                            # heat build-up from the day before
    d["hdd_prev"] = d["hdd"].shift(1)
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays(d.index.min(), d.index.max())
    d["is_holiday"] = d.index.isin(hol).astype(int)
    d["peak_lag2"] = d["peak"].shift(2)   # known the morning before
    d["peak_lag7"] = d["peak"].shift(7)
    d["lag2_ratio"] = d["peak_lag2"] / d["peak_lag7"]
    d["cdd_lag7"] = d["cdd"].shift(7)     # weather on the comparison day
    d["hdd_lag7"] = d["hdd"].shift(7)
    return d.dropna()


FEATURES = ["tmax", "tmin", "tmean", "cdd", "hdd", "cdd_max", "hdd_min", "cdd_prev", "hdd_prev",
            "dayofweek", "month", "is_weekend", "is_holiday", "lag2_ratio", "cdd_lag7", "hdd_lag7"]


def regression_inputs(d, columns=None):
    X = pd.get_dummies(d[["dayofweek", "month"]].astype(str), drop_first=True)
    for c in ["cdd", "hdd", "cdd_max", "hdd_min", "cdd_prev", "hdd_prev",
              "is_holiday", "lag2_ratio", "cdd_lag7", "hdd_lag7"]:
        X[c] = d[c]
    X = X.astype(float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def main():
    d = build_daily()
    train, test = d.loc[:"2024-12-31"].copy(), d.loc["2025-01-01":].copy()
    train = train[~((train.index >= URI[0]) & (train.index <= URI[1]))]
    target = train["peak"] / train["peak_lag7"]

    test["baseline"] = test["peak_lag7"]
    Xtr = regression_inputs(train)
    reg = LinearRegression().fit(Xtr, target)
    test["regression"] = reg.predict(regression_inputs(test, Xtr.columns)) * test["peak_lag7"]
    model, name = make_model()
    model.fit(train[FEATURES], target)
    test["model"] = model.predict(test[FEATURES]) * test["peak_lag7"]

    forecasts = ["baseline", "regression", "model"]
    hourly_path = PROCESSED_DIR / "predictions_2025_48h.csv"
    if hourly_path.exists():
        h = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
        test["hourly_model_max"] = h["model"].resample("D").max().reindex(test.index)
        forecasts.append("hourly_model_max")

    summer = test.index.month.isin([6, 7, 8, 9])
    winter = test.index.month.isin([12, 1, 2])
    top10 = test["peak"] >= test["peak"].quantile(0.9)
    rows = []
    for f in forecasts:
        rows.append({"forecast": f,
                     "MAPE_%": mape(test["peak"], test[f]),
                     "bias_%": bias(test["peak"], test[f]),
                     "summer_MAPE_%": mape(test.loc[summer, "peak"], test.loc[summer, f]),
                     "winter_MAPE_%": mape(test.loc[winter, "peak"], test.loc[winter, f]),
                     "top10pct_peak_days_MAPE_%": mape(test.loc[top10, "peak"], test.loc[top10, f])})
    m = pd.DataFrame(rows).set_index("forecast").round(2)
    print(f"2025 daily peak results, day-ahead ({name} as 'model', {len(test)} days)")
    print(m.to_string())
    print(f"Model reduces MAPE by {(1 - m.loc['model','MAPE_%'] / m.loc['regression','MAPE_%']) * 100:.1f}% versus regression")

    t = test.copy()
    t["ape"] = (t["model"] - t["peak"]).abs() / t["peak"] * 100
    worst = t.sort_values("ape", ascending=False).head(10)[["ape", "tmax", "tmin", "peak", "model", "peak_hour"]]
    worst[["peak", "model"]] /= 1000
    worst.columns = ["APE_%", "tmax", "tmin", "actual_peak_gw", "pred_peak_gw", "actual_peak_hour"]
    print("\n10 worst days:")
    print(worst.round(1).to_string())

    m.to_csv(PROCESSED_DIR / "daily_peak_metrics.csv")
    test[["peak", "peak_hour", "tmax", "tmin"] + forecasts].to_csv(PROCESSED_DIR / "daily_peak_predictions_2025.csv")


if __name__ == "__main__":
    main()
