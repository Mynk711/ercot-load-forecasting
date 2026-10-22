"""Step 4: Train and evaluate hourly load forecasts on 2025 (unseen) data.

Compares three forecasts, each run for two forecast horizons:
  - baseline:   seasonal naive (same hour last week), no training
  - regression: linear regression on weather and calendar inputs
  - model:      XGBoost on the same information
Horizons:
  - 24h: rolling 24-hour-ahead (uses load from 24 hours earlier)
  - 48h: true day-ahead (uses load from 48 hours earlier, known the morning before)

Both learned models predict load / load_lag168 (ratio to last week), so they
don't need to extrapolate load growth.

Outputs: results tables in the terminal, data/processed/metrics.csv,
data/processed/predictions_2025_<horizon>.csv, data/processed/worst_days_<horizon>.csv,
figures/*.png (figures use the 48h day-ahead version).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from config import PROCESSED_DIR, FIG_DIR, TRAIN_END, TEST_START

BASE_FEATURES = ["hour", "dayofweek", "month", "is_weekend", "is_holiday", "hour_sin", "hour_cos",
                 "temp", "temp_sq", "cdd", "hdd", "temp_roll24", "temp_lag24", "temp_lag168"]
HORIZONS = {"24h": "lag24_ratio", "48h": "lag48_ratio"}
FORECASTS = ["baseline", "regression", "model"]


def make_model():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=7,
                            subsample=0.8, colsample_bytree=0.8, random_state=42), "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, random_state=42), "HistGradientBoosting"


def regression_inputs(df, ratio_col, columns=None):
    """Linear regression needs hour/day/month as separate on-off columns (dummies),
    because a straight line can't learn that hour 17 is high and hour 3 is low."""
    X = pd.get_dummies(df[["hour", "dayofweek", "month"]].astype(str), drop_first=True)
    for c in ["cdd", "hdd", "temp_roll24", "is_holiday", "temp_lag168", ratio_col]:
        X[c] = df[c]
    X = X.astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def mape(actual, pred):
    return float(((actual - pred).abs() / actual).mean() * 100)


def bias(actual, pred):
    """Positive = over-forecasting on average, negative = under-forecasting."""
    return float(((pred - actual) / actual).mean() * 100)


def peak_mape(df, col):
    """Error on each day's single highest hour, the number operators care most about."""
    daily = df.groupby(df.index.date).agg(actual=("load", "max"), pred=(col, "max"))
    return mape(daily["actual"], daily["pred"])


def run_horizon(train, test, ratio_col):
    test = test.copy()
    features = BASE_FEATURES + [ratio_col]
    target = train["load"] / train["load_lag168"]

    test["baseline"] = test["load_lag168"]

    X_train = regression_inputs(train, ratio_col)
    reg = LinearRegression().fit(X_train, target)
    test["regression"] = reg.predict(regression_inputs(test, ratio_col, X_train.columns)) * test["load_lag168"]

    model, model_name = make_model()
    model.fit(train[features], target)
    test["model"] = model.predict(test[features]) * test["load_lag168"]

    hot = test["temp"] >= test["temp"].quantile(0.9)
    metrics = pd.DataFrame([{
        "forecast": name,
        "MAPE_%": mape(test["load"], test[name]),
        "bias_%": bias(test["load"], test[name]),
        "peak_day_MAPE_%": peak_mape(test, name),
        "hottest_10pct_hours_MAPE_%": mape(test.loc[hot, "load"], test.loc[hot, name]),
    } for name in FORECASTS]).set_index("forecast").round(2)
    return test, metrics, model, model_name, features


def worst_days(test, n=10):
    t = test.copy()
    t["ape"] = (t["model"] - t["load"]).abs() / t["load"] * 100
    days = t.groupby(t.index.date).agg(
        mape=("ape", "mean"), max_temp=("temp", "max"), min_temp=("temp", "min"),
        actual_peak_gw=("load", "max"), pred_peak_gw=("model", "max"))
    days[["actual_peak_gw", "pred_peak_gw"]] /= 1000
    days.index.name = "date"
    return days.sort_values("mape", ascending=False).head(n).round(1)


def main():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", index_col="timestamp", parse_dates=True)
    train, test = df.loc[:TRAIN_END], df.loc[TEST_START:]
    uri = (train.index >= "2021-02-10") & (train.index < "2021-02-21")
    train = train[~uri]  # Winter Storm Uri: recorded load was cut by load shedding
    print(f"Training on {len(train):,} hours (2019-2024), testing on {len(test):,} hours (2025)")

    all_metrics, results = [], {}
    for horizon, ratio_col in HORIZONS.items():
        print(f"\nRunning {horizon} horizon ...")
        t, metrics, model, model_name, features = run_horizon(train, test, ratio_col)
        results[horizon] = (t, model, model_name, features)
        m = metrics.copy()
        m.insert(0, "horizon", horizon)
        all_metrics.append(m)

        print(f"\n2025 test results, {horizon} horizon ({model_name})")
        print(metrics.to_string())
        for ref in ["baseline", "regression"]:
            imp = (1 - metrics.loc["model", "MAPE_%"] / metrics.loc[ref, "MAPE_%"]) * 100
            print(f"Model reduces MAPE by {imp:.1f}% versus {ref}")

        wd = worst_days(t)
        print(f"\n10 worst days, {horizon} horizon (model)")
        print(wd.to_string())
        wd.to_csv(PROCESSED_DIR / f"worst_days_{horizon}.csv")
        t[["load", "baseline", "regression", "model", "temp"]].to_csv(PROCESSED_DIR / f"predictions_2025_{horizon}.csv")

    pd.concat(all_metrics).to_csv(PROCESSED_DIR / "metrics.csv")

    # Figures use the true day-ahead (48h) version.
    t, model, model_name, features = results["48h"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    week = t.loc["2025-07-14":"2025-07-20"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week["load"] / 1000, label="Actual", color="black", lw=2)
    ax.plot(week.index, week["model"] / 1000, label=model_name, lw=1.5)
    ax.plot(week.index, week["regression"] / 1000, label="Regression", lw=1)
    ax.plot(week.index, week["baseline"] / 1000, label="Seasonal naive", lw=1, ls="--")
    ax.set_ylabel("ERCOT load (GW)")
    ax.set_title("Day-ahead forecast vs actual, week of July 14, 2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_vs_actual_july.png", dpi=150)

    by_hour = t.groupby("hour")[["load"] + FORECASTS].apply(
        lambda g: pd.Series({n: mape(g["load"], g[n]) for n in FORECASTS}))
    fig, ax = plt.subplots(figsize=(10, 4))
    by_hour.plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour of day (hour beginning, Central time)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Day-ahead forecast error by hour of day, 2025")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mape_by_hour.png", dpi=150)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot(kind="barh", ax=ax)
        ax.set_title("Feature importance (day-ahead model)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)

    print("\nFigures (day-ahead version) saved to figures/")


if __name__ == "__main__":
    main()