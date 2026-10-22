"""Step 7: Monthly energy model with weather normalization and an economic driver.

Predicts average daily energy (GWh/day) for each month from monthly heating and
cooling degree days plus Texas nonfarm employment (economic growth).
Average per day is used so 28-day and 31-day months are comparable.

Economic data: download the CSV for FRED series TXNA (Texas, All Employees:
Total Nonfarm) from fred.stlouisfed.org/series/TXNA and save it as
data/raw/TXNA.csv.

Weather normalization: also predicts each 2025 month under NORMAL weather (the
2019-2024 average degree days for that calendar month), which separates load
growth from weather swings.
Train 2019-2024 (Feb 2021, Winter Storm Uri, excluded), test 2025.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from config import ROOT, PROCESSED_DIR, BALANCE_POINT_F

FRED_PATH = ROOT / "data" / "raw" / "TXNA.csv"


def load_employment():
    if not FRED_PATH.exists():
        raise SystemExit(f"Missing {FRED_PATH}. Download TXNA from FRED (see the top of this file).")
    e = pd.read_csv(FRED_PATH)
    date_col = next(c for c in e.columns if "date" in c.lower())
    value_col = next(c for c in e.columns if c != date_col)
    e[date_col] = pd.to_datetime(e[date_col])
    s = pd.to_numeric(e.set_index(date_col)[value_col], errors="coerce")
    s.index = s.index.to_period("M")
    return s.rename("employment_k")


def build_monthly():
    load = pd.read_csv(PROCESSED_DIR / "load_hourly.csv", index_col="timestamp", parse_dates=True)["ERCOT"]
    temp = pd.read_csv(PROCESSED_DIR / "weather_hourly.csv", index_col="timestamp", parse_dates=True)["temp_wavg"]
    daily_t = temp.resample("D").mean()
    daily = pd.DataFrame({
        "energy_gwh": load.resample("D").sum() / 1000,
        "cdd": (daily_t - BALANCE_POINT_F).clip(lower=0),
        "hdd": (BALANCE_POINT_F - daily_t).clip(lower=0),
    })
    m = daily.groupby(daily.index.to_period("M")).mean()   # averages per day within each month
    m.columns = ["gwh_per_day", "cdd_per_day", "hdd_per_day"]
    m = m.join(load_employment(), how="left")
    m["month"] = m.index.month
    m["trend"] = (m.index.year - 2019) * 12 + m.index.month  # months since start
    return m


def main():
    m = build_monthly()
    missing_emp = m["employment_k"].isna().sum()
    if missing_emp:
        print(f"Warning: {missing_emp} months have no employment data and are dropped")
        m = m.dropna(subset=["employment_k"])

    train = m[(m.index.year <= 2024) & (m.index != pd.Period("2021-02", "M"))]
    test = m[m.index.year == 2025].copy()
    features = ["cdd_per_day", "hdd_per_day", "employment_k", "trend"]
    reg = LinearRegression().fit(train[features], train["gwh_per_day"])
    test["predicted"] = reg.predict(test[features])

    # Weather-only model (no economic driver), to show what the economy term adds.
    reg_w = LinearRegression().fit(train[["cdd_per_day", "hdd_per_day"]], train["gwh_per_day"])
    test["predicted_weather_only"] = reg_w.predict(test[["cdd_per_day", "hdd_per_day"]])

    # Normal weather: 2019-2024 average degree days for each calendar month.
    normals = train.groupby("month")[["cdd_per_day", "hdd_per_day"]].mean()
    normal_inputs = test[["month", "employment_k", "trend"]].join(normals, on="month")
    test["weather_normalized"] = reg.predict(normal_inputs[features])

    def mape(a, p):
        return float(((a - p).abs() / a).mean() * 100)

    def bias(a, p):
        return float(((p - a) / a).mean() * 100)

    print("Monthly energy model coefficients (GWh/day):")
    for f, c in zip(features, reg.coef_):
        print(f"  {f:14s} {c:+.3f}")
    print(f"  intercept      {reg.intercept_:+.1f}")
    print("\n2025 monthly results:")
    for col in ["predicted", "predicted_weather_only"]:
        print(f"  {col:24s} MAPE {mape(test['gwh_per_day'], test[col]):.2f}%   "
              f"bias {bias(test['gwh_per_day'], test[col]):+.2f}%")

    out = test[["gwh_per_day", "predicted", "predicted_weather_only", "weather_normalized",
                "cdd_per_day", "hdd_per_day", "employment_k"]].round(1)
    out.index = out.index.astype(str)
    print("\n" + out.to_string())
    actual_annual = (test["gwh_per_day"] * test.index.days_in_month).sum() / 1000
    normal_annual = (test["weather_normalized"] * test.index.days_in_month).sum() / 1000
    print(f"\n2025 actual energy: {actual_annual:.1f} TWh   weather-normalized: {normal_annual:.1f} TWh")
    out.to_csv(PROCESSED_DIR / "monthly_energy_2025.csv")


if __name__ == "__main__":
    main()
