"""Step 3: Combine load and weather and build the model's input features.

Output: data/processed/features.csv
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from config import PROCESSED_DIR, BALANCE_POINT_F


def main():
    load = pd.read_csv(PROCESSED_DIR / "load_hourly.csv", index_col="timestamp", parse_dates=True)
    weather = pd.read_csv(PROCESSED_DIR / "weather_hourly.csv", index_col="timestamp", parse_dates=True)
    df = load[["ERCOT"]].rename(columns={"ERCOT": "load"}).join(weather[["temp_wavg"]], how="inner")
    df = df.rename(columns={"temp_wavg": "temp"})

    # Calendar features: daily and weekly cycles, seasons, holidays.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(df.index.min(), df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)
    # Cyclical encoding so hour 23 and hour 0 are "close" to the model.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Weather features. Load vs temperature is U-shaped: high in heat (AC)
    # and in cold (heating), lowest around 65F.
    df["temp_sq"] = df["temp"] ** 2
    df["cdd"] = (df["temp"] - BALANCE_POINT_F).clip(lower=0)  # cooling degrees
    df["hdd"] = (BALANCE_POINT_F - df["temp"]).clip(lower=0)  # heating degrees
    df["temp_roll24"] = df["temp"].rolling(24).mean()         # heat build-up over the past day
    df["temp_lag24"] = df["temp"].shift(24)                   # yesterday's temp at this hour
    df["temp_lag168"] = df["temp"].shift(168)                 # last week's temp at this hour

    # Lagged load. The model predicts load relative to last week (load / load_lag168),
    # so recent load enters as ratios to last week rather than raw levels.
    df["load_lag24"] = df["load"].shift(24)
    df["load_lag48"] = df["load"].shift(48)
    df["load_lag168"] = df["load"].shift(168)                 # same hour last week
    df["lag24_ratio"] = df["load_lag24"] / df["load_lag168"]  # rolling 24-hour-ahead version
    df["lag48_ratio"] = df["load_lag48"] / df["load_lag168"]  # true day-ahead version

    df = df.dropna()
    df.to_csv(PROCESSED_DIR / "features.csv")
    print(f"Saved {len(df):,} rows and {df.shape[1] - 1} columns to data/processed/features.csv")


if __name__ == "__main__":
    main()