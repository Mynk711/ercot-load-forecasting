"""Step 2: Download hourly temperatures and build one ERCOT-wide temperature.

Uses the free meteostat library (which pulls NOAA station observations).
Output: data/processed/weather_hourly.csv with each station's temperature (F)
        and a load-weighted average, 'temp_wavg', in local time.
"""
from datetime import datetime
import pandas as pd
from meteostat import Point, Hourly
from config import PROCESSED_DIR, STATIONS, START, END


def main():
    start = datetime.fromisoformat(START)
    end = datetime.fromisoformat(END).replace(hour=23)

    series = {}
    for name, (lat, lon, _) in STATIONS.items():
        print(f"Downloading {name} ...")
        df = Hourly(Point(lat, lon), start, end, timezone="America/Chicago").fetch()
        if df.empty:
            raise SystemExit(f"No weather data returned for {name}")
        temp = df["temp"]
        temp.index = temp.index.tz_localize(None)           # local wall-clock time, like the load data
        temp = temp[~temp.index.duplicated(keep="first")]   # drop the repeated DST hour
        series[name] = temp * 9 / 5 + 32                    # Celsius to Fahrenheit

    weather = pd.DataFrame(series)
    weather = weather.reindex(pd.date_range(start, end, freq="h"))
    weather = weather.interpolate(limit=6)  # fill short station outages

    # Load-weighted average across stations (skips any station missing that hour).
    weights = pd.Series({n: w for n, (_, _, w) in STATIONS.items()})
    present = weather.notna()
    weather["temp_wavg"] = (weather.fillna(0) * weights).sum(axis=1) / (present * weights).sum(axis=1)
    weather.index.name = "timestamp"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(PROCESSED_DIR / "weather_hourly.csv")
    print(f"Saved {len(weather):,} hours to data/processed/weather_hourly.csv")


if __name__ == "__main__":
    main()
