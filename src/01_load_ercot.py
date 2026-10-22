"""Step 1: Read ERCOT's yearly hourly load archives and clean them.

Input : yearly files downloaded from ercot.com/gridinfo/load/load_hist,
        unzipped into data/raw/load/ (xlsx, xls, or csv).
Output: data/processed/load_hourly.csv, one row per hour, in local time
        (hour-BEGINNING timestamps), with all 8 zones plus the ERCOT total.
"""
from datetime import datetime
import pandas as pd
from config import RAW_LOAD_DIR, PROCESSED_DIR, ZONE_NAME_MAP, START, END


def read_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def to_hour_beginning(value):
    """ERCOT labels each hour by when it ENDS ('01/01/2024 01:00' = 00:00-01:00).

    Returns (hour-beginning timestamp, is_dst_duplicate). Handles both text
    labels and cells that Excel auto-converted into real dates.
    """
    if isinstance(value, datetime):  # Excel converted this cell into a real date
        return pd.Timestamp(value) - pd.Timedelta(hours=1), False

    text = str(value).strip().upper()
    is_dst = text.endswith("DST")
    text = text.replace("DST", "").strip()
    date_part, time_part = text.split()[:2]
    hour_ending = int(time_part.split(":")[0])  # runs 1..24

    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            day = pd.to_datetime(date_part, format=fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unrecognized Hour Ending value: {value!r}")

    return day + pd.Timedelta(hours=hour_ending - 1), is_dst


def clean_one(df):
    df.columns = [str(c).strip().upper().replace(" ", "") for c in df.columns]
    hour_col = next(c for c in df.columns if "HOUR" in c)

    # Data quality: report timestamps that aren't text (e.g., Excel-converted dates).
    n_non_text = int((~df[hour_col].map(lambda v: isinstance(v, str))).sum())
    print(f"  {len(df):,} rows, {n_non_text} non-text timestamps")

    parsed = df[hour_col].apply(to_hour_beginning)
    df["timestamp"] = [p[0] for p in parsed]
    df["is_dst_dup"] = [p[1] for p in parsed]
    df = df.rename(columns={c: ZONE_NAME_MAP[c] for c in df.columns if c in ZONE_NAME_MAP})
    keep = ["timestamp", "is_dst_dup"] + [c for c in ZONE_NAME_MAP.values() if c in df.columns]
    return df[list(dict.fromkeys(keep))]


def main():
    files = sorted(p for p in RAW_LOAD_DIR.iterdir() if p.suffix.lower() in {".xlsx", ".xls", ".csv"})
    if not files:
        raise SystemExit(f"No load files found in {RAW_LOAD_DIR}. See README step 1.")

    frames = []
    for f in files:
        print(f"Reading {f.name}")
        frames.append(clean_one(read_file(f)))
    load = pd.concat(frames, ignore_index=True)

    # Data quality: drop the repeated fall-back DST hour and any duplicate rows.
    n_dst = int(load["is_dst_dup"].sum())
    load = load[~load["is_dst_dup"]].drop(columns="is_dst_dup")
    n_dupes = int(load["timestamp"].duplicated().sum())
    load = load.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    # Fill any missing hours (e.g., the skipped spring-forward hour) by interpolation.
    full_index = pd.date_range(START, f"{END} 23:00", freq="h")
    load = load.reindex(full_index)
    n_missing = int(load["ERCOT"].isna().sum())
    load = load.interpolate(limit=3)
    load.index.name = "timestamp"

    print(f"Dropped {n_dst} DST duplicate hours and {n_dupes} duplicate rows")
    print(f"Filled {n_missing} missing hours by interpolation")
    print(f"Remaining missing hours: {int(load['ERCOT'].isna().sum())}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load.to_csv(PROCESSED_DIR / "load_hourly.csv")
    print(f"Saved {len(load):,} hours to data/processed/load_hourly.csv")


if __name__ == "__main__":
    main()