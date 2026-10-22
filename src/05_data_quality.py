"""Step 5: Automated data quality checks on hourly load, plus a test of how well they work.

Checks (run on every hour):
  - missing:   no value recorded
  - range:     outside a physically plausible range for ERCOT
  - spike:     far from the surrounding hours (bad meter reading, glitch)
  - flatline:  the same exact value repeated for 4+ hours (stuck data feed)
  - zone_sum:  the 8 weather zones don't add up to the ERCOT total

Test: inject known fake anomalies into a copy of the clean data, run the checks,
and measure what share were caught (detection rate) and how often clean hours are
wrongly flagged (false positive rate).
Outputs: data/processed/dq_flags.csv, data/processed/dq_test_results.csv
"""
import numpy as np
import pandas as pd
from config import PROCESSED_DIR, ZONES

RANGE_MW = (15_000, 100_000)   # ERCOT total has never been outside this
SPIKE_PCT = 0.10               # >10% away from the median of the surrounding hours
FLAT_HOURS = 4                 # identical values for 4+ hours in a row
ZONE_SUM_PCT = 0.005           # zones vs total differ by more than 0.5%


def run_checks(df):
    total = df["ERCOT"]
    flags = pd.DataFrame(index=df.index)
    flags["missing"] = total.isna()
    flags["range"] = total.notna() & ((total < RANGE_MW[0]) | (total > RANGE_MW[1]))

    # Compare each hour to the median of the 2 hours before and 2 after (not itself).
    neighbors = pd.concat([total.shift(k) for k in (-2, -1, 1, 2)], axis=1).median(axis=1)
    flags["spike"] = ((total - neighbors).abs() / neighbors) > SPIKE_PCT

    run_id = (total != total.shift()).cumsum()
    run_len = total.groupby(run_id).transform("size")
    flags["flatline"] = total.notna() & (run_len >= FLAT_HOURS)

    zones = [z for z in ZONES if z in df.columns]
    if len(zones) == len(ZONES):
        flags["zone_sum"] = ((df[zones].sum(axis=1, min_count=len(ZONES)) - total).abs() / total) > ZONE_SUM_PCT
    else:
        flags["zone_sum"] = False

    flags["any"] = flags.any(axis=1)
    return flags


def inject_anomalies(df, n_each=50, seed=42):
    """Plant fake errors in one zone and the total together, like a bad meter reading would."""
    rng = np.random.default_rng(seed)
    dirty = df.copy()
    zone_cols = [z for z in ZONES if z in df.columns] or ["ERCOT"]
    truth = []  # (anomaly type, list of affected timestamps)
    candidates = rng.permutation(np.arange(10, len(df) - 10))
    used, picks = set(), iter(candidates)

    def next_start(width):
        for i in picks:
            if all(j not in used for j in range(i - 3, i + width + 3)):
                used.update(range(i, i + width))
                return i

    for kind in ["spike", "drop", "zero", "flatline", "missing"]:
        for _ in range(n_each):
            width = 6 if kind == "flatline" else 1
            i = next_start(width)
            ts = df.index[i:i + width]
            zone = rng.choice(zone_cols)
            if kind == "spike":
                delta = df.iloc[i]["ERCOT"] * rng.uniform(0.15, 0.40)
            elif kind == "drop":
                delta = -df.iloc[i]["ERCOT"] * rng.uniform(0.15, 0.40)
            elif kind == "zero":
                delta = -df.iloc[i]["ERCOT"]
            if kind in ("spike", "drop", "zero"):
                if zone != "ERCOT":
                    dirty.loc[ts, zone] = dirty.loc[ts, zone] + delta
                dirty.loc[ts, "ERCOT"] = dirty.loc[ts, "ERCOT"] + delta
            elif kind == "flatline":
                for c in set(zone_cols + ["ERCOT"]):
                    dirty.loc[ts, c] = dirty.loc[ts[0], c]
            elif kind == "missing":
                dirty.loc[ts, "ERCOT"] = np.nan
            truth.append((kind, ts))
    return dirty, truth


def main():
    load = pd.read_csv(PROCESSED_DIR / "load_hourly.csv", index_col="timestamp", parse_dates=True)

    # 1. Checks on the real cleaned data.
    flags = run_checks(load)
    flags.to_csv(PROCESSED_DIR / "dq_flags.csv")
    print("Flags on real data (hours):")
    print(flags.drop(columns="any").sum().to_string())
    print(f"Total flagged hours: {int(flags['any'].sum())} of {len(flags):,} "
          f"({flags['any'].mean() * 100:.2f}%)")

    # 2. Injection test: how many planted errors do the checks catch?
    dirty, truth = inject_anomalies(load)
    dflags = run_checks(dirty)
    planted = pd.Index(sorted({t for _, ts in truth for t in ts}))
    rows = []
    for kind in ["spike", "drop", "zero", "flatline", "missing"]:
        events = [ts for k, ts in truth if k == kind]
        caught = sum(dflags.loc[ts, "any"].any() for ts in events)
        rows.append({"anomaly": kind, "planted": len(events), "caught": caught,
                     "detection_rate_%": round(caught / len(events) * 100, 1)})
    results = pd.DataFrame(rows).set_index("anomaly")
    total_caught, total_planted = results["caught"].sum(), results["planted"].sum()

    # False positives: the real data has already been cleaned, so any hour flagged
    # there is most likely a false alarm (or a real issue worth inspecting by hand).
    fp_rate = flags["any"].mean() * 100

    print("\nInjection test (planted anomalies):")
    print(results.to_string())
    print(f"Overall detection rate: {total_caught}/{total_planted} "
          f"({total_caught / total_planted * 100:.1f}%)")
    print(f"Flag rate on real cleaned data (upper bound on false positives): {fp_rate:.2f}%")
    results.to_csv(PROCESSED_DIR / "dq_test_results.csv")


if __name__ == "__main__":
    main()
