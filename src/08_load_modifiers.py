"""Step 8: Load modifier scenarios: EVs, rooftop solar, and demand response.

Adds hourly profiles for each modifier to actual 2025 ERCOT load and measures how
the peaks change. These are stylized what-if scenarios built from stated
assumptions (listed below), not measured data, so results show the direction and
rough size of each effect rather than a forecast.

Assumptions:
  EVs:       ~10 kWh per vehicle per day (about 3,600 kWh/yr).
             Unmanaged charging starts when people get home (evening-heavy).
             Managed charging is shifted to overnight (midnight-6 AM).
  Solar:     Rooftop output follows a daylight bell curve peaking around 1 PM,
             scaled by season. Output per kW of capacity is set so a typical
             Texas system produces ~1,500 kWh/yr per kW.
  Demand
  response:  On the 20 highest-load days, cut a fixed amount of load during the
             4 hours around that day's peak.

Outputs: data/processed/load_modifier_scenarios.csv, figures/load_modifiers_summer_day.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROCESSED_DIR, FIG_DIR

KWH_PER_EV_DAY = 10
# Share of each day's EV charging energy in each hour (sums to 1).
EV_UNMANAGED = np.array([4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 2, 3, 5, 8, 11, 12, 12, 11, 9, 6], dtype=float)
EV_MANAGED = np.array([16, 18, 18, 16, 14, 10, 4, 1, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=float)
EV_UNMANAGED /= EV_UNMANAGED.sum()
EV_MANAGED /= EV_MANAGED.sum()
SOLAR_KWH_PER_KW_YEAR = 1500
DR_DAYS, DR_HOURS = 20, 4

SCENARIOS = {
    "Base (actual 2025)":            {},
    "+1M EVs, unmanaged":            {"ev_m": 1.0, "ev": "unmanaged"},
    "+1M EVs, managed overnight":    {"ev_m": 1.0, "ev": "managed"},
    "+3M EVs, unmanaged":            {"ev_m": 3.0, "ev": "unmanaged"},
    "+5 GW rooftop solar":           {"solar_gw": 5.0},
    "+10 GW rooftop solar":          {"solar_gw": 10.0},
    "3 GW demand response":          {"dr_gw": 3.0},
    "+3M EVs + 10 GW solar + 3 GW DR": {"ev_m": 3.0, "ev": "unmanaged", "solar_gw": 10.0, "dr_gw": 3.0},
}


def ev_profile(index, millions, mode):
    shares = EV_UNMANAGED if mode == "unmanaged" else EV_MANAGED
    daily_mwh = millions * 1e6 * KWH_PER_EV_DAY / 1000
    return pd.Series(daily_mwh * shares[index.hour], index=index)  # MW in each hour


def solar_profile(index, gw):
    hour = index.hour + 0.5
    doy = index.dayofyear
    day_length = 12 + 2 * np.sin(2 * np.pi * (doy - 80) / 365)        # ~10 h winter, ~14 h summer
    x = (hour - 13) / (day_length / 2)                                  # solar noon ~1 PM in CDT
    shape = np.clip(np.cos(np.pi / 2 * x), 0, None) ** 1.5
    season = 0.75 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)       # stronger sun in summer
    raw = pd.Series(shape * season, index=index)
    # Scale so annual output = capacity * 1,500 kWh/kW.
    annual_mwh_per_mw = SOLAR_KWH_PER_KW_YEAR
    factor = annual_mwh_per_mw * (len(raw) / 8760) / raw.sum()
    return raw * factor * gw * 1000  # MW


def dr_profile(load, gw):
    daily_peak = load.resample("D").max()
    top_days = daily_peak.nlargest(DR_DAYS).index
    cut = pd.Series(0.0, index=load.index)
    for day in top_days:
        day_load = load.loc[day.strftime("%Y-%m-%d")]
        peak_ts = day_load.idxmax()
        window = pd.date_range(peak_ts - pd.Timedelta(hours=DR_HOURS // 2 - 1),
                               periods=DR_HOURS, freq="h")
        cut.loc[cut.index.intersection(window)] = gw * 1000
    return cut


def main():
    load = pd.read_csv(PROCESSED_DIR / "load_hourly.csv", index_col="timestamp", parse_dates=True)["ERCOT"]
    load = load.loc["2025-01-01":"2025-12-31"]

    results, profiles = [], {}
    for name, s in SCENARIOS.items():
        net = load.copy()
        if "ev_m" in s:
            net += ev_profile(net.index, s["ev_m"], s["ev"])
        if "solar_gw" in s:
            net -= solar_profile(net.index, s["solar_gw"])
        if "dr_gw" in s:
            net -= dr_profile(net, s["dr_gw"])
        profiles[name] = net
        daily_peak = net.resample("D").max()
        summer = net[net.index.month.isin([6, 7, 8])]
        results.append({
            "scenario": name,
            "annual_peak_gw": net.max() / 1000,
            "annual_peak_hour": net.idxmax().hour,
            "avg_daily_peak_gw": daily_peak.mean() / 1000,
            "avg_summer_peak_hour": float(summer.groupby(summer.index.date).idxmax().map(lambda t: t.hour).mean()),
            "annual_energy_twh": net.sum() / 1e6,
        })

    res = pd.DataFrame(results).set_index("scenario")
    base = res.loc["Base (actual 2025)"]
    res["peak_change_%"] = (res["annual_peak_gw"] / base["annual_peak_gw"] - 1) * 100
    res["energy_change_%"] = (res["annual_energy_twh"] / base["annual_energy_twh"] - 1) * 100
    res = res.round(2)
    print("2025 load modifier scenarios")
    print(res.to_string())
    res.to_csv(PROCESSED_DIR / "load_modifier_scenarios.csv")

    # Average summer weekday shape for selected scenarios.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ["Base (actual 2025)", "+3M EVs, unmanaged", "+3M EVs, managed overnight",
                 "+10 GW rooftop solar", "+3M EVs + 10 GW solar + 3 GW DR"]:
        if name not in profiles:
            continue
        p = profiles[name]
        p = p[(p.index.month.isin([6, 7, 8])) & (p.index.dayofweek < 5)]
        ax.plot(p.groupby(p.index.hour).mean() / 1000, label=name, lw=2 if name.startswith("Base") else 1.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average net load (GW)")
    ax.set_title("Average summer weekday load under load modifier scenarios, 2025")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "load_modifiers_summer_day.png", dpi=150)
    print("\nFigure saved to figures/load_modifiers_summer_day.png")


if __name__ == "__main__":
    main()
