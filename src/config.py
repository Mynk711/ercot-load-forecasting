"""Shared settings for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_LOAD_DIR = ROOT / "data" / "raw" / "load"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

START = "2019-01-01"
END = "2025-12-31"
TRAIN_END = "2024-12-31 23:00"   # train on 2019-2024
TEST_START = "2025-01-01 00:00"  # test on all of 2025

# ERCOT's eight weather zones. Older archive files use different names,
# so every known variant is mapped to one standard name.
ZONE_NAME_MAP = {
    "COAST": "COAST",
    "EAST": "EAST",
    "FWEST": "FWEST", "FAR_WEST": "FWEST", "FARWEST": "FWEST",
    "NORTH": "NORTH",
    "NCENT": "NCENT", "NORTH_C": "NCENT", "NORTHCENTRAL": "NCENT",
    "SOUTH": "SOUTH", "SOUTHERN": "SOUTH",
    "SCENT": "SCENT", "SOUTH_C": "SCENT", "SOUTHCENTRAL": "SCENT",
    "WEST": "WEST",
    "ERCOT": "ERCOT",
}
ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]

# Weather stations for the four big load centers: (latitude, longitude, weight).
# Weights roughly reflect each metro's share of ERCOT load, so the
# blended temperature tracks what actually drives system demand.
STATIONS = {
    "houston":     (29.98, -95.36, 0.35),  # Coast zone
    "dfw":         (32.90, -97.04, 0.35),  # North Central zone
    "san_antonio": (29.53, -98.47, 0.15),  # South Central zone
    "austin":      (30.19, -97.67, 0.15),  # South Central zone
}

BALANCE_POINT_F = 65  # standard base temperature for degree days
