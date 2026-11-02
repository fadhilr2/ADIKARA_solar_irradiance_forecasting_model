"""
Download hourly NASA POWER data for one target point (P) and its neighbors.

Target P is the center of a 5x5 grid at 0.5 degree spacing (25 points total).
One CSV is saved per point, plus manifest.csv describing every request.

Run:  python download_power_grid.py
Needs: pip install requests pandas
"""
import os
import time
import requests
import pandas as pd

# ---- settings you may change -------------------------------------------
TARGET_LAT, TARGET_LON = -10.17, 123.67   # Kupang airport area (check on a map)
STEP = 0.5                                # grid spacing in degrees
HALF_WIDTH = 2                            # 2 -> 5x5 grid (25 points)
START_YEAR, END_YEAR = 2021, 2025
PARAMS = "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"   # add T2M,RH2M for the target only

# Quick test: one day, only the target point + one neighbor, saved to data/test.
# Set TEST_MODE = False for the full download.
TEST_MODE = False
TEST_DATE = "20250115"                    # YYYYMMDD, any day inside 2023-2025

OUT_DIR = "data/test" if TEST_MODE else "data/raw"
# -------------------------------------------------------------------------

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_range(lat, lon, start, end, retries=3):
    """Hourly data for one point between two YYYYMMDD dates. Returns (DataFrame, (lat, lon) returned)."""
    params = {
        "parameters": PARAMS,
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=180)
            r.raise_for_status()
            js = r.json()
            df = pd.DataFrame(js["properties"]["parameter"])
            df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
            lon_r, lat_r = js["geometry"]["coordinates"][:2]
            return df.replace(-999, float("nan")), (lat_r, lon_r)
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} for {lat},{lon} {start}-{end}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed: {lat},{lon},{start}-{end}")


def fetch_point(lat, lon):
    frames, returned = [], None
    if TEST_MODE:
        df, returned = fetch_range(lat, lon, TEST_DATE, TEST_DATE)
        frames.append(df)
    else:
        for year in range(START_YEAR, END_YEAR + 1):
            df, returned = fetch_range(lat, lon, f"{year}0101", f"{year}1231")
            frames.append(df)
            time.sleep(8)  # be polite to the API
    out = pd.concat(frames)
    out.index.name = "time_utc"
    return out, returned


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    offsets = range(-HALF_WIDTH, HALF_WIDTH + 1)
    for i in offsets:            # latitude offset
        for j in offsets:        # longitude offset
            if TEST_MODE and (i, j) not in [(0, 0), (0, 1)]:
                continue         # test: target + one neighbor only
            lat = round(TARGET_LAT + i * STEP, 4)
            lon = round(TARGET_LON + j * STEP, 4)
            role = "TARGET" if (i == 0 and j == 0) else "neighbor"
            path = os.path.join(OUT_DIR, f"point_{i:+d}_{j:+d}.csv")
            if os.path.exists(path):          # resume support
                print(f"skip (exists) {path}")
                continue
            print(f"[{role}] requesting {lat}, {lon}")
            df, (lat_r, lon_r) = fetch_point(lat, lon)
            df.to_csv(path)
            rows.append({
                "file": path, "role": role,
                "req_lat": lat, "req_lon": lon,
                "got_lat": lat_r, "got_lon": lon_r,
                "rows": len(df),
                "missing_frac": round(df.isna().mean().mean(), 4),
            })
    if rows:
        manifest = pd.DataFrame(rows)
        manifest.to_csv(os.path.join(OUT_DIR, "manifest.csv"), mode="a",
                        header=not os.path.exists(os.path.join(OUT_DIR, "manifest.csv")),
                        index=False)
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()