"""
Download hourly Open-Meteo (ERA5 reanalysis) data for one target point (P) and its neighbors.

Target P is the center of a 5x5 grid at 0.5 degree spacing (25 points total).
Points are downloaded center-out: target first, then nearest neighbors, then the corners.
One CSV is saved per point, plus manifest.csv describing every finished point.

Each CSV has: time_utc, ALLSKY_SFC_SW_DWN (Open-Meteo), CLRSKY_SFC_SW_DWN (pvlib)

Resume support:
  - finished points (point_*.csv exists) are skipped
  - inside a point, each year is cached in <OUT_DIR>/cache/, so a crash only costs one year

Run:  python download_openmeteo_grid.py
Needs: pip install requests pandas pvlib
"""
import json
import os
import shutil
import sys
import time

import pandas as pd
import pvlib
import requests

# ---- settings you may change -------------------------------------------
TARGET_LAT, TARGET_LON = -10.17, 123.67   # Kupang airport area (check on a map)
STEP = 0.5                                # grid spacing in degrees
HALF_WIDTH = 2                            # 2 -> 5x5 grid (25 points)
START_YEAR, END_YEAR = 2001, 2025
SLEEP_BETWEEN_REQUESTS = 5                # seconds

VARIABLES = {
    "shortwave_radiation": "ALLSKY_SFC_SW_DWN",   # GHI, W/m2, mean of the preceding hour
}
MODEL = "era5"   # fixed model so 2001-2025 is consistent (~0.25 deg grid)

# Quick test: one day, only the target point + one neighbor, saved to data/test.
TEST_MODE = False
TEST_DATE = "2025-01-15"                  # YYYY-MM-DD

OUT_DIR = "data/test" if TEST_MODE else "data/raw"
# -------------------------------------------------------------------------

URL = "https://archive-api.open-meteo.com/v1/archive"


class DailyLimit(Exception):
    pass


def fetch_range(lat, lon, start, end, retries=6):
    """Hourly data for one point between two YYYY-MM-DD dates.
    Returns (DataFrame, (lat, lon, elevation) of the grid cell actually used)."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES),
        "models": MODEL,
        "timezone": "GMT",
    }
    for attempt in range(retries):
        wait = 15 * (attempt + 1)
        try:
            r = requests.get(URL, params=params, timeout=180)
            if r.status_code == 429:
                try:
                    reason = r.json().get("reason", "")
                except Exception:
                    reason = ""
                low = reason.lower()
                if "daily" in low:
                    raise DailyLimit(reason or "daily limit reached")
                wait = 600 if "hourly" in low else 90     # hourly cap vs per-minute cap
                print(f"  rate limited: {reason or '429'} -> waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            js = r.json()
            hourly = js["hourly"]
            df = pd.DataFrame(
                {name: hourly[name] for name in VARIABLES},
                index=pd.to_datetime(hourly["time"]),
            ).rename(columns=VARIABLES)
            return df, (js["latitude"], js["longitude"], js.get("elevation", 0.0))
        except DailyLimit:
            raise
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} for {lat},{lon} {start}-{end}: {e} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed: {lat},{lon},{start}-{end}")


def add_clearsky(df, lat, lon, elevation):
    """Add CLRSKY_SFC_SW_DWN (pvlib Ineichen). Open-Meteo radiation is the mean of the
    preceding hour, so clear-sky is evaluated at timestamp - 30 min."""
    times = (df.index - pd.Timedelta(minutes=30)).tz_localize("UTC")
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=elevation)
    out = df.copy()
    out["CLRSKY_SFC_SW_DWN"] = loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy()
    return out


def write_atomic(df, path):
    """Write CSV via a temp file so an interrupted run never leaves a truncated file."""
    tmp = path + ".tmp"
    df.to_csv(tmp)
    os.replace(tmp, path)


def fetch_point(lat, lon, tag):
    cache = os.path.join(OUT_DIR, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    meta_path = os.path.join(cache, "meta.json")
    returned = tuple(json.load(open(meta_path))) if os.path.exists(meta_path) else None

    if TEST_MODE:
        chunks = [("test", TEST_DATE, TEST_DATE)]
    else:
        chunks = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(START_YEAR, END_YEAR + 1)]

    frames = []
    for label, start, end in chunks:
        cpath = os.path.join(cache, f"{label}.csv")
        if os.path.exists(cpath):
            frames.append(pd.read_csv(cpath, index_col=0, parse_dates=True))
            continue
        df, returned = fetch_range(lat, lon, start, end)
        write_atomic(df, cpath)
        json.dump(list(returned), open(meta_path, "w"))
        frames.append(df)
        print(f"  {label} done")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    out = pd.concat(frames)
    out = add_clearsky(out, *returned)
    out.index.name = "time_utc"
    return out, returned, cache


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")

    offsets = range(-HALF_WIDTH, HALF_WIDTH + 1)
    pairs = [(i, j) for i in offsets for j in offsets]
    if TEST_MODE:
        pairs = [p for p in pairs if p in [(0, 0), (0, 1)]]
    pairs.sort(key=lambda p: (p[0] ** 2 + p[1] ** 2, p))     # center-out order

    for i, j in pairs:
        lat = round(TARGET_LAT + i * STEP, 4)
        lon = round(TARGET_LON + j * STEP, 4)
        role = "TARGET" if (i == 0 and j == 0) else "neighbor"
        tag = f"point_{i:+d}_{j:+d}"
        path = os.path.join(OUT_DIR, f"{tag}.csv")
        if os.path.exists(path):
            print(f"skip (exists) {path}")
            continue
        print(f"[{role}] requesting {lat}, {lon}")
        try:
            df, (lat_r, lon_r, elev), cache = fetch_point(lat, lon, tag)
        except DailyLimit as e:
            print(f"\nDaily API limit reached ({e}). Re-run tomorrow; progress is saved.")
            sys.exit(0)
        write_atomic(df, path)
        shutil.rmtree(cache, ignore_errors=True)

        day = df["CLRSKY_SFC_SW_DWN"] >= 50
        kt = df.loc[day, "ALLSKY_SFC_SW_DWN"] / df.loc[day, "CLRSKY_SFC_SW_DWN"]
        row = pd.DataFrame([{
            "file": path, "role": role,
            "req_lat": lat, "req_lon": lon,
            "got_lat": lat_r, "got_lon": lon_r, "elev_m": elev,
            "rows": len(df),
            "missing_frac": round(df["ALLSKY_SFC_SW_DWN"].isna().mean(), 4),
            "kt_p95": round(kt.quantile(0.95), 3),
        }])
        row.to_csv(manifest_path, mode="a", header=not os.path.exists(manifest_path), index=False)
        print(row.to_string(index=False, header=False))


if __name__ == "__main__":
    main()