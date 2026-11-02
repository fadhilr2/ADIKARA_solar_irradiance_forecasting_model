"""
Phase 2: data selection. Rank the neighbor sites against the target P and keep the top 3.

Scores (the same three as the paper):
  1. Pearson correlation   (formula implemented by hand, as given)
  2. Spearman correlation  (formula implemented by hand, as given)
  3. XGBoost feature importance (gain)

Run:   python select_sites.py                 # reads data/raw, writes data/selected
       python select_sites.py data/test       # point it at another folder
Needs: pip install pandas numpy xgboost
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ---- settings you may change -------------------------------------------
RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "../data_collection/data/raw"
OUT_DIR = os.path.join(os.path.dirname(RAW_DIR.rstrip("/")), "selected")
TARGET_NAME = "+0_+0"          # the file point_+0_+0.csv is P
TOP_K = 3                      # how many neighbors to keep
LAG_HOURS = 0                  # 0 = same-hour scoring (as in the paper).
                               # 24 = score neighbors 24 h earlier (closer to a real day-ahead setting)
TRAIN_END = "2023-12-31 23:00" # scores use training years only, so the test years stay unseen
MIN_CLEAR_SKY = 50.0           # W/m2; hours below this are treated as night and dropped
# -------------------------------------------------------------------------


def pearson(x, y):
    """Corr_xy = (n*sum(xy) - sum(x)sum(y)) / (sqrt(n*sum(x^2) - (sum x)^2) * sqrt(n*sum(y^2) - (sum y)^2))"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    num = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    den = np.sqrt(n * np.sum(x ** 2) - np.sum(x) ** 2) * np.sqrt(n * np.sum(y ** 2) - np.sum(y) ** 2)
    return num / den


def spearman(x, y):
    """Corr = 1 - 6*sum(d_i^2) / (n(n^2 - 1)), where d_i is the rank difference.
    Tied values get their average rank."""
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    d = rx - ry
    n = len(rx)
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def load_all():
    allsky, clrsky = {}, {}
    files = sorted(glob.glob(os.path.join(RAW_DIR, "point_*.csv")))
    if not files:
        sys.exit(f"No point_*.csv files found in {RAW_DIR}")
    for path in files:
        name = re.search(r"point_(.+)\.csv$", os.path.basename(path)).group(1)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        allsky[name] = df["ALLSKY_SFC_SW_DWN"]
        clrsky[name] = df["CLRSKY_SFC_SW_DWN"]
    allsky, clrsky = pd.DataFrame(allsky), pd.DataFrame(clrsky)
    if TARGET_NAME not in allsky.columns:
        sys.exit(f"Target file point_{TARGET_NAME}.csv not found in {RAW_DIR}")
    return allsky, clrsky


def drop_duplicate_sites(kt):
    """Remove neighbors whose series is identical to P (same NASA cell) or to an earlier neighbor."""
    keep, dropped = [TARGET_NAME], []
    for col in kt.columns:
        if col == TARGET_NAME:
            continue
        same = [k for k in keep if np.allclose(kt[col].fillna(-1), kt[k].fillna(-1), atol=1e-9)]
        if same:
            dropped.append((col, same[0]))
        else:
            keep.append(col)
    return keep, dropped


def main():
    allsky, clrsky = load_all()
    print(f"Loaded {allsky.shape[1]} sites, {len(allsky)} hourly rows "
          f"({allsky.index.min()} to {allsky.index.max()})")

    # Clear-sky index removes the day/night cycle, so the scores reflect cloud behavior.
    kt = (allsky / clrsky.where(clrsky >= MIN_CLEAR_SKY)).clip(0, 1.5)

    keep, dropped = drop_duplicate_sites(kt)
    for col, same in dropped:
        print(f"  dropped {col}: identical to {same} (same NASA cell)")
    kt = kt[keep]

    target = kt[TARGET_NAME]
    neighbors = kt.drop(columns=TARGET_NAME).shift(LAG_HOURS)

    data = pd.concat([target.rename("target"), neighbors], axis=1)
    data = data[clrsky[TARGET_NAME] >= MIN_CLEAR_SKY].dropna()   # daylight hours only
    train = data[data.index <= pd.Timestamp(TRAIN_END)]
    if len(train) < 1000:
        sys.exit(f"Only {len(train)} training rows before {TRAIN_END}; check your date range and TRAIN_END.")
    print(f"Scoring on {len(train)} daylight hours up to {TRAIN_END} (lag = {LAG_HOURS} h)")

    y = train["target"].to_numpy()
    X = train.drop(columns="target")

    # 1 and 2: Pearson and Spearman
    pear = {c: pearson(X[c], y) for c in X.columns}
    spear = {c: spearman(X[c], y) for c in X.columns}

    # sanity check of the hand-written formulas against pandas
    chk_p = max(abs(pear[c] - X[c].corr(train["target"])) for c in X.columns)
    chk_s = max(abs(spear[c] - X[c].corr(train["target"], method="spearman")) for c in X.columns)
    print(f"Formula check vs pandas: max |diff| Pearson = {chk_p:.2e}, Spearman = {chk_s:.2e}")

    # 3: XGBoost importance (gain), normalised to sum to 1
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=1.0,
                         importance_type="gain", random_state=42, n_jobs=-1)
    model.fit(X, y)
    xgb_imp = pd.Series(model.feature_importances_, index=X.columns)
    xgb_imp = xgb_imp / xgb_imp.sum()

    table = pd.DataFrame({"pearson": pd.Series(pear), "spearman": pd.Series(spear),
                          "xgb_importance": xgb_imp})
    # rank 1 = best on each score, then average the three ranks
    ranks = pd.DataFrame({
        "r_pearson": table["pearson"].abs().rank(ascending=False),
        "r_spearman": table["spearman"].abs().rank(ascending=False),
        "r_xgb": table["xgb_importance"].rank(ascending=False),
    })
    table["mean_rank"] = ranks.mean(axis=1)
    table = table.sort_values(["mean_rank", "xgb_importance"], ascending=[True, False])
    table["selected"] = False
    table.iloc[:TOP_K, table.columns.get_loc("selected")] = True

    os.makedirs(OUT_DIR, exist_ok=True)
    table.round(4).to_csv(os.path.join(OUT_DIR, "site_ranking.csv"), index_label="site")
    print("\nRanking (best first):")
    print(table.round(4).to_string())

    chosen = [TARGET_NAME] + list(table.index[:TOP_K])
    out = pd.concat([allsky[chosen].add_prefix("ALL_"), clrsky[chosen].add_prefix("CLR_")], axis=1)
    out.to_csv(os.path.join(OUT_DIR, "selected_dataset.csv"), index_label="time_utc")
    print(f"\nKept P + top {TOP_K}: {chosen}")
    print(f"Saved {OUT_DIR}/site_ranking.csv and {OUT_DIR}/selected_dataset.csv")


if __name__ == "__main__":
    main()