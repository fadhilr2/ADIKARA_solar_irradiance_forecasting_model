"""
Solar irradiance forecasting at target P: the next 24 hours, hour by hour.

THE FLOW (each step is one function below, and the script prints a banner for each)

    STEP 1  Data collection       read selected_dataset.csv and prepare the numbers
    STEP 2  Split                 cut the days into TRAIN / VALIDATION / TEST, in time order
    STEP 3  Feature engineering   turn raw hourly data into a table: one row per hour to predict
    STEP 4  Train                 learn from the TRAIN rows (climatology + LightGBM)
    STEP 5  Evaluate              predict, then score every model on VALIDATION and TEST
    STEP 6  Save                  write results and models for the dashboard

WORDS USED IN THIS FILE
    clear-sky irradiance   how much sun would arrive with NO clouds (from NASA, known in advance)
    clear-sky index        actual irradiance / clear-sky irradiance. 1.0 = clear sky, 0.2 = very cloudy.
                           We predict this instead of raw irradiance because it removes the
                           sunrise-to-sunset curve, so the model only has to learn the clouds.
    issue time             the moment the forecast is made (16:00 UTC = 00:00 WITA, every day)
    step                   which hour after the issue time we are predicting (0 to 23)
    persistence            simple guess: "tomorrow at this hour = yesterday at this hour"
    climatology            simple guess: "the average for this month and this hour"
    train / val / test     train = the model learns from it, val = used to stop training at the
                           right time, test = never touched until the end (the honest score)

Run:    set DATA_FILE and OUT_DIR below, then:  python train_models.py
Needs:  pip install pandas numpy lightgbm
"""
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================== SETTINGS ====================================
DATA_FILE = "data/selected/selected_dataset.csv"   # <- set your file path here
OUT_DIR = "data/model"                             # <- results are saved here

TARGET = "+0_+0"               # site name of P (file point_+0_+0.csv)
ISSUE_HOUR_UTC = 16            # forecast is issued at this UTC hour each day
MIN_CLEAR_SKY = 50.0           # W/m2; hours below this count as night and are ignored
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2    # the remaining 20% of days is the test set
SEED = 42
# ============================================================================

MODELS = ["persistence", "climatology", "lightgbm"]

# Newer LightGBM versions print a harmless notice about eval_set; older versions need it.
warnings.filterwarnings("ignore", message=".*eval_set.*")


def banner(n, title):
    print(f"\n{'=' * 8} STEP {n}/6: {title} {'=' * 8}")


# ============================================================================
# STEP 1. DATA COLLECTION
# ============================================================================
def step1_data_collection():
    """Read the file made by select_sites.py (P + the 3 chosen neighbors).

    Returns three tables with one column per site and one row per hour:
      allsky = actual irradiance, clr = clear-sky irradiance, kt = clear-sky index
    """
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated()]
    sites = [c[4:] for c in df.columns if c.startswith("ALL_")]
    if TARGET not in sites:
        sys.exit(f"Column ALL_{TARGET} not found in {DATA_FILE}. Columns: {list(df.columns)}")

    # make sure every hour exists (a missing hour becomes an empty value, not a skipped row)
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    allsky = pd.DataFrame({s: df[f"ALL_{s}"] for s in sites})
    clr = pd.DataFrame({s: df[f"CLR_{s}"] for s in sites})

    # clear-sky index = actual / clear-sky; night hours are left empty
    kt = (allsky / clr.where(clr >= MIN_CLEAR_SKY)).clip(0, 1.5)
    neighbors = [s for s in sites if s != TARGET]

    hrs = sorted(set(clr.index[clr[TARGET] >= MIN_CLEAR_SKY].hour))
    print(f"Loaded {allsky.shape[1]} sites, {len(allsky)} hourly rows "
          f"({allsky.index.min()} to {allsky.index.max()})")
    print(f"Neighbors (best-ranked first): {neighbors}")
    print(f"Max clear-sky irradiance at P: {clr[TARGET].max():.0f} "
          f"(should be roughly 1000 to 1100 if the unit is W/m2)")
    print(f"Daylight UTC hours at P: {hrs}  -> should be a block of about 12 hours "
          f"(for Kupang roughly 22:00 to 10:00 UTC); if not, check the time zone")
    return allsky, clr, kt, neighbors


# ============================================================================
# STEP 2. SPLIT
# ============================================================================
def step2_split(allsky):
    """Decide which forecast days belong to train, validation and test.

    The cut is by DATE, in time order (never random): the model learns from the past and is
    judged on a later period it has never seen. With 2021-2025 data this gives
    train 2021-2023, validation 2024, test 2025.

    Returns {"train": [...], "val": [...], "test": [...]}, each a list of issue-time positions.
    """
    idx = allsky.index
    positions = np.where(idx.hour == ISSUE_HOUR_UTC)[0]            # one issue time per day
    # need 7 days of history before the issue time and 24 hours of future after it
    positions = positions[(positions >= 168) & (positions + 24 <= len(idx))]
    n = len(positions)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    split = {"train": positions[:a], "val": positions[a:b], "test": positions[b:]}

    print(f"{n} forecast days in total, cut in time order:")
    for name, pos in split.items():
        if len(pos) == 0:
            sys.exit(f"The {name} split is empty; you need more days of data.")
        print(f"  {name:5s}: {idx[pos[0]].date()} to {idx[pos[-1]].date()}  ({len(pos)} days)")
    return split


# ============================================================================
# STEP 3. FEATURE ENGINEERING
# ============================================================================
def step3_feature_engineering(allsky, clr, kt, neighbors, split):
    """Build the table the models learn from: ONE ROW PER DAYLIGHT HOUR TO PREDICT.

    Each row says: "at issue time X, for the hour h after it, here is everything we knew
    (the inputs) and here is what really happened (the answer)".
    Every input comes from BEFORE the issue time, so nothing from the future leaks in.
    """
    idx = allsky.index
    ktP = kt[TARGET].to_numpy(float)
    clrP = clr[TARGET].to_numpy(float)
    allP = allsky[TARGET].to_numpy(float)
    ktN = [kt[s].to_numpy(float) for s in neighbors]

    def mean_or_nan(a, lo, hi):
        w = a[lo:hi]
        return np.nan if np.isnan(w).all() else float(np.nanmean(w))

    rows = []
    for split_name, positions in split.items():
        for p in positions:                      # p = position of the issue time
            # (a) summary inputs, the same for all 24 hours of this day:
            #     average clear-sky index over the last 1, 3 and 7 days
            base = {"P_m1": mean_or_nan(ktP, p - 24, p),
                    "P_m3": mean_or_nan(ktP, p - 72, p),
                    "P_m7": mean_or_nan(ktP, p - 168, p)}
            for k, a in enumerate(ktN, start=1):          # N1 = best-ranked neighbor
                base[f"N{k}_m1"] = mean_or_nan(a, p - 24, p)
                base[f"N{k}_m3"] = mean_or_nan(a, p - 72, p)

            # (b) hour-specific inputs, one row per hour of the next day
            for h in range(24):                           # h = hours after the issue time
                t = p + h                                 # t = the hour being predicted
                if not clrP[t] >= MIN_CLEAR_SKY:          # night: nothing to predict
                    continue
                ts = idx[t]
                ang = 2 * np.pi * ts.dayofyear / 365.25   # position in the year (season)
                row = {"split": split_name, "issue_time": idx[p], "target_time": ts,
                       "step": h, "month": ts.month,
                       "clr": clrP[t],                    # clear-sky irradiance of that hour
                       "doy_sin": np.sin(ang), "doy_cos": np.cos(ang),
                       "P_d1": ktP[t - 24],               # same hour 1 / 2 / 3 days ago
                       "P_d2": ktP[t - 48],
                       "P_d3": ktP[t - 72],
                       "actual_kt": ktP[t],               # THE ANSWER (index)
                       "actual": allP[t],                 # THE ANSWER (W/m2)
                       **base}
                for k, a in enumerate(ktN, start=1):
                    row[f"N{k}_d1"] = a[t - 24]           # neighbor, same hour yesterday
                rows.append(row)

    s = pd.DataFrame(rows)
    if s.empty:
        sys.exit("No rows were built. Check ISSUE_HOUR_UTC, MIN_CLEAR_SKY and the date range.")

    # the columns the models are allowed to look at (everything except the answers)
    features = ["step", "clr", "doy_sin", "doy_cos", "P_m1", "P_m3", "P_m7", "P_d1", "P_d2", "P_d3"]
    for k in range(1, len(neighbors) + 1):
        features += [f"N{k}_d1", f"N{k}_m1", f"N{k}_m3"]

    print(f"Built {len(s)} rows (one per daylight hour to predict) with {len(features)} input columns:")
    print(f"  {features}")
    print(s["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    return s, features


# ============================================================================
# STEP 4. TRAIN
# ============================================================================
def step4_train(s, features):
    """Learn from the TRAIN rows. Returns the climatology table and the 3 LightGBM models."""
    train = s[(s["split"] == "train") & s["actual_kt"].notna()]
    val = s[(s["split"] == "val") & s["actual_kt"].notna()]

    # 4a. Climatology baseline: average clear-sky index for each (month, hour step) in TRAIN
    clim_table = train.groupby(["month", "step"])["actual_kt"].mean()
    clim_default = train["actual_kt"].mean()
    print(f"4a. Climatology table built from {len(train)} training rows")

    # 4b. LightGBM. Three models on the same inputs:
    #     - mean model: the best single guess
    #     - q10 / q90 models: a low and a high guess, so we can draw an uncertainty band
    # The VALIDATION rows are used only to decide when to stop adding trees (early stopping).
    params = dict(n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)

    def fit_model(**extra):
        m = lgb.LGBMRegressor(**params, **extra)
        m.fit(train[features], train["actual_kt"],
              eval_set=[(val[features], val["actual_kt"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m

    print("4b. Training LightGBM (mean, 10th percentile, 90th percentile)...")
    m_mean = fit_model(objective="regression")
    m_q10 = fit_model(objective="quantile", alpha=0.1)
    m_q90 = fit_model(objective="quantile", alpha=0.9)
    print(f"    trees used: mean={m_mean.best_iteration_}, q10={m_q10.best_iteration_}, "
          f"q90={m_q90.best_iteration_}")

    imp = pd.Series(m_mean.booster_.feature_importance("gain"), index=features)
    print("    most used inputs (share of the model's gain):")
    print((imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_string())
    return (clim_table, clim_default), (m_mean, m_q10, m_q90)


# ============================================================================
# STEP 5. EVALUATE
# ============================================================================
def score(y, yhat):
    e = np.asarray(yhat, float) - np.asarray(y, float)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),    # typical error, big misses count extra
            "mae": float(np.mean(np.abs(e))),           # average error
            "bias": float(np.mean(e))}                  # positive = forecasts too sunny


def breakdown(d, key):
    out = []
    for k, g in d.groupby(key):
        row = {key: k, "n": len(g)}
        for m in MODELS:
            row[f"rmse_{m}"] = score(g["actual"], g[m])["rmse"]
        out.append(row)
    return pd.DataFrame(out)


def step5_evaluate(s, features, clim, models):
    """Predict every row with every model, then score on VALIDATION and TEST."""
    clim_table, clim_default = clim
    m_mean, m_q10, m_q90 = models

    # 5a. Predictions. Models predict the clear-sky index; multiply by the hour's clear-sky
    #     irradiance to get W/m2, the same unit as the real measurements.
    key = pd.MultiIndex.from_arrays([s["month"], s["step"]])
    clim_kt = clim_table.reindex(key).to_numpy()
    clim_kt = np.where(np.isnan(clim_kt), clim_default, clim_kt)
    X = s[features]
    kt_mean = np.clip(m_mean.predict(X), 0, 1.5)
    kt_q10 = np.clip(m_q10.predict(X), 0, 1.5)
    kt_q90 = np.clip(m_q90.predict(X), 0, 1.5)
    s["persistence"] = s["P_d1"] * s["clr"]
    s["climatology"] = clim_kt * s["clr"]
    s["lightgbm"] = kt_mean * s["clr"]
    s["q10"] = np.minimum(kt_q10, kt_q90) * s["clr"]
    s["q90"] = np.maximum(kt_q10, kt_q90) * s["clr"]

    # 5b. Score. Every model is scored on the SAME rows, daylight hours only.
    ev = s[s["actual"].notna() & s["P_d1"].notna()].copy()
    print(f"Scoring on {len(ev)} of {len(s)} rows (rows with a missing answer or a missing "
          f"persistence value are dropped for all models)")
    recs = []
    for sp in ["val", "test"]:
        d = ev[ev["split"] == sp]
        r_pers = score(d["actual"], d["persistence"])["rmse"]
        r_clim = score(d["actual"], d["climatology"])["rmse"]
        for m in MODELS:
            r = score(d["actual"], d[m])
            r.update(split=sp, model=m, n=len(d),
                     skill_vs_persistence=1 - r["rmse"] / r_pers,
                     skill_vs_climatology=1 - r["rmse"] / r_clim)
            recs.append(r)
    metrics = pd.DataFrame(recs)[["split", "model", "n", "rmse", "mae", "bias",
                                  "skill_vs_persistence", "skill_vs_climatology"]]
    print("\nMetrics in W/m2. skill = 1 - RMSE / RMSE of the reference; above 0 means better than it.")
    print("The TEST rows are the honest ones to report (validation was used to stop training).")
    print(metrics.round(3).to_string(index=False))

    test = ev[ev["split"] == "test"]
    cover = float(((test["actual"] >= test["q10"]) & (test["actual"] <= test["q90"])).mean())
    print(f"\nTest: share of real values inside the 10th-90th percentile band = {cover:.1%} "
          f"(ideal is about 80%)")

    by_step = breakdown(test, "step")
    by_month = breakdown(test, "month")
    print("\nTest RMSE by hour after issue time (step):")
    print(by_step.round(1).to_string(index=False))
    print("\nTest RMSE by month:")
    print(by_month.round(1).to_string(index=False))
    return ev, metrics, by_step, by_month


# ============================================================================
# STEP 6. SAVE
# ============================================================================
def step6_save(ev, metrics, by_step, by_month, features, models):
    m_mean, m_q10, m_q90 = models
    metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    by_step.to_csv(os.path.join(OUT_DIR, "metrics_by_step.csv"), index=False)
    by_month.to_csv(os.path.join(OUT_DIR, "metrics_by_month.csv"), index=False)
    cols = ["issue_time", "target_time", "step", "split", "clr", "actual",
            "persistence", "climatology", "lightgbm", "q10", "q90"]
    ev.loc[ev["split"] != "train", cols].to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    m_mean.booster_.save_model(os.path.join(OUT_DIR, "lgbm_mean.txt"))
    m_q10.booster_.save_model(os.path.join(OUT_DIR, "lgbm_q10.txt"))
    m_q90.booster_.save_model(os.path.join(OUT_DIR, "lgbm_q90.txt"))
    with open(os.path.join(OUT_DIR, "features.txt"), "w") as f:
        f.write("\n".join(features))
    print(f"Saved metrics, predictions and models to {OUT_DIR}")


# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    banner(1, "DATA COLLECTION")
    allsky, clr, kt, neighbors = step1_data_collection()

    banner(2, "SPLIT")
    split = step2_split(allsky)

    banner(3, "FEATURE ENGINEERING")
    s, features = step3_feature_engineering(allsky, clr, kt, neighbors, split)

    banner(4, "TRAIN")
    clim, models = step4_train(s, features)

    banner(5, "EVALUATE")
    ev, metrics, by_step, by_month = step5_evaluate(s, features, clim, models)

    banner(6, "SAVE")
    step6_save(ev, metrics, by_step, by_month, features, models)


if __name__ == "__main__":
    main()