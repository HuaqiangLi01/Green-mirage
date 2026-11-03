
# -*- coding: utf-8 -*-
"""
Multi-model machine learning reconstruction of global GPP (1982–2023)
=====================================================================

Models:
    - Random Forest (RF)
    - Extra Trees (ET)
    - XGBoost
    - LightGBM

Workflow:
    1. Load and preprocess FLUXNET-based training data.
    2. Train and evaluate four machine learning models.
    3. Reconstruct monthly global GPP for 1982–2023.
    4. Apply a vegetation land-cover mask.
    5. Aggregate monthly GPP into annual GPP.

Missing NDVI values are handled using two predictors:
    - ndvi_filled: NDVI with missing values replaced by zero.
    - ndvi_isnan: Binary indicator of missing NDVI.

Outputs:
    ML/<ModelName>/month/GPP_YYYYMM.tif
    ML/<ModelName>/year/GPP_YYYY.tif

Note:
    Update DATA_ROOT to match the local data directory.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import rasterio

from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# 0. General settings and directory configuration
# ============================================================

rcParams["font.family"] = "Arial"
rcParams["font.size"] = 13
rcParams["axes.unicode_minus"] = False

# Root directory containing all input datasets.
# Replace this example path with the actual local directory.

DATA_ROOT = r"D:\ResearchData\GPP_Project"

# Output directory for machine learning predictions.
base_ml_dir = os.path.join(DATA_ROOT, "ML")
os.makedirs(base_ml_dir, exist_ok=True)


# ============================================================
# 1. Load and preprocess training data
# ============================================================

# Input: FLUXNET-based GPP training dataset (Excel).
train_fp = os.path.join(
    DATA_ROOT,
    "training",
    "GPP_global_train2.xlsx"
)

df = pd.read_excel(train_fp)

# Remove GPP observations outside the selected range.
df = df[(df["GPP"] > 1) & (df["GPP"] <= 500)]

# NDVI is allowed to contain missing values.
subset_no_ndvi = [
    "pet", "pre", "tmp", "vap", "vpd",
    "sw", "ssr", "ca", "GPP"
]

df2 = df.dropna(subset=subset_no_ndvi).copy()

# Create an NDVI missingness indicator and fill missing NDVI with zero.
df2["ndvi_isnan"] = df2["ndvi"].isna().astype(int)
df2["ndvi_filled"] = df2["ndvi"].fillna(0)

# Predictor variables used in all four models.
input_vars = [
    "pet", "pre", "tmp", "vap", "vpd",
    "sw", "ssr", "ndvi_filled", "ndvi_isnan", "ca"
]

X = df2[input_vars].values
y = df2["GPP"].values

print("Number of valid training observations:", len(df2))


# ============================================================
# 2. Training and testing data partition
# ============================================================

# All models use the same random 70%/30% partition.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ============================================================
# 3. Initialize machine learning models
# ============================================================

# Fixed hyperparameters; no additional hyperparameter tuning.
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}


# ============================================================
# 4. Train and evaluate all models
# ============================================================

trained_models = {}

print("\n========== Model Training ==========")

for name, model in models.items():

    print(f"\n=== Training {name} ===")

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    R2 = r2_score(y_test, y_pred_test)

    RMSE = np.sqrt(
        mean_squared_error(y_test, y_pred_test)
    )

    print(f"{name} Test R²   = {R2:.4f}")
    print(f"{name} Test RMSE = {RMSE:.4f}")

    trained_models[name] = model


# ============================================================
# 5. Input raster directory configuration
# ============================================================

# Monthly raster datasets:
#
# climate/
#   pet/month/cru_pet_YYYY_MM.tif
#   pre/month/cru_pre_YYYY_MM.tif
#   tmp/month/tmp_YYYYMM.tif
#   vap/month/vap_YYYYMM.tif
#   vpd/month/vpd_YYYYMM.tif
#
# soil/
#   sw/month/sw_YYYYMM.tif
#
# radiation/
#   ssr/month/ssr_YYYYMM.tif
#
# vegetation/
#   ndvi/month/ndvi_erl_YYYYMM.tif
#   MCD12Q1_2001_vegetatedLands.tif
#
# atmospheric/
#   co2_concentration.xlsx

paths = {
    "pet": os.path.join(
        DATA_ROOT, "climate", "pet", "month",
        "cru_pet_{year}_{month}.tif"
    ),

    "pre": os.path.join(
        DATA_ROOT, "climate", "pre", "month",
        "cru_pre_{year}_{month}.tif"
    ),

    "tmp": os.path.join(
        DATA_ROOT, "climate", "tmp", "month",
        "tmp_{year}{month}.tif"
    ),

    "vap": os.path.join(
        DATA_ROOT, "climate", "vap", "month",
        "vap_{year}{month}.tif"
    ),

    "vpd": os.path.join(
        DATA_ROOT, "climate", "vpd", "month",
        "vpd_{year}{month}.tif"
    ),

    "sw": os.path.join(
        DATA_ROOT, "soil", "sw", "month",
        "sw_{year}{month}.tif"
    ),

    "ssr": os.path.join(
        DATA_ROOT, "radiation", "ssr", "month",
        "ssr_{year}{month}.tif"
    ),

    "ndvi": os.path.join(
        DATA_ROOT, "vegetation", "ndvi", "month",
        "ndvi_erl_{year}{month}.tif"
    )
}


# ============================================================
# 6. Load vegetation mask
# ============================================================

# Land-cover classes 1–14 are treated as vegetated lands.
mask_fp = os.path.join(
    DATA_ROOT,
    "vegetation",
    "MCD12Q1_2001_vegetatedLands.tif"
)

with rasterio.open(mask_fp) as msrc:
    mask = msrc.read(1)
    valid_mask = (mask >= 1) & (mask <= 14)


# ============================================================
# 7. Load atmospheric CO2 concentration
# ============================================================

# The Excel file contains monthly CO2 concentrations.
# Required columns: year, month, average.
# Required worksheet: monthly.

co2_fp = os.path.join(
    DATA_ROOT,
    "atmospheric",
    "co2_concentration.xlsx"
)

co2 = pd.read_excel(
    co2_fp,
    sheet_name="monthly"
)


# ============================================================
# 8. Raster reading utility
# ============================================================

def load_raster(fp):
    """
    Read the first band of a GeoTIFF raster.

    Parameters
    ----------
    fp : str
        Input raster file path.

    Returns
    -------
    array : numpy.ndarray
        Raster values.
    metadata : dict
        Raster metadata.
    """

    with rasterio.open(fp) as src:
        return src.read(1), src.meta.copy()


# ============================================================
# 9. Reconstruction period
# ============================================================

years = range(1982, 2024)


# ============================================================
# 10. Monthly GPP reconstruction
# ============================================================

for model_name, model in trained_models.items():

    print(
        f"\n=============== Predicting with "
        f"{model_name} ==============="
    )

    # Model-specific monthly and annual output directories.
    out_month_dir = os.path.join(
        base_ml_dir, model_name, "month"
    )

    out_year_dir = os.path.join(
        base_ml_dir, model_name, "year"
    )

    os.makedirs(out_month_dir, exist_ok=True)
    os.makedirs(out_year_dir, exist_ok=True)

    for year in years:

        for m in range(1, 13):

            month = f"{m:02d}"

            print(
                f"  -> {model_name}: "
                f"Predicting GPP {year}-{month}"
            )

            rasters = []
            meta = None
            skip = False

            # -----------------------------------------------
            # 10.1 Load eight monthly predictor rasters
            # -----------------------------------------------

            for var in [
                "pet", "pre", "tmp", "vap",
                "vpd", "sw", "ssr", "ndvi"
            ]:

                fp = paths[var].format(
                    year=year,
                    month=month
                )

                if not os.path.exists(fp):

                    print(
                        "WARNING: Missing input raster, "
                        "skipping month:",
                        fp
                    )

                    skip = True
                    break

                arr, meta_tmp = load_raster(fp)

                rasters.append(arr)

                if meta is None:
                    meta = meta_tmp

            if skip:
                continue

            H, W = rasters[0].shape

            # -----------------------------------------------
            # 10.2 Handle missing NDVI values
            # -----------------------------------------------

            ndvi = rasters[-1]

            ndvi_isnan = np.isnan(ndvi).astype(int)

            ndvi_filled = np.nan_to_num(
                ndvi,
                nan=0.0
            )

            # -----------------------------------------------
            # 10.3 Retrieve monthly atmospheric CO2
            # -----------------------------------------------

            try:

                ca_val = float(
                    co2.loc[
                        (co2["year"] == year)
                        & (co2["month"] == m),
                        "average"
                    ]
                )

            except:

                print(
                    f"WARNING: Missing CO2 for "
                    f"{year}-{month}; using NaN."
                )

                ca_val = np.nan

            ca_arr = np.full(
                (H, W),
                ca_val
            )

            # -----------------------------------------------
            # 10.4 Construct predictor matrix
            # -----------------------------------------------

            X_pred = np.column_stack([
                rasters[0].ravel(),       # PET
                rasters[1].ravel(),       # Precipitation
                rasters[2].ravel(),       # Temperature
                rasters[3].ravel(),       # Vapour pressure
                rasters[4].ravel(),       # VPD
                rasters[5].ravel(),       # Soil water
                rasters[6].ravel(),       # Solar radiation
                ndvi_filled.ravel(),      # Filled NDVI
                ndvi_isnan.ravel(),       # NDVI missing indicator
                ca_arr.ravel()           # Atmospheric CO2
            ])

            # -----------------------------------------------
            # 10.5 Predict monthly GPP
            # -----------------------------------------------

            gpp_pred = model.predict(
                X_pred
            ).reshape(H, W)

            # -----------------------------------------------
            # 10.6 Apply vegetation mask
            # -----------------------------------------------

            gpp_pred[~valid_mask] = np.nan

            # -----------------------------------------------
            # 10.7 Export monthly GPP raster
            # -----------------------------------------------

            out_fp = os.path.join(
                out_month_dir,
                f"GPP_{year}{month}.tif"
            )

            meta.update(
                count=1,
                dtype="float32"
            )

            with rasterio.open(
                out_fp, "w", **meta
            ) as dst:

                dst.write(
                    gpp_pred.astype("float32"),
                    1
                )

            print("     Saved:", out_fp)

    # ========================================================
    # 11. Aggregate monthly GPP to annual GPP
    # ========================================================

    print(
        f"\n=== Annual GPP aggregation: "
        f"{model_name} ==="
    )

    for year in years:

        year_sum = None
        meta_year = None
        has_data = False

        for m in range(1, 13):

            month = f"{m:02d}"

            fp_m = os.path.join(
                out_month_dir,
                f"GPP_{year}{month}.tif"
            )

            # Skip missing monthly predictions.
            if not os.path.exists(fp_m):
                continue

            with rasterio.open(fp_m) as src:

                arr_m = src.read(1)
                meta_year = src.meta.copy()

            if year_sum is None:

                year_sum = arr_m.astype(
                    "float32"
                )

            else:

                year_sum += arr_m.astype(
                    "float32"
                )

            has_data = True

        if not has_data:

            print(
                f"WARNING: No monthly GPP data "
                f"for {model_name}, {year}. "
                f"Skipping annual output."
            )

            continue

        # -----------------------------------------------
        # 11.1 Export annual cumulative GPP
        # -----------------------------------------------

        out_year_fp = os.path.join(
            out_year_dir,
            f"GPP_{year}.tif"
        )

        meta_year.update(
            count=1,
            dtype="float32"
        )

        with rasterio.open(
            out_year_fp, "w", **meta_year
        ) as dst:

            dst.write(
                year_sum.astype("float32"),
                1
            )

        print(
            "Annual GPP saved:",
            out_year_fp
        )


print(
    "\n======= Monthly and annual GPP "
    "reconstruction completed ======="
)
