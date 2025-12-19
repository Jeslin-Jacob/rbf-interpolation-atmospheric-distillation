# validate_rbf.py
"""
Validate RBF-populated data against ASPEN HYSYS ground truth.

Outputs:
- rbf_validation_point_errors.csv  : per-ASPEN-point error details (joined with nearest RBF point)
- rbf_error_matrix.csv             : per-variable error metrics (MAE, RMSE, MAPE(%), R2, Bias)
- rbf_error_matrix_per_blend.csv   : per-blend aggregated RMSE / MAE
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial import cKDTree
import os

# ----------------------------
# CONFIG
# ----------------------------
INPUT_CSV = "Yield_RBF_Populated_per_blend.csv"   # replace path if needed
OUTPUT_ERRORS_CSV = "rbf_validation_point_errors.csv"
OUTPUT_ERROR_MATRIX = "rbf_error_matrix.csv"
OUTPUT_ERROR_MATRIX_PER_BLEND = "rbf_error_matrix_per_blend.csv"

# the 12 continuous process variables to validate/scale (must match your file)
CONTINUOUS_VARS = [
    "Feed_Temp_C",
    "Feed_Flow_kg_h",
    "Feed_Flow_m3_h",
    "Steam_Rate_kg_h",
    "Naphtha_Yield_%",
    "Naphtha_Yield_kg_h",
    "Naphtha_API",
    "Naphtha_RVP_bar",
    "Reflux_Ratio",
    "QLine_Duty_J_h",
    "Condenser_Duty_J_h",
    "Overall_Heat_Duty_J_h"
]

SOURCE_COL = "Source"        # column that marks ASPEN_HYSYS vs RBF_INTERPOLATED
BLEND_COL = "Blend_ID"

# For MAPE denominator guard (avoid divide-by-very-small)
MAPE_EPS = 1e-8

# ----------------------------
# Utilities
# ----------------------------
def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < MAPE_EPS, MAPE_EPS, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------
# Load data
# ----------------------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

# Basic checks
missing = [c for c in CONTINUOUS_VARS if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in input CSV: {missing}")

if SOURCE_COL not in df.columns:
    # If Source not present, assume all rows are ASPEN (no interpolated) — still proceed but warn
    df[SOURCE_COL] = "UNKNOWN"
    print("Warning: no 'Source' column found. All rows marked 'UNKNOWN'")

# Separate ASPEN (ground-truth) and RBF interpolated rows
asp_df = df[df[SOURCE_COL] == "ASPEN_HYSYS"].copy()
rbf_df = df[df[SOURCE_COL] == "RBF_INTERPOLATED"].copy()

if asp_df.shape[0] == 0:
    raise ValueError("No rows labeled 'ASPEN_HYSYS' found — cannot validate without ground truth ASPEN points.")

if rbf_df.shape[0] == 0:
    raise ValueError("No rows labeled 'RBF_INTERPOLATED' found — nothing to validate against.")

print(f"Total rows: {len(df):,}, ASPEN rows: {len(asp_df):,}, RBF_interpolated rows: {len(rbf_df):,}")

# ----------------------------
# Scaling
# ----------------------------
# Fit scaler on ASPEN (ground-truth) points only (so distances reflect the original operating region)
scaler = StandardScaler()
scaler.fit(asp_df[CONTINUOUS_VARS].values)   # fit using ASPEN rows

# Transform both ASPEN and RBF points for distance computations
X_asp_scaled = scaler.transform(asp_df[CONTINUOUS_VARS].values)
X_rbf_scaled = scaler.transform(rbf_df[CONTINUOUS_VARS].values)

# ----------------------------
# Nearest-neighbour matching in scaled space
# ----------------------------
# Build KDTree on RBF interpolated scaled points
tree = cKDTree(X_rbf_scaled)

# Query nearest RBF point for each ASPEN point
distances, indices = tree.query(X_asp_scaled, k=1)

# Prepare result DataFrame: join ASPEN point with its nearest RBF point
asp_index = asp_df.index.to_numpy()
rbf_index = rbf_df.index.to_numpy()

matched_rbf_indices = rbf_index[indices]   # original row indices in rbf_df corresponding to NN

# Extract matched rows
asp_matched = asp_df.reset_index(drop=True).loc[np.arange(len(asp_df))].copy()
rbf_matched = rbf_df.loc[matched_rbf_indices].reset_index(drop=True).copy()

# compute per-variable errors (predicted = rbf, actual = asp)
errors = rbf_matched[CONTINUOUS_VARS].values - asp_matched[CONTINUOUS_VARS].values
abs_errors = np.abs(errors)
sq_errors = errors ** 2

# create detailed per-point results
detailed = pd.DataFrame({
    "asp_index": asp_df.index,
    "rbf_index": matched_rbf_indices,
    "blend": asp_matched[BLEND_COL].values,
    "nn_distance_scaled": distances
})

# add actual / predicted / error columns for each continuous var
for i, var in enumerate(CONTINUOUS_VARS):
    detailed[f"{var}_asp"] = asp_matched[var].values
    detailed[f"{var}_rbf"] = rbf_matched[var].values
    detailed[f"{var}_err"] = errors[:, i]
    detailed[f"{var}_abs_err"] = abs_errors[:, i]
    detailed[f"{var}_sq_err"] = sq_errors[:, i]

# Save detailed errors
detailed.to_csv(OUTPUT_ERRORS_CSV, index=False)
print(f"Saved per-point matched errors to: {OUTPUT_ERRORS_CSV}")

# ----------------------------
# Error metrics per variable (global)
# ----------------------------
metrics = []
for i, var in enumerate(CONTINUOUS_VARS):
    y_true = asp_matched[var].values
    y_pred = rbf_matched[var].values
    mae_v = mean_absolute_error(y_true, y_pred)
    rmse_v = rmse(y_true, y_pred)
    mape_v = mape(y_true, y_pred)
    # R2 may give negative values if poor match
    try:
        r2_v = r2_score(y_true, y_pred)
    except Exception:
        r2_v = np.nan
    bias_v = np.mean(y_pred - y_true)  # mean error (prediction - truth)

    metrics.append({
        "variable": var,
        "MAE": mae_v,
        "RMSE": rmse_v,
        "MAPE_%": mape_v,
        "R2": r2_v,
        "Bias": bias_v
    })

errmat = pd.DataFrame(metrics).set_index("variable")

# Add an overall summary row (mean of metrics)
summary = {
    "variable": "MEAN_ACROSS_VARS",
    "MAE": errmat["MAE"].mean(),
    "RMSE": errmat["RMSE"].mean(),
    "MAPE_%": errmat["MAPE_%"].mean(),
    "R2": errmat["R2"].mean(),
    "Bias": errmat["Bias"].mean()
}
# Use pd.concat instead of the deprecated .append()
summary_df = pd.DataFrame([summary]).set_index("variable")
errmat = pd.concat([errmat, summary_df])

errmat.index = list(errmat.index[:-1]) + ["MEAN_ACROSS_VARS"]

errmat.to_csv(OUTPUT_ERROR_MATRIX)
print(f"Saved error matrix to: {OUTPUT_ERROR_MATRIX}")

# ----------------------------
# Per-blend aggregated metrics (RMSE and MAE)
# ----------------------------
# We already stored 'blend' in detailed; compute per-blend RMSE / MAE for each variable and overall RMSE
blends = detailed["blend"].unique()
rows = []
for b in blends:
    sub = detailed[detailed["blend"] == b]
    if sub.shape[0] == 0:
        continue
    row = {"blend": b, "n_points": len(sub)}
    # compute per-variable RMSE and MAE averaged across points (i.e., as for that blend)
    mae_vals = []
    rmse_vals = []
    for var in CONTINUOUS_VARS:
        mae_v = sub[f"{var}_abs_err"].mean()
        rmse_v = np.sqrt(sub[f"{var}_sq_err"].mean())
        row[f"{var}_MAE"] = mae_v
        row[f"{var}_RMSE"] = rmse_v
        mae_vals.append(mae_v)
        rmse_vals.append(rmse_v)
    row["mean_MAE_across_vars"] = np.mean(mae_vals)
    row["mean_RMSE_across_vars"] = np.mean(rmse_vals)
    rows.append(row)

per_blend_df = pd.DataFrame(rows).set_index("blend")
per_blend_df.to_csv(OUTPUT_ERROR_MATRIX_PER_BLEND)
print(f"Saved per-blend aggregated error matrix to: {OUTPUT_ERROR_MATRIX_PER_BLEND}")

# ----------------------------
# Print a short summary to console
# ----------------------------
pd.set_option("display.precision", 6)
print("\nPer-variable error matrix (top lines):")
print(errmat.head(15))

print("\nPer-blend summary (top lines):")
print(per_blend_df.head())

print("\nValidation completed.")
