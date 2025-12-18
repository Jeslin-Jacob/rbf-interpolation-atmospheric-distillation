
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator
from joblib import Parallel, delayed
import warnings

INPUT_FILENAME = "synthetic_base_data_labeled.csv"  
OUTPUT_FILENAME = "Yield_RBF_Populated_per_blend.csv"
CONT_VARS = [
    "Feed_Temp_C",
    "Feed_Flow_kg_h",
    "Feed_Flow_m3_h",
    "Steam_Rate_kg_h",
    "Reflux_Ratio",
    "Naphtha_Yield_%",
    "Naphtha_Yield_kg_h",
    "Naphtha_API",
    "Naphtha_RVP_bar",
    "QLine_Duty_J_h",
    "Condenser_Duty_J_h",
    "Overall_Heat_Duty_J_h",
]
ID_COL = "Blend_ID"  
TARGET_PER_BLEND = 700  
KERNEL = "thin_plate_spline"  
SMOOTHING = 1e-5  
RANDOM_SEED = 42
N_JOBS = -1  

rng = np.random.default_rng(RANDOM_SEED)


def load_input(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(p)
    return df


def sample_points_in_convex_hull(X, n_samples, rng, points_per_sample=None):
    
    X = np.asarray(X)
    n_points, dim = X.shape
    if n_points == 0:
        raise ValueError("Empty X in convex hull sampler.")
    if points_per_sample is None:
        k = min(n_points, dim + 1)
    else:
        k = int(min(n_points, max(1, points_per_sample)))

    
    X_new = np.empty((n_samples, dim), dtype=X.dtype)
    for i in range(n_samples):
        idx = rng.choice(n_points, size=k, replace=False)
        w = rng.exponential(scale=1.0, size=k)
        w /= w.sum()
        X_new[i] = w @ X[idx]
    return X_new


def build_rbf_interpolator(X, y, kernel=KERNEL, smoothing=SMOOTHING):
        
    try:
        rbf = RBFInterpolator(X, y, kernel=kernel, smoothing=smoothing)
    except Exception as e:
        raise RuntimeError(
            "RBFInterpolator construction failed. Try increasing SMOOTHING or switching kernel. "
            f"Original error: {e}"
        )
    return rbf


def process_blend(group_df, target_count, rng):
        n_existing = len(group_df)
    n_needed = max(0, target_count - n_existing)
    blend_id = group_df[ID_COL].iat[0] if ID_COL in group_df.columns else "Unknown"

    if n_needed == 0:
        print(f"{blend_id}: already has {n_existing} rows (>= {target_count}), skipping interpolation.")
        return pd.DataFrame(columns=group_df.columns)  # empty

    
    X_orig = group_df[CONT_VARS].astype(float).values 

        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)

        X_new_scaled = sample_points_in_convex_hull(X_scaled, n_needed, rng=rng)

        Y_scaled = X_scaled.copy()

    dim = X_scaled.shape[1]

    
    def fit_one(j):
        return build_rbf_interpolator(X_scaled, Y_scaled[:, j], kernel=KERNEL, smoothing=SMOOTHING)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interpolators = Parallel(n_jobs=N_JOBS)(
            delayed(fit_one)(j) for j in range(dim)
        )

        Y_new_scaled = np.column_stack([interp(X_new_scaled) for interp in interpolators])

        Y_new_orig = scaler.inverse_transform(Y_new_scaled)

        df_new = pd.DataFrame(Y_new_orig, columns=CONT_VARS)

        for col in group_df.columns:
        if col not in CONT_VARS:
            if col == ID_COL:
                df_new[col] = blend_id
            else:
                                df_new[col] = "Interpolated"

    
    df_new = df_new[group_df.columns]

        df_new["Source"] = "RBF_INTERPOLATED"
    return df_new


def main():
    df = load_input(INPUT_FILENAME)

    
    missing = [c for c in CONT_VARS if c not in df.columns]
    if missing:
        raise ValueError("Missing required continuous columns: " + ", ".join(missing))
    if ID_COL not in df.columns:
        raise ValueError(f"Grouping column '{ID_COL}' not found in input.")

    
    if "Source" not in df.columns:
        df["Source"] = "ASPEN_HYSYS"

    blends = sorted(df[ID_COL].unique())
    print(f"Found blends: {blends}")

    all_new_dfs = []
    
    for i, blend in enumerate(blends):
        group_df = df[df[ID_COL] == blend].reset_index(drop=True)
        print(f"Processing {blend}: existing rows = {len(group_df)}")
        
        blend_rng = np.random.default_rng(RANDOM_SEED + i + 1)
        df_new = process_blend(group_df, TARGET_PER_BLEND, blend_rng)
        print(f"{blend}: generated new interpolated rows = {len(df_new)}")
        all_new_dfs.append(df_new)

    
    df_new_all = pd.concat([d for d in all_new_dfs if not d.empty], ignore_index=True, sort=False)
    df_out = pd.concat([df, df_new_all], ignore_index=True, sort=False)

        counts = df_out.groupby(ID_COL).size().to_dict()
    for blend, cnt in counts.items():
        print(f"{blend}: final count = {cnt}")
        if cnt < TARGET_PER_BLEND:
            print(f"WARNING: {blend} has {cnt} rows, which is less than TARGET_PER_BLEND={TARGET_PER_BLEND}.")

    
    df_out.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Saved populated dataset to: {OUTPUT_FILENAME}")
    print("Done.")

    
    print("\nRationale summary (also present as comments in the script):")
    print("- Interpolation used to populate intermediate operating points between expensive ASPEN runs.")
    print("- Extrapolation avoided by sampling strictly inside convex hull (barycentric sampling).")
    print("- This reduces ASPEN simulation cost because the surrogate lets you explore many points cheaply;")
    print("  validate only a subset by running the simulator when needed.")


if __name__ == "__main__":
    main()
