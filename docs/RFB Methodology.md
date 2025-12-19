## Radial Basis Function (RBF) Interpolation Methodology

### Objective
This workflow generates a dense, thermodynamically consistent dataset from sparse
ASPEN HYSYS atmospheric distillation simulations using **multivariate Radial Basis
Function (RBF) interpolation**.  
The purpose is **data generation**, not machine learning prediction, to enable
downstream ML-based naphtha yield prediction and anomaly detection.

---

### Key Constraints (Strictly Enforced)

- Multivariate interpolation (**not** a supervised ML model)
- No Random Forest, XGBoost, Neural Networks, or regression models
- No extrapolation beyond simulated operating envelopes
- One independent RBF interpolator per variable
- Blend-wise interpolation to preserve crude-specific behavior
- Reproducible results via fixed random seeds

Libraries used:
- `pandas`, `numpy`
- `scipy.interpolate.RBFInterpolator`
- `scikit-learn`
- `joblib`

---

### Input Data

- Input file: `synthetic_base_data_labeled.csv`
- Source: ASPEN HYSYS atmospheric distillation simulations
- Grouping column: `Blend_ID` (e.g., Blend_1 to Blend_4)

#### Continuous Variables Interpolated (12)

```

Feed_Temp_C
Feed_Flow_kg_h
Feed_Flow_m3_h
Steam_Rate_kg_h
Reflux_Ratio
Naphtha_Yield_%
Naphtha_Yield_kg_h
Naphtha_API
Naphtha_RVP_bar
QLine_Duty_J_h
Condenser_Duty_J_h
Overall_Heat_Duty_J_h

```

---

### Blend-wise Processing Strategy

- Each `Blend_ID` is processed independently
- Target population per blend: **≥ 700 records**
- If a blend already exceeds the target size, interpolation is skipped

This avoids mixing crude-specific thermodynamic behavior across blends.

---

### Scaling and Numerical Conditioning

For each blend:
- Continuous variables are scaled using `StandardScaler`
- Scaling is performed **per blend**, not globally

Benefits:
- Improved numerical conditioning of RBF kernels
- Stable distance calculations in high-dimensional space
- Reduced interpolation artifacts

---

### Convex Hull Sampling (No Extrapolation)

New samples are generated **strictly inside** the convex hull of the original
simulation points using **barycentric (convex-combination) sampling**.

Key properties:
- New points are linear convex combinations of existing points
- All generated samples lie within the original operating envelope
- Extrapolation is mathematically impossible by construction

Sampling is performed in **scaled feature space**.

---

### RBF Construction

For each blend:

- One `RBFInterpolator` is fitted **per variable**
- Inputs: scaled operating points
- Targets: corresponding scaled variable values

Configuration:
- Kernel: `thin_plate_spline` (default)
- Regularization (smoothing): `1e-5`
- Purpose of smoothing:
  - Avoid ill-conditioned kernel matrices
  - Improve numerical robustness in dense regions

---

### Interpolation and Reconstruction

Workflow:
1. Generate new interior points in scaled space
2. Evaluate each RBF interpolator at new locations
3. Reconstruct full multivariate samples
4. Inverse-transform results to original engineering units
5. Assign metadata:
   - `Blend_ID` retained
   - `Source = RBF_INTERPOLATED`

Original simulation points are labeled:
```

Source = ASPEN_HYSYS

```

---

### Reproducibility

- Global random seed is fixed
- Each blend uses a deterministic, derived RNG
- Parallel execution (`joblib`) does not affect reproducibility

---

### Output

- Output file: `Yield_RBF_Populated_per_blend.csv`
- Contains:
  - Original ASPEN simulation data
  - RBF-interpolated interior samples
- Guaranteed:
  - ≥ target records per blend
  - No extrapolated operating points

---

### Rationale for Using RBF Interpolation

- Suitable for sparse, high-dimensional simulation data
- Preserves smooth local behavior
- Avoids linearity assumptions
- Significantly reduces the need for expensive ASPEN simulations

This method enables efficient exploration of intermediate operating conditions
while maintaining thermodynamic consistency within defined bounds.

---


