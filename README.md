

---

# RBF-Based Data Generation for Naphtha Yield Prediction in an Atmospheric Distillation Unit

## Overview

This repository contains a **Radial Basis Function (RBF)–based multivariate interpolation framework** developed to generate a **high-resolution, physically realistic dataset** for naphtha yield prediction in an **atmospheric distillation unit (ADU)** processing diverse crude oil blends.

The dataset is generated from **rigorously converged Aspen HYSYS steady-state simulations** and expanded using RBF interpolation to populate intermediate operating conditions within the feasible operating envelope.

This work represents **Phase I (Data Generation & Pre-processing)** of the academic project:

> **ML-Driven Prediction of Naphtha Yield and Anomaly Detection for Diverse Crude Oil Blends**

The repository is intentionally focused on **data generation and reproducibility**, not on ML model training or optimization.

---

## Problem Statement

Atmospheric distillation units operate in a **high-dimensional, strongly coupled thermodynamic space**.
Although Aspen HYSYS simulations are reliable, they are:

* Computationally expensive
* Sparse across the operating envelope
* Unsuitable, in raw form, for data-driven modeling

Sparse simulation datasets lead to **poor generalization and unstable ML models**. Simply increasing the number of simulations is impractical.

This repository addresses that limitation by using **RBF interpolation** to generate a **dense, smooth, and continuous dataset** that remains constrained within the simulated operating space.

---

## Scope of This Repository

This repository covers:

* Multivariate data generation from Aspen HYSYS simulation outputs
* RBF-based interpolation in a 12-dimensional operating space
* Creation of a dense, ML-ready dataset suitable for regression, surrogate modeling, and sensitivity studies

Out of scope:

* Machine learning model development
* Optimization studies
* Real-time plant deployment

Those components are intentionally separated to keep this repository focused and auditable.

---

## Dataset Description

* **Base data source**: Aspen HYSYS steady-state atmospheric distillation simulations
* **Number of Aspen cases**: 160 converged operating points
* **Number of interpolated cases**: ~2,400 total data points
* **Dimensionality**: 12 continuous variables

Each row represents **one physically feasible operating scenario** of the atmospheric distillation column.

### Variable Categories

The dataset includes variables from the following categories:

* **Feed and operating conditions**
  (e.g., feed temperature, flow rate, steam injection rate, reflux ratio)

* **Product properties**
  (e.g., naphtha yield, API gravity, Reid Vapor Pressure)

* **Energy and performance indicators**
  (e.g., condenser duty, overall heat duty)

All variables are continuous and suitable for regression-based ML workflows.

---

## Why Radial Basis Function (RBF) Interpolation?

RBF interpolation was selected because it:

* Handles **multivariate, nonlinear relationships** effectively
* Produces **smooth and differentiable response surfaces**
* Avoids discontinuities common in grid-based interpolation
* Is well-suited for **physics-based surrogate data generation**

Most importantly, interpolation is **restricted to the convex hull of the simulated data**, ensuring no uncontrolled extrapolation beyond the modeled operating envelope.

---

## Intended Use Cases

This dataset is suitable for:

* Machine learning–based naphtha yield prediction
* Surrogate modeling for atmospheric distillation units
* Sensitivity analysis and feature importance studies
* Process digitalization and offline analytics
* Academic research and benchmarking of regression models

If you plan to use this data for **plant decision-making**, independent validation against site-specific operating data is mandatory.


---

## Reproducibility Notes

* The interpolation code is deterministic when a fixed random seed is used
* All interpolated data are generated strictly **within simulated bounds**
* No black-box ML models are used at this stage

This design choice ensures transparency and traceability of the generated data.

---

## Disclaimer

This dataset is generated from **steady-state simulation data**, not from live plant measurements.
While care has been taken to maintain physical realism, the data **should not be interpreted as actual plant operating data**.

Use at your own risk for any application beyond academic research and offline modeling.

---

## License

This repository is intended for **academic and research use**.
Commercial use requires explicit permission from the author.

---


