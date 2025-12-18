RBF-Based Data Generation for Naphtha Yield Prediction in Atmospheric Distillation Unit

Overview

This repository presents a Radial Basis Function (RBF) interpolation framework for generating a thermodynamically consistent, high-resolution dataset for naphtha yield prediction in atmospheric distillation units processing diverse crude oil blends.

This work corresponds to Phase I of the academic project:

                        ML-Driven Prediction of Naphtha Yield and Anomaly Detection for Diverse Crude Oil Blends

The scope of this repository is data generation, multivariate interpolation, and thermodynamic validation.

Problem Motivation

Rigorous process simulations in Aspen HYSYS, while thermodynamically accurate, are computationally intensive and yield sparse datasets across the high-dimensional operating space of atmospheric distillation units processing multi-crude blends. This sparsity renders the raw simulation data inadequate for training robust machine learning models. To overcome this challenge, a space-filling design of experiments is integrated with multivariate Radial Basis Function interpolation to generate a dense, physics-consistent dataset that preserves thermodynamic relationships while enabling effective downstream ML model development.

Methodology Summary
1. Design of Experiments (DOE) for Base Simulations
   
	To ensure numerical stability and reliable interpolation performance, base Aspen HYSYS simulations were generated using a Latin Hypercube Sampling (LHS) strategy rather than simple parametric sweeps. A total 		of 	160 rigorously converged simulation cases were produced to provide uniform coverage of the multidimensional operating space, reduce clustering of input points, and improve the conditioning of the RBF 				kernel matrix. This space-filling design ensures that the interpolation framework captures the underlying thermodynamic relationships across the full range of operating conditions without introducing 						numerical artifacts or extra.

2. Multivariate Input Space (12 Variables)

	RBF interpolation was performed in a 12-dimensional input–output space representing key operating and product variables of the atmospheric distillation unit. The variables include feed and operating 							conditions (e.g., temperature, flow rate, steam rate, reflux ratio), product and column performance indicators (e.g., naphtha yield, API gravity, Reid Vapor Pressure), and energy variables (e.g., condenser 			duty, overall heat duty). Each data row represents a single operating scenario in this 12-variable space, enabling comprehensive characterization of the distillation column's thermodynamic and operational 				behavior across diverse crude blend compositions.

3. Radial Basis Function (RBF) Interpolation

	Multivariate Radial Basis Function interpolation was applied to populate intermediate operating conditions between the LHS-selected simulation points, generating a dense dataset while preserving thermodynamic 		consistency. The interpolation employs multiquadric or thin-plate spline kernels based on Euclidean distance in normalized feature space, with strict enforcement of interpolation within the convex hull of 				simulated data to avoid extrapolation beyond thermodynamically meaningful bounds.

	The RBF formulation is expressed as

											f(x) = ∑ᵢ₌₁ᴺ wᵢ φ(‖x − xᵢ‖)
   
	where 	φ represents the radial basis function and wᵢ are the interpolation weights determined by satisfying the boundary conditions at all N simulation points.

	This approach ensures smooth, differentiable transitions between known simulation states while maintaining physical plausibility across the entire operating envelope.

4. Dataset Expansion and Final Output

	Approximately 700 interpolated points were generated per crude blend, resulting in an overall dataset expansion of approximately 14× relative to the base simulation count. The final dataset comprises 						approximately 2,400 thermodynamically consistent data points spanning the full range of operating conditions and crude blend compositions. This expanded dataset provides sufficient density and coverage for 			downstream applications including machine learning regression model training and process optimization studies, while maintaining computational tractability and physical interpretability.

Thermodynamic Validation of RBF-Interpolated Dataset

1. Purpose of Validation

	Radial Basis Function interpolation can mathematically generate smooth data; however, numerical smoothness does not guarantee physical realism. All interpolated data generated in this work were therefore 				subjected to rigorous thermodynamic validation to ensure consistency with fundamental distillation behavior, published literature ranges, and actual refinery plant operating envelopes. This validation step is 		critical to ensure that the interpolated dataset is physically meaningful and suitable for downstream machine learning, optimization, and anomaly detection tasks, preventing the propagation of non-physical 			predictions into operational decision-making frameworks.

2. Validation Data Sources

	Thermodynamic validation was performed using two independent reference datasets to establish physically acceptable bounds. Literature data were collected from atmospheric distillation design textbooks, peer-			reviewed journal articles, and university course materials, defining the theoretical operating limits for atmospheric distillation units processing mixed crude feeds. Actual plant operating data were compiled 		from industrial case studies, open-literature refinery reports, and consolidated operating envelopes, representing realistic constraints observed in commercial refineries rather than idealized simulation 				conditions. The combination of theoretical and empirical validation sources ensures that interpolated values remain both thermodynamically sound and practically implementable.

3. Interpolated Variable Set (12 Variables)

	RBF interpolation was performed in a 12-dimensional variable space representing both operating conditions and product properties of the atmospheric distillation unit. The validated variables include naphtha 			yield (vol% and kg/h), naphtha API gravity, naphtha Reid Vapor Pressure (RVP), reflux ratio, feed temperature (°C), feed flow rate (kg/h and m³/h), steam injection rate (kg/h), condenser duty (GJ/h), overall 		heat duty (GJ/h) . Each interpolated data point represents a single physically feasible operating scenario in this 12-variable space, capturing the coupled thermodynamic relationships between feed 								characteristics, column operations, and product quality specifications.

4. Validation Methodology
   
	For each of the 12 variables, the minimum and maximum values from the RBF-interpolated dataset were extracted and systematically compared against literature-reported ranges and typical plant operating limits. 		Any interpolated values falling outside physically meaningful bounds were identified and rejected to prevent the inclusion of thermodynamically inconsistent data points. Validation was conducted after 						interpolation rather than before, ensuring that RBF-generated intermediate points did not violate thermodynamic constraints introduced by the nonlinear interactions between operating variables. This post-				interpolation validation approach guarantees that the final expanded dataset maintains both mathematical smoothness and physical plausibility across the entire operating envelope.
	
	The Proof of Validation
                   [Thermodynamic Validation.xlsx](https://github.com/user-attachments/files/24232161/Thermodynamic.Validation.xlsx)

	The observed differences between the RBF-interpolated condenser duty and overall heat duty values and those reported in literature or actual plant data are primarily attributable to differences in system 				boundary definition, duty normalization, and unit representation rather than thermodynamic inconsistency. In the present work, condenser duty and overall heat duty were extracted directly from steady-state 			ASPEN HYSYS simulation outputs and expressed on a per-case, normalized energy basis suitable for multivariate interpolation, whereas literature and plant data often report aggregate heat duties that include 			upstream feed preheating, furnace firing, pump-around heat recovery, and auxiliary exchanger networks. Additionally, plant-reported duties are frequently averaged over long operating periods and influenced by 		non-ideal factors such as fouling, heat losses, control margins, and partial load operation, which are not explicitly modeled in idealized steady-state simulations. As a result, while the absolute magnitudes 		differ due to scope and reporting conventions, the interpolated condenser and overall heat duty values remain thermodynamically consistent, exhibit correct directional trends with respect to feed rate and 				operating severity, and lie within physically meaningful bounds when interpreted under consistent energy accounting assumptions.
