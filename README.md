# Statistical-and-Machine-Learning
Course Project
# Environment-Compensated SHM on openLAB Research Bridge (No Deep Learning)

This repository contains a course-project pipeline for **Structural Health Monitoring (SHM)** that reduces **environment-driven drift** (temperature/humidity/solar) in vibration-based features using **classical regression** (OLS, Ridge, Lasso) and a **lightweight nonlinear** extension (polynomial + regularization). The output is a residual-based **Environment-Normalized Damage/Novelty Index (ENDI)** and a clean anomaly timeline.

---

**Live Demo (Streamlit):** https://YOUR-APP-NAME.streamlit.app *(replace with your actual link)*

This repository contains a course-project pipeline for **Structural Health Monitoring (SHM)** that reduces **environment-driven drift** (temperature/humidity/solar) in vibration-based features using **classical regression** (OLS, Ridge, Lasso) and a **lightweight nonlinear** extension (polynomial + regularization). The output is a residual-based **Environment-Normalized Damage/Novelty Index (ENDI)** and a clean anomaly/**alarm** timeline.

---

## Abstract

Vibration-based SHM indicators often vary with environmental conditions, which can mask or mimic damage. Using the openLAB Research Bridge dataset (accelerations with air temperature, humidity, and solar radiation), this project learns an environment-to-feature baseline and uses the resulting residuals as an environment-normalized health index. For each time window, we extract interpretable vibration features (e.g., dominant spectral peak proxies, bandpower ratios, RMS/crest factor) and model each feature \(y\) as a function of environmental variables \(x\) using Ordinary Least Square and regularized regression (Ridge and Lasso). To capture mild nonlinearity, we extend the design matrix with polynomial/interactions (or splines) and apply Ridge/Lasso to control complexity. Hyperparameters are selected via time-aware cross-validation, and performance is analyzed through bias–variance trade-offs. The main output is an Environment-Normalized Damage/Novelty Index defined from the residuals \(r=y-\hat{y}\), aggregated across sensors/features; reduced residual variance under normal periods and improved separation of abnormal periods will be quantified (e.g., lower false-alarm rate for a fixed threshold, stability across seasons). The final deliverable is a reproducible Python pipeline that fits the compensation models and generates before/after drift plots plus a residual-based health timeline for anomaly flagging.

---

**Pipeline**
1. Segment triggered acceleration data into fixed windows (e.g., 5 s).
2. Extract interpretable vibration features per sensor and window.
3. Align each window with environment values (previous record or interpolation).
4. Train regression models (OLS, Ridge, Lasso, Poly(deg=2)+Ridge) with **time-aware cross-validation**.
5. Compute residuals \(r(t)=y(t)-\hat{y}(t)\).
6. Aggregate standardized residual magnitudes into **ENDI(t)** (alarm score).

**Interpretation**
- **Low ENDI** → normal behavior after compensation (**no alarm**)
- **High ENDI** → candidate abnormal period (**alarm candidate**; investigate further)

---

## Methodology

Acceleration signals from the openLAB Research Bridge are segmented into fixed time windows, and for each sensor and window we compute interpretable vibration features in both the time and frequency domains, including RMS and crest factor as amplitude measures, and Welch-PSD–based spectral features such as dominant peak frequency proxies, bandpowers in selected frequency bands, and bandpower ratios. Each window is then aligned with environmental measurements (air temperature, humidity, and solar radiation) using timestamp-based matching to form a supervised regression dataset with predictors \(x(t)=[T(t),RH(t),S(t)]\) and responses \(y(t)\) given by the extracted features. For environment compensation, we fit an environment-to-feature baseline \(\hat{y}(t)=f(x(t))\) using OLS as a reference model and regularized regression (Ridge and Lasso) to improve stability and generalization under correlated predictors; mild nonlinear effects are captured by expanding the design matrix with polynomial and interaction terms and applying Ridge/Lasso to control complexity. Hyperparameters (e.g., regularization strength \(\alpha\) and polynomial degree) are selected using time-aware cross-validation (blocked/forward splits) to avoid temporal leakage and to reflect real SHM deployment, and model behavior is interpreted through a bias–variance perspective via generalization error trends. The final SHM output is a residual-based Environment-Normalized Damage/Novelty Index computed from \(r(t)=y(t)-\hat{y}(t)\), standardized per channel and robustly aggregated across sensors/features (median of standardized absolute residuals) to produce a single health timeline that can be thresholded for anomaly flagging; improvements over a raw-feature threshold baseline are quantified through drift/variance reduction and false-alarm comparisons, supported by a statistical validation component using bootstrapping.

---

## Data Source

This project uses the **openLAB Research Bridge** monitoring dataset (TU Dresden / IDA-KI openLAB), which provides:
- **Acceleration** measurements (including triggered recordings)
- **Environment** measurements such as air temperature, humidity, and solar radiation
- Reference period: **2024-02-01 to 2024-10-31** (undamaged baseline/reference condition)

---

## Information About the Data Source

**Dataset:** openLAB Research Bridge dataset (TU Dresden / IDA-KI openLAB; OpARA repository)

**Official links (download & info)**
- OpARA dataset record (download page): https://opara.zih.tu-dresden.de/items/6653124a-8659-40b8-817e-51250639c95b  
- openLAB project overview: https://tu-dresden.de/bu/bauingenieurwesen/imb/forschung/grossprojekte/openLAB?set_language=en

**Which folders to use (after download/unzip)**
- `01_acceleration_trigger/`  → select/upload **multiple** `acc_*.csv` trigger files  
- `02_environment/`           → select/upload **one or more** `environment_YYYY_MM.csv` files  

**Default environment columns (editable in Streamlit sidebar)**
- Temperature: `G_HTST_ENVR_EN0000_0`
- Humidity: `G_HTSH_ENVR_EN0000_0`
- Solar radiation: `G_PYRS_ENVR_EN0000_0`

---

## A List of Packages Required

Create a `requirements.txt` in the repo root with:

```txt
streamlit
numpy
pandas
matplotlib
scikit-learn

## References (IEEE)

[1] A. Jansen, M. Herbers, B. Richter, M. Walker, F. Jesse, and S. Marx, “Monitoring data of the openLAB research bridge – Part 1: Reference condition,” *Data in Brief*, vol. 60, art. no. 111624, 2025, doi: 10.1016/j.dib.2025.111624. :contentReference[oaicite:3]{index=3}

[2] TU Dresden / OpARA Repository, “Monitoring Data of the openLAB Research Bridge (2024-02-01 to 2024-10-31) and building information,” Dataset record, 2025. :contentReference[oaicite:4]{index=4}

[3] TU Dresden — Institute of Concrete Structures, “openLAB — A research bridge in Lusatia,” Project overview, 2025. :contentReference[oaicite:5]{index=5}

---

## Acknowledgment

This project was completed with limited assistance from generative AI tools (ChatGPT) as a supplementary resource, similar to a textbook or search engine. ChatGPT was used to help improve clarity of writing, organize the methodology and presentation narrative, and provide guidance on implementing standard machine-learning workflows (e.g., regression with regularization, time-aware cross-validation, and bootstrapping) in Python. All modeling choices, code, results, and interpretations were reviewed, verified, and are fully understood by the author, who assumes full responsibility for the final submitted work.

---
