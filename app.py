
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="openLAB ENDI Demo (No DL)", layout="wide")
st.title("openLAB ENDI Demo WebApp (Streamlit) — No Deep Learning")
st.write("Upload openLAB trigger acceleration CSVs and environment CSVs, run env compensation (OLS/Ridge/Lasso/Poly+Ridge), and compute ENDI.")

DEFAULT_BANDS = [(0.5,2.0),(2.0,5.0),(5.0,10.0),(10.0,20.0)]

def estimate_fs_from_timestamp(ts: pd.Series) -> float:
    dt = ts.diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        return float("nan")
    med = float(dt.median())
    return 1.0/med if med > 0 else float("nan")

def welch_psd_numpy(x: np.ndarray, fs: float, nperseg: int = 2048, noverlap: int = 1024):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 128:
        win = np.hanning(max(n, 2))
        x0 = x - np.mean(x) if n > 0 else x
        X = np.fft.rfft(x0 * win[:n])
        psd = (np.abs(X) ** 2) / (fs * np.sum(win[:n] ** 2))
        freqs = np.fft.rfftfreq(n, d=1 / fs) if n > 0 else np.array([0.0])
        return freqs, psd

    if n < nperseg:
        nperseg = 2 ** int(np.floor(np.log2(n)))
        nperseg = max(nperseg, 128)
        noverlap = nperseg // 2

    step = nperseg - noverlap
    win = np.hanning(nperseg)
    scale = fs * np.sum(win ** 2)

    psds = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = seg - np.mean(seg)
        X = np.fft.rfft(seg * win)
        psds.append((np.abs(X) ** 2) / scale)

    if len(psds) == 0:
        win = np.hanning(n)
        x0 = x - np.mean(x)
        X = np.fft.rfft(x0 * win)
        psd = (np.abs(X) ** 2) / (fs * np.sum(win ** 2))
        freqs = np.fft.rfftfreq(n, d=1 / fs)
        return freqs, psd

    psd = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    return freqs, psd

def bandpower(freqs, psd, f1, f2):
    mask = (freqs >= f1) & (freqs < f2)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))

def extract_features_for_window(window_df: pd.DataFrame, acc_cols, fs: float, bands, peak_band=(0.5,20.0)):
    feats = {}
    for ch in acc_cols:
        x = window_df[ch].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        x = x - np.mean(x)
        rms = float(np.sqrt(np.mean(x**2)))
        std = float(np.std(x))
        crest = float(np.max(np.abs(x)) / (rms + 1e-12))

        freqs, psd = welch_psd_numpy(x, fs=fs)
        f_lo, f_hi = peak_band
        pmask = (freqs >= f_lo) & (freqs <= f_hi)
        pk = float(freqs[pmask][np.argmax(psd[pmask])]) if np.any(pmask) else float(freqs[np.argmax(psd)])

        feats[f"{ch}_rms"] = rms
        feats[f"{ch}_std"] = std
        feats[f"{ch}_crest"] = crest
        feats[f"{ch}_peakHz"] = pk

        for (b1,b2) in bands:
            feats[f"{ch}_bp_{b1:g}_{b2:g}"] = bandpower(freqs, psd, b1, b2)

        bp_hi = feats.get(f"{ch}_bp_2_5", np.nan)
        bp_lo = feats.get(f"{ch}_bp_0.5_2", np.nan)
        if np.isfinite(bp_hi) and np.isfinite(bp_lo):
            feats[f"{ch}_bpr_2_5_over_0.5_2"] = float(bp_hi / (bp_lo + 1e-12))

    return feats

def build_feature_table_from_uploaded_acc(acc_dfs, window_seconds: float, bands, peak_band=(0.5,20.0), max_windows_per_file=0):
    all_rows = []
    summary = []
    for fname, acc in acc_dfs:
        if "Timestamp" not in acc.columns:
            continue
        acc = acc.copy()
        acc["Timestamp"] = pd.to_datetime(acc["Timestamp"])

        all_acc_cols = [c for c in acc.columns if c != "Timestamp"]
        acc_cols = [c for c in all_acc_cols if not acc[c].isna().all()]
        fs = estimate_fs_from_timestamp(acc["Timestamp"])
        if not np.isfinite(fs):
            continue

        win_n = int(round(window_seconds * fs))
        if win_n < 64:
            continue

        starts = list(range(0, len(acc) - win_n + 1, win_n))
        if max_windows_per_file and max_windows_per_file > 0:
            starts = starts[:max_windows_per_file]

        used = 0
        for st_i in starts:
            w = acc.iloc[st_i:st_i+win_n]
            feats = extract_features_for_window(w, acc_cols=acc_cols, fs=fs, bands=bands, peak_band=peak_band)
            if len(feats) == 0:
                continue
            feats["timestamp"] = w["Timestamp"].iloc[0]
            feats["acc_file"] = fname
            feats["fs_estimated_hz"] = fs
            feats["window_seconds"] = window_seconds
            all_rows.append(feats)
            used += 1

        summary.append({"file": fname, "fs_hz": fs, "windows": used, "channels": len(acc_cols)})

    feat = pd.DataFrame(all_rows)
    if len(feat) == 0:
        return feat, pd.DataFrame(summary)
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    return feat, pd.DataFrame(summary)

def align_environment(feat_df: pd.DataFrame, env_df: pd.DataFrame, env_cols: dict, method: str):
    f = feat_df.sort_values("timestamp").reset_index(drop=True).copy()
    e = env_df.sort_values("Timestamp").reset_index(drop=True).copy()

    needed = list(env_cols.values())
    for c in ["Timestamp"] + needed:
        if c not in e.columns:
            raise ValueError(f"Environment column missing: {c}")

    if method == "previous":
        merged = pd.merge_asof(f, e, left_on="timestamp", right_on="Timestamp", direction="backward")
        for new, old in env_cols.items():
            merged[new] = merged[old]
        return merged

    if method == "interpolate":
        prev = pd.merge_asof(
            f[["timestamp"]],
            e[["Timestamp"] + needed],
            left_on="timestamp",
            right_on="Timestamp",
            direction="backward",
        ).rename(columns={"Timestamp": "prev_ts"})
        nxt = pd.merge_asof(
            f[["timestamp"]],
            e[["Timestamp"] + needed],
            left_on="timestamp",
            right_on="Timestamp",
            direction="forward",
        ).rename(columns={"Timestamp": "next_ts"})

        merged = f.copy()
        merged["prev_ts"] = prev["prev_ts"]
        merged["next_ts"] = nxt["next_ts"]

        t = merged["timestamp"].astype("int64") / 1e9
        t0 = merged["prev_ts"].astype("int64") / 1e9
        t1 = merged["next_ts"].astype("int64") / 1e9
        w = ((t - t0) / (t1 - t0)).clip(0, 1)

        for new, old in env_cols.items():
            v0 = prev[old].astype(float)
            v1 = nxt[old].astype(float)
            merged[new] = (1 - w) * v0 + w * v1

        return merged

    raise ValueError("method must be 'previous' or 'interpolate'")

def select_alpha_time_cv(X, y, model_kind="ridge", poly_degree=None, alphas=None, n_splits=6):
    if alphas is None:
        alphas = np.logspace(-4, 2, 18)
    n_splits = min(n_splits, max(2, len(y)//5))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mse = []

    for a in alphas:
        steps = [("scaler", StandardScaler())]
        if poly_degree is not None:
            steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
        if model_kind == "ridge":
            steps.append(("model", Ridge(alpha=float(a))))
        elif model_kind == "lasso":
            steps.append(("model", Lasso(alpha=float(a), max_iter=20000)))
        else:
            raise ValueError("model_kind must be ridge or lasso")

        pipe = Pipeline(steps)
        mses = []
        for tr, te in tscv.split(X):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            mses.append(mean_squared_error(y[te], pred))
        cv_mse.append(float(np.mean(mses)))

    cv_mse = np.array(cv_mse)
    best_idx = int(np.argmin(cv_mse))
    return float(alphas[best_idx]), np.array(alphas), cv_mse

def choose_best_model(X, y, poly_degree=2, n_splits=6):
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(y)//5)))

    ols = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    mse_ols = np.mean([mean_squared_error(y[te], ols.fit(X[tr], y[tr]).predict(X[te])) for tr, te in tscv.split(X)])

    a_r, alphas_r, curve_r = select_alpha_time_cv(X, y, "ridge", None, None, n_splits=n_splits)
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=a_r))])
    mse_r = float(curve_r.min())

    a_l, alphas_l, curve_l = select_alpha_time_cv(X, y, "lasso", None, None, n_splits=n_splits)
    lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=a_l, max_iter=20000))])
    mse_l = float(curve_l.min())

    a_pr, alphas_pr, curve_pr = select_alpha_time_cv(X, y, "ridge", poly_degree, None, n_splits=n_splits)
    poly_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ("model", Ridge(alpha=a_pr))
    ])
    mse_pr = float(curve_pr.min())

    candidates = [
        ("ridge", ridge, mse_r, {"alpha": a_r}),
        ("ols", ols, float(mse_ols), {}),
        ("lasso", lasso, mse_l, {"alpha": a_l}),
        ("poly_ridge", poly_ridge, mse_pr, {"alpha": a_pr, "degree": poly_degree}),
    ]
    candidates.sort(key=lambda x: (x[2], 0 if x[0] in ("ridge","ols") else 1))
    best_name, best_pipe, best_mse, meta = candidates[0]
    best_pipe.fit(X, y)
    return best_name, best_pipe, best_mse, meta, candidates

def compute_ENDI(df: pd.DataFrame, res_cols, normal_frac=0.6):
    n = len(df)
    n0 = max(10, int(n * normal_frac))
    normal_mask = np.arange(n) < n0
    Z = []
    for c in res_cols:
        r = df[c].to_numpy(dtype=float)
        sigma = np.std(r[normal_mask])
        sigma = sigma if sigma > 1e-12 else 1.0
        Z.append(np.abs(r) / sigma)
    Z = np.vstack(Z).T
    return np.median(Z, axis=1)

st.sidebar.header("Controls")
window_seconds = st.sidebar.number_input("Window size (seconds)", min_value=1.0, max_value=600.0, value=5.0, step=1.0)
align_method = st.sidebar.selectbox("Environment alignment", ["interpolate", "previous"], index=0)
poly_degree = st.sidebar.slider("Polynomial degree", 1, 3, 2)
n_splits = st.sidebar.slider("TimeSeriesSplit folds", 2, 10, 6)
normal_frac = st.sidebar.slider("Normal fraction for sigma", 0.2, 0.9, 0.6)
max_files = st.sidebar.number_input("Max trigger files (0=all)", min_value=0, max_value=5000, value=0, step=10)
max_windows_per_file = st.sidebar.number_input("Max windows per file (0=all)", min_value=0, max_value=10000, value=0, step=10)

st.sidebar.subheader("Environment columns")
temp_col = st.sidebar.text_input("Temperature", "G_HTST_ENVR_EN0000_0")
rh_col   = st.sidebar.text_input("Humidity",    "G_HTSH_ENVR_EN0000_0")
sol_col  = st.sidebar.text_input("Solar",       "G_PYRS_ENVR_EN0000_0")
env_cols = {"temp": temp_col, "rh": rh_col, "solar": sol_col}

st.subheader("Upload Data")
cA, cB = st.columns(2)
with cA:
    acc_files = st.file_uploader("Acceleration triggers (acc_*.csv) — multiple", type=["csv"], accept_multiple_files=True)
with cB:
    env_files = st.file_uploader("Environment files (environment_*.csv) — one or more", type=["csv"], accept_multiple_files=True)

run_btn = st.button("Run Pipeline ✅", type="primary")

def read_uploaded_csvs(uploaded_files):
    out = []
    for f in uploaded_files:
        out.append((f.name, pd.read_csv(f)))
    return out

if run_btn:
    if not acc_files or not env_files:
        st.error("Upload at least one acceleration file and one environment file.")
        st.stop()

    acc_dfs = read_uploaded_csvs(acc_files)
    env_dfs = read_uploaded_csvs(env_files)

    env_parts = []
    for name, e in env_dfs:
        if "Timestamp" not in e.columns:
            st.warning(f"Skipping env file without Timestamp: {name}")
            continue
        e = e.copy()
        e["Timestamp"] = pd.to_datetime(e["Timestamp"])
        e["env_file"] = name
        env_parts.append(e)

    if not env_parts:
        st.error("No valid environment tables found (need Timestamp column).")
        st.stop()

    env = pd.concat(env_parts, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)

    if max_files and max_files > 0:
        acc_dfs = acc_dfs[:max_files]

    feat, summary = build_feature_table_from_uploaded_acc(
        acc_dfs=acc_dfs,
        window_seconds=window_seconds,
        bands=DEFAULT_BANDS,
        peak_band=(0.5, 20.0),
        max_windows_per_file=max_windows_per_file
    )

    if len(feat) == 0:
        st.error("No features extracted. Check your window size and input format.")
        st.stop()

    st.success(f"Extracted {len(feat)} windows from {len(acc_dfs)} trigger files.")
    st.dataframe(summary, use_container_width=True)

    feat2 = align_environment(feat, env, env_cols=env_cols, method=align_method)
    x_cols = ["temp", "rh", "solar"]
    feat2 = feat2.replace([np.inf, -np.inf], np.nan).dropna(subset=x_cols).reset_index(drop=True)

    y_cols = [c for c in feat2.columns if c.endswith("_peakHz") or c.endswith("_bpr_2_5_over_0.5_2")]
    if len(y_cols) == 0:
        st.error("No target features found (expected *_peakHz or *_bpr_2_5_over_0.5_2).")
        st.stop()

    X_all = feat2[x_cols].to_numpy(dtype=float)
    residuals = pd.DataFrame({"timestamp": feat2["timestamp"], "acc_file": feat2["acc_file"]})
    model_manifest = {}

    for ycol in y_cols:
        y = feat2[ycol].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.all(np.isfinite(X_all), axis=1)
        if mask.sum() < 30:
            continue

        X = X_all[mask]
        yy = y[mask]

        best_name, best_pipe, best_mse, meta, candidates = choose_best_model(X, yy, poly_degree=poly_degree, n_splits=n_splits)
        yhat = np.full(len(y), np.nan)
        yhat[mask] = best_pipe.predict(X)
        r = y - yhat

        residuals[ycol] = y
        residuals[ycol + "_yhat"] = yhat
        residuals[ycol + "_r"] = r

        model_manifest[ycol] = {
            "best_model": best_name,
            "best_cv_mse": float(best_mse),
            "best_alpha": meta.get("alpha", None),
            "best_degree": meta.get("degree", None),
            "candidates": {c[0]: float(c[2]) for c in candidates},
        }

    res_cols = [c for c in residuals.columns if c.endswith("_r")]
    residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna(subset=res_cols, how="all").reset_index(drop=True)

    residuals["ENDI"] = compute_ENDI(residuals, res_cols=res_cols, normal_frac=normal_frac)

    st.subheader("Plots")
    rep = next((c for c in y_cols if c in residuals.columns), y_cols[0])

    p1, p2 = st.columns(2)
    with p1:
        fig = plt.figure()
        plt.scatter(feat2["timestamp"], feat2[rep], s=8, alpha=0.7)
        plt.title(f"Raw feature: {rep}")
        plt.xlabel("time"); plt.ylabel("feature value")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with p2:
        fig = plt.figure()
        plt.scatter(residuals["timestamp"], residuals[rep + "_r"], s=8, alpha=0.7)
        plt.title(f"Residual: {rep}")
        plt.xlabel("time"); plt.ylabel("r = y - yhat")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    fig = plt.figure()
    plt.scatter(residuals["timestamp"], residuals["ENDI"], s=10, alpha=0.8)
    plt.title("ENDI timeline")
    plt.xlabel("time"); plt.ylabel("ENDI")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Download results")
    feat_csv = feat2.to_csv(index=False).encode("utf-8")
    res_csv = residuals.to_csv(index=False).encode("utf-8")
    manifest_json = json.dumps(model_manifest, indent=2).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download feature table CSV", feat_csv, file_name="windowed_feature_table_with_env.csv")
    with d2:
        st.download_button("Download residuals + ENDI CSV", res_csv, file_name="residuals_and_ENDI.csv")
    with d3:
        st.download_button("Download model manifest JSON", manifest_json, file_name="model_manifest.json")

    st.success("Finished ✅")
