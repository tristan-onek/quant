# Diebold-Li Model Yield Curve Forecast

"""
Quick manual test: Extending Diebold–Li yield-curve forecasts with ML

What this does...

- Get US Treasury yield data across different maturities
- Fits the Diebold-Li model each month to extract the three factors (level, slope, curvature)
- Finds the best lambda parameter through a simple grid search (minimizing fit errors)
- Creates a baseline forecast using simple AR(1) models on those factors
- Then tries a GradientBoosting approach that directly predicts each maturity's yield
  using the lagged DL factors (keeping it simple - no macro variables or fancy stuff)
- Tests both methods on rolling windows and compares RMSE performance

Data sources
-----------
By default, pulls data from FRED (needs internet). If that's not working or you want 
to use your own data, just drop a CSV called 'yields.csv' in the same folder with:
- A 'DATE' column
- Yield columns named like: DGS3MO, DGS6MO, DGS1, DGS2, etc.
- Values should be in percent (the script handles the rest)

Getting started
--------------
pip install pandas numpy scikit-learn pandas_datareader
python dl_ml_quicktest.py

Caveats & warnings
-----------
Lambda search range is 0.3 to 1.5 (where maturity is in years). I kept error 
handling minimal since this is meant for quick experimentation rather 
than production use. Problems left as excercises to end user, not financial advice, etc etc
"""

import warnings
warnings.filterwarnings("ignore") # we dont need that lol

import math
import numpy as np
import pandas as pd

# scikit-learn only; no statsmodels required
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Optional: FRED
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False


# ---------------------------
# Data utilities
# ---------------------------

FRED_CODES = {
    "DGS3MO": 0.25,  # years
    "DGS6MO": 0.50,
    "DGS1":   1.00,
    "DGS2":   2.00,
    "DGS3":   3.00,
    "DGS5":   5.00,
    "DGS7":   7.00,
    "DGS10": 10.00,
    "DGS20": 20.00,
    "DGS30": 30.00,
}

def load_yields_monthly(prefer_fred=True, start="1990-01-01", local_csv="yields.csv"):
    """
    Returns:
      df (pd.DataFrame): monthly, end-of-month, columns subset of FRED_CODES keys, pct levels
      maturities (np.ndarray): maturities (years) aligned to df columns
    """
    if prefer_fred and HAS_PDR:
        df = pdr.DataReader(list(FRED_CODES.keys()), "fred", start=start)
        # Resample daily->monthly, end-of-month last obs
        df = df.resample("M").last()
    else:
        df = pd.read_csv(local_csv, parse_dates=["DATE"])
        df = df.set_index("DATE")
        # If daily, coerce to monthly last obs
        if df.index.inferred_type != "datetime64":
            df.index = pd.to_datetime(df.index)
        df = df.resample("M").last()
        # Keep only known columns present
        keep = [c for c in FRED_CODES.keys() if c in df.columns]
        df = df[keep]

    # Drop columns with almost all NaN or too short
    good_cols = [c for c in df.columns if df[c].notna().sum() > 120]  # ≥10 years of data
    df = df[good_cols]

    # Drop rows with any NaN (keeps the intersection window)
    df = df.dropna(how="any")

    maturities = np.array([FRED_CODES[c] for c in df.columns], dtype=float)
    return df, maturities


# ---------------------------
# Diebold–Li core
# ---------------------------

def dl_loadings(maturities_years, lam):
    """
    Build the DL loading matrix X (M x 3) for given maturities and lambda.
    Loadings:
      L1 = 1
      L2 = (1 - e^{-λ m})/(λ m)
      L3 = L2 - e^{-λ m}
    """
    m = np.asarray(maturities_years, dtype=float).reshape(-1, 1)
    # Avoid division warning at m=0 (we don't have m=0 anyway)
    exp_term = np.exp(-lam * m)
    L1 = np.ones_like(m)
    L2 = (1 - exp_term) / (lam * m)
    L3 = L2 - exp_term
    X = np.hstack([L1, L2, L3])  # M x 3
    return X


def fit_factors_cross_section(yields_vec, X):
    """
    OLS per date: beta = argmin ||y - X beta||^2
    yields_vec: (M,)
    X: (M x 3)
    returns beta: (3,)
    """
    # Solve via least squares
    beta, *_ = np.linalg.lstsq(X, yields_vec, rcond=None)
    return beta


def estimate_factors_over_time(Y, X):
    """
    Y: (T x M) matrix of yields
    X: (M x 3) DL loadings
    Returns:
      F: (T x 3) [level, slope, curvature]
      SSE: (T,) squared fit error per date
    """
    T, M = Y.shape
    F = np.zeros((T, 3))
    sse = np.zeros(T)
    XtX_inv_Xt = None  # Not used; simple lstsq per row for clarity

    for t in range(T):
        beta = fit_factors_cross_section(Y[t, :], X)
        F[t, :] = beta
        resid = Y[t, :] - X @ beta
        sse[t] = (resid ** 2).sum()
    return F, sse


def choose_lambda(Y_train, maturities, lam_grid):
    """
    Pick λ minimizing average cross-sectional SSE on train window.
    Y_train: (T_train x M)
    """
    best = None
    best_sse = np.inf
    for lam in lam_grid:
        X = dl_loadings(maturities, lam)
        _, sse = estimate_factors_over_time(Y_train, X)
        avg = sse.mean()
        if avg < best_sse:
            best_sse = avg
            best = lam
    return best


# ---------------------------
# Forecasting pieces
# ---------------------------

def ar1_fit(y):
    """
    Simple AR(1): y_t = c + phi*y_{t-1} + eps
    Returns (c, phi)
    """
    y_lag = y[:-1].reshape(-1, 1)
    y_curr = y[1:]
    reg = LinearRegression().fit(y_lag, y_curr)
    phi = float(reg.coef_.ravel()[0])
    c = float(reg.intercept_.ravel()[0])
    return c, phi


def ar1_forecast_next(y_hist):
    """
    y_hist: array-like, time-ordered
    """
    c, phi = ar1_fit(np.asarray(y_hist))
    return c + phi * y_hist[-1]


def make_factor_lags(F, k_lags=6):
    """
    Build features X_t = [F_{t-1},...,F_{t-k}] flattened (3*k dims).
    Returns:
      X (T x 3k), valid from t = k onward
    """
    T, K = F.shape
    cols = []
    for lag in range(1, k_lags + 1):
        cols.append(np.roll(F, lag, axis=0))
    X = np.concatenate(cols, axis=1)
    X[:k_lags, :] = np.nan
    return X


# ---------------------------
# Rolling evaluation
# ---------------------------

def rolling_eval(Y, maturities, train_min=120, test_horizon=None, k_lags=6, random_state=7):
    """
    Y: (T x M) monthly yields
    Returns:
      dict with RMSE tables for DL-AR1 and ML, and chosen λ
    """
    T, M = Y.shape
    if test_horizon is None:
        test_horizon = T - train_min

    # Lambda grid (years-based)
    lam_grid = np.linspace(0.3, 1.5, 25)

    # Split
    train_end0 = train_min  # first forecast made for t = train_end0 -> predict t+1
    # Choose λ on initial train window
    lam = choose_lambda(Y[:train_end0, :], maturities, lam_grid)
    X_load = dl_loadings(maturities, lam)

    # Estimate initial factors to start lag feature construction
    F_all, _ = estimate_factors_over_time(Y, X_load)  # Full-period factors
    Xlags_all = make_factor_lags(F_all, k_lags=k_lags)  # factor-lag features

    # Storage
    preds_dl = np.full((T, M), np.nan)
    preds_ml = np.full((T, M), np.nan)

    for t in range(train_end0, T - 1):
        # Recompute λ occasionally? Quick test: keep fixed after initial selection.

        # ---- Baseline DL(AR1) forecast of factors, then reconstruct yields ----
        # AR(1) on each factor up to time t
        f_next = np.zeros(3)
        for j in range(3):
            f_hist = F_all[:t+1, j]  # up to t
            f_next[j] = ar1_forecast_next(f_hist)
        # Reconstruct yields at t+1
        yhat_dl = X_load @ f_next
        preds_dl[t+1, :] = yhat_dl

        # ---- ML: predict each maturity yield from lagged factors ----
        # Build train set from [k_lags .. t]
        Xtrain = Xlags_all[k_lags:t+1, :]
        Ytrain = Y[k_lags:t+1, :]       # align with Xtrain
        # One-step-ahead: target for t+1 uses Xlags at t+1 (which uses F up to t)
        Xtest = Xlags_all[t+1, :].reshape(1, -1)

        # Fit a small GBM per maturity (fast enough for quick test)
        for m in range(M):
            # Defensive: drop rows with NaN (shouldn't have, but safe)
            mask = np.isfinite(Xtrain).all(1) & np.isfinite(Ytrain[:, m])
            Xtr = Xtrain[mask]
            ytr = Ytrain[mask, m]
            if len(ytr) < 50:  # not enough history
                continue
            gbr = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=2,
                subsample=0.9, random_state=random_state
            )
            gbr.fit(Xtr, ytr)
            preds_ml[t+1, m] = gbr.predict(Xtest)[0]

    # Compute RMSE over out-of-sample (from first valid prediction)
    start_eval = train_end0 + 1
    y_true = Y[start_eval:, :]
    yhat_dl_eval = preds_dl[start_eval:, :]
    yhat_ml_eval = preds_ml[start_eval:, :]

    def rmse(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return math.sqrt(np.mean((a[mask] - b[mask]) ** 2))

    rmse_dl = [rmse(y_true[:, m], yhat_dl_eval[:, m]) for m in range(M)]
    rmse_ml = [rmse(y_true[:, m], yhat_ml_eval[:, m]) for m in range(M)]
    overall_dl = rmse(y_true, yhat_dl_eval)
    overall_ml = rmse(y_true, yhat_ml_eval)

    out = pd.DataFrame({
        "maturity_years": maturities,
        "RMSE_DL_AR1": rmse_dl,
        "RMSE_ML_GBM": rmse_ml
    }).sort_values("maturity_years").reset_index(drop=True)

    summary = {
        "lambda": lam,
        "overall_RMSE_DL_AR1": overall_dl,
        "overall_RMSE_ML_GBM": overall_ml,
        "per_maturity": out
    }
    return summary


# ---------------------------
# Main
# ---------------------------

def main():
    # 1) Load data (prefer FRED; fall back to local CSV)
    try:
        df, maturities = load_yields_monthly(prefer_fred=True, start="1990-01-01")
    except Exception:
        df, maturities = load_yields_monthly(prefer_fred=False, local_csv="yields.csv")

    # Align to numpy
    Y = df.values.astype(float)
    # Optional: keep a later start to avoid sparse early periods (comment out if not needed)
    # df = df[df.index >= "1995-01-01"]; Y = df.values; maturities = maturities[[df.columns.get_loc(c) for c in df.columns]]

    # 2) Run quick rolling evaluation
    res = rolling_eval(Y, maturities, train_min=120, k_lags=6)

    # 3) Print results
    print("\n==== Diebold–Li + ML Quick Test ====")
    print(f"Obs (months): {Y.shape[0]} | Maturities used: {Y.shape[1]}")
    print(f"Chosen lambda (years-based): {res['lambda']:.4f}")
    print(f"Overall RMSE — DL(AR1): {res['overall_RMSE_DL_AR1']:.4f}")
    print(f"Overall RMSE — ML(GBM from DL lags): {res['overall_RMSE_ML_GBM']:.4f}")
    print("\nPer-maturity RMSE (percent points):")
    print(res["per_maturity"].to_string(index=False))

    # Minimal sanity check: show last few actual vs predicted (ML) on 10Y if present
    try:
        ten_idx = list(df.columns).index("DGS10")
        print("\nTail check (last 10 months) — 10Y actual vs ML forecast:")
        # Recreate ML preds aligned (reuse the path from rolling_eval if needed)
        # For simplicity, just print actual and NaNs placeholder; quick test keeps core metrics.
        tail = pd.DataFrame({
            "date": df.index[-10:],
            "DGS10": df.iloc[-10:, ten_idx].values
        })
        print(tail.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()

