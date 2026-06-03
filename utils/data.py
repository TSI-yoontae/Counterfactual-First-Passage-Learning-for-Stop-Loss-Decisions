import math
import numpy as np
import pandas as pd

EQUITY_FILES = {
    "US_SP500": {
        "daily": "US_SP500_daily.parquet",
        "id_col": "permno",
        "price_col": "prc",
        "ret_col": "ret",
        "mktcap_col": "mktcap",
        "volume_col": "volume",
    },
    "DE_DAX": {
        "daily": "DE_DAX_daily.parquet",
        "id_col": "gvkey",
        "price_col": "prc_close",
        "ret_col": "ret_local",
        "mktcap_col": "mktcap_local",
        "volume_col": "volume",
    },
    "JP_Nikkei225": {
        "daily": "JP_Nikkei_225_daily.parquet",
        "id_col": "gvkey",
        "price_col": "prc_close",
        "ret_col": "ret_local",
        "mktcap_col": "mktcap_local",
        "volume_col": "volume",
    },
    "KR_KOSPI200": {
        "daily": "KR_KOSPI_200_daily.parquet",
        "id_col": "gvkey",
        "price_col": "prc_close",
        "ret_col": "ret_local",
        "mktcap_col": "mktcap_local",
        "volume_col": "volume",
    },
}


def equal_weight_portfolio(ret, vol, assets, min_coverage=0.6):
    x = ret.reindex(columns=assets).astype(float)
    valid = x.notna().sum(axis=1)
    min_count = max(1, int(math.ceil(min_coverage * max(1, len(assets)))))
    port = x.mean(axis=1, skipna=True).where(valid >= min_count).dropna()
    if vol is not None:
        pvol = vol.reindex(columns=assets).sum(axis=1, skipna=True).reindex(port.index)
    else:
        pvol = None
    return port, pvol, x.reindex(port.index)


def trend_scanning_label(cum_path):
    y = np.asarray(cum_path, dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return 0
    logw_full = np.log1p(np.r_[0.0, y])
    best_t = 0.0
    for j in range(2, len(logw_full)):
        yy = logw_full[: j + 1]
        x = np.arange(len(yy), dtype=float)
        x = x - x.mean()
        denom = float(np.sum(x * x))
        if denom <= 0:
            continue
        beta = float(np.sum(x * (yy - yy.mean())) / denom)
        resid = yy - (yy.mean() + beta * x)
        dof = len(yy) - 2
        s2 = float(np.sum(resid * resid) / dof) if dof > 0 else 0.0
        se = math.sqrt(s2 / denom) if s2 > 0 else np.inf
        t_value = beta / se if np.isfinite(se) and se > 0 else 0.0
        if abs(t_value) > abs(best_t):
            best_t = t_value
    return int(best_t > 0.0)


def tb_label_from_path(cum_path, barrier):
    for val in np.asarray(cum_path, dtype=float):
        if np.isfinite(val) and val <= -float(barrier):
            return 0
        if np.isfinite(val) and val >= float(barrier):
            return 1
    return int(float(cum_path[-1]) > 0.0)


def future_returns_from_return_series(returns, horizon, stop_levels, stride=None):
    if stride is None:
        stride = horizon
    returns = returns.astype(float)
    idx = returns.index
    r = returns.to_numpy(dtype=float)
    rows = []
    for i in range(0, len(r) - horizon, stride):
        future = r[i + 1 : i + 1 + horizon]
        if not np.all(np.isfinite(future)):
            continue
        cum = np.cumprod(1.0 + future) - 1.0
        row = {
            "date": idx[i],
            "path_end": idx[i + horizon],
            "terminal_return": float(cum[-1]),
            "min_path_return": float(cum.min()),
            "max_path_return": float(cum.max()),
            "trend_scan_label": trend_scanning_label(cum),
        }
        for s in stop_levels:
            row[f"tb_{float(s):0.3f}"] = tb_label_from_path(cum, float(s))
        rows.append(row)
    if rows:
        return pd.DataFrame(rows).set_index("date")
    return pd.DataFrame(columns=["terminal_return", "min_path_return"])
