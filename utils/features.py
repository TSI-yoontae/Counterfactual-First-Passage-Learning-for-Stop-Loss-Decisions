import numpy as np
import pandas as pd


def rolling_mdd(returns, lookback):
    arr = returns.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(lookback, len(arr)):
        w = arr[i - lookback:i]
        if not np.all(np.isfinite(w)):
            continue
        wealth = np.cumprod(1.0 + w)
        peak = np.maximum.accumulate(wealth)
        out[i] = -(wealth / peak - 1.0).min()
    return pd.Series(out, index=returns.index)


def make_features(returns, lookback, volume=None, xret=None):
    r = returns.astype(float)
    f = pd.DataFrame(index=r.index)
    short = max(3, min(5, lookback // 4))
    mid = max(5, min(20, lookback // 2))
    f["ret_1"] = r
    f["mean_lb"] = r.rolling(lookback).mean()
    f["vol_lb"] = r.rolling(lookback).std()
    f["downvol_lb"] = r.where(r < 0.0, 0.0).rolling(lookback).std()
    f["neg_frac_lb"] = (r < 0.0).rolling(lookback).mean()
    f["mom_short"] = (1.0 + r).rolling(short).apply(np.prod, raw=True) - 1.0
    f["mom_mid"] = (1.0 + r).rolling(mid).apply(np.prod, raw=True) - 1.0
    f["mom_lb"] = (1.0 + r).rolling(lookback).apply(np.prod, raw=True) - 1.0
    f["mdd_lb"] = rolling_mdd(r, lookback)
    f["skew_lb"] = r.rolling(lookback).skew()
    if volume is not None:
        lv = np.log1p(volume.astype(float).replace([np.inf, -np.inf], np.nan))
        f["log_volume_lb"] = lv.rolling(lookback).mean()
        f["volume_trend"] = lv.rolling(mid).mean() - lv.rolling(lookback).mean()
    return f.replace([np.inf, -np.inf], np.nan)
