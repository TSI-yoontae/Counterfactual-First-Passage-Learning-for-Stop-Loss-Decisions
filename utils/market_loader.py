import numpy as np
import pandas as pd
from utils.data import EQUITY_FILES


def patch_pyarrow_unregister():
    try:
        import pyarrow as pa
    except Exception:
        return
    if getattr(pa.unregister_extension_type, "_cfpl_safe", False):
        return
    original = pa.unregister_extension_type

    def safe_unregister(name):
        try:
            return original(name)
        except Exception as exc:
            if exc.__class__.__name__ in {"ArrowKeyError", "KeyError"}:
                return None
            raise

    safe_unregister._cfpl_safe = True
    pa.unregister_extension_type = safe_unregister


def clean_market_frame(data_dir, market):
    patch_pyarrow_unregister()
    spec = EQUITY_FILES[market]
    cols = ["date", spec["id_col"], spec["price_col"], spec["ret_col"], spec["mktcap_col"], spec["volume_col"]]
    path = data_dir / spec["daily"]
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_parquet(path, columns=cols)
    df = df.rename(columns={
        spec["id_col"]: "asset_id",
        spec["price_col"]: "close",
        spec["ret_col"]: "ret",
        spec["mktcap_col"]: "mktcap",
        spec["volume_col"]: "volume",
    })
    df = df.dropna(subset=["date", "asset_id", "close"])
    df["asset_id"] = df["asset_id"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").abs()
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["asset_id", "date"])
    df["ret"] = df["ret"].where(df["ret"].notna(), df.groupby("asset_id")["close"].pct_change())
    df.loc[(df["ret"] < -0.95) | (df["ret"] > 2.0), "ret"] = np.nan
    return df


def rank_assets(df, test_start, max_k):
    train = df[df["date"] < pd.Timestamp(test_start)]
    obs = train.groupby("asset_id")["ret"].count()
    medcap = train.groupby("asset_id")["mktcap"].median()
    n_dates = train["date"].nunique()
    min_obs = max(20, min(252, int(n_dates * 0.1)))
    ranked = medcap[obs.reindex(medcap.index).fillna(0) >= min_obs].dropna().sort_values(ascending=False)
    return ranked.index.astype(str).tolist()[: int(max_k)]


def load_market_frame(data_dir, market, max_k, test_start):
    df = clean_market_frame(data_dir, market)
    ranked_assets = rank_assets(df, test_start, max_k)
    df = df[df["asset_id"].isin(ranked_assets)].copy()
    return df, ranked_assets


def frame_to_subset(df, ranked_assets, max_k):
    top_assets = ranked_assets[: int(max_k)]
    sdf = df[df["asset_id"].isin(top_assets)].copy()
    ret = sdf.pivot_table(index="date", columns="asset_id", values="ret", aggfunc="last").sort_index()
    vol = sdf.pivot_table(index="date", columns="asset_id", values="volume", aggfunc="last").sort_index()
    ret = ret.reindex(columns=top_assets)
    vol = vol.reindex(columns=top_assets)
    return ret, vol, top_assets


def load_subset(data_dir, market, max_k, test_start):
    df, ranked_assets = load_market_frame(data_dir, market, max_k, test_start)
    return frame_to_subset(df, ranked_assets, max_k)
