import gc
import numpy as np
import pandas as pd
from utils.data import equal_weight_portfolio, future_returns_from_return_series
from utils.features import make_features
from utils.market_loader import load_market_frame, frame_to_subset
from utils.metrics import action_matrix, result_row, count_actions, utility
from utils.policies import fit_cfpl, fit_label_policy_fair, fit_direct_utility_policy_fair, build_label_targets

LABELS = ["Ordinary Label", "SL-adjusted Label", "Triple-Barrier", "Trend-Scanning"]


def purged_fit_val_indices(train_dates, train_path_end, val_fraction=0.2):
    dates = pd.to_datetime(pd.Index(train_dates))
    ends = pd.to_datetime(pd.Index(train_path_end))
    n = len(dates)
    if n < 3:
        return np.arange(max(0, n - 1)), np.arange(max(0, n - 1), n), {"purge_status": "too_few"}
    order = np.argsort(dates.values)
    n_val = max(1, int(np.ceil(n * float(val_fraction))))
    n_val = min(n_val, max(1, n - 2))
    candidates = [n - n_val, int(np.floor(n * 0.75)), int(np.floor(n * 0.70)), int(np.floor(n * 0.65)), int(np.floor(n * 0.60))]
    best = None
    for raw_pos in candidates:
        pos = int(np.clip(raw_pos, 1, n - 1))
        val_start = dates[order[pos]]
        fit_mask = ends < val_start
        val_mask = dates >= val_start
        fit_idx = np.flatnonzero(fit_mask)
        val_idx = np.flatnonzero(val_mask)
        if len(fit_idx) >= 20 and len(val_idx) >= 10:
            info = {
                "purge_status": "ok",
                "val_start": str(val_start.date()),
                "fit_n": int(len(fit_idx)),
                "val_n": int(len(val_idx)),
                "purged_between_fit_val": int(np.sum((dates < val_start) & (ends >= val_start))),
            }
            return fit_idx, val_idx, info
        best = (fit_idx, val_idx, val_start)
    fit_idx, val_idx, val_start = best
    info = {
        "purge_status": "fallback_small",
        "val_start": str(val_start.date()),
        "fit_n": int(len(fit_idx)),
        "val_n": int(len(val_idx)),
        "purged_between_fit_val": int(np.sum((dates < val_start) & (ends >= val_start))),
    }
    return fit_idx, val_idx, info


def evaluate_cell(samples, feats, market, k, k_eff, cfg, collect_returns=True):
    df = samples.join(feats, how="left")
    if "path_end" not in df.columns:
        raise ValueError("samples must include path_end for purged splitting")
    df["path_end"] = pd.to_datetime(df["path_end"])
    feat_cols = list(feats.columns)
    df = df.dropna(subset=["terminal_return", "min_path_return", "path_end"]).dropna(subset=feat_cols, how="all")
    if df.empty:
        return [], [], [], []

    test_start = pd.Timestamp(cfg.test_start)
    train_mask = np.asarray((df.index < test_start) & (df["path_end"] < test_start))
    test_mask = np.asarray(df.index >= test_start)
    if train_mask.sum() < 80 or test_mask.sum() < 20:
        return [], [], [], []

    X = df[feat_cols].to_numpy(float)
    terminal = df["terminal_return"].to_numpy(float)
    minp = df["min_path_return"].to_numpy(float)
    am = action_matrix(
        terminal,
        minp,
        cfg.stop_levels,
        cfg.default_leverage,
        cfg.default_cost_bps,
        cfg.default_gamma,
        cfg.default_stop_slippage_bps,
    )
    actions, net, H, hit = am["actions"], am["net"], am["utility"], am["hit"]
    Xtr, Xte = X[train_mask], X[test_mask]
    Htr = H[train_mask]
    nettr = net[train_mask]
    nette = net[test_mask]
    train_dates = df.index[train_mask]
    train_path_end = df.loc[train_mask, "path_end"].to_numpy()
    fit_idx, val_idx, purge_info = purged_fit_val_indices(train_dates, train_path_end)
    if len(fit_idx) < 10 or len(val_idx) < 5:
        return [], [], [], []

    meta = {
        "market": market,
        "asset_count": int(k),
        "asset_count_effective": int(k_eff),
        "lookback": int(cfg.default_lookback),
        "cost_bps": float(cfg.default_cost_bps),
        "stop_slippage_bps": float(cfg.default_stop_slippage_bps),
        "leverage": float(cfg.default_leverage),
        "gamma": float(cfg.default_gamma),
        "horizon_days": int(cfg.horizon_days),
        "test_start": cfg.test_start,
        "train_n_purged": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
        "fit_n": int(len(fit_idx)),
        "val_n": int(len(val_idx)),
        "purged_between_fit_val": int(purge_info.get("purged_between_fit_val", 0)),
    }
    rows, counts, returns, params = [], [], [], []

    def add(name, out):
        r = np.asarray(out["returns"], float)
        chosen = np.asarray(out["chosen"], object)
        rows.append(result_row(meta, r, chosen, name, cfg.periods_per_year, cfg.default_gamma, float(np.nanmean(out.get("score", np.nan))), float(out.get("threshold", np.nan))))
        counts.append(count_actions(meta, chosen, name, cfg.stop_levels))
        par = dict(meta)
        par.update({"strategy": name})
        par.update(out.get("params", {}) or {})
        par.update(purge_info)
        params.append(par)
        if collect_returns:
            u = utility(r, cfg.default_gamma)
            w = np.cumprod(1.0 + r)
            dates = df.index[test_mask]
            for dt, rr, uu, ww, cc in zip(dates, r, u, w, chosen):
                item = dict(meta)
                item.update({"date": dt, "strategy": name, "return": float(rr), "utility": float(uu), "wealth": float(ww), "action": str(cc)})
                returns.append(item)

    add("CFPL", fit_cfpl(Xtr, Xte, Htr, nette, actions, cfg.default_cost_bps, cfg.default_leverage, cfg.default_gamma, cfg.default_stop_slippage_bps, fit_idx=fit_idx, val_idx=val_idx))
    targets = build_label_targets(df, actions, hit, cfg.stop_levels)
    for lab in LABELS:
        add(lab, fit_label_policy_fair(Xtr, Xte, targets[lab][train_mask], nettr, nette, actions, cfg.default_cost_bps, cfg.default_leverage, cfg.default_gamma, cfg.default_stop_slippage_bps, fit_idx=fit_idx, val_idx=val_idx))
    add("Direct Utility Ridge", fit_direct_utility_policy_fair(Xtr, Xte, Htr, nettr, nette, actions, cfg.default_cost_bps, cfg.default_leverage, cfg.default_gamma, cfg.default_stop_slippage_bps, fit_idx=fit_idx, val_idx=val_idx))
    add("No Trade", {"returns": np.zeros(test_mask.sum()), "chosen": np.array(["no_trade"] * test_mask.sum(), dtype=object), "score": np.zeros(test_mask.sum()), "threshold": np.nan})
    return rows, counts, returns, params



def run_market_k(market, k, ret, vol, ranked_assets, cfg):
    if len(ranked_assets) < int(k):
        info = {
            "market": market,
            "asset_count": int(k),
            "asset_count_effective": len(ranked_assets),
            "status": "skipped_insufficient_assets",
        }
        return [], [], [info]
    assets = ranked_assets[: int(k)]
    port, pvol, xret = equal_weight_portfolio(ret, vol, assets)
    samples = future_returns_from_return_series(port, cfg.horizon_days, cfg.stop_levels, stride=cfg.horizon_days)
    feats = make_features(port, cfg.default_lookback, pvol, xret).reindex(samples.index)
    rows, counts, returns, params = evaluate_cell(samples, feats, market, int(k), len(assets), cfg, collect_returns=True)
    info = {
        "market": market,
        "asset_count": int(k),
        "asset_count_effective": len(assets),
        "status": "used" if rows else "skipped_no_valid_cell",
        "n_samples": len(samples),
        "train_n": int((samples.index < pd.Timestamp(cfg.test_start)).sum()) if len(samples) else 0,
        "test_n": int((samples.index >= pd.Timestamp(cfg.test_start)).sum()) if len(samples) else 0,
        "sample_start": str(samples.index.min()) if len(samples) else "",
        "sample_end": str(samples.index.max()) if len(samples) else "",
    }
    del port, pvol, xret, samples, feats
    gc.collect()
    return rows, returns, [info]


def run_default_experiment(data_dir, cfg):
    all_rows = []
    all_returns = []
    sample_info = []
    for market in cfg.markets:
        print(f"load {market}", flush=True)
        max_k = max(cfg.asset_counts[market])
        market_frame, ranked_assets = load_market_frame(data_dir, market, max_k, cfg.test_start)
        for k in cfg.asset_counts[market]:
            if len(ranked_assets) < int(k):
                info = {
                    "market": market,
                    "asset_count": int(k),
                    "asset_count_effective": len(ranked_assets),
                    "status": "skipped_insufficient_assets",
                }
                sample_info.append(info)
                print(f"skip {market} K={k}: skipped_insufficient_assets", flush=True)
                continue
            ret, vol, top_assets = frame_to_subset(market_frame, ranked_assets, k)
            rows, returns, info = run_market_k(market, k, ret, vol, top_assets, cfg)
            all_rows.extend(rows)
            all_returns.extend(returns)
            sample_info.extend(info)
            if rows:
                print(f"done {market} K={k}: result rows={len(rows)}, return rows={len(returns)}", flush=True)
            else:
                print(f"skip {market} K={k}: {info[0].get('status')}", flush=True)
            del ret, vol
            gc.collect()
        del market_frame
        gc.collect()
    return pd.DataFrame(all_rows), pd.DataFrame(all_returns), pd.DataFrame(sample_info)
