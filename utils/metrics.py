import math
import numpy as np


def stopped_gross_return(terminal, min_path, stop):
    terminal = np.asarray(terminal, dtype=float)
    min_path = np.asarray(min_path, dtype=float)
    if stop is None:
        return terminal.copy(), np.zeros_like(terminal, dtype=bool)
    hit = min_path <= -float(stop)
    gross = np.where(hit, -float(stop), terminal)
    return gross, hit


def utility(net, gamma):
    net = np.asarray(net, dtype=float)
    return net - 0.5 * float(gamma) * net * net


def action_matrix(terminal, min_path, stop_levels, leverage, cost_bps, gamma, stop_slippage_bps):
    action_names = ["nostop"] + [f"stop_{float(s):0.3f}" for s in stop_levels]
    gross_cols = []
    hit_cols = []
    gross, hit = stopped_gross_return(terminal, min_path, None)
    gross_cols.append(gross)
    hit_cols.append(hit)
    for s in stop_levels:
        gross, hit = stopped_gross_return(terminal, min_path, float(s))
        gross_cols.append(gross)
        hit_cols.append(hit)
    gross_mat = np.column_stack(gross_cols)
    hit_mat = np.column_stack(hit_cols)
    net_mat = float(leverage) * gross_mat - 2.0 * float(cost_bps) / 10000.0 * float(leverage)
    if float(stop_slippage_bps) != 0.0:
        net_mat = net_mat - float(leverage) * float(stop_slippage_bps) / 10000.0 * hit_mat.astype(float)
    return {
        "actions": np.array(action_names, dtype=object),
        "gross": gross_mat,
        "hit": hit_mat,
        "net": net_mat,
        "utility": utility(net_mat, gamma),
    }


def max_drawdown(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float(-(wealth / peak - 1.0).min())


def performance(returns, periods_per_year):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in ["n", "ann_return", "cagr", "vol", "sharpe", "mdd", "var95", "cvar95", "hit_rate"]}
    wealth = np.cumprod(1.0 + r)
    ann_return = float(np.mean(r) * periods_per_year)
    vol = float(np.std(r, ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan")
    sharpe = ann_return / vol if np.isfinite(vol) and vol > 0 else 0.0
    cagr = float(wealth[-1] ** (periods_per_year / n) - 1.0) if wealth[-1] > 0 else float("nan")
    losses = -r
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(np.mean(tail)) if len(tail) else float("nan")
    return {
        "n": int(n),
        "ann_return": ann_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": max_drawdown(r),
        "var95": var95,
        "cvar95": cvar95,
        "hit_rate": float(np.mean(r > 0.0)),
    }


def result_row(meta, returns, chosen, strategy, periods_per_year, gamma, score_mean=float("nan"), threshold=float("nan")):
    row = dict(meta)
    row.update(performance(returns, periods_per_year))
    row["strategy"] = strategy
    row["trade_rate"] = float(np.mean(np.asarray(chosen, dtype=object) != "no_trade"))
    row["avg_utility"] = float(np.nanmean(utility(returns, gamma)))
    row["mean_period_return"] = float(np.nanmean(returns))
    row["score_mean"] = score_mean
    row["threshold"] = threshold
    row["terminal_wealth"] = float(np.prod(1.0 + np.asarray(returns, dtype=float))) if len(returns) else float("nan")
    return row


def count_actions(meta, chosen, strategy, stop_levels):
    row = {k: meta.get(k, np.nan) for k in ["market", "asset_count", "asset_count_effective", "lookback", "cost_bps", "leverage", "gamma", "stop_slippage_bps"]}
    row["strategy"] = strategy
    for name in ["nostop"] + [f"stop_{float(s):0.3f}" for s in stop_levels]:
        row[f"action_{name}"] = int(np.sum(np.asarray(chosen, dtype=object) == name))
    row["action_no_trade"] = int(np.sum(np.asarray(chosen, dtype=object) == "no_trade"))
    row["n"] = int(len(chosen))
    return row
