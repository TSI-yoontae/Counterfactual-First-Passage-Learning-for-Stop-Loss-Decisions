import warnings
import numpy as np
import pandas as pd
from utils.metrics import utility

DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


class RidgeValueModel:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.constant_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y)
        X = X[mask]
        y = y[mask]
        if len(y) == 0:
            self.constant_ = 0.0
            return self
        if np.nanstd(y) < 1e-14:
            self.constant_ = float(np.nanmean(y))
            return self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isfinite(med), med, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med_)
        self.mu_ = Xi.mean(axis=0)
        sd = Xi.std(axis=0)
        self.sd_ = np.where((sd > 1e-12) & np.isfinite(sd), sd, 1.0)
        Xs = (Xi - self.mu_) / self.sd_
        self.ymean_ = float(y.mean())
        yc = y - self.ymean_
        p = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        b = Xs.T @ yc
        try:
            self.beta_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.pinv(A) @ b
        self.constant_ = None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.constant_ is not None:
            return np.full(X.shape[0], self.constant_, dtype=float)
        Xi = np.where(np.isfinite(X), X, self.med_)
        Xs = (Xi - self.mu_) / self.sd_
        return self.ymean_ + Xs @ self.beta_


def action_stop_value(action_name):
    if action_name == "nostop":
        return -1.0
    return float(str(action_name).split("_")[1])


def expand_action_features(X, actions, cost_bps, leverage, gamma, stop_slippage_bps):
    n = X.shape[0]
    a = np.array([action_stop_value(x) for x in actions], dtype=float)
    return np.hstack([
        np.repeat(X, len(actions), axis=0),
        np.tile(a, n).reshape(-1, 1),
        np.full((n * len(actions), 1), float(cost_bps) / 10000.0),
        np.full((n * len(actions), 1), float(leverage)),
        np.full((n * len(actions), 1), float(gamma)),
        np.full((n * len(actions), 1), float(stop_slippage_bps) / 10000.0),
    ])


def quantile_candidates(x, qs, include_zero=True, lower_bound=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    vals = [0.0] if include_zero else []
    if len(x):
        vals += [float(v) for v in np.nanquantile(x, list(qs))]
        vals += [float(np.nanmin(x) - 1e-12), float(np.nanmax(x))]
    out = []
    for v in vals:
        if not np.isfinite(v):
            continue
        if lower_bound is not None and v < lower_bound:
            continue
        out.append(float(v))
    return sorted(set(out))


def fit_margin_predictions(X_fit, X_pred, H_fit, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha):
    y_mu = H_fit[:, 0]
    model_mu = RidgeValueModel(alpha).fit(X_fit, y_mu)
    stop_actions = actions[1:]
    X_stop_fit = expand_action_features(X_fit, stop_actions, cost_bps, leverage, gamma, stop_slippage_bps)
    y_rho = (H_fit[:, 1:] - H_fit[:, [0]]).reshape(-1)
    model_rho = RidgeValueModel(alpha).fit(X_stop_fit, y_rho)
    mu_pred = model_mu.predict(X_pred)
    rho_pred = model_rho.predict(expand_action_features(X_pred, stop_actions, cost_bps, leverage, gamma, stop_slippage_bps)).reshape(X_pred.shape[0], len(stop_actions))
    return {"mu": mu_pred, "rho": rho_pred}


def action_consistent_scores(mu, rho_j, theta):
    theta = max(0.0, float(theta))
    protect = rho_j > theta
    selected_value = np.where(protect, mu + rho_j, mu)
    return selected_value, protect


def choose_cfpl_risk_budget_params(pred, H_val, min_trade=0.05):
    mu, rho = pred["mu"], pred["rho"]
    n, J = rho.shape
    best, par = -np.inf, None
    theta_q = [0.05, 0.20, 0.40, 0.60, 0.80]
    kappa_q = [0.05, 0.20, 0.40, 0.60, 0.80]
    for fallback in range(J):
        theta_grid = quantile_candidates(rho[:, fallback], theta_q, include_zero=True, lower_bound=0.0)
        for theta in theta_grid:
            selected_value, protect = action_consistent_scores(mu, rho[:, fallback], theta)
            kappa_grid = quantile_candidates(selected_value, kappa_q, include_zero=True, lower_bound=None)
            for kappa in kappa_grid:
                trade = selected_value > kappa
                if trade.mean() < min_trade:
                    continue
                choice = np.zeros(n, dtype=int)
                choice[trade & protect] = fallback + 1
                val = float(np.where(trade, H_val[np.arange(n), choice], 0.0).mean())
                cand = {
                    "variant": "action_consistent_purged_risk_budget",
                    "kappa_e": float(kappa),
                    "theta_p": float(theta),
                    "fallback": int(fallback),
                    "val_utility": val,
                    "val_trade_rate": float(trade.mean()),
                    "val_protect_rate_cond_trade": float(protect[trade].mean()) if trade.any() else 0.0,
                }
                if val > best:
                    best, par = val, cand
    if par is None:
        return {"variant": "action_consistent_purged_risk_budget", "kappa_e": 0.0, "theta_p": 0.0, "fallback": 0, "val_utility": float("nan")}
    return par


def apply_cfpl_risk_budget_policy(pred, net_test, actions, params):
    mu, rho = pred["mu"], pred["rho"]
    n, J = rho.shape
    fallback = int(params["fallback"])
    selected_value, protect = action_consistent_scores(mu, rho[:, fallback], float(params["theta_p"]))
    trade = selected_value > float(params["kappa_e"])
    choice = np.zeros(n, dtype=int)
    choice[trade & protect] = fallback + 1
    returns = np.where(trade, net_test[np.arange(n), choice], 0.0)
    chosen = np.where(trade, actions[choice], "no_trade")
    return {"returns": returns, "chosen": chosen, "score": selected_value, "threshold": params["kappa_e"], "params": params, "trade": trade, "protect": protect}


def build_label_targets(samples, actions, hit, stop_levels):
    terminal = samples["terminal_return"].to_numpy(dtype=float)
    ordinary = np.tile((terminal > 0.0).astype(float).reshape(-1, 1), (1, len(actions)))
    sl = np.zeros((len(samples), len(actions)), dtype=float)
    sl[:, 0] = (terminal > 0.0).astype(float)
    for j in range(1, len(actions)):
        sl[:, j] = ((terminal > 0.0) & (~hit[:, j])).astype(float)
    tb = np.zeros((len(samples), len(actions)), dtype=float)
    tb[:, 0] = (terminal > 0.0).astype(float)
    for j, s in enumerate(stop_levels, start=1):
        col = f"tb_{float(s):0.3f}"
        if col in samples.columns:
            tb[:, j] = samples[col].astype(float).to_numpy()
        else:
            tb[:, j] = sl[:, j]
    ts_y = samples.get("trend_scan_label", pd.Series(np.zeros(len(samples)), index=samples.index)).astype(float).to_numpy()
    trend = np.tile(ts_y.reshape(-1, 1), (1, len(actions)))
    return {"Ordinary Label": ordinary, "SL-adjusted Label": sl, "Triple-Barrier": tb, "Trend-Scanning": trend}


def fit_action_score_model(X_fit, y_fit, X_pred, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha):
    model = RidgeValueModel(alpha).fit(expand_action_features(X_fit, actions, cost_bps, leverage, gamma, stop_slippage_bps), y_fit.reshape(-1))
    return model.predict(expand_action_features(X_pred, actions, cost_bps, leverage, gamma, stop_slippage_bps)).reshape(X_pred.shape[0], len(actions))


def choose_label_threshold_fair(scores_val, net_val, gamma, min_trade=0.05):
    scores_val = np.asarray(scores_val, dtype=float)
    best_score = np.nanmax(scores_val, axis=1)
    best_idx = np.nanargmax(scores_val, axis=1)
    finite = best_score[np.isfinite(best_score)]
    if finite.size == 0:
        return 0.5, float("nan"), 0.0
    qs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    candidates = sorted(set([0.0, 0.5] + [float(x) for x in np.nanquantile(finite, qs)]))
    best_u, best_th, best_tr = -np.inf, float(candidates[0]), 0.0
    for th in candidates:
        trade = best_score > th
        tr = float(np.mean(trade))
        if tr < min_trade:
            continue
        returns = np.where(trade, net_val[np.arange(len(best_idx)), best_idx], 0.0)
        u = float(utility(returns, gamma).mean())
        if u > best_u:
            best_u, best_th, best_tr = u, float(th), tr
    if not np.isfinite(best_u):
        th = float(np.nanmin(finite)) - 1e-12
        trade = best_score > th
        returns = np.where(trade, net_val[np.arange(len(best_idx)), best_idx], 0.0)
        best_u, best_th, best_tr = float(utility(returns, gamma).mean()), th, float(np.mean(trade))
    return best_th, best_u, best_tr


def default_fit_val_indices(n):
    n_fit = int(n * 0.8)
    if n_fit < 100:
        n_fit = max(50, int(n * 0.7))
    n_fit = min(max(1, n_fit), max(1, n - 1))
    return np.arange(n_fit), np.arange(n_fit, n)


def purged_fit_val_indices(n, dates=None, path_end=None):
    base_fit, base_val = default_fit_val_indices(n)
    if dates is None or path_end is None or len(base_val) == 0:
        return base_fit, base_val
    dates_arr = pd.to_datetime(np.asarray(dates))
    end_arr = pd.to_datetime(np.asarray(path_end))
    first_val_date = dates_arr[base_val][0]
    fit_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    fit_mask[base_fit] = True
    val_mask[base_val] = True
    fit_mask &= end_arr < first_val_date
    fit_idx = np.flatnonzero(fit_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(fit_idx) == 0 or len(val_idx) == 0:
        return base_fit, base_val
    return fit_idx, val_idx


def fit_label_policy_fair(X_train, X_test, y_train, net_train, net_test, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha_grid=DEFAULT_ALPHA_GRID, min_trade=0.05, fit_idx=None, val_idx=None, train_dates=None, train_path_end=None):
    if fit_idx is None or val_idx is None:
        fit_idx, val_idx = purged_fit_val_indices(len(X_train), dates=train_dates, path_end=train_path_end)
    fit_idx = np.asarray(fit_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    best = {"val_utility": -np.inf, "alpha": None, "threshold": 0.5, "val_trade_rate": np.nan, "fit_n": int(len(fit_idx)), "val_n": int(len(val_idx))}
    for alpha in alpha_grid:
        scores_val = fit_action_score_model(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], actions, cost_bps, leverage, gamma, stop_slippage_bps, float(alpha))
        th, val_u, val_tr = choose_label_threshold_fair(scores_val, net_train[val_idx], gamma, min_trade=min_trade)
        if val_u > best["val_utility"]:
            best.update({"val_utility": float(val_u), "alpha": float(alpha), "threshold": float(th), "val_trade_rate": float(val_tr)})
    alpha = float(best["alpha"] if best["alpha"] is not None else 1.0)
    scores_test = fit_action_score_model(X_train, y_train, X_test, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha)
    best_idx = np.nanargmax(scores_test, axis=1)
    best_score = scores_test[np.arange(len(best_idx)), best_idx]
    trade = best_score > float(best["threshold"])
    returns = np.where(trade, net_test[np.arange(len(best_idx)), best_idx], 0.0)
    chosen = np.where(trade, actions[best_idx], "no_trade")
    return {"returns": returns, "chosen": chosen, "score": best_score, "threshold": float(best["threshold"]), "params": best}


def fit_direct_utility_policy_fair(X_train, X_test, H_train, net_train, net_test, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha_grid=DEFAULT_ALPHA_GRID, min_trade=0.05, fit_idx=None, val_idx=None, train_dates=None, train_path_end=None):
    return fit_label_policy_fair(X_train, X_test, H_train, net_train, net_test, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha_grid=alpha_grid, min_trade=min_trade, fit_idx=fit_idx, val_idx=val_idx, train_dates=train_dates, train_path_end=train_path_end)


def fit_cfpl(X_train, X_test, H_train, net_test, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha_grid=DEFAULT_ALPHA_GRID, min_trade=0.05, fit_idx=None, val_idx=None, train_dates=None, train_path_end=None):
    if fit_idx is None or val_idx is None:
        fit_idx, val_idx = purged_fit_val_indices(len(X_train), dates=train_dates, path_end=train_path_end)
    fit_idx = np.asarray(fit_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    best = {"val_utility": -np.inf, "alpha": None, "policy_params": None, "fit_n": int(len(fit_idx)), "val_n": int(len(val_idx))}
    for alpha in alpha_grid:
        pred_val = fit_margin_predictions(X_train[fit_idx], X_train[val_idx], H_train[fit_idx], actions, cost_bps, leverage, gamma, stop_slippage_bps, float(alpha))
        params = choose_cfpl_risk_budget_params(pred_val, H_train[val_idx], min_trade=min_trade)
        val_u = float(params.get("val_utility", -np.inf))
        if val_u > best["val_utility"]:
            best.update({"val_utility": val_u, "alpha": float(alpha), "policy_params": params})
    alpha = float(best["alpha"] if best["alpha"] is not None else 1.0)
    params = best["policy_params"] or {"variant": "action_consistent_purged_risk_budget", "kappa_e": 0.0, "theta_p": 0.0, "fallback": 0, "val_utility": float("nan")}
    pred_test = fit_margin_predictions(X_train, X_test, H_train, actions, cost_bps, leverage, gamma, stop_slippage_bps, alpha)
    out = apply_cfpl_risk_budget_policy(pred_test, net_test, actions, params)
    out["params"] = dict(out.get("params", {}) or {})
    out["params"].update({"alpha": alpha, "val_utility_alpha_selection": best["val_utility"], "fit_n": int(len(fit_idx)), "val_n": int(len(val_idx))})
    return out
