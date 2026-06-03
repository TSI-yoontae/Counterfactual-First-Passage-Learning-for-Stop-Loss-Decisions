import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROPOSED = "CFPL"
ACTIVE_METHODS = ["CFPL", "Ordinary Label", "SL-adjusted Label", "Triple-Barrier", "Trend-Scanning", "Direct Utility Ridge"]
ALL_METHODS = ACTIVE_METHODS + ["No Trade"]
ORDER = {s: i for i, s in enumerate(ALL_METHODS)}
PCT_COLS = {"avg_utility", "ann_return", "cagr", "vol", "mdd", "var95", "cvar95", "hit_rate", "trade_rate", "mean_period_return"}
METRIC_COLS = ["avg_utility", "cagr", "sharpe", "mdd", "cvar95", "trade_rate", "terminal_wealth"]
TABLE_COLS = ["Utility", "CAGR", "Sharpe", "MDD", "CVaR95", "Trade Rate", "Wealth"]
DISPLAY_NAMES = {"CFPL": "CFPL (Ours)"}
FIGURE_MARKET_LABELS = {
    "DE_DAX": "DAX",
    "JP_Nikkei225": "Nikkei 225",
    "KR_KOSPI200": "KOSPI 200",
    "US_SP500": "S&P 500",
}


def scale_percent_columns(df):
    out = df.copy()
    for col in out.columns:
        if col in PCT_COLS and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col] * 100.0
    return out


def aggregate_main_table(results):
    out = results.groupby(["market", "strategy"], as_index=False).agg({m: "mean" for m in METRIC_COLS})
    out["strategy_order"] = out["strategy"].map(ORDER).fillna(99)
    out["market_order"] = out["market"].map({"DE_DAX": 0, "JP_Nikkei225": 1, "KR_KOSPI200": 2, "US_SP500": 3}).fillna(99)
    out = out.sort_values(["market_order", "strategy_order"]).drop(columns=["strategy_order", "market_order"])
    out = scale_percent_columns(out)
    out = out[out["strategy"].isin(ACTIVE_METHODS)].copy()
    return out


def make_average_wealth(returns):
    rows = []
    ret_default = returns.copy()
    for market, market_df in ret_default.groupby("market"):
        date_grid = pd.Index(sorted(market_df["date"].unique()), name="date")
        for strategy in ALL_METHODS:
            strategy_df = market_df[market_df.strategy == strategy]
            if strategy_df.empty:
                continue
            curves = []
            for asset_count, k_df in strategy_df.groupby("asset_count"):
                curve = (
                    k_df.sort_values("date")
                    .drop_duplicates("date")
                    .set_index("date")["wealth"]
                    .reindex(date_grid)
                    .ffill()
                    .fillna(1.0)
                    .rename(asset_count)
                )
                curves.append(curve)
            matrix = pd.concat(curves, axis=1)
            mean_wealth = matrix.mean(axis=1)
            n_k = matrix.notna().sum(axis=1)
            rows.append(pd.DataFrame({"market": market, "date": date_grid, "strategy": strategy, "wealth": mean_wealth.values, "n_K": n_k.values}))
    avg = pd.concat(rows, ignore_index=True)
    avg = avg.sort_values(["market", "strategy", "date"])
    avg["return_from_mean_wealth"] = avg.groupby(["market", "strategy"])["wealth"].pct_change().fillna(avg["wealth"] - 1.0)
    avg["strategy_order"] = avg["strategy"].map(ORDER).fillna(99)
    avg["market_order"] = avg["market"].map({"DE_DAX": 0, "JP_Nikkei225": 1, "KR_KOSPI200": 2, "US_SP500": 3}).fillna(99)
    avg = avg.sort_values(["market_order", "date", "strategy_order"]).drop(columns=["market_order", "strategy_order"])
    return avg


def write_main_table_csv(main_table, out_path):
    renamed = main_table.rename(columns={
        "strategy": "Method",
        "avg_utility": "Utility",
        "cagr": "CAGR",
        "sharpe": "Sharpe",
        "mdd": "MDD",
        "cvar95": "CVaR95",
        "trade_rate": "Trade Rate",
        "terminal_wealth": "Wealth",
    })
    renamed["Method"] = renamed["Method"].map(lambda x: DISPLAY_NAMES.get(x, x))
    renamed["Market"] = renamed["market"].map({
        "DE_DAX": "DAX",
        "JP_Nikkei225": "Nikkei 225",
        "KR_KOSPI200": "KOSPI 200",
        "US_SP500": "S&P 500",
    })
    renamed = renamed[["Market", "Method"] + TABLE_COLS]
    renamed.to_csv(out_path, index=False)


def write_wealth_figure(avg_wealth, out_path):
    plot_df = avg_wealth[avg_wealth["strategy"].isin(ACTIVE_METHODS)].copy()
    markets = ["DE_DAX", "JP_Nikkei225", "KR_KOSPI200", "US_SP500"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, market in zip(axes, markets):
        market_df = plot_df[plot_df["market"] == market]
        for strategy in ACTIVE_METHODS:
            line_df = market_df[market_df["strategy"] == strategy].sort_values("date")
            if line_df.empty:
                continue
            linewidth = 2.2 if strategy == PROPOSED else 1.1
            ax.plot(line_df["date"], line_df["wealth"], label=DISPLAY_NAMES.get(strategy, strategy), linewidth=linewidth)
        ax.axhline(1.0, linewidth=0.8, linestyle=":")
        ax.set_title(FIGURE_MARKET_LABELS.get(market, market))
        ax.set_xlabel("Date")
        ax.set_ylabel("Average normalized wealth")
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(results, returns, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    main_table = aggregate_main_table(results)
    avg_wealth = make_average_wealth(returns)
    write_main_table_csv(main_table, results_dir / "main_table.csv")
    write_wealth_figure(avg_wealth, results_dir / "wealth_figure.pdf")
    return main_table, avg_wealth
