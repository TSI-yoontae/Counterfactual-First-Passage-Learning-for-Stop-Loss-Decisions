class ExperimentConfig:
    def __init__(self):
        self.stop_levels = (0.02, 0.03, 0.05, 0.08)
        self.horizon_days = 5
        self.test_start = "2019-01-01"
        self.periods_per_year = 252.0 / 5.0
        self.default_lookback = 60
        self.default_cost_bps = 5.0
        self.default_stop_slippage_bps = 10.0
        self.default_leverage = 1.0
        self.default_gamma = 3.0
        self.asset_counts = {
            "DE_DAX": (10, 25, 50, 100, 200),
            "JP_Nikkei225": (10, 25, 50, 100, 200),
            "KR_KOSPI200": (10, 25, 50, 100, 200),
            "US_SP500": (10, 25, 50, 100, 200),
        }
        self.markets = ("DE_DAX", "JP_Nikkei225", "KR_KOSPI200", "US_SP500")
        self.methods = (
            "CFPL",
            "Ordinary Label",
            "SL-adjusted Label",
            "Triple-Barrier",
            "Trend-Scanning",
            "Direct Utility Ridge",
        )
        self.all_methods = self.methods + ("No Trade",)
