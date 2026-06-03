import argparse
import shutil
from pathlib import Path
from exp.config import ExperimentConfig
from exp.experiment import run_default_experiment
from exp.reporting import write_outputs

REQUIRED_FILES = [
    "DE_DAX_daily.parquet",
    "JP_Nikkei_225_daily.parquet",
    "KR_KOSPI_200_daily.parquet",
    "US_SP500_daily.parquet",
]


def resolve_data_dir(arg_value, base_dir):
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    sandbox_dir = Path("/mnt/data")
    local_data_dir = (base_dir.parent / "data").resolve()
    if all((sandbox_dir / name).exists() for name in REQUIRED_FILES):
        return sandbox_dir
    if all((local_data_dir / name).exists() for name in REQUIRED_FILES):
        return local_data_dir
    return sandbox_dir


def check_data_dir(data_dir):
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Data directory is missing required parquet files:\n  - {joined}\nData directory: {data_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the default CFPL experiment and write only the main table CSV and wealth figure.")
    parser.add_argument("--data-dir", default=None, help="Directory containing the four daily parquet files.")
    parser.add_argument("--results-dir", default=None, help="Output directory. Default: code/results.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, base_dir)
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else base_dir / "results"
    check_data_dir(data_dir)

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig()
    print(f"data_dir={data_dir}", flush=True)
    print(f"results_dir={results_dir}", flush=True)
    results, returns, sample_info = run_default_experiment(data_dir, cfg)
    if results.empty or returns.empty:
        raise RuntimeError("No valid result rows were produced. Check the input data and configuration.")
    main_table, avg_wealth = write_outputs(results, returns, results_dir)
    print("saved results/main_table.csv", flush=True)
    print("saved results/wealth_figure.pdf", flush=True)
    skipped = sample_info[sample_info["status"] != "used"] if not sample_info.empty else sample_info
    if skipped is not None and not skipped.empty:
        for _, row in skipped.iterrows():
            print(f"skipped {row['market']} K={row['asset_count']}: {row['status']} (effective={row['asset_count_effective']})", flush=True)
    print("completed", flush=True)


if __name__ == "__main__":
    main()
