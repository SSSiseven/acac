"""Aggregate completed ovcA7 ACAC baseline runs without editing source CSVs."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (0, 1, 2)
T_CRITICAL_95_DF2 = 4.3026527297

def load_run(log_root: Path, seed: int) -> tuple[Path, dict[str, np.ndarray]]:
    candidates = sorted(log_root.glob(f"s{seed}_*/progress.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one progress.csv for seed {seed}, found {len(candidates)}: {candidates}")
    path = candidates[0]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(line for line in handle if line.strip()))
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    columns = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in ("Timesteps", "Episodes", "Eval/ReturnAverage", "TotalTime")
    }
    if not all(np.isfinite(values).all() for values in columns.values()):
        raise RuntimeError(f"Non-finite aggregation input in {path}")
    if not np.all(np.diff(columns["Timesteps"]) > 0):
        raise RuntimeError(f"Timesteps are not strictly increasing in {path}")
    return path, columns

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=40_000)
    args = parser.parse_args()
    runs = [load_run(args.log_root, seed) for seed in SEEDS]
    start = int(np.ceil(max(run[1]["Timesteps"][0] for run in runs) / args.step) * args.step)
    stop = int(np.floor(min(run[1]["Timesteps"][-1] for run in runs) / args.step) * args.step)
    grid = np.arange(start, stop + 1, args.step, dtype=np.float64)
    interpolated = np.vstack([
        np.interp(grid, data["Timesteps"], data["Eval/ReturnAverage"])
        for _, data in runs
    ])
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0, ddof=1)
    sem = std / np.sqrt(len(SEEDS))
    ci95 = T_CRITICAL_95_DF2 * sem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "eval_return_three_seed.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timesteps", *[f"Seed{seed}" for seed in SEEDS], "Mean", "Std", "SEM", "CI95"])
        writer.writerows(zip(grid.astype(np.int64), *interpolated, mean, std, sem, ci95))
    summary_path = args.output_dir / "run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Seed", "ProgressCSV", "Rows", "Episodes", "Timesteps", "FinalEvalReturn", "TotalTimeSeconds"])
        for seed, (path, data) in zip(SEEDS, runs):
            writer.writerow([seed, path, len(data["Timesteps"]), int(data["Episodes"][-1]), int(data["Timesteps"][-1]), data["Eval/ReturnAverage"][-1], data["TotalTime"][-1]])
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    colors = ("#4477AA", "#EE6677", "#228833")
    for seed, color, (_, data) in zip(SEEDS, colors, runs):
        ax.plot(data["Timesteps"], data["Eval/ReturnAverage"], color=color, alpha=0.35, linewidth=0.8, label=f"Seed {seed}")
    ax.fill_between(grid, mean - ci95, mean + ci95, color="#332288", alpha=0.16, label="95% CI")
    ax.plot(grid, mean, color="#332288", linewidth=2.0, label="Three-seed mean")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Eval/ReturnAverage")
    ax.set_title("ACAC baseline on ovcA7")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, frameon=False)
    fig.savefig(args.output_dir / "eval_return_three_seed.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {summary_path}")
    print(f"Final three-seed mean: {mean[-1]:.6f}")
    print(f"Final 95% CI half-width: {ci95[-1]:.6f}")

if __name__ == "__main__":
    main()
