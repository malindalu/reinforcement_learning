import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def plot_reward_and_cost_for_run(
    progress_path: str,
    window: int = 10,
) -> None:
    """Make reward/cost rolling plots for a single progress.csv."""
    run_dir = os.path.dirname(progress_path)
    print(f"[INFO] Processing {progress_path}")

    df = pd.read_csv(progress_path)

    required_cols = ["TotalEnvSteps", "Metrics/EpRet", "Metrics/EpCost"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] Missing column '{col}' in {progress_path}, skipping.")
            return

    if df.empty:
        print(f"[WARN] Empty CSV at {progress_path}, skipping.")
        return

    steps = df["TotalEnvSteps"]

    # Rolling stats for reward
    df["EpRet_mean"] = df["Metrics/EpRet"].rolling(window=window, min_periods=1).mean()
    df["EpRet_std"] = (
        df["Metrics/EpRet"].rolling(window=window, min_periods=1).std().fillna(0.0)
    )

    # Rolling stats for cost
    df["EpCost_mean"] = df["Metrics/EpCost"].rolling(window=window, min_periods=1).mean()
    df["EpCost_std"] = (
        df["Metrics/EpCost"].rolling(window=window, min_periods=1).std().fillna(0.0)
    )

    # --- Plot reward + cost in one figure ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Reward
    ax = axes[0]
    ax.plot(steps, df["EpRet_mean"], label="EpRet (rolling mean)")
    ax.fill_between(
        steps,
        df["EpRet_mean"] - df["EpRet_std"],
        df["EpRet_mean"] + df["EpRet_std"],
        alpha=0.3,
        label="±1 std",
    )
    ax.set_ylabel("Episode Return")
    ax.set_title("CPO Training (reward & cost over time)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Cost
    ax = axes[1]
    ax.plot(steps, df["EpCost_mean"], label="EpCost (rolling mean)")
    ax.fill_between(
        steps,
        df["EpCost_mean"] - df["EpCost_std"],
        df["EpCost_mean"] + df["EpCost_std"],
        alpha=0.3,
        label="±1 std",
    )
    ax.set_xlabel("Total Env Steps")
    ax.set_ylabel("Episode Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(run_dir, "reward_cost_curves_rolling.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs",
        help="Base directory to search for progress.csv files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size (in epochs) for mean/std.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    window = args.window

    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"Base directory '{base_dir}' does not exist.")

    # Walk the tree and find all progress.csv files
    progress_files = []
    for root, dirs, files in os.walk(base_dir):
        if "progress.csv" in files:
            progress_files.append(os.path.join(root, "progress.csv"))

    if not progress_files:
        print(f"[WARN] No progress.csv files found under {base_dir}")
        return

    print(f"[INFO] Found {len(progress_files)} progress.csv files.")

    for path in progress_files:
        try:
            plot_reward_and_cost_for_run(path, window=window)
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")


if __name__ == "__main__":
    main()
