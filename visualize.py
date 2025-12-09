#!/usr/bin/env python
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------
# EDIT THIS: list of run folder names under runs/simglucose/
# ---------------------------------------------------------------------
RUN_IDS = [
    "adolescent2-v0__ppo__1__1765172126__2025-12-08_00-35-26_lagrangian",
    "adolescent2-v0__ppo__1__1765171548__2025-12-08_00-25-48_lagrangian",
    "adolescent2-v0__ppo__1__1765168393__2025-12-07_23-33-13_lagrangian",
    "adolescent2-v0__ppo__1__1765168278__2025-12-07_23-31-18_lagrangian"
    # add more here...
]

LAG_STEPS = [0]   # e.g. [0, 3, 6] if you want multiple
NUM_SEEDS = 10
PYTHON = "python"  # or "python3" if needed


def run(cmd):
    """Helper that prints and runs a command."""
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    for run_id in RUN_IDS:
        logdir = f"runs/simglucose/{run_id}"
        outdir = f"plots/{run_id}"
        model_full_path = f"{logdir}/adolescent2_ppo.pt"
        model_path = run_id

        # make sure output directory exists
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # 1) export_tensorboard_images.py
        # python export_tensorboard_images.py \
        #   --logdir runs/simglucose/<run_id> \
        #   --out   plots/<run_id>
        run([
            PYTHON, "export_tensorboard_images.py",
            "--logdir", logdir,
            "--out", outdir,
        ])

        # 2) evaluate_with_lag.py for each lag in LAG_STEPS
        # python evaluate_with_lag.py \
        #   --model_full_path runs/simglucose/<run_id>/adolescent2_ppo.pt \
        #   --model ppo \
        #   --model_path <run_id> \
        #   --lag_steps <lag>
        for lag in LAG_STEPS:
            run([
                PYTHON, "evaluate_with_lag.py",
                "--model_full_path", model_full_path,
                "--model", "ppo",
                "--model_path", model_path,
                "--lag_steps", str(lag),
            ])

        # 3) evaluate.py
        # python evaluate.py \
        #   --model_full_path runs/simglucose/<run_id>/adolescent2_ppo.pt \
        #   --model ppo \
        #   --model_path <run_id> \
        #   --num_seeds 10
        run([
            PYTHON, "evaluate.py",
            "--model_full_path", model_full_path,
            "--model", "ppo",
            "--model_path", model_path,
            "--num_seeds", str(NUM_SEEDS),
        ])


if __name__ == "__main__":
    main()
