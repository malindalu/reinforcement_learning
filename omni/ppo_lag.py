from __future__ import annotations

from typing import Any, ClassVar

import torch
import gymnasium as gym
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register

from gymnasium.envs.registration import register
import numpy as np
import tyro
from typing import Optional
from dataclasses import dataclass

import omnisafe
from cpo import *

def build_config(args: Args):
    return {
        "seed": args.seed,

        "algo_cfgs": {
            # "cost_limit": args.cost_limit,     # constraint C ≤ limit
        },

        "train_cfgs": {
            "total_steps": args.total_steps,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },

        # "env_cfgs": {
        #     "normalize_obs": args.normalize_obs,
        #     "normalize_rew": args.normalize_rew,
        #     "normalize_cost": args.normalize_cost,
        # },

        "logger_cfgs": {
            "use_wandb": args.use_wandb,
        },
    }

def main():
    args = tyro.cli(Args)
    cfgs = build_config(args)

    print("Launching OmniSafe with config:")
    print(cfgs)
    # Instantiate your agent
    agent = omnisafe.Agent(
        algo='PPOLag',                   # Algorithm
        env_id= 'simglucose/adolescent2-v0',  # Your custom environment
        custom_cfgs=cfgs,
    )

    # Train the agent
    agent.learn()

if __name__ == "__main__":
    main()
