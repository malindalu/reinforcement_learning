# Copyright 2025 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OmniSafe CMDP wrapper for simglucose adolescent environment."""

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

@dataclass
class Args:
    # Experiment + environment settings
    algo: str = "CPO"                     # CPO, PCPO, PPO-Lag, etc.
    env_id: str = "simglucose/adolescent2-v0"

    patient: str = "adolescent2"
    patient_name_hash: str = "adolescent#002"

    # RL settings
    seed: int = 0
    total_steps: int = 500000
    cost_limit: float = 10.0

    # Logging settings
    use_wandb: bool = False
    wandb_project: str = "SimGlucose-CPO"
    wandb_entity: Optional[str] = None

    normalize_obs: bool = False
    normalize_rew: bool = False
    normalize_cost: bool = False

@env_register
class SimGlucoseAdolescentEnv(CMDP):
    """CMDP wrapper for simglucose adolescent environment."""

    _support_envs: ClassVar[list[str]] = ['simglucose/adolescent2-v0']

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        # Create underlying gym environment
        register(
            id="simglucose/adolescent2-v0",
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=288,
            kwargs={"patient_name": "adolescent#002"},
        )
        self._env = gym.make("simglucose/adolescent2-v0", render_mode="human")
        self._count = 0

        # Convert Gym spaces to Torch spaces
        obs_shape = self._env.observation_space.shape
        act_shape = self._env.action_space.shape

        self._observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=obs_shape, dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=np.array(self._env.action_space.low, dtype=np.float32),
            high=np.array(self._env.action_space.high, dtype=np.float32),
            shape=act_shape,
            dtype=np.float32
        )
        self._num_envs = 1

        # --- new state for cost + rendering ---
        self._prev_bg: float | None = None
        self.cost_scale: float = 1.0
        self._bg_history: list[float] = []

    @property
    def max_episode_steps(self) -> int:
        return 288  # typical simglucose episode length
    
    def compute_cost(self, obs) -> float:
        """Return a cost for unsafe glucose levels based on BG changes."""
        bg = float(obs[0])
        if self._prev_bg is None:
            cost = 0.0
        else:
            diff = bg - self._prev_bg
            cost = self.cost_scale * float(diff ** 2)
        self._prev_bg = bg
        return cost

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action.numpy())
        self._last_obs = np.array(obs, copy=True)

        # track BG history for rendering
        bg = float(obs[0])
        self._bg_history.append(bg)

        # Define cost for safe RL; e.g., glucose out-of-range risk surrogate
        cost = torch.as_tensor(self.compute_cost(obs), dtype=torch.float32)

        self._count += 1
        truncated = torch.as_tensor(truncated or self._count >= self.max_episode_steps)
        terminated = torch.as_tensor(terminated)

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            cost,
            terminated,
            truncated,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict]:
        obs, info = self._env.reset(seed=seed)
        self._count = 0
        self._prev_bg = None

        # reset BG history and start with initial BG
        self._bg_history = [float(obs[0])]

        return torch.as_tensor(obs, dtype=torch.float32), info

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        """
        Return an RGB frame of BG over time with horizontal bounds at 70 (red)
        and 180 (green).
        """
        import matplotlib.pyplot as plt
        plt.use("Agg")

        # If no history yet but we have an observation, seed it
        if not self._bg_history and self._last_obs is not None:
            self._bg_history.append(float(self._last_obs[0]))

        # Make sure we have something to show
        if not self._bg_history:
            return None

        bg_vals = np.array(self._bg_history, dtype=float)
        x = np.arange(len(bg_vals))

        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        ax.plot(x, bg_vals, linewidth=2)

        # Bounds: 70 (red) and 180 (green)
        ax.axhline(70.0, color="red", linestyle="--", linewidth=1.5, label="70 mg/dL")
        ax.axhline(180.0, color="green", linestyle="--", linewidth=1.5, label="180 mg/dL")

        ax.set_xlabel("Time step")
        ax.set_ylabel("BG (mg/dL)")
        ax.set_title("Blood Glucose Trace")
        # Set a reasonable y-limit
        ymin = min(bg_vals.min(), 50.0)
        ymax = max(bg_vals.max(), 250.0)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        # Convert figure to RGB array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(h, w, 3)
        plt.close(fig)

        return frame
    
    def set_seed(self, seed: int | None = None) -> None:
        """Set the RNG seed for reproducibility."""
        import random, numpy as np
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._env.reset(seed=seed)

def build_config(args: Args):
    return {
        "seed": args.seed,

        "algo_cfgs": {
            "cost_limit": args.cost_limit,     # constraint C ≤ limit
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
        algo='CPO',                   # Algorithm
        env_id= 'simglucose/adolescent2-v0',  # Your custom environment
        custom_cfgs=cfgs,
    )

    # Train the agent
    agent.learn()

if __name__ == "__main__":
    main()
