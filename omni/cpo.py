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

import omnisafe

class Args:
    # Experiment + environment settings
    algo: str = "CPO"                     # CPO, PCPO, PPO-Lag, etc.
    env_id: str = "simglucose/adolescent2-v0"

    patient: str = "adolescent2"
    patient_name_hash: str = "adolescent#002"

    # RL settings
    seed: int = 0
    total_steps: int = 500000
    cost_limit: float = 0.0

    # Logging settings
    use_wandb: bool = False
    wandb_project: str = "SimGlucose-CPO"
    wandb_entity: Optional[str] = None

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

    @property
    def max_episode_steps(self) -> int:
        return 288  # typical simglucose episode length
    
    def compute_cost(self,obs) -> float:
        """Return a cost for unsafe glucose levels."""
        bg = obs[0].item()
        # Example: penalize hypoglycemia (<70) and hyperglycemia (>180)
        if bg < 70 or bg > 180:
            return 1.0
        return 0.0

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action.numpy())
        # Define cost for safe RL; e.g., glucose out-of-range risk
        cost = torch.as_tensor(self.compute_cost(obs), dtype=torch.float32)
        print(cost)

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
        return torch.as_tensor(obs, dtype=torch.float32), info

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        return self._env.render()
    
    def set_seed(self, seed: int | None = None) -> None:
        """Set the RNG seed for reproducibility."""
        import random, numpy as np
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # If your simglucose env supports gym seeding:
            self._env.reset(seed=seed)


def build_config(args: Args):
    return {
        "seed": args.seed,

        "algo_cfgs": {
            "cost_limit": args.cost_limit,     # constraint C ≤ limit
        },

        "train_cfgs": {
            "total_steps": args.total_steps,
        },

        "logger_cfgs": {
            "use_wandb": args.use_wandb,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
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
