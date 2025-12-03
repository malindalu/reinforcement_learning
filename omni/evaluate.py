# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Evaluate a saved OmniSafe policy from a given log directory."""

import os
import argparse

import omnisafe
from gymnasium.envs.registration import register


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help=(
            "Path to the experiment log directory, e.g. "
            "'~/omnisafe/examples/runs/PPOLag-SafetyPointGoal1-v0/seed-000-2023-03-07-20-25-48'"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to render/evaluate per checkpoint.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Render mode passed to omnisafe.Evaluator.",
    )
    args = parser.parse_args()

    log_dir = os.path.expanduser(args.log_dir)
    torch_save_dir = os.path.join(log_dir, "torch_save")

    if not os.path.isdir(torch_save_dir):
        raise FileNotFoundError(f"'torch_save' directory not found at: {torch_save_dir}")

    evaluator = omnisafe.Evaluator(render_mode=args.render_mode)

    # Loop through all saved checkpoints (*.pt) in torch_save
    with os.scandir(torch_save_dir) as it:
        for item in it:
            if item.is_file() and item.name.endswith(".pt"):
                print(f"\n=== Evaluating checkpoint: {item.name} ===")
                evaluator.load_saved(
                    save_dir=log_dir,
                    model_name=item.name,
                    camera_name="track",
                    width=256,
                    height=256,
                )
                # will render and then run evaluation rollouts
                evaluator.render(num_episodes=args.episodes)
                evaluator.evaluate(num_episodes=args.episodes)


if __name__ == "__main__":
    register(
            id="simglucose/adolescent2-v0",
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=288,
            kwargs={"patient_name": "adolescent#002"},
        )
    main()
