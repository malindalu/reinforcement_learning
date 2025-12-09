# compare_models_ttest.py

import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt  # not really needed but kept in case you extend
import random
from gymnasium.envs.registration import register
from scipy.stats import ttest_rel, ttest_ind

from simglucose.envs import T1DSimGymnaisumEnv  # Make sure this is available
from cpo import AgentContinuous as CPOAgent     # Adjust import for your agent
from ppo import AgentContinuous as PPOAgent     # Adjust import for your agent
from ppo import BGDerivativeWrapper, BGTimeOutsideCostWrapper, BGSmoothControlCostWrapper
from evaluate import make_test_env


# # 1. Register the environment for testing
# def make_test_env(patient="adolescent2", patient_name_hash="adolescent#002", seed=123):
#     np.random.seed(seed)
#     random.seed(seed)
#     env_id = f"simglucose/{patient}-test-v0"
#     # avoid double-registration if you call multiple times
#     try:
#         gym.spec(env_id)
#     except gym.error.Error:
#         register(
#             id=env_id,
#             entry_point="simglucose.envs:T1DSimGymnaisumEnv",
#             max_episode_steps=60,  # keep as in your snippet
#             kwargs={"patient_name": patient_name_hash},
#         )
#     env = gym.make(env_id)
#     # env = BGDerivativeWrapper(env)
#     env = BGTimeOutsideCostWrapper(env)
#     return env


# 2. Load trained agent (same as before but used for both models)
def load_trained_agent(env, model_path, model="cpo"):
    if model == "cpo":
        ModelAgent = CPOAgent
    else:  # "ppo"
        ModelAgent = PPOAgent

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low = env.action_space.low
    act_high = env.action_space.high
    print(f"[{model}] Action bounds: low {act_low}, high {act_high}")

    agent = ModelAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        clip_actions=True,
    )

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
        print(f"[{model}] Loaded model (state_dict) with CUDA available.")
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        print(f"[{model}] Loaded model on CPU (CUDA not available).")

    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def compute_time_in_range(bg_trajectory, low=70.0, high=180.0):
    """
    Returns time-in-range (%) for BG trajectory.

    NOTE: matches your original function: assumes 288 steps for 24h,
    but as long as both models use the same horizon, comparisons are valid.
    """
    bg = np.asarray(bg_trajectory, dtype=float)
    if bg.size == 0:
        return 0.0
    in_range = (bg >= low) & (bg <= high)
    return sum(in_range) / 288 * 100.0


def evaluate_policy(agent, env, num_steps=288, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)

    bg_trajectory = []
    u_trajectory = []

    for step in range(num_steps):
        with torch.no_grad():
            mu = agent.actor_mean(obs)
        action = torch.clamp(mu, agent.act_low, agent.act_high)
        action_np = action.cpu().numpy().reshape(env.action_space.shape)

        if action_np.size == 1:
            u_trajectory.append(float(action_np.item()))
        else:
            u_trajectory.append(float(action_np[0]))

        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        bg_value = float(raw_obs[0])
        bg_trajectory.append(bg_value)

        if done:
            break

        obs = torch.tensor(raw_obs, dtype=torch.float32).reshape(1, -1)

    return np.array(bg_trajectory), np.array(u_trajectory)


def evaluate_model_tir(
    agent,
    patient,
    patient_hash,
    base_seed=123,
    num_seeds=10,
    num_steps=288,
    label="model",
    state = "basic",
    cost = "time_outside",
    reward = "risk",
):
    """
    Run multiple seeds for one model and return TIR array (one TIR per seed).
    """
    tir_list = []

    for i in range(num_seeds):
        this_seed = base_seed + i
        print(f"[{label}] Evaluating seed {this_seed}...")
        env_i = make_test_env(
            patient=patient,
            patient_name_hash=patient_hash,
            seed=this_seed,
            state=state,
            cost =cost,
            reward=reward
        )

        bg_trajectory, _ = evaluate_policy(
            agent, env_i, num_steps=num_steps, seed=this_seed
        )
        env_i.close()

        tir = compute_time_in_range(bg_trajectory)
        tir_list.append(tir)
        print(f"  [{label}] Seed {this_seed}: TIR = {tir:.2f}%")

    return np.asarray(tir_list, dtype=float)


def build_model_full_path(runs_root, model_folder, patient, model_type):
    """
    Helper to construct .pt path from folder name, patient, and model type.

    Example:
        runs_root = 'runs/simglucose'
        model_folder = 'adolescent2-v0__ppo__1__...'
        patient = 'adolescent2'
        model_type = 'ppo'

    -> runs/simglucose/adolescent2-v0__ppo__1__.../adolescent2_ppo.pt
    """
    filename = f"{patient}_{model_type}.pt"
    return os.path.join(runs_root, model_folder, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two models via TIR and t-test across seeds."
    )

    # model folders (under runs_root)
    parser.add_argument(
        "--model1_folder",
        type=str,
        required=True,
        help="Folder name for model 1 under runs_root.",
    )
    parser.add_argument(
        "--model2_folder",
        type=str,
        required=True,
        help="Folder name for model 2 under runs_root.",
    )

    # model types / architectures
    parser.add_argument(
        "--model1_type",
        type=str,
        default="ppo",
        choices=["cpo", "ppo"],
        help="Model 1 type (affects agent class and filename).",
    )
    parser.add_argument(
        "--model2_type",
        type=str,
        default="ppo",
        choices=["cpo", "ppo"],
        help="Model 2 type (affects agent class and filename).",
    )

    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs/simglucose",
        help="Root directory holding model folders.",
    )

    parser.add_argument("--model_1_state", type=str, default="basic") # or deriv
    parser.add_argument("--model_1_cost", type=str, default="time_outside") # or smooth
    parser.add_argument("--model_1_reward", type=str, default="risk") # or proportional or regioned

    parser.add_argument("--model_2_state", type=str, default="basic") # or deriv
    parser.add_argument("--model_2_cost", type=str, default="time_outside") # or smooth
    parser.add_argument("--model_2_reward", type=str, default="risk") # or proportional or regioned

    parser.add_argument("--patient", type=str, default="adolescent2")
    parser.add_argument("--patient_hash", type=str, default="adolescent#002")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="How many seeds (episodes) per model.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=288,
        help="Max steps per evaluation episode.",
    )

    args = parser.parse_args()

    # Build full paths
    model1_full_path = build_model_full_path(
        args.runs_root, args.model1_folder, args.patient, args.model1_type
    )
    model2_full_path = build_model_full_path(
        args.runs_root, args.model2_folder, args.patient, args.model2_type
    )

    print(f"Model 1 path: {model1_full_path}")
    print(f"Model 2 path: {model2_full_path}")

    # Create a template env just to size the networks
    env_template_1 = make_test_env(
        patient=args.patient,
        patient_name_hash=args.patient_hash,
        seed=args.seed,
        state=args.model_1_state,
        cost = args.model_1_cost,
        reward = args.model_1_reward
    )

    env_template_2 = make_test_env(
        patient=args.patient,
        patient_name_hash=args.patient_hash,
        seed=args.seed,
        state=args.model_2_state,
        cost = args.model_2_cost,
        reward = args.model_2_reward
    )


    # Load agents
    agent1 = load_trained_agent(
        env=env_template_1,
        model_path=model1_full_path,
        model=args.model1_type,
    )
    agent2 = load_trained_agent(
        env=env_template_2,
        model_path=model2_full_path,
        model=args.model2_type,
    )
    env_template_1.close()
    env_template_2.close()

    # Evaluate both models on the same seeds
    tir1 = evaluate_model_tir(
        agent1,
        patient=args.patient,
        patient_hash=args.patient_hash,
        base_seed=args.seed,
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        label=f"{args.model1_type}_1",
        state=args.model_1_state,
        cost = args.model_1_cost,
        reward = args.model_1_reward
    )
    tir2 = evaluate_model_tir(
        agent2,
        patient=args.patient,
        patient_hash=args.patient_hash,
        base_seed=args.seed,
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        label=f"{args.model2_type}_2",
        state=args.model_2_state,
        cost = args.model_2_cost,
        reward = args.model_2_reward
    )

    print("\n===== TIR Summary =====")
    print(f"Model 1 ({args.model1_type}): mean={tir1.mean():.2f}%, std={tir1.std(ddof=1):.2f}%")
    print(f"Model 2 ({args.model2_type}): mean={tir2.mean():.2f}%, std={tir2.std(ddof=1):.2f}%")

    # Paired t-test (since same seeds / scenarios)
    t_rel, p_rel = ttest_rel(tir1, tir2)
    # Also independent t-test with unequal variances, just in case
    t_ind, p_ind = ttest_ind(tir1, tir2, equal_var=False)

    print("\n===== Paired t-test on TIR (ttest_rel; same seeds) =====")
    print(f"t = {t_rel:.4f}, p = {p_rel:.4e}")

    print("\n===== Independent t-test on TIR (Welch; ttest_ind) =====")
    print(f"t = {t_ind:.4f}, p = {p_ind:.4e}")

    # Optional: simple effect size (Cohen's d for paired samples)
    diff = tir1 - tir2
    d = diff.mean() / diff.std(ddof=1)
    print(f"\nApprox. Cohen's d (paired) = {d:.3f}")


if __name__ == "__main__":
    main()
