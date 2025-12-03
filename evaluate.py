import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from simglucose.envs import T1DSimGymnaisumEnv  # Make sure this is available
from cpo import AgentContinuous as CPOAgent # Adjust import for your agent
from ppo import AgentContinuous as PPOAgent  # Adjust import for your agent

# 1. Register the environment for testing (with 24-hour horizon)
def make_test_env(patient="adolescent2", patient_name_hash="adolescent#002"):
    env_id = f"simglucose/{patient}-test-v0"
    # avoid double-registration if you call multiple times
    try:
        gym.spec(env_id)
    except gym.error.Error:
        register(
            id=env_id,  # Create a new test env
            entry_point="simglucose.envs:T1DSimGymnaisumEnv",
            max_episode_steps=288,  # 24 hours (288 × 5 minutes = 24 hours)
            kwargs={"patient_name": patient_name_hash},
        )
    return gym.make(env_id)

# 2. Load trained agent
def load_trained_agent(env,
                       model_path="runs/simglucose/adolescent2-v0__cpo_cleanrl__1__1764654879__2025-12-02_00-54-39/adolescent2_cpo.pt",
                       model="cpo_cleanrl"):
    if model == "cpo":
        ModelAgent = CPOAgent
    else:  # model == "ppo"
        ModelAgent = PPOAgent

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low = env.action_space.low
    act_high = env.action_space.high
    print(f"Action bounds: low {act_low}, high {act_high}")

    agent = ModelAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        clip_actions=True,  # if that’s what you used in training
    )

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)  # load directly to GPU mem (still fine on CPU model)
        print("Loaded model (state_dict) with CUDA available.")
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        print("Loaded model on CPU (CUDA not available).")

    agent.load_state_dict(state_dict)
    agent.eval()
    return agent

# 3. Test the policy on the environment
def evaluate_policy(agent, env, num_steps=288, seed=123):
    obs, _ = env.reset(seed=seed)  # Reset to start
    obs = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)  # Flatten observation

    bg_trajectory = []   # BG over time
    u_trajectory = []    # insulin control over time (actions)

    for step in range(num_steps):
        # Get action from policy (deterministic: no exploration noise)
        with torch.no_grad():
            mu = agent.actor_mean(obs)  # Determine the mean action (no noise)
        action = torch.clamp(mu, agent.act_low, agent.act_high)
        action_np = action.cpu().numpy().reshape(env.action_space.shape)

        # store the (scalar) control, assuming 1D action
        if action_np.size == 1:
            u_trajectory.append(float(action_np.item()))
        else:
            # if multi-dim, you can adjust this to pick the insulin component you care about
            u_trajectory.append(float(action_np[0]))

        # Execute the action in the environment
        raw_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # Record BG (assuming flattened obs; BG is first element)
        bg_value = float(raw_obs[0])
        bg_trajectory.append(bg_value)

        if done:
            break

        # Prepare the next observation
        obs = torch.tensor(raw_obs, dtype=torch.float32).reshape(1, -1)

    return np.array(bg_trajectory), np.array(u_trajectory)

# 4. Visualize BG + control on twin axes
def plot_bg_and_control(bg_trajectory, u_trajectory, save_path="bg_trajectory.png"):
    # time axis in hours (5-min steps)
    timesteps = len(bg_trajectory)
    t = np.arange(timesteps) * 5.0 / 60.0  # hours

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # BG curve (left axis)
    bg_line, = ax1.plot(t, bg_trajectory, label="Blood Glucose (mg/dL)")
    ax1.axhline(70, color="red", linestyle="--", label="70 mg/dL")
    ax1.axhline(180, color="green", linestyle="--", label="180 mg/dL")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Blood Glucose (mg/dL)")
    ax1.grid(True)

    # Insulin control (right axis)
    ax2 = ax1.twinx()
    u_line, = ax2.step(t, u_trajectory, where="post",
                       label="Insulin Control (action)", alpha=0.8)
    ax2.set_ylabel("Control (action units)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("BG and Insulin Control Trajectories (24h)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_bg_and_control_side_by_side(bg_trajectory, u_trajectory, save_path="bg_control_side_by_side.png"):
    # time axis in hours (5-min steps)
    timesteps = len(bg_trajectory)
    t = np.arange(timesteps) * 5.0 / 60.0  # hours

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # ----- Left: BG -----
    ax_bg = axes[0]
    ax_bg.plot(t, bg_trajectory, label="BG (mg/dL)")
    ax_bg.axhline(70, color="red", linestyle="--", label="70 mg/dL")
    ax_bg.axhline(180, color="green", linestyle="--", label="180 mg/dL")
    ax_bg.set_title("Blood Glucose")
    ax_bg.set_xlabel("Time (hours)")
    ax_bg.set_ylabel("BG (mg/dL)")
    ax_bg.grid(True)
    ax_bg.legend(loc="upper right")

    # ----- Right: insulin control -----
    ax_u = axes[1]
    ax_u.step(t, u_trajectory, where="post", label="Insulin control", alpha=0.9)
    ax_u.set_title("Insulin Control (action)")
    ax_u.set_xlabel("Time (hours)")
    ax_u.set_ylabel("Action units")
    ax_u.grid(True)
    ax_u.legend(loc="upper right")

    plt.suptitle("BG and Insulin Control Trajectories (24h)", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cpo",
                        choices=["cpo", "ppo"])
    parser.add_argument("--model_full_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--patient", type=str, default="adolescent2")
    parser.add_argument("--patient_hash", type=str, default="adolescent#002")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_test_env(patient=args.patient, patient_name_hash=args.patient_hash)

    # Load the trained agent
    agent = load_trained_agent(env=env, model_path=args.model_full_path, model=args.model)

    # Evaluate the policy and get BG + control trajectories
    bg_trajectory, u_trajectory = evaluate_policy(agent, env, num_steps=288, seed=args.seed)

    # Visualize
    plot_bg_and_control_side_by_side(
        bg_trajectory,
        u_trajectory,
        save_path=f"plots/{args.model_path}/bg_and_control_trajectory_eval.png",
    )

    env.close()

if __name__ == "__main__":
    main()
