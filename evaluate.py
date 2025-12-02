import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from simglucose.envs import T1DSimGymnaisumEnv  # Make sure this is available
from cpo_cleanrl import AgentContinuous  # Adjust import for your agent

# 1. Register the environment for testing (with 24-hour horizon)
def make_test_env(patient_name="adolescent#002"):
    register(
        id="simglucose/adolescent2-test-v0",  # Create a new test env
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=288,  # 24 hours (288 × 5 minutes = 24 hours)
        kwargs={"patient_name": patient_name},
    )
    return gym.make("simglucose/adolescent2-test-v0")

# 2. Load trained agent
def load_trained_agent(model_path="runs/simglucose/adolescent2-v0__cpo_cleanrl__1__1764654879__2025-12-02_00-54-39/adolescent2_cpo.pt"):
    # Initialize your agent, matching the architecture used during training
    agent = AgentContinuous(obs_dim=1, act_dim=1, act_low=-1, act_high=1)  # Adjust as necessary
    agent.load_state_dict(torch.load(model_path))  # Load the model
    agent.eval()  # Switch to evaluation mode
    return agent

# 3. Test the policy on the environment
def evaluate_policy(agent, env, num_steps=288):
    obs, _ = env.reset()  # Reset to start
    obs = torch.Tensor(obs).reshape(1, -1)  # Flatten observation
    bg_trajectory = []  # To store BG over time

    for step in range(num_steps):
        # Get action from policy (deterministic: no exploration noise)
        with torch.no_grad():
            mu = agent.actor_mean(obs)  # Determine the mean action (no noise)
        action = mu.cpu().numpy()  # Convert to numpy array

        # Execute the action in the environment
        raw_obs, reward, done, trunc, info = env.step(action)

        # Record BG (assuming BG is in the "BG" field of the raw observation)
        bg_value = raw_obs[0]  # Or raw_obs["blood_glucose"], depending on version
        bg_trajectory.append(bg_value)

        # Prepare the next observation
        obs = torch.Tensor(raw_obs).reshape(1, -1)

    return bg_trajectory

# 4. Visualize the BG trajectory
def plot_bg_trajectory(bg_trajectory):
    # BG plot over 24 hours
    plt.figure(figsize=(12, 6))
    plt.plot(bg_trajectory, label="Blood Glucose (mg/dL)")
    plt.axhline(70, color="red", linestyle="--", label="Hypoglycemia (70 mg/dL)")
    plt.axhline(180, color="green", linestyle="--", label="Hyperglycemia (180 mg/dL)")
    plt.title("Blood Glucose (BG) Trajectory - 24 Hours")
    plt.xlabel("Time (5 min intervals)")
    plt.ylabel("Blood Glucose (mg/dL)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def main():
    # Load the trained agent
    agent = load_trained_agent("runs/simglucose/adolescent2-v0__cpo_cleanrl__1__1764654879__2025-12-02_00-54-39/adolescent2_cpo.pt")

    # Initialize the environment for testing (24-hour duration)
    env = make_test_env(patient_name="adolescent#002")

    # Evaluate the policy and get BG trajectory
    bg_trajectory = evaluate_policy(agent, env, num_steps=288)

    # Visualize the BG trajectory
    plot_bg_trajectory(bg_trajectory)

    env.close()

if __name__ == "__main__":
    main()
