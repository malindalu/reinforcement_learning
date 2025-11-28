import os
import time
import random
import argparse
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from collections import deque


from gymnasium.envs.registration import register


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu = self.actor(x)
        std = torch.exp(self.logstd)
        dist = Normal(mu, std)
        
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        value = self.critic(x)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value
        

# --- Environment Factory --------------------------------------------------

def make_simglucose_env(patient="adolescent2", render_mode=None):
    register(
        id="simglucose/adolescent2-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=10,
        kwargs={"patient_name": "adolescent#002"},
    )

    env_id = f"simglucose/{patient}-v0"
    env = gym.make(env_id, render_mode=render_mode)

    # Flatten dict observation → continuous vector
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env


# --- Main PPO Training ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="adolescent2")
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    env = make_simglucose_env(args.patient, render_mode=None)
    obs_shape = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_shape, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # PPO Buffers
    num_steps = 2048
    batch_size = 64
    minibatch = num_steps // batch_size
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 10

    obs = np.zeros((num_steps, obs_shape))
    actions = np.zeros((num_steps, act_dim))
    logprobs = np.zeros(num_steps)
    rewards = np.zeros(num_steps)
    dones = np.zeros(num_steps)
    values = np.zeros(num_steps)

    next_obs, info = env.reset()
    next_done = False
    global_step = 0

    # Training Loop
    for step in range(num_steps * (args.total_timesteps // num_steps)):
        global_step += 1
        obs[step % num_steps] = next_obs
        dones[step % num_steps] = next_done

        # Agent step
        obs_tensor = torch.tensor(next_obs, device=device).float()
        with torch.no_grad():
            action, logprob, entropy, value = agent.get_action_and_value(obs_tensor)

        a = action.cpu().item()
        next_obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        actions[step % num_steps] = a
        logprobs[step % num_steps] = logprob.cpu().item()
        rewards[step % num_steps] = reward
        values[step % num_steps] = value.cpu().item()

        next_done = done
        if done:
            next_obs, info = env.reset()

        # PPO update every 2048 steps
        if step % num_steps == 0 and step > 0:
            with torch.no_grad():
                next_value = agent.get_value(torch.tensor(next_obs).float().to(device)).cpu().item()

            # --- Compute GAE-Lambda ---
            advantages = np.zeros(num_steps)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value if t == num_steps - 1 else values[t + 1]
                delta = rewards[t] + gamma * next_nonterminal * next_val - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam

            returns = advantages + values

            # Convert buffers to torch
            b_obs = torch.tensor(obs, dtype=torch.float32).to(device)
            b_actions = torch.tensor(actions, dtype=torch.float32).to(device)
            b_logprobs = torch.tensor(logprobs).to(device)
            b_returns = torch.tensor(returns).to(device)
            b_advantages = torch.tensor(advantages).to(device)

            # PPO update
            inds = np.arange(num_steps)
            for _ in range(update_epochs):
                np.random.shuffle(inds)
                for start in range(0, num_steps, batch_size):
                    end = start + batch_size
                    mb_inds = inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                    pg1 = ratio * b_advantages[mb_inds]
                    pg2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * b_advantages[mb_inds]
                    policy_loss = -torch.min(pg1, pg2).mean()

                    value_loss = ((newvalue.flatten() - b_returns[mb_inds]) ** 2).mean()

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f"Step {global_step}, PPO updated.")

    print("Training complete!")
    torch.save(agent.state_dict(), "ppo_simglucose_cleanrl.pt")


if __name__ == "__main__":
    main()
