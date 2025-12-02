# cpo_simglucose.py
# CleanRL-style PPO adapted with CPO (Lagrangian constrained PPO)
# Continuous action space for SimGlucose

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.registration import register

# ----------------------------
# CLI / hyperparameters
# ----------------------------
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "simglucose/adolescent2-v0"
    patient: str = "adolescent2"
    patient_name_hash: str = "adolescent#002"
    total_timesteps: int = 1000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # CPO / Lagrangian
    cost_threshold: float = 1.0
    cost_lr: float = 1e-2

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    clip_actions: bool = False


# ----------------------------
# Environment setup
# ----------------------------
def make_env(env_id, patient, patient_name_hash, render_mode=None):
    register(
        id=f"simglucose/{patient}-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=288,
        kwargs={"patient_name": patient_name_hash},
    )
    env_id = f"simglucose/{patient}-v0"

    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        return env
    return thunk


# ----------------------------
# Layer init helper
# ----------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ----------------------------
# Agent with cost critic
# ----------------------------
class AgentContinuous(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, clip_actions=False):
        super().__init__()
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Cost Critic
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))
        self.clip_actions = clip_actions

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_cost_value(self, x):
        return self.cost_critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        mu = self.actor_mean(x)
        std = torch.exp(self.logstd).expand_as(mu)
        dist = Normal(mu, std)
        if action is None:
            action = dist.rsample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)
        if self.clip_actions:
            action = torch.max(torch.min(action, self.act_high), self.act_low)
        return action, logprob, entropy, value


# ----------------------------
# Cost function
# ----------------------------
def cost_fn(next_obs):
    BG = next_obs[:, 0]

    hypo_cost  = torch.clamp(70 - BG, min=0)   
    hyper_cost = torch.clamp(BG - 180, min=0)

    return hypo_cost + hyper_cost


# ----------------------------
# Main training loop
# ----------------------------
def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{timestamp}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %
        ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.patient, args.patient_name_hash, None) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high

    agent = AgentContinuous(obs_dim, act_dim, act_low, act_high, clip_actions=args.clip_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    lambda_cost = torch.tensor(1.0, requires_grad=False, device=device)

    # STORAGE
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    episode_returns = torch.zeros(args.num_envs).to(device)
    episode_lengths = torch.zeros(args.num_envs).to(device)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                flat_obs = next_obs.reshape(args.num_envs, -1)
                action, logprob, _, value = agent.get_action_and_value(flat_obs)
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            action_np = action.cpu().numpy()
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action_np)
            next_done = np.logical_or(terminations, truncations)

            next_obs = torch.Tensor(next_obs_np).to(device)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32).to(device)
            costs[step] = cost_fn(next_obs)
            next_done = torch.Tensor(next_done).to(device)

            episode_returns += rewards[step]
            episode_lengths += 1
            for i, done_flag in enumerate(next_done):
                if done_flag:
                    writer.add_scalar("charts/episodic_return", episode_returns[i].item(), global_step)
                    writer.add_scalar("charts/episodic_length", episode_lengths[i].item(), global_step)
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        rewards = (rewards - reward_mean) / reward_std

        cost_mean = costs.mean()
        cost_std = costs.std() + 1e-8
        costs = (costs - cost_mean) / cost_std

        # --- GAE for rewards ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs.reshape(args.num_envs, -1)).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        # --- GAE for costs ---
        with torch.no_grad():
            next_cost_value = agent.get_cost_value(next_obs.reshape(args.num_envs, -1)).reshape(1, -1)
        cost_advantages = torch.zeros_like(costs).to(device)
        last_cost_lam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextcost = next_cost_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextcost = agent.get_cost_value(obs[t + 1].reshape(args.num_envs, -1))
            delta = costs[t] + args.gamma * nextcost * nextnonterminal - agent.get_cost_value(obs[t].reshape(args.num_envs, -1))
            last_cost_lam = delta + args.gamma * args.gae_lambda * nextnonterminal * last_cost_lam
            cost_advantages[t] = last_cost_lam
        cost_returns = cost_advantages + agent.get_cost_value(obs.reshape(args.num_steps, args.num_envs, -1))

        # flatten batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1).detach()
        b_returns = returns.reshape(-1).detach()
        b_cost_adv = cost_advantages.reshape(-1).detach()
        b_values = values.reshape(-1).detach()

        batch_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = b_obs[mb_inds].to(device).reshape(len(mb_inds), -1)
                mb_actions = b_actions[mb_inds].to(device).reshape(len(mb_inds), -1)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)

                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = logratio.exp()

                # PPO policy loss
                mb_advantages = b_advantages[mb_inds].to(device)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds].to(device)) ** 2
                    v_clipped = b_values[mb_inds].to(device) + torch.clamp(newvalue - b_values[mb_inds].to(device),
                                                                            -args.clip_coef, args.clip_coef)
                    v_loss_max = torch.max(v_loss_unclipped, v_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()

                # entropy
                entropy_loss = entropy.mean()

                # cost penalty (detach cost advantages)
                mb_cost_adv = b_cost_adv[mb_inds].to(device)
                if args.norm_adv:
                    mb_cost_adv = (mb_cost_adv - mb_cost_adv.mean()) / (mb_cost_adv.std() + 1e-8)
                cost_penalty = (lambda_cost.detach() * mb_cost_adv).mean()

                # total loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + cost_penalty

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # update lambda (detach mb_cost_adv for safe update)
                batch_expected_cost = mb_cost_adv.mean().detach()
                lambda_cost += args.cost_lr * (batch_expected_cost - args.cost_threshold)
                lambda_cost = torch.clamp(lambda_cost, 0, 1000)


        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/cost_lambda", lambda_cost.item(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()
    model_file = os.path.join(f"runs/{run_name}", f"{args.patient}_cpo.pt")
    torch.save(agent.state_dict(), model_file)
    print(f"Saved model to {model_file}")


if __name__ == "__main__":
    main()
