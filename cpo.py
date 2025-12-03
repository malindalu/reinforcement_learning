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
    cost_threshold: float = 1.0     # desired average discounted cost per step
    cost_lr: float = 1e-2           # step size for lambda update
    cost_method: str = "time_outside" 

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    clip_actions: bool = False


# ----------------------------
# Environment setup (matches PPO)
# ----------------------------
def make_env(env_id, patient, patient_name_hash, render_mode=None):
    """Same registration pattern as the working PPO code."""
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
# Layer init helper (matches PPO)
# ----------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ----------------------------
# Agent with cost critic (PPO-style + extra head)
# ----------------------------
class AgentContinuous(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, clip_actions=False):
        super().__init__()
        # Reward critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Cost critic
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor (same structure as PPO, just no initial bias here by default)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(act_dim))

        # action bounds (for optional clipping)
        self.register_buffer("act_low", torch.tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(act_high, dtype=torch.float32))
        self.clip_actions = clip_actions

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_cost_value(self, x):
        return self.cost_critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        """
        x: [batch, obs_dim]
        Returns: action, logprob, entropy, value
        """
        mu = self.actor_mean(x)
        std = torch.exp(self.logstd).expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            action = dist.rsample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(x)

        if self.clip_actions:
            action = torch.max(torch.min(action, self.act_high), self.act_low)

        return action, logprob, entropy, value


# ----------------------------
# Cost function in BG space (same semantics as your original)
# ----------------------------
def cost_proportion_to_condition(next_obs: torch.Tensor) -> torch.Tensor:
    """
    next_obs: [num_envs, obs_dim] tensor on device
    Uses BG in column 0 and measures magnitude of hypo/hyper.
    """
    BG = next_obs[:, 0]
    hypo_cost = torch.clamp(70.0 - BG, min=0.0)
    hyper_cost = torch.clamp(BG - 180.0, min=0.0)
    return hypo_cost + hyper_cost

def cost_time_outside(next_obs: torch.Tensor) -> torch.Tensor:
    BG = next_obs[:, 0]
    return ((BG < 70.0) | (BG > 180.0)).float()


# ----------------------------
# Main training loop (PPO-style structure + CPO bits)
# ----------------------------
def main():
    args = tyro.cli(Args)

    # derived sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    cost_fn = cost_time_outside if args.cost_method == "time_outside" else cost_proportion_to_condition

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{timestamp}"

    # hyperparams dump (same as PPO)
    run_dir = f"runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    hyperparams_file = os.path.join(run_dir, "hyperparams.txt")
    with open(hyperparams_file, "w") as f:
        f.write("=== RUN PARAMETERS ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("=======================\n")

    # optional WandB
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

    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding (same pattern as PPO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # vectorized envs (same as PPO)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.patient, args.patient_name_hash, None) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Expect Box action space."

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low = envs.single_action_space.low
    act_high = envs.single_action_space.high

    agent = AgentContinuous(obs_dim, act_dim, act_low, act_high, clip_actions=args.clip_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Lagrange multiplier for cost
    lambda_cost = torch.tensor(1.0, device=device)

    # STORAGE (matches PPO style, plus cost + cost_values)
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cost_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    episode_returns = torch.zeros(args.num_envs).to(device)
    episode_lengths = torch.zeros(args.num_envs).to(device)
    episode_costs = torch.zeros(args.num_envs).to(device)

    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing (same as PPO)
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
                cost_value = agent.get_cost_value(flat_obs)

            values[step] = value
            cost_values[step] = cost_value
            actions[step] = action
            logprobs[step] = logprob

            action_np = action.cpu().numpy()
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action_np)
            next_done = np.logical_or(terminations, truncations)

            next_obs = torch.Tensor(next_obs_np).to(device)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32).to(device)
            costs[step] = cost_fn(next_obs).view(args.num_envs, -1).squeeze(-1)
            next_done = torch.Tensor(next_done).to(device)

            episode_returns += rewards[step]
            episode_costs += costs[step]
            episode_lengths += 1

            for i, done_flag in enumerate(next_done):
                if done_flag:
                    writer.add_scalar("charts/episodic_return", episode_returns[i].item(), global_step)
                    writer.add_scalar("charts/episodic_cost", episode_costs[i].item(), global_step)
                    writer.add_scalar("charts/episodic_length", episode_lengths[i].item(), global_step)
                    episode_returns[i] = 0.0
                    episode_costs[i] = 0.0
                    episode_lengths[i] = 0.0

        # --- GAE for rewards (same structure as PPO) ---
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

        # --- GAE for costs (parallel to rewards, but using cost_values) ---
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
                nextcost = cost_values[t + 1]
            delta_cost = costs[t] + args.gamma * nextcost * nextnonterminal - cost_values[t]
            last_cost_lam = delta_cost + args.gamma * args.gae_lambda * nextnonterminal * last_cost_lam
            cost_advantages[t] = last_cost_lam

        cost_returns = cost_advantages + cost_values

        # flatten batch (PPO-style)
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_cost_adv = cost_advantages.reshape(-1)
        b_cost_returns = cost_returns.reshape(-1)

        # PPO/CPO update
        batch_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = b_obs[mb_inds].to(device).reshape(len(mb_inds), -1)
                mb_actions = b_actions[mb_inds].to(device).reshape(len(mb_inds), -1)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                new_cost_value = agent.get_cost_value(mb_obs)

                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # reward advantages
                mb_advantages = b_advantages[mb_inds].to(device)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO policy loss (reward part)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss for reward critic
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds].to(device)) ** 2
                    v_clipped = b_values[mb_inds].to(device) + torch.clamp(
                        newvalue - b_values[mb_inds].to(device),
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds].to(device)) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()

                # cost advantage (for penalty and cost value loss)
                mb_cost_adv = b_cost_adv[mb_inds].to(device)
                mb_cost_returns = b_cost_returns[mb_inds].to(device)
                if args.norm_adv:
                    mb_cost_adv = (mb_cost_adv - mb_cost_adv.mean()) / (mb_cost_adv.std() + 1e-8)

                # cost penalty term (Lagrangian)
                cost_penalty = (lambda_cost.detach() * mb_cost_adv).mean()

                # cost value loss (train cost_critic)
                cost_v_loss = 0.5 * ((new_cost_value.view(-1) - mb_cost_returns) ** 2).mean()

                # entropy
                entropy_loss = entropy.mean()

                # total loss
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + cost_v_loss   # train cost critic
                    + cost_penalty  # enforce constraint
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # update lambda (using raw discounted costs)
                batch_expected_cost = mb_cost_returns.mean().detach()
                lambda_cost = lambda_cost + args.cost_lr * (batch_expected_cost - args.cost_threshold)
                lambda_cost = torch.clamp(lambda_cost, 0.0, 1e6)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # diagnostics (similar to PPO)
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/cost_value_loss", cost_v_loss.item(), global_step)
        writer.add_scalar("charts/lagrange_lambda_cost", lambda_cost.item(), global_step)
        writer.add_scalar("charts/batch_avg_cost", costs.mean().item(), global_step)

        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()

    model_file = os.path.join(run_dir, f"{args.patient}_cpo.pt")
    torch.save(agent.state_dict(), model_file)
    print(f"Saved model to {model_file}")


if __name__ == "__main__":
    main()
