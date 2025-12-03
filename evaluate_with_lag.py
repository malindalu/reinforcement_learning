import torch
import numpy as np
import argparse
from evaluate import make_test_env, load_trained_agent, plot_bg_and_control_side_by_side

def evaluate_policy(agent, env, num_steps=288, seed=123, lag_steps=0):
    """
    Evaluate a policy with an optional action delay.

    lag_steps = 0  -> no delay (current action applied immediately)
    lag_steps = k  -> action computed at time t is applied at time t + k

    For t < k, the applied action is 0 (clipped into the action space).
    """
    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)

    bg_trajectory = []   # BG over time
    u_trajectory = []    # actually APPLIED insulin control over time

    # --- set up lag buffer ---
    lag_steps = max(0, int(lag_steps))
    # "no insulin" baseline action, then clip into valid Box range
    zero_action_np = np.zeros_like(env.action_space.low, dtype=np.float32)
    zero_action_np = np.clip(zero_action_np, env.action_space.low, env.action_space.high)

    # buffer holds the *planned* actions that will be applied in the future
    # initialize with zeros so the first `lag_steps` applied actions are 0
    action_buffer = [zero_action_np.copy() for _ in range(lag_steps)]

    for step in range(num_steps):
        # 1) Policy looks at CURRENT obs and proposes an action
        with torch.no_grad():
            mu = agent.actor_mean(obs)
        action = torch.clamp(mu, agent.act_low, agent.act_high)
        planned_action_np = action.cpu().numpy().reshape(env.action_space.shape)

        # 2) Decide what to actually APPLY to the env this step
        if lag_steps == 0:
            applied_action_np = planned_action_np
        else:
            # oldest buffered action is applied now
            applied_action_np = action_buffer.pop(0)
            # newly computed action enters the buffer and will be applied later
            action_buffer.append(planned_action_np)

        # 3) Log the *applied* insulin for plotting
        if applied_action_np.size == 1:
            u_trajectory.append(float(applied_action_np.item()))
        else:
            u_trajectory.append(float(applied_action_np[0]))

        # 4) Step the environment with the APPLIED action
        raw_obs, reward, terminated, truncated, info = env.step(applied_action_np)
        done = terminated or truncated

        # Record BG (assuming first element is BG)
        bg_value = float(raw_obs[0])
        bg_trajectory.append(bg_value)

        if done:
            break

        obs = torch.tensor(raw_obs, dtype=torch.float32).reshape(1, -1)

    return np.array(bg_trajectory), np.array(u_trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cpo_cleanrl",
                        choices=["cpo_cleanrl", "ppo"])
    parser.add_argument("--model_full_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--patient", type=str, default="adolescent2")
    parser.add_argument("--patient_hash", type=str, default="adolescent#002")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--lag_steps",
        type=int,
        default=0,
        help="Action delay (in env steps). 0 = no delay.",
    )
    args = parser.parse_args()

    env = make_test_env(patient=args.patient, patient_name_hash=args.patient_hash)

    # Load the trained agent
    agent = load_trained_agent(env=env, model_path=args.model_full_path, model=args.model)

    # Evaluate the policy and get BG + control trajectories
    bg_trajectory, u_trajectory = evaluate_policy(
        agent,
        env,
        num_steps=288,
        seed=args.seed,
        lag_steps=args.lag_steps,
    )

    # Visualize
    plot_bg_and_control_side_by_side(
        bg_trajectory,
        u_trajectory,
        save_path=f"plots/{args.model_path}/bg_and_control_trajectory_eval_lag{args.lag_steps}.png",
    )

    env.close()

if __name__ == "__main__":
    main()
