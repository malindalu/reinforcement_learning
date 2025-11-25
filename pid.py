# pid_baseline.py
import gymnasium
import numpy as np
from gymnasium.envs.registration import register

class PIDController:
    """Simple PID controller for insulin dosing."""

    def __init__(self, Kp=0.1, Ki=0.001, Kd=0.05, target_bg=110):
        self.Kp = Kp # proportional gain e.g. bigger correction for bigger error
        self.Ki = Ki # integral gain (accumulated error) e.g. can correct if overall slowly drifting
        self.Kd = Kd # derivative gain (rate of change) e.g. can slow down if BG changes too fast
        self.target = target_bg

        self.integral = 0
        self.prev_error = 0

    def compute_action(self, bg):
        error = bg - self.target
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        action = max(0, action)  # insulin cannot be negative
        return action

def run_episode(env, controller, render=False, max_steps=200):
    obs, info = env.reset()
    t = 0
    total_reward = 0

    while t < max_steps:
        bg = obs[0]
        action = controller.compute_action(bg)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
        if terminated or truncated:
            break
        t += 1

    return total_reward, t

def main():

    register(
        id="simglucose/adolescent2-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=10,
        kwargs={"patient_name": "adolescent#002"},
    )

    env = gymnasium.make(
        "simglucose/adolescent2-v0",
        render_mode=None,  # set 'human' if you want plots
    )

    # Initialize PID
    # TODO: either hand tune these better for simple intelligent method
    pid = PIDController(Kp=0.05, Ki=0.001, Kd=0.01, target_bg=110)

    # Run simulation
    reward, steps = run_episode(env, pid)
    print(f"Episode finished in {steps} steps, total reward: {reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
