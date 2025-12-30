import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from agent import DQNAgent, RecurrentDQNAgent
from controllers import PIDController
from environment_3d import LineFollowEnv3D

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

RUN_DIR = Path(__file__).parent / "run_logs"
RUN_DIR.mkdir(exist_ok=True)


class PolicyWrapper:
    def __init__(self, name, act_fn, reset_fn=None):
        self.name = name
        self._act = act_fn
        self._reset = reset_fn or (lambda: None)

    def reset(self):
        self._reset()

    def act(self, obs, offset, heading_error, env):
        return self._act(obs, offset, heading_error, env)


def make_feedforward_policy(weights_path, eval_epsilon=0.05):
    agent = DQNAgent()
    if weights_path and os.path.exists(weights_path):
        agent.load(weights_path)
    agent.epsilon = eval_epsilon
    return PolicyWrapper(
        "feedforward",
        lambda obs, _o, _h, _env: agent.act(obs),
        reset_fn=lambda: None,
    )


def make_recurrent_policy(weights_path, eval_epsilon=0.05, rnn_type="gru"):
    agent = RecurrentDQNAgent(rnn_type=rnn_type)
    if weights_path and os.path.exists(weights_path):
        agent.load(weights_path)
    agent.epsilon = eval_epsilon
    return PolicyWrapper(
        f"{rnn_type}_rnn",
        lambda obs, _o, _h, _env: agent.act(obs),
        reset_fn=agent.reset_hidden,
    )


def make_pid_policy():
    pid = PIDController()
    return PolicyWrapper(
        "pid_baseline",
        lambda _obs, offset, heading_error, env: pid.act(offset, heading_error, env.turn_limit),
        reset_fn=pid.reset,
    )


def recovery_time(offsets, perturb_idx, settle_radius=6.0, window=12, step_dt=1 / 60.0):
    if perturb_idx is None or perturb_idx >= len(offsets):
        return None
    for idx in range(perturb_idx, len(offsets) - window):
        if np.all(np.abs(offsets[idx : idx + window]) < settle_radius):
            return (idx - perturb_idx) * step_dt
    return None


def episode_metrics(run, step_dt=1 / 60.0):
    offsets = np.array(run["offsets"])
    heading = np.array(run["heading_errors"])
    steer = np.array(run["steer"])
    sats = np.array(run["saturation_flags"]) if run["saturation_flags"] else np.array([])

    if len(offsets) == 0:
        return {}

    osc = float(np.mean(np.abs(np.diff(np.sign(offsets))))) if len(offsets) > 1 else 0.0
    metrics = {
        "tracking_error_px": float(np.mean(np.abs(offsets))),
        "oscillation": osc,
        "control_energy": float(np.sum(np.square(steer))),
        "heading_var": float(np.var(heading)),
        "saturation_ratio": float(np.mean(sats)) if sats.size else 0.0,
    }
    metrics["recovery_time_s"] = recovery_time(offsets, run.get("perturb_idx"))
    return metrics


def classify_failures(run):
    offsets = np.array(run["offsets"])
    steer = np.array(run["steer"])
    failures = []
    if np.any(np.abs(offsets) > 140):
        failures.append("divergence")
    osc = float(np.mean(np.abs(np.diff(np.sign(offsets))))) if len(offsets) > 1 else 0.0
    if osc > 0.35 and np.mean(np.abs(offsets)) > 8:
        failures.append("limit_cycle")
    if run["saturation_flags"] and np.mean(run["saturation_flags"]) > 0.35:
        failures.append("saturation")
    if np.isnan(steer).any():
        failures.append("nan_control")
    return failures


def plot_episode(run, policy_name, ep_idx):
    if not HAS_MPL or len(run["offsets"]) == 0:
        return None
    path = RUN_DIR / f"{policy_name}_ep{ep_idx}_plot.png"
    t = np.arange(len(run["offsets"])) * (1 / 60.0)
    plt.figure(figsize=(9, 5))
    plt.subplot(3, 1, 1)
    plt.plot(t, run["offsets"], label="offset px")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.ylabel("Offset (px)")
    if run.get("perturb_idx") is not None:
        plt.axvline(run["perturb_idx"] * (1 / 60.0), color="orange", linestyle="--", label="perturb")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, run["heading_errors"], color="teal", label="heading error rad")
    plt.ylabel("Heading (rad)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, run["steer"], color="magenta", label="steer command")
    plt.ylabel("Steer (rad)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def evaluate_policy(env, policy, args):
    runs = []
    for ep in range(args.episodes):
        obs = env.reset()
        policy.reset()
        run = {
            "offsets": [],
            "heading_errors": [],
            "actions": [],
            "steer": [],
            "saturation_flags": [],
            "rewards": [],
            "perturb_idx": None,
            "domain": env.domain_params,
            "done_step": None,
        }
        for step in range(args.max_steps):
            offset, tangent, _ = env._closest_path_info()
            heading_error = math.atan2(math.sin(env.angle - tangent), math.cos(env.angle - tangent))
            action = policy.act(obs, offset, heading_error, env)
            obs, reward, done = env.step(action)

            if args.perturb_step >= 0 and step == args.perturb_step:
                env.inject_disturbance(angle_kick=args.angle_kick, lateral_kick=args.lateral_kick)
                run["perturb_idx"] = step

            if args.render:
                env.render(
                    info={
                        "policy": policy.name,
                        "offset": f"{offset:.1f}px",
                        "heading": f"{heading_error:.2f}rad",
                        "sat": env.saturation_events,
                    }
                )

            run["offsets"].append(offset)
            run["heading_errors"].append(heading_error)
            run["actions"].append(action)
            run["steer"].append(env.last_steer)
            run["saturation_flags"].append(abs(env.last_steer) >= env.turn_limit * 0.99)
            run["rewards"].append(reward)

            if done:
                run["done_step"] = step
                break
        runs.append(run)
    return runs


def summarize_policy(env, policy, args):
    runs = evaluate_policy(env, policy, args)
    summaries = []
    plots = []
    for idx, run in enumerate(runs):
        metrics = episode_metrics(run)
        failures = classify_failures(run)
        plot_path = plot_episode(run, policy.name, idx) if failures else None
        if plot_path:
            plots.append(str(plot_path))
        summaries.append({"metrics": metrics, "failures": failures, "domain": run["domain"]})
    return summaries, plots


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for line follower")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--render", action="store_true", help="Show pygame window")
    parser.add_argument("--partial-obs", action="store_true", help="Mask geometric state in observations")
    parser.add_argument("--latency", type=int, default=2, help="Action latency in steps")
    parser.add_argument("--perturb-step", type=int, default=180)
    parser.add_argument("--angle-kick", type=float, default=0.4)
    parser.add_argument("--lateral-kick", type=float, default=10.0)
    parser.add_argument("--feedforward-weights", type=str, default=str(Path("models") / "dqn_final.pth"))
    parser.add_argument("--recurrent-weights", type=str, default=str(Path("models") / "dqn_rnn_model.pth"))
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    parser.add_argument("--rnn-type", type=str, default="gru", choices=["gru", "lstm"])
    args = parser.parse_args()

    env = LineFollowEnv3D(
        partial_obs=args.partial_obs,
        latency_steps=args.latency,
        actuation_limit=0.16,
        speed_limit_range=(2.4, 4.6),
        noise_std=0.08,
        bias_drift=0.001,
        domain_randomization=True,
        headless=not args.render,
    )

    policies = [
        make_feedforward_policy(args.feedforward_weights, args.eval_epsilon),
        make_recurrent_policy(args.recurrent_weights, args.eval_epsilon, args.rnn_type),
        make_pid_policy(),
    ]

    all_results = {}
    all_plots = []
    start = time.time()
    for policy in policies:
        summaries, plots = summarize_policy(env, policy, args)
        all_results[policy.name] = summaries
        all_plots.extend(plots)
    env.close()

    summary_path = RUN_DIR / f"summary_{int(start)}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("Robustness evaluation complete")
    print(f"- Episodes per policy: {args.episodes}")
    print(f"- Partial observability: {args.partial_obs}")
    print(f"- Results saved to: {summary_path}")
    if all_plots:
        print(f"- Failure plots: {', '.join(all_plots)}")
    else:
        print("- No failure plots generated")


if __name__ == "__main__":
    main()
