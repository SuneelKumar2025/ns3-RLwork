#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained REINFORCE (Policy Gradient) agent on ns3-gym over
N randomized episodes.

Each env.reset() triggers a fresh ns-3 run, and sim.cc redraws
case1/case2 randomly at the start of every run — so simply calling
env.reset() N times already gives N independent random test scenarios.
No extra scaffolding needed on the C++ side for this.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from Source import Classfile as C
from ns3gym import ns3env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Config ---
MODEL_PATH = "wifi_mlo_policy_gradient.pt"
N_EVAL_EPISODES = 100
MAX_T = 1000          # safety cap only; real episode length is driven by sim's GameOver
H_SIZE = 24            # must match hyperparameters["h_size"] used during training

# --- Connect to ns-3 ---
env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
s_size = ob_space.shape[0]
a_size = ac_space.n
print("Observation space:", ob_space, "| s_size =", s_size)
print("Action space:", ac_space, "| a_size =", a_size)

# --- Load trained policy ---
policy = C.Policy(s_size, a_size, H_SIZE).to(device)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy.eval()
print(f"Loaded weights from {MODEL_PATH}")

# --- Evaluation loop ---
episode_rewards = []
episode_switch_counts = []
episode_lengths = []

delay_history = []
l1_delay_history = []
l2_delay_history = []
action_history = []
step_times = []
episode_id_history = []   # which eval episode each step belongs to
step_counter = 0

for ep in range(1, N_EVAL_EPISODES + 1):
    state = env.reset()
    state = np.asarray(state, dtype=np.float32)

    total_reward = 0.0
    prev_action = None
    n_switches = 0
    n_steps = 0

    for t in range(MAX_T):
        action, _ = policy.act(state, deterministic=True)
        next_state, reward, done, info = env.step(action)

        # --- step-level logging (mirrors the training histories) ---
        try:
            delay_history.append(float(info))
        except (TypeError, ValueError):
            delay_history.append(np.nan)
        l1_delay_history.append(float(next_state[0]))
        l2_delay_history.append(float(next_state[1]))
        action_history.append(action)
        step_times.append(step_counter * 0.1)
        episode_id_history.append(ep)
        step_counter += 1
        n_steps += 1

        if prev_action is not None and action != prev_action:
            n_switches += 1
        prev_action = action
        # -------------------------------------------------------------

        total_reward += reward
        state = np.asarray(next_state, dtype=np.float32)

        if done:
            break

    episode_rewards.append(total_reward)
    episode_switch_counts.append(n_switches)
    episode_lengths.append(n_steps)

    print(f"Eval episode {ep:3d}/{N_EVAL_EPISODES} | "
          f"reward={total_reward:8.2f} | steps={n_steps:4d} | switches={n_switches:3d}")

env.close()
print("Evaluation complete.")

# --- Summary stats ---
episode_rewards = np.array(episode_rewards)
episode_switch_counts = np.array(episode_switch_counts)
episode_lengths = np.array(episode_lengths)

print("\n================ Summary over", N_EVAL_EPISODES, "episodes ================")
print(f"Reward   : mean={episode_rewards.mean():.2f}  std={episode_rewards.std():.2f}  "
      f"min={episode_rewards.min():.2f}  max={episode_rewards.max():.2f}")
print(f"Switches : mean={episode_switch_counts.mean():.2f}  std={episode_switch_counts.std():.2f}  "
      f"max={episode_switch_counts.max()}")
print(f"Length   : mean={episode_lengths.mean():.1f} steps")
print("===========================================================================")

# --- Save raw per-episode results ---
np.savetxt(
    "eval_episode_summary.csv",
    np.column_stack((np.arange(1, N_EVAL_EPISODES + 1),
                      episode_rewards, episode_switch_counts, episode_lengths)),
    delimiter=",",
    header="episode,reward,switch_count,n_steps",
    comments="",
)

# --- Save step-level results ---
step_matrix = np.column_stack(
    (episode_id_history, step_times, delay_history, l1_delay_history, l2_delay_history, action_history)
)
np.savetxt(
    "eval_step_metrics.csv",
    step_matrix,
    delimiter=",",
    header="episode,time_s,active_delay,l1_delay,l2_delay,selected_link",
    comments="",
)
print("Saved eval_episode_summary.csv and eval_step_metrics.csv")

# --- Plots ---
mpl.rcdefaults()
mpl.rcParams.update({"font.size": 13})

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].plot(range(1, N_EVAL_EPISODES + 1), episode_rewards, marker="o", color="green")
axes[0].axhline(episode_rewards.mean(), color="black", linestyle="--",
                 label=f"Mean = {episode_rewards.mean():.1f}")
axes[0].set_title(f"Evaluation Reward per Episode (n={N_EVAL_EPISODES}, deterministic policy)")
axes[0].set_xlabel("Evaluation Episode")
axes[0].set_ylabel("Total Reward")
axes[0].grid(True, linestyle="--")
axes[0].legend()

axes[1].hist(episode_rewards, bins=20, color="steelblue", edgecolor="black")
axes[1].set_title("Distribution of Episode Rewards")
axes[1].set_xlabel("Total Reward")
axes[1].set_ylabel("Count")
axes[1].grid(True, linestyle="--")

axes[2].bar(range(1, N_EVAL_EPISODES + 1), episode_switch_counts, color="purple")
axes[2].set_title("Link Switches per Evaluation Episode")
axes[2].set_xlabel("Evaluation Episode")
axes[2].set_ylabel("Switch Count")
axes[2].grid(True, linestyle="--")

plt.tight_layout()
plt.savefig("eval_performance_results.pdf")
plt.show()
print("Saved eval_performance_results.pdf")