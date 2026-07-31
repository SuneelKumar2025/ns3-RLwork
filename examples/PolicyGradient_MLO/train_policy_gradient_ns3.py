# We need to have old gym

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Source import Classfile as C
#pytorch
import torch

import torch.optim as optim
# from torch.distributions import Categorical
from Source import helper as help
from ns3gym import ns3env
# import gymnasium as gym 
import matplotlib as mpl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checking GPU

# step 1: Enviroment 
# --- Connect to ns-3 simulation ---
env = ns3env.Ns3Env()
env.reset()
# step 2:--- Hyperparameters (Policy Gradient / REINFORCE) ---
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space:", ob_space, ob_space.dtype)
print("Action space:", ac_space, ac_space.dtype)

s_size = ob_space.shape[0]
a_size = ac_space.n
hyperparameters = {
    "h_size": 24,
    "n_training_episodes": 1000,
    "max_t": 1000,
    "gamma": .95,
    "lr": 1e-3,
    "print_every": 1,
    # "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}





#step 3: Start training model

policy = C.Policy(s_size, a_size, hyperparameters["h_size"]).to(device)
optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])

# --- Train with REINFORCE ---
scores, histories = help.reinforce_ns3(
    policy,
    optimizer,
    hyperparameters["n_training_episodes"],
    hyperparameters["max_t"],
    hyperparameters["gamma"],
    hyperparameters["print_every"],
    env,
)



# step 4 --- Save trained model ---
torch.save(policy.state_dict(), "wifi_mlo_policy_gradient.pt")
print("Model saved successfully!")
env.close()
print("Training complete.")


#step 5 printing
# --- Train with REINFORCE ---



# --- Unpack step-level histories ---
delay_history = histories["delay_history"]
l1_delay_history = histories["l1_delay_history"]
l2_delay_history = histories["l2_delay_history"]
action_history = histories["action_history"]
step_times = histories["step_times"]

# --- Plotting ---
print("Plotting Learning and Delay Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 14})

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))

ax1.grid(True, linestyle='--')
ax1.plot(scores, label='Total Reward', color='blue', marker='o')
ax1.set_title('Learning Performance (Policy Gradient / REINFORCE)')
ax1.set_ylabel('Reward Sum')
ax1.set_xlabel('Episodes')
ax1.legend()

ax2.grid(True, linestyle='--')
ax2.plot(delay_history, label='Packet Delay (ms)', color='red', alpha=0.7)
ax2.axhline(y=200, color='black', linestyle='--', label='Threshold (200ms)')
ax2.set_title('Delay per Step')
ax2.set_ylabel('Delay (ms)')
ax2.legend()

ax3.grid(True, linestyle='--')
ax3.plot(l1_delay_history, color='teal', alpha=0.8, label='Link 1 Delay')
ax3.plot(l2_delay_history, color='orange', alpha=0.8, label='Link 2 Delay')
ax3.axhline(y=200, color='black', linestyle='--', label='Threshold (200ms)')
ax3.set_title('Per-Link Delay Comparison')
ax3.set_ylabel('Delay (ms)')
ax3.legend()

ax4.grid(True, linestyle='--')
ax4.plot(action_history, color='purple', linewidth=1.5, label='Selected Link', drawstyle='steps-pre')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['Link 1', 'Link 2'])
ax4.set_title('Agent Link Selection Action per Step')
ax4.set_xlabel('Total Simulated Steps')
ax4.set_ylabel('Action Index')
ax4.legend()

plt.tight_layout()
plt.savefig('pg_performance_results.pdf')
plt.show()

# --- Save raw data ---
np.savetxt("pg_rew_history.csv", scores, delimiter=",", header="reward")
np.savetxt("pg_delay_history.csv", delay_history, delimiter=",", header="delay_ms")

step_matrix = np.column_stack(
    (step_times, delay_history, l1_delay_history, l2_delay_history, action_history)
)
np.savetxt(
    "pg_training_step_metrics.csv",
    step_matrix,
    delimiter=",",
    header="time_s,active_delay_ms,l1_delay_ms,l2_delay_ms,selected_link",
    comments='',
)

print("Raw training data saved successfully!")