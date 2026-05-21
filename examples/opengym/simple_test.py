# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
 
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from ns3gym import ns3env
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from collections import deque
# import random
# from tqdm import tqdm

 
# tf.get_logger().setLevel('ERROR')
 
# # Connect to ns-3 simulation
# env = ns3env.Ns3Env()
# env.reset()
 
# ob_space = env.observation_space
# ac_space = env.action_space
# print("Observation space:", ob_space, ob_space.dtype)
# print("Action space:", ac_space, ac_space.dtype)
 
# s_size = ob_space.shape[0]
# a_size = ac_space.n
# print("s_size =", s_size)
# print("a_size =", a_size)
 
# delay_history = []
# time_history  = []
# rew_history   = []


# l1_delay_history = []
# l2_delay_history = []
# action_history   = []
# step_times       = []
# step_counter     = 0
 
# # --- Build model ---
# model = keras.Sequential([
#     keras.layers.Input(shape=(s_size,)),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(a_size, activation='linear')
# ])
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss='mse'
# )
 
# # --- Hyperparameters ---
# total_episodes = 200
# max_env_steps  = 151
# gamma          = 0.95
# epsilon        = 1.0
# epsilon_min    = 0.01
# epsilon_decay  = 0.97
# batch_size     = 32
 
# # --- Replay buffer ---
# memory = deque(maxlen=2000)
 
# def remember(state, action, reward, next_state, done):
#     memory.append((state, action, reward, next_state, done))
 
# def replay():
#     if len(memory) < batch_size:
#         return  # wait until buffer has enough samples
 
#     batch = random.sample(memory, batch_size)
 
#     # Build full arrays in one shot
#     states      = np.array([x[0][0] for x in batch])   # shape (32, s_size)
#     actions     = np.array([x[1]    for x in batch])   # shape (32,)
#     rewards     = np.array([x[2]    for x in batch])   # shape (32,)
#     next_states = np.array([x[3][0] for x in batch])   # shape (32, s_size)
#     dones       = np.array([x[4]    for x in batch])   # shape (32,)
 
#     # ONE predict call for current + next states (was 64 calls before)
#     current_q = model.predict(states,      verbose=0)   # (32, a_size)
#     next_q    = model.predict(next_states, verbose=0)   # (32, a_size)
 
#     # Update only the taken action's Q-value
#     for i in range(batch_size):
#         target = rewards[i]
#         if not dones[i]:
#             target = rewards[i] + gamma * np.amax(next_q[i])
#         current_q[i][actions[i]] = target
 
#     # ONE fit call for the whole batch (was 32 calls before)
#     model.fit(states, current_q, epochs=1, verbose=0)
 
# # --- Training loop ---
# for e in tqdm(range(total_episodes)):
#     # --- NEW LINE: Generate a new random seed for the C++ ns-3 execution ---
#     env.simSeed = random.randint(1, 100000)
#     state = env.reset()
#     state = np.reshape(state, [1, s_size])
#     rewardsum = 0
 
#     for time in range(max_env_steps):
 
#         # Choose action: explore or exploit
#         if np.random.rand() < epsilon:
#             action = np.random.randint(a_size)
#         else:
#             action = np.argmax(model.predict(state, verbose=0)[0])
 
#         next_state, reward, done, info = env.step(action)
#         # print(f"Action: {action}, Observed State: {next_state}, Reward: {reward}")
 
#         current_delay = float(info)
#         delay_history.append(current_delay)

#         # #### EXTRACT AND APPEND STEP HISTORIES IMMEDIATELY ####
#         l1_delay_history.append(float(next_state[0]))
#         l2_delay_history.append(float(next_state[1]))
#         action_history.append(action)
#         step_times.append(step_counter * 0.1)  # Matches your 0.05s step time interval
#         step_counter += 1
 
#         rewardsum += reward
 
#         if done:
#             print("episode: {}/{}, time: {}, rew: {:.2f}, eps: {:.3f}"
#                   .format(e, total_episodes, time, rewardsum, epsilon))
#             break
 
#         next_state = np.reshape(next_state, [1, s_size])
 
#         remember(state, action, reward, next_state, done)  # store
#         replay()                                            # train on batch
 
#         state = next_state
 
#     if epsilon > epsilon_min:
#         epsilon *= epsilon_decay
#     time_history.append(time)
#     rew_history.append(rewardsum)
#     # --- NEW: SAVE MODEL HERE ---
# model.save("wifi_mlo_model.keras")
# print("Model saved successfully!")
 
# env.close()
# print("Training complete.")
 
# # --- Plotting ---
# print("Plotting Learning and Delay Performance")
# mpl.rcdefaults()
# mpl.rcParams.update({'font.size': 14})
 
# # #### EXPAND SUBPLOTS CONFIGURATION FROM 2 TO 4 ROWS ####
# # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))

# ax1.grid(True, linestyle='--')
# ax1.plot(rew_history, label='Total Reward', color='blue', marker='o')
# ax1.set_title('Learning Performance')
# ax1.set_ylabel('Reward Sum')
# ax1.set_xlabel('Episodes') # Rew_history tracks per episode
# ax1.legend()
 
# # Plot 2: Active Combined Delay per Step
# ax2.grid(True, linestyle='--')
# ax2.plot(delay_history, label='Packet Delay (ms)', color='red', alpha=0.7)
# ax2.axhline(y=170, color='black', linestyle='--', label='Threshold (170ms)')
# ax2.set_title('Delay per Step')
# ax2.set_ylabel('Delay (ms)')
# ax2.legend()

# # Plot 3: Link 1 and Link 2 Delays Over Steps
# ax3.grid(True, linestyle='--')
# ax3.plot(l1_delay_history, color='teal', alpha=0.8, label='Link 1 Delay')
# ax3.plot(l2_delay_history, color='orange', alpha=0.8, label='Link 2 Delay')
# ax3.axhline(y=170, color='black', linestyle='--', label='Threshold (170ms)')
# ax3.set_title('Per-Link Delay Comparison')
# ax3.set_ylabel('Delay (ms)')
# ax3.legend()

# # Plot 4: Selected Link Action Taken per Step
# ax4.grid(True, linestyle='--')
# ax4.plot(action_history, color='purple', linewidth=1.5, label='Selected Link', drawstyle='steps-pre')
# ax4.set_yticks([0, 1])
# ax4.set_yticklabels(['Link 1', 'Link 2'])
# ax4.set_title('Agent Link Selection Action per Step')
# ax4.set_xlabel('Total Simulated Steps') # <--- FIXED: Now correctly matches your indexing approach
# ax4.set_ylabel('Action Index')
# ax4.legend()
 
# plt.tight_layout()
# plt.savefig('performance_results.pdf')
# plt.show()
# np.savetxt("rew_history.csv", rew_history, delimiter=",", header="reward")
# np.savetxt("delay_history.csv", delay_history, delimiter=",", header="delay_ms")
# print("Raw training data saved successfully!")



# # np.savetxt("rew_history.csv", rew_history, delimiter=",", header="reward")

# # =====================================================================
# # #### SAVE COMPLETE STEP TIMELINE DATA MATRIX TO A SINGLE CSV FILE ####
# step_matrix = np.column_stack((step_times, delay_history, l1_delay_history, l2_delay_history, action_history))
# np.savetxt("training_step_metrics.csv", step_matrix, delimiter=",", 
#            header="time_s,active_delay_ms,l1_delay_ms,l2_delay_ms,selected_link", comments='')
# # =====================================================================
# print("Raw training data saved successfully!")






import numpy as np
import tensorflow as tf
from tensorflow import keras
from ns3gym import ns3env
import matplotlib.pyplot as plt
import matplotlib as mpl

tf.get_logger().setLevel('ERROR')

# Connect to ns-3 simulation
env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space:", ob_space, ob_space.dtype)
print("Action space:", ac_space, ac_space.dtype)

s_size = ob_space.shape[0]
a_size = ac_space.n
print("s_size =", s_size)
print("a_size =", a_size)

delay_history = []
step_times = []
l1_delay_history = []
l2_delay_history = []
action_history = []

step_counter = 0

total_episodes = 1
max_env_steps  = 5000

# --- Load trained model ---
model = tf.keras.models.load_model("/home/suneel/Videos/ns-allinone-3.36.1/ns-3.36.1/contrib/opengym/examples/opengym/wifi_mlo_model.keras")
print("Model loaded successfully!")

# --- Inference loop ---
for e in range(total_episodes):
    state = env.reset()
    state = np.reshape(state, [1, s_size])

    for time in range(max_env_steps):
        action = np.argmax(model.predict(state, verbose=0)[0])

        next_state, reward, done, info = env.step(action)

        l1_delay = float(next_state[0])
        l2_delay = float(next_state[1])
        action_history.append(action)


        current_delay = float(info)
        delay_history.append(current_delay)
        l1_delay_history.append(l1_delay)
        l2_delay_history.append(l2_delay)
        step_times.append(step_counter * 0.1)
        step_counter += 1

        # print(f"t={step_counter*0.1:.1f}s | Action: {action} | Delay: {current_delay:.2f} us")

        if done:
            break

        state = np.reshape(next_state, [1, s_size])

env.close()
print("Inference complete.")

# --- Plot ---
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 14})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Top plot: active delay
ax1.plot(step_times, delay_history, color='red', alpha=0.7, label='Active Delay (us)')
ax1.axhline(y=170, color='black', linestyle='--', label='Threshold (170us)')
ax1.set_ylabel('Delay (us)')
ax1.set_title('Active Delay Every 0.1s - Using Trained model')
ax1.legend()
ax1.grid(True, linestyle='--')

# Bottom plot: both link delays
ax2.plot(step_times, l1_delay_history, color='blue', alpha=0.7, label='Link 1 Delay (us)')
ax2.plot(step_times, l2_delay_history, color='green', alpha=0.7, label='Link 2 Delay (us)')
ax2.axhline(y=170, color='black', linestyle='--', label='Threshold (170us)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Delay (us)')
ax2.set_title('Per-Link Delay Comparison')
ax2.legend()
ax2.grid(True, linestyle='--')


# Bottom plot: selected link at each step
ax3.step(step_times, action_history, color='purple', linewidth=1.5, label='Selected Link')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Link 1', 'Link 2'])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Selected Link')
ax3.set_title('Agent Link Selection per Step')
ax3.legend()
ax3.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig('inference_delay.pdf')
plt.show()