import numpy as np
from collections import deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reinforce_ns3(policy, optimizer, n_training_episodes, max_t, gamma, print_every, env):
    """
    REINFORCE (Monte-Carlo Policy Gradient) training loop for an ns3-gym
    environment.

    NOTE: ns3gym's env.step() returns a 4-tuple (obs, reward, done, info),
    unlike gymnasium's 5-tuple (obs, reward, terminated, truncated, info).
    This loop is written for the ns3gym API.
    """
    scores_deque = deque(maxlen=100)
    scores = []

    # step-level logs (mirrors the DQN script) for later plotting
    delay_history = []
    l1_delay_history = []
    l2_delay_history = []
    action_history = []
    step_times = []
    step_counter = 0

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []

        state = env.reset()
        state = np.asarray(state, dtype=np.float32)

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            next_state, reward, done, info = env.step(action)

            # --- step-level logging ---
            try:
                current_delay = float(info)
                delay_history.append(current_delay)
            except (TypeError, ValueError):
                pass
            l1_delay_history.append(float(next_state[0]))
            l2_delay_history.append(float(next_state[1]))
            action_history.append(action)
            step_times.append(step_counter * 0.1)
            step_counter += 1
            # ---------------------------

            rewards.append(reward)
            state = np.asarray(next_state, dtype=np.float32)

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # discounted returns
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    histories = {
        "delay_history": delay_history,
        "l1_delay_history": l1_delay_history,
        "l2_delay_history": l2_delay_history,
        "action_history": action_history,
        "step_times": step_times,
    }
    return scores, histories


def evaluate_agent_ns3(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate a trained REINFORCE agent on an ns3-gym environment
    (deterministic / greedy action selection).
    """
    episode_rewards = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        state = np.asarray(state, dtype=np.float32)
        total_reward = 0

        for step in range(max_steps):
            action, _ = policy.act(state, deterministic=True)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = np.asarray(next_state, dtype=np.float32)

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1:2d}: Reward = {total_reward}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("\n==============================")
    print(f"Average Reward : {mean_reward:.2f}")
    print(f"Std Reward     : {std_reward:.2f}")
    print("==============================")

    return mean_reward, std_reward
