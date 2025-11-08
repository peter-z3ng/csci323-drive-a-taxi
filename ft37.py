import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm import trange  # for progress bar

# Initialize the Taxi environment (text world)
env = gym.make("Taxi-v3", render_mode="ansi")

# Initialize Q-table (500 states x 6 actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 1.0      # exploration rate
min_epsilon = 0.1  # minimum exploration
decay_rate = 0.0005 # epsilon decay speed
episodes = 30000   # total training episodes

# Performance trackers
rewards_per_episode = []
steps_list = []
avg_q_values = []
epsilon_values = []
success_count = 0
success_flags = []

# Create progress bar once before loop
progress_bar = trange(episodes, desc="Training Taxi Agent")
step_limit = 200

for i in progress_bar:
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    success = False

    while not done and steps < step_limit:
        # Choose random or best action (explore vs exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action and get feedback
        next_state, reward, done, truncated, info = env.step(action)

        # Update Q-value (Bellman equation)
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        total_reward += reward
        steps += 1
        state = next_state

        # Count successful drop-offs
        if reward == 20:
            success = True
            success_count += 1

    # Record metrics per episode
    rewards_per_episode.append(total_reward)
    steps_list.append(steps)
    avg_q_values.append(np.mean(q_table))
    epsilon_values.append(epsilon)
    success_flags.append(1 if success else 0)

    # Print average metrics every 1000 episodes
    if (i + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-1000:])
        avg_steps = np.mean(steps_list[-1000:])
        success_rate = (success_count / (i + 1)) * 100
        avg_exploration = np.mean(epsilon_values[-1000:]) * 100
        start_ep = i - 999 if i >= 999 else 0
        end_ep = i + 1
        progress_bar.write(
            f"Episode {start_ep + 1:5d}-{end_ep:<5d} : "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Avg Steps: {avg_steps:6.1f} | "
            f"Success Rate: {success_rate:6.2f}% | "
            f"Avg Exploration: {avg_exploration:6.1f}%")

    # Gradually reduce exploration & learning rate
    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate))
    alpha = max(0.1, 1.0 / (1 + 0.0001 * i))

# Final success rate
success_rate = (success_count / episodes) * 100
print(f"\nTraining Complete! Final Success Rate: {success_rate:.2f}%")

# Visualize Q-Table
plt.figure(figsize=(10, 5))
plt.axis('off')
plt.title("Q-Table (First 15 States)")

subset = q_table[:15]
table = plt.table(
    cellText=np.round(subset, 2),
    colLabels=[f"A{i}" for i in range(env.action_space.n)],
    rowLabels=[f"S{i}" for i in range(15)],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)
plt.show()

# Visualize average Q-Value growth
plt.figure(figsize=(10,5))
plt.plot(avg_q_values, color='purple', label='Average Q-Value per Episode')
plt.xlabel("Episode")
plt.ylabel("Average Q-Value")
plt.title("Taxi-v3 Agent Knowledge Growth (Average Q-Value Over Time)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize success rate improvement
success_over_time = np.cumsum(success_flags) / np.arange(1, len(success_flags)+1) * 100
plt.figure(figsize=(10, 5))
plt.plot(success_over_time, color='purple', label='Success Rate (%)')
plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Taxi-v3 Success Rate Improvement Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Steps vs Reward scatter plot
plt.figure(figsize=(10,5))
plt.scatter(steps_list, rewards_per_episode, color='red', alpha=0.3)
plt.xlabel("Steps per Episode")
plt.ylabel("Total Reward")
plt.title("Efficiency vs. Performance (Steps vs. Reward)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize trained taxi routes
env = gym.make("Taxi-v3", render_mode="human")
print("\nVisualizing multiple trained taxi routes\n")

num_episodes_to_show = 5  # show 5 trips
for episode in range(num_episodes_to_show):
    state, info = env.reset()
    done = False
    total_reward = 0
    print(f"--- Route {episode + 1} ---")

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        time.sleep(0.2)  # animation speed

    print(f"Route {episode + 1} complete! Total Reward: {total_reward}\n")
    time.sleep(1.0)

env.close()
print("All routes complete!")
