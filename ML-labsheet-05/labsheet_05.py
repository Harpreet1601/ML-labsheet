import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

plt.rcParams['figure.figsize'] = (8, 5)
print("Gymnasium version:", gym.__version__)
print("PyTorch version   :", torch.__version__)


# ## 1. Install and Configure the Gymnasium (or OpenAI Gym) Library
# 
# Run the following once in a terminal (not needed again if already installed):
# 
# ```bash
# pip install gymnasium
# pip install gymnasium[toy-text]   # for FrozenLake and other toy-text envs
# pip install torch                 # for the Deep Q-Network experiments
# ```
# 

# Configuration check: confirm gymnasium is installed and list a few registered envs
import gymnasium as gym
from gymnasium import envs

print("Gymnasium successfully imported. Version:", gym.__version__)

sample_env_ids = [e for e in envs.registry.keys() if 'FrozenLake' in e or 'CartPole' in e]
print("Relevant registered environments found:", sample_env_ids)


# ## 2. Create and Execute a Simple Reinforcement Learning Environment

env = gym.make('CartPole-v1')
observation, info = env.reset(seed=42)
print("Initial observation:", observation)

total_reward = 0
for step in range(20):
    action = env.action_space.sample()          # random action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps.")
        break

print("Total reward collected:", total_reward)
env.close()


# ## 3. Explore the Observation Space and Action Space of an RL Environment

for env_id in ['CartPole-v1', 'FrozenLake-v1']:
    e = gym.make(env_id)
    print(f"--- {env_id} ---")
    print("Observation space:", e.observation_space)
    print("Action space     :", e.action_space)
    e.close()
    print()


# ## 4. Display the States, Actions, Rewards, and Termination Conditions of an Environment

env = gym.make('FrozenLake-v1', is_slippery=False)
state, info = env.reset(seed=42)
print("Start state:", state, "| info:", info)

for t in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}: state={state} action={action} -> next_state={next_state}, "
          f"reward={reward}, terminated={terminated}, truncated={truncated}")
    state = next_state
    if terminated or truncated:
        print("Episode terminated/truncated.")
        break
env.close()


# ## 5. Simulate Random Actions in the FrozenLake Environment

env = gym.make('FrozenLake-v1', is_slippery=True)
n_episodes_random = 20
random_rewards = []

for ep in range(n_episodes_random):
    state, info = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated
    random_rewards.append(ep_reward)

env.close()
print("Rewards for 20 random-action episodes:", random_rewards)
print(f"Random policy success rate: {np.mean(random_rewards) * 100:.1f}%")


# ## 6. Implement the Q-Learning Algorithm for the FrozenLake Environment

def epsilon_greedy_action(Q, state, n_actions, epsilon):
    """Task 15: epsilon-greedy action-selection policy (used by all Q-Learning runs)."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def q_learning(env, n_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
    """Tabular Q-Learning for a discrete-state, discrete-action Gymnasium environment."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))            # Task 7: initialize Q-table

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = epsilon_greedy_action(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 7: update the Q-table (Bellman update)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next * (not done) - Q[state, action])

            state = next_state
            ep_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

    return Q, episode_rewards

print("Q-Learning function defined.")


# ## 7. Initialize and Update the Q-Table during Training

demo_env = gym.make('FrozenLake-v1', is_slippery=False)
n_states = demo_env.observation_space.n
n_actions = demo_env.action_space.n

Q_demo = np.zeros((n_states, n_actions))
print("Initialized Q-table shape:", Q_demo.shape)
print(Q_demo)

# One manual update step to illustrate the Bellman rule used inside q_learning()
state, _ = demo_env.reset()
action = demo_env.action_space.sample()
next_state, reward, terminated, truncated, info = demo_env.step(action)

alpha, gamma = 0.1, 0.99
old_value = Q_demo[state, action]
Q_demo[state, action] += alpha * (reward + gamma * np.max(Q_demo[next_state]) - Q_demo[state, action])

print(f"\nQ[{state},{action}] updated from {old_value:.4f} to {Q_demo[state, action]:.4f}")
demo_env.close()


# ## 8. Train the Agent for Multiple Episodes using Q-Learning

env_fl = gym.make('FrozenLake-v1', is_slippery=True)

t0 = time.time()
Q_fl, rewards_fl = q_learning(env_fl, n_episodes=8000, alpha=0.1, gamma=0.99,
                               epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995)
train_time_fl = time.time() - t0

print(f"Training completed in {train_time_fl:.2f} seconds over {len(rewards_fl)} episodes.")
print(f"Success rate over last 1000 episodes: {np.mean(rewards_fl[-1000:]) * 100:.2f}%")


# ## 9. Display the Learned Q-Table after Training

q_table_df = pd.DataFrame(
    Q_fl,
    columns=['Left', 'Down', 'Right', 'Up'],
    index=[f"S{s}" for s in range(Q_fl.shape[0])]
)
q_table_df.round(3)


# ## 10. Evaluate the Trained Q-Learning Agent

def evaluate_policy(env, Q, n_episodes=200):
    successes = 0
    total_rewards = []
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = int(np.argmax(Q[state]))     # pure greedy (exploitation only)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1
    return successes / n_episodes, np.mean(total_rewards)

success_rate_fl, avg_reward_fl = evaluate_policy(env_fl, Q_fl, n_episodes=200)
print(f"Evaluation over 200 greedy episodes -> Success rate: {success_rate_fl * 100:.2f}%, "
      f"Average reward: {avg_reward_fl:.3f}")
env_fl.close()


# ## 11. Plot the Cumulative Rewards Obtained during Training

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window) / window, mode='valid')

plt.figure()
plt.plot(moving_average(rewards_fl, 100))
plt.xlabel('Episode')
plt.ylabel('Success rate (100-episode moving average)')
plt.title('Q-Learning Training Progress on FrozenLake')
plt.show()


# ## 12. Study the Effect of Different Learning Rates on Q-Learning Performance

learning_rates = [0.01, 0.1, 0.5, 0.9]
lr_results = {}

for lr in learning_rates:
    env_lr = gym.make('FrozenLake-v1', is_slippery=True)
    _, rew = q_learning(env_lr, n_episodes=4000, alpha=lr, gamma=0.99,
                         epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999)
    lr_results[lr] = rew
    env_lr.close()

plt.figure()
for lr, rew in lr_results.items():
    plt.plot(moving_average(rew, 100), label=f"alpha={lr}")
plt.xlabel('Episode')
plt.ylabel('Success rate (moving average)')
plt.title('Effect of Learning Rate (alpha) on Q-Learning')
plt.legend()
plt.show()

for lr, rew in lr_results.items():
    print(f"alpha={lr}: final success rate = {np.mean(rew[-500:]) * 100:.2f}%")


# ## 13. Study the Effect of Different Discount Factor (Gamma) Values

gammas = [0.5, 0.9, 0.99, 0.999]
gamma_results = {}

for g in gammas:
    env_g = gym.make('FrozenLake-v1', is_slippery=True)
    _, rew = q_learning(env_g, n_episodes=4000, alpha=0.1, gamma=g,
                         epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999)
    gamma_results[g] = rew
    env_g.close()

plt.figure()
for g, rew in gamma_results.items():
    plt.plot(moving_average(rew, 100), label=f"gamma={g}")
plt.xlabel('Episode')
plt.ylabel('Success rate (moving average)')
plt.title('Effect of Discount Factor (gamma) on Q-Learning')
plt.legend()
plt.show()

for g, rew in gamma_results.items():
    print(f"gamma={g}: final success rate = {np.mean(rew[-500:]) * 100:.2f}%")


# ## 14. Compare Exploration and Exploitation using Different Epsilon Values

# Fixed (non-decaying) epsilon values, to purely contrast exploration vs exploitation
epsilons = [0.0, 0.1, 0.3, 0.8]
epsilon_results = {}

for eps in epsilons:
    env_e = gym.make('FrozenLake-v1', is_slippery=True)
    _, rew = q_learning(env_e, n_episodes=4000, alpha=0.1, gamma=0.99,
                         epsilon_start=eps, epsilon_min=eps, epsilon_decay=1.0)  # fixed epsilon
    epsilon_results[eps] = rew
    env_e.close()

plt.figure()
for eps, rew in epsilon_results.items():
    plt.plot(moving_average(rew, 100), label=f"epsilon={eps}")
plt.xlabel('Episode')
plt.ylabel('Success rate (moving average)')
plt.title('Exploration (high epsilon) vs Exploitation (low epsilon)')
plt.legend()
plt.show()

print("epsilon=0.0  -> pure exploitation (greedy from the start; can get stuck in a suboptimal policy)")
print("epsilon=0.8  -> heavy exploration (learns slowly but explores more of the state space)")


# ## 15. Implement an Epsilon-Greedy Action Selection Policy
# 
# The `epsilon_greedy_action()` function defined in Experiment 6 already
# implements this policy and has been used throughout every Q-Learning run
# above. A quick standalone demonstration:

demo_Q = np.array([[0.1, 0.8, 0.3, 0.05]])   # pretend Q-values for a single state
for eps in [0.0, 0.2, 1.0]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(1000):
        a = epsilon_greedy_action(demo_Q, 0, 4, eps)
        counts[a] += 1
    print(f"epsilon={eps}: action distribution over 1000 draws -> {counts}")


# ## 16. Design a Simple Grid World Environment

class GridWorld:
    """A simple deterministic NxN Grid World with a start, a goal, and obstacles.

    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Reward : -1 per step, +10 for reaching the goal, -10 for hitting an obstacle
    """
    def __init__(self, size=5, obstacles=None, goal=None, start=(0, 0)):
        self.size = size
        self.start = start
        self.goal = goal if goal else (size - 1, size - 1)
        self.obstacles = obstacles if obstacles else [(1, 1), (2, 3), (3, 1)]
        self.action_space_n = 4
        self.observation_space_n = size * size
        self.state = start

    def reset(self):
        self.state = self.start
        return self._to_index(self.state)

    def _to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action):
        r, c = self.state
        if action == 0:   r = max(0, r - 1)          # Up
        elif action == 1: r = min(self.size - 1, r + 1)  # Down
        elif action == 2: c = max(0, c - 1)           # Left
        elif action == 3: c = min(self.size - 1, c + 1)  # Right

        new_pos = (r, c)
        done = False
        if new_pos in self.obstacles:
            reward = -10
            new_pos = self.state          # bounce back
        elif new_pos == self.goal:
            reward = 10
            done = True
        else:
            reward = -1

        self.state = new_pos
        return self._to_index(new_pos), reward, done, {}

grid_env = GridWorld(size=5)
print(f"Grid World created: {grid_env.size}x{grid_env.size} grid")
print("Start:", grid_env.start, "| Goal:", grid_env.goal, "| Obstacles:", grid_env.obstacles)
print("Observation space size:", grid_env.observation_space_n, "| Action space size:", grid_env.action_space_n)


# ## 17. Train a Q-Learning Agent in the Custom Grid World

def q_learning_gridworld(env, n_episodes=2000, alpha=0.1, gamma=0.95,
                          epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                          max_steps=100):
    Q = np.zeros((env.observation_space_n, env.action_space_n))
    episode_rewards = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action = epsilon_greedy_action(Q, state, env.action_space_n, epsilon)
            next_state, reward, done, _ = env.step(action)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next * (not done) - Q[state, action])
            state = next_state
            ep_reward += reward
            steps += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

    return Q, episode_rewards

t0 = time.time()
Q_grid, rewards_grid = q_learning_gridworld(grid_env, n_episodes=2000)
train_time_grid = time.time() - t0

print(f"Grid World training completed in {train_time_grid:.2f} seconds.")
print(f"Average reward over last 100 episodes: {np.mean(rewards_grid[-100:]):.2f}")


# ## 18. Visualize the Optimal Path Learned by the Agent

def get_optimal_path(env, Q, max_steps=50):
    state = env.reset()
    path = [env.start]
    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        state, reward, done, _ = env.step(action)
        pos = (state // env.size, state % env.size)
        path.append(pos)
        if done:
            break
    return path

optimal_path = get_optimal_path(grid_env, Q_grid)
print("Optimal path (row, col):", optimal_path)

grid_display = np.zeros((grid_env.size, grid_env.size))
for obs in grid_env.obstacles:
    grid_display[obs] = -1
grid_display[grid_env.goal] = 2

plt.figure()
plt.imshow(grid_display, cmap='Pastel1')
path_rows = [p[0] for p in optimal_path]
path_cols = [p[1] for p in optimal_path]
plt.plot(path_cols, path_rows, marker='o', color='blue', linewidth=2, label='Learned path')
plt.scatter([grid_env.start[1]], [grid_env.start[0]], color='green', s=150, marker='s', label='Start')
plt.scatter([grid_env.goal[1]], [grid_env.goal[0]], color='red', s=150, marker='*', label='Goal')
plt.title('Optimal Path Learned by Q-Learning Agent in Grid World')
plt.legend()
plt.gca().invert_yaxis()
plt.show()


# ## 19. Compare Agent Performance in FrozenLake and Grid World Environments

env_fl_eval = gym.make('FrozenLake-v1', is_slippery=True)
success_rate_fl2, avg_reward_fl2 = evaluate_policy(env_fl_eval, Q_fl, n_episodes=200)
env_fl_eval.close()

def evaluate_gridworld(env, Q, n_episodes=100, max_steps=100):
    total_rewards = []
    successes = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action = int(np.argmax(Q[state]))
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1
        total_rewards.append(ep_reward)
        if done and ep_reward > 0:
            successes += 1
    return successes / n_episodes, np.mean(total_rewards)

success_rate_grid, avg_reward_grid = evaluate_gridworld(grid_env, Q_grid, n_episodes=100)

env_comparison_df = pd.DataFrame({
    'Environment': ['FrozenLake-v1', 'Custom GridWorld'],
    'Success Rate (%)': [success_rate_fl2 * 100, success_rate_grid * 100],
    'Average Reward': [avg_reward_fl2, avg_reward_grid],
    'Training Episodes': [len(rewards_fl), len(rewards_grid)],
    'Training Time (s)': [round(train_time_fl, 2), round(train_time_grid, 2)]
})
env_comparison_df


# ## 20. Analyze the Convergence Behavior of the Q-Learning Algorithm

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(moving_average(rewards_fl, 200))
axes[0].set_title('FrozenLake: Convergence of Q-Learning')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Success rate (200-ep moving avg)')

axes[1].plot(moving_average(rewards_grid, 50))
axes[1].set_title('Grid World: Convergence of Q-Learning')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Reward (50-ep moving avg)')

plt.tight_layout()
plt.show()

print("Interpretation: both curves rise steeply while epsilon is still high (lots of")
print("exploration/learning) and then plateau once the Q-table has converged toward the")
print("optimal action-values and epsilon has decayed close to its minimum -- indicating")
print("the policy has stabilized.")


# # Deep Q-Network (DQN) Experiments

# ## 21. Install the Required Libraries for Deep Q-Network Implementation
# 
# ```bash
# pip install torch          # or: pip install tensorflow
# pip install gymnasium[classic-control]   # for CartPole-v1
# ```

import torch
import torch.nn as nn
import torch.optim as optim

print("Torch available:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device   :", device)


# ## 22. Implement a Basic Deep Q-Network using PyTorch

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Task 27 relies on this class -- experience replay memory."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


print("QNetwork and ReplayBuffer classes defined.")


# ### DQN Training Function
# A single reusable training function is defined below; it is used for the
# main training run (Experiment 23) as well as the replay-memory (27),
# target-network (28), and hyperparameter (34) ablation studies, so results
# are directly comparable.

def train_dqn(env_id='CartPole-v1', n_episodes=200, lr=1e-3, gamma=0.99,
              epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.97,
              batch_size=64, buffer_capacity=10000, use_replay=True,
              use_target_network=True, target_update_every=10, seed=0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity if use_replay else batch_size)

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q_values).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Learn from a batch (with replay: random past transitions;
            # without replay: effectively only the most recent transitions)
            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64)
                r = torch.tensor(r, dtype=torch.float32)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)

                q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if use_target_network:
                        q_next = target_net(s2).max(1)[0]
                    else:
                        q_next = policy_net(s2).max(1)[0]   # bootstrap off the same network
                    q_target = r + gamma * q_next * (1 - d)

                loss = nn.functional.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if use_target_network and ep % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

    env.close()
    return policy_net, episode_rewards

print("train_dqn() function defined.")


# ## 23. Train a DQN Agent on the CartPole Environment

t0 = time.time()
dqn_policy, dqn_rewards = train_dqn(env_id='CartPole-v1', n_episodes=250, lr=1e-3, gamma=0.99,
                                     epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.97,
                                     batch_size=64, buffer_capacity=10000,
                                     use_replay=True, use_target_network=True, seed=42)
train_time_dqn = time.time() - t0

print(f"DQN training completed in {train_time_dqn:.2f} seconds over {len(dqn_rewards)} episodes.")
print(f"Average reward over last 20 episodes: {np.mean(dqn_rewards[-20:]):.2f}")


# ## 24. Plot the Episode-Wise Reward Obtained during DQN Training

plt.figure()
plt.plot(dqn_rewards, alpha=0.4, label='Episode reward')
plt.plot(moving_average(dqn_rewards, 10), color='red', linewidth=2, label='10-episode moving average')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('DQN Training Reward on CartPole-v1')
plt.legend()
plt.show()


# ## 25. Evaluate the Trained DQN Agent

def evaluate_dqn(policy_net, env_id='CartPole-v1', n_episodes=20):
    env = gym.make(env_id)
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                action = int(torch.argmax(policy_net(torch.tensor(state, dtype=torch.float32))).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    env.close()
    return rewards

dqn_eval_rewards = evaluate_dqn(dqn_policy, n_episodes=20)
print("Evaluation rewards (20 greedy episodes):", dqn_eval_rewards)
print(f"Average evaluation reward: {np.mean(dqn_eval_rewards):.2f} (max possible = 500)")


# ## 26. Compare Q-Learning and DQN Performance
# 
# For a fair, same-environment comparison, a **discretized tabular
# Q-Learning** agent is trained on CartPole (its continuous state space is
# binned into discrete buckets) and evaluated the same way as the DQN agent.

cartpole_bins = [
    np.linspace(-2.4, 2.4, 10),
    np.linspace(-3.0, 3.0, 10),
    np.linspace(-0.21, 0.21, 10),
    np.linspace(-3.0, 3.0, 10),
]

def discretize_cartpole(state):
    return tuple(int(np.digitize(state[i], cartpole_bins[i])) for i in range(4))


def q_learning_cartpole(n_episodes=3000, alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999, seed=42):
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    Q = {}
    epsilon = epsilon_start
    episode_rewards = []

    def get_q(s):
        if s not in Q:
            Q[s] = np.zeros(env.action_space.n)
        return Q[s]

    for ep in range(n_episodes):
        state, _ = env.reset()
        ds = discretize_cartpole(state)
        done = False
        ep_reward = 0
        while not done:
            q_s = get_q(ds)
            action = int(np.argmax(q_s)) if random.random() > epsilon else env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ds2 = discretize_cartpole(next_state)
            q_s2 = get_q(ds2)
            q_s[action] += alpha * (reward + gamma * np.max(q_s2) * (not done) - q_s[action])
            ds = ds2
            ep_reward += reward
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

    env.close()
    return Q, episode_rewards


t0 = time.time()
Q_cartpole, rewards_qcartpole = q_learning_cartpole(n_episodes=3000)
train_time_qcartpole = time.time() - t0
print(f"Discretized Q-Learning on CartPole trained in {train_time_qcartpole:.2f}s")
print(f"Average reward over last 100 episodes: {np.mean(rewards_qcartpole[-100:]):.2f}")


def evaluate_q_cartpole(Q, n_episodes=20, seed=123):
    env = gym.make('CartPole-v1')
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        ds = discretize_cartpole(state)
        done = False
        ep_reward = 0
        while not done:
            q_s = Q.get(ds, np.zeros(env.action_space.n))
            action = int(np.argmax(q_s))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ds = discretize_cartpole(state)
            ep_reward += reward
        rewards.append(ep_reward)
    env.close()
    return rewards

q_cartpole_eval_rewards = evaluate_q_cartpole(Q_cartpole, n_episodes=20)

algo_comparison_df = pd.DataFrame({
    'Algorithm': ['Discretized Q-Learning', 'Deep Q-Network (DQN)'],
    'Avg Training Reward (last 100 ep)': [np.mean(rewards_qcartpole[-100:]), np.mean(dqn_rewards[-100:])],
    'Avg Evaluation Reward (20 ep)': [np.mean(q_cartpole_eval_rewards), np.mean(dqn_eval_rewards)],
    'Training Episodes': [len(rewards_qcartpole), len(dqn_rewards)],
    'Training Time (s)': [round(train_time_qcartpole, 2), round(train_time_dqn, 2)]
})
algo_comparison_df


# ## 27. Analyze the Effect of Replay Memory on DQN Performance

t0 = time.time()
_, rewards_no_replay = train_dqn(n_episodes=150, use_replay=False, buffer_capacity=64,
                                  use_target_network=True, seed=42)
time_no_replay = time.time() - t0

t0 = time.time()
_, rewards_with_replay = train_dqn(n_episodes=150, use_replay=True, buffer_capacity=10000,
                                    use_target_network=True, seed=42)
time_with_replay = time.time() - t0

plt.figure()
plt.plot(moving_average(rewards_no_replay, 10), label='Without replay memory (tiny buffer)')
plt.plot(moving_average(rewards_with_replay, 10), label='With replay memory (buffer=10000)')
plt.xlabel('Episode')
plt.ylabel('Reward (10-episode moving average)')
plt.title('Effect of Experience Replay Memory on DQN Training')
plt.legend()
plt.show()

print(f"Without replay memory -> avg reward (last 30 ep): {np.mean(rewards_no_replay[-30:]):.2f}")
print(f"With replay memory    -> avg reward (last 30 ep): {np.mean(rewards_with_replay[-30:]):.2f}")
print("Replay memory de-correlates consecutive samples and reuses past experience,")
print("which typically produces more stable and sample-efficient learning.")


# ## 28. Study the Role of the Target Network in DQN

t0 = time.time()
_, rewards_no_target = train_dqn(n_episodes=150, use_replay=True, use_target_network=False, seed=42)
time_no_target = time.time() - t0

t0 = time.time()
_, rewards_with_target = train_dqn(n_episodes=150, use_replay=True, use_target_network=True, seed=42)
time_with_target = time.time() - t0

plt.figure()
plt.plot(moving_average(rewards_no_target, 10), label='Without target network')
plt.plot(moving_average(rewards_with_target, 10), label='With target network')
plt.xlabel('Episode')
plt.ylabel('Reward (10-episode moving average)')
plt.title('Effect of the Target Network on DQN Stability')
plt.legend()
plt.show()

print(f"Without target network -> avg reward (last 30 ep): {np.mean(rewards_no_target[-30:]):.2f}, "
      f"reward std: {np.std(rewards_no_target[-30:]):.2f}")
print(f"With target network    -> avg reward (last 30 ep): {np.mean(rewards_with_target[-30:]):.2f}, "
      f"reward std: {np.std(rewards_with_target[-30:]):.2f}")
print("The target network provides a slowly-updated, stable bootstrap target, which")
print("prevents the 'moving target' problem and reduces training instability/oscillation.")


# ## 29. Save the Trained DQN Model

import os
os.makedirs('models', exist_ok=True)

torch.save(dqn_policy.state_dict(), 'models/dqn_cartpole.pth')
print("DQN model saved to: models/dqn_cartpole.pth")


# ## 30. Load the Saved DQN Model and Perform Testing

loaded_policy = QNetwork(state_dim=4, action_dim=2)
loaded_policy.load_state_dict(torch.load('models/dqn_cartpole.pth'))
loaded_policy.eval()

loaded_eval_rewards = evaluate_dqn(loaded_policy, n_episodes=10)
print("Rewards using the loaded model (10 episodes):", loaded_eval_rewards)
print(f"Average reward: {np.mean(loaded_eval_rewards):.2f}")


# ## 31. Compare Cumulative Rewards Obtained using Different RL Algorithms

def cumulative(rewards):
    return np.cumsum(rewards)

plt.figure()
plt.plot(cumulative(rewards_qcartpole[:250]), label='Discretized Q-Learning (CartPole)')
plt.plot(cumulative(dqn_rewards[:250]), label='DQN (CartPole)')
plt.xlabel('Episode')
plt.ylabel('Cumulative reward')
plt.title('Cumulative Reward Comparison: Q-Learning vs DQN (first 250 episodes)')
plt.legend()
plt.show()


# ## 32. Visualize the Learning Curve of the RL Agent

plt.figure()
plt.plot(moving_average(dqn_rewards, 15), label='DQN (CartPole)', linewidth=2)
plt.plot(moving_average(rewards_grid, 50) * 50, label='Q-Learning GridWorld (scaled x50 for visibility)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Smoothed reward')
plt.title('Learning Curves of Different RL Agents')
plt.legend()
plt.show()

print("A rising, then plateauing learning curve indicates the agent is successfully")
print("converging toward a near-optimal policy for its environment.")


# ## 33. Compare Training Time and Convergence of Q-Learning and DQN

def episodes_to_reach_threshold(rewards, threshold, window=20):
    ma = moving_average(rewards, window)
    idx = np.argmax(ma >= threshold) if np.any(ma >= threshold) else -1
    return idx if idx >= 0 else None

threshold = 100   # reward threshold considered 'reasonably solved' for this comparison
q_conv = episodes_to_reach_threshold(rewards_qcartpole, threshold)
dqn_conv = episodes_to_reach_threshold(dqn_rewards, threshold)

time_conv_df = pd.DataFrame({
    'Algorithm': ['Discretized Q-Learning', 'DQN'],
    'Training Time (s)': [round(train_time_qcartpole, 2), round(train_time_dqn, 2)],
    f'Episodes to reach reward >= {threshold}': [q_conv, dqn_conv],
    'Total Episodes Trained': [len(rewards_qcartpole), len(dqn_rewards)]
})
time_conv_df


# ## 34. Analyze the Impact of Hyperparameters on Learning Performance

hyperparam_settings = {
    'baseline (lr=1e-3, decay=0.97)': dict(lr=1e-3, epsilon_decay=0.97),
    'high lr (lr=1e-2)':               dict(lr=1e-2, epsilon_decay=0.97),
    'low lr (lr=1e-4)':                dict(lr=1e-4, epsilon_decay=0.97),
    'slow epsilon decay (0.995)':      dict(lr=1e-3, epsilon_decay=0.995),
}

hp_results = {}
for name, params in hyperparam_settings.items():
    _, rew = train_dqn(n_episodes=120, seed=42, **params)
    hp_results[name] = rew

plt.figure()
for name, rew in hp_results.items():
    plt.plot(moving_average(rew, 10), label=name)
plt.xlabel('Episode')
plt.ylabel('Reward (10-episode moving average)')
plt.title('Impact of Hyperparameters on DQN Learning Performance')
plt.legend()
plt.show()

for name, rew in hp_results.items():
    print(f"{name}: avg reward (last 20 ep) = {np.mean(rew[-20:]):.2f}")

print("\nToo high a learning rate can destabilize training (large, noisy updates);")
print("too low a learning rate slows convergence. Slower epsilon decay keeps the agent")
print("exploring longer, which can delay convergence but sometimes avoids poor local optima.")


# ## 35. Prepare a Comparative Report Summarizing the Performance of RL Algorithms

report_df = pd.DataFrame({
    'Algorithm / Setup': [
        'Random Policy (FrozenLake)',
        'Q-Learning (FrozenLake)',
        'Q-Learning (Custom GridWorld)',
        'Discretized Q-Learning (CartPole)',
        'DQN (CartPole)',
        'DQN without Replay Memory',
        'DQN without Target Network',
    ],
    'Avg Reward / Success Rate': [
        f"{np.mean(random_rewards) * 100:.1f}% success",
        f"{success_rate_fl2 * 100:.1f}% success",
        f"{avg_reward_grid:.2f} reward",
        f"{np.mean(q_cartpole_eval_rewards):.1f} reward",
        f"{np.mean(dqn_eval_rewards):.1f} reward",
        f"{np.mean(rewards_no_replay[-30:]):.1f} reward (train)",
        f"{np.mean(rewards_no_target[-30:]):.1f} reward (train)",
    ],
    'Notes': [
        'No learning; baseline for comparison',
        'Converges to a good policy on a small discrete state space',
        'Learns an efficient obstacle-avoiding path to the goal',
        'Works, but coarse discretization limits precision',
        'Handles the continuous state space directly via function approximation',
        'Less stable / less sample-efficient without replay',
        'More unstable training without a stable bootstrap target',
    ]
})
report_df
