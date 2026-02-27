# Chapter 23: Reinforcement Learning

> *"Reinforcement learning is the study of how to map situations to actions — so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them."* — Sutton & Barto (2018)

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Formalize sequential decision-making problems as Markov Decision Processes (MDPs) and derive the Bellman equations.
2. Implement dynamic programming algorithms (policy iteration, value iteration) for environments with known dynamics.
3. Apply Monte Carlo and Temporal Difference methods to learn from experience without a model of the environment.
4. Build Deep Q-Networks (DQN) with experience replay and target networks, and understand key extensions (Double DQN, Dueling DQN, PER).
5. Derive the REINFORCE policy gradient algorithm and explain how baselines reduce variance.
6. Implement actor-critic architectures (A2C, A3C) and understand Generalized Advantage Estimation (GAE).
7. Explain PPO's clipped surrogate objective and implement it for continuous control tasks.
8. Connect reinforcement learning theory to its application in LLM alignment via RLHF.
9. Apply multi-armed bandit algorithms (epsilon-greedy, UCB, Thompson Sampling) and extend to contextual settings.
10. Understand model-based RL (world models, Dreamer) and offline RL (CQL, Decision Transformer).

---

## 23.1 Markov Decision Processes

The mathematical framework underlying reinforcement learning is the Markov Decision Process (MDP). An MDP formalizes the interaction between an agent and its environment as a discrete-time stochastic process.

### 23.1.1 MDP Definition

An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- **States** $\mathcal{S}$: The set of possible states the environment can be in. A state $s \in \mathcal{S}$ contains all information needed to predict the future (the Markov property).
- **Actions** $\mathcal{A}$: The set of actions available to the agent. In some formulations, the available actions depend on the state: $\mathcal{A}(s)$.
- **Transition function** $P$: The probability of transitioning to state $s'$ given state $s$ and action $a$:

$$P(s' | s, a) = \Pr[S_{t+1} = s' | S_t = s, A_t = a]$$

- **Reward function** $R$: The expected immediate reward:

$$R(s, a, s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$$

Often simplified to $R(s, a)$ or $R(s)$.

- **Discount factor** $\gamma \in [0, 1]$: Trades off immediate versus future rewards.

**The Markov Property.** The defining characteristic is that the future is conditionally independent of the past given the present:

$$\Pr[S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0] = \Pr[S_{t+1} | S_t, A_t]$$

### 23.1.2 Return

The *return* $G_t$ is the total discounted reward from time step $t$ onward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The discount factor $\gamma$ serves multiple purposes: it ensures the return is finite for continuing tasks, expresses preference for sooner rewards, and mathematically enables recursive formulations.

### 23.1.3 Policy

A *policy* $\pi$ maps states to actions. It can be:
- **Deterministic:** $a = \pi(s)$
- **Stochastic:** $\pi(a|s) = \Pr[A_t = a | S_t = s]$

The goal of RL is to find a policy $\pi^*$ that maximizes expected return.

### 23.1.4 Value Functions

The *state-value function* $V^\pi(s)$ is the expected return starting from state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

The *action-value function* $Q^\pi(s, a)$ is the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

The relationship between them: $V^\pi(s) = \sum_{a} \pi(a|s) Q^\pi(s, a)$.

### 23.1.5 Bellman Equations

The value functions satisfy recursive relationships known as the Bellman equations.

**Bellman Expectation Equation** (for a given policy $\pi$):

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$

**Bellman Optimality Equation** (for the optimal policy):

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]$$

The optimal policy is greedy with respect to $Q^*$: $\pi^*(s) = \arg\max_a Q^*(s, a)$.

---

## 23.2 Dynamic Programming

When the MDP dynamics ($P$ and $R$) are known, dynamic programming (DP) algorithms can compute optimal policies exactly.

### 23.2.1 Policy Evaluation

Given a policy $\pi$, compute its value function $V^\pi$ by iteratively applying the Bellman expectation equation as an update rule:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]$$

This converges to $V^\pi$ as $k \to \infty$ under the contraction mapping theorem, since the Bellman operator is a $\gamma$-contraction in the supremum norm.

### 23.2.2 Policy Improvement

Given $V^\pi$, we can improve the policy by acting greedily:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

The *policy improvement theorem* guarantees that $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$, with equality if and only if $\pi$ is already optimal.

### 23.2.3 Policy Iteration

Alternates between policy evaluation and policy improvement:

1. Initialize $\pi$ arbitrarily.
2. **Policy Evaluation:** Compute $V^\pi$ (by iterating the Bellman expectation equation to convergence).
3. **Policy Improvement:** Set $\pi'(s) = \arg\max_a Q^\pi(s, a)$ for all $s$.
4. If $\pi' = \pi$, stop (optimal policy found). Otherwise, $\pi \leftarrow \pi'$ and go to step 2.

Policy iteration converges in a finite number of iterations for finite MDPs (at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations, though in practice convergence is much faster).

### 23.2.4 Value Iteration

Combines evaluation and improvement into a single update by applying the Bellman optimality equation:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]$$

Iterate until convergence ($\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$), then extract the optimal policy:

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

```python
import numpy as np

def value_iteration(P, R, gamma=0.99, theta=1e-8):
    """
    Value iteration for a finite MDP.

    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: Optimal value function, shape (num_states,)
        policy: Optimal deterministic policy, shape (num_states,)
    """
    num_states, num_actions = P.shape[0], P.shape[1]
    V = np.zeros(num_states)

    while True:
        V_new = np.zeros(num_states)
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for s_next in range(num_states):
                    q_values[a] += P[s, a, s_next] * (
                        R[s, a, s_next] + gamma * V[s_next]
                    )
            V_new[s] = np.max(q_values)

        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    # Extract optimal policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for s_next in range(num_states):
                q_values[a] += P[s, a, s_next] * (
                    R[s, a, s_next] + gamma * V[s_next]
                )
        policy[s] = np.argmax(q_values)

    return V, policy
```

---

## 23.3 Monte Carlo Methods

When the environment dynamics are unknown, the agent must learn from experience. Monte Carlo (MC) methods estimate value functions by averaging returns over complete episodes.

### 23.3.1 First-Visit Monte Carlo Prediction

To estimate $V^\pi(s)$, generate episodes following $\pi$ and average the returns observed after the *first visit* to $s$ in each episode:

1. Generate episode: $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T$.
2. For each state $s$ appearing in the episode:
   - $G \leftarrow$ return following the first occurrence of $s$.
   - Append $G$ to $\text{Returns}(s)$.
   - $V(s) \leftarrow \text{average}(\text{Returns}(s))$.

By the law of large numbers, $V(s) \to V^\pi(s)$ as the number of episodes $\to \infty$.

### 23.3.2 Exploring Starts

For MC control (finding optimal policies), the agent must explore all state-action pairs. The *exploring starts* assumption guarantees this by starting each episode from a random state-action pair. While unrealistic in many domains, it provides a clean theoretical foundation for MC control.

### 23.3.3 On-Policy vs Off-Policy

- **On-policy** methods improve the same policy used to generate behavior. The agent follows an $\epsilon$-greedy policy (choosing the greedy action with probability $1 - \epsilon$ and a random action with probability $\epsilon$) and learns the value of this $\epsilon$-greedy policy.

- **Off-policy** methods learn about a *target* policy $\pi$ while following a different *behavior* policy $b$. This uses *importance sampling* to correct for the mismatch:

$$V^\pi(s) = \mathbb{E}_b\left[\frac{\prod_{t=0}^{T-1} \pi(A_t|S_t)}{\prod_{t=0}^{T-1} b(A_t|S_t)} G_t \bigg| S_0 = s\right]$$

The importance sampling ratio can have very high variance, especially for long episodes. Weighted importance sampling helps by normalizing the ratios, reducing variance at the cost of introducing bias.

---

## 23.4 Temporal Difference Learning

Temporal Difference (TD) methods combine ideas from MC and DP: they learn from experience (like MC) but update estimates based on other learned estimates without waiting for the episode to end (like DP). This *bootstrapping* is the key idea.

### 23.4.1 TD(0)

The simplest TD method updates the value estimate after each time step using the TD error:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

The quantity $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called the *TD error*. The target $R_{t+1} + \gamma V(S_{t+1})$ is called the *TD target*.

**Advantages over MC:**
- Updates after each step (no need to wait for episode end).
- Works for continuing (non-episodic) tasks.
- Lower variance (but introduces bias from bootstrapping).
- Typically converges faster in practice.

### 23.4.2 SARSA (On-Policy TD Control)

SARSA learns $Q^\pi$ for the behavior policy by updating after each transition $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

The name comes from the quintuple used in each update: $(S, A, R, S, A)$. SARSA is on-policy because $A_{t+1}$ is the action actually taken by the current policy.

### 23.4.3 Q-Learning (Off-Policy TD Control)

Q-Learning (Watkins & Dayan, 1992) directly approximates $Q^*$ regardless of the policy being followed:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

The critical difference from SARSA: the update uses $\max_{a'} Q(S_{t+1}, a')$ instead of $Q(S_{t+1}, A_{t+1})$. This makes Q-learning *off-policy* — it learns the optimal Q-function independent of the exploration policy.

**Convergence.** Q-Learning converges to $Q^*$ with probability 1 under these conditions:
1. All state-action pairs are visited infinitely often.
2. The learning rate $\alpha_t$ satisfies $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$.

```python
import numpy as np
import gymnasium as gym

def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Tabular Q-Learning for discrete environments.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning update (off-policy: uses max over next actions)
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return Q, episode_rewards

# Example: Solve FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=True)
Q, rewards = q_learning(env, num_episodes=20000)

# Extract learned policy
policy = np.argmax(Q, axis=1)
action_names = ["Left", "Down", "Right", "Up"]
print("Learned policy:")
for s in range(env.observation_space.n):
    print(f"  State {s}: {action_names[policy[s]]}")
```

---

## 23.5 Deep Q-Networks (DQN)

Tabular methods fail when the state space is large or continuous. Deep Q-Networks (DQN) use neural networks to approximate $Q^*(s, a)$, enabling RL in high-dimensional spaces like Atari games (Mnih et al., 2015).

### 23.5.1 Core Architecture

DQN approximates $Q^*(s, a; \theta)$ using a deep neural network with parameters $\theta$. For Atari, the input is a stack of 4 grayscale $84 \times 84$ frames, processed by 3 convolutional layers followed by 2 fully connected layers, outputting Q-values for all actions simultaneously.

The loss function is the mean squared TD error:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

### 23.5.2 Key Innovations

Naively applying neural networks to Q-learning fails due to instability. DQN introduced two critical stabilization techniques:

**Experience Replay.** Transitions $(s_t, a_t, r_{t+1}, s_{t+1})$ are stored in a replay buffer $\mathcal{D}$. Mini-batches are sampled uniformly from $\mathcal{D}$ for training. This breaks temporal correlations between consecutive samples and enables data reuse. The buffer typically stores the last 1 million transitions.

**Target Network.** A separate target network $Q(s, a; \theta^-)$ is used in the TD target. Its parameters $\theta^-$ are copied from the online network $\theta$ every $C$ steps (e.g., $C = 10{,}000$). This prevents the moving target problem where the network chases its own rapidly changing predictions.

**Reward Clipping.** All rewards are clipped to $[-1, +1]$, normalizing across different games and improving training stability.

**Atari Results.** DQN achieved human-level performance on 29 of 49 Atari games, using the same architecture and hyperparameters for all games — a landmark demonstration of general-purpose RL.

### 23.5.3 DQN Extensions

**Double DQN (van Hasselt et al., 2016).** Standard DQN uses the same network to both select and evaluate actions in the TD target: $\max_{a'} Q(s', a'; \theta^-)$. This causes overestimation because the max operator is positively biased. Double DQN decouples selection and evaluation:

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

The online network $\theta$ selects the best action; the target network $\theta^-$ evaluates it.

**Dueling DQN (Wang et al., 2016).** Decomposes $Q(s, a)$ into a state-value stream $V(s)$ and an advantage stream $A(s, a)$:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$

The advantage normalization ensures identifiability. This architecture helps when the value of a state is important but actions do not significantly affect the outcome.

**Prioritized Experience Replay (PER) (Schaul et al., 2016).** Instead of sampling uniformly from the replay buffer, PER samples transitions proportional to their TD error magnitude — transitions the agent learns the most from are replayed more frequently. Importance sampling weights correct the bias introduced by non-uniform sampling.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network for Atari-style environments."""
    def __init__(self, input_channels=4, num_actions=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.uint8),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.uint8),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network."""
    def __init__(self, num_actions, lr=2.5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000000,
                 target_update_freq=10000, batch_size=32):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = DQN(num_actions=num_actions).to(self.device)
        self.target_net = DQN(num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.steps = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.uint8).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        """Perform one gradient step on the DQN loss."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return loss.item()
```

---

## 23.6 Policy Gradient Methods

Value-based methods (DQN) learn a value function and derive a policy from it. Policy gradient methods directly parameterize and optimize the policy $\pi_\theta(a|s)$.

### 23.6.1 Why Policy Gradients?

1. **Continuous actions:** Value-based methods require $\arg\max_a Q(s, a)$, which is intractable for continuous action spaces. Policy gradients naturally handle continuous actions by outputting distribution parameters (e.g., mean and variance of a Gaussian).
2. **Stochastic policies:** Some tasks require stochastic policies (e.g., rock-paper-scissors). Policy gradients represent stochastic policies directly.
3. **Better convergence:** In some cases, small changes to the policy produce small changes in behavior, leading to smoother optimization.

### 23.6.2 The Policy Gradient Theorem

The objective is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \right] = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

The *policy gradient theorem* (Sutton et al., 2000) provides the gradient without requiring differentiation through the environment dynamics:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_{k+1}$ is the return from time $t$.

**Derivation sketch.** The probability of a trajectory $\tau = (s_0, a_0, r_1, s_1, a_1, \ldots)$ under policy $\pi_\theta$ is:

$$P(\tau | \theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) P(s_{t+1} | s_t, a_t)$$

Taking the gradient of $J(\theta) = \mathbb{E}_{\tau \sim P(\tau|\theta)} [R(\tau)]$:

$$\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) R(\tau) \, d\tau$$

Using the log-derivative trick ($\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)$):

$$= \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log P(\tau|\theta) \cdot R(\tau)]$$

Since $\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t)$ (transition dynamics and initial state distribution do not depend on $\theta$), we obtain the result.

### 23.6.3 REINFORCE

The REINFORCE algorithm (Williams, 1992) uses the policy gradient theorem with Monte Carlo return estimates:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t$$

**Baseline Subtraction.** REINFORCE suffers from high variance. A *baseline* $b(s_t)$ can be subtracted from the return without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t)) \right]$$

This is unbiased because $\mathbb{E}_{a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$.

The optimal baseline is $b^*(s) = \frac{\mathbb{E}[(\nabla_\theta \log \pi)^2 G_t]}{\mathbb{E}[(\nabla_\theta \log \pi)^2]}$, but in practice, using $b(s_t) = V(s_t)$ (a learned value function) works well. The quantity $G_t - V(s_t)$ is called the *advantage*: it measures how much better action $a_t$ is compared to the average action at state $s_t$.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def reinforce_with_baseline(env_name="CartPole-v1", num_episodes=1000,
                             gamma=0.99, lr_policy=3e-4, lr_value=1e-3):
    """REINFORCE with learned value baseline."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim)
    value_fn = ValueNetwork(obs_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_fn.parameters(), lr=lr_value)

    for episode in range(num_episodes):
        # Collect episode
        states, actions, rewards = [], [], []
        state, _ = env.reset()

        while True:
            state_t = torch.tensor(state, dtype=torch.float32)
            dist = policy(state_t)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            states.append(state_t)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if terminated or truncated:
                break

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        states = torch.stack(states)
        actions = torch.stack(actions)

        # Update value function
        values = value_fn(states)
        value_loss = nn.functional.mse_loss(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Update policy with advantage = return - baseline
        with torch.no_grad():
            advantages = returns - value_fn(states)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = policy(states)
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Return = {sum(rewards):.1f}")

    return policy
```

---

## 23.7 Actor-Critic Methods

REINFORCE uses full episode returns, which have high variance. Actor-critic methods reduce variance by using a *critic* (value function) to estimate returns, updated at every step via bootstrapping.

### 23.7.1 The Actor-Critic Architecture

- **Actor** $\pi_\theta(a|s)$: The policy that selects actions.
- **Critic** $V_w(s)$ or $Q_w(s, a)$: The value function that evaluates states or state-action pairs.

The actor is updated using policy gradients with the critic's estimate as a baseline:

$$\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]$$

where $\hat{A}_t$ is an estimate of the advantage function.

### 23.7.2 A2C: Advantage Actor-Critic

A2C uses the one-step advantage estimate:

$$\hat{A}_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$$

This is equivalent to using the TD error as the advantage estimate. A2C is the synchronous version of A3C (Mnih et al., 2016) — multiple workers collect experience in parallel, and gradients are accumulated synchronously before updating the shared model.

### 23.7.3 A3C: Asynchronous Advantage Actor-Critic

A3C (Mnih et al., 2016) runs multiple agent instances in parallel across CPU threads, each interacting with its own copy of the environment. Each thread computes gradients locally and applies them to a shared global model asynchronously. This parallelism serves as an alternative to experience replay for decorrelating samples.

### 23.7.4 Generalized Advantage Estimation (GAE)

GAE (Schulman et al., 2016) provides a principled way to trade off bias and variance in advantage estimation using an exponentially-weighted average of $n$-step advantages:

$$\hat{A}_t^\text{GAE}(\gamma, \lambda) = \sum_{\ell=0}^{\infty} (\gamma \lambda)^\ell \delta_{t+\ell}$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error.

- $\lambda = 0$: One-step TD advantage (high bias, low variance).
- $\lambda = 1$: Monte Carlo advantage (low bias, high variance).
- $\lambda \in (0, 1)$: Interpolation between the two extremes.

In practice, $\lambda = 0.95$ works well for most tasks.

---

## 23.8 Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is the most widely used policy gradient algorithm due to its simplicity, stability, and strong empirical performance. It addresses a fundamental challenge: policy gradient updates that are too large can catastrophically degrade the policy.

### 23.8.1 The Problem with Large Policy Updates

Standard policy gradient methods take gradient steps proportional to the advantage-weighted log-probabilities. If the step is too large, the policy can change dramatically, leading to poor data collection in subsequent iterations — a vicious cycle that collapses training. The challenge is taking the largest possible improvement step without stepping so far that the policy degrades.

### 23.8.2 The Clipped Surrogate Objective

PPO constrains policy updates via a clipped objective function. Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ be the probability ratio between the new and old policies.

The clipped surrogate objective is:

$$\mathcal{L}^\text{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

where $\epsilon$ is the clipping parameter (typically $0.2$).

**Why clipping works:** Consider two cases:
- **Positive advantage** ($\hat{A}_t > 0$): The action was better than average, so we want to increase its probability ($r_t > 1$). But the clipping caps $r_t$ at $1 + \epsilon$, preventing the probability from increasing too much.
- **Negative advantage** ($\hat{A}_t < 0$): The action was worse than average, so we want to decrease its probability ($r_t < 1$). Clipping caps $r_t$ at $1 - \epsilon$, preventing the probability from decreasing too much.

The $\min$ operation takes the more pessimistic (conservative) bound, ensuring the objective is a lower bound on the true performance.

### 23.8.3 Full PPO Objective

The complete PPO loss combines three terms:

$$\mathcal{L}(\theta) = \mathcal{L}^\text{CLIP}(\theta) - c_1 \mathcal{L}^\text{VF}(\theta) + c_2 S[\pi_\theta]$$

where:
- $\mathcal{L}^\text{VF}(\theta) = (V_\theta(s_t) - V_t^\text{target})^2$ is the value function loss.
- $S[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ is an entropy bonus that encourages exploration.
- $c_1 \approx 0.5$ and $c_2 \approx 0.01$ are coefficients.

### 23.8.4 PPO Implementation Details

Modern PPO implementations include numerous details critical for performance:

1. **Vectorized environments:** Run $N$ environments in parallel (e.g., $N = 8$).
2. **Mini-batch updates:** Collect rollouts of length $T$ from $N$ environments, forming a batch of $N \times T$ transitions. Perform $K$ epochs (e.g., $K = 4$) of mini-batch SGD on this data.
3. **Advantage normalization:** Normalize advantages across the batch to zero mean and unit variance.
4. **Gradient clipping:** Clip gradient norms (typically to 0.5).
5. **Learning rate annealing:** Linearly decay the learning rate.
6. **Orthogonal initialization:** Initialize policy and value network weights with orthogonal initialization.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for continuous control."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal initialization
        for layer in [*self.shared, self.actor_mean, self.critic]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        value = self.critic(features).squeeze(-1)
        return Normal(mean, std), value


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    values = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_log_probs,
               advantages, returns, clip_eps=0.2, epochs=4, batch_size=64,
               value_coef=0.5, entropy_coef=0.01):
    """PPO clipped surrogate objective update."""
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        # Shuffle and create mini-batches
        indices = torch.randperm(len(states))
        for start in range(0, len(states), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # Get current policy and value
            dist, values = model(batch_states)
            log_probs = dist.log_prob(batch_actions).sum(dim=-1)

            # PPO clipped surrogate loss
            ratio = (log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.functional.mse_loss(values, batch_returns)

            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
```

---

## 23.9 Trust Region Policy Optimization (TRPO)

TRPO (Schulman et al., 2015) was the precursor to PPO and addresses the same problem — constraining policy updates — but with a theoretically grounded trust region constraint rather than a heuristic clipping mechanism.

### 23.9.1 Trust Region Constraint

TRPO maximizes the surrogate objective subject to a constraint on the KL divergence between old and new policies:

$$\max_\theta \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \mathbb{E}_t \left[ D_\text{KL}(\pi_{\theta_\text{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)) \right] \leq \delta$$

### 23.9.2 Solution via Conjugate Gradient

This constrained optimization problem is solved approximately:
1. Compute the policy gradient $\mathbf{g} = \nabla_\theta \mathcal{L}$.
2. Compute the natural gradient direction $\mathbf{x} = \mathbf{F}^{-1} \mathbf{g}$ using the conjugate gradient method, where $\mathbf{F}$ is the Fisher information matrix (avoiding explicit construction of $\mathbf{F}$, which is $|\theta| \times |\theta|$, by computing Fisher-vector products).
3. Determine step size $\beta = \sqrt{2\delta / \mathbf{x}^T \mathbf{F} \mathbf{x}}$.
4. Apply line search with backtracking to ensure the KL constraint is satisfied.

### 23.9.3 Why PPO is Preferred

While TRPO provides stronger theoretical guarantees, PPO is preferred in practice because:
- PPO is simpler to implement (no conjugate gradient, no Fisher-vector products, no line search).
- PPO works with standard first-order optimizers (Adam).
- PPO achieves comparable or better empirical performance.
- PPO naturally handles shared actor-critic architectures (TRPO's constraint only applies to the policy, making joint optimization awkward).

---

## 23.10 Multi-Armed Bandits

The multi-armed bandit problem is a simplified RL setting with a single state — the agent repeatedly selects from $K$ actions (arms) and receives rewards. It isolates the fundamental exploration-exploitation tradeoff.

### 23.10.1 Problem Formulation

At each time step $t$, the agent selects an arm $a_t \in \{1, \ldots, K\}$ and receives a reward $r_t$ drawn from an unknown distribution with mean $\mu_{a_t}$. The goal is to maximize the cumulative reward $\sum_{t=1}^{T} r_t$, or equivalently, minimize the *regret*:

$$\text{Regret}(T) = T \mu^* - \sum_{t=1}^{T} \mu_{a_t}$$

where $\mu^* = \max_a \mu_a$ is the mean reward of the best arm.

### 23.10.2 Algorithms

**Epsilon-Greedy.** With probability $1 - \epsilon$, select the arm with the highest estimated mean reward; with probability $\epsilon$, select a random arm. The estimate is $\hat{\mu}_a = \frac{1}{N_a} \sum_{t: a_t = a} r_t$ where $N_a$ is the number of times arm $a$ has been pulled. Regret grows linearly: $O(\epsilon T)$ for constant $\epsilon$.

**Upper Confidence Bound (UCB).** Select the arm that maximizes:

$$a_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}} \right]$$

The second term is an exploration bonus that grows logarithmically with time and shrinks as an arm is pulled more. UCB achieves logarithmic regret: $O(\sqrt{KT \ln T})$.

**Thompson Sampling.** A Bayesian approach. Maintain a posterior distribution over each arm's mean reward. At each step:
1. Sample a value $\tilde{\mu}_a$ from each arm's posterior.
2. Select the arm with the highest sample: $a_t = \arg\max_a \tilde{\mu}_a$.
3. Update the posterior based on the observed reward.

For Bernoulli rewards, the posterior is a Beta distribution: $\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$, updated as $\alpha_a \leftarrow \alpha_a + r_t$ and $\beta_a \leftarrow \beta_a + (1 - r_t)$ when arm $a$ is selected. Thompson Sampling achieves near-optimal regret and adapts naturally to changing environments.

### 23.10.3 Contextual Bandits

In contextual bandits, the agent observes a *context* (feature vector) $x_t$ before selecting an arm. The expected reward depends on both the arm and the context: $\mathbb{E}[r_t | x_t, a_t] = f(x_t, a_t)$. This setting lies between standard bandits and full RL — there is no state transition and no long-term credit assignment, but the optimal action depends on context.

Applications include ad placement (context = user features, arms = ad choices), recommendation systems, and clinical trials (context = patient features, arms = treatment options). Linear contextual bandits (LinUCB, LinTS) model $f(x, a) = x^T \theta_a$ and maintain uncertainty estimates over $\theta_a$.

---

## 23.11 Model-Based RL

Model-free methods (DQN, PPO) learn directly from interaction without modeling the environment dynamics. Model-based methods learn a model of the environment and use it for planning, potentially achieving much better sample efficiency.

### 23.11.1 World Models

Ha and Schmidhuber (2018) proposed *World Models* — an agent that learns a compressed spatial and temporal representation of its environment:

1. **Vision model (V):** A Variational Autoencoder (VAE) compresses high-dimensional observations into a low-dimensional latent vector $z$.
2. **Memory model (M):** An RNN (specifically an MDN-RNN — Mixture Density Network RNN) predicts future latent states: $P(z_{t+1} | a_t, z_t, h_t)$.
3. **Controller (C):** A simple linear policy $a_t = W_c [z_t; h_t] + b_c$ maps from the latent state to actions.

The key insight: the controller can be trained entirely inside the "dream" — using the learned world model to simulate trajectories, without any interaction with the real environment. This approach achieved strong results on car racing tasks with a remarkably simple controller.

### 23.11.2 Dreamer

Hafner et al. (2020, 2023) developed the Dreamer family of model-based agents, which learn world models and train policies entirely in imagination:

1. **Representation model** (encoder): Maps observations to latent states.
2. **Transition model** (dynamics): Predicts future latent states given actions.
3. **Reward model:** Predicts rewards from latent states.
4. **Value model** (critic): Estimates expected returns in imagination.
5. **Policy model** (actor): Selects actions to maximize imagined returns.

DreamerV3 (Hafner et al., 2023) introduced techniques for robust training across diverse domains — from Atari to continuous control to Minecraft — without domain-specific hyperparameter tuning. It uses symlog predictions (stabilizing learning with rewards of varying magnitudes), free bits for KL balancing, and percentile-based return normalization.

### 23.11.3 Planning with Learned Dynamics

Given a learned dynamics model $\hat{f}(s_{t+1} | s_t, a_t)$ and reward model $\hat{r}(s_t, a_t)$, several planning approaches are possible:

- **Model Predictive Control (MPC):** At each step, optimize an action sequence over a finite horizon using the model, execute the first action, re-plan at the next step.
- **Monte Carlo Tree Search (MCTS):** Build a search tree using the model, as in AlphaZero (Silver et al., 2018).
- **Dyna-style:** Interleave real experience with simulated experience from the model to accelerate value function learning.

---

## 23.12 Offline RL

Standard RL requires online interaction with the environment — the agent collects data, updates its policy, and collects more data with the updated policy. Offline RL (also called batch RL) learns policies from a fixed dataset of previously collected transitions, without any additional environment interaction (Levine et al., 2020).

### 23.12.1 The Distributional Shift Problem

Offline RL faces a critical challenge: the learned policy may select actions that are not well-represented in the dataset, leading to inaccurate Q-value estimates and catastrophic overestimation. Standard Q-learning exploits these overestimated values, producing policies that fail when deployed.

### 23.12.2 Conservative Q-Learning (CQL)

CQL (Kumar et al., 2020) addresses this by learning a conservative Q-function that lower-bounds the true Q-values:

$$\min_Q \alpha \, \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta} [Q(s, a)] \right] + \frac{1}{2} \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (Q(s,a) - \hat{\mathcal{B}}^\pi Q(s,a))^2 \right]$$

The first term pushes down Q-values for actions not in the dataset (the $\log\sum\exp$ term pushes up all actions, while the expectation under the behavior policy $\hat{\pi}_\beta$ pushes down dataset actions — the net effect pushes down out-of-distribution actions). The second term is the standard Bellman error.

### 23.12.3 Decision Transformer

Chen et al. (2021) reframed offline RL as a sequence modeling problem. Instead of fitting value functions, the Decision Transformer uses a GPT-style causal transformer to model trajectories:

$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots)$$

where $\hat{R}_t$ is the *return-to-go* (desired future return). At inference, conditioning on a high return-to-go prompts the model to generate actions from the best trajectories in the training data. This approach leverages the sequence modeling capabilities of transformers without requiring Bellman backups, temporal difference learning, or discount factors.

---

## 23.13 RLHF: Connecting RL to LLM Alignment

Reinforcement Learning from Human Feedback (RLHF) applies RL — specifically PPO — to align large language models with human preferences (Ouyang et al., 2022). Understanding the RL theory from this chapter is essential for understanding RLHF.

### 23.13.1 RLHF as an RL Problem

The RLHF pipeline can be mapped to RL concepts:

| RL Concept | RLHF Equivalent |
|------------|-----------------|
| Environment | The language generation process |
| State | Prompt + generated tokens so far |
| Action | Next token to generate |
| Policy | The LLM being fine-tuned: $\pi_\theta(\text{token} | \text{prompt + context})$ |
| Reward | Reward model score $r_\phi(x, y)$ for prompt $x$ and response $y$ |
| Episode | Generating a complete response |

### 23.13.2 The RLHF Pipeline

1. **Supervised Fine-Tuning (SFT):** Fine-tune the base LLM on high-quality demonstrations to obtain $\pi_\text{SFT}$.

2. **Reward Modeling:** Train a reward model $r_\phi$ on human preference data. Given pairs of responses $(y_w, y_l)$ where $y_w$ is preferred, minimize:

$$\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

This is the Bradley-Terry model of pairwise preferences.

3. **RL Fine-Tuning:** Optimize the LLM to maximize the reward model score while staying close to the SFT policy. The objective is:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta D_\text{KL}(\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)) \right]$$

The KL penalty $\beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ serves as a trust region constraint (cf. TRPO): it prevents the policy from deviating too far from the reference model $\pi_\text{ref}$ (usually $\pi_\text{SFT}$), which would lead to reward hacking — generating outputs that score highly on the imperfect reward model but are actually low quality.

**Why PPO for RLHF?** PPO is used because:
- Its clipped objective naturally constrains update size (combined with the KL penalty for additional safety).
- It handles the enormous action space (vocabulary size) and long horizons (token sequences) of language generation.
- It is relatively stable and does not require second-order optimization (unlike TRPO).

### 23.13.3 DPO: Direct Preference Optimization

Rafailov et al. (2023) showed that the RLHF objective has a closed-form solution, enabling direct optimization without training a separate reward model or running PPO:

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

DPO is simpler, more stable, and computationally cheaper than PPO-based RLHF, though debate continues about whether it is as effective for complex alignment tasks.

---

## Exercises

### Conceptual Questions

1. **Bellman equations.** Derive the Bellman expectation equation for $Q^\pi(s, a)$ from first principles. Show that the Bellman optimality operator is a contraction mapping and explain why this guarantees convergence of value iteration.

2. **On-policy vs off-policy.** Explain why SARSA is on-policy and Q-learning is off-policy. In what scenarios would SARSA produce a better practical policy than Q-learning? (Hint: consider the cliff-walking environment.)

3. **PPO vs TRPO.** Compare the trust region mechanisms in PPO (clipping) and TRPO (KL constraint). Why might clipping be less theoretically sound but more practical? Can you construct a scenario where clipping fails to properly constrain the update?

4. **RLHF theory.** Explain how the KL penalty in RLHF serves a similar role to the trust region in TRPO. What happens if $\beta$ is too small? Too large?

### Implementation Exercises

5. **Tabular Q-learning.** Implement Q-learning for the Taxi-v3 environment. Plot the learning curve. Implement Double Q-learning and compare the Q-value estimates — verify that standard Q-learning overestimates.

6. **DQN from scratch.** Implement DQN with experience replay and target networks for CartPole-v1. Add Double DQN and compare training stability. Visualize Q-values during training to observe overestimation.

7. **PPO for continuous control.** Implement PPO for the Pendulum-v1 environment (continuous action space). Include GAE, advantage normalization, and entropy bonus. Ablate each component — remove them one at a time and measure the impact on training.

8. **Multi-armed bandit comparison.** Implement epsilon-greedy, UCB, and Thompson Sampling. Run them on a 10-armed testbed with Gaussian rewards. Plot cumulative regret over 10,000 steps. Verify that Thompson Sampling achieves the lowest regret.

### Research Questions

9. **Sample efficiency.** Compare the sample efficiency of DQN, PPO, and DreamerV3 on Atari games. Why does model-based RL (Dreamer) require orders of magnitude fewer environment interactions? What are the tradeoffs?

10. **RLHF alternatives.** Compare PPO-based RLHF, DPO, and RAFT (Dong et al., 2023) for LLM alignment. When would each approach be most appropriate? Design an experiment to evaluate their relative strengths.

---

## References

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *CCS*.

2. Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Abbeel, P. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. *NeurIPS*.

3. Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122*.

4. Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR*.

5. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*.

6. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS*.

7. Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. *arXiv:2005.01643*.

8. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harber, T., ... & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.

9. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-Level Control Through Deep Reinforcement Learning. *Nature*, 518(7540), 529-533.

10. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS*.

11. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.

12. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. *ICLR*.

13. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust Region Policy Optimization. *ICML*.

14. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.

15. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

16. Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go Through Self-Play. *Science*, 362(6419), 1140-1144.

17. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

18. Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. *NeurIPS*.

19. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. *AAAI*.

20. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.

21. Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3-4), 279-292.

22. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*, 8(3-4), 229-256.
