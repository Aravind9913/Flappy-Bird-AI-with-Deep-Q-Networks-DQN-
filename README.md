# 🎮 Flappy Bird AI with Deep Q-Networks (DQN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-darkgreen.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Reinforcement Learning (DRL) agent trained to master the classic **Flappy Bird** game using **Deep Q-Learning (DQN)** with **Experience Replay** and **Target Networks** implemented in PyTorch and Gymnasium.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [How Deep Q-Learning (DQN) Works](#-how-deep-q-learning-dqn-works)
  - [1. Reinforcement Learning Basics](#1-reinforcement-learning-basics)
  - [2. Deep Q-Network Architecture](#2-deep-q-network-architecture)
  - [3. Experience Replay](#3-experience-replay)
  - [4. Dual-Network Stability (Policy & Target Networks)](#4-dual-network-stability-policy--target-networks)
  - [5. Epsilon-Greedy Exploration Strategy](#5-epsilon-greedy-exploration-strategy)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [File-by-File Explanation](#-file-by-file-explanation)
- [Hyperparameters Explained](#-hyperparameters-explained)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution](#-usage--execution)
  - [1. Play the Game Manually](#1-play-the-game-manually)
  - [2. Train the DQN Agent](#2-train-the-dqn-agent)
  - [3. Test / Watch the Trained Agent Play](#3-test--watch-the-trained-agent-play)
- [Training Workflow & Math](#-training-workflow--math)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Overview

The goal of this project is to train an autonomous agent to play **Flappy Bird** without human intervention. Instead of relying on hardcoded heuristics, the bird learns optimal flapping actions solely through trial, error, and scalar reward signals received from the environment (`flappy-bird-gymnasium`).

### Key Highlights
- **Framework**: Built on modern [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).
- **Accelerated Compute**: Automatically detects and utilizes **NVIDIA CUDA**, **Apple Silicon MPS**, or **CPU**.
- **Double DQN Mechanism**: Decouples action selection from action evaluation using a separate Target Network to avoid overestimation bias.
- **Experience Replay Buffer**: Breaks correlation between consecutive transitions for stable gradient updates.
- **Configurable Hyperparameters**: Clean separation of configuration (`parameters.yaml`) from implementation logic.

---

## 🧠 How Deep Q-Learning (DQN) Works

### 1. Reinforcement Learning Basics

In reinforcement learning, an **agent** interacts with an **environment** over discrete time steps:
- **State ($s_t$)**: A 12-dimensional continuous feature vector describing the bird's vertical position, velocity, and relative coordinates to upcoming pipes.
- **Action ($a_t$)**: The action taken by the bird at time $t$:
  - `0`: Do nothing (free-fall under gravity)
  - `1`: Flap wings (upward impulse)
- **Reward ($r_t$)**: Environmental feedback signal:
  - $+0.1$: Survived a frame
  - $+1.0$: Successfully passed through a pipe gap
  - $-1.0$: Collided with a pipe or ground (terminal state)
- **Next State ($s_{t+1}$)**: The resulting state after taking action $a_t$.
- **Done ($d_t$)**: Boolean flag indicating if the bird died or reached the terminal condition.

### 2. Deep Q-Network Architecture

In traditional Q-learning, a table stores values for each $(s, a)$ pair. Because Flappy Bird has a continuous state space, tabular Q-learning is impractical. 

DQN approximates the optimal action-value function $Q^*(s, a)$ using a neural network:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

- **Input Layer**: $12$ neurons (state dimensions)
- **Hidden Layer**: $256$ neurons with **ReLU** activation
- **Output Layer**: $2$ neurons (estimated Q-values for action `0` and action `1`)

The optimal action at inference is chosen greedily:
$$a^* = \arg\max_{a} Q(s, a; \theta)$$

### 3. Experience Replay

Consecutive gameplay frames are highly correlated. If the network trained directly on sequential samples, updates would suffer from high variance and catastrophic forgetting.

**Experience Replay** stores past transitions in a cyclic FIFO buffer:
$$e_t = (s_t, a_t, s_{t+1}, r_t, \text{terminated})$$

During training, uniform random minibatches of size $32$ are sampled from this buffer. This:
1. Breaks temporal correlation between samples.
2. Reuses rare high-reward experiences multiple times.
3. Smooths the training distribution over past states.

### 4. Dual-Network Stability (Policy & Target Networks)

Standard Q-learning with neural networks can oscillate or diverge because the target value depends on the very weights being updated. DQN introduces two networks:

1. **Policy Network ($Q_{\text{policy}}$ with weights $\theta$)**:
   Updated at every training step via gradient descent.
2. **Target Network ($Q_{\text{target}}$ with weights $\theta^-$)**:
   Kept stationary and periodically synchronized with $\theta$ every $N$ steps (`network_sync_rate`).

The **Bellman Target** is formulated as:
$$y_i = r_i + (1 - \text{terminated}_i) \cdot \gamma \max_{a'} Q_{\text{target}}(s'_{i}, a'; \theta^-)$$

The loss is the Mean Squared Error (MSE) between predicted Q-values and Bellman targets:
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left( Q_{\text{policy}}(s_i, a_i; \theta) - y_i \right)^2$$

### 5. Epsilon-Greedy Exploration Strategy

To balance **exploration** (discovering new strategies) and **exploitation** (choosing the known best action):
- With probability $\epsilon$, the agent chooses a random action.
- With probability $1 - \epsilon$, the agent chooses $\arg\max_a Q(s, a)$.

$\epsilon$ begins at $\epsilon_{\text{init}} = 1.0$ (100% random) and decays multiplicatively after each episode:
$$\epsilon_{k+1} = \max(\epsilon_{\text{min}},\, \epsilon_k \times \epsilon_{\text{decay}})$$
where $\epsilon_{\text{decay}} = 0.9995$ and $\epsilon_{\text{min}} = 0.05$.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Environment ["Gymnasium: FlappyBird-v0"]
        ENV["Flappy Bird Game Engine"]
    end

    subgraph AgentPipeline ["DQN Agent Pipeline"]
        direction TB
        OBS["State Observation (12D Vector)"]
        EPS{"random() < epsilon?"}
        ACT_RAND["Random Action (Explore)"]
        ACT_POL["ArgMax Q_policy(s) (Exploit)"]
        ACTION["Action: 0 (Idle) or 1 (Flap)"]

        MEM[("Replay Memory (100k transitions)")]
        
        POL_NET["Policy Network Q(s, a; θ)"]
        TAR_NET["Target Network Q(s', a'; θ⁻)"]
        
        LOSS["MSE Loss: (Q_policy - Target_Q)²"]
        OPT["Adam Optimizer (lr = 0.001)"]
    end

    ENV -->|State, Reward, Done| OBS
    OBS --> EPS
    EPS -->|Yes| ACT_RAND --> ACTION
    EPS -->|No| POL_NET --> ACT_POL --> ACTION
    ACTION -->|Execute| ENV

    OBS -.->|Store (s, a, s', r, done)| MEM
    MEM -->|Sample Mini-batch (size=32)| POL_NET
    MEM -->|Sample Mini-batch (size=32)| TAR_NET
    
    POL_NET --> LOSS
    TAR_NET --> LOSS
    LOSS --> OPT
    OPT -->|Update θ| POL_NET
    POL_NET -.->|Sync weights every 10 steps| TAR_NET
```

---

## 📂 Project Directory Structure

```text
RL_DQN-main/
│
├── agent.py               # Core DQN training and inference loop
├── dqn.py                 # PyTorch Neural Network architecture (MLP)
├── experience_replay.py   # ReplayMemory data structure (FIFO buffer)
├── game_flappy_bird.py    # Manual playable mode (PyGame human control)
├── parameters.yaml        # YAML configuration for hyperparameters
├── runs/                  # Saved model checkpoints and training logs
│   ├── flappybirdv0.log   # Training logs with best reward progression
│   └── flappybirdv0.pt    # PyTorch state_dict checkpoint of best model
├── .gitignore             # Git ignore file for Python & model artifacts
└── README.md              # Detailed documentation and project guide
```

---

## 🔍 File-by-File Explanation

### 1. `dqn.py`
Defines the `DQN` class extending `torch.nn.Module`.
- **`__init__(state_dim=12, action_dim=2, hidden_dim=256)`**: Instantiates a 2-layer Multi-Layer Perceptron (Linear $\to$ ReLU $\to$ Linear).
- **`forward(x)`**: Computes raw Q-value predictions for each discrete action given a batch of states.

### 2. `experience_replay.py`
Implements the experience replay mechanism via the `ReplayMemory` class.
- **`__init__(maxlen)`**: Initializes a bounded `collections.deque` with the specified maximum buffer capacity.
- **`append(new_exp)`**: Pushes transition tuple `(state, action, next_state, reward, terminated)` into the queue.
- **`sample(sample_size)`**: Selects a uniform random batch of experiences using `random.sample`.
- **`__len__()`**: Returns current stored experience count.

### 3. `parameters.yaml`
Centralized YAML configuration holding training parameters and environment IDs under named profiles (e.g., `flappybirdv0`). Allows adjusting hyperparameters without modifying code.

### 4. `agent.py`
The orchestrator of the project.
- **`Agent.__init__(param_set)`**: Loads parameters from YAML, configures MSE loss, initializes paths for logs and model checkpoints (`runs/`), and detects device (`cuda`, `mps`, or `cpu`).
- **`Agent.run(is_training=True, render=False)`**:
  - Training loop: Initializes `Policy DQN` and `Target DQN`, manages $\epsilon$-greedy exploration, records experiences into `ReplayMemory`, triggers optimization, syncs networks, and checkpoints the best policy whenever a new high score is achieved.
  - Evaluation loop: Loads `runs/<param_set>.pt`, disables exploration ($\epsilon = 0$), and visualizes the bird playing autonomously in real time.
- **`Agent.optimize(mini_batch, policy_dqn, target_dqn)`**: Unpacks experiences, computes target Q-values using the Bellman equation, evaluates predicted Q-values via `.gather()`, computes MSE loss, and runs backpropagation with `Adam`.
- **`CLI Entrypoint`**: Uses `argparse` to handle `--train` flag and parameter set selection.

### 5. `game_flappy_bird.py`
A standalone script that launches the `FlappyBird-v0` environment in human-playable mode:
- Opens a PyGame window.
- Captures the `SPACE` key to flap the bird.
- Allows you to experience the physics, gravity, and difficulty firsthand to appreciate the agent's task.

---

## ⚙️ Hyperparameters Explained

All hyperparameters are declared in [`parameters.yaml`](parameters.yaml):

| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| `env_id` | `FlappyBird-v0` | Name of the Gymnasium environment to instantiate. |
| `epsilon_init` | `1.0` | Initial exploration rate (100% random actions at start). |
| `epsilon_min` | `0.05` | Minimum exploration floor to maintain lifelong exploration. |
| `epsilon_decay` | `0.9995` | Multiplicative decay applied to $\epsilon$ per episode ($\epsilon \leftarrow \epsilon \times \text{decay}$). |
| `replay_memory_size`| `100000` | Maximum capacity of the Experience Replay FIFO buffer. |
| `mini_batch_size` | `32` | Number of transition tuples sampled for each gradient step. |
| `network_sync_rate` | `10` | Frequency (in steps) to copy weights from Policy DQN to Target DQN. |
| `alpha` | `0.001` | Learning rate for the `Adam` optimizer. |
| `gamma` | `0.99` | Discount factor for future rewards ($\gamma \in [0, 1)$). |
| `reward_threshold` | `1000` | Episode reward threshold considered as game mastery. |

---

## 🚀 Installation & Setup

### Prerequisites
- Python **3.8+** (recommended: Python 3.10 or 3.11)
- Windows, macOS, or Linux

### 1. Clone or Navigate to the Repository
```bash
git clone https://github.com/your-username/RL_DQN.git
cd RL_DQN/RL_DQN-main
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
You can install all dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

Or install manually based on your hardware:
```bash
# PyTorch with NVIDIA CUDA GPU acceleration:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# OR PyTorch for CPU:
pip install torch torchvision

# Core RL dependencies:
pip install gymnasium flappy-bird-gymnasium pygame pyyaml
```

---

## 🎮 Usage & Execution

### 1. Play the Game Manually
Test your own Flappy Bird skills against the environment physics:
```bash
python game_flappy_bird.py
```
- **Controls**: Press <kbd>Spacebar</kbd> to flap.
- Close the window to exit.

---

### 2. Train the DQN Agent
Train the agent using the `flappybirdv0` parameter set:
```bash
python agent.py flappybirdv0 --train
```

**What happens during training:**
- A `runs/` folder is created automatically.
- Logs are written to `runs/flappybirdv0.log` whenever the agent reaches a new highest reward.
- The best performing model weights are saved to `runs/flappybirdv0.pt`.
- Terminal prints episode statistics:
  ```text
  episode=1 with total reward=3.4000000000000012 & epsilon=0.9995
  episode=2 with total reward=2.8 & epsilon=0.99900025
  ...
  episode=450 with total reward=104.2 & epsilon=0.7984...
  ```

---

### 3. Test / Watch the Trained Agent Play
Once trained (or using an existing checkpoint in `runs/flappybirdv0.pt`), watch the trained model play with real-time rendering:
```bash
python agent.py flappybirdv0
```
*(Notice: omit the `--train` flag. The script automatically loads `runs/flappybirdv0.pt` and sets `render=True`).*

---

## 📐 Training Workflow & Math

### Step-by-Step Training Iteration:

1. **State Observation**: Current state $s_t$ is extracted from `env.reset()` or `env.step()`.
2. **Action Selection**:
   $$a_t = \begin{cases} \text{random action from } \{0, 1\} & \text{if } u < \epsilon \\ \arg\max_a Q_{\text{policy}}(s_t, a) & \text{otherwise} \end{cases}$$
   where $u \sim \mathcal{U}(0, 1)$.
3. **Environment Step**: Execute $a_t$ in environment to obtain $(s_{t+1}, r_t, \text{terminated})$.
4. **Buffer Push**: Store $(s_t, a_t, s_{t+1}, r_t, \text{terminated})$ in `ReplayMemory`.
5. **Batch Sampling**: When memory size exceeds `mini_batch_size` (32), sample a random minibatch.
6. **Bellman Target Formulation**:
   $$y_i = r_i + (1 - \text{terminated}_i) \cdot \gamma \cdot \max_{a'} Q_{\text{target}}(s'_{i}, a')$$
7. **Loss Calculation & Optimization**:
   $$\mathcal{L} = \text{MSE}\left( Q_{\text{policy}}(s_i, a_i),\, y_i \right)$$
   $$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$$
8. **Target Synchronization**: Every $10$ environment steps, clone weights:
   $$\theta^- \leftarrow \theta$$
9. **Anneal Epsilon**: At the end of each episode:
   $$\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \times \epsilon_{\text{decay}})$$

---

## ❓ Troubleshooting & FAQs

#### Q1: "ModuleNotFoundError: No module named 'flappy_bird_gymnasium'"
> **Solution**: Ensure your virtual environment is active and install the package with:
> ```bash
> pip install flappy-bird-gymnasium
> ```

#### Q2: The PyGame window does not appear during evaluation
> **Solution**: Make sure PyGame has display privileges. If running on a headless Linux server or Docker container, configure a virtual display:
> ```bash
> sudo apt-get install -y xvfb
> xvfb-run python agent.py flappybirdv0
> ```

#### Q3: How do I train on GPU?
> `agent.py` automatically checks for GPU availability:
> ```python
> if torch.backends.mps.is_available():
>     device = "mps"
> elif torch.cuda.is_available():
>     device = "cuda"
> else: 
>     device = "cpu"
> ```
> To verify PyTorch detects your GPU, run:
> ```bash
> python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
> ```

#### Q4: Why does the bird perform poorly in the first 100 episodes?
> This is completely normal. In the beginning, $\epsilon \approx 1.0$, meaning the bird takes almost entirely random actions to explore the state space. As $\epsilon$ decays and the replay buffer gathers high-reward pipe traversals, the policy converges rapidly.

---

## 📜 License

This project is licensed under the MIT License - feel free to modify, extend, and use it for research and educational purposes.
