# 🔥 Wildfire Containment & Resource Deployment — RL + MLOps

> **RL Problem Statement:** An agent decides where to deploy firefighting resources
> (ground crews, helicopters) on a grid-based wildfire simulator to **minimise
> burned area**.

## 🌍 SDG Alignment

| SDG | Connection |
|-----|-----------|
| **SDG 13 — Climate Action** | Reducing wildfire damage lowers CO₂ emissions and helps combat climate change |
| **SDG 15 — Life on Land** | Preserving forest ecosystems protects terrestrial biodiversity and habitats |

---

## 📁 Project Structure

```
wildfire-rl/
├── sim/                        # Wildfire simulator
│   ├── __init__.py
│   └── wildfire_env.py         # Grid-based fire spread environment
├── src/                        # RL agent source
│   ├── __init__.py
│   └── agent.py                # Q-Learning agent (tabular, ε-greedy)
├── configs/                    # Experiment configurations (YAML)
│   ├── qlearning_v1.yaml       # Conservative exploration
│   └── qlearning_v2.yaml       # More exploration + higher LR
├── models/                     # Saved policy checkpoints (.pkl)
├── results/                    # Experiment logs, CSVs, plots
├── experiments/                # Experiment notes
├── train.py                    # Main training script
├── evaluate.py                 # Baseline vs RL comparison + plots
├── requirements.txt
├── .gitignore
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (Experiment 1)

```bash
python train.py --config configs/qlearning_v1.yaml
```

### 3. Train (Experiment 2)

```bash
python train.py --config configs/qlearning_v2.yaml
```

### 4. Evaluate — Baseline vs RL

```bash
python evaluate.py --config configs/qlearning_v1.yaml
```

---

## 🎮 Simulator

The **WildfireEnv** is a grid-based wildfire spread simulator:

- **Grid:** 10×10 cells, each cell is one of: `EMPTY`, `TREE 🌲`, `BURNING 🔥`, `BURNED ⬛`, `FIREBREAK 🧱`
- **Fire Spread:** Probabilistic — each burning cell may ignite neighbouring trees, biased by wind direction
- **Resources:** The agent deploys firefighting resources to cells to create firebreaks or suppress active fires
- **Episode ends** when: fire is fully contained OR max steps reached

### State, Action, Reward

| Component | Description |
|-----------|------------|
| **State** | Flattened grid (10×10 = 100 cells, each with value 0–4) — represents fire spread map + resource positions |
| **Action** | Index 0–99 — the grid cell where the agent deploys resources next |
| **Reward** | `−(newly burned cells) + 0.5 × (cells suppressed)` — penalises fire spread, rewards suppression |

---

## 🤖 RL Algorithm

### Algorithm Choice: **Q-Learning**

> *"Q-learning is chosen because the state space (discretised grid) is manageable
> for a tabular approach, and Q-learning's off-policy nature allows efficient
> learning from exploratory actions."*

### Exploration Strategy: **ε-greedy with decay**

- Start with ε = 1.0 (fully random)
- Decay by factor of 0.995 per episode
- Minimum ε = 0.05

### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

Where:
- α = learning rate (0.1)
- γ = discount factor (0.95)
- r = immediate reward

### Convergence

> *"Average reward improves over time and stabilises — the agent learns to
> strategically place firebreaks near the fire front rather than deploying
> resources randomly."*

---

## 📊 Experiment Tracking (MLOps)

### Results CSV

Each training run produces `results/results_<experiment>.csv` containing:

| Column | Description |
|--------|------------|
| `run-id` | Unique run identifier |
| `episode` | Episode number |
| `reward` | Total episode reward |
| `total_burned` | Total cells burned |
| `epsilon` | Current exploration rate |
| `steps` | Steps taken in episode |
| `q_table_size` | Number of states discovered |

### Log JSON

Each run also produces `results/log_<experiment>.json` with:
- `run_id`, `episodes`, `average_reward`, `average_burned`
- `parameters` (ε, learning rate, etc.)
- `policy_files` — paths to saved policy checkpoints

### Policy Versions

| Policy File | Description |
|------------|------------|
| `policy_exp-qlearning-1_ep250.pkl` | V1 policy at 250 episodes (still exploring) |
| `policy_exp-qlearning-1_final.pkl` | V1 final policy (500 episodes, converged) |
| `policy_exp-qlearning-2_ep250.pkl` | V2 policy at 250 episodes |
| `policy_exp-qlearning-2_final.pkl` | V2 final policy |

---

## 🔁 Reproducibility

To reproduce a run:

```bash
python train.py --config configs/qlearning_v1.yaml
```

> Anyone should be able to clone this repo and run the same experiment.

**Steps:**
1. `git clone <repo-url>`
2. `pip install -r requirements.txt`
3. `python train.py --config configs/qlearning_v1.yaml`
4. `python evaluate.py --config configs/qlearning_v1.yaml`

---

## 📈 Baseline vs RL Comparison

After running `evaluate.py`, you'll get a comparison table:

| Metric | Random Baseline | RL Policy |
|--------|---------------:|----------:|
| Avg Reward | *(varies)* | *(higher)* |
| Avg Burned Cells | *(varies)* | *(lower)* |

### Plots Generated

1. **`results/reward_curve.png`** — Average reward over episodes (shows convergence)
2. **`results/burned_area_curve.png`** — Burned cells over episodes (shows improvement)

### When RL Performs Better

- When fire starts in predictable locations — agent learns optimal firebreak placement
- When wind direction is consistent — agent exploits directional knowledge

### When RL May Struggle

- Highly stochastic fire spread (high base probability)
- Very large grids where Q-table becomes too sparse
- Multiple simultaneous fire outbreaks overwhelming resources

### Sensitivity to Changes

- Changing wind direction requires retraining (different optimal strategies)
- Increasing fire count (3→5) degrades performance — agent needs more episodes

---

## 🌱 SDG Impact

> *"Reducing average burned area by X% supports SDG 13 (Climate Action) by
> lowering CO₂ and particulate emissions from wildfires, and SDG 15 (Life on
> Land) by preserving forest ecosystems and protecting biodiversity."*

Optimised resource deployment also:
- Reduces firefighting costs and response times
- Minimises risk to human lives and infrastructure
- Supports data-driven disaster management strategies

---

## 📡 Monitoring Plan (Design Only — No Live Deployment)

If this system were deployed in real-world wildfire management, we would monitor:

- **Burned area per incident** — primary KPI; should decrease over time
- **Resource utilisation rate** — fraction of deployed resources actively suppressing fire
- **Response time** — time between fire detection and resource deployment
- **False positive rate** — resources deployed to non-threatened areas
- **Model drift** — degradation of policy performance as climate/vegetation patterns change
- **Safety constraints** — ensure no resources deployed into actively burning zones (crew safety)

---

## 🏷️ Git Versioning

Use Git commits/tags for different experiments:

```bash
git tag exp-qlearning-1   # After first experiment
git tag exp-qlearning-2   # After second experiment
```

---

## 📋 Results & Limitations

### Results
- Q-learning agent learns to strategically deploy resources near fire fronts
- Trained policy consistently outperforms random baseline
- Performance improves and stabilises after ~300 episodes

### Limitations
- Tabular Q-learning doesn't scale well to larger grids (20×20+)
- Simplified fire model doesn't capture terrain, humidity, vegetation types
- Single-agent control — real wildfire response involves multi-agent coordination
- No temporal fire prediction — agent is reactive, not predictive

### Future Work
- Deep Q-Network (DQN) for larger state spaces
- Multi-agent RL for coordinated resource deployment
- Integration with real satellite fire detection data
