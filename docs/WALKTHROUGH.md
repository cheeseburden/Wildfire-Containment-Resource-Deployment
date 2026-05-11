# 🔥 PyroShield AI — Final Report & Walkthrough

> **Project:** Wildfire Containment & Resource Deployment  
> **Team Repository:** [GitHub](https://github.com/cheeseburden/Wildfire-Containment-Resource-Deployment)  
> **SDG Alignment:** SDG 13 (Climate Action) + SDG 15 (Life on Land)

---

## Table of Contents

1. [Problem Statement & SDG Link](#1-problem-statement--sdg-link)
2. [Simulator](#2-simulator)
3. [Part A — RL Methodology](#3-part-a--rl-methodology)
   - [Algorithm Choice](#31-algorithm-choice)
   - [State, Action, Reward](#32-state-action-reward)
   - [Exploration & Convergence](#33-exploration--convergence)
   - [Saved Policies](#34-saved-policies)
4. [Part B — MLOps Implementation](#4-part-b--mlops-implementation)
   - [Versioning](#41-versioning)
   - [Experiment Tracking](#42-experiment-tracking)
   - [Reproducibility](#43-reproducibility)
   - [Monitoring Plan](#44-monitoring-plan)
5. [Baseline vs RL Comparison](#5-baseline-vs-rl-comparison)
6. [Results & Analysis](#6-results--analysis)
7. [SDG Impact Section](#7-sdg-impact-section)
8. [Project Structure](#8-project-structure)
9. [How to Run](#9-how-to-run)
10. [Limitations & Future Work](#10-limitations--future-work)

---

## 1. Problem Statement & SDG Link

**Problem:** Wildfires destroy millions of hectares of forest every year, releasing approximately 8 billion tonnes of CO₂ annually. Firefighting resources (ground crews, helicopters) are limited and must be deployed strategically to minimize burned area.

**Our Solution:** We built a **Reinforcement Learning agent** that learns optimal resource deployment strategies on a grid-based wildfire simulator. The agent decides *which sector of the fire to target* each timestep, aiming to minimize total burned area.

**SDG Connection:**

| SDG | How We Support It |
|-----|-------------------|
| **SDG 13 — Climate Action** | Reducing burned area directly lowers CO₂ and particulate emissions from wildfires |
| **SDG 15 — Life on Land** | Preserving forest cells protects terrestrial biodiversity, wildlife corridors, and prevents soil degradation |

---

## 2. Simulator

**File:** [`sim/wildfire_env.py`](../sim/wildfire_env.py) (339 lines)

Our simulator is a **10×10 grid-based wildfire environment** implementing the OpenAI Gym-style `reset()` / `step(action)` interface.

**Grid Cell States:**
| Value | State | Symbol |
|-------|-------|--------|
| 0 | Empty | `.` |
| 1 | Tree | `T` |
| 2 | Burning | `*` |
| 3 | Burned | `#` |
| 4 | Firebreak | `B` |

**Dynamics:**
- Trees are placed with 85% density at initialization
- Fire starts at 2 random tree cells
- Each timestep, burning cells spread fire to adjacent tree cells probabilistically
- **Wind direction** biases spread probability via dot-product bonus (8 directions supported)
- Burning cells have a 15% chance of burning out each step
- Episode ends when no fire remains or 50 steps are reached

**Key Design Choice:** The grid is divided into **4 sectors** (2×2 quadrants). The agent operates at the sector level rather than individual cell level, keeping the state space tractable for tabular methods while still providing meaningful strategic choices.

---

## 3. Part A — RL Methodology

### 3.1 Algorithm Choice

**Primary Algorithm:** Q-Learning (tabular, off-policy, ε-greedy)

> "Q-Learning is chosen because the state space (discretized grid sectors with binned burning/tree counts) is manageable for a tabular approach, and Q-Learning's off-policy nature allows efficient learning from exploratory actions without requiring on-policy samples."

**Additional Algorithms Implemented:**

| Algorithm | File | Why |
|-----------|------|-----|
| **Q-Learning** | `src/agent.py` | Primary — off-policy, simple, effective for discrete states |
| **SARSA** | `src/models/sarsa_agent.py` | On-policy — more conservative, safer deployment strategies |
| **Double Q-Learning** | `src/models/double_q_agent.py` | Reduces maximization bias for robust value estimation |
| **DQN** | `src/models/dqn_agent.py` | Neural network — scales to larger state spaces |

### 3.2 State, Action, Reward

**State** — Compact, hashable tuple for Q-table:
```
(burning_bin₁, tree_bin₁, burning_bin₂, tree_bin₂, burning_bin₃, tree_bin₃, burning_bin₄, tree_bin₄)
```
Each sector gets a binned burning count (0=none, 1=1-3, 2=4+) and tree fraction (0=<30%, 1=30-60%, 2=>60%).  
**State space:** 3⁸ = 6,561 possible states — tractable for tabular Q-learning.

**Action** — Select sector index (0–3) to deploy firefighting resources:
- Priority 1: Suppress a burning cell in the chosen sector (convert to firebreak)
- Priority 2: Create firebreak on a tree cell adjacent to fire in the sector
- Also suppresses neighboring burning cells with 35% probability

**Reward:**
```python
reward = -(newly_burned_cells) + (cells_suppressed × 2.0)
```
Negative reward for fire spread, positive reward for successful suppression.

### 3.3 Exploration & Convergence

**Exploration Strategy:** ε-greedy with exponential decay
- Start: ε = 1.0 (fully random)
- Decay: ε *= 0.995 per episode
- Minimum: ε = 0.05 (always 5% exploration)

**Convergence — Training over 800 episodes:**

| Phase | Episodes | ε Range | Behavior |
|-------|----------|---------|----------|
| **Exploration** | 1–300 | 1.0 → 0.22 | Agent explores widely, Q-table grows rapidly, rewards are noisy |
| **Convergence** | 300–600 | 0.22 → 0.05 | Agent starts exploiting learned strategy, reward stabilizes |
| **Exploitation** | 600–800 | 0.05 (min) | Consistent deployment near fire fronts, burned area minimized |

> "Average reward improves over time and stabilizes. The agent learns to prioritize sectors with active fires and create strategic firebreaks, resulting in a 36.9% reduction in burned area compared to random deployment."

Training results are saved to:
- `results/results_exp-qlearning-1.csv` — per-episode metrics
- `results/log_exp-qlearning-1.json` — run summary
- `results/reward_curve_exp-qlearning-1.png` — reward over episodes
- `results/burned_area_curve_exp-qlearning-1.png` — burned area over episodes

### 3.4 Saved Policies

We save **at least two policy versions** per experiment:

| Policy File | Description |
|-------------|-------------|
| `models/policy_exp-qlearning-1_ep400.pkl` | v1 checkpoint at episode 400 (still exploring) |
| `models/policy_exp-qlearning-1_ep800.pkl` | v1 checkpoint at episode 800 |
| `models/policy_exp-qlearning-1_final.pkl` | v1 final trained policy (**best performer**) |
| `models/policy_exp-qlearning-2_ep400.pkl` | v2 checkpoint at episode 400 |
| `models/policy_exp-qlearning-2_ep800.pkl` | v2 checkpoint at episode 800 |
| `models/policy_exp-qlearning-2_final.pkl` | v2 final trained policy |

---

## 4. Part B — MLOps Implementation

### 4.1 Versioning

**Git Branching Strategy:**
```
main            ← Production-ready, tagged releases
  └── dev       ← Integration branch, CI runs here
       └── feature/*    ← Individual feature branches
       └── docs/*       ← Documentation updates
```

**Experiment Tags:** Git commits and branch names map to experiments:
- `exp-qlearning-1` — Conservative exploration (v1 config)
- `exp-qlearning-2` — Higher LR + harder scenario (v2 config)

**Data Versioning:** DVC tracks raw and processed data files. The `dvc.yaml` defines the full pipeline, and `dvc.lock` pins exact data versions.

**Collaboration Tools:**
- `CODEOWNERS` — auto-assigns reviewers per directory
- PR template — standardized checklist for code review
- Issue templates — bug reports and feature requests
- `CONTRIBUTING.md` — branching workflow, commit conventions, testing requirements

### 4.2 Experiment Tracking

Each training run produces a **`results.csv`** and **`log.json`** containing:

```json
{
  "run_id": "exp-qlearning-1_20260511_104547",
  "experiment_name": "exp-qlearning-1",
  "timestamp": "2026-05-11T10:45:47",
  "episodes": 800,
  "average_reward": -271.17,
  "average_burned": 312.93,
  "final_reward_avg_50": -258.52,
  "final_burned_avg_50": 299.84,
  "parameters": {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "grid_size": 10,
    "wind_direction": "N",
    "base_spread_prob": 0.3
  },
  "training_time_seconds": 20.6,
  "q_table_states": 1539
}
```

**MLflow Integration** (`src/tracking.py`):
- Auto-logs hyperparameters, per-episode metrics, and artifacts
- Model Registry for policy versioning
- Cross-experiment comparison via MLflow UI
- Tags: `team=ml_engineering`, `sdg_alignment=SDG13,SDG15`

### 4.3 Reproducibility

**How to reproduce any experiment:**

```bash
# 1. Clone the repository
git clone https://github.com/cheeseburden/Wildfire-Containment-Resource-Deployment
cd Wildfire-Containment-Resource-Deployment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full DVC pipeline (data generation → cleaning → training → evaluation)
dvc repro

# 4. Or run individual experiments:
python train.py --config configs/qlearning_v1.yaml    # Experiment 1
python train.py --config configs/qlearning_v2.yaml    # Experiment 2

# 5. Evaluate
python evaluate.py --config configs/qlearning_v1.yaml
python evaluate.py --config configs/qlearning_v2.yaml
```

> "Anyone should be able to clone this repo and run the same experiment. All configurations are stored in YAML files under `configs/`, and the `dvc.yaml` pipeline ensures end-to-end reproducibility."

**Configuration files:**
- `configs/qlearning_v1.yaml` — Experiment 1: Conservative (N wind, 2 fires, α=0.1)
- `configs/qlearning_v2.yaml` — Experiment 2: Harder (NE wind, 3 fires, α=0.15)
- `configs/sarsa_v1.yaml` — SARSA variant
- `configs/double_q_v1.yaml` — Double Q-Learning variant
- `configs/dqn_v1.yaml` — DQN variant

### 4.4 Monitoring Plan

If this system were deployed for real-world wildfire management, we would monitor the following:

> **"We would track average burned area per incident, resource utilization rate (% of deployments that suppressed fire), response time (steps to containment), and data drift detection (monitoring whether the distribution of fire conditions in production diverges from training data). A 15% drift threshold on reward distribution would trigger automated retraining alerts."**

**Implementation available:**
- `GET /metrics` — Prometheus-compatible metrics endpoint (predictions count, uptime, Q-table size)
- `GET /drift/report` — Automated data drift detection comparing recent vs baseline reward distributions
- `infra/prometheus.yml` — Prometheus scrape configuration
- `docker-compose.yml` — Includes Prometheus service on port 9090

---

## 5. Baseline vs RL Comparison

We evaluate by running **100 episodes** each of a random baseline vs. the trained RL policy on the same simulator:

### Experiment 1 (N wind, 2 fires, spread=0.3)

| Metric | Random Baseline | RL Policy (Q-Learning) | Improvement |
|--------|----------------|----------------------|-------------|
| Avg Reward | -358.45 | -220.26 | +38.5% |
| Avg Burned Cells | 404.8 | 255.4 | **−36.9%** |

### Experiment 2 (NE wind, 3 fires, spread=0.35)

| Metric | Random Baseline | RL Policy (Q-Learning) | Improvement |
|--------|----------------|----------------------|-------------|
| Avg Reward | -375.23 | -346.55 | +7.6% |
| Avg Burned Cells | 424.1 | 396.3 | −6.5% |

---

## 6. Results & Analysis

### When RL Performs Well
- **Standard conditions** (Exp 1: 2 fires, N wind): The RL agent achieves **36.9% reduction** in burned area. The Q-table effectively captures the compressed sector states, and the agent learns to prioritize sectors with active fire.

### When RL Struggles
- **Harder scenarios** (Exp 2: 3 fires, NE wind, higher spread): Only 6.5% improvement. Multiple simultaneous fire fronts with diagonal wind create states the agent hasn't seen frequently enough during training, and the compressed state representation loses spatial detail.

### Sensitivity Analysis
- **Wind direction**: Diagonal winds (NE, SW) are harder — fire spreads to more neighbors simultaneously
- **Fire sources**: More initial fires = exponentially more complex state transitions
- **Spread probability**: Higher spread probability reduces the agent's window of opportunity to contain fire before it spreads beyond control

### Key Insight
> The tabular Q-Learning approach works well for standard scenarios but shows limitations for complex multi-fire situations. This motivates the DQN implementation (`src/models/dqn_agent.py`) which can handle higher-dimensional state representations via the feature engineering module (`src/features/engineering.py`).

---

## 7. SDG Impact Section

### SDG 13 — Climate Action 🌍
Reducing average burned area by **36.9%** directly lowers CO₂ and particulate emissions from wildfires. Wildfires release approximately 8 billion tonnes of CO₂ annually — optimized containment through RL-based resource deployment can significantly reduce this climate impact.

### SDG 15 — Life on Land 🌲
Every cell saved from burning represents preserved forest ecosystem and biodiversity. In our simulation, the RL agent saves approximately **149 cells** (out of ~100 grid cells per episode) compared to random deployment. Targeted resource deployment protects critical habitats, wildlife corridors, and prevents soil degradation from uncontrolled fires.

> **"Reducing average burned area by 36.9% supports SDG 13 (Climate Action) by lowering CO₂ emissions from wildfires, and SDG 15 (Life on Land) by preserving forest ecosystems and protecting biodiversity. Optimized resource deployment saves firefighting costs and lives."**

---

## 8. Project Structure

```
Wildfire-Containment-Resource-Deployment/
│
├── sim/                          # SIMULATOR
│   ├── __init__.py
│   └── wildfire_env.py           # Grid-based wildfire environment (339 lines)
│
├── src/                          # CORE ML CODE
│   ├── __init__.py
│   ├── agent.py                  # Q-Learning agent (113 lines)
│   ├── tracking.py               # MLflow tracking integration (270 lines)
│   ├── tuning.py                 # Hyperparameter tuning — grid/random search (395 lines)
│   ├── data/
│   │   ├── generate.py           # Data generation from simulations (172 lines)
│   │   └── clean.py              # Data cleaning pipeline (185 lines)
│   ├── models/
│   │   ├── sarsa_agent.py        # SARSA agent (109 lines)
│   │   ├── double_q_agent.py     # Double Q-Learning agent (123 lines)
│   │   └── dqn_agent.py          # Deep Q-Network in pure NumPy (242 lines)
│   └── features/
│       └── engineering.py        # Feature engineering for DQN (173 lines)
│
├── api/                          # FASTAPI SERVICE
│   ├── __init__.py
│   └── app.py                    # REST API with 7 endpoints (407 lines)
│
├── frontend/                     # INTERACTIVE DASHBOARD
│   ├── index.html                # Main page — 7 sections (557 lines)
│   ├── style.css                 # Premium dark theme CSS (363 lines)
│   ├── app.js                    # App orchestrator + animations (209 lines)
│   ├── simulator.js              # Browser wildfire simulator (346 lines)
│   ├── charts.js                 # Canvas training charts (149 lines)
│   └── data.js                   # Pre-baked training data
│
├── configs/                      # EXPERIMENT CONFIGS
│   ├── qlearning_v1.yaml         # Exp 1: Conservative (N wind, 2 fires)
│   ├── qlearning_v2.yaml         # Exp 2: Harder (NE wind, 3 fires)
│   ├── sarsa_v1.yaml             # SARSA config
│   ├── double_q_v1.yaml          # Double Q-Learning config
│   └── dqn_v1.yaml               # DQN config
│
├── models/                       # TRAINED POLICIES (DVC-tracked)
│   ├── policy_exp-qlearning-1_ep400.pkl
│   ├── policy_exp-qlearning-1_final.pkl
│   ├── policy_exp-qlearning-2_ep400.pkl
│   └── policy_exp-qlearning-2_final.pkl
│
├── results/                      # EXPERIMENT RESULTS
│   ├── results_exp-qlearning-1.csv
│   ├── log_exp-qlearning-1.json
│   ├── evaluation_exp-qlearning-1.json
│   ├── reward_curve_exp-qlearning-1.png
│   └── burned_area_curve_exp-qlearning-1.png
│
├── tests/                        # TEST SUITE
│   └── test_all.py               # 13 tests: env, agents, features, integration
│
├── k8s/                          # KUBERNETES
│   ├── deployment.yaml           # Deployment + Service + HPA + ConfigMap
│   └── argocd-app.yaml           # ArgoCD GitOps auto-sync
│
├── helm/pyroshield/              # HELM CHART
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│
├── infra/                        # INFRASTRUCTURE
│   └── prometheus.yml            # Prometheus monitoring config
│
├── docs/                         # DOCUMENTATION
│   ├── WALKTHROUGH.md            # This file — final report
│   └── ADR.md                    # Architecture Decision Records
│
├── .github/                      # CI/CD & COLLABORATION
│   ├── workflows/ci-cd.yml       # 5-stage GitHub Actions pipeline
│   ├── CODEOWNERS                # Auto-assign reviewers
│   ├── pull_request_template.md  # PR checklist
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── train.py                      # Main training entry point
├── evaluate.py                   # Baseline vs RL evaluation + plotting
├── Dockerfile                    # Multi-stage production Docker image
├── docker-compose.yml            # Full stack: API + MLflow + Prometheus
├── dvc.yaml                      # DVC reproducible pipeline (5 stages)
├── requirements.txt              # Python dependencies
├── CONTRIBUTING.md               # Branching strategy & contribution guide
├── CHANGELOG.md                  # Version history
└── README.md                     # Project overview & quick start
```

---

## 9. How to Run

### Quick Start (Local)
```bash
git clone https://github.com/cheeseburden/Wildfire-Containment-Resource-Deployment
cd Wildfire-Containment-Resource-Deployment
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
dvc repro    # Runs: generate → clean → train (×2) → evaluate (×2)
```

### Run Individual Steps
```bash
# Generate data
python -m src.data.generate

# Clean data
python -m src.data.clean

# Train
python train.py --config configs/qlearning_v1.yaml

# Evaluate
python evaluate.py --config configs/qlearning_v1.yaml

# Start API
python -m api.app     # http://localhost:8000/docs

# Start Frontend
python -m http.server 8080 -d frontend   # http://localhost:8080

# Run Tests
python -m pytest tests/ -v

# Docker
docker-compose up -d  # API + MLflow + Prometheus
```

---

## 10. Limitations & Future Work

**Current Limitations:**
- Tabular Q-Learning struggles with complex multi-fire scenarios (Exp 2 shows only 6.5% improvement)
- State compression loses spatial detail — the agent can't distinguish *where* in a sector the fire is
- Wind dynamics are simplified to a constant direction per episode
- No real-world fire data integration — simulator uses synthetic stochastic spread

**Future Improvements:**
- Upgrade to DQN with enriched features (spatial fire centroid, edge proximity — already implemented in `src/features/engineering.py`)
- Add multi-agent RL for coordinated deployment of multiple resource teams
- Integrate satellite fire detection data for real-world validation
- Implement curriculum learning — start with simple scenarios, gradually increase complexity
