# 🔥 PyroShield AI — Wildfire Containment & Resource Deployment

> **MLOps & RL Project:** An intelligent agent decides where to deploy firefighting resources on a grid-based wildfire simulator to minimise burned area, wrapped in a full production-ready MLOps platform.

![Wildfire Simulation](https://img.shields.io/badge/RL-Q--Learning%20%7C%20DQN%20%7C%20SARSA-orange)
![MLOps](https://img.shields.io/badge/MLOps-MLflow%20%7C%20DVC-blue)
![API](https://img.shields.io/badge/API-FastAPI-green)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-lightgrey)

## 🌍 SDG Alignment
| SDG | Connection |
|-----|-----------|
| **SDG 13 — Climate Action** | Reducing wildfire damage lowers CO₂ emissions and combats climate change |
| **SDG 15 — Life on Land** | Preserving forest ecosystems protects terrestrial biodiversity |

---

## 🏗️ System Architecture

Our MLOps architecture consists of four main components: Data Pipeline, Model Training & Tracking, CI/CD, and Serving.

```mermaid
graph TD
    subgraph Data Pipeline
        D[Data Generator] -->|DVC| R[Raw Data]
        R --> C[Cleaner/Normalizer]
        C --> F[Feature Engineering]
    end

    subgraph Training & Tracking
        F --> T[Model Tuning & Training]
        T -->|Logs| M[MLflow Tracking]
        T -->|Registers| MR[MLflow Registry]
    end

    subgraph CI/CD Pipeline
        GH[GitHub Actions] -->|Triggers| T
        GH -->|Builds| DB[Docker Build]
        DB -->|Deploys| K[Kubernetes/Helm]
    end

    subgraph Serving & Monitoring
        MR --> API[FastAPI Prediction Service]
        API -->|Prometheus Metrics| PROM[Prometheus]
        API -->|Drift Reports| DR[Data Drift Monitor]
    end
```

---

## 🚀 Quick Start

### 1. Local Setup
```bash
git clone <repo-url>
cd Wildfire-Containment-Resource-Deployment
pip install -r requirements.txt
```

### 2. DVC Data Pipeline
```bash
# Generate, clean data, and run model training stages
dvc repro
```

### 3. Training & MLflow Tracking
```bash
# Start MLflow UI
mlflow ui &

# Train a model (tracked automatically)
python src/tracking.py --config configs/qlearning_v1.yaml --algorithm qlearning
```

### 4. Start the Prediction API
```bash
# Using Docker Compose (API + MLflow + Prometheus)
docker-compose up -d

# Or locally
python -m api.app
```
*API available at `http://localhost:8000/docs`*

---

## 🤖 Models Supported

We have expanded the RL algorithms from a simple tabular agent to multiple options:
1. **Q-Learning** (Tabular, Off-Policy) - Good baseline.
2. **SARSA** (Tabular, On-Policy) - Safer, conservative deployment strategy.
3. **Double Q-Learning** (Tabular) - Reduces maximization bias for robust value estimation.
4. **DQN** (Neural Network, Replay Buffer) - Scales to complex, high-dimensional state spaces.

---

## 🔬 Experiment Tracking (MLflow & DVC)

- **Data Versioning:** Handled by DVC (`.dvc/config`, `dvc.yaml`).
- **Experiment Tracking:** MLflow logs parameters, metrics (Reward, Burned Cells, Convergence), and models.
- **Model Registry:** Best policies are registered and pulled automatically by the API container.

---

## ☸️ Production Deployment

The project is packaged for Kubernetes using Helm.
```bash
# Deploy to K8s cluster
helm upgrade --install pyroshield ./helm/pyroshield --namespace pyroshield --create-namespace
```

---

## 🛠️ Design Choices & Lessons Learned

### Key Design Choices
- **RL over Heuristics:** We chose Reinforcement Learning (Q-Learning/SARSA/DQN) because wildfire spread is highly stochastic. RL learns an optimal policy that adapts to changing wind dynamics rather than relying on rigid rules.
- **Microservices Architecture:** By decoupling the RL training loop from the FastAPI serving layer, we allow the API to scale independently and pull updated policies from the MLflow registry dynamically.
- **GitOps for K8s:** Using ArgoCD ensures that our deployment state always matches our Git repository, eliminating configuration drift and simplifying rollback.

### Challenges & Lessons Learned
- **Challenge:** Managing the curse of dimensionality in the RL state space.
  - **Lesson:** We had to implement feature engineering to compress the grid into macroscopic "sector states" to keep the tabular Q-learning matrix manageable before upgrading to DQN.
- **Challenge:** Reproducibility across different environments.
  - **Lesson:** Implementing DVC saved us hours of debugging by perfectly versioning the data generation and cleaning steps.

---

## 🧰 New Tools Explored
To achieve an MLOps 'Excellent' standard, we explored and integrated the following tools:
- **FastAPI:** For high-performance, asynchronous REST API serving.
- **MLflow:** For comprehensive experiment tracking and the Model Registry.
- **DVC (Data Version Control):** To version our datasets just like we version code.
- **ArgoCD:** To implement true GitOps continuous deployment for Kubernetes.
- **Prometheus:** For monitoring API latency, prediction volume, and data drift.


## 📋 Git Branches & Collaboration
- `main`: Production-ready code.
- `dev`: Integration branch where CI/CD runs.
- `feature/*`: For new work. PRs require reviews (see `CODEOWNERS`).

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## ✅ Version Management & Collaboration
This repository integrates several advanced tools for Version Management & Collaboration:
1. **Full Git Branching**: Strict `main`/`dev`/`feature` workflow documented in `CONTRIBUTING.md`.
2. **GitOps Auto-Sync**: An ArgoCD application manifest (`k8s/argocd-app.yaml`) tracks the `helm/pyroshield` directory to automatically sync configuration as code.
3. **Data Versioning**: `dvc.yaml` fully orchestrates the generation and tracking of data.
4. **Model Registry**: `train.py` and `evaluate.py` use MLflow (`mlflow.start_run()`, `mlflow.log_artifact()`) to actively track and register model binaries to the MLflow Registry.
5. **Team Collaboration**: Configured GitHub Issue Templates (`bug_report.md`, `feature_request.md`), a PR Template (`.github/pull_request_template.md`), and `CODEOWNERS` for automated review tracking.
