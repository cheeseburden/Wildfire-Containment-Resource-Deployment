# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-10
### Added
- **MLOps Architecture**: Fully transformed the project to an MLOps platform.
- **Data Pipeline**: New `src/data/generate.py` and `src/data/clean.py` for structured episode trajectories.
- **Feature Engineering**: Advanced spatial, temporal, and contextual features (`src/features/engineering.py`).
- **Multiple Models**: Added SARSA, Double Q-Learning, and a custom NumPy DQN to the model zoo.
- **Hyperparameter Tuning**: Custom Grid and Random Search implementation with cross-validation (`src/tuning.py`).
- **MLflow**: Experiment tracking and model registry integration (`src/tracking.py`).
- **FastAPI Service**: Production prediction service with drift detection and Prometheus metrics (`api/app.py`).
- **Docker & Compose**: Multi-stage `Dockerfile` and `docker-compose.yml` for local API and monitoring spin-up.
- **Kubernetes & Helm**: Deployment manifests, Services, HPA, and a fully featured Helm Chart.
- **CI/CD**: GitHub Actions pipeline covering linting, testing, training, Docker build, and K8s deployment.
- **DVC**: Added Data Version Control config and `dvc.yaml` pipeline.
- **Collaboration**: PR Templates, Issue Templates, CODEOWNERS, and CONTRIBUTING guidelines.
- **Testing**: Pytest suite for environment, agents, features, and full training loop integration.

### Changed
- Reorganized directory structure into `src/`, `sim/`, `api/`, `k8s/`, `helm/`, `tests/`, etc.
- Updated `requirements.txt` and `.gitignore`.
- Refactored `src/agent.py` and simulation interactions to allow dynamic agent plugging.

## [1.0.0] - Initial Version
### Added
- Tabular Q-Learning agent with ε-greedy exploration.
- Basic grid-based Wildfire Simulator (`sim/wildfire_env.py`).
- Training and evaluation scripts with CSV/JSON logging.
- Static dashboard visualization.
