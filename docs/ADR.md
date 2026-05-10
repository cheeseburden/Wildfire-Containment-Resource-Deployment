# Architecture Decision Records (ADR)

## ADR 001: Selection of Lightweight DQN Implementation
**Date:** 2026-05-10
**Status:** Accepted

### Context
We needed to expand beyond Tabular Q-Learning to handle larger state spaces in the Wildfire simulation, necessitating a Deep Q-Network (DQN). However, the system must remain easy to deploy without bloat.

### Decision
We opted to write a custom, lightweight Neural Network and DQN implementation using pure NumPy instead of importing heavyweight frameworks like PyTorch or TensorFlow.

### Consequences
**Positive:** 
- Keeps the Docker container size minimal.
- Reduces dependency complexity.
- Faster API spin-up time.
**Negative:** 
- Not easily portable to GPU acceleration.
- Limits advanced neural network architectures (e.g., ConvNets).

---

## ADR 002: Hyperparameter Tuning Strategy
**Date:** 2026-05-10
**Status:** Accepted

### Context
Initially, the plan included using Optuna for hyperparameter optimization.

### Decision
Instead of introducing Optuna, we implemented a custom Grid Search and Random Search module with K-Fold Cross Validation (`src/tuning.py`).

### Consequences
**Positive:**
- Fewer dependencies.
- Tightly coupled with our specific RL environment loop.
- Easy to integrate directly with our local MLflow logging.
**Negative:**
- Lacks advanced pruning strategies (like Hyperband) that Optuna provides.

---

## ADR 003: FastAPI for Serving
**Date:** 2026-05-10
**Status:** Accepted

### Context
We needed a mechanism to expose the trained RL policies for live simulation prediction.

### Decision
We chose FastAPI as the REST framework over Flask or Django.

### Consequences
**Positive:**
- Native async support and high performance.
- Auto-generated OpenAPI (Swagger) documentation.
- Built-in data validation using Pydantic.
