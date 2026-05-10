# Contributing to PyroShield AI 🔥

Thank you for your interest in contributing to the Wildfire Containment & Resource Deployment project!

## 🌿 Branch Strategy

We follow a structured Git branching model:

```
main        ← Production-ready, tagged releases
  └── dev   ← Integration branch, CI runs here
       └── feature/xyz  ← Individual feature branches
```

### Workflow

1. **Create a feature branch** from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**, commit with descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add SARSA agent implementation"
   ```

3. **Push and open a PR** to `dev`:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Review** → At least one approval required
5. **Merge to dev** → CI runs tests + training
6. **Release** → `dev` merged to `main` with version tag

### Commit Message Convention

```
type: short description

Types:
  feat     — New feature
  fix      — Bug fix
  docs     — Documentation
  refactor — Code restructuring
  test     — Adding/updating tests
  infra    — CI/CD, Docker, K8s changes
  data     — Data pipeline changes
  model    — ML model changes
```

## 🧪 Testing

Before submitting a PR:

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run training (quick check)
python train.py --config configs/qlearning_v1.yaml

# Check API
python -m api.app  # then curl http://localhost:8000/health
```

## 📊 Model Changes

If your PR modifies the ML pipeline:

1. Run both experiments and log to MLflow
2. Compare metrics (reward, burned area) before and after
3. Include comparison in PR description
4. Update configs if hyperparameters changed

## 🐳 Docker

Test Docker build locally:
```bash
docker build -t pyroshield .
docker run -p 8000:8000 pyroshield
```

## 📝 Code Style

- Follow PEP 8
- Max line length: 120 characters
- Use type hints where practical
- Add docstrings to all public functions
- Keep functions focused and under 50 lines where possible

## 🤝 Code Review Guidelines

Reviewers should check:
- [ ] Correctness of algorithm implementation
- [ ] Test coverage for new code
- [ ] No hardcoded paths or credentials
- [ ] Documentation updated
- [ ] CI pipeline passes
