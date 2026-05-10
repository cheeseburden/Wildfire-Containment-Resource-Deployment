"""
FastAPI REST API — Wildfire Containment Prediction Service
=============================================================
Production-grade API for real-time wildfire resource deployment
predictions using trained RL policies.

Endpoints:
  POST /predict        — Get optimal sector deployment for given state
  POST /simulate       — Run a full simulation episode
  GET  /health         — Health check
  GET  /metrics        — Prometheus-compatible metrics
  GET  /model/info     — Current model metadata
  POST /retrain        — Trigger model retraining
  GET  /drift/report   — Data drift detection report
"""

import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import List, Optional

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")

import numpy as np

from sim.wildfire_env import WildfireEnv
from src.agent import QLearningAgent

# ─── Logging Setup ───────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# Prediction logger
pred_logger = logging.getLogger("predictions")
pred_handler = logging.FileHandler("logs/predictions.jsonl")
pred_handler.setFormatter(logging.Formatter("%(message)s"))
pred_logger.addHandler(pred_handler)
pred_logger.setLevel(logging.INFO)

# API logger
api_logger = logging.getLogger("api")
api_handler = logging.FileHandler("logs/api.log")
api_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
api_logger.addHandler(api_handler)
api_logger.setLevel(logging.INFO)

# ─── Pydantic Models ─────────────────────────────────────────


class PredictRequest(BaseModel):
    state: List[int] = Field(
        ..., description="Compressed state tuple (8 values for 4 sectors)"
    )
    model_version: Optional[str] = Field(
        None, description="Specific model version to use"
    )


class PredictResponse(BaseModel):
    action: int
    sector_name: str
    confidence: float
    q_values: List[float]
    model_version: str
    timestamp: str
    latency_ms: float


class SimulateRequest(BaseModel):
    wind_direction: str = Field("N", description="Wind direction (N/S/E/W/NE/NW/SE/SW)")
    num_fires: int = Field(2, ge=1, le=5)
    spread_prob: float = Field(0.3, ge=0.1, le=0.6)
    use_rl: bool = Field(True, description="Use RL agent vs random baseline")
    episodes: int = Field(10, ge=1, le=100)


class SimulateResponse(BaseModel):
    avg_reward: float
    avg_burned: float
    episodes_run: int
    agent_type: str
    results: List[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int
    version: str


class ModelInfoResponse(BaseModel):
    algorithm: str
    model_path: str
    q_table_size: int
    epsilon: float
    loaded_at: str


class DriftReport(BaseModel):
    drift_detected: bool
    metric: str
    baseline_mean: float
    current_mean: float
    pct_change: float
    window_size: int
    recommendation: str


# ─── Application ──────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="🔥 PyroShield AI — Wildfire Containment API",
        description="Production RL prediction service for optimal firefighting resource deployment",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Global State ────────────────────────────────────────
    STARTUP_TIME = time.time()
    TOTAL_PREDICTIONS = 0
    PREDICTION_HISTORY = deque(maxlen=1000)  # Last 1000 predictions for drift
    REWARD_HISTORY = deque(maxlen=500)

    SECTOR_NAMES = {
        0: "NW Quadrant",
        1: "NE Quadrant",
        2: "SW Quadrant",
        3: "SE Quadrant",
    }

    # Load model
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/policy_exp-qlearning-1_final.pkl")
    AGENT = None
    MODEL_LOADED_AT = None

    def load_model(path=None):
        global AGENT, MODEL_LOADED_AT, MODEL_PATH
        if path:
            MODEL_PATH = path

        if not os.path.exists(MODEL_PATH):
            api_logger.warning(f"Model not found: {MODEL_PATH}")
            return False

        env = WildfireEnv(grid_size=10, num_sectors_per_side=2)
        AGENT = QLearningAgent(
            state_size=env.state_size,
            action_size=env.action_size,
        )
        AGENT.load(MODEL_PATH)
        AGENT.epsilon = 0.0  # Pure exploitation
        MODEL_LOADED_AT = datetime.now().isoformat()
        api_logger.info(f"Model loaded: {MODEL_PATH}")
        return True

    @app.on_event("startup")
    async def startup():
        load_model()

    # ─── Middleware: Request Logging ──────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        latency = (time.time() - start) * 1000
        api_logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latency={latency:.1f}ms"
        )
        return response

    # ─── Endpoints ────────────────────────────────────────────

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        """Get optimal sector deployment for a given wildfire state."""
        global TOTAL_PREDICTIONS

        if AGENT is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(req.state) != 8:
            raise HTTPException(
                status_code=400,
                detail=f"State must have 8 values, got {len(req.state)}",
            )

        start = time.time()
        state = tuple(req.state)

        # Get Q-values
        q_values = AGENT.q_table[state].tolist()
        action = int(np.argmax(q_values))

        # Confidence = softmax probability of chosen action
        q_arr = np.array(q_values)
        exp_q = np.exp(q_arr - np.max(q_arr))
        confidence = float(exp_q[action] / exp_q.sum())

        latency = (time.time() - start) * 1000
        TOTAL_PREDICTIONS += 1

        # Log prediction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "state": req.state,
            "action": action,
            "q_values": q_values,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency, 2),
            "model_version": MODEL_PATH,
        }
        pred_logger.info(json.dumps(log_entry))
        PREDICTION_HISTORY.append(log_entry)

        return PredictResponse(
            action=action,
            sector_name=SECTOR_NAMES.get(action, f"Sector {action}"),
            confidence=round(confidence, 4),
            q_values=[round(q, 4) for q in q_values],
            model_version=os.path.basename(MODEL_PATH),
            timestamp=datetime.now().isoformat(),
            latency_ms=round(latency, 2),
        )

    @app.post("/simulate", response_model=SimulateResponse)
    async def simulate(req: SimulateRequest):
        """Run simulation episodes and return results."""
        env_cfg = {
            "grid_size": 10,
            "num_resources": 2,
            "base_spread_prob": req.spread_prob,
            "wind_spread_bonus": 0.2,
            "wind_direction": req.wind_direction,
            "max_steps": 50,
            "num_initial_fires": req.num_fires,
            "tree_density": 0.85,
            "num_sectors_per_side": 2,
        }

        results = []
        for ep in range(req.episodes):
            env = WildfireEnv(**env_cfg)
            state = env.reset()
            total_reward = 0

            while True:
                if req.use_rl and AGENT is not None:
                    action = AGENT.choose_action(state)
                else:
                    action = np.random.randint(0, env.action_size)

                state, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break

            results.append(
                {
                    "episode": ep,
                    "reward": round(float(total_reward), 2),
                    "burned": int(env.total_burned),
                }
            )
            REWARD_HISTORY.append(total_reward)

        avg_r = float(np.mean([r["reward"] for r in results]))
        avg_b = float(np.mean([r["burned"] for r in results]))

        return SimulateResponse(
            avg_reward=round(avg_r, 2),
            avg_burned=round(avg_b, 1),
            episodes_run=req.episodes,
            agent_type="rl_qlearning" if req.use_rl else "random_baseline",
            results=results,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Service health check."""
        return HealthResponse(
            status="healthy",
            model_loaded=AGENT is not None,
            uptime_seconds=round(time.time() - STARTUP_TIME, 1),
            total_predictions=TOTAL_PREDICTIONS,
            version="2.0.0",
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info():
        """Current model metadata."""
        if AGENT is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        return ModelInfoResponse(
            algorithm="Q-Learning (tabular)",
            model_path=MODEL_PATH,
            q_table_size=AGENT.get_q_table_size(),
            epsilon=AGENT.epsilon,
            loaded_at=MODEL_LOADED_AT or "unknown",
        )

    @app.get("/drift/report", response_model=DriftReport)
    async def drift_report():
        """Data drift detection based on prediction distribution."""
        if len(REWARD_HISTORY) < 50:
            return DriftReport(
                drift_detected=False,
                metric="reward",
                baseline_mean=0.0,
                current_mean=0.0,
                pct_change=0.0,
                window_size=len(REWARD_HISTORY),
                recommendation="Insufficient data. Need at least 50 observations.",
            )

        rewards = list(REWARD_HISTORY)
        mid = len(rewards) // 2
        baseline = np.mean(rewards[:mid])
        current = np.mean(rewards[mid:])
        pct_change = ((current - baseline) / (abs(baseline) + 1e-8)) * 100

        drift = abs(pct_change) > 15.0  # 15% threshold

        return DriftReport(
            drift_detected=drift,
            metric="reward",
            baseline_mean=round(float(baseline), 2),
            current_mean=round(float(current), 2),
            pct_change=round(float(pct_change), 2),
            window_size=len(rewards),
            recommendation=(
                "ALERT: Significant performance drift detected. Consider retraining."
                if drift
                else "Performance stable. No action needed."
            ),
        )

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus-compatible metrics endpoint."""
        uptime = time.time() - STARTUP_TIME
        metrics = f"""# HELP pyroshield_predictions_total Total predictions served
# TYPE pyroshield_predictions_total counter
pyroshield_predictions_total {TOTAL_PREDICTIONS}

# HELP pyroshield_uptime_seconds Service uptime
# TYPE pyroshield_uptime_seconds gauge
pyroshield_uptime_seconds {uptime:.1f}

# HELP pyroshield_model_loaded Model loaded status
# TYPE pyroshield_model_loaded gauge
pyroshield_model_loaded {1 if AGENT else 0}

# HELP pyroshield_q_table_size Number of states in Q-table
# TYPE pyroshield_q_table_size gauge
pyroshield_q_table_size {AGENT.get_q_table_size() if AGENT else 0}
"""
        return Response(content=metrics, media_type="text/plain")


def main():
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        print("Install FastAPI: pip install fastapi uvicorn")
        sys.exit(1)

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    print(f"\n🔥 PyroShield AI — Starting API on {host}:{port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   Health: http://localhost:{port}/health\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
