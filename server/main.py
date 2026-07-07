from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from config import GLOBAL_ROUNDS, NODE_IDS
from ml.model import get_resnet18, load_state_dict_from_base64, state_dict_to_base64
from server.auth import (
    Token,
    User,
    login_for_access_token,
    require_admin,
    require_client,
)

ADVERSARIAL_THRESHOLD = 0.7

app = FastAPI(
    title="FedVault Central Server",
    description="Decentralized federated learning orchestrator for chest X-ray classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelUpdateRequest(BaseModel):
    node_id: str = Field(..., description="Hospital node identifier")
    weights: str = Field(..., description="Base64-encoded PyTorch state dict")
    train_loss: float = Field(..., ge=0.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    num_samples: int = Field(..., ge=1)


class GlobalModelResponse(BaseModel):
    round_number: int
    weights: str
    is_ready: bool


class RoundStatusResponse(BaseModel):
    current_round: int
    max_rounds: int
    round_active: bool
    pending_nodes: list[str]
    completed_nodes: list[str]
    aggregated: bool
    round_history: list[dict[str, Any]]
    client_logs: list[dict[str, Any]]
    global_metrics: dict[str, Any]
    adversarial_flags: dict[str, Any]


class StartRoundResponse(BaseModel):
    message: str
    round_number: int
    participating_nodes: list[str]


def _flatten_state_dict(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all tensors in a state dict into a single 1D vector for cosine similarity."""
    return torch.cat([tensor.detach().float().flatten() for tensor in state_dict.values()])


def _detect_adversarial_nodes(
    updates: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:

    if len(updates) < 2:
        return {
            node_id: {"similarity": 1.0, "flagged": False, "status": "safe"}
            for node_id in updates
        }

    flattened = {
        node_id: _flatten_state_dict(payload["state_dict"])
        for node_id, payload in updates.items()
    }

    all_vectors = torch.stack(list(flattened.values()))
    mean_vector = all_vectors.mean(dim=0)

    results = {}
    for node_id, vector in flattened.items():
        similarity = F.cosine_similarity(
            vector.unsqueeze(0),
            mean_vector.unsqueeze(0)
        ).item()

        flagged = similarity < ADVERSARIAL_THRESHOLD
        results[node_id] = {
            "similarity": round(similarity, 4),
            "flagged": flagged,
            "status": "SUSPICIOUS - possible data poisoning" if flagged else "safe",
            "threshold": ADVERSARIAL_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if flagged:
            print(f"[SECURITY ALERT] {node_id} flagged as adversarial! "
                  f"Cosine similarity: {similarity:.4f} < {ADVERSARIAL_THRESHOLD}")
        else:
            print(f"[SECURITY OK] {node_id} passed adversarial check. "
                  f"Cosine similarity: {similarity:.4f}")

    return results


class FedVaultState:

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model = get_resnet18(pretrained=True)
        self.current_round = 0
        self.max_rounds = GLOBAL_ROUNDS
        self.round_active = False
        self.pending_updates: dict[str, dict[str, Any]] = {}
        self.completed_nodes: list[str] = []
        self.round_history: list[dict[str, Any]] = []
        self.client_logs: list[dict[str, Any]] = []
        self.last_aggregation: dict[str, Any] | None = None
        self.adversarial_flags: dict[str, Any] = {}

    def get_global_weights_b64(self) -> str:
        with self._lock:
            return state_dict_to_base64(self.model.state_dict())

    def reset_for_new_round(self) -> int:
        with self._lock:
            if self.current_round >= self.max_rounds:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Maximum global rounds ({self.max_rounds}) already completed.",
                )
            if self.round_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A global round is already in progress.",
                )

            self.current_round += 1
            self.round_active = True
            self.pending_updates = {}
            self.completed_nodes = []
            return self.current_round

    def record_client_update(self, update: ModelUpdateRequest) -> dict[str, Any]:
        with self._lock:
            if not self.round_active:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No active global round. Admin must start a round first.",
                )

            if update.node_id not in NODE_IDS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown node_id: {update.node_id}",
                )

            if update.node_id in self.pending_updates:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Node {update.node_id} already submitted an update for round {self.current_round}.",
                )

            try:
                state_dict = load_state_dict_from_base64(update.weights)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid weight payload from {update.node_id}: {exc}",
                ) from exc

            self.pending_updates[update.node_id] = {
                "state_dict": state_dict,
                "train_loss": update.train_loss,
                "accuracy": update.accuracy,
                "num_samples": update.num_samples,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.completed_nodes.append(update.node_id)

            log_entry = {
                "round": self.current_round,
                "node_id": update.node_id,
                "train_loss": update.train_loss,
                "accuracy": update.accuracy,
                "num_samples": update.num_samples,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "submitted",
            }
            self.client_logs.append(log_entry)

            aggregation_result: dict[str, Any] = {
                "aggregated": False,
                "round": self.current_round,
                "completed_nodes": list(self.completed_nodes),
            }

            if len(self.pending_updates) >= len(NODE_IDS):
                aggregation_result = self._aggregate_updates_locked()

            return aggregation_result

    def _aggregate_updates_locked(self) -> dict[str, Any]:

        if not self.pending_updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No client updates available for aggregation.",
            )

        print(f"\n[SECURITY] Running adversarial detection for round {self.current_round}...")
        adversarial_results = _detect_adversarial_nodes(self.pending_updates)
        self.adversarial_flags = adversarial_results

        flagged_nodes = [
            node_id for node_id, result in adversarial_results.items()
            if result["flagged"]
        ]
        if flagged_nodes:
            print(f"[SECURITY WARNING] Flagged nodes: {flagged_nodes}")
        else:
            print(f"[SECURITY] All nodes passed adversarial check.")

        total_samples = sum(item["num_samples"] for item in self.pending_updates.values())
        if total_samples <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total sample count must be positive for FedAvg.",
            )

        aggregated_state: dict[str, torch.Tensor] | None = None

        try:
            for node_id, payload in self.pending_updates.items():
                weight = payload["num_samples"] / total_samples
                state_dict = payload["state_dict"]

                if aggregated_state is None:
                    aggregated_state = {
                        key: tensor.clone().detach().float() * weight
                        for key, tensor in state_dict.items()
                    }
                else:
                    for key, tensor in state_dict.items():
                        aggregated_state[key] += tensor.detach().float() * weight

            if aggregated_state is None:
                raise RuntimeError("Aggregation produced an empty state dict.")

            self.model.load_state_dict(aggregated_state)

            avg_loss = sum(
                item["train_loss"] * item["num_samples"]
                for item in self.pending_updates.values()
            ) / total_samples

            avg_accuracy = sum(
                item["accuracy"] * item["num_samples"]
                for item in self.pending_updates.values()
            ) / total_samples

            round_summary = {
                "round": self.current_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "avg_train_loss": round(avg_loss, 6),
                "avg_accuracy": round(avg_accuracy, 6),
                "participating_nodes": list(self.pending_updates.keys()),
                "adversarial_flags": adversarial_results,
                "flagged_nodes": flagged_nodes,
                "node_metrics": {
                    node_id: {
                        "train_loss": payload["train_loss"],
                        "accuracy": payload["accuracy"],
                        "num_samples": payload["num_samples"],
                        "cosine_similarity": adversarial_results.get(node_id, {}).get("similarity", 1.0),
                        "security_status": adversarial_results.get(node_id, {}).get("status", "safe"),
                    }
                    for node_id, payload in self.pending_updates.items()
                },
            }

            self.round_history.append(round_summary)
            self.last_aggregation = round_summary
            self.round_active = False

            for log_entry in self.client_logs:
                if log_entry.get("round") == self.current_round:
                    log_entry["status"] = "aggregated"

            return {
                "aggregated": True,
                "round": self.current_round,
                "avg_train_loss": round_summary["avg_train_loss"],
                "avg_accuracy": round_summary["avg_accuracy"],
                "participating_nodes": round_summary["participating_nodes"],
                "flagged_nodes": flagged_nodes,
                "adversarial_flags": adversarial_results,
            }
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"FedAvg aggregation failed: {exc}",
            ) from exc

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            pending_nodes = [node for node in NODE_IDS if node not in self.pending_updates]
            return {
                "current_round": self.current_round,
                "max_rounds": self.max_rounds,
                "round_active": self.round_active,
                "pending_nodes": pending_nodes,
                "completed_nodes": list(self.completed_nodes),
                "aggregated": not self.round_active and self.current_round > 0 and bool(self.last_aggregation),
                "round_history": copy.deepcopy(self.round_history),
                "client_logs": copy.deepcopy(self.client_logs),
                "global_metrics": copy.deepcopy(self.last_aggregation) if self.last_aggregation else {},
                "adversarial_flags": copy.deepcopy(self.adversarial_flags),
            }


fed_state = FedVaultState()


@app.on_event("startup")
async def startup_event() -> None:
    """Ensure the global model is initialized on server startup."""
    _ = fed_state.get_global_weights_b64()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "fedvault-central-server"}


@app.post("/token", response_model=Token)
async def issue_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """Issue JWT access tokens for admin and client roles."""
    return login_for_access_token(form_data)


@app.get("/global-model", response_model=GlobalModelResponse)
async def download_global_model(
    _client: User = Depends(require_client),
) -> GlobalModelResponse:
    """Allow authenticated hospital nodes to download current global weights."""
    try:
        weights = fed_state.get_global_weights_b64()
        return GlobalModelResponse(
            round_number=fed_state.current_round,
            weights=weights,
            is_ready=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve global model: {exc}",
        ) from exc


@app.get("/global-model/admin", response_model=GlobalModelResponse)
async def download_global_model_admin(
    _admin: User = Depends(require_admin),
) -> GlobalModelResponse:
    """Allow admin dashboard to fetch global weights for explainability."""
    try:
        weights = fed_state.get_global_weights_b64()
        return GlobalModelResponse(
            round_number=fed_state.current_round,
            weights=weights,
            is_ready=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve global model: {exc}",
        ) from exc


@app.post("/submit-update")
async def submit_update(
    update: ModelUpdateRequest,
    client: User = Depends(require_client),
) -> dict[str, Any]:
    """Receive localized model updates from hospital nodes."""
    if client.node_id != update.node_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token node_id does not match submitted node_id.",
        )

    try:
        result = fed_state.record_client_update(update)
        return {
            "message": "Update received",
            "node_id": update.node_id,
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process update: {exc}",
        ) from exc


@app.post("/start-round", response_model=StartRoundResponse)
async def start_global_round(
    _admin: User = Depends(require_admin),
) -> StartRoundResponse:
    """Admin endpoint to begin a new federated global round."""
    round_number = fed_state.reset_for_new_round()
    return StartRoundResponse(
        message=f"Global round {round_number} started",
        round_number=round_number,
        participating_nodes=NODE_IDS,
    )


@app.get("/status", response_model=RoundStatusResponse)
async def get_round_status(
    _admin: User = Depends(require_admin),
) -> RoundStatusResponse:
    """Return federated training status for the admin dashboard."""
    status_payload = fed_state.get_status()
    return RoundStatusResponse(**status_payload)


@app.get("/status/public")
async def get_public_status() -> dict[str, Any]:
    """Lightweight public status endpoint for dashboard polling."""
    status_payload = fed_state.get_status()
    return {
        "current_round": status_payload["current_round"],
        "max_rounds": status_payload["max_rounds"],
        "round_active": status_payload["round_active"],
        "completed_nodes": status_payload["completed_nodes"],
        "pending_nodes": status_payload["pending_nodes"],
        "round_history": status_payload["round_history"],
        "client_logs": status_payload["client_logs"],
        "adversarial_flags": status_payload["adversarial_flags"],
    }