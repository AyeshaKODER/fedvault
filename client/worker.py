"""FedVault hospital node worker — local training and weight submission."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests
import torch

from config import (
    CENTRAL_SERVER_URL,
    CLIENT_CREDENTIALS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    LOCAL_EPOCHS,
    NODE_DATA_PATHS,
    NODE_IDS,
)
from ml.model import (
    apply_state_dict_to_model,
    get_resnet18,
    state_dict_to_base64,
    train_model,
)

# Gradient compression: keep only top-k% of weights by absolute magnitude
# Setting to 0.10 means only top 10% of weights are transmitted
COMPRESSION_RATIO = 0.10


def _apply_top_k_compression(
    state_dict: dict[str, torch.Tensor],
    k: float = COMPRESSION_RATIO,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply top-k sparsification to model weights before transmission.

    Only the top k% of weight values (by absolute magnitude) are kept.
    All other values are set to zero. This reduces communication cost
    while preserving the most impactful weight updates.

    Args:
        state_dict: Model state dictionary to compress
        k: Fraction of weights to keep (0.10 = top 10%)

    Returns:
        Compressed state dict and compression statistics
    """
    compressed = {}
    total_params = 0
    kept_params = 0

    for key, tensor in state_dict.items():
        if tensor.dtype in (torch.float32, torch.float16, torch.float64):
            flat = tensor.flatten()
            n_keep = max(1, int(len(flat) * k))

            # Find threshold: keep top-k by absolute magnitude
            threshold = torch.topk(flat.abs(), n_keep, largest=True).values.min()

            # Zero out weights below threshold
            mask = tensor.abs() >= threshold
            compressed[key] = tensor * mask.float()

            total_params += flat.numel()
            kept_params += mask.sum().item()
        else:
            # Keep non-float tensors (e.g., batch norm running stats) as-is
            compressed[key] = tensor.clone()
            total_params += tensor.numel()
            kept_params += tensor.numel()

    actual_ratio = kept_params / total_params if total_params > 0 else 1.0
    bytes_full = total_params * 4  # float32 = 4 bytes
    bytes_compressed = int(bytes_full * actual_ratio)
    bytes_saved = bytes_full - bytes_compressed

    stats = {
        "total_params": total_params,
        "kept_params": int(kept_params),
        "compression_ratio": round(actual_ratio, 4),
        "bytes_full": bytes_full,
        "bytes_compressed": bytes_compressed,
        "bytes_saved": bytes_saved,
        "savings_percent": round((1 - actual_ratio) * 100, 2),
    }

    return compressed, stats


class FedVaultClientWorker:
    """Edge hospital node that participates in federated learning rounds."""

    def __init__(self, node_id: str, server_url: str = CENTRAL_SERVER_URL) -> None:
        if node_id not in NODE_IDS:
            raise ValueError(f"Invalid node_id '{node_id}'. Expected one of: {NODE_IDS}")

        self.node_id = node_id
        self.server_url = server_url.rstrip("/")
        self.credentials = CLIENT_CREDENTIALS[node_id]
        self.data_path = NODE_DATA_PATHS[node_id] / "train"
        self.token: str | None = None
        self.device = torch.device("cpu")
        self.model = get_resnet18(pretrained=True).to(self.device)
        self.logs: list[dict[str, Any]] = []
        self.compression_stats: list[dict[str, Any]] = []

    def _record_log(self, message: str, level: str = "info", extra: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": self.node_id,
            "level": level,
            "message": message,
        }
        if extra:
            entry.update(extra)
        self.logs.append(entry)
        print(f"[{self.node_id}] {message}")

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = f"{self.server_url}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, HTTP_MAX_RETRIES + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_payload,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                self._record_log(
                    f"HTTP {method} {endpoint} failed (attempt {attempt}/{HTTP_MAX_RETRIES}): {exc}",
                    level="warning",
                )
                time.sleep(min(2 * attempt, 6))

        raise RuntimeError(f"Request failed after {HTTP_MAX_RETRIES} attempts: {last_error}")

    def authenticate(self) -> str:
        """Obtain a JWT access token from the central server."""
        try:
            response = self._request_with_retry(
                "POST",
                "/token",
                data={
                    "username": self.credentials["username"],
                    "password": self.credentials["password"],
                },
            )
            payload = response.json()
            self.token = payload["access_token"]
            self._record_log("Authentication successful")
            return self.token
        except Exception as exc:
            self._record_log(f"Authentication failed: {exc}", level="error")
            raise

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("Client is not authenticated. Call authenticate() first.")
        return {"Authorization": f"Bearer {self.token}"}

    def fetch_global_model(self) -> dict[str, Any]:
        """Download the latest global model weights from the central server."""
        try:
            response = self._request_with_retry(
                "GET",
                "/global-model",
                headers=self._auth_headers(),
            )
            payload = response.json()
            apply_state_dict_to_model(self.model, payload["weights"])
            self._record_log(
                f"Fetched global model for round {payload.get('round_number', 'unknown')}",
                extra={"round_number": payload.get("round_number")},
            )
            return payload
        except Exception as exc:
            self._record_log(f"Failed to fetch global model: {exc}", level="error")
            raise

    def run_local_training(self) -> dict[str, Any]:
        """Execute local PyTorch training on the node's private dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Local training data not found: {self.data_path}")

        try:
            self._record_log(f"Starting local training for {LOCAL_EPOCHS} epoch(s)")
            metrics = train_model(
                model=self.model,
                data_path=self.data_path,
                epochs=LOCAL_EPOCHS,
                device=self.device,
            )
            self._record_log(
                f"Local training complete — loss={metrics['train_loss']:.4f}, accuracy={metrics['accuracy']:.4f}",
                extra=metrics,
            )
            return metrics
        except Exception as exc:
            self._record_log(f"Local training failed: {exc}", level="error")
            raise

    def submit_update(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Upload compressed model weights and metrics to the central server.

        Applies top-k gradient compression before transmission to reduce
        communication overhead while preserving model quality.
        """
        try:
            # Apply gradient compression before sending
            compressed_state_dict, compression_stats = _apply_top_k_compression(
                self.model.state_dict(),
                k=COMPRESSION_RATIO,
            )

            self.compression_stats.append({
                "round": metrics.get("round"),
                "node_id": self.node_id,
                **compression_stats,
            })

            self._record_log(
                f"Gradient compression: kept {compression_stats['savings_percent']}% savings, "
                f"transmitted {compression_stats['bytes_compressed']:,} / {compression_stats['bytes_full']:,} bytes",
            )

            encoded_weights = state_dict_to_base64(compressed_state_dict)
            payload = {
                "node_id": self.node_id,
                "weights": encoded_weights,
                "train_loss": float(metrics["train_loss"]),
                "accuracy": float(metrics["accuracy"]),
                "num_samples": int(metrics["num_samples"]),
            }
            response = self._request_with_retry(
                "POST",
                "/submit-update",
                headers=self._auth_headers(),
                json_payload=payload,
            )
            result = response.json()
            result["compression_stats"] = compression_stats
            self._record_log(
                f"Submitted compressed update — aggregated={result.get('aggregated', False)}",
                extra=result,
            )
            return result
        except Exception as exc:
            self._record_log(f"Failed to submit model update: {exc}", level="error")
            raise

    def run_federated_round(self) -> dict[str, Any]:
        """Complete one full federated learning round for this node."""
        self.authenticate()
        global_payload = self.fetch_global_model()
        metrics = self.run_local_training()
        submission = self.submit_update(metrics)
        return {
            "node_id": self.node_id,
            "global_round": global_payload.get("round_number"),
            "metrics": metrics,
            "submission": submission,
            "compression_stats": self.compression_stats,
            "logs": self.logs,
        }


def run_node(node_id: str, server_url: str = CENTRAL_SERVER_URL) -> dict[str, Any]:
    """Run a single hospital node through one federated round."""
    worker = FedVaultClientWorker(node_id=node_id, server_url=server_url)
    return worker.run_federated_round()


def run_all_nodes(server_url: str = CENTRAL_SERVER_URL) -> list[dict[str, Any]]:
    """Run all configured hospital nodes sequentially."""
    results: list[dict[str, Any]] = []
    for node_id in NODE_IDS:
        try:
            result = run_node(node_id, server_url=server_url)
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "node_id": node_id,
                    "error": str(exc),
                    "logs": [],
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="FedVault hospital node federated learning worker.")
    parser.add_argument(
        "--node-id",
        choices=NODE_IDS,
        help="Run a specific hospital node. If omitted, all nodes are executed sequentially.",
    )
    parser.add_argument(
        "--server-url",
        default=CENTRAL_SERVER_URL,
        help=f"Central server base URL (default: {CENTRAL_SERVER_URL})",
    )
    args = parser.parse_args()

    try:
        if args.node_id:
            result = run_node(args.node_id, server_url=args.server_url)
            print(f"Node {args.node_id} completed round successfully.")
            print(result)
        else:
            results = run_all_nodes(server_url=args.server_url)
            print("All nodes completed.")
            print(results)
        return 0
    except Exception as exc:
        print(f"Worker execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())