import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

NODE_DATA_PATHS = {
    "node_1": DATA_ROOT / "node_1",
    "node_2": DATA_ROOT / "node_2",
    "node_3": DATA_ROOT / "node_3",
}

NODE_IDS = list(NODE_DATA_PATHS.keys())

CENTRAL_SERVER_HOST = "127.0.0.1"
CENTRAL_SERVER_PORT = 8000
CENTRAL_SERVER_URL = f"http://{CENTRAL_SERVER_HOST}:{CENTRAL_SERVER_PORT}"

JWT_SECRET_KEY = "fedvault-research-prototype-secret-key-2026"
JWT_ALGORITHM = "HS256"
JWT_TOKEN_EXPIRE_MINUTES = 120

# Pre-configured credentials (research prototype only — not for production)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

CLIENT_CREDENTIALS = {
    "node_1": {"username": "client_node_1", "password": "node1pass"},
    "node_2": {"username": "client_node_2", "password": "node2pass"},
    "node_3": {"username": "client_node_3", "password": "node3pass"},
}

LEARNING_RATE = 0.001
BATCH_SIZE = 4
LOCAL_EPOCHS = 2
GLOBAL_ROUNDS = 10
NUM_CLASSES = 2
CLASS_NAMES = ["normal", "pneumonia"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SAMPLES_PER_CLASS = 10
DATA_SPLITS = ["train", "test"]

ACCENT_COLOR = "#00f0ff"
DARK_BG = "#0e1117"
DARK_SURFACE = "#1a1f2e"

HTTP_TIMEOUT_SECONDS = 60
HTTP_MAX_RETRIES = 3
