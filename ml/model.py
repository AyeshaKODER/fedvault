"""ResNet-18 model utilities for FedVault federated chest X-ray classification."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_CLASSES,
)


class ChestXrayDataset(Dataset):
    """PyTorch dataset for folder-structured chest X-ray images."""

    def __init__(self, root_dir: str | Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, class_idx))
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, class_idx))
            for image_path in sorted(class_dir.glob("*.jpeg")):
                self.samples.append((image_path, class_idx))

        if not self.samples:
            raise ValueError(f"No image samples found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {image_path}: {exc}") from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_default_transform() -> transforms.Compose:
    """Return the standard ImageNet-normalized transform for ResNet-18."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build a ResNet-18 model adapted for binary chest X-ray classification."""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _create_dataloader(data_path: str | Path, shuffle: bool) -> DataLoader:
    dataset = ChestXrayDataset(data_path, transform=get_default_transform())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_model(
    model: nn.Module,
    data_path: str | Path,
    epochs: int = LOCAL_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run a local training loop and return metrics."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.train()

    try:
        dataloader = _create_dataloader(data_path, shuffle=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to create training dataloader for {data_path}: {exc}") from exc

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_loss = 0.0
    total_batches = 0

    for _epoch in range(epochs):
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    eval_metrics = evaluate_model(model, data_path, device=device)

    return {
        "train_loss": round(avg_loss, 6),
        "accuracy": eval_metrics["accuracy"],
        "correct": eval_metrics["correct"],
        "total": eval_metrics["total"],
        "num_samples": len(dataloader.dataset),
    }


def evaluate_model(
    model: nn.Module,
    data_path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate model accuracy on a local dataset directory."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    try:
        dataloader = _create_dataloader(data_path, shuffle=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to create evaluation dataloader for {data_path}: {exc}") from exc

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)
            predictions = torch.argmax(outputs, dim=1)
            correct += int((predictions == batch_labels).sum().item())
            total += int(batch_labels.size(0))

    accuracy = correct / max(total, 1)
    return {
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
    }


def predict_single_image(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run inference on a single image and return class probabilities."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    transform = get_default_transform()
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess image {image_path}: {exc}") from exc

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_idx = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))},
        "input_tensor": tensor,
    }


def state_dict_to_base64(state_dict: dict[str, torch.Tensor]) -> str:
    """Serialize a PyTorch state dict to a base64-encoded string."""
    try:
        buffer = io.BytesIO()
        cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
        torch.save(cpu_state, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
    except Exception as exc:
        raise RuntimeError(f"Failed to serialize model weights: {exc}") from exc


def load_state_dict_from_base64(encoded_weights: str) -> dict[str, torch.Tensor]:
    """Deserialize a base64-encoded PyTorch state dict."""
    try:
        raw_bytes = base64.b64decode(encoded_weights.encode("utf-8"))
        buffer = io.BytesIO(raw_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        return state_dict
    except Exception as exc:
        raise RuntimeError(f"Failed to deserialize model weights: {exc}") from exc


def apply_state_dict_to_model(model: nn.Module, encoded_weights: str) -> nn.Module:
    """Load serialized weights into a model instance."""
    state_dict = load_state_dict_from_base64(encoded_weights)
    model.load_state_dict(state_dict)
    return model


def count_dataset_distribution(data_root: str | Path) -> dict[str, dict[str, int]]:
    """Count images per split and class for dashboard display."""
    data_root = Path(data_root)
    distribution: dict[str, dict[str, int]] = {}

    for split in ("train", "test"):
        split_counts: dict[str, int] = {}
        for class_name in CLASS_NAMES:
            class_dir = data_root / split / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
            else:
                count = 0
            split_counts[class_name] = count
        distribution[split] = split_counts

    return distribution
