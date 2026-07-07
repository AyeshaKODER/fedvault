from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    CLASS_NAMES,
    DATA_ROOT,
    DATA_SPLITS,
    IMAGE_SIZE,
    NODE_DATA_PATHS,
    SAMPLES_PER_CLASS,
)

# Non-IID distribution per node
# Each value = fraction of PNEUMONIA samples (1 - value = fraction of NORMAL)
NODE_PNEUMONIA_RATIO = {
    "node_1": 0.70,  # 70% pneumonia — simulates high-risk hospital
    "node_2": 0.30,  # 30% pneumonia — simulates low-risk hospital
    "node_3": 0.50,  # 50% pneumonia — balanced hospital
}


def _generate_normal_pattern(size: int, seed: int) -> np.ndarray:
    """Create a synthetic 'normal' chest X-ray-like grayscale image."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.45, scale=0.08, size=(size, size)).astype(np.float32)
    base = np.clip(base, 0.0, 1.0)

    y_coords, x_coords = np.mgrid[0:size, 0:size]
    lung_mask = ((x_coords - size * 0.28) ** 2 + (y_coords - size * 0.5) ** 2 < (size * 0.22) ** 2) | (
        (x_coords - size * 0.72) ** 2 + (y_coords - size * 0.5) ** 2 < (size * 0.22) ** 2
    )
    base[lung_mask] *= 0.55

    rib_lines = np.sin(y_coords / 18.0 + rng.uniform(0, 2 * np.pi)) * 0.03
    base += rib_lines
    base += rng.normal(0, 0.015, size=(size, size))

    return np.clip(base, 0.0, 1.0)


def _generate_pneumonia_pattern(size: int, seed: int) -> np.ndarray:
    """Create a synthetic 'pneumonia' chest X-ray-like grayscale image with opacity."""
    rng = np.random.default_rng(seed)
    base = _generate_normal_pattern(size, seed + 1000)

    y_coords, x_coords = np.mgrid[0:size, 0:size]
    opacity_center_x = int(size * rng.uniform(0.35, 0.65))
    opacity_center_y = int(size * rng.uniform(0.35, 0.65))
    opacity_radius = int(size * rng.uniform(0.12, 0.22))

    opacity_mask = (x_coords - opacity_center_x) ** 2 + (y_coords - opacity_center_y) ** 2 < opacity_radius**2
    cloud = rng.normal(0.75, 0.12, size=(size, size)).astype(np.float32)
    cloud = np.clip(cloud, 0.0, 1.0)
    base[opacity_mask] = np.maximum(base[opacity_mask], cloud[opacity_mask] * 0.85)

    speckle = rng.random((size, size)) > 0.97
    base[speckle] = np.minimum(base[speckle] + 0.25, 1.0)

    return np.clip(base, 0.0, 1.0)


def _array_to_pil_image(array: np.ndarray) -> Image.Image:
    """Convert a float32 [0,1] grayscale array to an 8-bit PIL image."""
    uint8_array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_array, mode="L")


def generate_node_dataset(node_id: str, node_path: Path) -> dict[str, int]:
    """Generate train/test splits with Non-IID class distribution for one node.
    
    Non-IID means each hospital has a different ratio of pneumonia vs normal,
    simulating real-world data heterogeneity across medical institutions.
    """
    counts: dict[str, int] = {}
    seed_offset = {"node_1": 0, "node_2": 10000, "node_3": 20000}.get(node_id, 0)

    # Get pneumonia ratio for this node
    pneumonia_ratio = NODE_PNEUMONIA_RATIO.get(node_id, 0.50)
    total_samples = SAMPLES_PER_CLASS * 2  # total samples per split

    # Calculate actual samples per class based on ratio
    pneumonia_samples = int(total_samples * pneumonia_ratio)
    normal_samples = total_samples - pneumonia_samples

    class_sample_counts = {
        "pneumonia": pneumonia_samples,
        "normal": normal_samples,
    }

    print(f"[NON-IID] {node_id}: {normal_samples} normal / {pneumonia_samples} pneumonia ({pneumonia_ratio*100:.0f}% pneumonia)")

    for split_idx, split in enumerate(DATA_SPLITS):
        for class_name in CLASS_NAMES:
            class_dir = node_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            n_samples = class_sample_counts[class_name]
            class_idx = CLASS_NAMES.index(class_name)

            for sample_idx in range(n_samples):
                seed = seed_offset + split_idx * 1000 + class_idx * 100 + sample_idx
                if class_name == "normal":
                    array = _generate_normal_pattern(IMAGE_SIZE, seed)
                else:
                    array = _generate_pneumonia_pattern(IMAGE_SIZE, seed)

                image = _array_to_pil_image(array)
                filename = f"{class_name}_{sample_idx:03d}.png"
                image.save(class_dir / filename)

            key = f"{split}/{class_name}"
            counts[key] = n_samples

    return counts


def generate_all_node_data(force: bool = False) -> None:
    """Generate synthetic datasets for all configured hospital nodes."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for node_id, node_path in NODE_DATA_PATHS.items():
        marker = node_path / ".generated"
        if marker.exists() and not force:
            print(f"[SKIP] {node_id}: data already exists at {node_path}")
            continue

        print(f"[GEN]  {node_id}: generating Non-IID synthetic X-ray data...")
        counts = generate_node_dataset(node_id, node_path)
        marker.write_text("FedVault synthetic dataset marker\n", encoding="utf-8")
        total = sum(counts.values())
        print(f"[DONE] {node_id}: {total} images written to {node_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FedVault synthetic hospital node datasets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate datasets even if they already exist.",
    )
    args = parser.parse_args()

    try:
        generate_all_node_data(force=args.force)
        print("Data generation completed successfully.")
        return 0
    except Exception as exc:
        print(f"Data generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())