"""Grad-CAM explainability for ResNet-18 chest X-ray classification."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet-18 layer4."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.layer4

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        """Remove registered hooks."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Compute a Grad-CAM heatmap for the given input tensor."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[:, target_class]
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

            gradients = self.gradients
            activations = self.activations

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=False)
            cam = torch.relu(cam)

            cam_np = cam.squeeze(0).cpu().numpy()
            cam_np = self._normalize(cam_np)
            return cam_np
        except Exception as exc:
            raise RuntimeError(f"Grad-CAM computation failed: {exc}") from exc

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        minimum = float(array.min())
        maximum = float(array.max())
        if maximum - minimum < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        normalized = (array - minimum) / (maximum - minimum)
        return normalized.astype(np.float32)

    def overlay_heatmap(
        self,
        input_tensor: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Overlay the Grad-CAM heatmap on the original input image."""
        try:
            original = self._tensor_to_display_image(input_tensor)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

            overlay = cv2.addWeighted(original, 1.0 - alpha, heatmap, alpha, 0)
            return overlay
        except Exception as exc:
            raise RuntimeError(f"Grad-CAM overlay generation failed: {exc}") from exc

    @staticmethod
    def _tensor_to_display_image(input_tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized 3-channel tensor back to an RGB uint8 image."""
        tensor = input_tensor.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0.0, 1.0)
        image = (denormalized.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return image

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        alpha: float = 0.45,
    ) -> dict[str, np.ndarray]:
        """Generate CAM heatmap, original display image, and overlay."""
        cam = self.generate_cam(input_tensor, target_class=target_class)
        original = self._tensor_to_display_image(input_tensor)
        overlay = self.overlay_heatmap(input_tensor, cam, alpha=alpha)
        return {
            "cam": cam,
            "original": original,
            "overlay": overlay,
        }


def run_gradcam_explanation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> dict[str, np.ndarray]:
    """Convenience wrapper that manages Grad-CAM hook lifecycle."""
    with GradCAM(model) as gradcam:
        return gradcam.explain(input_tensor, target_class=target_class)
