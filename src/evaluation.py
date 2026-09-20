"""Evaluación compartida de acciones y eficiencia para baseline y VLA."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from torch import nn

CONTINUOUS_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def action_loss_components(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Devuelve MSE físico y BCE de terminate y pinza por separado."""
    return {
        "actions_loss": nn.functional.mse_loss(predictions[:, 1:7], targets[:, 1:7]),
        "terminate_loss": nn.functional.binary_cross_entropy(predictions[:, 0], targets[:, 0]),
        "gripper_loss": nn.functional.binary_cross_entropy(predictions[:, 7], targets[:, 7]),
    }


def evaluate_actions(predictions, targets, minimum, scale) -> dict:
    """Métricas predictivas; no incluye tiempos ni parámetros."""
    predicted = np.asarray(predictions, dtype=np.float32)
    expected = np.asarray(targets, dtype=np.float32)
    if predicted.shape != expected.shape or predicted.ndim != 2 or predicted.shape[1] != 8:
        raise ValueError("Las acciones deben tener forma (muestras, 8)")
    error = predicted[:, 1:7] - expected[:, 1:7]
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    # El coseno se calcula exclusivamente con XYZ desnormalizado.
    xyz_predicted = predicted[:, 1:4] * scale[1:4] + minimum[1:4]
    xyz_expected = expected[:, 1:4] * scale[1:4] + minimum[1:4]
    divisor = np.linalg.norm(xyz_predicted, axis=1) * np.linalg.norm(xyz_expected, axis=1)
    cosine = np.divide((xyz_predicted * xyz_expected).sum(axis=1), divisor, out=np.zeros_like(divisor), where=divisor > 0)

    def classification(index):
        real = expected[:, index] >= 0.5
        estimated = predicted[:, index] >= 0.5
        tp = int(np.logical_and(real, estimated).sum())
        fp = int(np.logical_and(~real, estimated).sum())
        fn = int(np.logical_and(real, ~estimated).sum())
        tn = int(np.logical_and(~real, ~estimated).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"accuracy": (tp + tn) / len(real), "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "real_positives": int(real.sum()), "predicted_positives": int(estimated.sum()), "true_positives": tp}

    return {"continuous_normalized": {"mae": float(mae.mean()), "rmse": float(rmse.mean()),
            "mae_by_component": dict(zip(CONTINUOUS_NAMES, mae.tolist())),
            "rmse_by_component": dict(zip(CONTINUOUS_NAMES, rmse.tolist()))},
            "xyz_denormalized_cosine_similarity": float(cosine.mean()),
            "terminate": classification(0), "gripper": classification(7)}


def benchmark_inference(predict: Callable[[], None], samples: int, repeats: int = 5) -> dict:
    """Mide una ruta de inferencia concreta y devuelve ms/muestra y desviación."""
    durations = []
    with torch.no_grad():
        for _ in range(repeats):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.perf_counter(); predict()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000 / samples)
    return {"samples": samples, "repeats": repeats, "mean_ms_per_sample": float(np.mean(durations)), "std_ms_per_sample": float(np.std(durations))}
