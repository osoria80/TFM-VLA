"""Funciones compartidas por los notebooks de comprobación y entrenamiento."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def action_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE para x..rz y BCE para terminate y la pinza binaria."""
    if predictions.shape != targets.shape or predictions.shape[-1] != 8:
        raise ValueError("predictions y targets deben tener forma (batch, 8)")
    terminate_loss = nn.functional.binary_cross_entropy(
        predictions[:, 0], targets[:, 0]
    )
    physical_loss = nn.functional.mse_loss(predictions[:, 1:7], targets[:, 1:7])
    gripper_loss = nn.functional.binary_cross_entropy(
        predictions[:, 7], targets[:, 7]
    )
    return physical_loss + terminate_loss + gripper_loss


def calcular_perdida_accion(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Alias en español para mantener una interfaz cómoda en los notebooks."""
    return action_loss(predictions, targets)


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    """Ejecuta una época y devuelve su pérdida media y duración."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    start = time.perf_counter()

    for image_embeddings, text_embeddings, targets in loader:
        image_embeddings = image_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(image_embeddings=image_embeddings, text_embeddings=text_embeddings)
        loss = action_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_samples += len(targets)

    return {
        "loss": total_loss / max(total_samples, 1),
        "time_seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, device) -> Dict[str, float]:
    """Calcula la pérdida media de validación sin modificar los pesos."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    start = time.perf_counter()

    for image_embeddings, text_embeddings, targets in loader:
        predictions = model(
            image_embeddings=image_embeddings.to(device),
            text_embeddings=text_embeddings.to(device),
        )
        loss = action_loss(predictions, targets.to(device))
        total_loss += loss.item() * len(targets)
        total_samples += len(targets)

    return {
        "loss": total_loss / max(total_samples, 1),
        "time_seconds": time.perf_counter() - start,
    }


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch: int = 0,
    validation_loss: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Guarda el estado del modelo y la información necesaria para recuperarlo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "validation_loss": float(validation_loss),
        "config": config or {},
        "history": history or [],
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location=None) -> Dict[str, Any]:
    """Carga un checkpoint en el modelo y, opcionalmente, en su optimizador."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model(
    model,
    train_loader,
    validation_loader,
    optimizer,
    device,
    epochs: int = 50,
    patience: int = 7,
    checkpoint_path=None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Entrena con early stopping y conserva únicamente el mejor checkpoint."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs y patience deben ser positivos")

    history: List[Dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics = validate_one_epoch(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "time_train_seconds": train_metrics["time_seconds"],
            "time_validation_seconds": validation_metrics["time_seconds"],
            "time_epoch_seconds": train_metrics["time_seconds"] + validation_metrics["time_seconds"],
        }
        history.append(row)

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    validation_loss=best_validation_loss,
                    config=config,
                    history=history,
                )
        else:
            epochs_without_improvement += 1

        print(
            f"Época {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"validation_loss={row['validation_loss']:.6f} | "
            f"{row['time_epoch_seconds']:.1f}s"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping: {patience} épocas sin mejora.")
            break

    total_time = time.perf_counter() - training_start
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_executed": len(history),
        "total_training_time_seconds": total_time,
    }
