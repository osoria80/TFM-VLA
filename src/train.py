"""Funciones compartidas por los notebooks de comprobación y entrenamiento."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from tqdm.auto import tqdm

from .evaluation import action_loss_components


def action_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE para x..rz y BCE para terminate y la pinza binaria."""
    if predictions.shape != targets.shape or predictions.shape[-1] != 8:
        raise ValueError("predictions y targets deben tener forma (batch, 8)")
    return sum(action_loss_components(predictions, targets).values())


def calcular_perdida_accion(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Alias en español para mantener una interfaz cómoda en los notebooks."""
    return action_loss(predictions, targets)


def train_one_epoch(model, loader, optimizer, device, progress_description: str = "Entrenamiento") -> Dict[str, float]:
    """Ejecuta una época y devuelve su pérdida media y duración."""
    model.train()
    total_loss = total_actions = total_terminate = total_gripper = 0.0
    total_samples = 0
    start = time.perf_counter()

    progress = tqdm(loader, desc=progress_description, unit="lote", leave=False, mininterval=1.0)
    for batch_index, (static_embeddings, gripper_embeddings, text_embeddings, targets) in enumerate(progress, start=1):
        static_embeddings = static_embeddings.to(device)
        gripper_embeddings = gripper_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            static_embeddings=static_embeddings,
            gripper_embeddings=gripper_embeddings,
            text_embeddings=text_embeddings,
        )
        components = action_loss_components(predictions, targets)
        loss = sum(components.values())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_actions += components["actions_loss"].item() * len(targets)
        total_terminate += components["terminate_loss"].item() * len(targets)
        total_gripper += components["gripper_loss"].item() * len(targets)
        total_samples += len(targets)
        if batch_index % 50 == 0 or batch_index == len(loader):
            progress.set_postfix(loss=f"{total_loss / total_samples:.6f}")

    return {
        "loss": total_loss / max(total_samples, 1),
        "actions_loss": total_actions / max(total_samples, 1),
        "terminate_loss": total_terminate / max(total_samples, 1),
        "gripper_loss": total_gripper / max(total_samples, 1),
        "time_seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, device, progress_description: str = "Validación") -> Dict[str, float]:
    """Calcula la pérdida media de validación sin modificar los pesos."""
    model.eval()
    total_loss = total_actions = total_terminate = total_gripper = 0.0
    total_samples = 0
    start = time.perf_counter()

    progress = tqdm(loader, desc=progress_description, unit="lote", leave=False, mininterval=1.0)
    for batch_index, (static_embeddings, gripper_embeddings, text_embeddings, targets) in enumerate(progress, start=1):
        predictions = model(
            static_embeddings=static_embeddings.to(device),
            gripper_embeddings=gripper_embeddings.to(device),
            text_embeddings=text_embeddings.to(device),
        )
        components = action_loss_components(predictions, targets.to(device))
        loss = sum(components.values())
        total_loss += loss.item() * len(targets)
        total_actions += components["actions_loss"].item() * len(targets)
        total_terminate += components["terminate_loss"].item() * len(targets)
        total_gripper += components["gripper_loss"].item() * len(targets)
        total_samples += len(targets)
        if batch_index % 50 == 0 or batch_index == len(loader):
            progress.set_postfix(loss=f"{total_loss / total_samples:.6f}")

    return {
        "loss": total_loss / max(total_samples, 1),
        "actions_loss": total_actions / max(total_samples, 1),
        "terminate_loss": total_terminate / max(total_samples, 1),
        "gripper_loss": total_gripper / max(total_samples, 1),
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
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            progress_description=f"Entrenamiento {epoch}/{epochs}",
        )
        validation_metrics = validate_one_epoch(
            model, validation_loader, device,
            progress_description=f"Validación {epoch}/{epochs}",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_actions_loss": train_metrics["actions_loss"],
            "train_terminate_loss": train_metrics["terminate_loss"],
            "train_gripper_loss": train_metrics["gripper_loss"],
            "validation_actions_loss": validation_metrics["actions_loss"],
            "validation_terminate_loss": validation_metrics["terminate_loss"],
            "validation_gripper_loss": validation_metrics["gripper_loss"],
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
