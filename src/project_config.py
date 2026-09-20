"""Rutas y configuración comunes para los notebooks del proyecto."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
PROCESSED_DIR = DATA_DIR / "procesado_clip"
CACHE_DIR = DATA_DIR / "cache_embeddings"


def taco_play_dir() -> Path:
    """Ruta de TACO Play configurable con la variable TACO_PLAY_DIR."""
    return Path(os.environ.get("TACO_PLAY_DIR", r"E:\TFM_datasets\taco_play_lerobot"))


def ensure_project_dirs() -> None:
    """Crea las carpetas de salida compartidas sin modificar datos de entrada."""
    for directory in (DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def experiment_dirs(experiment_name: str) -> dict[str, Path]:
    """Crea una carpeta aislada por experimento para evitar sobrescrituras."""
    ensure_project_dirs()
    paths = {
        "checkpoints": CHECKPOINTS_DIR / experiment_name,
        "results": RESULTS_DIR / experiment_name,
        "figures": FIGURES_DIR / experiment_name,
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths
