# src/scripts/metrics.py

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ———————————————— CSV INDIVIDUAL ————————————————

def init_metrics_csv(csv_path: Path) -> None:
    """
    Crea el fichero CSV con cabeceras si no existe.
    Columnas: config (JSON), round, rmse, train_time.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["config", "round", "rmse", "train_time"])


def append_round(
    csv_path: Path,
    config: Dict[str, Any],
    round_idx: int,
    rmse: float,
    train_time: float,
) -> None:
    """
    Añade una fila al CSV con las métricas de una ronda.
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            json.dumps(config, sort_keys=True),
            round_idx,
            rmse,
            f"{train_time:.3f}"
        ])


# ———————————————— CSV GLOBAL ————————————————

def init_server_csv(csv_path: Path, metrics: Optional[List[str]] = None) -> None:
    """
    Crea el fichero CSV de métricas globales con cabeceras según la lista 'metrics'.
    Columnas: experiment, round, <metrics...>
    """
    if metrics is None:
        metrics = ["rmse"]
    header = ["experiment", "round"] + metrics
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_server_metric(
    csv_path: Path,
    experiment: str,
    round_idx: int,
    **metric_values: float,       # ej. rmse=..., mae=..., r2=...
) -> None:
    """
    Añade una fila al CSV global con las métricas dadas.
    metric_values: clave=nombre de la métrica, valor=float.
    """
    # Asegurarse de que el header coincide con metric_values.keys()
    # (opcionalmente, podrías re-inicializar si cambian las métricas)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [experiment, round_idx] + [metric_values[k] for k in metric_values]
        writer.writerow(row)
