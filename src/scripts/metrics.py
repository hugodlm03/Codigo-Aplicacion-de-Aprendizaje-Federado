# src/utils/metrics.py
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Utilidades para manejar métricas de entrenamiento y evaluación en CSV
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

# Utilidades para medir el tiempo de ejecución de funciones
def measure_time(fn, *args, **kwargs) -> Tuple[Any, float]:
    """
    Ejecuta fn(*args, **kwargs) y devuelve (resultado, tiempo_en_segundos).
    """
    start = time.time()
    res = fn(*args, **kwargs)
    return res, time.time() - start
