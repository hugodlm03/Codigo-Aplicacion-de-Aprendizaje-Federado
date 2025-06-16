# xgboost_comprehensive/metrics.py
import csv
from pathlib import Path

def init_server_csv(path: Path):
    """Crea el CSV con cabecera si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "round", "rmse"])

def append_server_metric(path: Path, experiment: str, rnd: int, rmse: float):
    """Añade una línea al CSV con la métrica de una ronda."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([experiment, rnd, rmse])
