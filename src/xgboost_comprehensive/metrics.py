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

# src/xgboost_comprehensive/metrics.py
import csv
from pathlib import Path
from typing import List, Optional

def init_server_csv(path: Path, metrics: Optional[List[str]] = None) -> None:
    if metrics is None:
        metrics = ["rmse", "mae", "r2"]
    print(f"[DEBUG init_server_csv] path={path} metrics={metrics}")
    header = ["experiment", "round"] + metrics + ["eval_time_round"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_server_metric(
    path: Path,
    experiment: str,
    round_idx: int,
    eval_time_round: float,
    **metric_values: float,
) -> None:
    print(f"[DEBUG append_server_metric] path={path} experiment={experiment} "f"round={round_idx} values={metric_values}")
    # orden: experiment, round, <métricas…>, train_time_round, eval_time_round
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [experiment, round_idx] + [metric_values[k] for k in metric_values] + [f"{eval_time_round:.3f}"]
        writer.writerow(row)

