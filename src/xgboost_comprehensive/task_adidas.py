"""
task_adidas.py

Módulo de tareas para preparar datos del caso Adidas en un entorno federado con XGBoost.

Incluye:
- Carga del dataset
- Particionado por distintos esquemas (región, retailer+región, retailer+ciudad)
- División en entrenamiento y validación
- Conversión a objetos `xgb.DMatrix` para entrenamiento
"""

import xgboost as xgb
import pandas as pd
from typing import Tuple
from logging import INFO

from flwr.common.logger import log
from xgboost_comprehensive.data_loader import (
    load_clean_adidas_data,
    partition_by_region,
    partition_by_retailer_region,
    partition_by_retailer_city,
    preprocess_features, 
    get_feature_template,
)

from pathlib import Path

# Ruta por defecto al archivo Excel de Adidas
EXCEL_DEFAULT = Path(__file__).resolve().parent.parent.parent / "datos" / "Adidas US Sales Datasets.xlsx"

def load_data(
    partitioner_type: str,
    partition_id: int,
    centralised_eval_client: bool,
    test_fraction: float,
    seed: int,
    excel_path: str | Path | None = None,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    """
    Carga y prepara los datos de un nodo federado para entrenamiento y validación.

    Args:
        partitioner_type (str): Esquema de partición ('region', 'retailer_region', 'retailer_city').
        partition_id (int): Índice del nodo actual.
        centralised_eval_client (bool): Si True, el conjunto de validación será el dataset completo.
        test_fraction (float): Porcentaje reservado para validación local (si no es centralizado).
        seed (int): Semilla para particionado reproducible.
        excel_path (str | Path | None): Ruta al Excel. Si None, usa ruta por defecto.

    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, int, int]: 
            - train_dmatrix: Conjunto de entrenamiento
            - valid_dmatrix: Conjunto de validación
            - num_train: Número de ejemplos de entrenamiento
            - num_valid: Número de ejemplos de validación
    """
    if excel_path is None:
        excel_path = EXCEL_DEFAULT
    else:
        excel_path = Path(excel_path).resolve()

    

    # Cargar y limpiar los datos
    df = load_clean_adidas_data(excel_path)

    # Particionar el dataset según el esquema indicado
    if partitioner_type == "region":
        parts = partition_by_region(df)
    elif partitioner_type == "retailer_region":
        parts = partition_by_retailer_region(df)
    elif partitioner_type == "retailer_city":
        parts = partition_by_retailer_city(df)
    else:
        raise ValueError(f"Unknown partitioner_type: {partitioner_type}")

    # Seleccionar la partición correspondiente a este cliente
    keys = sorted(parts.keys())
    key = keys[partition_id % len(keys)]
    client_df = parts[key]

    # Dividir en entrenamiento y validación
    if centralised_eval_client:
        train_df = client_df
        valid_df = df  # validación contra todo el dataset (centralizada)
    else:
        train_df = client_df.sample(frac=1 - test_fraction, random_state=seed)
        valid_df = client_df.drop(train_df.index)

    num_train = len(train_df)
    num_valid = len(valid_df)

    log(INFO, f"Client partition '{key}': {num_train} train rows, {num_valid} valid rows")

    cols = get_feature_template(df) 

    X_tr, y_tr = preprocess_features(train_df)
    X_va, y_va = preprocess_features(valid_df)

    # re-index a plantilla (garantiza misma dimensión y orden)
    X_tr = X_tr.reindex(columns=cols, fill_value=0.0)
    X_va = X_va.reindex(columns=cols, fill_value=0.0)

    train_dmatrix = xgb.DMatrix(data=X_tr.values, label=y_tr.values, feature_names=cols)
    valid_dmatrix = xgb.DMatrix(data=X_va.values, label=y_va.values, feature_names=cols)
    
    return train_dmatrix, valid_dmatrix, num_train, num_valid


