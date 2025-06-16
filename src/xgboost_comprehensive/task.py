"""
xgboost_comprehensive: Aplicación federada con XGBoost usando Flower y flwr_datasets.

Este módulo permite cargar y particionar un dataset público (como Higgs) en múltiples
clientes para simular un entorno federado. Utiliza particionadores configurables y 
convierte los datos a formato `xgb.DMatrix` para su uso con XGBoost.

Dependencias:
- flwr-datasets
- datasets (HuggingFace)
- xgboost
"""

from logging import INFO

import xgboost as xgb
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    ExponentialPartitioner,
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
)

from datasets import DatasetDict, concatenate_datasets
from flwr.common import log

# Mapeo entre tipo de partición y su clase correspondiente
CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

# Cache de FederatedDataset para evitar re-carga
fds = None  # Cache FederatedDataset


def train_test_split(partition, test_fraction, seed):
    """
    Divide un subconjunto en entrenamiento y validación.

    Args:
        partition: subconjunto (Dataset) de HuggingFace.
        test_fraction (float): Porcentaje destinado a validación.
        seed (int): Semilla para reproducibilidad.

    Returns:
        Tuple[Dataset, Dataset, int, int]: train, valid, num_train, num_valid
    """
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """
    Transforma un dataset HuggingFace a DMatrix de XGBoost.

    Args:
        data: Dataset HuggingFace con claves "inputs" y "label".

    Returns:
        xgb.DMatrix
    """
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def instantiate_fds(partitioner_type, num_partitions):
    """
    Inicializa el objeto FederatedDataset con el particionador deseado.

    Args:
        partitioner_type (str): Clave del tipo de particionador.
        num_partitions (int): Número total de clientes.

    Returns:
        FederatedDataset
    """
    global fds
    if fds is None:
        partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
            num_partitions=num_partitions
        )
        fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
            preprocessor=resplit,
        )
    return fds


def load_data(
    partitioner_type,
    partition_id,
    num_partitions,
    centralised_eval_client,
    test_fraction,
    seed,
):
    """Load partition data."""
    fds_ = instantiate_fds(partitioner_type, num_partitions)
    partition = fds_.load_partition(partition_id)
    partition.set_format("numpy")

    if centralised_eval_client:
        train_data = partition
        num_train = train_data.shape[0]

        # Use centralised test set for evaluation
        valid_data = fds_.load_split("test")
        valid_data.set_format("numpy")
        num_val = valid_data.shape[0]
    else:
        # Train/test splitting
        train_data, valid_data, num_train, num_val = train_test_split(
            partition, test_fraction=test_fraction, seed=seed
        )

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def resplit(dataset: DatasetDict) -> DatasetDict:
    """Increase the quantity of centralised test samples from 500K to 1M."""
    return DatasetDict(
        {
            "train": dataset["train"].select(
                range(0, dataset["train"].num_rows - 500_000)
            ),
            "test": concatenate_datasets(
                [
                    dataset["train"].select(
                        range(
                            dataset["train"].num_rows - 500_000,
                            dataset["train"].num_rows,
                        )
                    ),
                    dataset["test"],
                ]
            ),
        }
    )
