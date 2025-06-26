"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import List, Optional, Tuple, Union

import torch
from flwr.common import NDArray
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.datasets import load_svmlight_file
from pathlib import Path
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


from hfedxgboost.dataset_preparation import (
    datafiles_fusion,
    download_data,
    modify_labels,
    train_test_split,
)

def load_and_divide_partitioned_dataset(
    dataset_name: str,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[
    List[DataLoader],                   # trainloaders
    List[Optional[DataLoader]],         # valloaders
    DataLoader,                         # testloader
    np.ndarray, np.ndarray,             # x_train, y_train
    np.ndarray, np.ndarray              # x_test, y_test
]:
    """
    Carga un dataset particionado en LIBSVM (silos + test) y devuelve:
      - trainloaders, valloaders, testloader
      - x_train, y_train, x_test, y_test

    Asume que tienes en 
      dataset/adidas_partitioned/ 
    los ficheros:
      train.libsvm, test.libsvm,
      silo_train_00-XX.libsvm, silo_val_00-XX.libsvm
    """
    # Ajusta esta ruta a donde estén tus .libsvm
    base_path = Path("hfedxgboost") / "dataset" / f"{dataset_name}_partitioned"


    # 1) CARGO test completo
    X_test, y_test = load_svmlight_file(str(base_path / "test.libsvm"))[:2]
    X_test = X_test.toarray().astype(np.float32)
    y_test = y_test.astype(np.float32)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    testloader = get_dataloader(test_ds, "test", batch_size)

    # ————————————————————————————
    # 2) CARGO SILOS uno a uno
    # ————————————————————————————
    trainloaders: List[DataLoader] = []
    valloaders: List[Optional[DataLoader]] = []

    for i in range(pool_size):
        # ficheros de entrenamiento y validación de cada silo
        X_tr, y_tr = load_svmlight_file(str(base_path / f"silo_train_{i:02d}.libsvm"))[:2]
        X_val, y_val = load_svmlight_file(str(base_path / f"silo_val_{i:02d}.libsvm"))[:2]

        X_tr = X_tr.toarray().astype(np.float32)
        X_val = X_val.toarray().astype(np.float32)
        y_tr = y_tr.astype(np.float32)
        y_val = y_val.astype(np.float32)

        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        trainloaders.append(get_dataloader(ds_tr, "train", batch_size))

        # si val_ratio==0 no necesitamos validación, pero aquí la devolvemos siempre
        valloaders.append(get_dataloader(ds_val, "val", batch_size))


    # 3) ARRANCO arrays completos de train (por si me hace falta que en verdad no)
    X_full, y_full = load_svmlight_file(str(base_path / "train.libsvm"))[:2]
    X_full = X_full.toarray().astype(np.float32)
    y_full = y_full.astype(np.float32)

    return trainloaders, valloaders, testloader, X_full, y_full, X_test, y_test

def load_single_dataset(
    task_type: str, dataset_name: str, train_ratio: Optional[float] = 0.75
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load a single dataset.

    Parameters
    ----------
        task_type (str): The type of task, either "BINARY" or "REG".
        dataset_name (str): The name of the dataset to load.
        train_ratio (float, optional): The ratio of training data to the total dataset.
        Default is 0.75.

    Returns
    -------
            x_train (numpy array): The training data features.
            y_train (numpy array): The training data labels.
            X_test (numpy array): The testing data features.
            y_test (numpy array): The testing data labels.
    """
    if dataset_name == "adidas":
        print("→ Cargando dataset Adidas desde archivos preprocesados")

        base_path = Path("hfedxgboost") / "dataset" / "adidas_partitioned"
        x_train, y_train = load_svmlight_file(str(base_path / "train.libsvm"))[:2]
        x_test, y_test = load_svmlight_file(str(base_path / "test.libsvm"))[:2]


        x_train = x_train.toarray().astype(np.float32)
        x_test = x_test.toarray().astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        print("Feature dimension of the dataset:", x_train.shape[1])
        print("Size of the trainset:", x_train.shape[0])
        print("Size of the testset:", x_test.shape[0])

        return x_train, y_train, x_test, y_test

    # 🔁 Si NO es Adidas, seguimos con la lógica original
    datafiles_paths = download_data(dataset_name)
    X, Y = datafiles_fusion(datafiles_paths)
    x_train, y_train, x_test, y_test = train_test_split(X, Y, train_ratio=train_ratio)
    if task_type.upper() == "BINARY":
        y_train, y_test = modify_labels(y_train, y_test)

        print(
            "First class ratio in train data",
            y_train[y_train == 0.0].size / x_train.shape[0],
        )
        print(
            "Second class ratio in train data",
            y_train[y_train != 0.0].size / x_train.shape[0],
        )
        print(
            "First class ratio in test data",
            y_test[y_test == 0.0].size / x_test.shape[0],
        )
        print(
            "Second class ratio in test data",
            y_test[y_test != 0.0].size / x_test.shape[0],
        )

    print("Feature dimension of the dataset:", x_train.shape[1])
    print("Size of the trainset:", x_train.shape[0])
    print("Size of the testset:", x_test.shape[0])

    return x_train, y_train, x_test, y_test


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    """Return a DataLoader object.

    Parameters
    ----------
        dataset (Dataset): The dataset object that contains the data.
        partition (str): The partition string that specifies the subset of data to use.
        batch_size (Union[int, str]): The batch size to use for loading data.
        It can be either an integer value or the string "whole".
        If "whole" is provided, the batch size will be set to the length of the dataset.

    Returns
    -------
        DataLoader: A DataLoader object that loads data from the dataset in batches.
    """
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


def divide_dataset_between_clients(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, Union[List[DataLoader], List[None]], DataLoader]:
    """Divide the data between clients with IID distribution.

    Parameters
    ----------
        trainset (Dataset): The  full training dataset.
        testset (Dataset): The full test dataset.
        pool_size (int): The number of partitions to create.
        batch_size (Union[int, str]): The size of the batches.
        val_ratio (float, optional): The ratio of validation data. Defaults to 0.0.

    Returns
    -------
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing
        the training loaders, validation loaders (or None), and test loader.
    """
    # Split training set into `num_clients` partitions to simulate
    # different local datasets
    trainset_length = len(trainset)
    lengths = [trainset_length // pool_size] * pool_size
    if sum(lengths) != trainset_length:
        lengths[-1] = trainset_length - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders: List[DataLoader] = []
    valloaders: Union[List[DataLoader], List[None]] = []
    for dataset in datasets:
        len_val = int(len(dataset) * val_ratio)
        len_train = len(dataset) - len_val
        ds_train, ds_val = random_split(
            dataset, [len_train, len_val], torch.Generator().manual_seed(0)
        )
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders.append(None)
    return trainloaders, valloaders, get_dataloader(testset, "test", batch_size)
