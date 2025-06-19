import syft as sy
import numpy as np
import numpy.typing as npt
from typing import Union, TypeVar, Any, TypedDict, TypeVar
import pandas as pd
from syft.service.policy.policy import MixedInputPolicy
import pickle
import cloudpickle
import random
import copy
import concurrent.futures


from fed_rf_mk.utils import check_status_last_code_requests


DataFrame = TypeVar("pandas.DataFrame")
NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
Dataset = TypeVar("Dataset", bound=tuple[NDArrayFloat, NDArrayInt])

class DataParamsDict(TypedDict):
    target: str
    ignored_columns: list[Any]

# Añadimos las nuevas métricas de regresión al diccionario de parámetros del modelo.
class ModelParamsDict(TypedDict, total=False):
    model: bytes
    n_base_estimators: int
    n_incremental_estimators: int
    train_size: float
    test_size: float
    sample_size: int
    mae: float       # nueva
    rmse: float      # nueva


DataParams = TypeVar("DataParams", bound=DataParamsDict)
ModelParams = TypeVar("ModelParams", bound=ModelParamsDict)

from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import concurrent.futures
import pickle, cloudpickle, random

class FLClient:
    def __init__(self, partition_scheme: str):
        self.datasites = {}
        self.eval_datasites = {}
        self.weights = {}
        self.dataParams = {}
        self.modelParams = {}
        self.model_parameters_history = {}
        self.partition_scheme = partition_scheme
    
    def add_train_client(self, name, url, email, password, weight = None):
        try:
            client = sy.login(email=email, password=password, url=url)
            self.datasites[name] = client
            self.weights[name] = weight
            print(f"Successfully connected to {name} at {url}")
        except Exception as e:
            print(f"Failed to connect to {name} at {url}: {e}")

    def add_eval_client(self, name, url, email, password):
        try:
            client = sy.login(email=email, password=password, url=url)
            self.eval_datasites[name] = client
            print(f"Successfully connected to {name} at {url}")
        except Exception as e:
            print(f"Failed to connect to {name} at {url}: {e}")
    
    def check_status(self):
        """
        Checks and prints the status of all connected silos.
        """
        for name, client in self.datasites.items():
            try:
                datasets = client.datasets
                print(f"{name}:  Connected ({len(datasets)} datasets available)")
            except Exception as e:
                print(f"{name}: Connection failed ({e})")

    def set_data_params(self, data_params):
        self.dataParams = data_params
        return f"Data parameters set: {data_params}"
    
    def set_model_params(self, model_params):
        self.modelParams = model_params
        return f"Model parameters set: {model_params}"

    def get_data_params(self):
        return self.dataParams

    def get_model_params(self):
        return self.modelParams


    
    def send_request(self):

        if not self.datasites:
            print("No clients connected. Please add clients first.")
            return
        
        if self.dataParams is None or self.modelParams is None:
            print("DataParams and ModelParams must be set before sending the request.")
            return
        
        for site in self.datasites:
            data_asset = self.datasites[site].datasets[0].assets[0]
            client = self.datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client,
                    data=data_asset,
                    dataParams=dict,
                    modelParams=dict
                )
            )(ml_experiment)
            ml_training_project = sy.Project(
                name="ML Experiment for FL",
                description="""Test project to run a ML experiment""",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            project = ml_training_project.send()

        for site in self.eval_datasites:
            data_asset = self.eval_datasites[site].datasets[0].assets[0]
            client = self.eval_datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client,
                    data=data_asset,
                    dataParams=dict,
                    modelParams=dict
                )
            )(evaluate_global_model)
            ml_training_project = sy.Project(
                name="ML Evaluation for FL",
                description="""Test project to evaluate a ML model""",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            project = ml_training_project.send()

    def check_status_last_code_requests(self):
        """
        Display status message of last code request sent to each datasite.
        """
        check_status_last_code_requests(self.datasites)
        check_status_last_code_requests(self.eval_datasites)

    # ------------------------------------------------------------------
    # interno: dado el histórico de la época, calcula nuevos pesos
    # ------------------------------------------------------------------
    def _update_weights(self, metrics, alpha=0.5, eps=1e-6):
        """
        metrics = dict{client: mae}   (sólo de la época actual)
        Calcula wᵢ(t)=α·wᵢ(t-1)+(1-α)·(1/maeᵢ)  (EMA inversa del error)
        """
        inv = {c: 1.0 / (mae + eps) for c, mae in metrics.items()}

        # normalizamos el nuevo vector
        s = sum(inv.values())
        inv = {c: v / s for c, v in inv.items()}

        # EMA
        for c in self.weights:
            prev = self.weights[c]
            self.weights[c] = alpha * prev + (1 - alpha) * inv.get(c, 0.0)

        # renormalizar por si acaso
        s = sum(self.weights.values())
        for c in self.weights:
            self.weights[c] /= s


    def run_model(self, output_root: Path = Path("results")):
        
        """
            Ejecuta el entrenamiento federado y guarda:
            - results/{scheme}/{timestamp}/history.csv
            - results/{scheme}/{timestamp}/model.pkl
        """

        # Obtener parámetros del modelo y datos
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()

        # Para guardar métricas de cada epoch y cada cliente
        history = []
        modelParams_history = {}

        # Preparar carpeta de resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = output_root / self.partition_scheme / timestamp
        base_out.mkdir(parents=True, exist_ok=True)

        # Ajuste inicial de pesos
        num_clients = len(self.weights)
        none_count = sum(1 for w in self.weights.values() if w is None)

        if none_count == num_clients:  
            # **Case 1: All weights are None → Assign equal weights**
            equal_weight = 1 / num_clients
            self.weights = {k: equal_weight for k in self.weights}
            print(f"All weights were None. Assigning equal weight: {equal_weight}")

        elif none_count > 0:
            # **Case 2: Some weights are None → Distribute remaining weight proportionally**
            defined_weights_sum = sum(w for w in self.weights.values() if w is not None)
            undefined_weight_share = (1 - defined_weights_sum) / none_count

            self.weights = {
                k: (undefined_weight_share if w is None else w) for k, w in self.weights.items()
            }
            print(f"Some weights were None. Distributing remaining weight: {self.weights}")

        if "column_template" not in self.modelParams:
            tpl_path = Path("artefactos/template_cols.json")
            if not tpl_path.exists():
                raise FileNotFoundError(
                    "No encuentro artefactos/template_cols.json con la plantilla "
                    "global de columnas.  Ejecútalo en la fase de particiones."
                )
            import json
            self.modelParams["column_template"] = json.loads(tpl_path.read_text())
            print(f"Plantilla global cargado: "
                  f"{len(self.modelParams['column_template'])} columnas")

        # --- Bucle federado  ---
        for epoch in range(self.modelParams["fl_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.modelParams['fl_epochs']}")

            if epoch == 0:
                # Parallel dispatch to all silos
                print("Lanzando epoch inicial en paralelo…") 
                
                def remote_train(name, ds):
                        t0 = time.time()
                        mp = ds.code.ml_experiment(
                            data=ds.datasets[0].assets[0],
                            dataParams=dataParams,
                            modelParams={**modelParams, "model": None}
                        ).get_from(ds)
                        if "error" in mp:
                            print(f"🚨  {name} devolvió ERROR:\n      {mp['error']}")
                        mp["_duration"] = time.time() - t0
                        return name, mp
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.datasites)) as executor:
                    # map future → client name

                    futures = {
                        executor.submit(remote_train, name, ds): name
                        for name, ds in self.datasites.items()
                    }

                    for fut in concurrent.futures.as_completed(futures):
                        name, mp = fut.result()
                        dur  = mp.pop("_duration", 0.0)
                        n    = mp.get("sample_size")
                        mae  = mp.get("mae")
                        rmse = mp.get("rmse")
                        print(f"   ✔ {name}: n={n}, mae={mae:.3f}, rmse={rmse:.3f}, time={dur:.2f}s")
                        history.append({
                            "epoch":       1,
                            "client":      name,
                            "sample_size": n,
                            "mae":         mae,
                            "rmse":        rmse,
                            "duration_s":  dur
                        })
                        # guardar mp para el merge más tarde
                        modelParams_history[name] = mp

                # reajustar pesos para la próxima ronda
                maes_epoch = {rec["client"]: rec["mae"] for rec in history if rec["epoch"]==epoch+1}
                self._update_weights(maes_epoch)
                print("  Pesos tras EMA:", self.weights)
             

                # Merge their estimators
                print("Merging stimators para clients exitosos…")

                # Renormalize weights to only the successful clients
                successful = list(modelParams_history.keys())
                total_w = sum(self.weights[n] for n in successful)
                self.weights = {n: self.weights[n] / total_w for n in successful}
                print(f"Re‐normalized weights among successful clients: {self.weights}")


                # re-indexamos los DataFrames que vienen de los silos antes de fusionar
                all_estimators = []
                merged_model   = None
                for name, mp in modelParams_history.items():
                    clf  = pickle.loads(mp["model"])
                    take = int(clf.n_estimators * self.weights[name])
                    all_estimators.extend(random.sample(clf.estimators_, take))
                    merged_model = clf          # lo reutilizamos como recipiente


                # attach the merged ensemble
                merged_model.estimators_ = all_estimators
                modelParams["model"] = cloudpickle.dumps(merged_model)


            # epoch > 0
            else:
                print(f"Refinamiento epoch {epoch+1} con modelo merged…")
                for name, ds in self.datasites.items():
                    t0 = time.time()
                    try:
                        mp = ds.code.ml_experiment(
                                data        = ds.datasets[0].assets[0],
                                dataParams  = dataParams,
                                modelParams = modelParams # contiene column_template
                        ).get_from(ds)
                        success = True
                    except Exception as e:
                        mp = {"error": str(e)}
                        success = False
                    duration = time.time() - t0    

                    if not success or "error" in mp:
                        print(f"⚠️  {name} devolvió error: {mp['error']}")
                        history.append({
                            "epoch": epoch+1,
                            "client": name,
                            "sample_size": None,
                            "mae": None,
                            "rmse": None,
                            "duration_s": duration,
                            "error": mp["error"],
                        })
                        continue

                    n    = mp["sample_size"]
                    mae  = mp.get("mae")
                    rmse = mp.get("rmse")
                    print(f"   ✔ {name}: n={n}, mae={mae:.3f}, rmse={rmse:.3f}, time={duration:.2f}s")

                    history.append({
                        "epoch":      epoch+1,
                        "client":     name,
                        "sample_size": n,
                        "mae":        mae,
                        "rmse":       rmse,
                        "duration_s": duration
                    })



        # Fin de todas las epochs
        print("\n✅ Entrenamiento federado completado.")

        # Guardar historial en CSV
        hist_df = pd.DataFrame(history)
        hist_path = base_out / "history.csv"
        hist_df.to_csv(hist_path, index=False)
        print(f"History saved to {hist_path}")

        # Guardar modelo final
        model_path = base_out / "model.pkl"
        with open(model_path, "wb") as f:
            f.write(modelParams["model"])
        print(f"Final model saved to {model_path}")

        # Info de la ejecución
        info_txt = base_out  / "run_info.txt"
        with open(info_txt, "w") as f:
            f.write(f"Scheme: {self.partition_scheme}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"DataParams: {dataParams}\n")
            f.write(f"ModelParams: {modelParams}\n")
        print(f"Info    → {info_txt}")
          
        # Actualizar modelParams internos
        self.set_model_params(modelParams)
        
            
    def run_evaluate(self):
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()

        print(f"Number of evaluation sites: {len(self.eval_datasites)}")

        for name, datasite in self.eval_datasites.items():
            data_asset = datasite.datasets[0].assets[0]
            print(f"\nEvaluating model at {name}")

            # Send evaluation request
            model = datasite.code.evaluate_global_model(
                data=data_asset, dataParams=dataParams, modelParams=modelParams
            ).get_from(datasite)

            return model

def evaluate_global_model(data: DataFrame, dataParams: dict, modelParams: dict) -> dict:
    from sklearn.model_selection import train_test_split
    # Imports para regresion
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, matthews_corrcoef as mcc
    #from sklearn.metrics import precision_score, recall_score, f1_score
    import pickle
    import numpy as np
    import pandas as pd

    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        # Step 1: Prepare the data for training
        # Drop rows with missing values in Q1
        data = data.dropna(subset=[dataParams["target"]])
        y = data[dataParams["target"]]


        # Aseguramos que las columnas ignoradas no estén en X y que eliminamos el target
        X = data.drop(dataParams["ignored_columns"] + [dataParams["target"]], axis=1)

        # Replace inf/-inf with NaN, cast to float64, drop NaNs
        X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        # Step 2: Split the data into training and testing sets
        # _, X_test, _, y_test = train_test_split(X, y, test_size=modelParams["test_size"], stratify=y, random_state=42)
        # Evitamos stratify si la variable objetivo es continua, porque va dar error
        _, X_test, _, y_test = train_test_split(X, y, test_size=modelParams["test_size"], random_state=42)
        return X_test, y_test
    
        

    def evaluate(model, data: tuple[pd.DataFrame, pd.Series]) -> dict:
        X, y_true = data
        X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y_true = y_true[mask]

        y_pred = model.predict(X)
        # print("after predict")

        return {
            # Comentamos las métricas de clasificación para el caso de regresión
            #"mcc": mcc(y_true, y_pred),
            #"cm": confusion_matrix(y_true, y_pred),
            #"accuracy": accuracy_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            #"precision": precision_score(y_true, y_pred, average='weighted'),
            #"recall": recall_score(y_true, y_pred, average='weighted'),
            #"f1_score": f1_score(y_true, y_pred, average='weighted')
        }
    try:
        testing_data = preprocess(data)
        print(f"Testing data shape: {testing_data[0].shape}")
        model = modelParams["model"]
        clf = pickle.loads(model)
        print(f"Model estimators: {clf.n_estimators}")

        test_metrics = evaluate(clf, testing_data)
        print("CHEGOU AQUI")
    except Exception as e:
        print(f"Error: {e}")
        test_metrics = {"error": str(e)}

    return test_metrics
    

def ml_experiment(data: DataFrame, dataParams: dict, modelParams: dict) -> dict:
    # preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor # Es necesario importar RandomForestRegressor para el caso de regresión
    from sklearn.metrics import mean_absolute_error, mean_squared_error # importar métricas de regresión
    from sklearn.metrics import accuracy_score
    import cloudpickle
    import pickle
    import numpy as np
    import pandas as pd

    
    # --------- lista blanca de kwargs que acepta RandomForest ----------
    RF_KWARGS = [
            "max_depth", "max_features", "min_samples_leaf",
            "min_samples_split", "criterion", "bootstrap",
            "max_samples", "max_leaf_nodes"
    ]  

    
    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        #  Eliminar filas sin target
        data = data.dropna(subset=[dataParams["target"]])
        y = data[dataParams["target"]]

        X = data.drop(dataParams["ignored_columns"] + [dataParams["target"]], axis=1)

        # Evitamos stratify si la variable objetivo es continua, porque va dar error
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=modelParams["train_size"], random_state=42)

        return (X_train, y_train), (X_test, y_test)

    def train(model, training_data: tuple[pd.DataFrame, pd.Series]) -> RandomForestRegressor:
        X_train, y_train = training_data

        model.fit(X_train, y_train)
        return model
    
    def evaluate(model, data: tuple[pd.DataFrame, pd.Series]) -> dict:
        X, y_true = data
        y_pred = model.predict(X)
        return {
            "mae": mean_absolute_error(y_true, y_pred), # Nueva métrica de error absoluto medio
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)) # Nueva métrica de raíz del error cuadrático medio
        }
    
    try:
        training_data, test_data = preprocess(data)
        # Si ya hay modelo previo, lo desempacamos y añadimos estimators
        if modelParams["model"]:
            clf = pickle.loads(modelParams["model"])
            clf.n_estimators += modelParams["n_incremental_estimators"]
        else:
            # recogemos SOLO los hiper-parámetros presentes en modelParams
            rf_kwargs = {k: modelParams[k] for k in RF_KWARGS if k in modelParams}
            clf = RandomForestRegressor(
                n_estimators = modelParams["n_base_estimators"],
                warm_start   = True,
                random_state = 42,
                **rf_kwargs              # ← ¡aquí entran los nuevos parámetros!
            )
        # Entrenamos el modelo con los datos de entrenamiento
        clf = train(clf, training_data)

        # Evaluamos localmente y añadimos métricas al retorno
        metrics = evaluate(clf, test_data)
    except Exception as e:
        print(f"Error en ml_experiment: {e}")
        return {"error": str(e)}
    
    column_template = list(training_data[0].columns)
    # Preparamos el diccionario de retorno con el modelo y las métricas. 
    # Hemos añadido las metricas de regresión y el tamaño de la muestra respecto al codigo original.
    return {
        "model": cloudpickle.dumps(clf),
        "n_base_estimators":      modelParams["n_base_estimators"],
        "n_incremental_estimators": modelParams["n_incremental_estimators"],
        "train_size":             modelParams["train_size"],
        "test_size":              modelParams["test_size"],
        "sample_size":            len(training_data[0]),
        **metrics,
        "columns": column_template          
    }

def hello_world():
    print("FedLearning RF is installed!")