"""
Este módulo implementa un cliente personalizado para entrenar modelos de regresión con XGBoost
en un entorno de aprendizaje federado utilizando Flower.

El cliente se conecta a un servidor federado, recibe un modelo global (o entrena desde cero si es
la primera ronda), realiza entrenamiento local sobre una partición de datos, calcula métricas como
el RMSE en validación y devuelve el modelo actualizado.

La arquitectura sigue el patrón de `ClientApp` de Flower, y permite controlar el número de rondas
locales, tipo de partición, escalado del learning rate, etc. Soporta estrategias de agregación
como "bagging" y entrenamiento cíclico.

────────────────────────────
Para ejecutar el cliente:

1. Abre una terminal y navega al directorio raíz del proyecto:

   "C:\\Users\\PC\\Desktop\\s.o.e\\Estudios\\U-4\\Segundo Cuatri\\TFG\\Codigo-Aplicación de Aprendizaje Federado\\src"

2. Activa el entorno virtual:

   .\\.venv\\Scripts\\Activate

3. Instala las dependencias del proyecto (modo editable):

   python -m pip install -e src\\xgboost_comprehensive

4. Lanza el cliente con Flower:

   flwr run .

────────────────────────────
Créditos y base del proyecto:

Este código parte y adapta gran parte de la lógica del ejemplo oficial:

> **Comprehensive Flower + XGBoost Example**  
> *Flower: A Friendly Federated Learning Framework*  
> Repositorio original: https://github.com/adap/flower  
> Más información: https://flower.ai

@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}

"""
# === Imports ===
import warnings
from pathlib import Path

# Ignorar ciertos warnings de usuario de XGBoost
warnings.filterwarnings("ignore", category=UserWarning)

# Librerías externas
import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context

# Módulos propios
from xgboost_comprehensive.task_adidas import load_data
from xgboost_comprehensive.task import replace_keys


class XgbClient(Client):
    """
    Cliente Flower personalizado para entrenamiento federado usando XGBoost.

    Atributos:
        train_dmatrix (xgb.DMatrix): Datos de entrenamiento.
        valid_dmatrix (xgb.DMatrix): Datos de validación.
        num_train (int): Número de muestras de entrenamiento.
        num_val (int): Número de muestras de validación.
        num_local_round (int): Número de rondas locales por federated round.
        params (dict): Parámetros base para XGBoost.
        train_method (str): Método de entrenamiento.
    """
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round


        # Extiende los parámetros con los específicos para regresión
        self.params = {
            **params,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
        self.train_method = train_method


    def _local_boost(self, bst_input):

        """
        Realiza el entrenamiento local de XGBoost.
        Args:
            bst_input (xgb.Booster): Modelo XGBoost inicial o global.
        Returns:
            xgb.Booster: Modelo XGBoost actualizado tras el entrenamiento local.

        """
        # Entrenamos localmente durante num_local_round rondas
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

         # Según el método, devolvemos:
        # - "bagging": solo las últimas rondas entrenadas (para agregación en servidor)
        #- cyclic: el modelo completo
        if self.train_method == "bagging":
            start = bst_input.num_boosted_rounds() - self.num_local_round
            end = bst_input.num_boosted_rounds()
            bst = bst_input[start:end]
        else:
            bst = bst_input
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        """
        Entrena el modelo localmente y devuelve el modelo actualizado.
        Args:
            ins (FitIns): Información de entrenamiento, incluyendo el modelo global.
        Returns:
            FitRes: Resultado del entrenamiento con el modelo actualizado.
        """
        # Cargamos el modelo global si existe
        global_round = int(ins.config["global_round"])


        if global_round == 1:
            # Primera ronda: entrenamiento local desde cero
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "rmse")],  # Validación con RMSE
            )
        else:
            # Rondas posteriores: cargamos el modelo global y reforzamos localmente
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])
            bst.load_model(global_model)
            # Reforzamos el modelo global con entrenamiento local
            bst = self._local_boost(bst)

        # Calcular RMSE en validación local
        preds = bst.predict(self.valid_dmatrix)
        true_labels = self.valid_dmatrix.get_label()
        rmse = float(((preds - true_labels) ** 2).mean() ** 0.5)

        # Serializamos el modelo en JSON (para compatibilidad con tu flujo)
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Devolvemos el modelo y número de ejemplos usados
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={"rmse_val": rmse},
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evalúa el modelo global recibido usando el conjunto de validación local.

        Args:
            ins (EvaluateIns): Instrucciones de evaluación del servidor.

        Returns:
            EvaluateRes: Resultado con la pérdida (RMSE) y métricas adicionales.
        """
        # Cargar modelo global
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Evaluar en la ronda actual (última iteración del booster)
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "rmse")],
            iteration=bst.num_boosted_rounds() - 1,
        )

        # Formato del resultado: "validation-rmse:<valor>"
        rmse = float(eval_results.split("\t")[1].split(":")[1])

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=rmse,
            num_examples=self.num_val,
            metrics={"rmse": rmse},
        )


def client_fn(context: Context) -> XgbClient:
    """Cliente federado personalizado que se configura mediante context.run_config."""

    # Parámetros de partición del contexto del nodo
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Leer configuración general
    cfg = context.run_config

    # Parámetros generales (con valores por defecto si no están en el config)
    num_local_round = int(cfg.get("local-epochs", 1))
    train_method = cfg.get("strategy", "bagging")
    partitioner_raw = cfg.get("partitioner", "region")
    partitioner_type = str(partitioner_raw)
    seed = int(cfg.get("seed", 42))
    test_fraction = float(cfg.get("test-fraction", 0.2))
    centralised_eval_client = bool(cfg.get("centralised-eval-client", False))
    # Leer hiperparámetros XGBoost
    raw_params = cfg.get("params", {})
    params = raw_params if isinstance(raw_params, dict) else {}
    
    # Convertir ruta en Path
    # Para obtener la raíz del proyecto a partir de src/xgboost_comprehensive
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    excel_path_raw = cfg.get("excel-path", None)
    if excel_path_raw is not None:
        excel_path = Path(str(excel_path_raw))
        if not excel_path.is_absolute():
            excel_path = PROJECT_ROOT / excel_path
    else:
        # Por defecto: datos/Adidas US Sales Datasets.xlsx (en la raíz)
        excel_path = PROJECT_ROOT / "datos" / "Adidas US Sales Datasets.xlsx"
        # Cargar datos particionados localmente
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partitioner_type=partitioner_type,
        partition_id=partition_id,
        centralised_eval_client=centralised_eval_client,
        test_fraction=test_fraction,
        seed=seed,
        excel_path=excel_path,
    )

    # Escalar el learning rate si aplica (por partición)
    if cfg.get("scaled-lr", False) and params.get("eta") is not None:
        params["eta"] = float(params["eta"]) / num_partitions

    # Instancia del cliente federado
    return XgbClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=num_local_round,
        params=params,
        train_method=train_method,
    )

# Registro de la aplicación cliente con Flower
app = ClientApp(client_fn=client_fn)