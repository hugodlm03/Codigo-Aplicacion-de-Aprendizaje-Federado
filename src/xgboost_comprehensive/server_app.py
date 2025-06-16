"""
xgboost_comprehensive: Aplicación de servidor para federated learning con Flower y XGBoost.

Este módulo define la lógica de servidor en un entorno de Aprendizaje Federado con XGBoost.
Permite lanzar una instancia de Flower ServerApp que coordina a múltiples nodos (clientes)
utilizando estrategias como FedXgbBagging o FedXgbCyclic. Incluye opciones de evaluación
centralizada, agregación de métricas, y configuración por ronda.

────────────────────────────
¿Cómo ejecutar el servidor?

1. Activa el entorno virtual:
   .\\.venv\\Scripts\\Activate

2. Instala las dependencias si aún no lo has hecho:
   pip install -e .

3. Lanza el servidor con Flower desde esta carpeta:
   flwr run .

────────────────────────────
Funcionalidades incluidas:
- Estrategias personalizadas: `FedXgbBagging`, `FedXgbCyclic`.
- Criterios de selección de clientes personalizados.
- Evaluación centralizada opcional con RMSE.
- Configuración dinámica por ronda (`on_fit_config_fn` y `on_evaluate_config_fn`).
- Agregación de métricas federadas (RMSE ponderado).

────────────────────────────
Créditos y base del proyecto:

Este código se inspira y adapta del ejemplo oficial de:

> **Comprehensive Flower + XGBoost Example**  
> Proyecto: [Flower - A Friendly Federated Learning Framework](https://flower.ai)  
> Repositorio original: https://github.com/adap/flower  

@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
"""


# === Librerías estándar ===
from logging import INFO
from typing import Dict, List, Optional
from pathlib import Path
# === Librerías externas ===
import xgboost as xgb
from datasets import load_dataset  # (Actualmente no usado, se puede eliminar si no se usa)

# === Flower: módulos comunes ===
from flwr.common import Context, Parameters, Scalar
from flwr.common.config import unflatten_dict
from flwr.common.logger import log

# === Flower: servidor ===
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

# === Módulos propios del proyecto ===
from xgboost_comprehensive.task import replace_keys, transform_dataset_to_dmatrix
from xgboost_comprehensive.data_loader import load_clean_adidas_data



class CyclicClientManager(SimpleClientManager):
    """
        ClientManager personalizado que selecciona todos los clientes disponibles
        en cada ronda de entrenamiento (comportamiento cíclico).

        A diferencia del muestreo aleatorio tradicional, este gestor devuelve todos
        los clientes conectados que cumplan con el criterio (si se define alguno).

        Ideal para estrategias tipo `FedXgbCyclic`.

    """

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """
        Selecciona un subconjunto (o todos) de clientes disponibles para la ronda actual.

        Args:
            num_clients (int): Número de clientes que solicita la estrategia.
            min_num_clients (Optional[int]): Número mínimo de clientes necesarios.
            criterion (Optional[Criterion]): Criterio para filtrar clientes válidos.

        Returns:
            List[ClientProxy]: Lista de proxies de clientes seleccionados.

        """

        # Esperar hasta que haya suficientes clientes conectados
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Filtrar clientes conectados según el criterio (si se proporciona)
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        # Si no hay suficientes clientes disponibles, registrar fallo
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Devolver todos los clientes disponibles (modo cíclico)
        return [self.clients[cid] for cid in available_cids]



def get_evaluate_fn(test_dmatrix: xgb.DMatrix, params: Dict[str, Scalar]):
    """
        Genera una función de evaluación centralizada basada en RMSE para el servidor.

        Esta función se ejecuta al final de cada ronda en el servidor (si se ha activado
        la evaluación centralizada), y evalúa el modelo global en un conjunto de test común.

        Args:
            test_dmatrix (xgb.DMatrix): Conjunto de datos de prueba para evaluación global.
            params (Dict[str, Scalar]): Parámetros base del modelo XGBoost.

        Returns:
            Callable: Función de evaluación compatible con Flower (recibe ronda, modelo, config).
    """

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Optional[tuple[float, Dict[str, Scalar]]]:
        # Saltar evaluación en la ronda 0
        if server_round == 0:
            return 0.0, {}
        # Cargar modelo recibido desde los parámetros
        bst = xgb.Booster(params=params)
        for tensor in parameters.tensors:
            bst.load_model(bytearray(tensor))
        # Evaluar modelo sobre el conjunto de prueba
        ev = bst.eval_set(
            evals=[(test_dmatrix, "rmse")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        # Formato esperado: "validation-rmse:<valor>"
        rmse = float(ev.split("\t")[1].split(":")[1])
        # Registrar resultado en un archivo de log local
        with open("./centralised_eval.txt", "a", encoding="utf-8") as fp:
            fp.write(f"Round:{server_round},rmse:{rmse}\n")
        return rmse, {"rmse": rmse}

    return evaluate_fn


def evaluate_metrics_aggregation(eval_metrics: List[tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Agrega las métricas RMSE reportadas por los clientes usando promedio ponderado.

    Cada cliente aporta su RMSE ponderado por el número de ejemplos que ha usado.

    Args:
        eval_metrics (List[tuple[int, Dict[str, float]]]): Lista de tuplas (n, métricas),
            donde `n` es el número de ejemplos usados por el cliente, y `métricas` es un
            diccionario con valores tipo {"rmse": valor}.

    Returns:
        Dict[str, float]: Métrica agregada global, como {"rmse": promedio_ponderado}.
    """
    total = sum(n for n, _ in eval_metrics)
    # Weighted RMSE
    rmse_agg = sum(metrics["rmse"] * n for n, metrics in eval_metrics) / total
    return {"rmse": rmse_agg}


def config_func(rnd: int) -> Dict[str, str]:
    """
    Devuelve la configuración específica para una ronda.

    Esta función se pasa a `on_fit_config_fn` y `on_evaluate_config_fn`, y se
    encarga de enviar el número de ronda actual al cliente, para permitir
    comportamientos condicionales (por ejemplo: entrenar desde cero en ronda 1).

    Args:
        rnd (int): Número de la ronda actual.

    Returns:
        Dict[str, str]: Diccionario de configuración enviado a los clientes.
    """
    return {"global_round": str(rnd)}

# === Crear la aplicación del servidor federado ===
def server_fn(context: Context) -> ServerAppComponents:
    """
    Función principal que configura y devuelve los componentes del servidor Flower.

    Se encarga de:
    - Leer la configuración del experimento.
    - Cargar datos para evaluación centralizada (opcional).
    - Seleccionar la estrategia de entrenamiento (bagging o cycling).
    - Definir el ClientManager en caso necesario.
    """
    cfg = context.run_config

    num_rounds       = cfg["num-server-rounds"]
    fraction_fit     = cfg["fraction-fit"]
    fraction_eval    = cfg["fraction-evaluate"]
    train_method     = cfg["strategy"]
    raw = cfg.get("params", {})  # diccionario con eta, max_depth...
    centralised_eval = cfg["centralised-eval"]
    
    num_rounds = int(cfg["num-server-rounds"])

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

    
    hyperparams: dict[str, float] = raw if isinstance(raw, dict) else {}
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        **hyperparams
    }
    # Cargar el dataset completo para evaluación centralizada (si aplica)
    test_dmatrix: Optional[xgb.DMatrix] = None
    if centralised_eval:
        df = load_clean_adidas_data(excel_path)
         # Codificar variables categóricas, string y datetime a valores numéricos
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].cat.codes
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category").cat.codes
        for col in df.select_dtypes(include=["datetime"]).columns:
            df[col] = df[col].astype("int64")
        X = df.drop(columns=["Units Sold"])
        y = df["Units Sold"]
        test_dmatrix = xgb.DMatrix(X, label=y)

    # Crear parámetros iniciales vacíos (modelo sin entrenar)
    initial_parameters = Parameters(tensor_type="", tensors=[])

    # Preparar función de evaluación centralizada (solo si está activada)
    evaluate_fn = None
    if centralised_eval and test_dmatrix is not None:
        evaluate_fn = get_evaluate_fn(test_dmatrix, params)

    # Definir la estrategia y el client manager
    if train_method == "bagging":
        strategy = FedXgbBagging(
            evaluate_function=evaluate_fn,
            fraction_fit=fraction_fit,
            fraction_evaluate=(0.0 if centralised_eval else fraction_eval),
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=(
                None if centralised_eval else evaluate_metrics_aggregation
            ),
            initial_parameters=initial_parameters,
        )
        client_manager = None # Usa el ClientManager por defecto
    else:  # "cyclic"
        strategy = FedXgbCyclic(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=initial_parameters,
        )
        client_manager = CyclicClientManager() 

     # 6) Devolver componentes listos para el servidor
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
        client_manager=client_manager,
    )

# === Instanciar el servidor federado ===
app = ServerApp(server_fn=server_fn)
