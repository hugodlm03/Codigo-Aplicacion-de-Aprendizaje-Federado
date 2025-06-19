---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
URL:  https://arxiv.org/abs/2304.07537
labels: [cross-silo, tree-based, XGBoost, Classification, Regression, Tabular] 
dataset: [a9a, cod-rna, ijcnn1, space_ga, cpusmall, YearPredictionMSD] 
---

# Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2304.07537](https://arxiv.org/abs/2304.07537)

**Authors:** Chenyang Ma, Xinchi Qiu, Daniel J. Beutel, Nicholas D. Lane

**Abstract:** The privacy-sensitive nature of decentralized datasets and the robustness of eXtreme Gradient Boosting (XGBoost) on tabular data raise the need to train XGBoost in the context of federated learning (FL). Existing works on federated XGBoost in the horizontal setting rely on the sharing of gradients, which induce per-node level communication frequency and serious privacy concerns. To alleviate these problems, we develop an innovative framework for horizontal federated XGBoost which does not depend on the sharing of gradients and simultaneously boosts privacy and communication efficiency by making the learning rates of the aggregated tree ensembles are learnable. We conduct extensive evaluations on various classification and regression datasets, showing our approach achieve performance comparable to the state-of-the-art method and effectively improves communication efficiency by lowering both communication rounds and communication overhead by factors ranging from 25x to 700x.


## 1  Puesta en marcha rápida

```bash
# Crear entorno virtual + instalar dependencias
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # o poetry install

# Baseline centralizado (70 / 30)
python -m hfedxgboost.main \
    --config-name Centralized_Baseline \
    dataset=adidas_us_sales \
    xgboost_params_centralized=adidas_us_sales_xgboost_centralized \
    dataset.train_ratio=0.7

# Aprendizaje federado (28 clientes, 100 rondas)
python -m hfedxgboost.main \
    dataset=adidas_us_sales \
    clients=adidas_28_clients \
    run_experiment.num_rounds=100
```

Los CSV se guardan automáticamente en `results/`:

- `results_centralized.csv`
- `results.csv` (federado)

---

## 2  Estructura del proyecto

```
hfedxgboost/
├── client.py               # FlClient: envuelve XGB + CNN para Flower
├── server.py               # FlServer: _early‑stopping_ + logs propios
├── strategy.py             # FedXgbNnAvg: FedAvg para conjuntos de árboles
├── adidas_dataset.py       # carga, limpieza y partición no‑IID
├── early_stop.py           # parada temprana basada en rondas
├── results_io.py           # create_res_csv, ResultsWriter, ...
├── utils.py                # helpers de split y Torch Dataset wrappers
├── main.py                 # _entry‑point_ CLI, arranca Hydra
└── conf/                   # configuraciones Hydra
    ├── base.yaml
    ├── clients/*.yaml      # nº clientes, params locales XGB…
    ├── dataset/*.yaml      # opciones de cada _task_
    └── xgboost_params_*.yaml
```



---

## 3  Clases principales

| Módulo              | Clase / función                             | Papel                                                                                                                                                                                |
| ------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `client.py`         | `FlClient`                                  | Implementa `get_parameters`, `fit`, `evaluate`. Acepta tanto la API antigua **(parameters + config)** como `EvaluateIns` (Flower ≥ 1.4). |
|                     | `train`, `test` (internas)                  | Ejecutan una época local de entrenamiento/evaluación; pueden usar **GPU** si se indica `device="cuda"`.                                                                              |
| `server.py`         | `FlServer`                                  | Subclase de *Flower Server* que enchufa un `EarlyStopper` y añade logs formateados.                                                                                                  |
| `strategy.py`       | `FedXgbNnAvg`                               | Variación de FedAvg capaz de promediar árboles XGBoost y la cabecera CNN. Controla `fraction_fit` / `fraction_evaluate`.                                                             |
| `early_stop.py`     | `EarlyStop`                                 | Supervisa la *loss* de test del servidor y detiene si no mejora tras *n* rondas.                                                                                                     |
| `results_io.py`     | `create_res_csv`, `ResultsWriter`           | Auxiliares de E/S que crean/actualizan los CSV de métricas.                                                                                                                          |
| `adidas_dataset.py` | `load_clean_adidas_data`, `adidas_to_numpy` | Limpia el CSV bruto de ventas Adidas US, codifica categóricas y devuelve `pd.DataFrame`/`np.ndarray`.                                                                                |
|                     | `adidas_split_non_iid`                      | Particiona filas por cliente con *label‑skew* para simular no‑IID.                                                                                                                   |

---

## 4  Flujo de un experimento

1. **Hydra** resuelve los YAML de `conf/*` y genera un objeto `OmegaConf`.
2. `main.py` imprime la config y decide:
   - **centralizado** → `run_centralized()` ejecuta `xgboost.train`, muestra RMSE y escribe CSV.
   - **federado** → construye `DataLoader` → instancia `FedXgbNnAvg` (+ callbacks) → lanza la **simulación Flower** sobre Ray.
3. Cada ronda, cada `FlClient` recibe el ensamble global, añade sus propios árboles (`n_estimators_client`), entrena `num_iterations` y devuelve los árboles nuevos.
4. El servidor agrega con `FedXgbNnAvg`, evalúa sobre el *testset* y vuelve a difundir.

---

## 5  GPU y rendimiento

Aqui tengo problemas, porque al no ser como kaggle que puedes utilizar un gpu extener, el rendimiento es muy bajo.

Si sabeis alguna alternativa, agradeceria el consejo.

---

## 6  Logs y resultados

- **CSV** → `results/` se crea al vuelo (centralizado vs. federado).


