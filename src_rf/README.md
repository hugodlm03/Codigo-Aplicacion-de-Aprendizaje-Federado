# Random Forest Federado sobre el **Adidas US Sales**

*Solución completa y reproducible construida con **PySyft ≥ 0.9***



## ¿Qué incluye el proyecto?

| Componente                                        | Propósito                                                                                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`data_loader.py` / `load_clean_adidas_data()`** | Limpieza del Excel original y conversión de tipos de datos.                                                                                                          |
| **Particionadores (`partition_by_*`)**            | Generan nodos federados según tres esquemas: `region`, `retailer_city`, `retailer_region`.                                                                           |
| **`export_partition()`**                          | Re-indexa cada CSV de nodo al **template global de 20 columnas** para garantizar idéntico orden de features.                                                         |
| **`server.py / FLServer`**                        | Mini-servidor Syft que expone un CSV como DataSite (modo *auto-accept* disponible).                                                                                  |
| **`client.py / FLClient`**                        | Orquestador: conecta con todos los silos, envía las funciones de entrenamiento/evaluación, fusiona bosques locales, aplica pesos adaptativos, guarda histórico, etc. |
| **`ml_experiment()`**                             | Rutina remota que cada silo ejecuta: preprocesado → entrenamiento (o incremento) de `RandomForestRegressor` → métricas + modelo serializado.                         |
| **`evaluate_global_model()`**                     | Rutina remota que puntúa el modelo global en el *hold-out*.                                                                                                          |
| **Sección de experimentos**                          | Lanza experimentos de *grid search*; cada configuración se almacena en `results/grid_runs/run_##/`.                                                                  |
| **`utils.py`**                                    | Auxiliares (EMA de pesos, logs bonitos, chequeo de proyectos Syft…).                                                                                                 |
| **Notebooks / scripts**                           | Ejemplos de uso, análisis visual (heat-maps, pair-plots, importancias, árboles…).                                                                                    |

---

## Estructura de carpetas

```text

src_rf/
│
├── fed_rf_mk/                     # paquete principal
│   ├── __init__.py
│   ├── client.py                  # FLClient
│   ├── server.py                  # FLServer
│   ├── datasets.py
│   ├── datasites.py
│   └── utils.py
│
├── ImplementacioADatasetAdidas/   # notebooks / pruebas
│   └── rf_adidas_datasets.ipynb
│   └── artefactos/
│       ├── template_cols.json         # 20 features globales
│       └── grid_results.csv           # resumen del último grid-search
│   └── nodos/                         # CSV de cada nodo + hold-out
│       ├── region/
│       ├── retailer_city/
│       └── retailer_region/
│   └── results/                       # salidas de los entrenamientos
│       ├── retailer_region/<timestamp>/…
│       └── grid_runs/run_##/…
│
└── requirements.txt




```

---

## Requisitos previos

* **Python 3.10 – 3.12**
* Instalar dependencias:

  ```bash
  pip install -r requirements.txt
  ```

  Bibliotecas clave: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, **`pysyft>=0.9`**, `cloudpickle`.



## Flujo de trabajo de extremo a extremo

| Paso                          | Script / función                            | Qué ocurre                                                                                                                     |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1. **Preparar datos**         | `load_clean_adidas_data()`                  | Elimina filas basura, castea tipos y muestra conteos.                                                                          |
| 2. **Ingeniería de features** | `preprocess_features()`                     | One-hot a categorías pequeñas, label-encoding a State/City, deja 20 columnas finales.                                          |
| 3. **Template global**        | `template_cols.json`                        | Asegura idéntico orden de columnas para todos los nodos.                                                                       |
| 4. **Particionado**           | `partition_by_*`                            | Exporta cada nodo como CSV ya re-indexado al template.                                                                         |
| 5. **Servidores**             | `FLServer`                                  | Cada CSV se sirve como DataSite Syft.                                                                                          |
| 6. **Cliente**                | `FLClient`                                  | Se conecta a N silos de entrenamiento + 1 de evaluación.                                                                       |
| 7. **Envío de código**        | `.send_request()`                           | Crea proyectos Syft y sube `ml_experiment` y `evaluate_global_model`.                                                          |
| 8. **Entrenamiento federado** | `.run_model()`                              | E0: cada silo entrena RF desde cero → merge.<br>E>0: reciben modelo global, añaden árboles.<br>Peso adaptativo = EMA de 1/MAE. |
| 9. **Evaluación global**      | `.run_evaluate()`                           | Hold-out produce MAE y RMSE finales.                                                                                           |
| 10. **Logs / artefactos**     | `history.csv`, `model.pkl`, `run_info.txt`. |                                                                                                                                |
| 11. **Grid search**           | `run_experiments.py`                        | Repite pasos 6-9 sobre una malla de hiper-parámetros.                                                                          |
| 12. **Análisis**              | notebooks / `inspect_flwr_config.py`        | Pair-plots, matrices de correlación, importancias, árboles, etc.                                                               |

---

## Cómo configurar los experimentos

### Malla de hiper-parámetros

Edita `RF_PARAM_GRID` en **`run_experiments.py`**:

```python
RF_PARAM_GRID = {
    "n_base_estimators":        [50, 100, 200],
    "n_incremental_estimators": [10],
    "train_size":               [0.7, 0.8],
    "max_depth":                [None, 5, 10],
    "max_features":             ["sqrt", "log2"],
    "min_samples_leaf":         [1, 5],
    "min_samples_split":        [2, 10],
}
```

Cualquier clave listada se reenvía tal cual a `RandomForestRegressor`.

### Pesos adaptativos

`utils.ema_weights(alpha=0.6)` actualiza `self.weights` tras cada epoch.
Puedes bajar `alpha` para suavizar más, o subirlo para reaccionar antes.


## Créditos y licencia

Basado y ampliado a partir de la propuesta **“Federated Random Forest” (Cotorobai et al., 2025)**.
Código bajo licencia **MIT** (consulta `LICENSE`).
