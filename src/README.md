# Aprendizaje Federado con XGBoost y Flower

> **Proyecto final de TFG – Ejemplo completo y adaptable**

Este repositorio muestra cómo entrenar modelos **XGBoost** de forma **federada** (simulada) con la ayuda de
[**Flower**](https://flower.ai). El objetivo es ofrecer un _framework_ ligero pero extensible que permita

* crear cientos de configuraciones de experimento mediante archivos **TOML**;
* lanzar esos experimentos de manera interactiva o en lote;
* recopilar y comparar métricas tanto del **servidor** como de los **clientes**;
* probar estrategias de agregación como **bagging** o **cyclic boosting**;
* mantener los datos *en origen*, preservando la privacidad de cada partición.


## Requisitos
* **Python ≥ 3.10**
* Sistema operativo Windows, macOS o Linux
* Conexión a Internet para descargar dependencias y datasets públicos

| Paquete       | Versión mínima |
|---------------|----------------|
| `flwr`        | 1.18.0         |
| `xgboost`     | 2.0.0          |
| `flwr-datasets` | 0.5.0       |
| `pandas`      | 2.3.0          |
| `questionary` | 2.1.0          |

> **Tip**: todas las dependencias se resuelven automáticamente al instalar el proyecto en modo editable; consulta la sección siguiente.

---

## Instalación
```bash
# 1) Clona el repositorio y entra en él
$ git clone <url-del-repo>
$ cd Codigo-Aplicacion-de-Aprendizaje-Federado

# 2) Crea y activa un entorno virtual (opcional pero recomendado)
$ python -m venv .venv
$ source .venv/bin/activate      # Linux/Mac
# .\.venv\Scripts\Activate      # Windows PowerShell

# 3) Instala el proyecto en modo editable (incluye dependencias)
$ pip install -e .
```

Si prefieres un enfoque tradicional, instala desde `requirements.txt`:
```bash
$ pip install -r requirements.txt
```

---

## Estructura del repositorio
```
.
├── configs/                 # Archivos TOML (autogenerados) con combinaciones de parámetros
├── results/                 # CSV de métricas producidos tras los experimentos
├── scripts/             # CLI: gen_configs, run_experiments, inspect_flwr_config
├──xgboost_comprehensive/
│       ├── client_app.py    # Lógica del cliente Flower
│       ├── server_app.py    # Lógica del servidor Flower
│       ├── data_loader.py   # Limpieza y particionado del dataset Adidas
│       ├── task.py          # Particionado basado en HuggingFace (dataset Higgs)
│       ├── task_adidas.py   # Particionado basado en Adidas (por región, retailer, ciudad)
│       └── …
└── README.md                # Este documento
└── requirements.txt
```

---

## Flujo de trabajo rápido

### 1. Generar configuraciones
Crea todos los archivos **TOML** necesarios para tus experimentos:
```bash
$ python src/scripts/gen_configs.py
```
Por defecto se exploran distintos valores de _partitioner_, _strategy_, _test‑fraction_, etc.

### 2. Lanzar experimentos
Ejecuta **uno** o **todos** los experimentos de la carpeta `configs/` con un sencillo asistente interactivo:
```bash
$ python src/scripts/run_experiments.py
```
Cada ejecución guarda las métricas globales en `results/<nombre‑config>.csv`.

### 3. Analizar resultados
Esto no he conseguido que me funcione, pero es la idea.

Abre los CSV de `results/` con tu herramienta favorita (Excel, pandas, Tableau…).
Cada fila incluye:
* `round`   — número de ronda
* `rmse`    — métrica de validación global o local
* `train_time` — duración total del experimento

---





## Detalles del dataset
* **Adidas US Sales** (`datos/Adidas US Sales Datasets.xlsx`)
  * Limpieza y tipado automático vía `data_loader.py`.
  * Particionamiento configurable:
    * `region` — 5 nodos (Oeste, Este, Centro…)
    * `retailer_region` — 28 nodos
    * `retailer_city` — 108 nodos

> Cada partición se transforma a `xgb.DMatrix`, manteniendo los datos in‑situ.

---

## Archivos de configuración
Los archivos **TOML** de `configs/` fusionan parámetros de Flower y de XGBoost.
Un ejemplo mínimo:
```toml
run-id          = "demo"
strategy        = "bagging"         # o "cycling"
partitioner     = "retailer_region"
centralised-eval = true              # eval server‑side
local-epochs    = 5                 # árboles entrenados por cliente y ronda

[params]
eta         = 0.1
max_depth   = 8
subsample   = 1.0
```
Modifica cualquier clave y vuelve a ejecutar el experimento.

---

## Créditos y licenciamiento
Basado en el ejemplo oficial _Comprehensive XGBoost + Flower_ © Flower Labs. Código bajo licencia **Apache 2.0**.

```
@article{beutel2020flower,
  title     = {Flower: A Friendly Federated Learning Research Framework},
  author    = {Beutel, Daniel J and Topal, Taner and Mathur, Akhil and …},
  journal   = {arXiv preprint arXiv:2007.14390},
  year      = {2020}
}
```

¡Felices experimentos federados! 🚀

