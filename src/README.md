# Aprendizaje Federado con XGBoost y Flower



## Introducción

Este proyecto muestra cómo **entrenar modelos *gradient-boosted decision trees* (XGBoost) en un entorno federado** usando la plataforma de investigación [**Flower**](https://flower.ai).
Su objetivo es proporcionar un marco **minimalista pero extensible** que te permita:

* definir **cientos de configuraciones de experimento** mediante archivos **TOML**;
* lanzar esos experimentos **automáticamente en lote** (o de forma interactiva);
* **registrar** y **comparar métricas** de servidor y clientes en archivos CSV;
* probar rápidamente **estrategias de agregación** (*bagging*, *cyclic boosting*, etc.);
* **conservar** los datos **en el origen** de cada partición, respetando la privacidad.



---

## Requisitos

| Paquete         | Versión mínima |
| --------------- | -------------- |
| Python          | **3.10**       |
| `flwr`          | 1.18.0         |
| `xgboost`       | 2.0.0          |
| `flwr-datasets` | 0.5.0          |
| `pandas`        | 2.3.0          |
| `questionary`   | 2.1.0          |

> **Sugerencia** · Instala el proyecto en modo editable (`pip install -e .`) para que se gestionen automáticamente todas las dependencias.

---

## Instalación paso a paso

```bash
# 1) Clona el repositorio y accede a él
$ git clone <url-del-repo>
$ cd xgboost-flwr-framework

# 2) (Opcional) Crea un entorno virtual aislado
$ python -m venv .venv
$ source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate  # Windows

# 3) Instala en modo editable + deps
$ pip install -e .
```

Si lo prefieres, puedes instalar desde un `requirements.txt` tradicional:

```bash
$ pip install -r requirements.txt
```

---

## Estructura del proyecto

```
.
├── configs/                 # ↳ Archivos .toml con los parámetros de cada experimento
├── results/                 # ↳ Salida en CSV de todas las métricas
├──notebooks/ 
│   └──adidasxgboost_comprehensive.ipynb  # Libreta de ejecución.
├── xgboost_comprehensive/
│   ├── client_app.py        # Lógica de cada cliente Flower
│   ├── server_app.py        # Lógica del servidor/agregador
│   ├── data_loader.py       # Limpieza y particionamiento de datasets
│   ├── task.py              # Particionamiento tipo «Higgs» (HuggingFace)
│   ├── task_adidas.py       # Particionamiento Adidas (región / retailer / ciudad)
│   └── metrics.py           # Funciones auxiliares de métricas
└── README.md                # (Este documento)
```



-

## Flujo de trabajo recomendado

### 1. Entrar en la libreta

---
### 2. Generar combinaciones

Crea automáticamente todas las combinaciones relevantes de parámetros (modelo, estrategia, fracciones, etc.).


Esto abonará la carpeta `configs/` con decenas (o cientos) de ficheros `.toml`.

---

### 3. Lanzar experimentos

Durante la ejecución, se mostrará un *stream* en vivo del log y el tiempo consumido por cada configuración.
Los resultados se guardan en `results/<run-id>.csv`.

---

### 5. Analizar resultados

Cada CSV generado contiene al menos:

| Columna      | Descripción                              |
| ------------ | ---------------------------------------- |
| `round`      | Ronda federada                           |
| `rmse`       | Métrica global/validación                |
| `eva_time  ` | Medida de tiempo de la evaluación        |

Puedes abrir los CSV con **pandas**, Excel, Tableau, Power BI o tu herramienta favorita para comparar estrategias.

---

## Datasets de ejemplo

| Dataset                 | Descripción breve                             | Script de carga  |
| ----------------------- | --------------------------------------------- | ---------------- |
| **Adidas US Sales**     | Ventas minoristas Adidas en EE. UU. (privado) | `data_loader.py` |

Cada partición se convierte internamente a `xgboost.DMatrix` y **no se comparte** entre nodos.

---

