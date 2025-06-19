#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path
import toml
import pandas as pd
import questionary
from scripts.metrics import init_metrics_csv, append_round

# Directorios y rutas
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent  # src/
CONFIG_DIR = SRC_DIR / "configs"
RESULTS_DIR = SRC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Etiquetas para cada nivel de filtrado
LEVEL_LABELS = {
    "partitioner": "Particionador",
    "strategy":    "Estrategia",
    "ce":          "Evaluación centralizada",
    "tf":          "Fracción test",
    "le":          "Épocas locales",
    "eta":         "Tasa de aprendizaje (eta)",
    "md":          "Profundidad máxima",
    "ss":          "Submuestreo",
}

def list_configs():
    return sorted(CONFIG_DIR.glob("*.toml"))

def get_config_levels(cfg: Path):
    parts = cfg.stem.split("-")
    return {
        "partitioner": parts[0],
        "strategy":    parts[1],
        "ce":          parts[3],
        "tf":          parts[5],
        "le":          parts[7],
        "eta":         parts[9],
        "md":          parts[11],
        "ss":          parts[13],
    }

def filter_configs(configs):
    niveles = ["partitioner", "strategy", "ce", "tf", "le", "eta", "md", "ss"]
    candidatos = configs
    for nivel in niveles:
        opciones = sorted({get_config_levels(c)[nivel] for c in candidatos})
        if len(opciones) <= 1:
            continue
        etiqueta = LEVEL_LABELS.get(nivel, nivel)
        sel = questionary.select(
            f"Filtra por {etiqueta}", choices=["(todos)"] + opciones
        ).ask()
        if sel and sel != "(todos)":
            candidatos = [c for c in candidatos if get_config_levels(c)[nivel] == sel]
        if len(candidatos) == 1:
            break
    return candidatos

def choose_config(configs):
    candidatos = filter_configs(configs)
    if len(candidatos) == 1:
        return candidatos[0]
    sel = questionary.select("Elige configuración", choices=[c.name for c in candidatos]).ask()
    return CONFIG_DIR / sel

def main():
    configs = list_configs()
    if not configs:
        print("No hay archivos TOML en src/configs/. Genera primero las configuraciones.")
        return

    accion = questionary.select(
        "¿Qué quieres hacer?",
        choices=["Ejecutar TODOS los experimentos", "Ejecutar UN experimento", "Salir"]
    ).ask()
    if accion == "Salir":
        return

    to_run = configs if accion.startswith("Ejecutar TODOS") else [choose_config(configs)]

    for cfg in to_run:
        # Leer run-id desde el TOML
        cfg_dict = toml.loads(cfg.read_text(encoding="utf-8"))
        experiment_id = cfg_dict.get("run-id", cfg.stem)

        levels = get_config_levels(cfg)
        partitioner = levels["partitioner"]
        nodes = {"region":5, "retailer_region":28, "retailer_city":108}.get(partitioner,1)
        print(f"Lanzando experimento {cfg.name}: {nodes} nodos para '{partitioner}'")

        # Preparar CSV local por experimento
        csv_path = RESULTS_DIR / f"{cfg.stem}.csv"
        init_metrics_csv(csv_path)

        # Ejecutar experimento mostrando salida en tiempo real
        cmd = ["flwr", "run", str(SRC_DIR), "--run-config", str(cfg)]
        start = time.time()
        result = subprocess.run(cmd, cwd=str(SRC_DIR))
        duration = time.time() - start

        if result.returncode != 0:
            print(f" Falló {cfg.name}, código {result.returncode}")
            continue

        # Leer métricas globales desde el CSV del servidor
        server_csv = RESULTS_DIR / "server_metrics.csv"
        if not server_csv.exists():
            print(f" No existe {server_csv}. ¿Se ha inicializado correctamente en el servidor?")
            continue

        df = pd.read_csv(server_csv)
        exp_df = df[df["experiment"] == experiment_id]
        if exp_df.empty:
            print(f"No hay filas en {server_csv} para run-id='{experiment_id}'")
            continue

        # Volcar cada ronda en el CSV local
        for _, row in exp_df.iterrows():
            append_round(
                csv_path,
                {"config": cfg.stem},
                int(row["round"]),
                float(row["rmse"]),
                duration,
            )
        print(f" Métricas guardadas en {csv_path}")

if __name__ == "__main__":
    main()

