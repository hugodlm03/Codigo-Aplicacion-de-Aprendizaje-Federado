#!/usr/bin/env python3
import itertools
from pathlib import Path
import toml

RUN_CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Esquemas de particionado de datos:
partitioners = ["retailer_region"]

# Estrategias de entrenamiento federado:
strategies = ["bagging", "cycling"]

# Evaluación centralizada en servidor (“on”) o solo evaluación local en clientes (“off”)
central_evals = ["on", "off"]

# Fracción de los datos locales reservada para validación/testing (por cliente)
test_fracs = [0.1, 0.2, 0.3]

# Número de epochs (rondas de boosting) que hace cada cliente en su partición
local_epochs = [1, 5, 10]

# Eta (learning rate) de XGBoost
etas = [0.05, 0.1]

# Profundidad máxima de los árboles
max_depths = [4, 6, 8]

# Submuestreo de filas por árbol
subsamples = [0.6, 0.8, 1.0]


def main():
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)

    for part, strat, ce, tf, le, eta, md, ss in itertools.product(
        partitioners, strategies, central_evals,
        test_fracs, local_epochs, etas, max_depths, subsamples
    ):
        # Construcción del nombre y del run-id
        name = (
            f"{part}-{strat}-ce-{ce}-tf-{tf}-le-{le}"
            f"-eta-{eta}-md-{md}-ss-{ss}"
        ).replace(".", "p")

        cfg = {
            "run-id": name,                          # Identificador único
            "strategy": strat,
            "partitioner": part,
            "centralised-eval": (ce == "on"),
            "test-fraction": tf,
            "local-epochs": le,
            "params": {
                "eta": eta,
                "max_depth": md,
                "subsample": ss
            },
        }

        filepath = RUN_CONFIGS_DIR / f"{name}.toml"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(toml.dumps(cfg))
        print("Creado", filepath)

if __name__ == "__main__":
    main()
