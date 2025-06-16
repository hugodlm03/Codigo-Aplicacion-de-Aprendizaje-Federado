# src/scripts/inspect_flwr_config.py
import pathlib
from flwr.common.config import get_fused_config_from_dir

if __name__ == "__main__":
    app_dir = pathlib.Path(__file__).parent.parent  # tu carpeta app
    default_cfg = get_fused_config_from_dir(app_dir, override_config={})
    print("Claves configurables por CLI (--run-config):")
    for key in default_cfg.keys():
        print(" -", key)

import pandas as pd

# Ruta a tu Excel (ajusta según tu estructura real)
ruta = "../datos/Adidas US Sales Datasets.xlsx"
# Si lo tienes en otro sitio pon el path relativo/correcto

# Carga el Excel
df = pd.read_excel(ruta, skiprows=3)  # saltar filas vacías del principio si hace falta

# ¡Opcional! Si tienes nombres raros de columnas tras limpiar:
# df.columns = [col.strip() for col in df.columns]

# Verifica que tienes las columnas correctas
print(df.columns)

# Contar particiones únicas de 'Retailer' + 'City'
particiones = df.groupby(['Retailer', 'City']).size().reset_index()
num_particiones = len(particiones)

print(f"Número de particiones (Retailer + City): {num_particiones}")
