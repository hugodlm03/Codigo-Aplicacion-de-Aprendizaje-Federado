# analiza_particiones_adidas.py

from src.xgboost_comprehensive.data_loader import load_clean_adidas_data

# Ruta al Excel (relativa a raíz del proyecto)
ruta = "datos/Adidas US Sales Datasets.xlsx"

# Cargar y limpiar los datos usando tu función oficial
df = load_clean_adidas_data(ruta)

# Número de particiones retailer + region
n_retailer_region = df[["Retailer", "Region"]].drop_duplicates().shape[0]
print(f"Particiones retailer + region: {n_retailer_region}")

# Número de particiones retailer + city
n_retailer_city = df[["Retailer", "City"]].drop_duplicates().shape[0]
print(f"Particiones retailer + city: {n_retailer_city}")

# valores únicos
# print(df[["Retailer", "Region"]].drop_duplicates())
# print(df[["Retailer", "City"]].drop_duplicates())
