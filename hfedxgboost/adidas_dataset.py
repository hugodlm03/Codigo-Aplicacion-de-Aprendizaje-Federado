# src_hfedxgboost/dataset_preparation.py
# ─────────────────────────────────────
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ruta del archivo de datos
DATA_FILE = Path(__file__).parent / ".." / ".." / "datos" / "Adidas US Sales Datasets.xlsx"

# Funnción para cargar y limpiar el dataset de Adidas
def load_clean_adidas_data(ruta_excel: Path = DATA_FILE) -> pd.DataFrame:
    """Lee el Excel original y devuelve un DataFrame limpio."""
    df = pd.read_excel(ruta_excel)

    # las tres primeras filas son texto decorativo
    df = df.drop([0, 1, 2])

    # fila 3 contiene los nombres reales de columna
    header_row = df.loc[3]
    df = df.rename(columns=header_row.to_dict())

    df = df.drop(df.columns[0], axis=1)  # primera col vacía
    df = df.drop(index=3).reset_index(drop=True)

    # convierte 'Invoice Date' a datetime y descarta nulos
    valid = pd.to_datetime(df["Invoice Date"], errors="coerce").notna()
    df = df[valid].copy()
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])

    # tipa columnas
    cat_cols = ["Retailer", "Region", "State", "City", "Product", "Sales Method"]
    num_cols = ["Price per Unit", "Total Sales", "Operating Profit", "Operating Margin"]
    target_col = "Units Sold"

    for c in cat_cols:
        df[c] = df[c].astype("category")

    df[num_cols + [target_col]] = df[num_cols + [target_col]].apply(
        pd.to_numeric, errors="coerce"
    )

    return df

# Función para transformar el DataFrame limpio en matrices NumPy
def adidas_to_numpy(df: pd.DataFrame, *, one_hot: bool = False):
    """Transforma el DF limpio en X (features) y y (objetivo)."""
    target = df["Units Sold"].to_numpy(dtype=np.float32)

    # fecha → ordinal (días desde epoch)
    df["Invoice_Ord"] = (df["Invoice Date"].view("int64") // 86_400_000_000_000).astype(
        np.int64
    )

    num_cols = [
        "Invoice_Ord",
        "Price per Unit",
        "Total Sales",
        "Operating Profit",
        "Operating Margin",
    ]
    cat_cols = ["Retailer", "Region", "State", "City", "Product", "Sales Method"]

    if one_hot:
        df_enc = pd.get_dummies(df[cat_cols], dtype=np.float32)
        X = np.hstack([df[num_cols].to_numpy(dtype=np.float32), df_enc.to_numpy()])
    else:
        # etiqueta entera por categoría + enable_categorical = True en XGB
        encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
        cat_matrix = np.stack([encoders[c].transform(df[c]) for c in cat_cols], axis=1)
        X = np.hstack([df[num_cols].to_numpy(dtype=np.float32), cat_matrix])

    return X, target, df[["Retailer", "Region"]]

# Funcion para dividir el dataset en clientes no IID
def adidas_split_non_iid(X, y, meta_df):
    """Devuelve lista de dicts (uno por cliente)."""
    clients = []
    grouped = meta_df.groupby(["Retailer", "Region"]).groups
    for (ret, reg), idx in grouped.items():
        clients.append(
            {
                "client_id": f"{ret}_{reg}",
                "x_train": X[idx],
                "y_train": y[idx],
            }
        )
    return clients
