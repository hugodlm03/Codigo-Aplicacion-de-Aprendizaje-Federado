from pathlib import Path
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np

# Ruta absoluta del Excel original
DATA_FILE = Path(
    r"C:\Users\PC\Desktop\s.o.e\Estudios\U-4\Segundo Cuatri\TFG\Codigo-Aplicacion de Aprendizaje Federado\datos\Adidas US Sales Datasets.xlsx"
)

# --------------------------------------------------
# Carga y limpieza del dataset
# --------------------------------------------------
def load_clean_adidas_data(ruta_excel: Path) -> pd.DataFrame:
    df = pd.read_excel(ruta_excel)
    df = df.drop([0, 1, 2])
    header_row = df.loc[3].to_dict()
    df = df.rename(columns=header_row)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(index=3).reset_index(drop=True)

    # Validar fechas
    mask = pd.to_datetime(df["Invoice Date"], errors="coerce").notna()
    df = df[mask].copy()
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])

    # Cast de columnas
    cat_cols = ["Retailer", "Region", "State", "City", "Product", "Sales Method"]
    num_cols = ["Price per Unit", "Total Sales", "Operating Profit", "Operating Margin", "Units Sold"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# --------------------------------------------------
# Preprocesado de features (X, y)
# --------------------------------------------------
def preprocess_features(raw: pd.DataFrame):
    y = raw["Units Sold"].values

    X = raw.drop(columns=["Total Sales", "Operating Profit", "Units Sold", "Invoice Date"]).copy()

    dates = pd.to_datetime(raw["Invoice Date"], errors="coerce")
    X["inv_year"], X["inv_month"], X["inv_day"] = dates.dt.year, dates.dt.month, dates.dt.day

    X = pd.get_dummies(X, columns=["Retailer", "Region", "Product", "Sales Method"], drop_first=True)

    for col in ["State", "City"]:
        X[col] = X[col].cat.codes

    return X.astype("float64"), y

# --------------------------------------------------
def main() -> None:
    df = load_clean_adidas_data(DATA_FILE)

    # Plantilla completa de columnas
    X_full, _ = preprocess_features(df)
    COL_TEMPLATE = sorted(X_full.columns)
    n_cols = len(COL_TEMPLATE)
    print(f"→ Plantilla fija de {n_cols} columnas")

    out_dir = Path("dataset") / "adidas_partitioned"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Particionar 70/30
    df_train, df_test = train_test_split(df, test_size=0.30, stratify=df["Product"], random_state=42)

    # --------------------------------------------------
    def export(lib_name: str, subset: pd.DataFrame) -> None:
        X_df, y_ser = preprocess_features(subset)

        # Reindexar con columnas ausentes → 0.0
        X_df = X_df.reindex(columns=COL_TEMPLATE, fill_value=0.0)

        assert X_df.shape[1] == n_cols, f"{lib_name}: columnas ≠ plantilla"

        dump_svmlight_file(
            X_df.values,
            np.asarray(y_ser),
            str(out_dir / f"{lib_name}.libsvm"),
            zero_based=True,
        )
        print(f"  • {lib_name}.libsvm → {len(subset)} filas, {n_cols} cols")

    # Centralizado y conjunto completo
    export("adidas", df)
    export("centralized", df_train)
    export("train", df_train)
    export("test", df_test)

    # Particiones por nodo federado (Retailer + Region)
    groups = df_train.groupby(["Retailer", "Region"], observed=True)
    for i, (_, part) in enumerate(groups):
        part_train, part_val = train_test_split(
            part, test_size=0.2, random_state=42
        )
        export(f"silo_train_{i:02d}", part_train.reset_index(drop=True))
        export(f"silo_val_{i:02d}", part_val.reset_index(drop=True))


if __name__ == "__main__":
    main()
