"""
Módulo de utilidades para carga y preprocesamiento de datos.

Este script importa las librerías necesarias para trabajar con los datos
y preparar su lectura desde archivos Excel. Puede ser usado como parte de
la cadena de procesamiento previa al entrenamiento federado.

────────────────────────────
Recuerda activar el entorno virtual antes de ejecutar:

    .\\fedenv\\Scripts\\Activate
────────────────────────────
"""

# === Librerías necesarias ===
import sys
from pathlib import Path
import pandas as pd


# FUNCIÓN: Cargar y limpiar datos 
def load_clean_adidas_data(ruta_excel: str | Path) -> pd.DataFrame:
    """
    Carga y limpia el conjunto de datos de Adidas desde un archivo Excel.

    Esta función:
    - Elimina filas vacías iniciales y ajusta la fila 4 como cabecera real.
    - Convierte la columna de fechas al formato datetime.
    - Realiza tipado explícito para variables categóricas y numéricas.
    - Elimina registros con fechas inválidas o datos nulos.

    Args:
        ruta_excel (str | Path): Ruta al archivo Excel (.xlsx) con los datos de ventas.

    Returns:
        pd.DataFrame: DataFrame limpio y listo para análisis o entrenamiento.
    """

    # Leer archivo
    df = pd.read_excel(ruta_excel)

     # Limpiar encabezado: eliminar filas vacías e interpretar fila 4 como cabecera real
    df = df.drop([0, 1, 2]) # Las primeras 3 filas no contienen datos útiles
    columnas = df.loc[3]
    df = df.rename(columns=columnas.to_dict()) # Fila 4 se convierte en encabezado
    df = df.drop(df.columns[0], axis=1) # Eliminar primera columna vacía
    df = df.drop(index=3).reset_index(drop=True)

     # Convertir 'Invoice Date' a datetime y eliminar filas con fechas no válidas
    df = df[pd.to_datetime(df['Invoice Date'], errors='coerce').notna()]
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    # Definir variables por tipo
    variables_categoricas = ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method']
    variables_numericas = ['Price per Unit', 'Total Sales', 'Operating Profit', 'Operating Margin']
    variable_objetivo = 'Units Sold'

    # Conversión de tipos
    for col in variables_categoricas:
        df[col] = df[col].astype('category')

    for col in variables_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[variable_objetivo] = pd.to_numeric(df[variable_objetivo], errors='coerce')

    return df

# GRUPO DE FUNCIONES PARA PARTICIONAR EL DATAFRAME

# FUNCIÓN: Particionar por nodo federado (Retailer + Región)
def partition_by_retailer_region(df: pd.DataFrame) -> dict:
    """
    Divide el DataFrame en subconjuntos por combinación de Retailer y Región.

    Cada subconjunto representa un nodo federado lógico (cliente) en el sistema FL.

    Args:
        df (pd.DataFrame): Conjunto de datos completo, ya limpio y preprocesado.

    Returns:
        dict[str, pd.DataFrame]: Diccionario donde cada clave tiene formato 'Retailer - Región'
        y cada valor es el DataFrame correspondiente al nodo.
    """
    # Agrupar por combinaciones únicas de 'Retailer' y 'Region'
    grupos = df.groupby(["Retailer", "Region"], observed=True)  # observed=True evita FutureWarning

    particiones = {
        f"{retailer} - {region}": sub_df.reset_index(drop=True)
        for (retailer, region), sub_df in grupos
    }

    return particiones


# FUNCIÓN: Particionar por nodo federado (Retailer + City)
def partition_by_retailer_city(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Divide el DataFrame en subconjuntos por combinación de Retailer y City.

    Cada subconjunto representa un nodo federado lógico (cliente) dentro del sistema FL.

    Args:
        df (pd.DataFrame): DataFrame preprocesado con los datos completos de Adidas.

    Returns:
        dict[str, pd.DataFrame]: Diccionario con claves tipo 'Retailer - City' y
        valores correspondientes a los DataFrames de cada nodo.
    """
    # Agrupar por combinación única de Retailer y City
    grupos = df.groupby(['Retailer', 'City'], observed=True)  # Agrupar por retailer + ciudad
    particiones = {}
    # Crear diccionario de particiones
    for (retailer, city), sub_df in grupos:
        clave = f"{retailer} - {city}"                        # Nombre del nodo
        particiones[clave] = sub_df.reset_index(drop=True)   # Guardar copia limpia del subgrupo

    return particiones

# FUNCIÓN: Particionar por nodo federado (REGION)
def partition_by_region(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Divide el DataFrame en subconjuntos por la columna 'Region'.

    Cada subconjunto representa un nodo federado lógico (cliente) agrupado solo por región.

    Args:
        df (pd.DataFrame): Conjunto de datos preprocesado de Adidas.

    Returns:
        dict[str, pd.DataFrame]: Diccionario con claves tipo 'Region' y
                                 DataFrames correspondientes a cada nodo.
    """
    grupos = df.groupby('Region', observed=True)              # Agrupar por región
    particiones = {}

    for region, sub_df in grupos:
        clave = region                                        # Clave del nodo (región)
        particiones[clave] = sub_df.reset_index(drop=True)

    return particiones

#  FUNCIÓN: Guardar csv por nodo federado.
def guardar_nodos_en_csv(particiones, carpeta_salida: Path | str):
    """
    Guarda cada partición del dataset como un archivo CSV independiente.

    Cada archivo tendrá como nombre el identificador del nodo (limpio de espacios y símbolos).

    Args:
        particiones (dict[str, pd.DataFrame]): Diccionario de nodos con sus datos.
        carpeta_salida (Path | str): Carpeta destino donde se guardarán los CSV.
    """
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    for nombre, df_nodo in particiones.items():
        # Limpiar nombre del archivo: sin espacios ni barras
        filename = f"{nombre.replace(' ', '_').replace('/', '_')}.csv"
        df_nodo.to_csv(Path(carpeta_salida) / filename, index=False)

def contar_particiones(partitioner_type, df):
    if partitioner_type == "retailer_city":
        n = df[["Retailer", "City"]].drop_duplicates().shape[0]
    elif partitioner_type == "retailer_region":
        n = df[["Retailer", "Region"]].drop_duplicates().shape[0]
    elif partitioner_type == "region":
        n = df[["Region"]].drop_duplicates().shape[0]
    else:
        n = None
    return n

if __name__ == "__main__":
    import sys

    # Obtener esquema de particionado desde la línea de comandos (por defecto: retailer_region)
    if len(sys.argv) > 1:
        scheme = sys.argv[1]
    else:
        scheme = "retailer_region"

    # Ruta al archivo Excel de origen
    ruta_excel = "datos/Adidas US Sales Datasets.xlsx"
    # Cargar y limpiar datos
    df = load_clean_adidas_data(ruta_excel)

    # Elegir función de particionado y carpeta destino
    base_dir = Path("nodos")            # carpeta raíz común
    if scheme == "retailer_city":
        particiones = partition_by_retailer_city(df)
        subdir = base_dir / "retailer_city"
    elif scheme == "region":
        particiones = partition_by_region(df)
        subdir = base_dir / "region"
    elif scheme == "retailer_region":
        particiones = partition_by_retailer_region(df)
        subdir = base_dir / "retailer_region"
    else:
        raise ValueError(f"Esquema no reconocido: {scheme}")


    # Mostrar una muestra de 3 nodos por consola
    for nombre, df_nodo in list(particiones.items())[:3]:
        print(f"\n--- {nombre} ({len(df_nodo)} registros) ---")
        print(df_nodo.head())

    # Guardar CSVs
    guardar_nodos_en_csv(particiones, subdir)
    print(f"Particiones '{scheme}' guardadas en '{subdir}/'")

