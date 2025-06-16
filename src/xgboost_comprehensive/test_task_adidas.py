# test_task_adidas.py
"""
Script de prueba para la función load_data() de task_adidas.py.

Este archivo permite comprobar localmente si los datos se cargan, particionan y transforman
correctamente a objetos DMatrix de XGBoost. Ideal para pruebas rápidas antes de ejecutar federado.

Ejemplo de ejecución:
    cd "C:\\Users\\PC\\Desktop\\s.o.e\\Estudios\\U-4\\Segundo Cuatri\\TFG\\Codigo-Aplicación de Aprendizaje Federado\\src"
    .\\flwr_tu\\Scripts\\Activate
    python test_task_adidas.py
"""

from task_adidas import load_data

if __name__ == "__main__":
    # Parámetros de prueba
    partitioner_type   = "region"       # uno de: region, retailer_region, retailer_city
    partition_id       = 0              # primer nodo
    num_partitions     = 5              # solo usado si hicieras modulo
    centralised_eval   = False          # prueba split local
    test_fraction      = 0.2
    seed               = 42
    excel_path         = "../datos/Adidas US Sales Datasets.xlsx"

    # Invocamos load_data
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partitioner_type=partitioner_type,
        partition_id=partition_id,
        centralised_eval_client=centralised_eval,
        test_fraction=test_fraction,
        seed=seed,
        excel_path=excel_path,
    )


    # Imprimimos tamaños y algunas comprobaciones
    print(f"Train partition #{partition_id} ({partitioner_type}):")
    print("  → num_train rows:", num_train, "| DMatrix rows:", train_dmatrix.num_row())
    print("  → num_valid rows:", num_val, "| DMatrix rows:", valid_dmatrix.num_row())
