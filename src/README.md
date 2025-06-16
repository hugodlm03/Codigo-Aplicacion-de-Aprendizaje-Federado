Descripción del proyecto

Un pequeño framework de Aprendizaje Federado usando Flower y XGBoost, pensado para que puedas:

- Crear docenas de configuraciones de experimento (variando particiones, estrategias y parámetros).

- Lanzar esos experimentos de forma interactiva o todos de golpe.

- Recolectar métricas tanto de cada cliente como del servidor central.

- Comparar resultados y aprender qué combinación de parámetros funciona mejor.

-------------------------------

1. Generación de configuraciones

python scripts/gen_configs.py

2. Ejecución de experimentos
python scripts/run_experiments.py

-----------------------------------------------------

A continuación se describe cómo funciona el enfoque de bagging federado en este proyecto, pasando por la lógica de muestreo, entrenamiento local y agregación de modelos:

- Bagging, o bootstrap aggregating, es una técnica de ensamblado que mejora la estabilidad y precisión de los modelos entrenando múltiples instancias en diferentes muestras con reemplazo del conjunto de datos y luego agregando sus predicciones 

- El aprendizaje federado permite que múltiples clientes colaboren en el entrenamiento de un modelo compartido sin intercambiar sus datos brutos, preservando así la privacidad de cada partición local 

- XGBoost es una librería optimizada de gradient boosting que construye un ensamblado de árboles de decisión de forma aditiva, añadiendo en cada iteración nuevos árboles que corrigen los errores de los anteriores 

-  En nuestro proyecto, la clase FedXgbBagging de Flower implementa esta estrategia: en cada ronda el servidor selecciona un subconjunto de clientes (determinado por el parámetro fraction_fit) para simular el muestreo bootstrap a nivel de clientes 

- Cada cliente elegido recibe los parámetros globales y un diccionario de configuración con hiperparámetros (learning rate, max_depth, subsample) definidos en los ficheros TOML, y entrena localmente un número fijo de epochs de boosting 

- Tras el entrenamiento local, los clientes devuelven los árboles recién entrenados en lugar de gradientes o pesos, preservando datos y reduciendo el tráfico de comunicación 

- El servidor agrega estas actualizaciones concatenando los nuevos árboles a los ya existentes en la instancia global, realizando así un bagging distribuido que reduce la varianza del modelo final 

- Dado que cada cliente aporta subconjuntos distintos de árboles entrenados en particiones diversas, el ensamblado aprovecha múltiples visiones de la distribución de datos, mejorando la robustez frente a datos no IID 

- Si está activada, la evaluación centralizada calcula la métrica RMSE global sobre un conjunto de test mantenido en el servidor después de cada ronda y registra los resultados en un CSV para seguimiento de experimentos 

- En conjunto, el bagging federado con FedXgbBagging combina las ventajas estadísti­cas del bootstrap aggregating con la preservación de privacidad del aprendizaje federado para entrenar un modelo XGBoost distribuido y con varianza reducida