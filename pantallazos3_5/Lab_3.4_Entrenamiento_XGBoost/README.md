# Laboratorio 3.4: Entrenamiento de un modelo de Machine Learning


## Paso 1. Importar las librerías y cargar los datos

![Importación y carga de datos](capturas/01_importacion_y_carga_de_datos.png)

**Qué hice:** importé las librerías (pandas, scipy, boto3, entre otras), descargué el archivo comprimido del repositorio de UCI, lo abrí con `arff.loadarff` y lo pasé a un DataFrame de pandas. Después cambié la clase de texto a número: `Abnormal` pasó a 1 y `Normal` a 0.

**Por qué:** XGBoost trabaja con números, no con texto, así que la clase tenía que estar como 0 y 1 antes de entrenar. En esta captura se ve que las cuatro celdas se ejecutaron bien, incluida la de `scipy`, que es la que suele fallar si el paquete no está instalado.

## Paso 2. Explorar los datos

![Forma y columnas del DataFrame](capturas/02_exploracion_shape_y_columnas.png)

**Qué hice:** usé `df.shape` y `df.columns` para revisar el tamaño y las columnas. Salió `(310, 7)`: 310 pacientes y 7 columnas, que son las 6 características biomecánicas más la columna `class`.

**Por qué:** antes de dividir o entrenar conviene confirmar que los datos se cargaron completos y saber con qué columnas se está trabajando.

## Paso 3. Mover la columna objetivo al inicio

![Columna objetivo en primera posición](capturas/03_mover_columna_objetivo.png)

**Qué hice:** reordené las columnas para que `class` quedara de primera, y lo comprobé con `df.columns`.

**Por qué:** el algoritmo XGBoost de SageMaker espera archivos CSV donde el valor a predecir esté en la primera columna. Si no se hace este cambio, el modelo tomaría una característica como si fuera la etiqueta.

## Paso 4. Dividir los datos en entrenamiento, validación y prueba

![División con train_test_split](capturas/04_division_entrenamiento_y_prueba.png)

**Qué hice:** usé `train_test_split` de scikit-learn en dos pasos. En el primero separé 80% para entrenamiento y 20% para un grupo temporal (`test_and_validate`). En el segundo dividí ese 20% por la mitad: una parte para prueba y otra para validación. Quedó 80% / 10% / 10%.

**Por qué:** cada conjunto cumple un papel distinto. El de entrenamiento enseña al modelo, el de validación sirve para vigilar cómo va mientras entrena, y el de prueba se guarda para evaluarlo al final con datos que nunca vio. Usé `stratify` porque hay pocos datos y así cada conjunto conserva la misma proporción de normales y anormales, y `random_state=42` para que la división se pueda repetir igual.

## Paso 5. Revisar el tamaño de los conjuntos

![Tamaño de los tres conjuntos](capturas/05_tamano_de_los_conjuntos.png)

**Qué hice:** imprimí la forma de cada conjunto. Salieron 248 filas para entrenamiento, 31 para prueba y 31 para validación, todos con 7 columnas.

**Por qué:** para verificar que la división quedó como la planeé (248 + 31 + 31 = 310, ningún paciente se perdió ni se repitió).

## Paso 6. Revisar la distribución de las clases

![Distribución de clases](capturas/06_distribucion_de_clases.png)

**Qué hice:** conté cuántos pacientes de cada clase hay en cada conjunto. En entrenamiento hay 168 anormales y 80 normales; en prueba y en validación, 21 anormales y 10 normales cada uno.

**Por qué:** quería confirmar que `stratify` funcionó. La proporción es de unos dos anormales por cada normal en los tres conjuntos, igual que en el dataset original (210 anormales y 100 normales).

## Paso 7. Crear la función para subir los datos a Amazon S3

![Función de carga a S3](capturas/07_funcion_de_carga_a_s3.png)

**Qué hice:** definí el nombre del bucket, el prefijo `lab3` y los nombres de los archivos, y escribí la función `upload_s3_csv`. La función convierte un DataFrame a CSV en un búfer de memoria y lo sube a S3 como un objeto.

**Por qué:** SageMaker lee los datos de entrenamiento desde S3, no desde el cuaderno. Escribí el CSV con `header=False` e `index=False` porque XGBoost no espera encabezados ni la columna del índice de pandas. Usé un búfer para no tener que guardar archivos locales.

## Paso 8. Subir los tres conjuntos

![Carga de los tres conjuntos a S3](capturas/08_carga_de_los_tres_conjuntos.png)

**Qué hice:** llamé a la función tres veces para subir los archivos de entrenamiento, prueba y validación, cada uno en su carpeta (`train`, `test`, `validate`).

**Por qué:** así los datos quedan listos en S3 y el entrenamiento puede leerlos por canales separados.

## Paso 9. Obtener el contenedor de XGBoost

![URI del contenedor XGBoost](capturas/09_contenedor_xgboost.png)

**Qué hice:** con `retrieve('xgboost', región, '1.0-1')` obtuve la imagen del contenedor del algoritmo XGBoost, en la región de mi sesión.

**Por qué:** SageMaker ejecuta el entrenamiento dentro de un contenedor, y esa es la imagen que trae el algoritmo ya instalado, así que no hay que programarlo desde cero.

## Paso 10. Definir los hiperparámetros

![Hiperparámetros](capturas/10_hiperparametros.png)

**Qué hice:** configuré tres hiperparámetros: `num_round = 42` (cantidad de rondas de entrenamiento), `eval_metric = auc` (métrica con la que se evalúa) y `objective = binary:logistic` (clasificación en dos clases).

**Por qué:** es la primera vez que entreno este modelo, así que usé valores iniciales sencillos. El objetivo `binary:logistic` encaja con el problema porque hay solo dos resultados (normal o anormal) y hace que el modelo devuelva una probabilidad. El ajuste fino se hace en un laboratorio posterior.

## Paso 11. Configurar el estimador

![Estimador de SageMaker](capturas/11_estimador.png)

**Qué hice:** creé el objeto `Estimator` con el contenedor, el rol de ejecución, una sola instancia (`instance_count=1`) de tipo `ml.m4.xlarge`, la carpeta de salida en S3 y los hiperparámetros.

**Por qué:** el estimador es el objeto que le dice a SageMaker qué algoritmo usar, con qué máquina entrenarlo y dónde guardar el modelo resultante. Con una instancia es suficiente porque el dataset es muy pequeño.

## Paso 12. Definir los canales de datos

![Canales de entrenamiento y validación](capturas/12_canales_de_entrenamiento.png)

**Qué hice:** creé el canal `train` y el canal `validation`, apuntando a las carpetas de S3 que subí antes, con formato `text/csv`.

**Por qué:** el estimador recibe los datos por canales. Con el de validación, XGBoost puede calcular la métrica AUC durante el entrenamiento y yo puedo ver si el modelo está aprendiendo o solo memorizando.

## Paso 13. Entrenar el modelo

![Entrenamiento con fit](capturas/13_entrenamiento_fit.png)

**Qué hice:** ejecuté `xgb_model.fit(inputs=data_channels, logs=False)`, que lanza el trabajo de entrenamiento en SageMaker. En la captura se ve cómo se crea el trabajo (`sagemaker-xgboost-2026-10-13-20-41-57-537`) y que empieza en estado *Starting*.

**Por qué:** es el paso que realmente entrena el modelo. Tarda unos minutos porque SageMaker levanta la instancia, descarga los datos y la imagen, entrena y guarda el modelo en S3. La captura la tomé con el entrenamiento recién iniciado; en el laboratorio 3.5 el mismo proceso se repite y ahí sí se ve el mensaje final *Training job completed