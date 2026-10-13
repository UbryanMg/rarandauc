# Laboratorio 3.5: Despliegue del modelo y predicciones

En este laboratorio tomé el modelo XGBoost del laboratorio anterior y lo usé de dos formas. Primero lo desplegué en un endpoint de SageMaker para hacer predicciones fila por fila. Después borré el endpoint y probé la transformación por lotes (batch transform) para predecir todo el conjunto de prueba de una vez. El trabajo se hizo en el cuaderno `3_5-machinelearning_es.ipynb`.

Abajo está cada paso con su captura, qué hice y por qué, y al final las respuestas a las preguntas del cuaderno.

---

## Paso 1. Configuración del laboratorio

![Bucket e importaciones](capturas/01_configuracion_imports.png)

**Qué hice:** definí el nombre del bucket de S3 de mi sesión e importé las librerías necesarias (pandas, scipy, boto3, sagemaker y scikit-learn).

**Por qué:** este laboratorio continúa los anteriores, así que el cuaderno necesita volver a cargar todo antes de poder desplegar nada. El bucket tiene que ser el de la sesión activa, porque cambia cada vez que se inicia el laboratorio.

## Paso 2. Cargar los datos y repetir la partición

![Carga de datos, partición y función de subida](capturas/02_configuracion_datos_y_particion.png)

**Qué hice:** ejecuté la celda de configuración, que descarga el dataset, convierte la clase a 0 y 1, mueve la columna objetivo al inicio, divide los datos en 80% / 10% / 10% con `train_test_split` y define la función `upload_s3_csv`.

**Por qué:** son los mismos pasos del laboratorio 3.4 agrupados en una sola celda. Hay que repetirlos porque el modelo que se va a desplegar necesita los datos ya preparados en S3, y porque la variable `test` se usa más adelante para las predicciones. Esta captura y la siguiente son una sola celda; la partí en dos para que se pueda leer el código.

## Paso 3. Subir los datos y reentrenar el modelo

![Subida a S3, contenedor, hiperparámetros, estimador y fit](capturas/03_configuracion_entrenamiento_codigo.png)

**Qué hice:** en la misma celda se suben los tres conjuntos a S3, se obtiene el contenedor de XGBoost, se definen los hiperparámetros (`num_round=42`, `auc`, `binary:logistic`), se configura el estimador con una instancia `ml.m4.xlarge`, se crean los canales de entrenamiento y validación y se llama a `fit`.

**Por qué:** el laboratorio se entrega con la configuración completa para que, aunque se empiece desde cero, exista un modelo entrenado listo para desplegar.

![Salida del entrenamiento](capturas/04_configuracion_entrenamiento_salida.png)

**Resultado:** el entrenamiento arrancó a las 21:10:46 y terminó a las 21:13:40 con el mensaje *Training job completed*, y el cuaderno imprimió `ready for hosting!`. Eso confirma que el modelo quedó listo para alojarse.

## Paso 4. Desplegar el modelo en un endpoint

![Despliegue del endpoint](capturas/05_despliegue_del_endpoint.png)

**Qué hice:** llamé a `xgb_model.deploy(...)` con una instancia `ml.m4.xlarge` y un `CSVSerializer`. SageMaker creó el modelo, la configuración del endpoint y el endpoint (`sagemaker-xgboost-2026-10-13-21-31-31-176`). Los guiones que se ven al final son el cuaderno esperando a que el endpoint quede activo, y el signo de exclamación indica que ya está listo.

**Por qué:** un endpoint deja el modelo "encendido" para recibir peticiones y devolver predicciones en tiempo real. Usé el `CSVSerializer` para poder enviarle los datos como texto separado por comas, que es el formato con el que se entrenó el modelo.

## Paso 5. Revisar los datos de prueba

![Datos de prueba](capturas/06_datos_de_prueba.png)

**Qué hice:** consulté `test.shape` y `test.head(5)`. El conjunto de prueba tiene 31 filas y 7 columnas, y se ven las primeras cinco con su clase real (la primera columna).

**Por qué:** antes de predecir quería recordar cómo son los datos y tener a mano la clase real para poder comparar con lo que responda el modelo.

## Paso 6. Predicción de la primera fila

![Predicción de la primera fila](capturas/07_prediccion_primera_fila.png)

**Qué hice:** tomé la primera fila del conjunto de prueba **sin la columna de clase** con `test.iloc[0:1,1:]`, la convertí a texto CSV en un búfer y la envié con `xgb_predictor.predict(test_row)`. La respuesta fue `0.9966`.

**Por qué:** la clase se quita porque es justo lo que el modelo debe adivinar. La respuesta no es un 0 ni un 1, sino una probabilidad: como el modelo usa `binary:logistic`, ese número es la probabilidad de que el paciente sea **anormal**. 0.9966 significa que el modelo está casi seguro de que es anormal.

## Paso 7. Validación con la entrada del profesor

![Validación con la entrada pedida](capturas/08_validacion_entrada_del_profesor.png)

**Qué hice:** envié al endpoint (`xgb_predictor`) esta entrada:

```
entrada = "63.0,12.5,45.0,50.5,125.0,3.0"
```

Los seis valores son, en orden: incidencia pélvica (63.0), inclinación pélvica (12.5), ángulo de lordosis lumbar (45.0), inclinación del sacro (50.5), radio pélvico (125.0) y grado de espondilolistesis (3.0).

**Resultado:** el endpoint devolvió `0.01694` (aproximadamente 0.0169).

**Qué lectura le doy al resultado:** ese número es la probabilidad de que el paciente sea anormal, o sea apenas un 1.7%. Como es mucho menor que 0.5, el modelo lo clasifica como **normal**, y con bastante seguridad. Tiene lógica con los datos de entrada, porque el grado de espondilolistesis es 3.0, un valor bajo, y en este conjunto de datos los pacientes con valores altos en esa medida son los que claramente salen anormales. Además, este resultado muestra que el modelo sí es capaz de responder "normal", aunque con los datos de prueba tendió a marcar casi todo como anormal (se ve más abajo en el paso del lote).

## Paso 8. Comparar la predicción con el dato real

![Comparación con los datos reales](capturas/09_comparacion_y_respuesta_exactitud.png)

**Qué hice:** volví a mostrar `test.head(5)` para comparar la predicción de la primera fila con su clase real, y contesté la pregunta del cuaderno (la respuesta está en la sección de preguntas, más abajo).

**Por qué:** una predicción solo tiene sentido si se contrasta con lo que realmente pasó. Aquí la primera fila tiene clase 1 y el modelo dio 0.9966, así que acertó.

## Paso 9. Eliminar el endpoint

![Eliminación del endpoint](capturas/10_eliminar_el_endpoint.png)

**Qué hice:** ejecuté `xgb_predictor.delete_endpoint(delete_endpoint_config=True)`, que borra el endpoint y también su configuración.

**Por qué:** un endpoint activo sigue consumiendo una instancia y generando costo aunque no se use. Por eso lo borré apenas terminé las predicciones individuales, y lo hice después de la validación del profesor, porque una vez borrado ya no se puede consultar.

## Paso 10. Preparar el conjunto de prueba para el lote

![Preparación del lote y subida a S3](capturas/11_preparar_lote_y_subir_a_s3.png)

**Qué hice:** con `test.iloc[:,1:]` tomé todas las filas de prueba sin la columna de clase y las subí a S3 como `batch-in.csv`.

**Por qué:** la transformación por lotes lee su entrada desde S3, y igual que antes la clase se quita porque es lo que se quiere predecir.

## Paso 11. Ejecutar la transformación por lotes

![Transformación por lotes](capturas/12_transformacion_por_lotes.png)

**Qué hice:** configuré las rutas de entrada y salida en S3, creé el transformador con una instancia `ml.m4.xlarge` (`MultiRecord`, ensamblado por línea) y ejecuté `transform` y `wait`. SageMaker creó el modelo y el trabajo de transformación (`sagemaker-xgboost-2026-10-13-21-42-21-164`). Los puntos son el cuaderno esperando a que termine.

**Por qué:** es una forma más cómoda que enviar filas una por una. SageMaker levanta una instancia con el modelo, predice todas las filas, guarda los resultados en S3 y apaga la instancia, así que no queda nada encendido que haya que acordarse de borrar.

## Paso 12. Leer los resultados del lote

![Resultados del lote](capturas/13_resultados_del_lote.png)

**Qué hice:** descargué el archivo de salida de S3 (`batch-in.csv.out`) y lo cargué en un DataFrame. Las primeras cinco filas dan 0.9966, 0.7773, 0.9946, 0.9937 y 0.9391.

**Por qué:** así puedo analizar las predicciones en pandas y compararlas con los datos reales. Un detalle que sirve de comprobación: la primera fila da 0.9966 tanto en el endpoint como en el lote, o sea que ambos métodos son consistentes.

## Paso 13. Convertir probabilidades en 0 o 1 y comparar

![Umbral y comparación con los datos reales](capturas/14_umbral_y_comparacion_con_datos_reales.png)

**Qué hice:** apliqué la función `binary_convert` con un umbral de 0.65: si la probabilidad es mayor, el resultado es 1 (anormal); si no, 0 (normal). Luego puse las predicciones junto a los datos reales de las primeras 10 filas.

**Por qué:** el modelo entrega probabilidades, pero para evaluarlo necesito una decisión (normal o anormal), y el umbral es la regla que la define.

Comparación de las 10 primeras filas (umbral 0.65):

| Fila (índice) | Probabilidad | Predicción | Clase real | ¿Acierta? |
|---|---|---|---|---|
| 136 | 0.9966 | 1 | 1 | Sí |
| 230 | 0.7773 | 1 | 0 | **No** |
| 134 | 0.9946 | 1 | 1 | Sí |
| 130 | 0.9937 | 1 | 1 | Sí |
| 47 | 0.9391 | 1 | 1 | Sí |
| 135 | 0.9974 | 1 | 1 | Sí |
| 100 | 0.9920 | 1 | 1 | Sí |
| 89 | 0.9875 | 1 | 1 | Sí |
| 297 | 0.9933 | 1 | 0 | **No** |
| 4 | 0.6828 | 1 | 1 | Sí |

El modelo acertó **8 de 10**.

---

## Respuestas a las preguntas del cuaderno

**¿Es exacta la predicción?**

Sí, en este caso acertó. La primera fila dio 0.9966, o sea que el modelo estaba casi seguro de que el paciente era anormal, y en los datos de prueba esa fila tiene clase 1, así que coincide. Igual, es solo un ejemplo, así que no se puede decir que el modelo sea exacto con una sola fila.

**Tarea de desafío: segunda fila y otras filas**

Revisé la segunda fila y varias más en el resultado del lote. En la segunda el modelo se equivocó: dio 0.777, que lo toma como anormal, pero en los datos esa fila es normal (clase 0). De las primeras 10 filas acertó 8. Los dos errores fueron pacientes normales (las filas 230 y 297) que el modelo marcó como anormales, y en la 297 con una probabilidad de 0.99. Parece que el modelo tiende a decir que casi todo es anormal.

**Tarea de desafío: cambiar el umbral, ¿afecta los resultados?**

Sí afecta, pero en este caso no mejora mucho. Estos son los aciertos en las 10 primeras filas con distintos umbrales:

| Umbral | Aciertos |
|---|---|
| 0.5 | 8 de 10 |
| 0.65 (el del cuaderno) | 8 de 10 |
| 0.7 | 7 de 10 |
| 0.8 | 8 de 10 |
| 0.95 | 7 de 10 |

Con 0.8 la fila 230 pasa a clasificarse bien como normal, pero la última fila (0.683), que sí es anormal, pasa a salir como normal, así que se gana una y se pierde otra. Con 0.95 empeora, porque empiezan a fallar pacientes anormales que el modelo sí tenía bien. El error de la fila 297 no se arregla cambiando el umbral porque el modelo le dio 0.99. Seguramente el modelo necesita ajustarse, como dice la nota del laboratorio. Estos conteos son solo de las 10 filas que muestra el cuaderno, no de las 31 de prueba.

