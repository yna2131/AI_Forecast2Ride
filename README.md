# AI_Forecast2Ride

| Nombre     | Matrícula |
| ---------- | --------- |
| Yuna Chung | A01709043 |

## Abstract

Este proyecto presenta un modeo de predicción de la demanda de bicicletas públicas utilizando técnicas de Machine Learning. 
Se describe el proceso de preprocesamiento de datos, incluyendo encoding, splitting y scaling, así como la selección de variables relevantes mediante análisis de correlación. 
Se implementan y comparan modelos de regresión, específicamente Random Forest y Redes Neuronales, evaluando su desempeño con métricas como MAE, RMSE y R². 
Los resultados muestran que el modelo Random Forest, tras la optimización de variables, ofrece la mejor precisión predictiva. 
El proyecto concluye que el modelo desarrollado tiene un nivel de precisión aceptable para estimar la demanda de bicicletas en función de variables climáticas y temporales, 
contribuyendo a la gestión eficiente del sistema de bicicletas compartidas en la ciudad de Seúl.

## Introducción

El uso de bicicletas públicas es una opción eficiente y ecológica para la movilidad en Seúl. Por la alta demanda, uno de los principales desafíos de estos sistemas de bicicletas compartidas es garantizar la disponibilidad estable a lo largo del día, minimizando el tiempo de espera para los usuarios.

## Dataset

Este proyecto utiliza técnicas de Machine Learning para predecir la cantidad de bicicletas que se rentarán utilizando los datos de Seúl, Corea del Sur. El dataset contiene el número de bicicletas públicas rentada por hora en el sistema de distribución de bicicletas de Seúl, junto con los datos climáticos correspondientes y la información de vacaciones.

### Tabla de Variables

| Nombre del Variable                               | Tipo          | Unidad |
| ------------------------------------------------- | ------------- | ------ |
| Date (Fecha)                                      | Fecha         | NA     |
| Rented Bike count (Número de Bicicletas Rentadas) | Número Entero | NA     |
| Hour (Hora)                                       | Número Entero | NA     |
| Temperature (Temperatura)                         | Continuo      | C°     |
| Humidity (Humedad)                                | Número Entero | %      |
| Wind Speed (Velocidad del Viento)                 | Continuo      | m/s    |
| Visibility (Visibilidad)                          | Número Entero | 10m    |
| Dew Point Temperature (Punto de Rocío)            | Continuo      | C      |
| Solar Radiation (Radiación Solar)                 | Continuo      | Mj/M2  |
| Rainfall (Lluvias)                                | Número Entero | mm     |
| Snowfall (Nevadas)                                | Número Entero | cm     |
| Seasons (Estación)                                | Categoría     | NA     |
| Holiday (Día Festival)                            | Binario       | NA     |
| Functioning Day (Día Funcional)                   | Binario       | NA     |

### Información Adicional de Variables

- Date : year-month-day
- Rented Bike count - Count of bikes rented at each hour
- Hour - Hour of the day
- Temperature - Temperature in Celsius
- Humidity - %
- Windspeed - m/s
- Visibility - 10m
- Dew point temperature - Celsius
- Solar radiation - MJ/m2
- Rainfall - mm
- Snowfall - cm
- Seasons - Winter, Spring, Summer, Autumn
- Holiday - Holiday/No holiday
- Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)

## Preprocesamiento

### Encoding.ipynb

Para poder predecir la cantidad de bicicletas que serán rentadas de acuerdo al contexto del día, es importante que los datos que estoy utilizando sean numéricos. Por esta razón, después de verificar que no hay datos vacíos en el dataset utilicé el método de One-Hot Encoding para la columna "Seasons" y Label Encoding para las columnas "Holiday" y "Functional Day". 
Desde el paquete de `sklearn.preprocessing` importé `OneHotEncoder` para el método de One-Hot Encoder y `LabelEncoder` el método de Label Encoder. Cada método fue realizado y después, concatené los datos para exportalos en un csv.
La razón por la cual utilicé One-Hot Encoding para la columna "Seasons" es porque la columna tiene 4 diferentes estaciones que no tienen tal cual una jerarquía u orden lógico entre ellas, crear una columna binara por cada estación para identificar el valor del dato fue muy efectivo evitando introducir una relación de orden falsa entre las estaciones. Por otra parte, para las columnas "Holiday" y "Functional Day" utilicé el método de Label Encoding ya que estas son cateogrías con solo dos valores posibles. Por lo que no hay necesidad de separar las columnas cuando los valores son binarios.

### DataSplitting.ipyb

Después de limpiar el dataset, dividí los datos para poder tener un dataset para entrenar mi modelo y otro dataset para probar mi modelo. Utilizando `random`, escogí 80% de los datos de manera aleatoria utilizando `sample` y los exporté a un nuevo csv con puros datos para entrenamiento del modelo y el 20% restante fue exportado a un nuevo csv conteniendo solamente los datos que se utilizarán para probar el modelo.

### Scaling.ipynb

Una vez que los datos ya estaban dividos para el entrenamiento y prueba, continué con el escalamiento de los datos utilizando `StandardScaler` de `sklearn.preprocessing` ya que la escala de mis datos era muy grande. Primero, eliminé la columna de `Date` de mis datos ya que las fechas no es parte de los features relevantes que quiero utilizar para hacer las predicciones.
Después, dividí los datos otra vez en `features (x)` y `target (y)`. Las caracterísicas (x) son todos esos valores que va a usar el modelo para hacer predicciones, y la variable objetivo (y) es lo que el modelo tiene que predecir, la cual es la cantidad de bicicletas rentadas. 
Una vez que los datos estuvieron nuevamente divididos, creé el escalador con `StandardScaler` de scikit-learn. Para los datos de `x_train`, utilicé `fit_transform()` para calcular la media y desviación estándar de cada columna y al final usar esos valores calculados para escalar los datos. En caso de `x_test`, solamente utilicé `transform()` para escalar los datos usando la media y desviación del entrenamiento pero sin `fit` porque esto causaría que use información del test para entrenar mi modelo.
Finalmente, exporté los datos de entrenamiento y pruebas escalados, la columna objetivo de train y test sin escalar a diferentes archivos csv.

## Modelo

### Heatmap.ipynb

Como tengo muchos features independientes, decidí evaluar la correlación entre estas categorías para visualizar la relevancia de cada uno de ellos con el target, que es la cantidad de bicicletas rentadas ya que los features irrelevantes afectan la exactitud del modelo [2].
Utilicé `seaborn` y `matplotlib.pyplot` para graficar este heat map. Eliminé la columna de fechas la cual consideré irrelevantes desde la fase de escalamiento de los datos, y utilicé `corr()` para obtener la correlación entre las columnas. 
Podemos observar que la hora y la temperatura fueron las columnas con más correlación con la cantidad de bicicletas rentadas.

<p align="center">
    <image src="./Images/image-4.png" alt="Heatmap de Correlación">
    <br>
    <em> Fig 1. Heatmap de Correlación</em>
</p>

### ModelTraining.ipynb

Utilizando `RandomForestRegressor` de `sklearn.ensemble`, entrené mi modelo con el algoritmo de **Random Forest**.
Para los datos objetivos que no están escalados, utilicé `values.ravel()` para convertir mi DataFrame en un arreglo NumPy unidimensional.
Después, creé mi modelo `RandomForestRegressor` usando scikit-learn. El parámetro `n_estimators` es para indicar el número de árboles en el bosque, y el `random_state` sirve para que mi modelo obtenga el mismo resultado siempre.
`y_pred` es el arreglo con las predicciones del modelo que se generó con los datos de prueba. Después, calculé el **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)** y **R²** usando `sklearn.metrics`. 
Mi modelo obtuvo 140.97 de MAE, 233.63 de RMSE y 0.87 de R². Generé 2 gráficas utililzando `matplotlib.pyplot` para ver qué tan separados están los valores predichos de los valores actuales.

<p align="center">
    <image src="./Images/image-5.png" alt="Actual vs Predicted">
    <br>
    <em> Fig 2. Comparación de datos reales vs datos predichos</em>
</p>

La figura 2 muestra todos los valores predichos y reales al mismo tiempo. La línea roja es lo que se considera la predicción
perfecta, y mientras más separadas de esta, más error tiene mi modelo. Como se puede observar, hay puntos muy separados de otros,
pero en general, todos los puntos están muy cerca de la línea roja.

<p align="center">
    <image src="./Images/image-6.png" alt="Actual vs Predicted - One Entry per Hour">
    <br>
    <em> Fig 3. Comparación de datos reales vs datos predichos - una entrada por hora</em>
</p>

En el caso de esta gráfica (figura 3), se recibe una entrada por hora para que sea más fácil visualizar la comparación.
Así es mucho más fácil observar y analizar los datos. En general, los valores predichos y actuales no están muy lejos,
sin embargo, el resultado de la gráfica es distinto cada vez que se corre el modelo ya que utilicé `randint()` y nada más se muestra una
parte muy pequeña del resultado. Hay veces donde las predicciones están muy lejos de los datos reales, y otras donde la predicción es muy similar al valor esperado.
Entonces esto ocasiona que sea complicado visualizar cómo se está desempeñando mi modelo en general.

#### Uso de Datos Externos

Ya con un modelo funcional, intenté generar una predicción utilizando los datos externos. El usuario da las variables independientes,
y el modelo predice la cantidad de bicicletas que se rentarán de acuerdo a los datos recibidos.
Al principio, estaba obteniendo predicciones bastantes cercanas a los valores esperados (ex. Si el valor real era 251, la predicción fue 227), pero me dí cuenta que mi dataset tenía una columna
extra de índice que pensé que había eliminado en las fases anteriores. Por esta razón, regresé a limpiar mis datos y entrenar de nuevo mi modelo para probar la predicción con los datos externos.
Una vez que esta columna fue eliminada, la predicción empeoró bastante porque la columna de índice introducía ruido que estaba haciendo un tipo de "trampa" a mi modelo. Antes, el modelo estaba obteniendo el R² de 0.88, pero después del ajuste, el modelo obtuvo el R² de 0.87 
lo cual no parece ser un gran cambio, pero afecta bastante en las predicciones. Aunque parece que empeoré mi modelo, al final, esto fue una mejora ya que eliminé el ruido que no me estaba dando una predicción más precisa y real.

## Mejorando el modelo

Como estoy utilizando muchos features para mi modelo:

1. Hora
2. Temperatura
3. Humedad
4. Velocidad del viento
5. Visibilidad
6. Punto de rocío
7. Radiación solar
8. Lluvias
9. Nevadas
10. Estación (Primavera, Verano, Otoño, Invierno)
11. Día Festival (Sí o No)
12. Día Funcional (Sí o No)

Y en teoría, al momento de eliminar algunos features que tiene una correlación débil, el modelo puede mejorar su rendimento [2].
Por esta razón, intenté eliminar algunos features que parecen estar muy relacionados con otros. Primero, intenté eliminar
punto de rocío el cual se relaciona con la temperatura de acuerdo a mi heatmap de correlación, y radiación solar que se relaciona bastante con
la temperatura y humedad. Originalmente, mi modelo tuvo 140.97 de MAE, 233.63 de RMSE y 0.87 de R²; y al eliminarlos, el resultado de las predicciones fue mucho más impreciso obteniendo los valores de MAE y RMSE más altos, 
y R² mucho más bajo que antes. Entonces, intenté solamente eliminando el punto de rocío que es un dato muy similar al de la temperatura, y esta vez, 
los resultados mejoraron con el MAE que bajó a 140.83, RMSE a 233.47 y el R² que se mantiene en 0.87.

<p align="center">
    <image src="./Images/image-1.png" alt="Random Forest (all features)">
    <br>
    <em> Fig 4. Comparación de los primeros 100 resultados utilizando todos los features</em>
</p>

<p align="center">
    <image src="./Images/image-7.png" alt="Random Forest (without Dew Point Temperature)">
    <br>
    <em> Fig 5. Comparación de los primeros 100 resultados sin punto de rocío</em>
</p>

Si comparamos la figura 4 y 5 a detalle, se percibe una diferencia aunque en general, es bastante similar. Hay predicciones que empeoran un poco más que antes, pero también hay predicciones que mejoran; podemos observar que los puntos
que estaban empalmados quedaron aun más empalmados, lo cual nos explica por qué el MAE y RMSE bajaron. El error bajó al momento de quitar el punto de rocío desde los features, aunque fue por muy poco.

### NeuralNetwork.ipynb

Ya que el rendimiento de mi modelo utilizando Random Forest no fue como yo esperaba, intenté comparar los resultados
con un modelo de **Neural Network**. Utilicé `tf.keras.Sequential`, con dos capas densas con la activación relu 
y una capa densa con 1 para obtener la salida numérica para el modelo de regresión. Entrené el modelo utilizando los datos de entrenamiento durante 100 épocas,
y al evaluar, obtuve el loss de 194.56 y el accuracy de 0.029. Sin embargo, como es un modelo de regresión que predice el valor de la salida de acuerdo a los valores de entrada, el accuracy no tiene mucha relevancia porque la probabilidad
de que una predicción sea exactamente igual que el valor real es muy baja.
Este modelo tuvo 194.56 de MAE, 315.91 de RMSE y 0.76 de R².

### Comparación de Modelos

#### Random Forest

<p align="center">
    <image src="./Images/image-1.png" alt="Random Forest (100 entries)">
    <br>
    <em> Fig 6. Comparación de los primeros 100 resultados de Random Forest</em>
</p>

<p align="center">
    <image src="./Images/image.png" alt="Random Forest (All entries)">
    <br>
    <em> Fig 7. Comparación de todos los resultados de Random Forest</em>
</p>

#### Deep Neural Network

<p align="center">
    <image src="./Images/image-2.png" alt="DNN (100 entries)">
    <br>
    <em> Fig 8. Comparación de los primeros 100 resultados de DNN</em>
</p>

<p align="center">
    <image src="./Images/image-3.png" alt="DNN (All entries)">
    <br>
    <em> Fig 9. Comparación de todos los resultados de DNN</em>
</p>

Para los ambos modelos, creé un scatter plot que muestra los valores de salida esperados en color azul, y las predicciones del modelo con color rojo.
Las primeras dos son de Random Forest; la figura 6 siendo el scatter plot de las primeras 100 entradas, 
y la figura 7 siendo la versión con todas las entradas del dataset. Las mismas gráficas se muestran para la sección de Neural Network en figura 8 y 9.
Como se puede observar en la figura 6 y figura 8, hay predicciones que son muy cercanas a los datos reales pero al mismo tiempo hay predicciones que están muy lejos de lo esperado.
Aunque la diferencia entre los resultados de los dos modelos parece ser muy similares al observar las gráficas y sobre todo los scatter plots que muestran todas las predicciones del dataset, pude analizar que el modelo
que utiliza Random Forest tiene mejor rendimiento que el modelo que utiliza Neural Network. Si nos enfocamos en las gráficas de 100 entradas, los puntos de predicción del modelo con Random Forest están más cercas 
de los resultados esperados que en el modelo de Neural Network.

## Conclusión

Después de comparar diferentes modelos, llegué a la conclusión que el modelo de Random Forest sin el punto de rocío obtuvo el mejor resultado. Aunque no fue el modelo óptimo para realizar predicciones ya que mi modelo tuvo 
el 140.83 de MAE, 233.47 de RMSE y 0.87 de R², cuando el modelo que encontré en el estado de arte obtuvo 121 de MAE, 210 de RMSEy 0.9 de R [3].

En conclusión, mi modelo no tuvo el mejor rendimiento, pero creo que puede ser utilizado para un sistema que predice el número de bicicletas que se rentarán de acuerdo a los datos climáticos de entrada ya que siendo un modelo de regresión, con un valor de R² mayor a 0.8 se considera aceptable indicando que una gran proporción de la varianza en la variable dependiente es explicada por las variables independientes.

## Referencias

[1] UCI Machine Learning Repository, “Seoul Bike Sharing Demand,” [Online]. Available: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand. [Accessed: May 12, 2025].

[2] X. Lin and C. Lu, “A Stacking-Based ensemble model for prediction of metropolitan bike sharing demand,” American Journal of Information Science and Technology, Apr. 2023, doi: 10.11648/j.ajist.20230702.13.

[3] T.-T. T. Ngo, H. T. Pham, J. G. Acosta, and S. Derrible, “Predicting Bike-Sharing demand using Random Forest,” Journal of Science and Transport Technology, pp. 13–21, May 2022, doi: 10.58845/jstt.utt.2022.en.2.13-21.
