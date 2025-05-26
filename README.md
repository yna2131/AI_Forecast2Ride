# AI_Forecast2Ride

| Nombre     | Matrícula |
| ---------- | --------- |
| Yuna Chung | A01709043 |

## Introducción

El uso de bicicletas públicas es una opción eficiente y ecológica para la movilidad en Seúl. Por la alta demanda, uno de los principales desafíos lde estos sistemass de bicicletas compartidas es garantizar la disponibilidad estable a lo largo del día, minimizando el tiempo de espera para los usuarios.

## Dataset

Este proyecto utiliza técnicas de Machine Learning para predecir la cantidad de bicicletas que se rentarán utilizando los datos de Seúl, Corea del Sur. El dataset contiene el número de bicicletas públicas rentada por hora en el sistema de distribución de bicicletas de Seúl, junto con los datos climáticos correspondientes y la información de vacaciones

### Tabla de Variables

| Nombre del Variable                               | Tipo          | Unidad |
| ------------------------------------------------- | ------------- | ------ |
| Date (Fecha)                                      | Fecha         | NA     |
| Rented Bike count (Número de Bicicletas Rentadas) | Número Entero | NA     |
| Hour (Hora)                                       | Número Entero | NA     |
| Temperature (Temperatura)                         | Continuo      | C      |
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
- Hour - Hour of he day
- Temperature-Temperature in Celsius
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
Desde el paquete de `sklearn.preprocessing` importé `OneHotEncoder` para el método de One-Hot Encoder y `LabelEncoder` el método de Label Encoder. Cada método fue realizada y después, concatené los datos para exportalos en un csv.
La razón por qué utilicé One-Hot Encoding para la columna "Seasons" fue principalmente para evitar asumir el orden de los valores. Como la columna tiene 4 diferentes estaciones que no tienen tal cual una jerarquía u orden lógico entre ellas, crear una columna binara por cada estación para identificar el valor del dato fue muy efectivo evitando introducir una relación de orden falsa entre las estaciones. Por otra parte, las columnas "Holiday" y "Functional Day" utilicé el método de Label Encoding ya que estos son cateogrías con solo dos valores posibles. No hay necesidad de separar las columnas cuando los valores son binarios.

### DataSplitting.ipyb

Después de limpiar el dataset, dividí los datos para poder tener un dataset para entrenar mi modelo y otro dataset para probar mi modelo. Utilizando `random`, escogí 80% de los datos de manera aleatoria utilizando `sample` y los exporté a un nuevo csv con puros datos para entrenamiento del modelo y el 20% restante fue exportado a un nuevo csv conteniendo solamente los datos que se utilizarán para probar el modelo.

## Referencias

[1] UCI Machine Learning Repository, “Seoul Bike Sharing Demand,” [Online]. Available: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand. [Accessed: May 12, 2025].
[2] X. Lin and C. Lu, “A Stacking-Based ensemble model for prediction of metropolitan bike sharing demand,” American Journal of Information Science and Technology, Apr. 2023, doi: 10.11648/j.ajist.20230702.13.
[3] T.-T. T. Ngo, H. T. Pham, J. G. Acosta, and S. Derrible, “Predicting Bike-Sharing demand using Random Forest,” Journal of Science and Transport Technology, pp. 13–21, May 2022, doi: 10.58845/jstt.utt.2022.en.2.13-21.