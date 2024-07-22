import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
import multiprocessing
from sklearn.metrics import mean_squared_error
from utils import *

# Título y descripción de la aplicación
st.title("Flotación Predictiva: Aplicación de Machine Learning para Predecir la Recuperación de Cobre")

st.markdown("""### Objetivo

**Optimización de un modelo de recuperación primaria a través de técnicas de Machine Learning.**

#### Objetivos Específicos

- Identificación de los distintos procesos metalúrgicos.
- Manejo de datos con técnicas de Data Science.
- Utilización de algoritmos de Machine Learning.
- Optimización de modelos.
""")

st.header("""
En que consiste la flotación
         """)
st.write("""
         El proceso de flotación es un método utilizado en la industria minera para separar minerales valiosos de otros materiales menos útiles.
          Se basa en la capacidad de ciertos minerales de adherirse a burbujas de aire en una suspensión acuosa. Al introducir aire en la suspensión,
          los minerales valiosos se adhieren a las burbujas y forman una espuma que se recoge en la parte superior,
          mientras que los materiales no deseados permanecen en la parte inferior como cola. Es un proceso fundamental en la concentración de minerales como el cobre,
          zinc, plomo y otros.""")

st.write(""" Los datos usados corresponden de una minera que contiene información de los años 2017, 2018, 2019, 2021, 2022. A continuación se describen 
         las variables que se utilizaran""")

st.markdown("""### Descripción de Variables

- **Fecha:** Fecha en la que se registraron los datos.

- **Total_tph:** Flujo másico entrante a la etapa de flotación primaria en toneladas por hora.

- **Total_Mina_tph:** Alimentación fresca de la mina en toneladas por hora.

- **Total_D1_tph:** Mineral de baja ley procesado en toneladas por hora.

- **Total_D2_tph:** Mineral de ley media procesado en toneladas por hora.

- **Total_D3_tph:** Mineral de alta ley procesado en toneladas por hora.

- **Total_Stock_tph:** Mineral de stock de baja ley procesado en toneladas por hora.

- **Ley_CuT_Mina_%:** Ley de cobre en el mineral proveniente de la mina, expresada en porcentaje.

- **Ley_CuT_Stock_%:** Ley de cobre en el mineral proveniente del stock, expresada en porcentaje.

- **P80_Op_um:** Tamaño de partícula P80 después de la operación de molienda, en micrómetros.

- **Cp_Op_%:** Porcentaje de capacidad operativa en la planta.

- **Flujo_Pulpa_m3/h:** Flujo de pulpa en metros cúbicos por hora.

- **Rec_D1_Lab_%:** Eficiencia de recuperación de cobre en la etapa de laboratorio para mineral de baja ley, expresada en porcentaje.

- **Rec_D2_Lab_%:** Eficiencia de recuperación de cobre en la etapa de laboratorio para mineral de ley media, expresada en porcentaje.

- **Rec_D3_Lab_%:** Eficiencia de recuperación de cobre en la etapa de laboratorio para mineral de alta ley, expresada en porcentaje.

- **Rec_Stock_Lab_%:** Eficiencia de recuperación de cobre en la etapa de laboratorio para mineral de stock, expresada en porcentaje.

- **Recuperación Lab:** Eficiencia global de recuperación de cobre en las etapas de laboratorio, expresada en porcentaje.

- **Recuperación Planta:** Eficiencia global de recuperación de cobre en la planta de procesamiento, expresada en porcentaje.
""")

# Definir la función para cargar los datos
def loadData(path):
    return pd.read_excel(path)

# Ruta al archivo de Excel
PATH = "Datos/BD_Rec_Centinela.xlsx"

data = loadData(PATH)

valores_nulos = data.isnull().sum()
st.write("Contamos los valores nulos de cada columna")
st.dataframe(valores_nulos)

st.write("La naturaleza de los datos permite rellenar los nulos con el valor 0")
data.fillna(0, inplace=True)
st.dataframe(data.isnull().sum())



# data["Recuperación Lab"] = (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100 * data["Rec_D1_Lab_%"]/100 + \
#                            data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100 * data["Rec_D2_Lab_%"]/100 + \
#                            data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100 * data["Rec_D3_Lab_%"]/100 + \
#                            data["Total_finos_Stock_tph"]* data["Rec_Stock_Lab_%"]/100) / \
#                            (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100  + \
#                            data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100  + \
#                            data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100  + \
#                            data["Total_finos_Stock_tph"])*100



# data = data[['Fecha','Total_tph', 'Total_Mina_tph', 'Total_D1_tph', 'Total_D2_tph', 'Total_D3_tph', 'Total_Stock_tph',
#              'Ley_CuT_Mina_%', 'Ley_CuT_Stock_%', 'P80_Op_um', 'Cp_Op_%', 'Flujo_Pulpa_m3/h', 'Rec_D1_Lab_%',
#              'Rec_D2_Lab_%', 'Rec_D3_Lab_%', 'Rec_Stock_Lab_%', 'Recuperación Lab', 'Recuperación Planta']]

# Mostrar el resumen estadístico
st.subheader("Resumen Estadístico de los Datos:")
st.write(data.describe())

st.subheader("Análisis de Estadísticas Descriptivas del Proceso de Flotación de Cobre")

st.markdown("""
    - El **Total_tph** tambien llamado **Tratamiento** muestra una media de 4684 toneladas por hora, y una máxima de 
            5987
    - La variable **Ley_CuT_Mina_%** o **Ley de cobre**, tiene una media de 0.71 y alcanza un máximo de 2.0
    - La media del **P80** es de 195 μm y tuvo un valor máximo de 270 μm
    - la **Recuperación de cobre** tiene una media de 88, una mínima de 62 y una máxima de 98
                    """)

st.subheader("Histograma de las variables")
fig, ax = plt.subplots(figsize=(30,30))
data.hist(ax=ax, bins = 25)
st.pyplot(fig)

st.write("""
**Total_tph**: Indica las toneladas por hora que entran al proceso, mostrando un pequeño sesgo a la derecha. ¿Tendrá relación con la recuperación de cobre?

**Total_mina_tph**: Representa el mineral proveniente de la mina. Muestra una buena distribución de valores, pero con muchos valores nulos, lo que indica que cuando es 0, el mineral proviene del stock.

**Total_Stock_tph**: Representa el aporte del stock. Tiene una buena distribución, aunque generalmente aporta menos que el mineral directo de la mina.

**Aporte_Mina_%**: Al ser un porcentaje del total de mineral, tiene una distribución similar a Total_mina_tph.

**Total_Stock_%**: Similar al caso anterior, muestra cuándo el aporte es del 100%. ¿Por qué a veces el mineral proviene completamente del stock?

**Ley_CuT_Mina_%**: El porcentaje de cobre en el mineral. Idealmente, debería ser alto, pero el histograma muestra una tendencia a valores bajos.

**Ley_CuT_Stock_%**: El porcentaje de cobre en el stock es aún más bajo que en el mineral de la mina.

**Total_Finos_mina_tph**: Tiene una distribución similar a Ley_CuT_Mina_%, lo cual es lógico ya que los finos se calculan directamente a partir de la ley de cobre y el tratamiento. $$ Finos = Tratamiento * \dfrac{Ley de cobre}{100}$$

**P80_Op_um**: En minería, es común tener una granulometría cercana a los 200 um. Aquí, los valores oscilan entre 150 y 250 um.

**Cp_Op_%**: Muestra una buena distribución con valores similares, pero hay una notable diferencia entre los valores de 35 y 37.

**Flujo Pulpa $m^3/h$**: Tiene una buena distribución con un leve sesgo a la izquierda, concentrándose alrededor de los 9000 $m^3/h$.

**Recuperación Planta**: Una medida crítica que se busca mantener lo más cercana posible al 100%. Normalmente, las recuperaciones rondan el 90%.
""")


st.subheader("""Boxplot de los datos""")
st.write("""Boxplot es una excelente herramiente para este caso, donde se quiere analizar los datos
         por año y mes, donde en el eje y corresponde a los datos de la variable en cuestión,
         en el eje x los meses y los colores a los años""")


# Boxplot de los datos
data['Fecha'] = pd.to_datetime(data['Fecha'])
# data['Fecha'] = data['Fecha'].astype(str)
data_copia = data.copy()
data_copia['Año'] = data_copia['Fecha'].dt.year
data_copia['Mes'] = data_copia['Fecha'].dt.month
data_copia = data_copia[data_copia['Año'].isin([2017,2018,2019,2021,2022])]

def generar_boxplot(columna, titulo):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Mes', y=columna, hue='Año', data=data_copia, palette='tab10')
    plt.title(titulo + " " + "por año y mes")
    plt.xlabel('Mes')
    plt.ylabel(columna)
    plt.legend(title='Año')
    plt.show()
    st.pyplot(plt)



st.write("""Se observa como el tratamiento en el año 2017 esta cerca de los 4500 toneladas por hora, y como ha aumentado a lo largo de los años
         hasta llego a las 5000 toneladas por hora en el año 2022.""")

generar_boxplot("Total_tph", "Tratamiento")

st.write("""Se observa en azul para el año 2017, que la ley alacanza los mas altos valores, y los cuales van disminuyendo a través de los años, observandose una
         tendencia de bajos valores de ley en el año 2022""")

generar_boxplot("Ley_CuT_Mina_%", "Ley de Cobre")

st.write("""Se observa que las mayores recuperaciones ocurren en los primeros años, para luego ver como disminuye año a año hasta el 2022, donde se regitran las menores recuperaciones""")

generar_boxplot("Recuperación Planta", "Recuperación de Cobre")

st.write("""Se observa un aumento del valor del P80 a través de los años, cerca de los 180 μm en 2017 hasta sobre los 200 para el 2022""")

generar_boxplot("P80_Op_um", "P80 en μm")

st.subheader("Resumen de los boxplot")
st.markdown("""
- La ley promedio ha disminuido del 0.92(%) al 0.48(%) entre 2017 y 2022.
- Esta disminución en la ley está asociada con una reducción en la recuperación promedio, que ha caído del 93(%) al 81(%) en el mismo periodo.
- Para compensar, el tratamiento promedio ha aumentado de 4341 tph en 2017 a 4946 tph en 2022 (un aumento del 13%).
- Se observa que este aumento promedio en el tratamiento ha incrementado el P80 de 177 μm a 217 μm entre 2017 y 2022, lo cual podría resultar en una menor recuperación.
""")


st.header("""Matriz de correlación de todas las variables con la variable Recuperación Planta""")
st.write("""
La matriz de correlación permite identificar y visualizar las relaciones lineales entre las variables
y la "Recuperación Planta". Esto es importante para entender qué variables podrían estar influenciando 
directamente la eficiencia del proceso de flotación.""")


corr_matrix = data.corr()
umbral = 0
high_corr_columns = corr_matrix[abs(corr_matrix["Recuperación Planta"]) >= umbral].index

matrix1 = data[high_corr_columns]
matrix2 = data.drop(columns=high_corr_columns)

matrix1["Recuperación Planta"] = data["Recuperación Planta"]

plt.figure(figsize=(10, 8))
sns.heatmap(matrix1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación para valores mayores a |0.30| con recuperación planta')
plt.xticks(rotation=90)
plt.show()
st.pyplot(plt)

st.write("""Estas correlaciones indican cómo cada variable puede afectar la recuperación de cobre en la planta.
          Variables con correlaciones negativas podrían requerir ajustes para mejorar la eficiencia de recuperación,
          mientras que aquellas con correlaciones positivas pueden ser áreas clave para maximizar la eficiencia del proceso.""")




st.header("Predicción de la recuperación de Cobre, a través de un modelo de Machine Learning XGBoost")

st.write("""Concluido el análisis de los, procedemos a crear un modelo de machine learning para
predicción de cobre
""")

data_ml = data.drop("Fecha", axis = 1)
X_train, X_test, y_train, y_test = ProcesarDatos(data_ml)

st.subheader("Modelo Base de XGBRegressor")

code = """
model_base = XGBRegressor()
model_base.fit(X_train, y_train)

y_pred = model_base.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Las predicciones de este modelo se alejan en promedio {rmse:.4f} 
de los valores reales')
"""
st.code(code, language='python')

model_base = XGBRegressor()
model_base.fit(X_train, y_train)

y_pred = model_base.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f'Las predicciones de este modelo se alejan en promedio {rmse:.4f} de los valores reales')

st.write("""
Ocupar un modelo base quiere decir que estamos usando los hiperparametros por defecto del modelo.

Al ser los hiperparámetros base, puede que el rendimiento del modelo no sea el mejor posible,
para eso podemos buscar los hiperparámetros que mejoren nuestro modelo.
         
El siguiente código define un diccionario `param_grid` que contiene una serie de hiperparámetros y sus posibles valores para la búsqueda de hiperparámetros en un modelo de `XGBoost`. Los hiperparámetros especificados son:

- `max_depth`: Profundidad máxima del árbol de decisión (valores posibles: None, 1, 3, 5, 10, 20).
- `subsample`: Proporción de muestras usadas para entrenar cada árbol (valores posibles: 0.5, 1).
- `learning_rate`: Tasa de aprendizaje para ajustar el modelo (valores posibles: 0.001, 0.01, 0.1).
- `booster`: Tipo de booster a utilizar en el modelo (valor posible: 'gbtree').

""")


code = """
param_grid = {'max_depth'        : [None, 1, 3, 5, 10, 20],
              'subsample'        : [0.5, 1],
              'learning_rate'    : [0.001, 0.01, 0.1],
              'booster'          : ['gbtree']
             }
"""

st.code(code, language='python')

param_grid = {'max_depth'        : [None, 1, 3, 5, 10, 20],
              'subsample'        : [0.5, 1],
              'learning_rate'    : [0.001, 0.01, 0.1],
              'booster'          : ['gbtree']
             }

st.write("""
El siguiente código realiza una partición de los datos de entrenamiento `X_train` y `y_train` para crear un conjunto de validación y ajustar los parámetros de entrenamiento del modelo de `XGBoost`.

1. Se establece una semilla aleatoria (`np.random.seed(123)`) para garantizar la reproducibilidad.
2. Se selecciona aleatoriamente el 10% de los datos de entrenamiento para crear un conjunto de validación (`idx_validacion`).
3. Se extraen las muestras correspondientes del conjunto de validación de `X_train` y `y_train` (`X_val` y `y_val`).
4. Se actualizan los conjuntos de entrenamiento (`X_train_grid` y `y_train_grid`) excluyendo las muestras seleccionadas para validación.
5. Se definen los parámetros de ajuste (`fit_params`) para el modelo de `XGBoost`, especificando el conjunto de validación y configurando la opción de salida detallada (`verbose`).

Este proceso ayuda a evaluar el rendimiento del modelo durante el entrenamiento utilizando un conjunto de validación.

""")

code = """
np.random.seed(123)
idx_validacion = np.random.choice(
                    X_train.shape[0],
                    size=int(X_train.shape[0]*0.1), #10% de los datos de entrenamiento 
                    replace=False
                 )

X_val = X_train.iloc[idx_validacion, :].copy()
y_val = y_train.iloc[idx_validacion].copy()

X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()

fit_params = {
              "eval_set": [(X_val, y_val)],
              "verbose": False
             }
"""
st.code(code, language='python')

np.random.seed(123)
idx_validacion = np.random.choice(
                    X_train.shape[0],
                    size=int(X_train.shape[0]*0.1), #10% de los datos de entrenamiento 
                    replace=False
                 )

X_val = X_train.iloc[idx_validacion, :].copy()
y_val = y_train.iloc[idx_validacion].copy()

X_train_grid = X_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()
y_train_grid = y_train.reset_index(drop = True).drop(idx_validacion, axis = 0).copy()


fit_params = {
              "eval_set": [(X_val, y_val)],
              "verbose": False
             }

st.write("""
Este código configura y ajusta un modelo de `XGBoost` utilizando `GridSearchCV` para la búsqueda de hiperparámetros:

1. Se crea un objeto `GridSearchCV` con los siguientes parámetros:
   - `estimator`: Un modelo `XGBRegressor` con 1000 estimadores, parada temprana después de 5 rondas, métrica de evaluación `rmse`, y semilla aleatoria 123.
   - `param_grid`: Un diccionario que define los hiperparámetros a probar (`param_grid`).
   - `scoring`: La métrica de evaluación utilizada es el error cuadrático medio negativo (`neg_root_mean_squared_error`).
   - `n_jobs`: Número de trabajos paralelos, ajustado al número de núcleos de CPU menos uno.
   - `cv`: Estrategia de validación cruzada con `RepeatedKFold` usando 3 particiones y 1 repetición.
   - `refit`: Ajusta el modelo final con el mejor conjunto de hiperparámetros encontrado.
   - `verbose`: Nivel de detalle de los mensajes durante el ajuste (0 significa sin mensajes).
   - `return_train_score`: Incluye la puntuación de entrenamiento en los resultados.

2. Se ajusta el objeto `GridSearchCV` a los datos de entrenamiento (`X_train_grid`, `y_train_grid`) utilizando los parámetros de ajuste (`fit_params`).

Este proceso busca los mejores hiperparámetros para el modelo de `XGBoost` y evalúa su rendimiento utilizando el conjunto de validación.

""")

code = """grid = GridSearchCV(
        estimator  = XGBRegressor(
                        n_estimators          = 1000,
                        early_stopping_rounds = 5,
                        eval_metric           = "rmse",
                        random_state          = 123
                    ),
        param_grid = param_grid,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits=3, n_repeats=1, random_state=123), 
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid.fit(X = X_train_grid, y = y_train_grid, **fit_params)"""

st.code(code, language='python')


grid = GridSearchCV(
        estimator  = XGBRegressor(
                        n_estimators          = 1000,
                        early_stopping_rounds = 5,
                        eval_metric           = "rmse",
                        random_state          = 123
                    ),
        param_grid = param_grid,
        scoring    = 'neg_root_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits=3, n_repeats=1, random_state=123), 
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid.fit(X = X_train_grid, y = y_train_grid, **fit_params) 



# st.write("""
# Este código procesa y visualiza los resultados de la búsqueda de hiperparámetros realizada con `GridSearchCV`:

# 1. Se crea un `DataFrame` de `pandas` a partir de los resultados de la validación cruzada (`grid.cv_results_`).
# 2. Se filtran las columnas del `DataFrame` para incluir solo las relacionadas con los parámetros (`param.*`), la puntuación media de la prueba (`mean_test_score`), y la desviación estándar de la puntuación de la prueba (`std_test_score`).
# 3. Se eliminan las columnas no necesarias (`params`).
# 4. Se ordenan los resultados en función de la puntuación media de la prueba (`mean_test_score`) en orden descendente.
# 5. Se muestran las 4 mejores combinaciones de hiperparámetros.

# Este proceso facilita la identificación de las mejores configuraciones de hiperparámetros y sus correspondientes puntuaciones de rendimiento.

# """)
# code = """
# resultados = pd.DataFrame(grid.cv_results_)
# resultados.filter(regex = '(param.*|mean_t|std_t)') \
#     .drop(columns = 'params') \
#     .sort_values('mean_test_score', ascending = False) \
#     .head(4)
# """

# st.code(code, language='python')

# resultados = pd.DataFrame(grid.cv_results_)
# resultados.filter(regex = '(param.*|mean_t|std_t)') \
#     .drop(columns = 'params') \
#     .sort_values('mean_test_score', ascending = False) \
#     .head(4)

# st.dataframe(resultados)

# st.write("""
# Este código imprime información sobre los mejores hiperparámetros encontrados y el número de árboles en el modelo ajustado:

# 1. Se imprime la mejor combinación de hiperparámetros encontrada por `GridSearchCV` (`grid.best_params_`), junto con la puntuación correspondiente (`grid.best_score_`) y la métrica de evaluación utilizada (`grid.scoring`).
# 2. Se determina el número de árboles incluidos en el modelo ajustado (`grid.best_estimator_`), accediendo a la información del modelo mediante `get_booster().get_dump()`.
# 3. Se imprime el número de árboles incluidos en el modelo.

# Este proceso permite evaluar la mejor configuración de hiperparámetros y obtener información sobre la complejidad del modelo ajustado.
# """)

# code = """
# print("Mejores hiperparámetros encontrados (cv)")
# print(grid.best_params_, ":", grid.best_score_, grid.scoring)


# n_arboles_incluidos = len(grid.best_estimator_.get_booster().get_dump())
# print(f"Número de árboles incluidos en el modelo: {n_arboles_incluidos}")
# """
# st.code(code, language='python')

# print("Mejores hiperparámetros encontrados (cv)")
# print(grid.best_params_, ":", grid.best_score_, grid.scoring)


# n_arboles_incluidos = len(grid.best_estimator_.get_booster().get_dump())
# print(f"Número de árboles incluidos en el modelo: {n_arboles_incluidos}")

# st.write(f'Mejores hiperparámetros encontrados (cv) {grid.best_params_} : {grid.best_score_}, {grid.scoring} \n Número de árboles incluidos en el modelo: {n_arboles_incluidos}')

# st.write("""
# Este código evalúa el modelo final ajustado y calcula su error en el conjunto de prueba:

# 1. Se obtiene el mejor modelo ajustado (`modelo_final`) de los resultados de `GridSearchCV` (`grid.best_estimator_`).
# 2. Se realizan predicciones sobre el conjunto de prueba (`X_test`) utilizando el modelo final.
# 3. Se calcula el error cuadrático medio (RMSE) entre las etiquetas verdaderas (`y_test`) y las predicciones (`predicciones`) utilizando `mean_squared_error`, configurado para devolver la raíz cuadrada del error cuadrático medio (`squared=False`).
# 4. Se imprime el valor del RMSE en el conjunto de prueba.

# Este proceso proporciona una medida de rendimiento del modelo en datos no vistos, evaluando su capacidad de generalización.

# """)

# code = """
# modelo_final = grid.best_estimator_
# y_pred = modelo_final.predict(X_test)
# rmse = mean_squared_error(
#         y_true  = y_test,
#         y_pred  = y_pred,
#         squared = False
#        )
# print(f"El error (rmse) de test es: {rmse}")
# """

# st.code(code, language='python')

# modelo_final = grid.best_estimator_
# y_pred = modelo_final.predict(X_test)
# rmse = mean_squared_error(
#         y_true  = y_test,
#         y_pred  = y_pred,
#         squared = False
#        )
# print(f"El error (rmse) de test es: {rmse}")

# st.write(f"El error (rmse) de test es: {rmse}")
# st.write(f'Las predicciones de este modelo se alejan en promedio {rmse} de los valores reales, una mejora respecto del rmse del modelo base de 3.0160')



# st.subheader("Gráfica de las predicciones")

# df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='y_test', y='y_pred', data=df)
# plt.xlabel('Valores Reales')
# plt.ylabel('Valores Predichos')
# plt.title('Valores Reales vs. Predichos')
# plt.plot([df['y_test'].min(), df['y_test'].max()], [df['y_test'].min(), df['y_test'].max()], color='red', linestyle='--')
# # plt.show()
# st.pyplot(plt)

