import streamlit as st
import pandas as pd
from utils import *

st.title("Aplicación de un modelo de Machine Learning para la predicción de cobre")

st.markdown("""
## Análisis de Modelos de Machine Learning

En este análisis, se compararon varios modelos de machine learning con el objetivo de evaluar su rendimiento en la predicción de resultados basados en un conjunto de datos específico.
             Para esta comparación, se utilizó la métrica **Root Mean Squared Error (RMSE)**,
             que permite medir la precisión de las predicciones de cada modelo. Los modelos evaluados incluyen tanto regresiones lineales y regularizadas,
             como algoritmos de árboles de decisión y técnicas de boosting.

A continuación, se presentan los resultados de la evaluación, destacando la variabilidad y precisión de cada modelo.
""")


st.header('Comparación de Modelos de Machine Learning')

st.markdown("""
## Análisis de Modelos de Machine Learning

En este análisis, comparé varios modelos de machine learning utilizando la métrica de **Root Mean Squared Error (RMSE)** para evaluar su rendimiento. Los modelos evaluados fueron:

- **Regresión Lineal (LR)**
- **Lasso**
- **Ridge**
- **ElasticNet (ELTN)**
- **Decision Tree Regressor (DTR)**
- **Random Forest Regressor (RFR)**
- **AdaBoost Regressor (ABR)**
- **SGD Regressor (SGDR)**
- **Gradient Boosting Regressor (GBR)**
- **XGBoost (XGB)**
- **CatBoost (CBR)**

### Resultados

- **XGBoost** y **CatBoost** mostraron el mejor rendimiento con medianas de RMSE bajas y menor variabilidad, indicando consistencia y solidez en sus predicciones.
- **Gradient Boosting Regressor** también tuvo un buen desempeño, cercano al de XGBoost.
- **Regresión Lineal** mostró un rendimiento decente, pero con mayor variabilidad comparado con XGBoost.
- Los modelos **Lasso**, **Ridge**, y **ElasticNet** presentaron medianas de RMSE más altas y mayor variabilidad, sugiriendo que pueden no ser ideales para este conjunto de datos.
- **Decision Tree Regressor** tuvo una mediana de RMSE alta y alta variabilidad, indicando un rendimiento inconsistente.

### Conclusión

Dado su rendimiento superior y consistencia, se seleccionó **XGBoost** para la prueba final y ajuste de hiperparámetros, con el objetivo de optimizar su desempeño en este conjunto de datos específico.

A continuación se presenta un boxplot que resume los resultados de RMSE para cada modelo:
""")

PATH = "Datos/BD_Rec_Centinela.xlsx"

def loadData(path=PATH):
  return pd.read_excel(path)

data = loadData()

data["Recuperación Lab"] = (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100 * data["Rec_D1_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100 * data["Rec_D2_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100 * data["Rec_D3_Lab_%"]/100 + \
                           data["Total_finos_Stock_tph"]* data["Rec_Stock_Lab_%"]/100) / \
                           (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100  + \
                           data["Total_finos_Stock_tph"])*100


data = data[['Total_tph', 'Total_Mina_tph', 'Total_D1_tph', 'Total_D2_tph', 'Total_D3_tph', 'Total_Stock_tph',
             'Ley_CuT_Mina_%', 'Ley_CuT_Stock_%', 'P80_Op_um', 'Cp_Op_%', 'Flujo_Pulpa_m3/h', 'Rec_D1_Lab_%',
             'Rec_D2_Lab_%', 'Rec_D3_Lab_%', 'Rec_Stock_Lab_%', 'Recuperación Lab', 'Recuperación Planta']]

X_train, X_test, y_train, y_test = ProcesarDatos(data)


# results_df = ComparacionDeModelos(X_train , y_train)

# boxplot = alt.Chart(results_df).mark_boxplot().encode(
#     x=alt.X('Model:N', title='Model', sort=None),
#     y=alt.Y('RMSE:Q', title='RMSE')
# ).properties(
#     title='Comparación de modelos de machine learning',
#     width=650,  
#     height=600  
# ).configure_view(
#     stroke='transparent'  
# )

# st.altair_chart(boxplot)

ruta_imagen = 'imagenes/ComparacionModelos.PNG'
st.image(ruta_imagen, caption='Comparación de Modelos')

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
### Procesamiento de Datos para Modelado Predictivo

La función `ProcesarDatos` realiza varias transformaciones fundamentales en los datos antes de aplicar modelos de aprendizaje automático:

1. **Llenado de Valores Nulos**: Todos los valores nulos en el conjunto de datos se rellenan con ceros para asegurar la integridad de los datos.

2. **Escalado de Características**: Se utiliza el escalador `MinMaxScaler` para normalizar todas las características en un rango de 0 a 1. Esto es crucial para asegurar que las características tengan una escala comparable y no dominen artificialmente el modelo.

3. **Preparación de Datos**: Después del escalado, los datos se convierten nuevamente en un dataframe de pandas manteniendo la estructura original pero con características escaladas.

4. **División en Conjuntos de Entrenamiento y Prueba**: Los datos se dividen en conjuntos de entrenamiento y prueba en una proporción del 80% para entrenamiento y 20% para prueba. El conjunto de entrenamiento se utiliza para entrenar el modelo, mientras que el conjunto de prueba se reserva para evaluar su rendimiento. Esta división asegura que el modelo no esté sobreajustado y pueda generalizar bien a nuevos datos.

Esta función es esencial en el proceso de preparación de datos para proyectos de análisis predictivo, garantizando que los datos estén limpios, escalados y listos para ser utilizados eficazmente en modelos de aprendizaje automático.
""")

model_base = xgb.XGBRegressor()
model_base.fit(X_train, y_train)

y_pred_base = model_base.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

st.markdown("### Rendimiento del Modelo Base")
st.write(f"RMSE del modelo base: {rmse_base:.2f}")

params = {
    'learning_rate': [0.01, 0.05],
    'n_estimators': [300, 500, 600],
    'max_depth': [3, 5],
    'colsample_bytree': [0.6, 0.8],
    'subsample': [0.6, 0.7]
}

grid_search = GridSearchCV(estimator=model_base, param_grid=params,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
st.markdown("### Optimización de Hiperparámetros")
st.write("Mejores parámetros encontrados:", best_params)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

st.markdown("### Rendimiento del Mejor Modelo")
st.write(f"RMSE del mejor modelo: {rmse_best:.2f}")

rendimiento_pct = ((rmse_base - rmse_best) / rmse_base) * 100
st.markdown("### Mejora del Modelo")
st.write(f"Mejora porcentual del modelo es: {rendimiento_pct:.2f}%")

st.markdown("""
### Conclusión
El proceso de optimización de hiperparámetros resultó en una mejora significativa del rendimiento del modelo. 
La reducción en RMSE y la mejora porcentual indican que el modelo optimizado es más preciso en sus predicciones, 
lo que destaca la importancia de la optimización de hiperparámetros en los modelos de machine learning.
""")


st.markdown("### Gráfica de Predicciones vs Valores Reales")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_best, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Valores Reales')
ax.set_ylabel('Predicciones')
ax.set_title('Comparación de Predicciones vs Valores Reales')
st.pyplot(fig)

st.markdown("""
## Análisis del Rendimiento del Modelo

El gráfico muestra un rendimiento razonablemente bueno del modelo, con predicciones cercanas a los valores reales y una buena captura de la tendencia de los datos. 

### Puntos a Considerar

Sin embargo, hay errores de predicción que podrían investigarse más a fondo. Se podrían considerar:

- Ajustes en los hiperparámetros.
- Incorporación de nuevos datos.
- Mejor uso de las variables disponibles.""")
