import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

data.fillna(0, inplace=True)

data["Recuperación Lab"] = (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100 * data["Rec_D1_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100 * data["Rec_D2_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100 * data["Rec_D3_Lab_%"]/100 + \
                           data["Total_finos_Stock_tph"]* data["Rec_Stock_Lab_%"]/100) / \
                           (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100  + \
                           data["Total_finos_Stock_tph"])*100



data = data[['Fecha','Total_tph', 'Total_Mina_tph', 'Total_D1_tph', 'Total_D2_tph', 'Total_D3_tph', 'Total_Stock_tph',
             'Ley_CuT_Mina_%', 'Ley_CuT_Stock_%', 'P80_Op_um', 'Cp_Op_%', 'Flujo_Pulpa_m3/h', 'Rec_D1_Lab_%',
             'Rec_D2_Lab_%', 'Rec_D3_Lab_%', 'Rec_Stock_Lab_%', 'Recuperación Lab', 'Recuperación Planta']]

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


st.subheader("""Boxplot de los datos""")
st.write("""Boxplot es una excelente herramiente para este caso, donde se quiere analizar los datos
         por año y mes, donde en el eje y corresponde a los datos de la variable en cuestión,
         en el eje x los meses y los colores a los años""")


# Boxplot de los datos
data['Fecha'] = pd.to_datetime(data['Fecha'])
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

# Calcula la matriz de correlación
corr_matrix = data.corr()

# Establece el umbral de correlación
umbral = 0.3

# Filtra las columnas con correlación igual o mayor al umbral
high_corr_columns = corr_matrix[abs(corr_matrix["Recuperación Planta"]) >= umbral].index

# Crea dos DataFrames separados
matrix1 = data[high_corr_columns]
matrix2 = data.drop(columns=high_corr_columns)

# Asegúrate de que "Recuperación Planta" esté presente en ambos DataFrames
matrix1["Recuperación Planta"] = data["Recuperación Planta"]

# Gráfico de calor para matrix1
plt.figure(figsize=(10, 8))
sns.heatmap(matrix1.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación para valores mayores a |0.30| con recuperación planta')
plt.xticks(rotation=90)
plt.show()
st.pyplot(plt)

# Correlaciones Negativas (menos recuperación cuando aumentan)
st.write("**Correlación Negativa (Menos recuperación cuando aumentan)**:")
st.text("Fecha: -0.624182")
st.text("Total_tph: -0.376385")
st.text("Total_D1_tph: -0.460980")
st.text("Total_D2_tph: -0.310243")
st.text("P80_Op_um: -0.449103")
st.text("Cp_Op_%: -0.458388")
st.text("Recuperación Lab: -0.353013")

# Correlaciones Positivas (más recuperación cuando aumentan)
st.write("**Correlación Positiva (Más recuperación cuando aumentan):**")
st.text("Total_D3_tph: 0.319408")
st.text("Total_Stock_tph: 0.138807")
st.text("Ley_CuT_Mina_%: 0.442830")
st.text("Ley_CuT_Stock_%: 0.186082")
st.text("Flujo_Pulpa_m3/h: 0.154533")
st.text("Rec_Stock_Lab_%: 0.482631")

# Correlaciones Moderadas
st.write("**Correlaciones Moderadas:**")
st.text("Total_Mina_tph: -0.252791")
st.text("Rec_D1_Lab_%: -0.413985")
st.text("Rec_D2_Lab_%: -0.190564")
st.text("Rec_D3_Lab_%: 0.144485")

st.write("""Estas correlaciones indican cómo cada variable puede afectar la recuperación de cobre en la planta.
          Variables con correlaciones negativas podrían requerir ajustes para mejorar la eficiencia de recuperación,
          mientras que aquellas con correlaciones positivas pueden ser áreas clave para maximizar la eficiencia del proceso.""")

st.subheader("Reducción de dimensionalidad con PCA")
st.write("""
         PCA ayuda a identificar patrones subyacentes o estructuras latentes en tus datos.
          Esto es especialmente útil cuando quieres entender qué variables están contribuyendo más significativamente a la variabilidad observada en la recuperación de cobre.""")

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

columnas_prim = ['Total_tph', 'Ley_CuT_Mina_%', 'Ley_CuT_Stock_%', 'P80_Op_um',  'Recuperación Planta']
data_prim = data[columnas_prim]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_prim)
data_prepared = pd.DataFrame(data_scaled, columns=data_prim.columns, index=data_prim.index)

labels = data['Recuperación Planta'].copy()

pca = PCA(n_components = 2)
X2D = pca.fit_transform(data_prepared)

print("Variabilidad explicada con",len(pca.components_),"componentes:",pca.explained_variance_ratio_.sum())

x2d_data = pd.DataFrame(data = X2D
             , columns = ['principal component 1', 'principal component 2'])

import matplotlib.pyplot as plt

# Crear una figura y ejes para el gráfico
plt.figure(figsize=(10, 8))

# Colormap "jet" para asignar colores a las etiquetas
cmap = plt.get_cmap("jet")

# Graficar las componentes principales 1 y 2, coloreando por las etiquetas
scatter = plt.scatter(x2d_data['principal component 1'], x2d_data['principal component 2'], c=labels, cmap=cmap, marker='o')

# Agregar etiquetas a los ejes
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Título del gráfico
plt.title('Gráfico de Componentes Principales 1 vs 2')

# Agregar una barra de color (colorbar)
cbar = plt.colorbar(scatter)
cbar.set_label('Recuperación Planta', rotation=90)

# Obtener las direcciones de las componentes principales
components = pca.components_

# Obtener los nombres de las columnas originales
column_names = data_prepared.columns  # Reemplaza 'data' con el nombre de tu DataFrame original

# Dibujar los vectores de las componentes principales con etiquetas

    
for i, (comp1, comp2) in enumerate(zip(components[0, :], components[1, :])):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.7)
    plt.text(comp1, comp2, column_names[i], color='k', fontsize=10)
    

# Mostrar el gráfico
plt.grid(True)
plt.show()
st.pyplot(plt)

st.subheader("Impacto de la Ley de Cobre")
st.write("""
La ley de cobre es una de las variables más importantes en el proceso de recuperación de cobre. 
Un contenido más alto de cobre en la mena generalmente resulta en una mayor eficiencia de recuperación. 
Esto se debe a que una mayor ley de cobre proporciona más mineral valioso para ser procesado.
""")

st.subheader("Relación con el Tratamiento")
st.write("""
El tratamiento se refiere a la cantidad de mineral procesado en toneladas por hora (tph). 
Existe una relación inversa entre el tratamiento y la recuperación de cobre; 
cuando se incrementa la cantidad de material procesado, la eficiencia de recuperación tiende a disminuir.
Esto podría deberse a varios factores, como la sobrecarga de los equipos y la reducción del tiempo de residencia del mineral en las celdas de flotación.
""")

st.subheader("Efecto del P80")
st.write("""
El P80 es el tamaño de partícula al cual el 80% del material pasa a través de una malla. 
Un aumento en el valor de P80 generalmente está asociado con una disminución en la recuperación de cobre. 
Esto se debe a que partículas más grandes pueden no estar suficientemente liberadas, lo que reduce la eficiencia del proceso de flotación.
""")

st.header("Conclusiones y Observaciones Adicionales")
st.write("""
Estos hallazgos resaltan cómo diferentes variables impactan la eficiencia de la recuperación de cobre en el proceso analizado. 
Es crucial monitorear y optimizar estos parámetros para mejorar la eficiencia y rentabilidad del proceso de flotación.
Las gráficas anteriores proporcionan una visión clara de las relaciones entre las variables clave y la recuperación de cobre, 
lo cual puede guiar futuras investigaciones y optimizaciones en el proceso.
""")

