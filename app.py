import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Título y descripción de la aplicación
st.title("Visualización de Datos: Flotación")
st.write("""
Análisis de datos correspondiente a un proceso de flotación
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
st.write("Resumen Estadístico de los Datos:")
st.write(data.describe())


st.write("""
            Boxplot de las columnas de interes: \n
                - Tratamiento \n 
                - Ley de Cobre
""")
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




generar_boxplot("Total_tph", "Tratamiento")
st.write("""Se observa como el tratamiento en el año 2017 esta cerca de los 4500 toneladas por hora, y como ha aumentado a lo largo de los años
         hasta llego a las 5000 toneladas por hora en el año 2022""")


generar_boxplot("Ley_CuT_Mina_%", "Ley de Cobre")
st.write("""Se observa en azul para el año 2017, que la ley alacanza los mas altos valores, y los cuales van disminuyendo a través de los años, observandose una
         tendencia de bajos valores de ley en el año 2022""")

generar_boxplot("Recuperación Planta", "Recuperación de Cobre")
st.write("""Se observa que las mayores recuperaciones ocurren en los primeros años, para luego ver como disminuye año a año hasta el 2022, donde se regitran las menores recuperaciones""")

generar_boxplot("P80_Op_um", "P80 en μm")
st.write("""Se observa un aumento del valor del P80 a través de los años, cerca de los 180 μm en 2017 hasta sobre los 200 para el 2022""")


st.title("""Matriz de correlación de todas las variables con la variable Recuperación Planta""")


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

st.title("Interpretación de Correlaciones con Recuperación Planta")
st.write("""
Aquí se muestra la interpretación de las correlaciones con la recuperación en planta.
""")

# Correlaciones Negativas (menos recuperación cuando aumentan)
st.header("Correlación Negativa (Menos recuperación cuando aumentan):")
st.text("Fecha: -0.624182")
st.text("Total_tph: -0.376385")
st.text("Total_D1_tph: -0.460980")
st.text("Total_D2_tph: -0.310243")
st.text("P80_Op_um: -0.449103")
st.text("Cp_Op_%: -0.458388")
st.text("Recuperación Lab: -0.353013")

# Correlaciones Positivas (más recuperación cuando aumentan)
st.header("Correlación Positiva (Más recuperación cuando aumentan):")
st.text("Total_D3_tph: 0.319408")
st.text("Total_Stock_tph: 0.138807")
st.text("Ley_CuT_Mina_%: 0.442830")
st.text("Ley_CuT_Stock_%: 0.186082")
st.text("Flujo_Pulpa_m3/h: 0.154533")
st.text("Rec_Stock_Lab_%: 0.482631")

# Correlaciones Moderadas
st.header("Correlaciones Moderadas:")
st.text("Total_Mina_tph: -0.252791")
st.text("Rec_D1_Lab_%: -0.413985")
st.text("Rec_D2_Lab_%: -0.190564")
st.text("Rec_D3_Lab_%: 0.144485")

st.write("""Estas correlaciones indican cómo cada variable puede afectar la recuperación de cobre en la planta.
          Variables con correlaciones negativas podrían requerir ajustes para mejorar la eficiencia de recuperación,
          mientras que aquellas con correlaciones positivas pueden ser áreas clave para maximizar la eficiencia del proceso.""")

st.title("Reducción de dimensionalidad con PCA")

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

st.title("Análisis de Componentes Principales")
st.write("""
Este análisis complementa lo observado en los histogramas y proporciona información adicional sobre la recuperación de cobre.
""")

# Impacto de la Ley de Cobre
st.header("Impacto de la Ley de Cobre:")
st.write("""
Se observa que la ley de cobre es el factor más significativo para la recuperación de cobre; 
a mayor ley de cobre, mayor es la recuperación.
""")

# Relación con el Tratamiento
st.header("Relación con el Tratamiento:")
st.write("""
Existe una relación inversa entre el tratamiento y la recuperación de cobre; 
a mayor tratamiento, menor es la recuperación.
""")

# Efecto del P80
st.header("Efecto del P80:")
st.write("""
Se identifica que un aumento en el valor de P80 está asociado con una disminución en la recuperación de cobre.
""")

# Conclusiones y observaciones adicionales
st.header("Conclusiones y Observaciones Adicionales:")
st.write("""
Estos hallazgos resaltan cómo diferentes variables impactan la eficiencia de la recuperación de cobre en el proceso analizado.
""")

