import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Título y descripción de la aplicación
st.title("Visualización de Datos: Flotación")
st.write("""
Gráficas de como se comporta la mina a través de los años
""")

# Definir la función para cargar los datos
def loadData(path):
    return pd.read_excel(path)

# Ruta al archivo de Excel
PATH = "Datos/BD_Rec_Centinela.xlsx"

# Definir la función para generar boxplots
def generar_boxplot(data, columnas_interes, titulos, años_de_interés, palette="tab10"):
    # Asegúrate de que las fechas estén en formato DateTime
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    
    # Crea una copia del DataFrame original
    data_copia = data.copy()
    
    # Extrae el año y el mes de cada fecha y guárdalos en nuevas columnas "Año" y "Mes"
    data_copia['Año'] = data_copia['Fecha'].dt.year
    data_copia['Mes'] = data_copia['Fecha'].dt.month
    
    # Filtra los años de interés
    data_copia = data_copia[data_copia['Año'].isin(años_de_interés)]
    
    # Genera un boxplot para cada columna de interés
    for columna, titulo in zip(columnas_interes, titulos):
        st.subheader(titulo)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Mes', y=columna, hue='Año', data=data_copia, palette=palette)
        plt.title(titulo + " por año y mes")
        plt.xlabel('Mes')
        plt.ylabel(columna)
        plt.legend(title='Año')
        st.pyplot(plt)

# Cargar los datos
data = loadData(PATH)

# Calcular la columna 'Recuperación Lab' (si es necesario)
data["Recuperación Lab"] = (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100 * data["Rec_D1_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100 * data["Rec_D2_Lab_%"]/100 + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100 * data["Rec_D3_Lab_%"]/100 + \
                           data["Total_finos_Stock_tph"]* data["Rec_Stock_Lab_%"]/100) / \
                           (data["Total_Finos_mina_tph"] * data["Finos_D1_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D2_%"]/100  + \
                           data["Total_Finos_mina_tph"] * data["Finos_D3_%"]/100  + \
                           data["Total_finos_Stock_tph"])*100

# Seleccionar columnas de interés
columnas_interes = ['Total_tph', 'Ley_CuT_Mina_%', 'Recuperación Planta', 'P80_Op_um']
titulos = ['Tratamiento', 'Ley de Cobre por tonelada', 'Recuperación Planta', 'P80 en μM']
años_de_interés = [2017, 2018, 2019, 2021, 2022]

# Ejecutar la función para generar los boxplots
generar_boxplot(data, columnas_interes, titulos, años_de_interés)
