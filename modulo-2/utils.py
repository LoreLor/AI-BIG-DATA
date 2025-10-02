import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def import_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset importado exitosamente desde {file_path}")
        return df
    except Exception as e:
        print(f"Error al importar el dataset: {e}")
        return None


def outliers_iqr(df, columna_grupo, valor_grupo, columna_analisis):
    """
    Analiza outliers en una columna específica de un DataFrame usando el método del IQR,
    filtrando por una categoría específica en otra columna.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna_grupo (str): El nombre de la columna para filtrar categorías.
    valor_grupo (str): El valor específico de la categoría para filtrar.
    columna_analisis (str): El nombre de la columna donde se buscarán outliers.

    Retorna:
    pd.Series: Una serie booleana indicando la presencia de outliers.
    """
    datos_filtrados = df[df[columna_grupo] == valor_grupo][columna_analisis]
    Q1 = datos_filtrados.quantile(0.25)
    Q3 = datos_filtrados.quantile(0.75)
    IQR = Q3 - Q1
    left = Q1 - 1.5 * IQR
    right = Q3 + 1.5 * IQR
    outliers = (datos_filtrados < left) | (datos_filtrados > right)
    return outliers.sum()

def estadisticas_descriptivas(df, columna):
    """
    Calcula estadísticas descriptivas para una columna numérica en un DataFrame.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columna (str): El nombre de la columna numérica para analizar.

    Retorna:
    dict: Un diccionario con las estadísticas calculadas.
    """
    datos = df[columna]
    estadisticas = {
        'media': datos.mean(),
        'mediana': datos.median(),
        'moda': datos.mode()[0] if not datos.mode().empty else np.nan,
        'desviacion_estandar': datos.std(),
        'varianza': datos.var(),
        'minimo': datos.min(),
        'maximo': datos.max(),
        'rango': datos.max() - datos.min(),
        'skewness': skew(datos),
        'kurtosis': kurtosis(datos)
    }
    return estadisticas