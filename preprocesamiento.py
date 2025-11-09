# preprocesamiento.py
# Proyecto: preprocesamiento-cienciadatos
# Rama: feature-preprocesamiento
# Autor: Tu Nombre

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def cargar_datos(ruta_archivo):
    """Carga un dataset CSV en un DataFrame de Pandas."""
    return pd.read_csv(ruta_archivo)

def eliminar_duplicados(df):
    """Elimina filas duplicadas del dataset."""
    return df.drop_duplicates()

def manejar_nulos(df):
    """Rellena los valores nulos: usa la media para números y la moda para texto."""
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def codificar_categoricas(df):
    """Convierte variables categóricas en valores numéricos."""
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def normalizar_datos(df):
    """Normaliza los valores numéricos entre 0 y 1."""
    scaler = MinMaxScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
        df.select_dtypes(include=['float64', 'int64'])
    )
    return df

def preprocesar_dataset(ruta_archivo):
    """Pipeline completo de preprocesamiento."""
    df = cargar_datos(ruta_archivo)
    df = eliminar_duplicados(df)
    df = manejar_nulos(df)
    df = codificar_categoricas(df)
    df = normalizar_datos(df)
    return df
