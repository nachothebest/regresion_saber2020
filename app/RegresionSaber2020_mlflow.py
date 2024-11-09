
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import warnings
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


# Importo los datos

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

archivo_2020_1 = '../data/Saber_11__2020-1_20241024.csv'
archivo_2020_2 = '../data/Saber_11__2020-2_20241024.csv'
saber_2020_1 = pd.read_csv(archivo_2020_1)
saber_2020_2 = pd.read_csv(archivo_2020_2)
saber_2020 = pd.concat([saber_2020_1, saber_2020_2], ignore_index=True)

from datetime import datetime


# Cambio Fecha a Edad 
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            return date_str

# Aplicar funcion a fecha nacimiento
saber_2020['ESTU_FECHANACIMIENTO'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(parse_date)

# Cacular edad
def calculate_age(birth_date):
    if isinstance(birth_date, pd.Timestamp):
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    else:
        return None

# Calculate age using the new function
saber_2020['AGE'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(calculate_age)

saber_2020.drop('ESTU_FECHANACIMIENTO', axis=1, inplace=True)

# Seleccionamos las columnas de interés
columns_of_interest = [
    'ESTU_NACIONALIDAD',
    'ESTU_GENERO',
    'AGE',
    'PERIODO',
    'ESTU_TIENEETNIA',
    'ESTU_COD_RESIDE_DEPTO',
    'FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR',
    'FAMI_EDUCACIONPADRE',
    'FAMI_EDUCACIONMADRE',
    'FAMI_TIENEINTERNET',
    'FAMI_TIENECONSOLAVIDEOJUEGOS',
    'FAMI_NUMLIBROS',
    'ESTU_DEDICACIONLECTURADIARIA',
    'ESTU_DEDICACIONINTERNET',
    'ESTU_HORASSEMANATRABAJA',
    'COLE_CODIGO_ICFES',
    'COLE_GENERO',
    'COLE_NATURALEZA',
    'COLE_CALENDARIO',
    'COLE_CARACTER',
    'COLE_AREA_UBICACION',
    'COLE_JORNADA',
    'COLE_COD_DEPTO_UBICACION'
]

# Dividimos columnas en categóricas y numericas
saber_2020_subset = saber_2020[columns_of_interest]

categorical_columns = [
    'ESTU_NACIONALIDAD',
    'ESTU_GENERO',
    'ESTU_TIENEETNIA',
    'FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR',
    'FAMI_EDUCACIONPADRE',
    'FAMI_EDUCACIONMADRE',
    'FAMI_TIENEINTERNET',
    'FAMI_TIENECONSOLAVIDEOJUEGOS',
    'FAMI_NUMLIBROS',
    'ESTU_DEDICACIONLECTURADIARIA',
    'ESTU_DEDICACIONINTERNET',
    'ESTU_HORASSEMANATRABAJA',
    'COLE_GENERO',
    'COLE_NATURALEZA',
    'COLE_CALENDARIO',
    'COLE_CARACTER',
    'COLE_AREA_UBICACION',
    'COLE_JORNADA'
]

numeric_columns = [
    'PERIODO',
    'AGE',
    'ESTU_COD_RESIDE_DEPTO',
    'COLE_CODIGO_ICFES',
    'COLE_COD_DEPTO_UBICACION'
]


# Imputamos valores faltante categoricos
categorical_imputer = SimpleImputer(strategy='most_frequent')
saber_2020_subset[categorical_columns] = categorical_imputer.fit_transform(saber_2020_subset[categorical_columns])

# Imputamos valores faltante numericos
numeric_imputer = SimpleImputer(strategy='mean')
saber_2020_subset[numeric_columns] = numeric_imputer.fit_transform(saber_2020_subset[numeric_columns])


# convertimos las variavles categóricas en dummies, transformando en columnas nuevas con 0 y 1.
saber_2020_encoded = pd.get_dummies(saber_2020_subset, columns=categorical_columns, drop_first=True)

# quitamos columnas que tienen mucha colinealidad
columns_to_drop = ['COLE_CODIGO_ICFES', 'ESTU_NACIONALIDAD_VENEZUELA', 'FAMI_PERSONASHOGAR_5 a 6', 'COLE_CALENDARIO_B']

# aplicamos cambios al df
saber_2020_encoded.drop(columns=columns_to_drop, inplace=True)

# Columnas numericas que vamos a estandarizar
numeric_columns = ['AGE', 'PERIODO', 'ESTU_COD_RESIDE_DEPTO', 'COLE_COD_DEPTO_UBICACION']

# Separamos los datos para no cometer errores
numeric_data = saber_2020_encoded[numeric_columns]
categorical_data = saber_2020_encoded.drop(columns=numeric_columns)

# estandarizar numericas
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

# Rehacemos el dataframe
numeric_data_scaled_df = pd.DataFrame(numeric_data_scaled, columns=numeric_columns, index=saber_2020_encoded.index)
saber_2020_scaled = pd.concat([numeric_data_scaled_df, categorical_data], axis=1)
