import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet

# Importo los datos
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

archivo_2020_1 = '../data/Saber_11__2020-1_20241024.csv'
archivo_2020_2 = '../data/Saber_11__2020-2_20241024.csv'
saber_2020_1 = pd.read_csv(archivo_2020_1)
saber_2020_2 = pd.read_csv(archivo_2020_2)
saber_2020 = pd.concat([saber_2020_1, saber_2020_2], ignore_index=True)

# Cambio Fecha a Edad 
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            return date_str

# Aplicar función a fecha nacimiento
saber_2020['ESTU_FECHANACIMIENTO'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(parse_date)

# Calcular edad
def calculate_age(birth_date):
    if isinstance(birth_date, pd.Timestamp):
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    else:
        return None

# Calcular edad usando la nueva función
saber_2020['AGE'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(calculate_age)

saber_2020.drop('ESTU_FECHANACIMIENTO', axis=1, inplace=True)

# Seleccionamos las columnas de interés
columns_of_interest = [
    'ESTU_NACIONALIDAD',
    'ESTU_GENERO',
    'AGE',
    'PERIODO',
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

# Dividimos columnas en categóricas y numéricas
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
    'AGE'
]

# Imputamos valores faltantes categóricos
categorical_imputer = SimpleImputer(strategy='most_frequent')
saber_2020_subset[categorical_columns] = categorical_imputer.fit_transform(saber_2020_subset[categorical_columns])

# Imputamos valores faltantes numéricos
numeric_imputer = SimpleImputer(strategy='mean')
saber_2020_subset[numeric_columns] = numeric_imputer.fit_transform(saber_2020_subset[numeric_columns])

# Encoding
saber_2020_encoded = pd.get_dummies(saber_2020_subset, columns=categorical_columns, drop_first=True)

# Quitamos columnas que tienen mucha colinealidad
columns_to_drop = ['ESTU_NACIONALIDAD_VENEZUELA', 'FAMI_PERSONASHOGAR_5 a 6', 'COLE_CALENDARIO_B']
saber_2020_encoded.drop(columns=columns_to_drop, inplace=True)

# Columnas numéricas que vamos a estandarizar
numeric_columns = ['AGE', 'PERIODO']

# Separar datos
numeric_data = saber_2020_encoded[numeric_columns]
categorical_data = saber_2020_encoded.drop(columns=numeric_columns)

# Estandarizar numéricas
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

# Rehacemos el dataframe
numeric_data_scaled_df = pd.DataFrame(numeric_data_scaled, columns=numeric_columns, index=saber_2020_encoded.index)
saber_2020_scaled = pd.concat([numeric_data_scaled_df, categorical_data], axis=1)

# MLflow
X = saber_2020_scaled
y = saber_2020['PUNT_GLOBAL']

y = y.values.reshape(-1, 1)

# Imputar valores faltantes en y
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y)
y = y.ravel()


# Inicializa el experimento de MLflow
mlflow.set_experiment("Experimientos_ElasticNet_Parametrizacion")

# Divide los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define los valores de alpha y l1_ratio para la búsqueda manual
alphas = [0.0001,0.001,0.01,0.1,1.0] # Valores de alpha desde 0.0001 a 1.0 
l1_ratios = [0.1, 0.5, 0.7, 0.9, 1.0]  # Valores comunes para l1_ratio

# Ciclo manual para probar cada combinación de alpha y l1_ratio
for alpha in alphas:
    for l1_ratio in l1_ratios:
        # Inicia una ejecución en MLflow para cada combinación
        with mlflow.start_run(run_name=f"ElasticNet_alpha_{alpha}_l1_ratio_{l1_ratio}"):
            
            # Configura y entrena el modelo ElasticNet
            elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            elastic_net.fit(X_train, y_train)
            
            # Realiza predicciones en el conjunto de prueba
            y_pred = elastic_net.predict(X_test)
            
            # Calcula las métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Registra los parámetros y métricas en MLflow
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            
            # Registra el modelo entrenado
            mlflow.sklearn.log_model(elastic_net, f"ElasticNet_Model_alpha_{alpha}_l1_ratio_{l1_ratio}")
            
            print(f"Registrado: alpha={alpha}, l1_ratio={l1_ratio}, MSE={mse}, R2={r2}")