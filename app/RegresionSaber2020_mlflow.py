
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

# Encoding

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

#mlflow

X = saber_2020_scaled

y = saber_2020['PUNT_GLOBAL']

y = y.values.reshape(-1, 1)

# Use an imputer to fill NaN values in y (e.g., with the mean of y)
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y)

# Convert back to a 1D array if necessary for your model
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y)

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn


# defina el servidor para llevar el registro de modelos y artefactos
mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("Saber_2020")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    alpha = 0.0001
    # Cree el modelo con los parámetros definidos y entrénelo
    la = Lasso(alpha=alpha)
    la.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = la.predict(X_test)
  
    # Registre los parámetros
    mlflow.log_param("alpha", alpha)

  
    # Registre el modelo
    mlflow.sklearn.log_model(la, "lasso")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)

# Define relevant features
relevant_features = [
    "COLE_JORNADA_SABATINA",
    "COLE_JORNADA_NOCHE",
    "COLE_GENERO_MIXTO",
    "COLE_JORNADA_TARDE",
    "COLE_JORNADA_MAÑANA",
    "COLE_JORNADA_UNICA"
    "FAMI_ESTRATOVIVIENDA_Sin Estrato",
    "FAMI_NUMLIBROS_26 A 100 LIBROS",
    "FAMI_NUMLIBROS_MÁS DE 100 LIBROS", 
    "ESTU_TIENEETNIA_Si", 
    "FAMI_ESTRATOVIVIENDA_Sin Estrato",
    "FAMI_EDUCACIONPADRE_Postgrado",
    "FAMI_EDUCACIONMADRE_Postgrado",
    "FAMI_EDUCACIONMADRE_No Aplica",
    "FAMI_EDUCACIONMADRE_Primaria incompleta", 
    "ESTU_DEDICACIONLECTURADIARIA_Más de 2 horas",
    "FAMI_EDUCACIONMADRE_Ninguno",
    "FAMI_TIENEINTERNET_Si" 
]

saber_2020_encoded['PUNT_GLOBAL'] = saber_2020['PUNT_GLOBAL']

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout for the app
app.layout = html.Div([
    html.H1("Relationship Between Relevant Features and [PUNT_GLOBAL]"),
    
    # Dropdown to select feature
    html.Label("Select a Feature:"),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": feature, "value": feature} for feature in relevant_features],
        value=relevant_features[0],  # Default value
    ),
    
    # Graph output
    dcc.Graph(id="feature-plot"),
    
    # Display additional information
    html.Div(id="feature-info", style={'padding': '20px'})
])

# Callback to update the graph based on the selected feature
@app.callback(
    [Output("feature-plot", "figure"),
     Output("feature-info", "children")],
    [Input("feature-dropdown", "value")]
)
def update_graph(selected_feature):
    # Check if the selected feature is categorical or numeric
    if saber_2020_encoded[selected_feature].dtype == 'object' or saber_2020_encoded[selected_feature].nunique() < 10:
        # Use box plot for categorical features
        fig = px.box(
            saber_2020_encoded,
            x=selected_feature,
            y="PUNT_GLOBAL",
            title=f"Box Plot of {selected_feature} vs. [PUNT_GLOBAL]"
        )
        description = f"The box plot shows the distribution of [PUNT_GLOBAL] for each category of {selected_feature}."
    else:
        # Use scatter plot for numerical features
        fig = px.scatter(
            saber_2020_encoded,
            x=selected_feature,
            y="PUNT_GLOBAL",
            trendline="ols",
            title=f"Scatter Plot of {selected_feature} vs. [PUNT_GLOBAL]"
        )
        description = f"The scatter plot shows the relationship between {selected_feature} and [PUNT_GLOBAL]."
    
    # Style the plot
    fig.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="[PUNT_GLOBAL]",
        template="plotly_white"
    )
    
    return fig, description

# Run the app
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False)
