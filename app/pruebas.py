import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import warnings
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np

# Configuración para ignorar advertencias
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Carga y procesamiento de datos
archivo_2020_1 = '../data/Saber_11__2020-1_20241024.csv'
archivo_2020_2 = '../data/Saber_11__2020-2_20241024.csv'
saber_2020_1 = pd.read_csv(archivo_2020_1)
saber_2020_2 = pd.read_csv(archivo_2020_2)
saber_2020 = pd.concat([saber_2020_1, saber_2020_2], ignore_index=True)

# Convertir fechas a edades
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            return date_str

saber_2020['ESTU_FECHANACIMIENTO'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(parse_date)

def calculate_age(birth_date):
    if isinstance(birth_date, pd.Timestamp):
        today = pd.Timestamp.now()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    else:
        return None

saber_2020['AGE'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(calculate_age)
saber_2020.drop('ESTU_FECHANACIMIENTO', axis=1, inplace=True)

# Columnas de interés y procesamiento de datos
columns_of_interest = [
    'ESTU_NACIONALIDAD', 'ESTU_GENERO', 'AGE', 'PERIODO', 'ESTU_TIENEETNIA',
    'ESTU_COD_RESIDE_DEPTO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 
    'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 
    'FAMI_TIENECONSOLAVIDEOJUEGOS', 'FAMI_NUMLIBROS', 'ESTU_DEDICACIONLECTURADIARIA',
    'ESTU_DEDICACIONINTERNET', 'ESTU_HORASSEMANATRABAJA', 'COLE_CODIGO_ICFES', 
    'COLE_GENERO', 'COLE_NATURALEZA', 'COLE_CALENDARIO', 'COLE_CARACTER', 
    'COLE_AREA_UBICACION', 'COLE_JORNADA', 'COLE_COD_DEPTO_UBICACION'
]
saber_2020_subset = saber_2020[columns_of_interest]

categorical_columns = [
    'ESTU_NACIONALIDAD', 'ESTU_GENERO', 'ESTU_TIENEETNIA', 'FAMI_ESTRATOVIVIENDA', 
    'FAMI_PERSONASHOGAR', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 
    'FAMI_TIENEINTERNET', 'FAMI_TIENECONSOLAVIDEOJUEGOS', 'FAMI_NUMLIBROS', 
    'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_DEDICACIONINTERNET', 'ESTU_HORASSEMANATRABAJA',
    'COLE_GENERO', 'COLE_NATURALEZA', 'COLE_CALENDARIO', 'COLE_CARACTER',
    'COLE_AREA_UBICACION', 'COLE_JORNADA'
]
numeric_columns = ['PERIODO', 'AGE', 'ESTU_COD_RESIDE_DEPTO', 'COLE_CODIGO_ICFES', 'COLE_COD_DEPTO_UBICACION']

categorical_imputer = SimpleImputer(strategy='most_frequent')
saber_2020_subset[categorical_columns] = categorical_imputer.fit_transform(saber_2020_subset[categorical_columns])
numeric_imputer = SimpleImputer(strategy='mean')
saber_2020_subset[numeric_columns] = numeric_imputer.fit_transform(saber_2020_subset[numeric_columns])

saber_2020_encoded = pd.get_dummies(saber_2020_subset, columns=categorical_columns, drop_first=True)
columns_to_drop = ['COLE_CODIGO_ICFES', 'ESTU_NACIONALIDAD_VENEZUELA', 'FAMI_PERSONASHOGAR_5 a 6', 'COLE_CALENDARIO_B']
saber_2020_encoded.drop(columns=columns_to_drop, inplace=True)

numeric_data = saber_2020_encoded[numeric_columns]
categorical_data = saber_2020_encoded.drop(columns=numeric_columns)
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

numeric_data_scaled_df = pd.DataFrame(numeric_data_scaled, columns=numeric_columns, index=saber_2020_encoded.index)
saber_2020_scaled = pd.concat([numeric_data_scaled_df, categorical_data], axis=1)

# Modelo de MLflow
X = saber_2020_scaled
y = saber_2020['PUNT_GLOBAL']
y = SimpleImputer(strategy='mean').fit_transform(y.values.reshape(-1, 1)).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlflow.set_tracking_uri('http://localhost:5000')
experiment = mlflow.set_experiment("Saber_2020")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    la = Lasso(alpha=0.0001)
    la.fit(X_train, y_train)
    predictions = la.predict(X_test)
    mlflow.log_param("alpha", 0.0001)
    mlflow.sklearn.log_model(la, "lasso")
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)

# Diccionario de nombres amigables
friendly_names = {
    "COLE_JORNADA_SABATINA": "Sabatina",
    "COLE_JORNADA_NOCHE": "Nocturna",
    "COLE_GENERO_MIXTO": "Colegio Mixto",
    "COLE_JORNADA_TARDE": "Jornada Tarde",
    "COLE_JORNADA_MAÑANA": "Jornada Mañana",
    "FAMI_EDUCACIONPADRE_Postgrado": "Padre con Postgrado",
    "FAMI_EDUCACIONMADRE_Postgrado": "Madre con Postgrado"
}

# Aplicación Dash
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

app.layout = html.Div(
    className="container mt-4",
    children=[
        html.H1("Relationship Between Features and PUNT_GLOBAL", className="text-center mb-4"),
        
        html.Div(className="row", children=[
            html.Div(className="col-md-4", children=[
                html.Label("Select a Feature:", className="font-weight-bold"),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": friendly_names.get(feature, feature), "value": feature} for feature in friendly_names.keys()],
                    value=list(friendly_names.keys())[0],
                    className="mb-4"
                ),
            ]),
            html.Div(className="col-md-8", children=[
                dcc.Graph(id="feature-plot"),
            ]),
        ]),
        
        html.Div(id="feature-info", className="mt-4", style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'border': '1px solid #ddd'})
    ]
)

@app.callback(
    [Output("feature-plot", "figure"),
     Output("feature-info", "children")],
    [Input("feature-dropdown", "value")]
)
def update_graph(selected_feature):
    filtered_data = saber_2020_encoded[selected_feature]
    
    fig = px.box(filtered_data, x=selected_feature, y="PUNT_GLOBAL", title=f"Box Plot of {friendly_names.get(selected_feature, selected_feature)}")
    fig.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="PUNT_GLOBAL",
        template="plotly_white"
    )

    description
