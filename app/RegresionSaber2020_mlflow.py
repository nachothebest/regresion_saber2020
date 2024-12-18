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

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Configuración de MLflow
mlflow.set_tracking_uri('http://localhost:5000')
experiment = mlflow.set_experiment("Saber_2020")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Parámetros del modelo
    alpha = 0.001
    la = Lasso(alpha=alpha)
    la.fit(X_train, y_train)
    predictions = la.predict(X_test)
  
    # Registrar parámetros y métricas
    mlflow.log_param("alpha", alpha)
    mlflow.sklearn.log_model(la, "lasso")
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)

# Cálculo de número total de observaciones
total_observations = len(saber_2020_encoded)

# Agregamos el puntaje global al dataframe de encoded para comparaciones
saber_2020_encoded['PUNT_GLOBAL'] = saber_2020['PUNT_GLOBAL']
coefficients = la.coef_

# Creo un DataFrame para asociar cada nombre de las variables con su coeficiente
feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

# Defino mis variables relevantes, aquellas que tienen un mayor coeficiente 
relevant_features_original = [
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
    'COLE_JORNADA',
    'PERIODO',
    'AGE'
]

relevant_features_encoded = [
    "COLE_JORNADA_SABATINA",
    "COLE_JORNADA_NOCHE",
    "COLE_GENERO_MIXTO",
    "COLE_JORNADA_TARDE",
    "COLE_JORNADA_MAÑANA",
    "COLE_JORNADA_UNICA",
    "FAMI_ESTRATOVIVIENDA_Sin Estrato",
    "FAMI_NUMLIBROS_26 A 100 LIBROS",
    "FAMI_NUMLIBROS_MÁS DE 100 LIBROS", 
    "ESTU_TIENEETNIA_Si", 
    "FAMI_EDUCACIONPADRE_Postgrado",
    "FAMI_EDUCACIONMADRE_Postgrado",
    "FAMI_EDUCACIONMADRE_No Aplica",
    "FAMI_EDUCACIONMADRE_Primaria incompleta", 
    "ESTU_DEDICACIONLECTURADIARIA_Más de 2 horas",
    "FAMI_EDUCACIONMADRE_Ninguno",
    "FAMI_TIENEINTERNET_Si" 
]

# Mapeo de nombres amigables para variables originales
friendly_names = {
    'ESTU_NACIONALIDAD': 'Nacionalidad',
    'ESTU_GENERO': 'Género del Estudiante',
    'AGE': 'Edad',
    'PERIODO': 'Periodo',
    'ESTU_TIENEETNIA': 'Tiene Etnia',
    'FAMI_ESTRATOVIVIENDA': 'Estrato de Vivienda',
    'FAMI_PERSONASHOGAR': 'Personas en el Hogar',
    'FAMI_EDUCACIONPADRE': 'Educación del Padre',
    'FAMI_EDUCACIONMADRE': 'Educación de la Madre',
    'FAMI_TIENEINTERNET': 'Tiene Internet',
    'FAMI_TIENECONSOLAVIDEOJUEGOS': 'Tiene Consola de Videojuegos',
    'FAMI_NUMLIBROS': 'Número de Libros',
    'ESTU_DEDICACIONLECTURADIARIA': 'Dedicación a Lectura Diaria',
    'ESTU_DEDICACIONINTERNET': 'Dedicación a Internet',
    'ESTU_HORASSEMANATRABAJA': 'Horas Semanales de Trabajo',
    'COLE_GENERO': 'Género del Colegio',
    'COLE_NATURALEZA': 'Naturaleza del Colegio',
    'COLE_CALENDARIO': 'Calendario del Colegio',
    'COLE_CARACTER': 'Carácter del Colegio',
    'COLE_AREA_UBICACION': 'Área de Ubicación del Colegio',
    'COLE_JORNADA': 'Jornada del Colegio'
}

# Mapeo de nombres amigables para variables codificadas
friendly_names_encoded = {
    'COLE_JORNADA_SABATINA': 'Jornada del Colegio: Sabatina',
    'COLE_JORNADA_NOCHE': 'Jornada del Colegio: Noche',
    'COLE_GENERO_MIXTO': 'Género del Colegio: Mixto',
    'COLE_JORNADA_TARDE': 'Jornada del Colegio: Tarde',
    'COLE_JORNADA_MAÑANA': 'Jornada del Colegio: Mañana',
    'COLE_JORNADA_UNICA': 'Jornada del Colegio: Única',
    'FAMI_ESTRATOVIVIENDA_Sin Estrato': 'Estrato de Vivienda: Sin Estrato',
    'FAMI_NUMLIBROS_26 A 100 LIBROS': 'Número de Libros: 26 a 100 libros',
    'FAMI_NUMLIBROS_MÁS DE 100 LIBROS': 'Número de Libros: Más de 100 libros',
    'ESTU_TIENEETNIA_Si': 'Tiene Etnia: Sí',
    'FAMI_EDUCACIONPADRE_Postgrado': 'Educación del Padre: Postgrado',
    'FAMI_EDUCACIONMADRE_Postgrado': 'Educación de la Madre: Postgrado',
    'FAMI_EDUCACIONMADRE_No Aplica': 'Educación de la Madre: No Aplica',
    'FAMI_EDUCACIONMADRE_Primaria incompleta': 'Educación de la Madre: Primaria Incompleta',
    'ESTU_DEDICACIONLECTURADIARIA_Más de 2 horas': 'Dedicación a Lectura Diaria: Más de 2 horas',
    'FAMI_EDUCACIONMADRE_Ninguno': 'Educación de la Madre: Ninguno',
    'FAMI_TIENEINTERNET_Si': 'Tiene Internet: Sí'
}

# Filtro de características relevantes y organización por valor absoluto del coeficiente
coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
coef_df = coef_df[coef_df['feature'].isin(relevant_features_encoded)]
coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)

# Aplicamos nombres amigables a las características
coef_df['feature'] = coef_df['feature'].map(friendly_names_encoded).fillna(coef_df['feature'])

# Inicializar la app de Dash
app = dash.Dash(__name__)

# Layout de la app con opciones de visualización
app.layout = html.Div([
    html.Div(
        [
            html.H2("Navegación", style={'textAlign': 'center'}),
            html.Hr(),
            html.P("Selecciona una sección:", style={'textAlign': 'center'}),
            html.Div(
                [
                    html.Button("Gráfico de Relevancia", id="btn-relevancia", className="menu-button"),
                    html.Br(),
                    html.Br(),
                    html.Button("Visualización Individual", id="btn-individual", className="menu-button"),
                ],
                style={'padding': '20px'}
            ),
            html.Hr(),
            html.P(f"Total de Observaciones:", style={'textAlign': 'center', 'fontSize': '18px'}),
            html.H2(f"{total_observations:,}", style={'textAlign': 'center', 'color': '#007bff'}),
        ],
        className="sidebar"
    ),
    
    html.Div(
        [
            html.H1("Análisis de Variables y Puntaje global"),
            html.Div(
                [
                    html.H3("Relevancia de las Características Codificadas en el Modelo de Regresión", id="relevance-chart"),
                    dcc.Graph(
                        id="relevance-bar-chart",
                        figure=px.bar(
                            coef_df,
                            x="feature",
                            y="abs_coefficient",
                            title="Relevancia de Cada Característica Codificada en el Modelo de Regresión",
                            labels={"feature": "Característica Codificada", "abs_coefficient": "Valor Absoluto del Coeficiente"},
                        ).update_layout(
                            xaxis_title="Característica Codificada",
                            yaxis_title="Relevancia (|Coeficiente|)",
                            template="plotly_white"
                        )
                    )
                ],
                style={'padding': '20px'},  # Mostrado por defecto
                id="div-relevancia"
            ),

            html.Div(
                [
                    html.H3("Selecciona una Característica Original:", id="individual-visualization"),
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[{"label": friendly_names.get(feature, feature), "value": feature} for feature in relevant_features_original],
                        value=relevant_features_original[0],
                    ),
                    html.Div([
                        html.H3("Relación con Puntaje global"),
                        dcc.Graph(id="feature-plot")
                    ]),
                ],
                style={'padding': '20px', 'display': 'none'},  # Oculto por defecto
                id="div-individual"
            ),
        ],
        className="content"
    ),
], className="container")

# Callback para mostrar la sección correspondiente
@app.callback(
    [Output("div-relevancia", "style"),
     Output("div-individual", "style")],
    [Input("btn-relevancia", "n_clicks"),
     Input("btn-individual", "n_clicks")]
)
def toggle_section(btn_relevancia, btn_individual):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'block'}, {'display': 'none'}
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "btn-relevancia":
            return {'display': 'block'}, {'display': 'none'}
        elif button_id == "btn-individual":
            return {'display': 'none'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}

# Callback para actualizar la visualización individual
@app.callback(
    Output("feature-plot", "figure"),
    Input("feature-dropdown", "value")
)
def update_feature_plot(selected_feature):
    friendly_label = friendly_names.get(selected_feature, selected_feature)
    if selected_feature in saber_2020.select_dtypes(include=['object']).columns:
        fig = px.box(
            saber_2020,
            x=selected_feature,
            y="PUNT_GLOBAL",
            title=f"Distribución de {friendly_label} en Relación al Puntaje global"
        )
    elif selected_feature in saber_2020.select_dtypes(include=[np.number]).columns:
        fig = px.scatter(
            saber_2020,
            x=selected_feature,
            y="PUNT_GLOBAL",
            trendline="ols",
            trendline_color_override="red",
            title=f"Relación entre {friendly_label} y Puntaje global con Línea de Tendencia"
        )
    else:
        fig = px.scatter(
            saber_2020,
            x=selected_feature,
            y="PUNT_GLOBAL",
            title=f"Relación entre {friendly_label} y Puntaje global"
        )
    
    fig.update_layout(
        xaxis_title=friendly_label,
        yaxis_title="PUNT_GLOBAL",
        template="plotly_white"
    )
    return fig

# CSS personalizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .container {
                display: flex;
                height: 100vh;
                font-family: Arial, sans-serif;
            }
            .sidebar {
                width: 20%;
                background-color: #f1f3f5;
                padding: 10px;
                border-right: 2px solid #ccc;
            }
            .content {
                width: 80%;
                padding: 20px;
            }
            .menu-button {
                display: block;
                width: 100%;
                padding: 10px;
                font-size: 16px;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px 0;
            }
            .menu-button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Ejecutar la app
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False)
