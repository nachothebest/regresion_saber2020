from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Ignorar advertencias y configurar visualización de pandas
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Cargar datos
archivo_2020_1 = '../data/Saber_11__2020-1_20241024.csv'
archivo_2020_2 = '../data/Saber_11__2020-2_20241024.csv'
saber_2020_1 = pd.read_csv(archivo_2020_1)
saber_2020_2 = pd.read_csv(archivo_2020_2)
saber_2020 = pd.concat([saber_2020_1, saber_2020_2], ignore_index=True)

# Preprocesamiento
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        return date_str

saber_2020['AGE'] = saber_2020['ESTU_FECHANACIMIENTO'].apply(parse_date).apply(
    lambda x: datetime.now().year - x.year if isinstance(x, pd.Timestamp) else None
)
saber_2020.drop('ESTU_FECHANACIMIENTO', axis=1, inplace=True)

columns_of_interest = [
    'ESTU_NACIONALIDAD', 'ESTU_GENERO', 'ESTU_TIENEETNIA', 'FAMI_ESTRATOVIVIENDA', 
    'FAMI_PERSONASHOGAR', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_NUMLIBROS', 
    'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_HORASSEMANATRABAJA', 'COLE_NATURALEZA', 'COLE_CALENDARIO', 'PUNT_GLOBAL', 'COLE_JORNADA'
]

saber_2020_subset = saber_2020[columns_of_interest]
categorical_columns = [col for col in saber_2020_subset.columns if saber_2020_subset[col].dtype == 'object']
numeric_columns = ['PUNT_GLOBAL']

imputer = SimpleImputer(strategy='mean')
saber_2020_subset[numeric_columns] = imputer.fit_transform(saber_2020_subset[numeric_columns])
saber_2020_encoded = pd.get_dummies(saber_2020_subset, columns=categorical_columns, drop_first=False)

# Iniciar el servidor de MLflow y registrar experimentos
mlflow.set_tracking_uri('http://localhost:5000')
experiment = mlflow.set_experiment("Saber_2020")
X_train, X_test, y_train, y_test = train_test_split(saber_2020_encoded.drop(columns='PUNT_GLOBAL'), saber_2020_encoded['PUNT_GLOBAL'])

with mlflow.start_run(experiment_id=experiment.experiment_id):
    la = Lasso(alpha=0.0001)
    la.fit(X_train, y_train)
    predictions = la.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(la, "lasso")

# Nombres más amigables para la interfaz
feature_names = {
    "ESTU_GENERO": "Género", "FAMI_ESTRATOVIVIENDA": "Estrato Vivienda",
    "FAMI_PERSONASHOGAR": "Personas en Hogar", "FAMI_EDUCACIONMADRE": "Educación Madre",
    "FAMI_EDUCACIONPADRE": "Educación Padre", "FAMI_TIENEINTERNET": "Internet en Hogar",
    "ESTU_TIENEETNIA": "Pertenencia Étnica", "PERIODO": "Periodo Escolar", "FAMI_NUMLIBROS": "Número de Libros",
    "COLE_NATURALEZA": "Naturaleza Colegio", "PUNT_GLOBAL": "Puntaje Global", "COLE_CALENDARIO": "Calendario colegio",
    "ESTU_HORASSEMANATRABAJA": "Horas de trabajo semanales", "ESTU_DEDICACIONLECTURADIARIA": "Lectura diaria",
    "ESTU_NACIONALIDAD": "Nacionalidad", "COLE_JORNADA": "Jornada"
}

# Configurar Dash
app = dash.Dash(__name__)
app.layout = html.Div(style={'font-family': 'Arial'}, children=[
    html.H1("Explorador de Datos Saber 2020", style={'textAlign': 'center', 'color': '#4CAF50'}),
    html.Div([
        html.Label("Seleccione una Característica para Visualizar:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": feature_names.get(f, f), "value": f} for f in columns_of_interest if f not in ['PUNT_GLOBAL', 'AGE']],
            value="ESTU_GENERO", clearable=False
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Br(),
    html.Div([
        dcc.Graph(id="feature-plot")
    ]),
    html.Div(id="feature-info", style={'padding': '20px', 'textAlign': 'center'}),
    html.Hr(),
    html.Div([
        html.H2("Distribución de Puntaje Global"),
        dcc.Graph(id="global-score-histogram", figure=px.histogram(saber_2020_encoded, x="PUNT_GLOBAL",
                nbins=30, title="Distribución del Puntaje Global", labels={'PUNT_GLOBAL': 'Puntaje Global'}))
    ])
])

# Callback para actualizar gráfico y descripción
@app.callback(
    [Output("feature-plot", "figure"), Output("feature-info", "children")],
    [Input("feature-dropdown", "value")]
)
def update_graph(selected_feature):
    if selected_feature in categorical_columns:
        # Variables categóricas: combinar las columnas de dummies y agrupar datos
        dummies = [col for col in saber_2020_encoded.columns if col.startswith(selected_feature)]
        melted_df = saber_2020_encoded[dummies].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
        saber_2020_encoded['category'] = melted_df
        
        fig = px.box(
            saber_2020_encoded, x="category", y="PUNT_GLOBAL",
            title=f"{feature_names.get(selected_feature, selected_feature)} vs. Puntaje Global",
            labels={"category": feature_names.get(selected_feature, selected_feature), "PUNT_GLOBAL": "Puntaje Global"}
        )
        description = f"Distribución del puntaje global para cada categoría de {feature_names.get(selected_feature, selected_feature)}."
    else:
        # Variables numéricas
        fig = px.scatter(
            saber_2020_encoded, x=selected_feature, y="PUNT_GLOBAL", trendline="ols",
            title=f"{feature_names.get(selected_feature, selected_feature)} vs. Puntaje Global",
            labels={selected_feature: feature_names.get(selected_feature, selected_feature), "PUNT_GLOBAL": "Puntaje Global"}
        )
        description = f"Relación entre {feature_names.get(selected_feature, selected_feature)} y el puntaje global."
    
    # Actualizar el diseño del gráfico
    fig.update_layout(template="plotly_white", margin=dict(t=50))
    return fig, description

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False)  # Mantiene el puerto predeterminado (8050)

