import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import requests
import json
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Inicialización de la app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# API URL
api_url = os.getenv('API_URL')
api_url = "http://{}:8001/api/v1/predict".format(api_url)

# Layout de la aplicación
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '20px'},
    children=[
        html.H2(
            "Predicción de Puntaje Global",
            style={'textAlign': 'center', 'color': '#333'}
        ),
        html.Div(
            "Por favor, selecciona las características del estudiante:",
            style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '16px'}
        ),
        html.Div(
            style={'maxWidth': '800px', 'margin': 'auto', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'},
            children=[
                html.Div([
                    html.Label("Jornada del colegio:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='COLE_JORNADA',
                        options=[{'label': i, 'value': i} for i in ['COMPLETA', 'MAÑANA', 'NOCHE', 'SABATINA', 'TARDE', 'UNICA']],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Género del colegio:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='COLE_GENERO',
                        options=[{'label': i, 'value': i} for i in ['FEMENINO', 'MASCULINO', 'MIXTO']],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Nivel educativo de la madre:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='FAMI_EDUCACIONMADRE',
                        options=[{'label': i, 'value': i} for i in [
                            'Educación profesional completa', 'Educación profesional incompleta', 'Ninguno', 'No sabe',
                            'Postgrado', 'Primaria completa', 'Primaria incompleta',
                            'Secundaria (Bachillerato) completa', 'Secundaria (Bachillerato) incompleta',
                            'Técnica o tecnológica completa', 'Técnica o tecnológica incompleta'
                        ]],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Nivel educativo del padre:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='FAMI_EDUCACIONPADRE',
                        options=[{'label': i, 'value': i} for i in [
                            'Educación profesional completa', 'Educación profesional incompleta', 'Ninguno', 'No sabe',
                            'Postgrado', 'Primaria completa', 'Primaria incompleta',
                            'Secundaria (Bachillerato) completa', 'Secundaria (Bachillerato) incompleta',
                            'Técnica o tecnológica completa', 'Técnica o tecnológica incompleta'
                        ]],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("¿Tiene etnia?", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='ESTU_TIENEETNIA',
                        options=[{'label': i, 'value': i} for i in ['No', 'Si']],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Estrato de vivienda:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='FAMI_ESTRATOVIVIENDA',
                        options=[{'label': i, 'value': i} for i in [
                            'Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6', 'Sin Estrato'
                        ]],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Número de libros en casa:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='FAMI_NUMLIBROS',
                        options=[{'label': i, 'value': i} for i in [
                            '0 A 10 LIBROS', '11 A 25 LIBROS', '26 A 100 LIBROS', 'MÁS DE 100 LIBROS'
                        ]],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("Dedicación diaria a la lectura:", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='ESTU_DEDICACIONLECTURADIARIA',
                        options=[{'label': i, 'value': i} for i in [
                            '30 minutos o menos', 'Entre 1 y 2 horas', 'Entre 30 y 60 minutos', 'Más de 2 horas',
                            'No leo por entretenimiento'
                        ]],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    html.Label("¿Tiene acceso a Internet en casa?", style={'fontSize': '14px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='FAMI_TIENEINTERNET',
                        options=[{'label': i, 'value': i} for i in ['Si', 'No']],
                        placeholder="Selecciona una opción"
                    ),
                ], style={'marginBottom': '15px'}),
            ]
        ),
        html.Br(),
        html.H4("Resultados", style={'textAlign': 'center', 'color': '#333'}),
        html.Div(
            id='resultado',
            style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px', 'fontWeight': 'bold', 'color': '#555'}
        ),
    ]
)

# Callback para enviar datos a la API con valores predeterminados si hay campos en blanco
@app.callback(
    Output(component_id='resultado', component_property='children'),
    [Input(component_id='COLE_JORNADA', component_property='value'),
     Input(component_id='COLE_GENERO', component_property='value'),
     Input(component_id='FAMI_EDUCACIONMADRE', component_property='value'),
     Input(component_id='FAMI_EDUCACIONPADRE', component_property='value'),
     Input(component_id='ESTU_TIENEETNIA', component_property='value'),
     Input(component_id='FAMI_ESTRATOVIVIENDA', component_property='value'),
     Input(component_id='FAMI_NUMLIBROS', component_property='value'),
     Input(component_id='ESTU_DEDICACIONLECTURADIARIA', component_property='value'),
     Input(component_id='FAMI_TIENEINTERNET', component_property='value')]
)
def enviar_a_api(*args):
    # Crear un diccionario con valores predeterminados si no hay selección
    payload_inputs = {
        "COLE_JORNADA": args[0] or "",
        "COLE_GENERO": args[1] or "",
        "FAMI_EDUCACIONMADRE": args[2] or "",
        "FAMI_EDUCACIONPADRE": args[3] or "",
        "ESTU_TIENEETNIA": args[4] or "",
        "FAMI_ESTRATOVIVIENDA": args[5] or "",
        "FAMI_NUMLIBROS": args[6] or "",
        "ESTU_DEDICACIONLECTURADIARIA": args[7] or "",
        "FAMI_TIENEINTERNET": args[8] or ""
    }

    payload = {"inputs": [payload_inputs]}
    headers = {"Content-Type": "application/json"}

    try:
        print("Payload enviado a la API:", json.dumps(payload, indent=4))  # Imprime el JSON en la consola
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        resultado = response.json()

        # Extraer el valor de "predictions"
        puntaje_global = resultado.get("predictions", [None])[0]
        if puntaje_global is not None:
            return f"Puntaje Global: {puntaje_global:.2f}"
        else:
            return "Error: No se encontró un puntaje en la respuesta de la API."
    except requests.exceptions.RequestException as e:
        return f"Error al conectar con la API: {str(e)}"


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
