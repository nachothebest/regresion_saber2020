from typing import Any, List, Optional

from pydantic import BaseModel
from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

# Esquema para inputs múltiples
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "COLE_JORNADA": "UNICA",
                        "COLE_GENERO": "MIXTO",
                        "FAMI_EDUCACIONMADRE": "Postgrado",
                        "FAMI_EDUCACIONPADRE": "Postgrado",
                        "ESTU_TIENEETNIA": "Si",
                        "FAMI_ESTRATOVIVIENDA": "Estrato 3",
                        "FAMI_NUMLIBROS": "26 A 100 LIBROS",
                        "ESTU_DEDICACIONLECTURADIARIA":"Más de 2 horas",
                        "FAMI_TIENEINTERNET":"Si",
                    }
                ]
            }
        }
