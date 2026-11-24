"""Limpieza y armonización de datos, igual que el cuaderno del capítulo 4.

Las mismas reglas se aplican al conjunto abierto (Kaggle) y a los registros de la
residencia, para que el modelo reciba la información con la misma forma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERICAS = [
    "edad",
    "nivel_educativo",
    "enfermedades_cronicas",
    "glucosa_mg_dl",
    "imc",
    "gds",
    "calidad_sueno",
    "actividad_fisica",
]
CATEGORICAS = ["genero", "estado_civil", "tabaquismo", "consumo_alcohol"]
PREDICTORAS = NUMERICAS + CATEGORICAS

ETIQUETAS = ["Bajo", "Medio", "Alto"]

NOMBRES_VARIABLES = {
    "edad": "Edad",
    "nivel_educativo": "Nivel educativo",
    "enfermedades_cronicas": "Enfermedades crónicas",
    "glucosa_mg_dl": "Glucosa",
    "imc": "IMC",
    "gds": "Puntaje GDS",
    "calidad_sueno": "Calidad de sueño",
    "actividad_fisica": "Actividad física",
    "genero": "Género",
    "estado_civil": "Estado civil",
    "tabaquismo": "Tabaquismo",
    "consumo_alcohol": "Consumo de alcohol",
    "genero_Masculino": "Género: masculino",
    "estado_civil_Divorciado": "Estado civil: divorciado",
    "estado_civil_Soltero": "Estado civil: soltero",
    "estado_civil_Viudo": "Estado civil: viudo",
    "tabaquismo_Sí": "Tabaquismo: sí",
    "consumo_alcohol_Sí": "Consumo de alcohol: sí",
}

# Rangos válidos (sección 4.1.1). Un valor fuera de rango se marca como faltante.
RANGOS = {
    "edad": (60, 110),
    "nivel_educativo": (0, 3),
    "enfermedades_cronicas": (0, 15),
    "glucosa_mg_dl": (40, 600),
    "imc": (10, 70),
    "gds": (0, 15),
    "calidad_sueno": (1, 5),
    "actividad_fisica": (0, 10),
    "mmse": (0, 30),
}

MAPA_COLUMNAS_KAGGLE = {
    "Participant_ID": "id_participante",
    "Age": "edad",
    "Gender": "genero",
    "Education_Level": "nivel_educativo",
    "Region": "region",
    "Marital_Status": "estado_civil",
    "Chronic_Diseases": "enfermedades_cronicas",
    "Glucose_Level": "glucosa_mg_dl",
    "BMI": "imc",
    "MMSE_Score": "mmse",
    "GDS_Score": "gds",
    "Sleep_Quality_Score": "calidad_sueno",
    "Physical_Activity_Score": "actividad_fisica",
    "Smoking_Status": "tabaquismo",
    "Alcohol_Use": "consumo_alcohol",
    "Cognitive_Impairment_Status": "etiqueta_kaggle",
}

# Equivalencias de categorías: aceptan inglés, español, mayúsculas y abreviaturas.
MAPAS_CATEGORIAS = {
    "genero": {
        "male": "Masculino", "masculino": "Masculino", "m": "Masculino", "hombre": "Masculino", "varon": "Masculino", "varón": "Masculino",
        "female": "Femenino", "femenino": "Femenino", "f": "Femenino", "mujer": "Femenino",
    },
    "estado_civil": {
        "single": "Soltero", "soltero": "Soltero", "soltera": "Soltero",
        "married": "Casado", "casado": "Casado", "casada": "Casado",
        "divorced": "Divorciado", "divorciado": "Divorciado", "divorciada": "Divorciado", "separado": "Divorciado", "separada": "Divorciado",
        "widowed": "Viudo", "viudo": "Viudo", "viuda": "Viudo",
    },
    "tabaquismo": {"yes": "Sí", "si": "Sí", "sí": "Sí", "1": "Sí", "no": "No", "0": "No"},
    "consumo_alcohol": {"yes": "Sí", "si": "Sí", "sí": "Sí", "1": "Sí", "no": "No", "0": "No"},
}

MAPA_EDUCACION_TEXTO = {
    "ninguno": 0, "ninguna": 0, "sin estudios": 0, "none": 0,
    "primaria": 1, "primary": 1,
    "secundaria": 2, "secondary": 2, "bachiller": 2,
    "superior": 3, "universitario": 3, "universitaria": 3, "higher": 3, "tecnico": 3, "técnico": 3,
}

NOMBRE_EDUCACION = {0: "Ninguno", 1: "Primaria", 2: "Secundaria", 3: "Superior"}


def nivel_riesgo_desde_mmse(mmse: pd.Series) -> pd.Series:
    """Regla de la Tabla 2: 0 = bajo (≥ 24), 1 = medio (18–23), 2 = alto (≤ 17)."""
    valores = pd.to_numeric(mmse, errors="coerce").astype(float)
    nivel = pd.Series(np.select([valores >= 24, valores >= 18], [0, 1], default=2), index=mmse.index)
    return nivel.where(valores.notna())


def grupo_edad(edad: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(edad, errors="coerce"),
        bins=[59, 69, 79, 89, 200],
        labels=["60–69", "70–79", "80–89", "90 o más"],
    ).astype(str).replace("nan", "Sin dato")


def normalizar_categoria(serie: pd.Series, columna: str) -> pd.Series:
    mapa = MAPAS_CATEGORIAS[columna]
    limpio = serie.astype("string").str.strip().str.lower()
    return limpio.map(mapa).astype(object).where(limpio.notna() & (limpio != ""), None)


def normalizar_educacion(serie: pd.Series) -> pd.Series:
    numerico = pd.to_numeric(serie, errors="coerce").astype(float)
    texto = serie.astype("string").str.strip().str.lower().map(MAPA_EDUCACION_TEXTO)
    return numerico.fillna(pd.to_numeric(texto, errors="coerce").astype(float))


def limpiar_kaggle(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce la limpieza del capítulo 4 sobre el archivo de Kaggle."""
    df = df_raw.rename(columns=MAPA_COLUMNAS_KAGGLE).copy()
    for columna in CATEGORICAS:
        df[columna] = normalizar_categoria(df[columna], columna)
    for columna, (minimo, maximo) in RANGOS.items():
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df.loc[~df[columna].between(minimo, maximo), columna] = np.nan
    df["mmse"] = df["mmse"].round()
    df["gds"] = df["gds"].round()
    df["glucosa_mg_dl"] = df["glucosa_mg_dl"].round(1)
    df["imc"] = df["imc"].round(1)
    df = df.drop(columns=[c for c in ("etiqueta_kaggle", "region") if c in df.columns])
    df["nivel_riesgo"] = nivel_riesgo_desde_mmse(df["mmse"]).astype(int)
    return df
