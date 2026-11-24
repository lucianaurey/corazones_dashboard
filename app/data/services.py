"""Importación del conjunto abierto y validación del archivo de la residencia."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app import db
from app.ml.preprocesamiento import (
    CATEGORICAS,
    NUMERICAS,
    PREDICTORAS,
    RANGOS,
    limpiar_kaggle,
    nivel_riesgo_desde_mmse,
    normalizar_categoria,
    normalizar_educacion,
)
from app.models import RegistroReferencia

COLUMNAS_PLANTILLA = ["codigo", "nombre", "fecha_evaluacion", *PREDICTORAS, "mmse"]

# Datos que el dashboard nunca necesita y que, por eso, no se aceptan en el archivo (sección 3.6)
COLUMNAS_PROHIBIDAS = {
    "ci", "carnet", "cedula", "cédula", "documento", "nro_documento", "dni",
    "direccion", "dirección", "domicilio", "telefono", "teléfono", "celular", "correo", "email",
}

ALIAS_COLUMNAS = {
    "código": "codigo", "id": "codigo", "id_residente": "codigo", "id_participante": "codigo", "id_persona": "codigo",
    "nombres": "nombre", "nombre_completo": "nombre", "nombre completo": "nombre",
    "nombre y apellido": "nombre", "nombre_residente": "nombre", "residente": "nombre",
    "fecha": "fecha_evaluacion", "fecha evaluacion": "fecha_evaluacion", "fecha_evaluación": "fecha_evaluacion",
    "género": "genero", "sexo": "genero",
    "educacion": "nivel_educativo", "escolaridad": "nivel_educativo",
    "enfermedades": "enfermedades_cronicas",
    "glucosa": "glucosa_mg_dl",
    "puntaje_gds": "gds", "gds15": "gds", "gds_15": "gds",
    "sueno": "calidad_sueno", "calidad_sueño": "calidad_sueno",
    "actividad": "actividad_fisica",
    "alcohol": "consumo_alcohol",
    "puntaje_mmse": "mmse",
}


def importar_referencia(ruta_csv: str | Path) -> int:
    """Carga los 5.000 registros limpios en la base (se reemplazan si ya existían)."""
    datos = limpiar_kaggle(pd.read_csv(ruta_csv))
    datos = datos.dropna(subset=PREDICTORAS + ["mmse"])
    db.session.query(RegistroReferencia).delete()
    filas = [
        RegistroReferencia(
            id_participante=str(r.id_participante),
            edad=int(r.edad), genero=r.genero, nivel_educativo=int(r.nivel_educativo),
            estado_civil=r.estado_civil, enfermedades_cronicas=int(r.enfermedades_cronicas),
            glucosa_mg_dl=float(r.glucosa_mg_dl), imc=float(r.imc), mmse=int(r.mmse), gds=int(r.gds),
            calidad_sueno=int(r.calidad_sueno), actividad_fisica=int(r.actividad_fisica),
            tabaquismo=r.tabaquismo, consumo_alcohol=r.consumo_alcohol, nivel_riesgo=int(r.nivel_riesgo),
        )
        for r in datos.itertuples(index=False)
    ]
    db.session.bulk_save_objects(filas)
    db.session.commit()
    return len(filas)


def referencia_dataframe() -> pd.DataFrame:
    consulta = db.session.query(RegistroReferencia).statement
    return pd.read_sql(consulta, db.engine)


@dataclass
class ResultadoValidacion:
    datos: pd.DataFrame | None = None  # una fila por residente (evaluación más reciente)
    errores: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    n_filas: int = 0
    n_excluidos: int = 0

    @property
    def ok(self) -> bool:
        return not self.errores and self.datos is not None and len(self.datos) > 0

def _leer_fechas(serie: pd.Series, res: "ResultadoValidacion") -> pd.Series:
    """Acepta día/mes/año (formato boliviano) y mes/día/año (Excel en inglés): elige el que lee más fechas."""
    texto = serie.astype("string").str.strip()
    dia_mes = pd.to_datetime(texto, errors="coerce", format="%d/%m/%Y")
    mes_dia = pd.to_datetime(texto, errors="coerce", format="%m/%d/%Y")
    otras = pd.to_datetime(texto, errors="coerce", format="mixed", dayfirst=True)
    if mes_dia.notna().sum() > dia_mes.notna().sum():
        res.avisos.append("Las fechas están en formato mes/día/año; se leyeron en ese formato.")
        elegida = mes_dia
    else:
        elegida = dia_mes
    elegida = elegida.fillna(otras)
    sin_fecha = texto.notna() & (texto != "") & elegida.isna()
    if sin_fecha.any():
        res.avisos.append(f"{int(sin_fecha.sum())} fecha(s) no reconocidas se dejaron vacías.")
    return elegida

def _parece_nombre(serie: pd.Series) -> bool:
    """True si la columna tiene texto con nombres (no números ni códigos como R01)."""
    valores = serie.dropna().astype(str).str.strip()
    valores = valores[valores != ""]
    if valores.empty:
        return False
    no_numericos = pd.to_numeric(valores.str.replace(",", ".", regex=False), errors="coerce").isna().mean()
    con_espacio = valores.str.contains(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{2,}\s+[A-Za-zÁÉÍÓÚáéíóúÑñ]").mean()
    return no_numericos > 0.9 and con_espacio > 0.5


def _normalizar_columnas(df: pd.DataFrame, res: "ResultadoValidacion") -> pd.DataFrame:
    """Aplica los alias sin crear columnas repetidas y encuentra la columna de nombres.

    - Si el archivo ya trae `codigo`, una columna `id_participante` no se toma como código.
    - Una columna con nombres de personas sin encabezado ("Unnamed: 1") o con otro encabezado
      (por ejemplo `id_participante` cuando ya existe `codigo`) se toma como `nombre`.
    """
    originales = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    nuevas, usadas = [], set(originales)
    for c in originales:
        destino = ALIAS_COLUMNAS.get(c, c)
        if destino != c and destino in usadas:
            destino = c  # el destino ya existe en el archivo: no se duplica
        usadas.add(destino)
        nuevas.append(destino)
    df.columns = nuevas
    if "codigo" in df.columns and "nombre" not in df.columns:
        candidatas = [c for c in df.columns if c.startswith("unnamed") or c in ("id_participante", "id_persona", "")]
        for c in candidatas:
            if _parece_nombre(df[c]):
                df = df.rename(columns={c: "nombre"})
                res.avisos.append(f"La columna '{c}' contiene nombres y se tomó como el nombre del residente.")
                break
    return df


def validar_csv_residencia(ruta: str | Path) -> ResultadoValidacion:
    """Valida y armoniza el CSV transcrito de los cuadernos de la residencia.

    - Exige la columna `codigo` (R01, R02…). `nombre` es opcional y solo se muestra
      dentro del dashboard; se rechazan documento de identidad, dirección y teléfono.
    - Las predictoras faltantes se imputan dentro del pipeline; aquí solo se registran.
    - Un valor fuera de rango se marca como faltante y se informa.
    - Si un residente tiene varias evaluaciones, se usa la más reciente (sección 3.3)
      y se guarda el MMSE anterior para detectar descensos.
    """
    res = ResultadoValidacion()
    try:
        df = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    except Exception as exc:  # noqa: BLE001
        res.errores.append(f"No se pudo leer el archivo: {exc}")
        return res

    df = _normalizar_columnas(df, res)
    res.n_filas = len(df)
    if res.n_filas == 0:
        res.errores.append("El archivo no tiene filas.")
        return res
    if "codigo" not in df.columns:
        res.errores.append("Falta la columna obligatoria 'codigo' (por ejemplo R01, R02…).")
        return res
    sensibles = [c for c in df.columns if c in COLUMNAS_PROHIBIDAS]
    if sensibles:
        res.errores.append(
            "El archivo contiene datos personales que el dashboard no necesita (" + ", ".join(sensibles) + "). "
            "Quita esas columnas y deja solo el código, el nombre y las variables clínicas (sección 3.6)."
        )
        return res

    # Nombre y apellido en columnas separadas → una sola columna "nombre"
    apellidos = next((c for c in ("apellidos", "apellido") if c in df.columns), None)
    if apellidos:
        base = df["nombre"].fillna("") if "nombre" in df.columns else ""
        df["nombre"] = (base + " " + df[apellidos].fillna("")).str.strip()
        df = df.drop(columns=[c for c in ("apellidos", "apellido") if c in df.columns])
    if "nombre" in df.columns:
        df["nombre"] = df["nombre"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.slice(0, 120)
        df.loc[df["nombre"] == "", "nombre"] = pd.NA
    else:
        df["nombre"] = pd.NA

    faltantes = [c for c in PREDICTORAS if c not in df.columns]
    for c in faltantes:
        df[c] = np.nan
    if faltantes:
        res.avisos.append("Columnas ausentes que se imputarán en todos los registros: " + ", ".join(faltantes) + ".")
    extra = [c for c in df.columns if c not in COLUMNAS_PLANTILLA]
    if extra:
        res.avisos.append("Columnas ignoradas: " + ", ".join(extra) + ".")

    df["codigo"] = df["codigo"].astype("string").str.strip().str.upper()
    sin_codigo = df["codigo"].isna() | (df["codigo"] == "")
    if sin_codigo.any():
        res.avisos.append(f"{int(sin_codigo.sum())} fila(s) sin código excluidas.")
        res.n_excluidos += int(sin_codigo.sum())
        df = df[~sin_codigo]

    if "fecha_evaluacion" in df.columns:
        df["fecha_evaluacion"] = _leer_fechas(df["fecha_evaluacion"], res)
    else:
        df["fecha_evaluacion"] = pd.NaT

    df["nivel_educativo"] = normalizar_educacion(df["nivel_educativo"])
    for c in CATEGORICAS:
        original = df[c].astype("string").str.strip()
        df[c] = normalizar_categoria(df[c], c)
        no_reconocidos = original.notna() & (original != "") & df[c].isna()
        if no_reconocidos.any():
            res.avisos.append(f"{c}: {int(no_reconocidos.sum())} valor(es) no reconocidos se imputarán.")

    df["mmse"] = df["mmse"] if "mmse" in df.columns else np.nan
    for c in NUMERICAS + ["mmse"]:
        texto = df[c].astype("string").str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(texto, errors="coerce").astype(float)
        minimo, maximo = RANGOS[c]
        fuera = df[c].notna() & ~df[c].between(minimo, maximo)
        if fuera.any():
            extra_txt = " (¿se registró la GDS-30? el modelo usa la GDS-15)" if c == "gds" else ""
            res.avisos.append(f"{c}: {int(fuera.sum())} valor(es) fuera de rango [{minimo}–{maximo}] se tratan como faltantes{extra_txt}.")
            df.loc[fuera, c] = np.nan
    df["mmse"] = df["mmse"].round()

    # Evaluación más reciente por residente y MMSE de la evaluación anterior
    df = df.sort_values(["codigo", "fecha_evaluacion"], na_position="first")
    df["mmse_anterior"] = df.groupby("codigo")["mmse"].shift(1)
    df["fecha_anterior"] = df.groupby("codigo")["fecha_evaluacion"].shift(1)
    nombre_por_codigo = df.dropna(subset=["nombre"]).groupby("codigo")["nombre"].last()
    ultimos = df.groupby("codigo").tail(1).copy()
    ultimos["nombre"] = ultimos["codigo"].map(nombre_por_codigo)
    if len(ultimos) < len(df):
        res.avisos.append(
            f"{len(df) - len(ultimos)} evaluación(es) anteriores se usaron solo para calcular la variación del MMSE."
        )
    separacion = (ultimos["fecha_evaluacion"] - ultimos["fecha_anterior"]).dt.days
    ultimos.loc[~(separacion >= 180), "mmse_anterior"] = np.nan  # solo si hay ≥ 6 meses entre evaluaciones

    ultimos["imputadas"] = ultimos[PREDICTORAS].isna().apply(
        lambda fila: ";".join(c for c, falta in fila.items() if falta), axis=1
    )
    ultimos["nivel_observado"] = nivel_riesgo_desde_mmse(ultimos["mmse"])

    con_imputacion = ultimos["imputadas"] != ""
    if con_imputacion.any():
        res.avisos.append(
            f"{int(con_imputacion.sum())} residente(s) con al menos una variable faltante: se imputó con la mediana o la moda del entrenamiento."
        )
    muy_incompletos = ultimos[PREDICTORAS].isna().sum(axis=1) > len(PREDICTORAS) / 2
    if muy_incompletos.any():
        res.avisos.append(
            f"{int(muy_incompletos.sum())} residente(s) con más de la mitad de las variables faltantes: "
            "su estimación es poco confiable."
        )
    mayores_90 = ultimos["edad"] >= 90
    if mayores_90.any():
        res.avisos.append(
            f"{int(mayores_90.sum())} residente(s) de 90 años o más: el conjunto de entrenamiento llega hasta 89 años."
        )

    res.datos = ultimos.reset_index(drop=True)
    return res