"""Aplicación del modelo a la residencia (Fase 2), evaluación descriptiva y alertas."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app import db
from app.ml.modelo import UMBRAL_MUY_ALTA, ModeloRiesgo, metricas_binarias
from app.ml.preprocesamiento import NOMBRES_VARIABLES, PREDICTORAS
from app.models import Alerta, CargaPrediccion, EvaluacionMMSE, ResultadoPrediccion

ACCIONES = {
    2: "Aplicar el MMSE completo en los próximos 7 días y revisar con el médico tratante.",
    1: "Programar el MMSE completo dentro de los próximos 30 días.",
    0: "Mantener el control rutinario según el calendario de la residencia.",
}
ACCION_MUY_ALTA = "Prioridad muy alta: aplicar el MMSE completo en los próximos días, antes que el resto."
ACCION_CORTA = {2: "MMSE ≤ 7 días", 1: "MMSE ≤ 30 días", 0: "Control rutinario"}


def _f(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value) else float(value)


def evaluar_contra_observado(resultados: pd.DataFrame, umbral: float, n_boot: int = 2000, semilla: int = 42) -> dict | None:
    """Contrasta el nivel estimado con el observado (MMSE), como describe la sección 3.8.

    Con tres clases se evalúa la clase de riesgo alto: estimado alto = P(alto) ≥ umbral clínico.
    Si alguna clase observada tiene menos de cinco residentes (regla de contingencia, sección 1.4.1),
    las predicciones se agregan: medio y alto se cuentan como riesgo elevado.
    """
    obs = resultados.dropna(subset=["nivel_observado"])
    if len(obs) < 5:
        return None
    nivel_obs = obs["nivel_observado"].to_numpy().astype(int)
    nivel_est = obs["nivel_estimado"].to_numpy().astype(int)
    conteo = {int(k): int(v) for k, v in obs["nivel_observado"].value_counts().items()}
    colapsa = any(conteo.get(k, 0) < 5 for k in (0, 1, 2))

    if colapsa:
        clase = "riesgo elevado (medio o alto)"
        real = (nivel_obs > 0).astype(int)
        pred = (nivel_est > 0).astype(int)
        puntaje = (obs["p_medio"] + obs["p_alto"]).to_numpy()
    else:
        clase = "riesgo alto"
        real = (nivel_obs == 2).astype(int)
        pred = (nivel_est == 2).astype(int)
        puntaje = obs["p_alto"].to_numpy()

    base = metricas_binarias(pred, real)
    auc = float(roc_auc_score(real, puntaje)) if 0 < real.sum() < len(real) else None

    rng = np.random.default_rng(semilla)
    muestras = {"sensibilidad": [], "especificidad": [], "exactitud_balanceada": [], "auc": []}
    n = len(obs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, p, s = real[idx], pred[idx], puntaje[idx]
        if r.sum() == 0 or r.sum() == n:
            continue
        m = metricas_binarias(p, r)
        muestras["sensibilidad"].append(m["sensibilidad"])
        muestras["especificidad"].append(m["especificidad"])
        muestras["exactitud_balanceada"].append(m["exactitud_balanceada"])
        muestras["auc"].append(roc_auc_score(r, s))
    ic = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 50 else None
        for k, v in muestras.items()
    }
    # Clasificador trivial: predice siempre la clase observada más frecuente
    mayoritaria_positiva = real.mean() > 0.5
    trivial = ({"sensibilidad": 1.0, "especificidad": 0.0, "exactitud_balanceada": 0.5, "auc": 0.5}
               if mayoritaria_positiva else
               {"sensibilidad": 0.0, "especificidad": 1.0, "exactitud_balanceada": 0.5, "auc": 0.5})
    matriz = pd.crosstab(
        obs["nivel_observado"].astype(int), obs["nivel_estimado"].astype(int)
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    return {
        "n": int(n),
        "clase": clase,
        "conteo_observado": conteo,
        "colapsa_dos_clases": colapsa,
        "sensibilidad": base["sensibilidad"],
        "especificidad": base["especificidad"],
        "exactitud_balanceada": base["exactitud_balanceada"],
        "vpp": base["vpp"],
        "auc": auc,
        "tp": base["tp"], "fn": base["fn"], "tn": base["tn"], "fp": base["fp"],
        "ic95": ic,
        "trivial": trivial,
        "matriz": matriz.values.tolist(),
    }


def _ordenar(resultados: pd.DataFrame) -> pd.DataFrame:
    """Orden de prioridad: primero el nivel asignado y, dentro de él, la mayor P(alto)."""
    resultados = resultados.sort_values(["nivel_estimado", "p_alto"], ascending=[False, False]).reset_index(drop=True)
    resultados["orden_prioridad"] = np.arange(1, len(resultados) + 1)
    return resultados


def ejecutar_prediccion(modelo: ModeloRiesgo, datos: pd.DataFrame, archivo: str, usuario_id: int | None,
                        descripcion: str = "", avisos: list[str] | None = None, n_filas: int = 0,
                        n_excluidos: int = 0) -> CargaPrediccion:
    pred = modelo.predecir(datos)
    resultados = _ordenar(pd.concat([datos, pred], axis=1))

    carga = CargaPrediccion(
        archivo=archivo[:200],
        descripcion=(descripcion or "")[:200],
        usuario_id=usuario_id,
        n_filas=n_filas or len(datos),
        n_residentes=len(resultados),
        n_excluidos=n_excluidos,
        umbral=modelo.umbral,
        version_modelo=modelo.version,
        avisos_json=json.dumps(avisos or [], ensure_ascii=False),
    )
    evaluacion = evaluar_contra_observado(resultados, modelo.umbral)
    carga.evaluacion_json = json.dumps(evaluacion, ensure_ascii=False) if evaluacion else None
    db.session.add(carga)
    db.session.flush()

    for fila in resultados.to_dict("records"):
        fecha = fila.get("fecha_evaluacion")
        r = ResultadoPrediccion(
            carga_id=carga.id,
            codigo=str(fila["codigo"])[:20],
            nombre=None if pd.isna(fila.get("nombre")) else str(fila["nombre"])[:120],
            fecha_evaluacion=fecha.date() if isinstance(fecha, pd.Timestamp) and not pd.isna(fecha) else None,
            edad=_f(fila.get("edad")), genero=fila.get("genero"), nivel_educativo=_f(fila.get("nivel_educativo")),
            estado_civil=fila.get("estado_civil"), enfermedades_cronicas=_f(fila.get("enfermedades_cronicas")),
            glucosa_mg_dl=_f(fila.get("glucosa_mg_dl")), imc=_f(fila.get("imc")), gds=_f(fila.get("gds")),
            calidad_sueno=_f(fila.get("calidad_sueno")), actividad_fisica=_f(fila.get("actividad_fisica")),
            tabaquismo=fila.get("tabaquismo"), consumo_alcohol=fila.get("consumo_alcohol"),
            mmse_observado=_f(fila.get("mmse")), mmse_anterior=_f(fila.get("mmse_anterior")),
            nivel_observado=None if pd.isna(fila.get("nivel_observado")) else int(fila["nivel_observado"]),
            p_bajo=float(fila["p_bajo"]), p_medio=float(fila["p_medio"]), p_alto=float(fila["p_alto"]),
            puntaje=float(fila["puntaje"]), nivel_estimado=int(fila["nivel_estimado"]),
            orden_prioridad=int(fila["orden_prioridad"]),
            factores_json=json.dumps(fila["factores"], ensure_ascii=False),
            imputadas=fila.get("imputadas") or "",
        )
        db.session.add(r)
        db.session.flush()
        for alerta in generar_alertas(r):
            alerta.carga_id = carga.id
            db.session.add(alerta)

    db.session.commit()
    return carga


def recalcular_cargas(modelo: ModeloRiesgo) -> int:
    """Vuelve a aplicar el modelo vigente a las cargas guardadas con una versión anterior.

    Usa las variables ya guardadas de cada residente (los faltantes siguen faltando y los imputa
    el pipeline), regenera las alertas y conserva la atención y las evaluaciones MMSE registradas.
    """
    pendientes = CargaPrediccion.query.filter(CargaPrediccion.version_modelo != modelo.version).all()
    for carga in pendientes:
        filas = carga.resultados.all()
        if not filas:
            carga.version_modelo, carga.umbral = modelo.version, modelo.umbral
            continue
        datos = pd.DataFrame([{**{c: getattr(r, c) for c in PREDICTORAS}, "id": r.id} for r in filas]).set_index("id")
        for c in PREDICTORAS:
            datos[c] = datos[c].where(datos[c].notna(), np.nan)
        pred = modelo.predecir(datos)
        por_id = {r.id: r for r in filas}
        for rid, p in pred.iterrows():
            r = por_id[rid]
            r.p_bajo, r.p_medio, r.p_alto = float(p.p_bajo), float(p.p_medio), float(p.p_alto)
            r.puntaje, r.nivel_estimado = float(p.puntaje), int(p.nivel_estimado)
            r.factores_json = json.dumps(p.factores, ensure_ascii=False)
        ordenados = sorted(filas, key=lambda r: (-r.nivel_estimado, -r.p_alto))
        for i, r in enumerate(ordenados, 1):
            r.orden_prioridad = i

        # Alertas: se regeneran conservando la atención y las evaluaciones por residente y tipo
        anteriores = carga.alertas.all()
        atencion = {(a.codigo, a.tipo): a for a in anteriores if a.estado == "atendida"}
        evaluaciones = EvaluacionMMSE.query.filter(EvaluacionMMSE.alerta_id.in_([a.id for a in anteriores])).all() if anteriores else []
        for ev in evaluaciones:
            ev.alerta_id = None
        db.session.flush()
        for a in anteriores:
            db.session.delete(a)
        db.session.flush()
        nuevas_por_codigo: dict[str, list[Alerta]] = {}
        for r in filas:
            for alerta in generar_alertas(r):
                alerta.carga_id = carga.id
                previa = atencion.get((alerta.codigo, alerta.tipo))
                if previa:
                    alerta.estado, alerta.fecha_atencion = "atendida", previa.fecha_atencion
                    alerta.atendida_por_id, alerta.nota_atencion = previa.atendida_por_id, previa.nota_atencion
                db.session.add(alerta)
                nuevas_por_codigo.setdefault(alerta.codigo, []).append(alerta)
        db.session.flush()
        orden_sev = {"alta": 0, "media": 1, "baja": 2}
        for ev in evaluaciones:
            candidatas = sorted(nuevas_por_codigo.get(ev.codigo, []), key=lambda a: orden_sev[a.severidad])
            if candidatas:
                ev.alerta_id = candidatas[0].id

        df = pd.DataFrame([{"nivel_observado": r.nivel_observado, "nivel_estimado": r.nivel_estimado,
                            "p_medio": r.p_medio, "p_alto": r.p_alto} for r in filas])
        evaluacion = evaluar_contra_observado(df, modelo.umbral)
        carga.evaluacion_json = json.dumps(evaluacion, ensure_ascii=False) if evaluacion else None
        carga.umbral, carga.version_modelo = modelo.umbral, modelo.version
    if pendientes:
        db.session.commit()
    return len(pendientes)


def generar_alertas(r: ResultadoPrediccion) -> list[Alerta]:
    alertas = []
    factores = ", ".join(f["variable"].lower() for f in r.factores) or "sin un factor dominante"
    p_alto = f"{r.p_alto:.2f}".replace(".", ",")
    if r.nivel_estimado == 2:
        muy_alta = r.p_alto >= UMBRAL_MUY_ALTA
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Riesgo alto estimado", severidad="alta",
                              mensaje=(f"P(alto) = {p_alto}{' (prioridad muy alta)' if muy_alta else ''}. "
                                       f"Factores que más elevan la estimación: {factores}."),
                              accion_sugerida=ACCION_MUY_ALTA if muy_alta else ACCIONES[2]))
    elif r.nivel_estimado == 1:
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Riesgo medio estimado", severidad="media",
                              mensaje=f"P(alto) = {p_alto}, bajo el umbral clínico; el nivel más probable es medio. "
                                      f"Factores que más elevan la estimación: {factores}.",
                              accion_sugerida=ACCIONES[1]))
    if r.mmse_observado is not None and r.mmse_anterior is not None and r.mmse_anterior - r.mmse_observado >= 3:
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Descenso de MMSE", severidad="alta",
                              mensaje=f"El MMSE bajó de {r.mmse_anterior:.0f} a {r.mmse_observado:.0f} puntos entre evaluaciones separadas por 6 meses o más.",
                              accion_sugerida="Revisar con el equipo médico y adelantar la reevaluación."))
    if r.nivel_observado == 2 and r.nivel_estimado != 2:
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Riesgo alto no señalado", severidad="alta",
                              mensaje=f"El MMSE observado ({r.mmse_observado:.0f}) corresponde a riesgo alto, pero P(alto) = {p_alto} quedó bajo el umbral clínico.",
                              accion_sugerida="Priorizar según el MMSE observado; el modelo no reemplaza la prueba."))
    elif r.nivel_observado == 1 and r.nivel_estimado == 0:
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Riesgo medio no señalado", severidad="media",
                              mensaje=f"El MMSE observado ({r.mmse_observado:.0f}) corresponde a riesgo medio, pero el modelo asignó riesgo bajo.",
                              accion_sugerida="Programar el seguimiento según el MMSE observado."))
    imputadas = r.lista_imputadas
    if len(imputadas) >= 3:
        nombres = ", ".join(NOMBRES_VARIABLES.get(v, v).lower() for v in imputadas)
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Datos incompletos", severidad="baja",
                              mensaje=f"{len(imputadas)} variables imputadas: {nombres}.",
                              accion_sugerida="Completar el registro en la próxima evaluación."))
    if r.edad is not None and r.edad >= 90:
        alertas.append(Alerta(resultado_id=r.id, codigo=r.codigo, tipo="Fuera del rango de entrenamiento", severidad="baja",
                              mensaje=f"Edad de {r.edad:.0f} años: el modelo se entrenó con personas de 60 a 89 años.",
                              accion_sugerida="Interpretar la estimación con cautela."))
    return alertas


def estimar_individual(modelo: ModeloRiesgo, valores: dict) -> dict:
    fila = pd.DataFrame([{c: valores.get(c) for c in PREDICTORAS}])
    for c in PREDICTORAS:
        if fila[c].iloc[0] in ("", None):
            fila[c] = np.nan
    numericas = [c for c in PREDICTORAS if c not in ("genero", "estado_civil", "tabaquismo", "consumo_alcohol")]
    fila[numericas] = fila[numericas].apply(pd.to_numeric, errors="coerce")
    pred = modelo.predecir(fila).iloc[0]
    imputadas = [NOMBRES_VARIABLES[c] for c in PREDICTORAS if pd.isna(fila[c].iloc[0])]
    nivel = int(pred["nivel_estimado"])
    muy_alta = nivel == 2 and float(pred["p_alto"]) >= UMBRAL_MUY_ALTA
    return {
        "p_bajo": float(pred["p_bajo"]), "p_medio": float(pred["p_medio"]), "p_alto": float(pred["p_alto"]),
        "puntaje": float(pred["puntaje"]), "nivel": nivel, "factores": pred["factores"],
        "imputadas": imputadas, "accion": ACCION_MUY_ALTA if muy_alta else ACCIONES[nivel],
        "prioridad": "Muy alta" if muy_alta else {2: "Alta", 1: "Media", 0: "Baja"}[nivel],
    }


def resultados_dataframe(carga: CargaPrediccion) -> pd.DataFrame:
    filas = []
    for r in carga.resultados.order_by(ResultadoPrediccion.orden_prioridad):
        filas.append({
            "orden_prioridad": r.orden_prioridad, "codigo": r.codigo, "nombre": r.nombre or "",
            "fecha_evaluacion": r.fecha_evaluacion.isoformat() if r.fecha_evaluacion else "",
            "nivel_estimado": ["Bajo", "Medio", "Alto"][r.nivel_estimado],
            "prioridad": r.prioridad_texto,
            "p_alto": round(r.p_alto, 4), "p_medio": round(r.p_medio, 4), "p_bajo": round(r.p_bajo, 4),
            "umbral_clinico": round(carga.umbral, 2),
            "mmse_observado": r.mmse_observado,
            "nivel_observado": ["Bajo", "Medio", "Alto"][r.nivel_observado] if r.nivel_observado is not None else "",
            "factores": "; ".join(f["variable"] for f in r.factores),
            "variables_imputadas": r.imputadas or "",
            "accion_sugerida": ACCION_MUY_ALTA if r.prioridad == 1 else ACCIONES[r.nivel_estimado],
        })
    return pd.DataFrame(filas)
