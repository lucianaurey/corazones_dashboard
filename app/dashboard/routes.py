from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from scipy import stats
from werkzeug.utils import secure_filename

from app import NIVELES, db, obtener_modelo
from app.auth.routes import log_access
from app.data.services import referencia_dataframe
from app.forms import AtenderAlertaForm, EvaluacionMMSEForm
from app.ml.preprocesamiento import NOMBRE_EDUCACION, NOMBRES_VARIABLES, NUMERICAS, grupo_edad
from app.models import Alerta, CargaPrediccion, EvaluacionMMSE, ResultadoPrediccion

dashboard_bp = Blueprint("dashboard", __name__)

COLOR_NIVEL = {0: "#7FB685", 1: "#E8B04B", 2: "#B8486B"}
ORDEN_EDAD = ["60–69", "70–79", "80–89", "90 o más"]


def _carga_seleccionada() -> CargaPrediccion | None:
    carga_id = request.args.get("carga", type=int)
    if carga_id:
        return db.session.get(CargaPrediccion, carga_id)
    return CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).first()


def _resultados_df(carga: CargaPrediccion) -> pd.DataFrame:
    filas = [
        {
            "codigo": r.codigo, "etiqueta": r.etiqueta_corta, "edad": r.edad, "genero": r.genero, "puntaje": r.puntaje,
            "nivel_estimado": r.nivel_estimado, "nivel_observado": r.nivel_observado,
            "n_imputadas": len(r.lista_imputadas), "orden": r.orden_prioridad,
        }
        for r in carga.resultados
    ]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- Panel principal
@dashboard_bp.route("/principal")
@login_required
def principal():
    modelo = obtener_modelo()
    cargas = CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).all()
    carga = _carga_seleccionada()
    contexto = {"title": "Panel principal", "cargas": cargas, "carga": carga, "modelo": modelo, "charts": [], "kpis": []}
    if carga is None or carga.n_residentes == 0:
        return render_template("principal.html", **contexto)

    df = _resultados_df(carga)
    n = len(df)
    elevados = int((df["nivel_estimado"] > 0).sum())
    altos = int((df["nivel_estimado"] == 2).sum())
    muy_alta = int(((df["nivel_estimado"] == 2) & (df["puntaje"] >= 0.70)).sum())
    abiertas = carga.alertas.filter_by(estado="abierta").count()
    abiertas_altas = carga.alertas.filter_by(estado="abierta", severidad="alta").count()
    pct_imputadas = df["n_imputadas"].sum() / (n * 12) if n else 0
    contexto["kpis"] = [
        {"titulo": "Residentes evaluados", "valor": f"{n}", "detalle": f"Carga #{carga.id} · {carga.fecha:%d/%m/%Y}", "icono": "fa-people-roof", "tono": "lila"},
        {"titulo": "Riesgo alto estimado", "valor": f"{altos}", "detalle": f"P(alto) ≥ {carga.umbral:.2f} · MMSE en ≤ 7 días".replace(".", ","), "icono": "fa-heart-pulse", "tono": "rosa"},
        {"titulo": "Prioridad muy alta", "valor": f"{muy_alta}", "detalle": "P(alto) ≥ 0,70 · evaluar primero", "icono": "fa-angles-up", "tono": "morado"},
        {"titulo": "Riesgo medio o alto", "valor": f"{elevados}", "detalle": f"{elevados / n:.0%} de los residentes".replace(".", ","), "icono": "fa-triangle-exclamation", "tono": "ambar"},
        {"titulo": "Alertas abiertas", "valor": f"{abiertas}", "detalle": f"{abiertas_altas} de severidad alta", "icono": "fa-bell", "tono": "lila"},
        {"titulo": "Datos imputados", "valor": f"{pct_imputadas:.1%}".replace(".", ","), "detalle": "de los valores de las 12 variables", "icono": "fa-puzzle-piece", "tono": "verde"},
    ]

    conteo = df["nivel_estimado"].value_counts().reindex([0, 1, 2], fill_value=0)
    ranking = df.sort_values("orden")
    df["grupo_edad"] = grupo_edad(df["edad"])
    tabla_edad = pd.crosstab(df["grupo_edad"], df["nivel_estimado"]).reindex(columns=[0, 1, 2], fill_value=0)
    tabla_edad = tabla_edad.reindex([g for g in ORDEN_EDAD + ["Sin dato"] if g in tabla_edad.index])
    charts = [
        {"id": "nivelEstimado", "titulo": "Residentes por nivel de riesgo estimado", "tipo": "doughnut",
         "labels": list(NIVELES.values()),
         "datasets": [{"label": "Residentes", "data": conteo.tolist(), "colores": [COLOR_NIVEL[k] for k in range(3)]}]},
        {"id": "edad", "titulo": "Nivel estimado por grupo de edad", "tipo": "bar",
         "labels": tabla_edad.index.tolist(),
         "datasets": [{"label": NIVELES[k], "data": tabla_edad[k].tolist(), "color": COLOR_NIVEL[k]} for k in range(3)],
         "opciones": {"apilado": True}},
        {"id": "ranking", "titulo": "Probabilidad de riesgo alto por residente (orden de prioridad)", "tipo": "bar", "ancho": True,
         "labels": ranking["etiqueta"].tolist(),
         "datasets": [{"label": "P(alto)", "data": ranking["puntaje"].round(3).tolist(),
                       "colores": [COLOR_NIVEL[int(k)] for k in ranking["nivel_estimado"]]}],
         "opciones": {"lineaUmbral": carga.umbral, "etiquetaUmbral": f"umbral clínico {carga.umbral:.2f}".replace(".", ","), "max": 1}},
    ]
    obs = df.dropna(subset=["nivel_observado"])
    if len(obs):
        cruce = pd.crosstab(obs["nivel_observado"].astype(int), obs["nivel_estimado"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        charts.append(
            {"id": "observado", "titulo": "Nivel observado (MMSE) frente al estimado", "tipo": "bar",
             "labels": [f"Observado {NIVELES[k].lower()}" for k in range(3)], "ancho": True, "horizontal": True,
             "datasets": [{"label": f"Estimado {NIVELES[k].lower()}", "data": cruce[k].tolist(), "color": COLOR_NIVEL[k]} for k in range(3)],
             "opciones": {"apilado": True}})
    contexto["charts"] = charts
    contexto["prioritarios"] = carga.resultados.order_by(ResultadoPrediccion.orden_prioridad).limit(10).all()
    return render_template("principal.html", **contexto)


# ---------------------------------------------------------------- Alertas
SEV_ORDEN = {"alta": 0, "media": 1, "baja": 2}


@dashboard_bp.route("/alertas")
@login_required
def alertas():
    cargas = CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).all()
    carga = _carga_seleccionada()
    filtros = {k: request.args.get(k, "") for k in ("estado", "severidad", "tipo")}
    if not filtros["estado"] and "estado" not in request.args:
        filtros["estado"] = "abierta"
    consulta = Alerta.query
    if carga:
        consulta = consulta.filter_by(carga_id=carga.id)
    todas = consulta.all()
    for campo in ("estado", "severidad", "tipo"):
        if filtros[campo]:
            consulta = consulta.filter(getattr(Alerta, campo) == filtros[campo])
    orden_sev = db.case(SEV_ORDEN, value=Alerta.severidad)
    lista = consulta.order_by(orden_sev, Alerta.codigo).all()

    def contar(**kw):
        return sum(all(getattr(a, k) == v for k, v in kw.items()) for a in todas)

    ids = [a.id for a in todas]
    programadas = EvaluacionMMSE.query.filter(EvaluacionMMSE.alerta_id.in_(ids), EvaluacionMMSE.estado == "programada").count() if ids else 0
    realizadas = EvaluacionMMSE.query.filter(EvaluacionMMSE.alerta_id.in_(ids), EvaluacionMMSE.estado == "realizada").count() if ids else 0
    kpis = [
        {"titulo": "Alertas abiertas", "valor": contar(estado="abierta"), "detalle": f"de {len(todas)} generadas", "icono": "fa-bell", "tono": "morado"},
        {"titulo": "Severidad alta", "valor": contar(estado="abierta", severidad="alta"), "detalle": "Atender esta semana", "icono": "fa-circle-exclamation", "tono": "rosa"},
        {"titulo": "Severidad media", "valor": contar(estado="abierta", severidad="media"), "detalle": "Programar en 30 días", "icono": "fa-clock", "tono": "ambar"},
        {"titulo": "Severidad baja", "valor": contar(estado="abierta", severidad="baja"), "detalle": "Calidad de datos", "icono": "fa-clipboard-check", "tono": "verde"},
        {"titulo": "MMSE programados", "valor": programadas, "detalle": f"{realizadas} ya realizados", "icono": "fa-calendar-check", "tono": "lila"},
        {"titulo": "Atendidas", "valor": contar(estado="atendida"), "detalle": "Con registro de quién y cuándo", "icono": "fa-check-double", "tono": "lila"},
    ]
    tipos = sorted({a.tipo for a in todas})
    por_tipo = pd.Series([a.tipo for a in todas if a.estado == "abierta"], dtype=object).value_counts()
    charts = [
        {"id": "porTipo", "titulo": "Alertas abiertas por tipo", "tipo": "bar", "horizontal": True,
         "labels": por_tipo.index.tolist(),
         "datasets": [{"label": "Alertas", "data": por_tipo.tolist(), "color": "#9D75AA"}]},
        {"id": "porSeveridad", "titulo": "Alertas abiertas por severidad", "tipo": "doughnut",
         "labels": ["Alta", "Media", "Baja"],
         "datasets": [{"label": "Alertas", "data": [kpis[1]["valor"], kpis[2]["valor"], kpis[3]["valor"]],
                       "colores": ["#B8486B", "#E8B04B", "#7FB685"]}]},
    ]
    return render_template(
        "alertas.html", title="Alertas", cargas=cargas, carga=carga, alertas=lista, kpis=kpis,
        charts=charts, filtros=filtros, tipos=tipos,
    )


def _carpeta_evaluaciones() -> Path:
    carpeta = Path(current_app.config["EVALUACIONES_FOLDER"])
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def _borrar_imagen(ev: EvaluacionMMSE) -> None:
    if ev.imagen_archivo:
        ruta = _carpeta_evaluaciones() / ev.imagen_archivo
        try:
            ruta.unlink(missing_ok=True)
        except OSError:
            current_app.logger.warning("No se pudo eliminar %s", ruta)
    ev.imagen_archivo = None
    ev.imagen_nombre = None


def _guardar_imagen(ev: EvaluacionMMSE, archivo) -> None:
    nombre = secure_filename(archivo.filename or "resultado")
    extension = nombre.rsplit(".", 1)[-1].lower() if "." in nombre else "jpg"
    _borrar_imagen(ev)
    interno = f"{ev.codigo}_{uuid.uuid4().hex[:12]}.{extension}"
    archivo.save(_carpeta_evaluaciones() / interno)
    ev.imagen_archivo = interno
    ev.imagen_nombre = (archivo.filename or nombre)[:200]


def _render_ficha(alerta: Alerta, form: EvaluacionMMSEForm, editando: EvaluacionMMSE | None = None, pestana: str = "seguimiento"):
    r = alerta.resultado
    otras = (Alerta.query.filter(Alerta.carga_id == alerta.carga_id, Alerta.codigo == alerta.codigo, Alerta.id != alerta.id)
             .order_by(db.case(SEV_ORDEN, value=Alerta.severidad)).all())
    historial = EvaluacionMMSE.query.filter_by(codigo=alerta.codigo).order_by(EvaluacionMMSE.fecha_registro.desc()).all()
    evaluaciones = [e for e in historial if e.alerta_id == alerta.id]
    otras_evaluaciones = [e for e in historial if e.alerta_id != alerta.id]
    datos = []
    if r:
        educacion = NOMBRE_EDUCACION.get(int(r.nivel_educativo)) if r.nivel_educativo is not None else None
        valores = [
            ("Edad", f"{r.edad:.0f} años" if r.edad is not None else None, "edad"),
            ("Género", r.genero, "genero"),
            ("Nivel educativo", educacion, "nivel_educativo"),
            ("Estado civil", r.estado_civil, "estado_civil"),
            ("Enfermedades crónicas", f"{r.enfermedades_cronicas:.0f}" if r.enfermedades_cronicas is not None else None, "enfermedades_cronicas"),
            ("Glucosa", f"{r.glucosa_mg_dl:.1f} mg/dL".replace(".", ",") if r.glucosa_mg_dl is not None else None, "glucosa_mg_dl"),
            ("IMC", f"{r.imc:.1f} kg/m²".replace(".", ",") if r.imc is not None else None, "imc"),
            ("Puntaje GDS-15", f"{r.gds:.0f}" if r.gds is not None else None, "gds"),
            ("Calidad de sueño", f"{r.calidad_sueno:.0f} / 5" if r.calidad_sueno is not None else None, "calidad_sueno"),
            ("Actividad física", f"{r.actividad_fisica:.0f} / 10" if r.actividad_fisica is not None else None, "actividad_fisica"),
            ("Tabaquismo", r.tabaquismo, "tabaquismo"),
            ("Consumo de alcohol", r.consumo_alcohol, "consumo_alcohol"),
        ]
        imputadas = set(r.lista_imputadas)
        datos = [{"nombre": n, "valor": v, "imputada": c in imputadas} for n, v, c in valores]
    return render_template(
        "alerta_ficha.html", title=f"Alerta · {r.etiqueta if r else alerta.codigo}", alerta=alerta, r=r,
        otras=otras, evaluaciones=evaluaciones, otras_evaluaciones=otras_evaluaciones, datos=datos,
        form=form, form_atencion=AtenderAlertaForm(), editando=editando, pestana=pestana,
        umbral=alerta.carga.umbral,
    )


@dashboard_bp.route("/alertas/<int:alerta_id>")
@login_required
def ficha_alerta(alerta_id: int):
    alerta = db.session.get(Alerta, alerta_id) or abort(404)
    form = EvaluacionMMSEForm()
    if not form.lugar.data:
        form.lugar.data = "Consultorio de la residencia"
    return _render_ficha(alerta, form, pestana=request.args.get("pestana", "seguimiento"))


def _aplicar_formulario(ev: EvaluacionMMSE, form: EvaluacionMMSEForm) -> None:
    ev.estado = form.estado.data
    ev.fecha_programada = form.fecha_programada.data
    ev.hora = form.hora.data or None
    ev.lugar = (form.lugar.data or "").strip() or None
    ev.responsable = (form.responsable.data or "").strip() or None
    ev.fecha_realizada = form.fecha_realizada.data if form.estado.data == "realizada" else None
    ev.mmse_puntaje = form.mmse_puntaje.data if form.estado.data == "realizada" else None
    ev.observaciones = (form.observaciones.data or "").strip() or None
    if form.quitar_imagen.data:
        _borrar_imagen(ev)
    if form.imagen.data and getattr(form.imagen.data, "filename", ""):
        _guardar_imagen(ev, form.imagen.data)


def _marcar_atendida(alerta: Alerta, nota: str) -> None:
    alerta.estado = "atendida"
    alerta.fecha_atencion = datetime.utcnow()
    alerta.atendida_por_id = current_user.id
    alerta.nota_atencion = nota[:255]


def _nota_desde_evaluacion(ev: EvaluacionMMSE) -> str:
    if ev.estado == "realizada":
        return f"MMSE aplicado el {ev.fecha_realizada:%d/%m/%Y}: {ev.mmse_puntaje} puntos."
    if ev.estado == "programada":
        return f"MMSE programado para el {ev.fecha_programada:%d/%m/%Y}{' a las ' + ev.hora if ev.hora else ''}."
    return "Evaluación cancelada."


@dashboard_bp.route("/alertas/<int:alerta_id>/evaluaciones", methods=["POST"])
@login_required
def nueva_evaluacion(alerta_id: int):
    alerta = db.session.get(Alerta, alerta_id) or abort(404)
    form = EvaluacionMMSEForm()
    if not form.validate_on_submit():
        flash("Revisa los datos de la evaluación.", "warning")
        return _render_ficha(alerta, form)
    ev = EvaluacionMMSE(alerta_id=alerta.id, codigo=alerta.codigo, registrado_por_id=current_user.id)
    db.session.add(ev)
    db.session.flush()
    _aplicar_formulario(ev, form)
    if form.marcar_atendida.data and alerta.estado == "abierta":
        _marcar_atendida(alerta, _nota_desde_evaluacion(ev))
    log_access("evaluacion_mmse", f"alerta={alerta.id}; codigo={alerta.codigo}; estado={ev.estado}")
    db.session.commit()
    flash("Evaluación registrada.", "success")
    return redirect(url_for("dashboard.ficha_alerta", alerta_id=alerta.id))


@dashboard_bp.route("/alertas/<int:alerta_id>/evaluaciones/<int:evaluacion_id>/editar", methods=["GET", "POST"])
@login_required
def editar_evaluacion(alerta_id: int, evaluacion_id: int):
    alerta = db.session.get(Alerta, alerta_id) or abort(404)
    ev = db.session.get(EvaluacionMMSE, evaluacion_id) or abort(404)
    if ev.codigo != alerta.codigo:
        abort(404)
    form = EvaluacionMMSEForm(obj=ev) if request.method == "GET" else EvaluacionMMSEForm()
    if request.method == "GET":
        form.imagen.data = None
        return _render_ficha(alerta, form, editando=ev)
    if not form.validate_on_submit():
        flash("Revisa los datos de la evaluación.", "warning")
        return _render_ficha(alerta, form, editando=ev)
    _aplicar_formulario(ev, form)
    ev.fecha_actualizacion = datetime.utcnow()
    if form.marcar_atendida.data and alerta.estado == "abierta":
        _marcar_atendida(alerta, _nota_desde_evaluacion(ev))
    log_access("editar_evaluacion_mmse", f"evaluacion={ev.id}; estado={ev.estado}")
    db.session.commit()
    flash("Evaluación actualizada.", "success")
    return redirect(url_for("dashboard.ficha_alerta", alerta_id=alerta.id))


@dashboard_bp.route("/evaluaciones/<int:evaluacion_id>/eliminar", methods=["POST"])
@login_required
def eliminar_evaluacion(evaluacion_id: int):
    ev = db.session.get(EvaluacionMMSE, evaluacion_id) or abort(404)
    alerta_id = ev.alerta_id
    if AtenderAlertaForm().validate_on_submit():
        _borrar_imagen(ev)
        db.session.delete(ev)
        log_access("eliminar_evaluacion_mmse", f"evaluacion={evaluacion_id}")
        db.session.commit()
        flash("Evaluación eliminada.", "info")
    return redirect(url_for("dashboard.ficha_alerta", alerta_id=alerta_id) if alerta_id else url_for("dashboard.alertas"))


@dashboard_bp.route("/evaluaciones/<int:evaluacion_id>/imagen")
@login_required
def imagen_evaluacion(evaluacion_id: int):
    ev = db.session.get(EvaluacionMMSE, evaluacion_id) or abort(404)
    if not ev.imagen_archivo:
        abort(404)
    ruta = _carpeta_evaluaciones() / ev.imagen_archivo
    if not ruta.exists():
        abort(404)
    return send_file(ruta, download_name=ev.imagen_nombre or ev.imagen_archivo,
                     as_attachment=request.args.get("descargar") == "1")


@dashboard_bp.route("/alertas/<int:alerta_id>/atender", methods=["POST"])
@login_required
def atender_alerta(alerta_id: int):
    alerta = db.session.get(Alerta, alerta_id) or abort(404)
    form = AtenderAlertaForm()
    if form.validate_on_submit():
        _marcar_atendida(alerta, form.nota.data or "")
        log_access("atender_alerta", f"alerta={alerta.id}; codigo={alerta.codigo}")
        db.session.commit()
        flash(f"Alerta de {alerta.resultado.etiqueta if alerta.resultado else alerta.codigo} marcada como atendida.", "success")
    return redirect(request.referrer or url_for("dashboard.alertas"))


@dashboard_bp.route("/alertas/<int:alerta_id>/reabrir", methods=["POST"])
@login_required
def reabrir_alerta(alerta_id: int):
    alerta = db.session.get(Alerta, alerta_id) or abort(404)
    if AtenderAlertaForm().validate_on_submit():
        alerta.estado = "abierta"
        alerta.fecha_atencion = None
        alerta.atendida_por_id = None
        log_access("reabrir_alerta", f"alerta={alerta.id}")
        db.session.commit()
    return redirect(request.referrer or url_for("dashboard.alertas"))


# ---------------------------------------------------------------- Analítica (5.000 registros)
def _referencia() -> pd.DataFrame:
    if "referencia_df" not in current_app.extensions:
        df = referencia_dataframe()
        df["grupo_edad"] = grupo_edad(df["edad"])
        df["educacion"] = df["nivel_educativo"].map(NOMBRE_EDUCACION)
        df["gds_cat"] = np.where(df["gds"] > 5, "Sugiere depresión (> 5)", "Normal (0–5)")
        current_app.extensions["referencia_df"] = df
    return current_app.extensions["referencia_df"]


VISTAS = {"datos", "modelo"}


@dashboard_bp.route("/analitica")
@login_required
def analitica():
    vista = request.args.get("vista", "")
    modelo = obtener_modelo()
    base = _referencia()
    contexto = {"title": "Analítica", "vista": vista if vista in VISTAS else "", "modelo": modelo,
                "n_total": len(base), "m": modelo.metricas}
    if vista == "datos":
        contexto.update(_contexto_datos(base))
    elif vista == "modelo":
        contexto.update(_contexto_modelo(modelo))
    else:
        contexto["resumen"] = {
            "alto": (base.nivel_riesgo == 2).mean(),
            "mmse": base.mmse.mean(),
        }
    return render_template("analitica.html", **contexto)


def _contexto_datos(base: pd.DataFrame) -> dict:
    """Capítulo 4: análisis exploratorio del conjunto abierto, con filtros."""
    filtros = {k: request.args.get(k, "") for k in ("genero", "educacion", "grupo_edad", "estado_civil", "gds_cat")}
    df = base.copy()
    for campo, valor in filtros.items():
        if valor:
            df = df[df[campo] == valor]
    opciones = {
        "genero": sorted(base["genero"].unique()),
        "educacion": [NOMBRE_EDUCACION[k] for k in range(4)],
        "grupo_edad": [g for g in ORDEN_EDAD if g in set(base["grupo_edad"])],
        "estado_civil": sorted(base["estado_civil"].unique()),
        "gds_cat": ["Normal (0–5)", "Sugiere depresión (> 5)"],
    }
    n = len(df)
    contexto = {"filtros": filtros, "opciones": opciones, "n": n, "kpis": [], "charts": [], "tabla": [], "efectos": []}
    if n == 0:
        return contexto

    contexto["kpis"] = [
        {"titulo": "Registros", "valor": f"{n:,}".replace(",", "."), "detalle": f"de {len(base):,} en el conjunto abierto".replace(",", "."), "icono": "fa-database", "tono": "lila"},
        {"titulo": "Riesgo alto", "valor": f"{(df.nivel_riesgo == 2).mean():.1%}".replace(".", ","), "detalle": "MMSE ≤ 17", "icono": "fa-heart-pulse", "tono": "rosa"},
        {"titulo": "Riesgo medio", "valor": f"{(df.nivel_riesgo == 1).mean():.1%}".replace(".", ","), "detalle": "MMSE 18–23", "icono": "fa-triangle-exclamation", "tono": "ambar"},
        {"titulo": "MMSE medio", "valor": f"{df.mmse.mean():.1f}".replace(".", ","), "detalle": f"DE {df.mmse.std():.1f}".replace(".", ","), "icono": "fa-brain", "tono": "morado"},
        {"titulo": "GDS > 5", "valor": f"{(df.gds > 5).mean():.1%}".replace(".", ","), "detalle": "Síntomas depresivos", "icono": "fa-cloud-rain", "tono": "verde"},
        {"titulo": "Edad media", "valor": f"{df.edad.mean():.1f}".replace(".", ","), "detalle": "años", "icono": "fa-person-cane", "tono": "lila"},
    ]

    conteo = df["nivel_riesgo"].value_counts().reindex([0, 1, 2], fill_value=0)
    comp_edu = pd.crosstab(df["educacion"], df["nivel_riesgo"], normalize="index").reindex(
        [NOMBRE_EDUCACION[k] for k in range(4)]).reindex(columns=[0, 1, 2], fill_value=0).fillna(0) * 100
    alto_edad = df.groupby("grupo_edad")["nivel_riesgo"].apply(lambda s: (s == 2).mean() * 100).reindex(
        [g for g in ORDEN_EDAD if g in set(df["grupo_edad"])])
    cronicas = df["enfermedades_cronicas"].clip(upper=5).map(lambda v: "5 o más" if v >= 5 else str(int(v)))
    alto_cronicas = df.groupby(cronicas)["nivel_riesgo"].apply(lambda s: (s == 2).mean() * 100)
    alto_cronicas = alto_cronicas.reindex([c for c in ["0", "1", "2", "3", "4", "5 o más"] if c in alto_cronicas.index])
    hist = df["mmse"].value_counts().sort_index()
    charts = [
        {"id": "nivel", "titulo": "Distribución del nivel de riesgo", "tipo": "doughnut", "labels": list(NIVELES.values()),
         "datasets": [{"label": "Registros", "data": conteo.tolist(), "colores": [COLOR_NIVEL[k] for k in range(3)]}]},
        {"id": "edadAlto", "titulo": "% de riesgo alto por grupo de edad", "tipo": "bar",
         "labels": alto_edad.index.tolist(),
         "datasets": [{"label": "% riesgo alto", "data": alto_edad.round(1).tolist(), "color": "#B8486B"}],
         "opciones": {"porcentaje": True}},
        {"id": "mmse", "titulo": "Distribución del MMSE y puntos de corte", "tipo": "bar", "ancho": True,
         "labels": [str(int(v)) for v in hist.index],
         "datasets": [{"label": "Registros", "data": hist.tolist(),
                       "colores": [COLOR_NIVEL[0 if v >= 24 else 1 if v >= 18 else 2] for v in hist.index]}]},
        {"id": "educacion", "titulo": "Composición del riesgo según nivel educativo (%)", "tipo": "bar", "horizontal": True,
         "labels": comp_edu.index.tolist(),
         "datasets": [{"label": NIVELES[k], "data": comp_edu[k].round(1).tolist(), "color": COLOR_NIVEL[k]} for k in range(3)],
         "opciones": {"apilado": True, "porcentaje": True, "max": 100}},
        {"id": "cronicasAlto", "titulo": "% de riesgo alto por número de enfermedades crónicas", "tipo": "line",
         "labels": alto_cronicas.index.tolist(),
         "datasets": [{"label": "% riesgo alto", "data": alto_cronicas.round(1).tolist(), "color": "#6E4A82"}],
         "opciones": {"porcentaje": True}},
    ]

    efectos = []
    for col in NUMERICAS:
        grupos = [df.loc[df.nivel_riesgo == k, col] for k in range(3) if (df.nivel_riesgo == k).sum() > 1]
        if len(grupos) == 3:
            h, p = stats.kruskal(*grupos)
            eps = max((h - 3 + 1) / (n - 3), 0)
            efectos.append({"variable": NOMBRES_VARIABLES[col], "eps2": eps, "p": p,
                            "medianas": " / ".join(f"{df.loc[df.nivel_riesgo == k, col].median():g}".replace(".", ",") for k in range(3))})
    efectos.sort(key=lambda e: e["eps2"], reverse=True)
    if efectos:
        charts.append({"id": "efectos", "titulo": "Asociación de cada predictora con el nivel de riesgo (ε² de Kruskal-Wallis)",
                       "tipo": "bar", "horizontal": True, "ancho": True,
                       "labels": [e["variable"] for e in efectos],
                       "datasets": [{"label": "ε²", "data": [round(e["eps2"], 4) for e in efectos], "color": "#9D75AA"}],
                       "opciones": {"lineaUmbral": 0.06, "etiquetaUmbral": "efecto mediano"}})

    tabla = (
        df.groupby(["educacion", "grupo_edad"], observed=True)
        .agg(n=("mmse", "size"), alto=("nivel_riesgo", lambda s: (s == 2).mean()), mmse=("mmse", "mean"), gds=("gds", "mean"))
        .reset_index()
    )
    contexto.update(charts=charts, efectos=efectos, tabla=tabla.to_dict("records"))
    return contexto


def _contexto_modelo(modelo) -> dict:
    """Capítulo 5: comparación, hiperparametrización, umbral clínico y evaluación final."""
    m = modelo.metricas
    cv = [c for c in m.get("validacion_cruzada", [])]
    busqueda = m.get("busqueda", [])
    ganador = m.get("modelo_ganador")
    cv_por_modelo = {c["modelo"]: c for c in cv}
    color = lambda nombre: "#5E3B73" if nombre == ganador else ("#C9B3D3" if nombre != "Modelo ingenuo" else "#E1D9E5")
    charts = []
    if cv:
        charts.append({
            "id": "cvRecallAp", "titulo": "Validación cruzada: recall y AP de riesgo alto (media de 5 pliegues)",
            "tipo": "bar", "horizontal": True,
            "labels": [c["modelo"] for c in cv],
            "datasets": [
                {"label": "Recall alto", "data": [round(c["recall_alto_media"], 3) for c in cv], "color": "#B8486B"},
                {"label": "AP alto", "data": [round(c["ap_alto_media"], 3) for c in cv], "color": "#9D75AA"},
            ],
            "opciones": {"min": 0, "max": 1}})
    if busqueda:
        charts.append({
            "id": "busquedaAp", "titulo": "AP de riesgo alto antes y después de la búsqueda de hiperparámetros",
            "tipo": "bar", "horizontal": True,
            "labels": [b["modelo"] for b in busqueda],
            "datasets": [
                {"label": "Configuración inicial", "data": [round(cv_por_modelo.get(b["modelo"], {}).get("ap_alto_media", 0), 3) for b in busqueda], "color": "#C9B3D3"},
                {"label": "Mejor combinación", "data": [round(b["ap_media"], 3) for b in busqueda], "color": "#5E3B73"},
            ],
            "opciones": {"min": 0.3, "max": 0.55}})
    curva = m.get("curva_umbral", [])
    if curva:
        charts.append({
            "id": "curvaUmbral", "titulo": "Elección del umbral clínico sobre P(alto) (probabilidades fuera de pliegue)",
            "tipo": "scatter", "ancho": True,
            "datasets": [
                {"label": "Recall de riesgo alto", "puntos": [{"x": c["umbral"], "y": round(c["recall_alto"], 4)} for c in curva], "color": "#B8486B"},
                {"label": "Precisión de riesgo alto", "puntos": [{"x": c["umbral"], "y": round(c["precision_alto"], 4)} for c in curva if c["pct_marcados_alto"] >= 1], "color": "#9D75AA"},
                {"label": "Proporción marcada como alto", "puntos": [{"x": c["umbral"], "y": round(c["pct_marcados_alto"] / 100, 4)} for c in curva], "color": "#E8B04B", "punteada": True},
            ],
            "opciones": {"lineaX": m["umbral"], "etiquetaX": f"umbral clínico {m['umbral']:.2f}".replace(".", ","),
                         "lineaUmbral": m.get("recall_minimo", 0.8), "etiquetaUmbral": "recall 0,80",
                         "min": 0, "max": 1, "xmin": 0.05, "xmax": 0.75, "tituloX": "Umbral sobre P(alto)"}})
    if m.get("roc"):
        charts.append({
            "id": "roc", "titulo": f"Curva ROC · riesgo alto frente al resto (AUC = {m['auc_alto_prueba']:.3f})".replace(".", ","),
            "tipo": "scatter",
            "datasets": [
                {"label": "Modelo final", "puntos": m["roc"], "color": "#5E3B73", "relleno": True},
                {"label": "Clasificador aleatorio", "puntos": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "color": "#6F5F78", "punteada": True},
            ],
            "opciones": {"min": 0, "max": 1, "xmin": 0, "xmax": 1, "tituloX": "Tasa de falsos positivos", "tituloY": "Recall (sensibilidad)"}})
    if m.get("pr"):
        charts.append({
            "id": "pr", "titulo": f"Curva precisión-recall · riesgo alto (AP = {m['ap_alto_prueba']:.3f})".replace(".", ","),
            "tipo": "scatter",
            "datasets": [
                {"label": "Modelo final", "puntos": m["pr"], "color": "#5E3B73"},
                {"label": f"Umbral clínico {m['umbral']:.2f}".replace(".", ","), "puntos": [m["punto_clinico"]], "color": "#B8486B", "soloPunto": True},
            ],
            "opciones": {"min": 0, "max": 1, "xmin": 0, "xmax": 1, "tituloX": "Recall (sensibilidad)", "tituloY": "Precisión",
                         "lineaUmbral": m["prevalencia_alto_prueba"], "etiquetaUmbral": f"línea basal {m['prevalencia_alto_prueba']:.3f}".replace(".", ",")}})
    coef = sorted(m.get("coeficientes", []), key=lambda c: c["odds_ratio"])
    if coef:
        charts.append({
            "id": "oddsRatios", "titulo": "Odds ratios: riesgo alto frente a bajo", "tipo": "bar", "horizontal": True, "ancho": True,
            "labels": [c["nombre"] for c in coef],
            "datasets": [{"label": "Odds ratio", "data": [round(c["odds_ratio"], 2) for c in coef],
                          "colores": ["#B8486B" if c["odds_ratio"] > 1 else "#7FB685" for c in coef]}],
            "opciones": {"lineaUmbral": 1, "etiquetaUmbral": "OR = 1", "min": 0.4}})
    return {"charts": charts, "coeficientes": sorted(coef, key=lambda c: c["odds_ratio"], reverse=True)}
