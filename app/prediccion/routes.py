from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app import db, obtener_modelo
from app.auth.routes import log_access, role_required
from app.data.services import COLUMNAS_PLANTILLA, validar_csv_residencia
from app.forms import CargaCsvForm, EvaluacionRapidaForm
from app.models import CargaPrediccion, EvaluacionMMSE, ResultadoPrediccion
from app.prediccion.services import ACCIONES, ejecutar_prediccion, estimar_individual, resultados_dataframe

prediccion_bp = Blueprint("prediccion", __name__)

BASE_DATA = Path(__file__).resolve().parent.parent.parent / "data"


@prediccion_bp.route("/", methods=["GET"])
@login_required
def inicio():
    modelo = obtener_modelo()
    cargas = CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).limit(20).all()
    return render_template(
        "prediccion.html",
        title="Predicción",
        form=CargaCsvForm(),
        form_rapida=EvaluacionRapidaForm(),
        cargas=cargas,
        modelo=modelo,
        estimacion=None,
        columnas=COLUMNAS_PLANTILLA,
    )


@prediccion_bp.route("/cargar", methods=["POST"])
@login_required
def cargar():
    form = CargaCsvForm()
    if not form.validate_on_submit():
        for errores in form.errors.values():
            for e in errores:
                flash(e, "danger")
        return redirect(url_for("prediccion.inicio"))

    carpeta = Path(current_app.config["CSV_UPLOAD_FOLDER"])
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = secure_filename(form.csv_file.data.filename or "residentes.csv")
    temporal = carpeta / f"{uuid.uuid4().hex}_{nombre}"
    form.csv_file.data.save(temporal)
    try:
        validacion = validar_csv_residencia(temporal)
    finally:
        try:
            os.remove(temporal)  # el archivo original no se conserva en el servidor
        except OSError:
            current_app.logger.warning("No se pudo eliminar %s", temporal)

    if not validacion.ok:
        for e in validacion.errores:
            flash(e, "danger")
        return redirect(url_for("prediccion.inicio"))

    carga = ejecutar_prediccion(
        obtener_modelo(), validacion.datos, nombre, current_user.id,
        descripcion=form.descripcion.data or "", avisos=validacion.avisos,
        n_filas=validacion.n_filas, n_excluidos=validacion.n_excluidos,
    )
    log_access("prediccion", f"carga={carga.id}; archivo={nombre}; residentes={carga.n_residentes}")
    db.session.commit()
    flash(f"Predicción ejecutada para {carga.n_residentes} residentes.", "success")
    return redirect(url_for("prediccion.detalle", carga_id=carga.id))


@prediccion_bp.route("/rapida", methods=["POST"])
@login_required
def rapida():
    modelo = obtener_modelo()
    form = EvaluacionRapidaForm()
    estimacion = None
    if form.validate_on_submit():
        valores = {k: v for k, v in form.data.items() if k not in ("submit", "csrf_token")}
        valores["nivel_educativo"] = int(valores["nivel_educativo"])
        for k in ("glucosa_mg_dl", "imc"):
            valores[k] = float(valores[k]) if valores[k] is not None else None
        estimacion = estimar_individual(modelo, valores)
    else:
        flash("Revisa los valores del formulario: hay campos fuera de rango.", "warning")
    cargas = CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).limit(20).all()
    return render_template(
        "prediccion.html", title="Predicción", form=CargaCsvForm(), form_rapida=form,
        cargas=cargas, modelo=modelo, estimacion=estimacion, columnas=COLUMNAS_PLANTILLA,
    )


@prediccion_bp.route("/carga/<int:carga_id>")
@login_required
def detalle(carga_id: int):
    carga = db.session.get(CargaPrediccion, carga_id) or abort(404)
    resultados = carga.resultados.order_by(ResultadoPrediccion.orden_prioridad).all()
    return render_template(
        "resultado_carga.html", title=f"Carga #{carga.id}", carga=carga,
        resultados=resultados, acciones=ACCIONES, modelo=obtener_modelo(),
    )


@prediccion_bp.route("/carga/<int:carga_id>/csv")
@login_required
def descargar(carga_id: int):
    carga = db.session.get(CargaPrediccion, carga_id) or abort(404)
    contenido = resultados_dataframe(carga).to_csv(index=False).encode("utf-8-sig")
    return Response(
        contenido, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=prediccion_carga_{carga.id}.csv"},
    )


@prediccion_bp.route("/carga/<int:carga_id>/eliminar", methods=["POST"])
@login_required
@role_required("admin")
def eliminar(carga_id: int):
    carga = db.session.get(CargaPrediccion, carga_id) or abort(404)
    ids = [a.id for a in carga.alertas]
    if ids:  # las evaluaciones MMSE se conservan en el historial del residente
        EvaluacionMMSE.query.filter(EvaluacionMMSE.alerta_id.in_(ids)).update({"alerta_id": None}, synchronize_session=False)
    db.session.delete(carga)
    log_access("eliminar_carga", f"carga={carga_id}")
    db.session.commit()
    flash(f"Carga #{carga_id} eliminada con sus resultados y alertas.", "info")
    return redirect(url_for("prediccion.inicio"))


@prediccion_bp.route("/plantilla.csv")
@login_required
def plantilla():
    return send_file(BASE_DATA / "plantilla_residentes.csv", as_attachment=True, download_name="plantilla_residentes.csv")


@prediccion_bp.route("/demo.csv")
@login_required
def demo():
    return send_file(BASE_DATA / "demo_residentes_ficticio.csv", as_attachment=True, download_name="demo_residentes_ficticio.csv")
