from __future__ import annotations

from pathlib import Path

import click
from flask import Flask, current_app, redirect, render_template, url_for
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash

from app.config import Config

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()

NIVELES = {0: "Bajo", 1: "Medio", 2: "Alto"}


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
        Path(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Inicia sesión para continuar."
    login_manager.login_message_category = "warning"

    from app.models import Usuario

    @login_manager.user_loader
    def load_user(user_id: str) -> Usuario | None:
        return db.session.get(Usuario, int(user_id))

    from app.auth.routes import auth_bp
    from app.dashboard.routes import dashboard_bp
    from app.prediccion.routes import prediccion_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp, url_prefix="/panel")
    app.register_blueprint(prediccion_bp, url_prefix="/prediccion")

    @app.route("/")
    def index():
        return redirect(url_for("dashboard.principal"))

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template("error.html", codigo=403, mensaje="No tienes permiso para ver esta página."), 403

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", codigo=404, mensaje="La página que buscas no existe."), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", codigo=413, mensaje="El archivo supera el tamaño permitido (8 MB)."), 413

    @app.before_request
    def asegurar_modelo():
        # Carga el modelo (y actualiza cargas antiguas) antes de cualquier panel
        from flask import request
        from flask_login import current_user

        if request.endpoint and not request.endpoint.startswith(("static", "auth.")) and current_user.is_authenticated:
            obtener_modelo()

    @app.context_processor
    def alertas_pendientes():
        from flask_login import current_user

        from app.models import Alerta, CargaPrediccion

        if not current_user.is_authenticated:
            return {}
        ultima = CargaPrediccion.query.order_by(CargaPrediccion.fecha.desc()).first()
        n = Alerta.query.filter_by(carga_id=ultima.id, estado="abierta", severidad="alta").count() if ultima else 0
        return {"alertas_altas_abiertas": n}

    register_cli(app)
    register_template_filters(app)
    with app.app_context():
        db.create_all()
        _agregar_columnas_nuevas()
    return app


def _agregar_columnas_nuevas() -> None:
    """Agrega a una base ya creada las columnas incorporadas después (por ejemplo, el nombre del residente)."""
    from sqlalchemy import inspect, text

    columnas = {c["name"] for c in inspect(db.engine).get_columns("resultados_prediccion")}
    if "nombre" not in columnas:
        with db.engine.begin() as conexion:
            conexion.execute(text("ALTER TABLE resultados_prediccion ADD COLUMN nombre VARCHAR(120)"))


def obtener_modelo():
    """Carga el modelo una vez por proceso. Si falta, es de una versión anterior o no es
    compatible, lo entrena de nuevo; después actualiza las cargas guardadas con otra versión."""
    from app.ml import modelo as ml

    if "modelo_riesgo" not in current_app.extensions:
        ruta = Path(current_app.config["MODEL_PATH"])
        try:
            current_app.extensions["modelo_riesgo"] = ml.cargar(ruta)
        except Exception:  # archivo ausente, versión anterior u otra versión de scikit-learn
            current_app.logger.warning("Modelo no disponible o desactualizado en %s; se entrena de nuevo.", ruta)
            paquete = ml.entrenar(current_app.config["DATASET_REFERENCIA"], ruta)
            current_app.extensions["modelo_riesgo"] = ml.ModeloRiesgo(paquete)
        _actualizar_cargas(current_app.extensions["modelo_riesgo"])
    return current_app.extensions["modelo_riesgo"]


def _actualizar_cargas(modelo) -> None:
    from app.prediccion.services import recalcular_cargas

    try:
        n = recalcular_cargas(modelo)
        if n:
            current_app.logger.info("Cargas recalculadas con el modelo %s: %s", modelo.version, n)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.exception("No se pudieron recalcular las cargas anteriores.")


def register_template_filters(app: Flask) -> None:
    @app.template_filter("dec")
    def dec(value, digits: int = 3):
        if value is None:
            return "—"
        try:
            return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("pct")
    def pct(value, digits: int = 1):
        if value is None:
            return "—"
        return f"{float(value) * 100:.{digits}f} %".replace(".", ",")

    @app.template_filter("nivel")
    def nivel(value):
        return NIVELES.get(value, "—") if value is not None else "—"

    @app.template_filter("fecha")
    def fecha(value):
        return value.strftime("%d/%m/%Y %H:%M") if hasattr(value, "hour") else (value.strftime("%d/%m/%Y") if value else "—")


def register_cli(app: Flask) -> None:
    @app.cli.command("create-admin")
    def create_admin_command():
        """Crea o actualiza el administrador con las variables del entorno."""
        _crear_admin(app)

    @app.cli.command("entrenar-modelo")
    def entrenar_modelo_command():
        """Entrena el modelo con el conjunto abierto y fija el umbral de decisión."""
        from app.ml.modelo import entrenar

        paquete = entrenar(app.config["DATASET_REFERENCIA"], app.config["MODEL_PATH"], verbose=True)
        m = paquete["metricas"]
        click.echo(
            f"Modelo guardado en {app.config['MODEL_PATH']} | {m['modelo_ganador']} {m['hiperparametros']} | "
            f"umbral clínico P(alto) ≥ {paquete['umbral']:.2f} | recall alto prueba={m['prueba']['clinico']['recall_alto']:.3f} | "
            f"AUC alto={m['auc_alto_prueba']:.3f}"
        )

    @app.cli.command("recalcular-cargas")
    def recalcular_cargas_command():
        """Aplica el modelo vigente a las cargas guardadas con una versión anterior."""
        from app.ml.modelo import cargar
        from app.prediccion.services import recalcular_cargas

        n = recalcular_cargas(cargar(app.config["MODEL_PATH"]))
        click.echo(f"Cargas recalculadas: {n}")

    @app.cli.command("init-data")
    def init_data_command():
        """Crea las tablas, el admin, importa los 5.000 registros y entrena el modelo."""
        from app.data.services import importar_referencia
        from app.ml.modelo import entrenar

        db.create_all()
        _crear_admin(app)
        n = importar_referencia(app.config["DATASET_REFERENCIA"])
        click.echo(f"Registros de referencia importados: {n}")
        paquete = entrenar(app.config["DATASET_REFERENCIA"], app.config["MODEL_PATH"], verbose=True)
        click.echo(f"Modelo entrenado ({paquete['modelo']}). Umbral clínico: P(alto) ≥ {paquete['umbral']:.2f}")
        from app.ml.modelo import ModeloRiesgo
        from app.prediccion.services import recalcular_cargas

        click.echo(f"Cargas recalculadas: {recalcular_cargas(ModeloRiesgo(paquete))}")


def _crear_admin(app: Flask) -> None:
    from app.models import Usuario

    password = app.config["ADMIN_PASSWORD"]
    if not password:
        raise click.ClickException("Define ADMIN_PASSWORD en el archivo .env antes de crear el administrador.")
    user = Usuario.query.filter_by(username=app.config["ADMIN_USERNAME"]).first()
    accion = "actualizado"
    if user is None:
        user = Usuario(username=app.config["ADMIN_USERNAME"])
        db.session.add(user)
        accion = "creado"
    user.nombre_completo = app.config["ADMIN_FULL_NAME"]
    user.email = app.config["ADMIN_EMAIL"]
    user.password_hash = generate_password_hash(password)
    user.rol = "admin"
    user.estado = "activo"
    db.session.commit()
    click.echo(f"Usuario admin {accion}: {user.username}")
