from __future__ import annotations

import json
from datetime import datetime

from flask_login import UserMixin
from sqlalchemy.orm import deferred

from app import db


class Usuario(UserMixin, db.Model):
    __tablename__ = "usuarios"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    nombre_completo = db.Column(db.String(160), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    rol = db.Column(db.String(20), nullable=False, default="personal", index=True)  # admin | personal
    estado = db.Column(db.String(20), nullable=False, default="activo", index=True)
    fecha_creacion = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    ultimo_acceso = db.Column(db.DateTime)

    accesos = db.relationship("BitacoraAcceso", back_populates="usuario", lazy="dynamic")

    @property
    def is_active(self) -> bool:
        return self.estado == "activo"


class BitacoraAcceso(db.Model):
    __tablename__ = "bitacora_acceso"

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=True, index=True)
    fecha_hora = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    ip = db.Column(db.String(64))
    accion = db.Column(db.String(80), nullable=False)
    detalle = db.Column(db.String(255))

    usuario = db.relationship("Usuario", back_populates="accesos")


class RegistroReferencia(db.Model):
    """Conjunto abierto de Kaggle ya limpio (Fase 1). Alimenta el panel de analítica."""

    __tablename__ = "registros_referencia"

    id = db.Column(db.Integer, primary_key=True)
    id_participante = db.Column(db.String(20), unique=True, nullable=False, index=True)
    edad = db.Column(db.Integer, nullable=False, index=True)
    genero = db.Column(db.String(20), nullable=False, index=True)
    nivel_educativo = db.Column(db.Integer, nullable=False, index=True)
    estado_civil = db.Column(db.String(20), nullable=False, index=True)
    enfermedades_cronicas = db.Column(db.Integer, nullable=False)
    glucosa_mg_dl = db.Column(db.Float, nullable=False)
    imc = db.Column(db.Float, nullable=False)
    mmse = db.Column(db.Integer, nullable=False)
    gds = db.Column(db.Integer, nullable=False)
    calidad_sueno = db.Column(db.Integer, nullable=False)
    actividad_fisica = db.Column(db.Integer, nullable=False)
    tabaquismo = db.Column(db.String(5), nullable=False)
    consumo_alcohol = db.Column(db.String(5), nullable=False)
    nivel_riesgo = db.Column(db.Integer, nullable=False, index=True)  # 0 bajo, 1 medio, 2 alto


class CargaPrediccion(db.Model):
    """Cada archivo de la residencia que se procesa con el modelo (Fase 2)."""

    __tablename__ = "cargas_prediccion"

    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=True)
    archivo = db.Column(db.String(200), nullable=False)
    descripcion = db.Column(db.String(200))
    n_filas = db.Column(db.Integer, nullable=False, default=0)
    n_residentes = db.Column(db.Integer, nullable=False, default=0)
    n_excluidos = db.Column(db.Integer, nullable=False, default=0)
    umbral = db.Column(db.Float, nullable=False)
    version_modelo = db.Column(db.String(60), nullable=False)
    evaluacion_json = db.Column(db.Text)  # comparación con el MMSE observado, si existe
    avisos_json = db.Column(db.Text)

    usuario = db.relationship("Usuario")
    resultados = db.relationship("ResultadoPrediccion", back_populates="carga", cascade="all, delete-orphan", lazy="dynamic")
    alertas = db.relationship("Alerta", back_populates="carga", cascade="all, delete-orphan", lazy="dynamic")

    @property
    def evaluacion(self) -> dict | None:
        return json.loads(self.evaluacion_json) if self.evaluacion_json else None

    @property
    def avisos(self) -> list[str]:
        return json.loads(self.avisos_json) if self.avisos_json else []


class ResultadoPrediccion(db.Model):
    __tablename__ = "resultados_prediccion"

    id = db.Column(db.Integer, primary_key=True)
    carga_id = db.Column(db.Integer, db.ForeignKey("cargas_prediccion.id"), nullable=False, index=True)
    codigo = db.Column(db.String(20), nullable=False, index=True)
    nombre = db.Column(db.String(120))
    fecha_evaluacion = db.Column(db.Date)
    edad = db.Column(db.Float)
    genero = db.Column(db.String(20))
    nivel_educativo = db.Column(db.Float)
    estado_civil = db.Column(db.String(20))
    enfermedades_cronicas = db.Column(db.Float)
    glucosa_mg_dl = db.Column(db.Float)
    imc = db.Column(db.Float)
    gds = db.Column(db.Float)
    calidad_sueno = db.Column(db.Float)
    actividad_fisica = db.Column(db.Float)
    tabaquismo = db.Column(db.String(5))
    consumo_alcohol = db.Column(db.String(5))
    mmse_observado = db.Column(db.Float)
    mmse_anterior = db.Column(db.Float)
    nivel_observado = db.Column(db.Integer)
    p_bajo = db.Column(db.Float, nullable=False)
    p_medio = db.Column(db.Float, nullable=False)
    p_alto = db.Column(db.Float, nullable=False)
    puntaje = db.Column(db.Float, nullable=False, index=True)
    nivel_estimado = db.Column(db.Integer, nullable=False, index=True)
    orden_prioridad = db.Column(db.Integer, nullable=False)
    factores_json = db.Column(db.Text)
    imputadas = db.Column(db.String(255))

    carga = db.relationship("CargaPrediccion", back_populates="resultados")

    @property
    def etiqueta(self) -> str:
        return f"{self.codigo} · {self.nombre}" if self.nombre else self.codigo

    @property
    def etiqueta_corta(self) -> str:
        """Código + nombre + primer apellido, para las etiquetas de los gráficos."""
        if not self.nombre:
            return self.codigo
        return f"{self.codigo} · {' '.join(self.nombre.split()[:2])}"

    @property
    def factores(self) -> list[dict]:
        return json.loads(self.factores_json) if self.factores_json else []

    @property
    def lista_imputadas(self) -> list[str]:
        return [v for v in (self.imputadas or "").split(";") if v]

    @property
    def prioridad(self) -> int:
        """1 muy alta (P(alto) ≥ 0,70) · 2 alta (riesgo alto) · 3 media · 4 baja (cuaderno del cap. 5)."""
        if self.nivel_estimado == 2 and self.p_alto >= 0.70:
            return 1
        return {2: 2, 1: 3, 0: 4}[self.nivel_estimado]

    @property
    def prioridad_texto(self) -> str:
        return {1: "Muy alta", 2: "Alta", 3: "Media", 4: "Baja"}[self.prioridad]


class Alerta(db.Model):
    __tablename__ = "alertas"

    id = db.Column(db.Integer, primary_key=True)
    carga_id = db.Column(db.Integer, db.ForeignKey("cargas_prediccion.id"), nullable=False, index=True)
    resultado_id = db.Column(db.Integer, db.ForeignKey("resultados_prediccion.id"), nullable=True)
    codigo = db.Column(db.String(20), nullable=False, index=True)
    tipo = db.Column(db.String(60), nullable=False, index=True)
    severidad = db.Column(db.String(10), nullable=False, index=True)  # alta | media | baja
    mensaje = db.Column(db.String(400), nullable=False)
    accion_sugerida = db.Column(db.String(255))
    estado = db.Column(db.String(15), nullable=False, default="abierta", index=True)  # abierta | atendida
    fecha_creacion = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    fecha_atencion = db.Column(db.DateTime)
    atendida_por_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=True)
    nota_atencion = db.Column(db.String(255))

    carga = db.relationship("CargaPrediccion", back_populates="alertas")
    resultado = db.relationship("ResultadoPrediccion")
    atendida_por = db.relationship("Usuario")
    evaluaciones = db.relationship("EvaluacionMMSE", back_populates="alerta", lazy="dynamic",
                                   order_by="EvaluacionMMSE.fecha_registro.desc()")

    @property
    def ultima_evaluacion(self) -> "EvaluacionMMSE | None":
        return self.evaluaciones.first()


class EvaluacionMMSE(db.Model):
    """Seguimiento de una alerta: evaluación MMSE programada o realizada en la residencia."""

    __tablename__ = "evaluaciones_mmse"

    id = db.Column(db.Integer, primary_key=True)
    alerta_id = db.Column(db.Integer, db.ForeignKey("alertas.id", ondelete="SET NULL"), nullable=True, index=True)
    codigo = db.Column(db.String(20), nullable=False, index=True)
    estado = db.Column(db.String(15), nullable=False, default="programada", index=True)  # programada | realizada | cancelada
    fecha_programada = db.Column(db.Date)
    hora = db.Column(db.String(5))
    lugar = db.Column(db.String(120))
    responsable = db.Column(db.String(120))
    fecha_realizada = db.Column(db.Date)
    mmse_puntaje = db.Column(db.Integer)
    observaciones = db.Column(db.Text)
    imagen_archivo = db.Column(db.String(200))      # nombre interno en instance/evaluaciones
    imagen_nombre = db.Column(db.String(200))       # nombre original del archivo subido
    # Contenido del archivo guardado en la base: en Render el disco se borra en cada reinicio.
    # deferred: solo se lee cuando se pide la imagen, no al listar evaluaciones.
    imagen_datos = deferred(db.Column(db.LargeBinary))
    imagen_tipo = db.Column(db.String(60))          # tipo MIME (image/jpeg, application/pdf, ...)
    registrado_por_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=True)
    fecha_registro = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    fecha_actualizacion = db.Column(db.DateTime)

    alerta = db.relationship("Alerta", back_populates="evaluaciones")
    registrado_por = db.relationship("Usuario")

    @property
    def nivel_mmse(self) -> int | None:
        """Nivel según la regla de la Tabla 2 a partir del puntaje registrado."""
        if self.mmse_puntaje is None:
            return None
        return 0 if self.mmse_puntaje >= 24 else 1 if self.mmse_puntaje >= 18 else 2

    @property
    def es_imagen(self) -> bool:
        return bool(self.imagen_archivo) and not self.imagen_archivo.lower().endswith(".pdf")
