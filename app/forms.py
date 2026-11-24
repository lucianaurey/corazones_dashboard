from __future__ import annotations

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import (
    BooleanField,
    DateField,
    DecimalField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import DataRequired, Length, NumberRange, Optional, Regexp


class LoginForm(FlaskForm):
    username = StringField("Usuario", validators=[DataRequired(), Length(max=80)])
    password = PasswordField("Contraseña", validators=[DataRequired(), Length(max=128)])
    remember = BooleanField("Recordarme")
    submit = SubmitField("Ingresar")


class CargaCsvForm(FlaskForm):
    csv_file = FileField(
        "Archivo CSV de la residencia",
        validators=[FileRequired("Selecciona un archivo."), FileAllowed(["csv"], "Solo se permiten archivos CSV.")],
    )
    descripcion = StringField("Descripción (opcional)", validators=[Optional(), Length(max=200)])
    submit = SubmitField("Ejecutar predicción")


class EvaluacionRapidaForm(FlaskForm):
    """Estimación individual, por ejemplo, para un residente nuevo aún no evaluado con el MMSE."""

    edad = IntegerField("Edad", validators=[DataRequired(), NumberRange(60, 110)])
    genero = SelectField("Género", choices=[("Femenino", "Femenino"), ("Masculino", "Masculino")])
    nivel_educativo = SelectField(
        "Nivel educativo",
        choices=[("0", "Ninguno"), ("1", "Primaria"), ("2", "Secundaria"), ("3", "Superior")],
    )
    estado_civil = SelectField(
        "Estado civil",
        choices=[("Casado", "Casado/a"), ("Viudo", "Viudo/a"), ("Soltero", "Soltero/a"), ("Divorciado", "Divorciado/a")],
    )
    enfermedades_cronicas = IntegerField("Enfermedades crónicas", validators=[Optional(), NumberRange(0, 15)])
    glucosa_mg_dl = DecimalField("Glucosa (mg/dL)", places=1, validators=[Optional(), NumberRange(40, 600)])
    imc = DecimalField("IMC (kg/m²)", places=1, validators=[Optional(), NumberRange(10, 70)])
    gds = IntegerField("Puntaje GDS-15", validators=[Optional(), NumberRange(0, 15)])
    calidad_sueno = IntegerField("Calidad de sueño (1–5)", validators=[Optional(), NumberRange(1, 5)])
    actividad_fisica = IntegerField("Actividad física (0–10)", validators=[Optional(), NumberRange(0, 10)])
    tabaquismo = SelectField("Tabaquismo", choices=[("No", "No"), ("Sí", "Sí")])
    consumo_alcohol = SelectField("Consumo de alcohol", choices=[("No", "No"), ("Sí", "Sí")])
    submit = SubmitField("Estimar riesgo")


class AtenderAlertaForm(FlaskForm):
    nota = StringField("Nota", validators=[Optional(), Length(max=255)])
    submit = SubmitField("Marcar como atendida")


class EvaluacionMMSEForm(FlaskForm):
    """Programación o registro de la evaluación MMSE que responde a una alerta."""

    estado = SelectField(
        "Estado",
        choices=[("programada", "Programada"), ("realizada", "Realizada"), ("cancelada", "Cancelada")],
        default="programada",
    )
    fecha_programada = DateField("Fecha programada", validators=[Optional()])
    hora = StringField("Hora", validators=[Optional(), Regexp(r"^([01]\d|2[0-3]):[0-5]\d$", message="Hora en formato HH:MM.")])
    lugar = StringField("Lugar", validators=[Optional(), Length(max=120)])
    responsable = StringField("Responsable de la evaluación", validators=[Optional(), Length(max=120)])
    fecha_realizada = DateField("Fecha en que se aplicó", validators=[Optional()])
    mmse_puntaje = IntegerField("Puntaje MMSE (0–30)", validators=[Optional(), NumberRange(0, 30, "El MMSE va de 0 a 30.")])
    observaciones = TextAreaField("Observaciones", validators=[Optional(), Length(max=2000)])
    imagen = FileField(
        "Imagen del resultado (foto o escaneo)",
        validators=[FileAllowed(["jpg", "jpeg", "png", "webp", "pdf"], "Solo se permiten imágenes JPG, PNG, WEBP o PDF.")],
    )
    quitar_imagen = BooleanField("Quitar la imagen actual")
    marcar_atendida = BooleanField("Marcar la alerta como atendida")
    submit = SubmitField("Guardar evaluación")

    def validate(self, extra_validators=None) -> bool:
        if not super().validate(extra_validators):
            return False
        ok = True
        if self.estado.data == "programada" and not self.fecha_programada.data:
            self.fecha_programada.errors.append("Indica la fecha en que se aplicará el MMSE.")
            ok = False
        if self.estado.data == "realizada":
            if not self.fecha_realizada.data:
                self.fecha_realizada.errors.append("Indica la fecha en que se aplicó el MMSE.")
                ok = False
            if self.mmse_puntaje.data is None:
                self.mmse_puntaje.errors.append("Registra el puntaje obtenido.")
                ok = False
        return ok
