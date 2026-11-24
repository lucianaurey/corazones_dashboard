"""Entrena el modelo sin conectarse a la base de datos.

Render lo ejecuta en el build (ver render.yaml): así el modelo ya está listo cuando
arranca el servicio y no se entrena dentro de una petición, que superaría el tiempo límite.
"""
from app.config import Config
from app.ml.modelo import entrenar

if __name__ == "__main__":
    paquete = entrenar(Config.DATASET_REFERENCIA, Config.MODEL_PATH, verbose=True)
    m = paquete["metricas"]
    print(
        f"Modelo guardado en {Config.MODEL_PATH} | {m['modelo_ganador']} {m['hiperparametros']} | "
        f"umbral clínico P(alto) >= {paquete['umbral']:.2f}"
    )
