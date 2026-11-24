# Dashboard Corazones de Alegría

Dashboard en Flask para estimar el riesgo de deterioro cognitivo de los residentes de la Residencia de Día "Corazones de Alegría" (Proyecto Integrador, UNIVALLE). Usa una regresión logística multinomial (C = 5, pesos de clase) con umbral clínico P(alto) ≥ 0,25.

## Ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Crea un archivo `.env` (no se sube a GitHub):

```
SECRET_KEY=una-clave-larga
ADMIN_PASSWORD=tu-contraseña
```

```bash
python entrenar_modelo.py   # entrena el modelo (≈ 1 min)
python run.py               # http://localhost:5000  (usuario: admin)
```

Al arrancar con una base vacía, la app crea el usuario `admin` con `ADMIN_PASSWORD` e importa los 5.000 registros de referencia.

## Publicar en Render

El repositorio incluye `render.yaml`, que crea el servicio web y una base Postgres gratuita.

1. En [Render](https://dashboard.render.com): **New → Blueprint** y elige este repositorio.
2. Render pide `ADMIN_PASSWORD`: escribe la contraseña del administrador.
3. **Deploy Blueprint**. El build instala las librerías y entrena el modelo; tarda unos minutos.
4. Entra a la URL `https://corazones-dashboard.onrender.com` (o la que asigne Render) con el usuario `admin`.

Notas del plan gratuito:

- El servicio se duerme tras 15 minutos sin visitas; la primera visita después tarda cerca de un minuto.
- El disco del servicio se borra en cada reinicio, por eso las cargas, alertas, evaluaciones MMSE y sus imágenes se guardan en Postgres.
- La base Postgres gratuita vence a los 30 días (hay 14 días más para pasarla a un plan pago antes de que se borre). Para la defensa, créala pocos días antes o vuelve a cargar el CSV en una base nueva.

## Privacidad

No subas al repositorio el CSV con los nombres reales de los residentes: cárgalo solo desde el dashboard. Los archivos de `data/` son el conjunto abierto y datos ficticios de demostración.
