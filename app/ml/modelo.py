"""Entrenamiento, umbral de decisión y aplicación del modelo (capítulo 5).

Reproduce el cuaderno del capítulo 5:
1. Partición estratificada 80/20.
2. Cuatro algoritmos sin balanceo y con pesos de clase (comparación en prueba).
3. Validación cruzada estratificada de 5 pliegues (modelo ingenuo + 4 modelos con pesos).
4. Hiperparametrización con GridSearchCV (refit = Average Precision de riesgo alto).
5. Selección del ganador técnico (mayor AP; competitivos a 0,01 o menos).
6. Umbral clínico sobre P(alto): el más alto con recall de riesgo alto ≥ 80 %,
   buscado con las probabilidades fuera de pliegue del conjunto de entrenamiento.
7. Evaluación final en prueba con el umbral tradicional (0,50) y el clínico.

El modelo se entrena una sola vez con el conjunto abierto (Fase 1) y se guarda en un
archivo .joblib. En la residencia (Fase 2) solo se aplica: no se reentrena ni se
modifica el umbral.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.preprocesamiento import (
    CATEGORICAS,
    ETIQUETAS,
    NOMBRES_VARIABLES,
    NUMERICAS,
    PREDICTORAS,
    limpiar_kaggle,
)

RANDOM_STATE = 42
CLASE_ALTO = 2
RECALL_MINIMO = 0.80           # criterio del umbral clínico (sección 3.8)
UMBRAL_TRADICIONAL = 0.50
UMBRAL_MUY_ALTA = 0.70         # prioridad "muy alta" (cuaderno del capítulo 5, sección 13)
MINUTOS_MMSE = 45
HORAS_MEDICO_SEMANA = 8
VERSION_MODELO = "cap5-gridsearch-umbral-alto-2.0"

# Compatibilidad con el nombre anterior
SENSIBILIDAD_OBJETIVO = RECALL_MINIMO


def construir_preprocesador() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numericas", Pipeline([("imputacion_mediana", SimpleImputer(strategy="median")),
                                    ("escalado", StandardScaler())]), NUMERICAS),
            ("categoricas", Pipeline([("imputacion_moda", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                               sparse_output=False))]), CATEGORICAS),
        ]
    )


def modelos_base(pesos: bool) -> dict[str, object]:
    """Configuración inicial de los cuatro algoritmos (Tabla 17)."""
    cw = "balanced" if pesos else None
    return {
        "Regresión logística": LogisticRegression(max_iter=1500, class_weight=cw, random_state=RANDOM_STATE),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, class_weight=cw,
                                                    random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=11, min_samples_leaf=10,
                                                class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),  # KNN no admite pesos de clase
    }


REJILLAS = {
    "Regresión logística": {"modelo__C": [0.01, 0.10, 1.00, 5.00]},
    "Árbol de decisión": {"modelo__max_depth": [3, 4, 6, 8], "modelo__min_samples_leaf": [20, 40]},
    "Random Forest": {"modelo__n_estimators": [150, 250], "modelo__max_depth": [8, 12, 20],
                      "modelo__min_samples_leaf": [5, 10]},
    "KNN": {"modelo__n_neighbors": [9, 15, 25, 35], "modelo__weights": ["uniform", "distance"], "modelo__p": [2]},
}


def _pipeline(algoritmo) -> Pipeline:
    return Pipeline([("preprocesador", construir_preprocesador()), ("modelo", clone(algoritmo))])


def metricas_clasificacion(y_real, predicciones, probabilidades) -> dict:
    y_real = np.asarray(y_real)
    return {
        "exactitud": float(accuracy_score(y_real, predicciones)),
        "precision_alto": float(precision_score(y_real, predicciones, labels=[CLASE_ALTO], average="macro", zero_division=0)),
        "recall_alto": float(recall_score(y_real, predicciones, labels=[CLASE_ALTO], average="macro", zero_division=0)),
        "f1_alto": float(f1_score(y_real, predicciones, labels=[CLASE_ALTO], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_real, predicciones, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_real, probabilidades, multi_class="ovr")),
        "ap_alto": float(average_precision_score((y_real == CLASE_ALTO).astype(int), probabilidades[:, CLASE_ALTO])),
    }


def metricas_binarias(pred: np.ndarray, real: np.ndarray) -> dict:
    """Sensibilidad, especificidad y demás para una clase frente al resto (pred y real en 0/1)."""
    pred = np.asarray(pred).astype(int)
    real = np.asarray(real).astype(int)
    tp = int(((pred == 1) & (real == 1)).sum())
    fn = int(((pred == 0) & (real == 1)).sum())
    tn = int(((pred == 0) & (real == 0)).sum())
    fp = int(((pred == 1) & (real == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    esp = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "sensibilidad": sens,
        "especificidad": esp,
        "exactitud_balanceada": (sens + esp) / 2,
        "vpp": tp / (tp + fp) if tp + fp else float("nan"),
        "vpn": tn / (tn + fn) if tn + fn else float("nan"),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "pct_senalados": float(pred.mean() * 100) if len(pred) else 0.0,
    }


def asignar_nivel(proba: np.ndarray, umbral: float) -> np.ndarray:
    """Alto si P(alto) ≥ umbral; si no, bajo o medio, el que tenga mayor probabilidad."""
    resto = np.where(proba[:, 1] >= proba[:, 0], 1, 0)
    return np.where(proba[:, CLASE_ALTO] >= umbral, CLASE_ALTO, resto)


def _resumen_umbral(p_alto: np.ndarray, y_alto: np.ndarray, umbral: float) -> dict:
    pred = (p_alto >= umbral).astype(int)
    return {
        "umbral": float(umbral),
        "recall_alto": float(recall_score(y_alto, pred, zero_division=0)),
        "precision_alto": float(precision_score(y_alto, pred, zero_division=0)),
        "pct_marcados_alto": float(pred.mean() * 100),
    }


def _evaluar_en_prueba(proba: np.ndarray, y_test: np.ndarray, umbral: float) -> dict:
    pred = asignar_nivel(proba, umbral)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    por_clase = []
    for k in range(3):
        por_clase.append({
            "clase": ETIQUETAS[k],
            "n": int((y_test == k).sum()),
            "precision": float(precision_score(y_test, pred, labels=[k], average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, pred, labels=[k], average="macro", zero_division=0)),
            "f1": float(f1_score(y_test, pred, labels=[k], average="macro", zero_division=0)),
        })
    binaria = metricas_binarias(pred == CLASE_ALTO, y_test == CLASE_ALTO)
    pct = float((pred == CLASE_ALTO).mean() * 100)
    return {
        "umbral": float(umbral),
        **metricas_clasificacion(y_test, pred, proba),
        "por_clase": por_clase,
        "precision_macro": float(np.mean([c["precision"] for c in por_clase])),
        "recall_macro": float(np.mean([c["recall"] for c in por_clase])),
        "exactitud_balanceada": float(np.mean([c["recall"] for c in por_clase])),
        "especificidad_alto": binaria["especificidad"],
        "detectados_alto": int(matriz[2, 2]),
        "total_alto": int(matriz[2].sum()),
        "falsas_alarmas": int(matriz[:, 2].sum() - matriz[2, 2]),
        "pct_marcados_alto": pct,
        "horas_mmse_por_100": pct * MINUTOS_MMSE / 60,
        "matriz": matriz.tolist(),
    }


def _reducir_curva(x: np.ndarray, y: np.ndarray, maximo: int = 120) -> list[dict]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(maximo, len(x))).astype(int))
    return [{"x": round(float(x[i]), 4), "y": round(float(y[i]), 4)} for i in idx]


def entrenar(ruta_csv: str | Path, ruta_salida: str | Path, comparar_modelos: bool = True, verbose: bool = False) -> dict:
    """Entrena como el cuaderno del capítulo 5 y guarda el pipeline y sus métricas."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    datos = limpiar_kaggle(pd.read_csv(ruta_csv))
    X = datos[PREDICTORAS]
    y = datos["nivel_riesgo"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    y_test_arr = y_test.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Entrenamiento tradicional en prueba: sin balanceo y con pesos de clase (Tabla 18)
    comparacion_balanceo = []
    if comparar_modelos:
        log("Comparando estrategias de balanceo…")
        for pesos, estrategia in ((False, "Sin balanceo"), (True, "Pesos de clase")):
            for nombre, algoritmo in modelos_base(pesos).items():
                if pesos and nombre == "KNN":
                    continue  # KNN no admite pesos: su fila es la misma
                pipe = _pipeline(algoritmo)
                inicio = time.perf_counter()
                pipe.fit(X_train, y_train)
                tiempo = time.perf_counter() - inicio
                proba = pipe.predict_proba(X_test)
                pred = pipe.predict(X_test)
                comparacion_balanceo.append({
                    "modelo": nombre,
                    "estrategia": "Sin pesos" if nombre == "KNN" else estrategia,
                    **metricas_clasificacion(y_test_arr, pred, proba),
                    "tp_alto": int(((pred == 2) & (y_test_arr == 2)).sum()),
                    "tiempo_seg": tiempo,
                })
        orden = ["Regresión logística", "Árbol de decisión", "Random Forest", "KNN"]
        comparacion_balanceo.sort(key=lambda f: (orden.index(f["modelo"]), f["estrategia"] != "Sin balanceo"))

    # 2. Validación cruzada: modelo ingenuo + cuatro modelos con pesos (Tabla 19)
    validacion_cruzada = []
    if comparar_modelos:
        log("Validación cruzada de 5 pliegues…")
        estudio = {"Modelo ingenuo": DummyClassifier(strategy="most_frequent"), **modelos_base(True)}
        for nombre, algoritmo in estudio.items():
            oof = np.zeros((len(X_train), 3))
            folds = []
            for idx_fit, idx_val in cv.split(X_train, y_train):
                pipe = _pipeline(algoritmo)
                pipe.fit(X_train.iloc[idx_fit], y_train.iloc[idx_fit])
                proba = pipe.predict_proba(X_train.iloc[idx_val])
                oof[idx_val] = proba
                folds.append(metricas_clasificacion(y_train.iloc[idx_val], pipe.predict(X_train.iloc[idx_val]), proba))
            folds = pd.DataFrame(folds)
            fila = {"modelo": nombre,
                    "ap_alto_oof": float(average_precision_score((y_train.to_numpy() == 2).astype(int), oof[:, 2]))}
            for m in folds.columns:
                fila[f"{m}_media"] = float(folds[m].mean())
                fila[f"{m}_de"] = float(folds[m].std(ddof=0))
            validacion_cruzada.append(fila)
        validacion_cruzada.sort(key=lambda f: f["ap_alto_oof"], reverse=True)

    # 3. Hiperparametrización con GridSearchCV (Tabla 20)
    def scorer_ap_alto(estimador, Xv, yv):
        p_alto = estimador.predict_proba(Xv)[:, list(estimador.classes_).index(CLASE_ALTO)]
        return average_precision_score((np.asarray(yv) == CLASE_ALTO).astype(int), p_alto)

    def scorer_recall_alto(estimador, Xv, yv):
        return recall_score(yv, estimador.predict(Xv), labels=[CLASE_ALTO], average="macro", zero_division=0)

    metricas_grid = {"ap_alto": scorer_ap_alto, "recall_alto": scorer_recall_alto, "f1_macro": "f1_macro"}
    busqueda, mejores = [], {}
    log("Búsqueda de hiperparámetros…")
    for nombre, algoritmo in modelos_base(True).items():
        grid = GridSearchCV(_pipeline(algoritmo), param_grid=REJILLAS[nombre], scoring=metricas_grid,
                            refit="ap_alto", cv=cv, n_jobs=-1)
        inicio = time.perf_counter()
        grid.fit(X_train, y_train)
        tiempo = time.perf_counter() - inicio
        i = grid.best_index_
        res = grid.cv_results_
        mejores[nombre] = grid.best_estimator_
        busqueda.append({
            "modelo": nombre,
            "rejilla": {k.replace("modelo__", ""): v for k, v in REJILLAS[nombre].items()},
            "mejores": {k.replace("modelo__", ""): (v.item() if hasattr(v, "item") else v)
                        for k, v in grid.best_params_.items()},
            "ap_media": float(res["mean_test_ap_alto"][i]),
            "ap_de": float(res["std_test_ap_alto"][i]),
            "recall_alto": float(res["mean_test_recall_alto"][i]),
            "f1_macro": float(res["mean_test_f1_macro"][i]),
            "combinaciones": int(len(res["params"])),
            "tiempo_seg": float(tiempo),
        })
        log(f"  {nombre}: AP={busqueda[-1]['ap_media']:.4f} {busqueda[-1]['mejores']}")
    busqueda.sort(key=lambda f: f["ap_media"], reverse=True)

    # 4. Selección: ganador técnico y competitivos (a 0,01 o menos)
    ganador = busqueda[0]["modelo"]
    mejor_ap = busqueda[0]["ap_media"]
    competitivos = [f["modelo"] for f in busqueda if f["ap_media"] >= mejor_ap - 0.01]
    pipeline = clone(mejores[ganador])

    # 5. Umbral clínico con probabilidades fuera de pliegue (sin tocar la prueba)
    oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)
    p_alto_oof = oof[:, CLASE_ALTO]
    y_alto_train = (y_train.to_numpy() == CLASE_ALTO).astype(int)
    candidatos = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curva_umbral = [_resumen_umbral(p_alto_oof, y_alto_train, u) for u in candidatos]
    validos = [c["umbral"] for c in curva_umbral if c["recall_alto"] >= RECALL_MINIMO]
    umbral = float(max(validos)) if validos else float(candidatos[0])

    # 6. Modelo final y evaluación en prueba con los dos umbrales
    pipeline.fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)
    y_alto_test = (y_test_arr == CLASE_ALTO).astype(int)
    p_alto_test = proba_test[:, CLASE_ALTO]
    fpr, tpr, _ = roc_curve(y_alto_test, p_alto_test)
    prec_c, rec_c, _ = precision_recall_curve(y_alto_test, p_alto_test)
    pred_cli = (p_alto_test >= umbral).astype(int)

    coeficientes = []
    modelo = pipeline.named_steps["modelo"]
    prep = pipeline.named_steps["preprocesador"]
    if isinstance(modelo, LogisticRegression):
        columnas = [c.split("__", 1)[1] for c in prep.get_feature_names_out()]
        escala = prep.named_transformers_["numericas"].named_steps["escalado"].scale_
        coef = modelo.coef_[2] - modelo.coef_[0]
        coeficientes = [
            {
                "variable": columna,
                "nombre": NOMBRES_VARIABLES.get(columna, columna),
                "tipo": "Numérica" if i < len(NUMERICAS) else "Categórica",
                "unidad": f"1 DE ({escala[i]:.2f})".replace(".", ",") if i < len(NUMERICAS) else "vs. referencia",
                "coeficiente": float(coef[i]),
                "odds_ratio": float(np.exp(coef[i])),
            }
            for i, columna in enumerate(columnas)
        ]

    distribucion = {
        "total": [int((y == k).sum()) for k in range(3)],
        "entrenamiento": [int((y_train == k).sum()) for k in range(3)],
        "prueba": [int((y_test == k).sum()) for k in range(3)],
    }
    metricas = {
        "n_total": int(len(datos)),
        "n_entrenamiento": int(len(X_train)),
        "n_prueba": int(len(X_test)),
        "distribucion": distribucion,
        "modelo_ganador": ganador,
        "hiperparametros": busqueda[0]["mejores"],
        "competitivos": competitivos,
        "umbral": umbral,
        "umbral_tradicional": UMBRAL_TRADICIONAL,
        "recall_minimo": RECALL_MINIMO,
        "comparacion_balanceo": comparacion_balanceo,
        "validacion_cruzada": validacion_cruzada,
        "busqueda": busqueda,
        "umbral_cv": {
            "tradicional": _resumen_umbral(p_alto_oof, y_alto_train, UMBRAL_TRADICIONAL),
            "clinico": _resumen_umbral(p_alto_oof, y_alto_train, umbral),
        },
        "curva_umbral": curva_umbral,
        "prueba": {
            "tradicional": _evaluar_en_prueba(proba_test, y_test_arr, UMBRAL_TRADICIONAL),
            "clinico": _evaluar_en_prueba(proba_test, y_test_arr, umbral),
        },
        "auc_alto_prueba": float(roc_auc_score(y_alto_test, p_alto_test)),
        "ap_alto_prueba": float(average_precision_score(y_alto_test, p_alto_test)),
        "auc_macro_prueba": float(roc_auc_score(y_test_arr, proba_test, multi_class="ovr")),
        "prevalencia_alto_prueba": float(y_alto_test.mean()),
        "roc": _reducir_curva(fpr, tpr),
        "pr": _reducir_curva(rec_c[::-1], prec_c[::-1]),
        "punto_clinico": {"x": float(recall_score(y_alto_test, pred_cli)),
                          "y": float(precision_score(y_alto_test, pred_cli, zero_division=0))},
        "coeficientes": coeficientes,
        "minutos_mmse": MINUTOS_MMSE,
        "horas_medico_semana": HORAS_MEDICO_SEMANA,
    }

    paquete = {
        "pipeline": pipeline,
        "umbral": umbral,
        "predictoras": PREDICTORAS,
        "version": VERSION_MODELO,
        "modelo": ganador,
        "sklearn": sklearn.__version__,
        "entrenado_en": datetime.now().isoformat(timespec="seconds"),
        "metricas": metricas,
    }
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta_salida)
    ruta_salida.with_suffix(".json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    return paquete


@dataclass
class ModeloRiesgo:
    paquete: dict

    @property
    def pipeline(self) -> Pipeline:
        return self.paquete["pipeline"]

    @property
    def umbral(self) -> float:
        return float(self.paquete["umbral"])

    @property
    def version(self) -> str:
        return self.paquete.get("version", "modelo")

    @property
    def nombre(self) -> str:
        return self.paquete.get("modelo", "Regresión logística")

    @property
    def metricas(self) -> dict:
        return self.paquete["metricas"]

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(X[PREDICTORAS])
        nivel = asignar_nivel(proba, self.umbral)
        salida = pd.DataFrame(
            {
                "p_bajo": proba[:, 0],
                "p_medio": proba[:, 1],
                "p_alto": proba[:, 2],
                "puntaje": proba[:, 2],  # el umbral clínico se aplica sobre P(alto)
                "nivel_estimado": nivel,
            },
            index=X.index,
        )
        salida["factores"] = self.explicar(X)
        return salida

    def explicar(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Aporte de cada variable al logit de riesgo alto frente a bajo (valor estandarizado × coeficiente)."""
        modelo = self.pipeline.named_steps["modelo"]
        if not hasattr(modelo, "coef_"):
            return [[] for _ in range(len(X))]
        prep = self.pipeline.named_steps["preprocesador"]
        matriz = prep.transform(X[PREDICTORAS])
        matriz = matriz.toarray() if hasattr(matriz, "toarray") else np.asarray(matriz)
        columnas = [c.split("__", 1)[1] for c in prep.get_feature_names_out()]
        coef = modelo.coef_[2] - modelo.coef_[0]
        aportes = matriz * coef
        resultado = []
        for fila in aportes:
            orden = np.argsort(fila)[::-1]
            resultado.append([
                {"variable": NOMBRES_VARIABLES.get(columnas[i], columnas[i]), "aporte": round(float(fila[i]), 3)}
                for i in orden[:top]
                if fila[i] > 0.05
            ])
        return resultado


def cargar(ruta: str | Path) -> ModeloRiesgo:
    paquete = joblib.load(ruta)
    if paquete.get("version") != VERSION_MODELO:
        raise ValueError(f"Modelo con versión {paquete.get('version')}; se requiere {VERSION_MODELO}")
    return ModeloRiesgo(paquete)
